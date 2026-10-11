"""Resumable frozen text-cache builder for pluggable Summarizer/Embedder backends.

This is the new entry point from replace.md Step 4. It implements the same
Algorithm 1 hourly -> daily -> embedding pipeline as `prepare_embeddings.py`,
but through the `src.text_backends` interface so the frozen summarizer and
embedder can each be mT5, Qwen3, or BGE-M3 (see `config.py`'s `summarizer` /
`embedder` switches). `prepare_embeddings.py` is untouched and remains the
tool for the original, single-model mT5 pipeline.

Two artifacts are produced per ticker, both resumable and namespaced by
`<summarizer>_<embedder>` so different backend combinations never collide:

    <cache-root>/texts/<TICKER>.jsonl   # inspectable hourly + daily text, JSON lines
    <cache-root>/<TICKER>.pt            # {dates, embeddings, preprocessing_version}

The second file has the *exact same format* `src/data.py` already reads for
the original mT5 cache, so `train.py --cache-root <cache-root>` works
unmodified against a Qwen3/BGE-M3 (or any other) cache: `text_embedding_dim`
is inferred straight from the saved tensor's width (768 for mT5, 1024 for
BGE-M3), satisfying replace.md Step 5 without touching the downstream model.

Run the summarization stage and the embedding stage separately (`--stage`)
to control GPU residency, e.g. unload an 8B Qwen3 model before loading
BGE-M3. For the mT5-only baseline (summarizer=mt5_xlsum, embedder=mt5_encoder)
both stages share one small loaded model, matching the original
`HierarchicalSummarizer.summarize_and_embed` exactly.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import traceback
from pathlib import Path

# Allow running as `python scripts/build_text_cache.py` from any working
# directory: Python only puts this script's own folder on sys.path, but
# `config` and `src` live at the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from config import Config
from src.data import PREPROCESSING_VERSION, available_tickers, prepared_prices
from src.news_preprocessing import hourly_documents
from src.text_backends import Embedder, Summarizer, build_embedder, build_summarizer
from src.text_backends.factory import EMBEDDER_CHOICES, SUMMARIZER_CHOICES


def _texts_path(cache_root: Path, ticker: str) -> Path:
    return cache_root / "texts" / f"{ticker}.jsonl"


def _embeddings_path(cache_root: Path, ticker: str) -> Path:
    return cache_root / f"{ticker}.pt"


def _embeddings_partial_path(cache_root: Path, ticker: str) -> Path:
    return cache_root / f"{ticker}.partial.pt"


def _load_daily_records(path: Path) -> dict[str, str]:
    """Return ``{date_iso: daily synopsis}`` already written for one ticker."""
    daily: dict[str, str] = {}
    if not path.exists():
        return daily
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("kind") == "daily":
            daily[record["date"]] = record["text"]
    return daily


def summarize_ticker(
    ticker: str,
    *,
    dataset_name: str,
    dataset_root: Path,
    cache_root: Path,
    summarizer: Summarizer,
    batch_days: int,
) -> bool:
    """Fill in missing hourly/daily text records for one ticker (Stage 1).

    Returns ``True`` if any summarization work was performed, ``False`` if
    the ticker's daily synopses were already fully cached (mirrors
    ``prepare_embeddings.py``'s succeeded/skipped distinction).
    """
    days, _, _ = prepared_prices(dataset_root, ticker)
    documents = hourly_documents(dataset_root / "news" / "preprocessed" / ticker, days)

    texts_path = _texts_path(cache_root, ticker)
    texts_path.parent.mkdir(parents=True, exist_ok=True)
    done_daily = _load_daily_records(texts_path)

    missing = [(day, docs) for day, docs in zip(days, documents) if day.isoformat() not in done_daily]
    if not missing:
        print(f"  {ticker}: all {len(days)} daily synopses already cached", flush=True)
        return False

    print(f"  {ticker}: summarizing {len(missing)}/{len(days)} missing day(s)", flush=True)
    with texts_path.open("a", encoding="utf-8") as sink:
        for start in range(0, len(missing), batch_days):
            chunk = missing[start : start + batch_days]

            # Flatten (chunk-local day index, hour text) so hourly summaries for
            # many days can be generated in a single batched call.
            flat_hours: list[tuple[int, str]] = []
            for day_index, (_day, docs) in enumerate(chunk):
                for hour_text in docs:
                    flat_hours.append((day_index, hour_text))

            hourly_texts = [text for _index, text in flat_hours]
            hourly_summaries = summarizer.summarize_hourly(hourly_texts, ticker=ticker)

            per_day_summaries: list[list[str]] = [[] for _ in chunk]
            for (day_index, _text), summary in zip(flat_hours, hourly_summaries):
                per_day_summaries[day_index].append(summary)

            # Empty day (no news) -> per_day_summaries[i] == []; summarize_daily
            # implementations fall back to the original "" placeholder, as
            # mT5's HierarchicalSummarizer already did.
            daily_summaries = summarizer.summarize_daily(per_day_summaries, ticker=ticker)

            for day_index, (day, _docs) in enumerate(chunk):
                date_iso = day.isoformat()
                for hour_index, summary in enumerate(per_day_summaries[day_index]):
                    sink.write(
                        json.dumps(
                            {
                                "dataset": dataset_name,
                                "ticker": ticker,
                                "date": date_iso,
                                "hour": hour_index,
                                "kind": "hourly",
                                "text": summary,
                            }
                        )
                        + "\n"
                    )
                sink.write(
                    json.dumps(
                        {
                            "dataset": dataset_name,
                            "ticker": ticker,
                            "date": date_iso,
                            "hour": None,
                            "kind": "daily",
                            "text": daily_summaries[day_index],
                        }
                    )
                    + "\n"
                )
            sink.flush()
            print(
                f"    {ticker}: {min(start + batch_days, len(missing))}/{len(missing)} days summarized",
                flush=True,
            )
    return True


def _has_current_embeddings(path: Path, expected_dates: list[str]) -> bool:
    """Return whether a completed embedding cache already covers every expected date."""
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return (
        payload.get("preprocessing_version") == PREPROCESSING_VERSION
        and payload.get("dates") == expected_dates
    )


def embed_ticker(
    ticker: str,
    *,
    dataset_root: Path,
    cache_root: Path,
    embedder: Embedder,
    batch_size: int,
    partial_save_interval: int,
) -> bool:
    """Encode cached daily synopses into the standard ``<TICKER>.pt`` cache (Stage 2).

    Batches ``batch_size`` days per call to ``embedder.encode(...)`` and writes
    an intra-stock ``<TICKER>.partial.pt`` checkpoint every
    ``partial_save_interval`` days, exactly like ``prepare_embeddings.py``'s
    own ``embed_ticker`` -- so a BGE-M3 (or any other embedder) run surviving
    a session interruption resumes from the last completed day instead of day
    1, and a completed ticker's final ``<TICKER>.pt`` is never recomputed.

    Returns ``True`` if any embedding work was performed, ``False`` if the
    ticker's embeddings were already fully cached.
    """
    days, _, _ = prepared_prices(dataset_root, ticker)
    output = _embeddings_path(cache_root, ticker)
    expected_dates = [day.isoformat() for day in days]
    if _has_current_embeddings(output, expected_dates):
        print(f"  {ticker}: embeddings already cached for all {len(days)} days", flush=True)
        return False

    daily_text = _load_daily_records(_texts_path(cache_root, ticker))
    missing_text = [day.isoformat() for day in days if day.isoformat() not in daily_text]
    if missing_text:
        raise ValueError(
            f"{ticker}: {len(missing_text)} day(s) missing a daily synopsis "
            f"(e.g. {missing_text[0]}); run `--stage summarize` first."
        )

    partial = _embeddings_partial_path(cache_root, ticker)
    total_days = len(days)

    start = 0
    embeddings: list[torch.Tensor] = []
    if partial.exists():
        saved = torch.load(partial, map_location="cpu", weights_only=False)
        if saved.get("preprocessing_version") == PREPROCESSING_VERSION:
            embeddings = list(saved["embeddings"])
            start = len(embeddings)
            print(f"  {ticker}: resuming embedding from day {start + 1}/{total_days}", flush=True)
        else:
            print(f"  {ticker}: ignoring partial embedding cache with old preprocessing", flush=True)

    pending_days = days[start:]
    for chunk_start in range(0, len(pending_days), batch_size):
        chunk_days = pending_days[chunk_start : chunk_start + batch_size]
        texts = [daily_text[day.isoformat()] for day in chunk_days]
        vectors = embedder.encode(texts)
        embeddings.extend(torch.from_numpy(vectors).float())

        done = start + chunk_start + len(chunk_days)
        if done % 25 == 0 or done == total_days:
            print(f"  {ticker}: {done}/{total_days} days embedded", flush=True)

        if partial_save_interval > 0 and done % partial_save_interval == 0 and done < total_days:
            torch.save(
                {
                    "dates": [d.isoformat() for d in days[:done]],
                    "embeddings": torch.stack(embeddings),
                    "preprocessing_version": PREPROCESSING_VERSION,
                },
                partial,
            )
            print(f"  {ticker}: partial embedding checkpoint saved at day {done}/{total_days}", flush=True)

    torch.save(
        {
            "dates": [day.isoformat() for day in days],
            "embeddings": torch.stack(embeddings),
            "preprocessing_version": PREPROCESSING_VERSION,
        },
        output,
    )
    if partial.exists():
        partial.unlink()
    print(f"  saved {output}", flush=True)
    return True


def _select_tickers(args: argparse.Namespace) -> list[str]:
    """Resolve the final ticker list from --ticker / --shard / --max-stocks.

    Same precedence and shard convention as ``prepare_embeddings.py``:
      1. --ticker  -> use exactly those tickers (ignores --shard / --max-stocks).
      2. Otherwise -> full dataset list, trimmed by --max-stocks, then sliced by --shard.
    """
    if args.ticker:
        return list(args.ticker)
    tickers = available_tickers(args.dataset_root)
    if args.max_stocks is not None:
        tickers = tickers[: args.max_stocks]
    if args.shard is not None:
        shard_idx, shard_total = args.shard
        if not (0 <= shard_idx < shard_total):
            raise ValueError(
                f"--shard INDEX must be in [0, TOTAL-1]; got INDEX={shard_idx}, TOTAL={shard_total}"
            )
        chunk = math.ceil(len(tickers) / shard_total)
        tickers = tickers[shard_idx * chunk : (shard_idx + 1) * chunk]
    return tickers


def _free_gpu(obj: object) -> None:
    del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _print_status(
    tickers: list[str], dataset_root: Path, cache_root: Path, shard_total: int | None
) -> None:
    """Print done/partial/pending counts and, when relevant, copy-paste shard commands.

    Mirrors ``prepare_embeddings.py``'s ``_print_status``, but a ticker here
    has two stages (summarize, embed) instead of one.
    """
    done: list[str] = []
    partial: list[str] = []
    pending: list[str] = []
    for ticker in tickers:
        days, _, _ = prepared_prices(dataset_root, ticker)
        expected_dates = [day.isoformat() for day in days]
        daily_done = len(_load_daily_records(_texts_path(cache_root, ticker)))
        embeddings_done = _has_current_embeddings(_embeddings_path(cache_root, ticker), expected_dates)
        if daily_done == len(days) and embeddings_done:
            done.append(ticker)
        elif daily_done == 0 and not _embeddings_partial_path(cache_root, ticker).exists():
            pending.append(ticker)
        else:
            partial.append(ticker)
    print(
        f"\nStatus \u2014 {len(tickers)} ticker(s) in scope: "
        f"{len(done)} done, {len(partial)} partial (will resume), {len(pending)} not started"
    )
    if partial:
        print(f"\nPartial \u2014 in progress, will auto-resume ({len(partial)}): {' '.join(partial)}")
    if pending:
        print(f"\nNot started ({len(pending)}): {' '.join(pending)}")
    print(f"\nDone ({len(done)}): {' '.join(done) if done else 'none'}")
    if (pending or partial) and shard_total is None:
        suggested = max(1, min(math.ceil(len(tickers) / 15), 8))
        print(f"\nTo split into {suggested} parallel shards (~15 stocks each):")
        for i in range(suggested):
            print(f"  python scripts/build_text_cache.py --shard {i} {suggested} --device cuda")
    print()


def _run_ticker_stage(
    stage_label: str,
    tickers: list[str],
    worker,
    *,
    config: Config,
    device: str | None,
) -> None:
    """Run ``worker(ticker)`` over every ticker, mirroring prepare_embeddings.py's
    continue-on-error batch loop: a failing ticker is logged and skipped rather
    than aborting the whole run, and a final succeeded/skipped/failed summary
    with copy-paste retry commands is printed at the end.

    ``worker`` must return ``True`` if it did work, ``False`` if the ticker was
    already fully cached (skipped).
    """
    total = len(tickers)
    skipped: list[str] = []
    succeeded: list[str] = []
    failed: list[str] = []

    for ticker_idx, ticker in enumerate(tickers, start=1):
        print(f"[{stage_label} {ticker_idx}/{total}] {ticker}", flush=True)
        try:
            did_work = worker(ticker)
        except Exception:
            print(
                f"  ERROR processing {ticker} \u2014 skipping and continuing:\n" + traceback.format_exc(),
                flush=True,
            )
            failed.append(ticker)
            continue
        (succeeded if did_work else skipped).append(ticker)

    print("\n" + "=" * 60, flush=True)
    print(
        f"{stage_label.capitalize()} run complete.\n"
        f"  Succeeded : {len(succeeded):>4}  {succeeded[:5]}{'...' if len(succeeded) > 5 else ''}\n"
        f"  Skipped   : {len(skipped):>4}  (already cached)\n"
        f"  Failed    : {len(failed):>4}  {failed if failed else ''}",
        flush=True,
    )
    if failed:
        print("\nFailed tickers \u2014 retry individually with:", flush=True)
        device_flag = f" --device {device}" if device else ""
        for ticker in failed:
            print(
                f"  python scripts/build_text_cache.py --ticker {ticker} "
                f"--summarizer {config.summarizer} --embedder {config.embedder} "
                f"--stage {stage_label}{device_flag}",
                flush=True,
            )
    print("=" * 60, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a resumable text cache (hourly + daily summaries, daily embeddings) "
            "using pluggable frozen Summarizer/Embedder backends."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Config C (paper default switch): Qwen3-8B summarizer + BGE-M3 embedder
  python scripts/build_text_cache.py --ticker AAPL --summarizer qwen3_8b --embedder bge_m3 --device cuda

  # Config B: keep mT5-XLSum summarizer, switch only the embedder to BGE-M3
  python scripts/build_text_cache.py --ticker AAPL --summarizer mt5_xlsum --embedder bge_m3 --device cuda

  # Config A (baseline, identical to prepare_embeddings.py's own cache):
  python scripts/build_text_cache.py --ticker AAPL --summarizer mt5_xlsum --embedder mt5_encoder --device cuda

  # Run stages separately (e.g. to unload Qwen3 before loading BGE-M3 on a smaller GPU)
  python scripts/build_text_cache.py --summarizer qwen3_8b --embedder bge_m3 --stage summarize --device cuda
  python scripts/build_text_cache.py --summarizer qwen3_8b --embedder bge_m3 --stage embed --device cuda

  # check overall progress without loading any model
  python scripts/build_text_cache.py --status

  # run 8 parallel Kaggle/cloud notebooks (~14 stocks each)
  python scripts/build_text_cache.py --shard 0 8 --device cuda
  python scripts/build_text_cache.py --shard 1 8 --device cuda
  ...up to...
  python scripts/build_text_cache.py --shard 7 8 --device cuda
""",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/CMIN-Dataset-official/CMIN-US"))
    parser.add_argument("--dataset-name", default="CMIN-US", help="Recorded in the JSONL text cache only.")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Defaults to data/cache/<summarizer>_<embedder>/ (see config.py Config.text_cache_subdir).",
    )
    parser.add_argument("--ticker", action="append", metavar="TICKER", help="Repeatable; defaults to all tickers.")
    parser.add_argument("--max-stocks", type=int, help="Cap to the first N tickers alphabetically (smoke tests).")
    parser.add_argument(
        "--shard",
        nargs=2,
        type=int,
        metavar=("INDEX", "TOTAL"),
        help=(
            "Divide all tickers into TOTAL equal chunks and process chunk INDEX (0-indexed), "
            "same convention as prepare_embeddings.py. Ignored if --ticker is given. "
            "Already-cached tickers/days are skipped; interrupted tickers resume from their "
            "partial checkpoint."
        ),
    )
    parser.add_argument("--summarizer", choices=SUMMARIZER_CHOICES, default=Config.summarizer)
    parser.add_argument("--embedder", choices=EMBEDDER_CHOICES, default=Config.embedder)
    parser.add_argument("--summarizer-model", default=None, help="Overrides the backend's default model name.")
    parser.add_argument("--embedder-model", default=None, help="Overrides the backend's default model name.")
    parser.add_argument(
        "--stage",
        choices=["all", "summarize", "embed"],
        default="all",
        help="Run both stages (default), or only one (e.g. to split across machines/GPUs).",
    )
    parser.add_argument("--batch-days", type=int, default=20, help="Days summarized per batched LLM call.")
    parser.add_argument("--embed-batch-size", type=int, default=32, help="Days embedded per batched call.")
    parser.add_argument("--partial-save-interval", type=int, default=100, help="Embedding checkpoint cadence (0 disables).")
    parser.add_argument("--device", help="Inference device (e.g. 'cuda', 'cpu'). Defaults per-backend.")
    parser.add_argument("--no-vllm", action="store_true", help="Force the transformers fallback for QwenSummarizer.")
    parser.add_argument("--status", action="store_true", help="Print per-ticker progress and exit.")
    args = parser.parse_args()

    config = Config()
    config.summarizer = args.summarizer
    config.embedder = args.embedder
    if args.summarizer_model:
        config.summarizer_model_name = args.summarizer_model
    if args.embedder_model:
        config.embedder_model_name = args.embedder_model

    cache_root = args.cache_root or Path("data/cache") / config.text_cache_subdir
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        tickers = _select_tickers(args)
    except ValueError as exc:
        parser.error(str(exc))
        return  # unreachable; satisfies type checkers

    if args.status:
        _print_status(tickers, args.dataset_root, cache_root, shard_total=args.shard[1] if args.shard else None)
        return

    print(
        f"build_text_cache: summarizer={config.summarizer} embedder={config.embedder} "
        f"cache_root={cache_root} tickers={len(tickers)}",
        flush=True,
    )

    shared_mt5_pair = config.summarizer == "mt5_xlsum" and config.embedder == "mt5_encoder"

    if args.stage in ("all", "summarize"):
        print(f"Loading summarizer '{config.summarizer}' ...", flush=True)
        summarizer_kwargs = {"device": args.device}
        if config.summarizer == "qwen3_8b" and args.no_vllm:
            from src.text_backends.qwen_backend import QwenSummarizer

            summarizer = QwenSummarizer(
                model_name=config.summarizer_model_name, device=args.device, use_vllm=False
            )
        else:
            summarizer = build_summarizer(config, **summarizer_kwargs)

        _run_ticker_stage(
            "summarize",
            tickers,
            lambda ticker: summarize_ticker(
                ticker,
                dataset_name=args.dataset_name,
                dataset_root=args.dataset_root,
                cache_root=cache_root,
                summarizer=summarizer,
                batch_days=args.batch_days,
            ),
            config=config,
            device=args.device,
        )

        if shared_mt5_pair and args.stage == "all":
            # Reuse the already-loaded mT5 model for embedding instead of a
            # second pass, mirroring the original single-model Algorithm 1 run.
            print("Embedding with the shared mT5 model (no reload) ...", flush=True)
            embedder = build_embedder(config, summarizer=summarizer, device=args.device)
            _run_ticker_stage(
                "embed",
                tickers,
                lambda ticker: embed_ticker(
                    ticker,
                    dataset_root=args.dataset_root,
                    cache_root=cache_root,
                    embedder=embedder,
                    batch_size=args.embed_batch_size,
                    partial_save_interval=args.partial_save_interval,
                ),
                config=config,
                device=args.device,
            )
            return

        if args.stage == "all":
            print("Unloading summarizer before loading the embedder ...", flush=True)
            _free_gpu(summarizer)

    if args.stage in ("all", "embed"):
        print(f"Loading embedder '{config.embedder}' ...", flush=True)
        embedder = build_embedder(config, device=args.device)
        _run_ticker_stage(
            "embed",
            tickers,
            lambda ticker: embed_ticker(
                ticker,
                dataset_root=args.dataset_root,
                cache_root=cache_root,
                embedder=embedder,
                batch_size=args.embed_batch_size,
                partial_save_interval=args.partial_save_interval,
            ),
            config=config,
            device=args.device,
        )


if __name__ == "__main__":
    main()
