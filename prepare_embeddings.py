"""Cache frozen hierarchical mT5 embeddings for CMIN news, once per stock/day.

Run without --ticker to process every CMIN-US stock (110 tickers).
Only caches bearing the current preprocessing version are skipped automatically,
so a preprocessing change safely triggers regeneration rather than mixing stale
text embeddings into a run.
"""

from __future__ import annotations

import argparse
import math
import traceback
from datetime import date
from pathlib import Path

import torch

from src.data import PREPROCESSING_VERSION, available_tickers, prepared_prices
from src.news_preprocessing import (
    available_trading_day as _available_trading_day,
    clean_text as _clean,
    hourly_documents,
    is_near_duplicate as _is_near_duplicate,
    levenshtein_distance,
    market_session as _market_session,
)
from src.summarization import HierarchicalSummarizer


def _has_current_cache(path: Path) -> bool:
    """Return whether a completed cache was produced by this exact pipeline."""
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return payload.get("preprocessing_version") == PREPROCESSING_VERSION
    except Exception:
        return False


def embed_ticker(
    ticker: str,
    *,
    dataset_root: Path,
    cache_root: Path,
    summarizer: HierarchicalSummarizer,
    partial_save_interval: int = 100,
) -> None:
    """Build and save the embedding cache for a single ticker.

    Each stock's final embedding is written to ``<cache_root>/<TICKER>.pt``
    immediately after all its trading days are processed — not at the end of
    the whole shard — so a completed stock is never reprocessed on resume.

    Intra-stock resumability: a ``<TICKER>.partial.pt`` file is written every
    ``partial_save_interval`` days.  If the process is interrupted mid-stock,
    the next run picks up from the last partial checkpoint instead of day 1.
    The partial file is deleted once the final ``.pt`` is saved.
    """
    days, _, _ = prepared_prices(dataset_root, ticker)
    documents = hourly_documents(dataset_root / "news" / "preprocessed" / ticker, days)
    total_days = len(days)

    output  = cache_root / f"{ticker}.pt"
    partial = cache_root / f"{ticker}.partial.pt"

    # ── Resume from partial checkpoint if available ───────────────────────────
    start = 0
    embeddings: list[torch.Tensor] = []
    if partial.exists():
        saved = torch.load(partial, map_location="cpu", weights_only=False)
        if saved.get("preprocessing_version") == PREPROCESSING_VERSION:
            embeddings = list(saved["embeddings"])   # Tensor → list for appending
            start = len(embeddings)
            print(f"  {ticker}: resuming from day {start + 1}/{total_days} (partial checkpoint found)", flush=True)
        else:
            print(f"  {ticker}: ignoring partial cache with old preprocessing", flush=True)

    for index, day_documents in enumerate(documents[start:], start=start + 1):
        _, embedding = summarizer.summarize_and_embed(day_documents)
        embeddings.append(embedding)

        if index % 25 == 0 or index == total_days:
            print(f"  {ticker}: {index}/{total_days} days embedded", flush=True)

        # ── Save partial checkpoint (intra-stock, for session-expiry resilience) ──
        if partial_save_interval > 0 and index % partial_save_interval == 0 and index < total_days:
            torch.save(
                {"dates": [d.isoformat() for d in days[:index]], "embeddings": torch.stack(embeddings),
                 "preprocessing_version": PREPROCESSING_VERSION},
                partial,
            )
            print(f"  {ticker}: partial checkpoint saved at day {index}/{total_days}", flush=True)

    # ── All days done: write final file and clean up partial ─────────────────
    torch.save(
        {"dates": [day.isoformat() for day in days], "embeddings": torch.stack(embeddings),
         "preprocessing_version": PREPROCESSING_VERSION},
        output,
    )
    if partial.exists():
        partial.unlink()
    print(f"  saved {output}", flush=True)


def _select_tickers(args: argparse.Namespace) -> list[str]:
    """Resolve the final ticker list from --ticker / --shard / --max-stocks.

    Priority:
      1. --ticker  → use exactly those tickers (ignores --shard / --max-stocks).
      2. Otherwise → full dataset list, trimmed by --max-stocks, then sliced by --shard.
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


def _print_status(tickers: list[str], cache_root: Path, shard_total: int | None) -> None:
    """Print done/partial/pending counts and, when relevant, copy-paste shard commands."""
    done    = [t for t in tickers if _has_current_cache(cache_root / f"{t}.pt")]
    partial = [t for t in tickers if (cache_root / f"{t}.partial.pt").exists()
               and not (cache_root / f"{t}.pt").exists()]
    pending = [t for t in tickers if t not in done and not (cache_root / f"{t}.partial.pt").exists()]
    print(
        f"\nStatus — {len(tickers)} ticker(s) in scope: "
        f"{len(done)} done, {len(partial)} partial (will resume), {len(pending)} not started"
    )
    if partial:
        print(f"\nPartial — interrupted mid-stock, will auto-resume ({len(partial)}): {' '.join(partial)}")
    if pending:
        print(f"\nNot started ({len(pending)}): {' '.join(pending)}")
    print(f"\nDone ({len(done)}): {' '.join(done) if done else 'none'}")
    if (pending or partial) and shard_total is None:
        # Suggest parallel shard commands only for a full-list status call.
        # Target ~15 stocks per shard.
        suggested = max(1, min(math.ceil(len(tickers) / 15), 8))
        print(f"\nTo split into {suggested} parallel shards (~15 stocks each):")
        for i in range(suggested):
            print(f"  python prepare_embeddings.py --shard {i} {suggested} --device cuda")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute frozen mT5 text embeddings for every CMIN-US ticker.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # check overall progress without loading the model
  python prepare_embeddings.py --status

  # run 8 parallel Kaggle/cloud notebooks (~14 stocks each)
  python prepare_embeddings.py --shard 0 8 --device cuda
  python prepare_embeddings.py --shard 1 8 --device cuda
  ...up to...
  python prepare_embeddings.py --shard 7 8 --device cuda

  # process specific tickers on any cloud instance
  python prepare_embeddings.py --ticker AAPL --ticker MSFT --device cuda

  # resume: completed tickers are skipped; interrupted stocks resume from partial checkpoint
  python prepare_embeddings.py --shard 0 8 --device cuda
""",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/CMIN-Dataset-official/CMIN-US"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/cache/cmin-us-mt5"),
    )
    parser.add_argument(
        "--ticker",
        action="append",
        metavar="TICKER",
        help=(
            "Process this ticker; may be repeated for an explicit list. "
            "Mutually exclusive with --shard: if both are given, --ticker wins."
        ),
    )
    parser.add_argument(
        "--shard",
        nargs=2,
        type=int,
        metavar=("INDEX", "TOTAL"),
        help=(
            "Divide all tickers into TOTAL equal chunks and process chunk INDEX (0-indexed). "
            "Use --shard 0 8 … --shard 7 8 for 8 parallel cloud jobs (~14 stocks each). "
            "Completed tickers are skipped; interrupted stocks resume from their partial checkpoint."
        ),
    )
    parser.add_argument(
        "--partial-save-interval",
        type=int,
        default=100,
        metavar="N",
        help=(
            "Save intra-stock partial progress every N days so a session expiry mid-stock "
            "doesn't lose all work (0 disables). Default: 100."
        ),
    )
    parser.add_argument(
        "--max-stocks",
        type=int,
        help="Cap the ticker list to the first N alphabetically before applying --shard (smoke tests).",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print done/pending ticker counts and suggested shard commands, then exit without processing.",
    )
    parser.add_argument(
        "--model",
        default="csebuetnlp/mT5_multilingual_XLSum",
        help="Hugging Face summarizer model name.",
    )
    parser.add_argument(
        "--device",
        help="Inference device (e.g. 'cuda', 'cpu'). Defaults to cuda if available.",
    )
    args = parser.parse_args()

    try:
        tickers = _select_tickers(args)
    except ValueError as exc:
        parser.error(str(exc))
        return  # unreachable; satisfies type checkers

    args.cache_root.mkdir(parents=True, exist_ok=True)

    # --status: show progress and exit without touching the model
    if args.status:
        _print_status(tickers, args.cache_root, shard_total=args.shard[1] if args.shard else None)
        return

    pending_tickers = [t for t in tickers if not _has_current_cache(args.cache_root / f"{t}.pt")]
    total = len(tickers)
    shard_label = f" (shard {args.shard[0]}/{args.shard[1]})" if args.shard else ""
    print(
        f"prepare_embeddings{shard_label}: {total} ticker(s) in scope, "
        f"{len(pending_tickers)} pending, {total - len(pending_tickers)} already cached → "
        f"cache at {args.cache_root}",
        flush=True,
    )

    if not pending_tickers:
        print("Nothing to do — all tickers in this scope are already cached.", flush=True)
        return

    print(f"Loading summarizer model '{args.model}' ...", flush=True)
    summarizer = HierarchicalSummarizer(args.model, device=args.device)
    print(f"Model loaded on device: {summarizer.device}\n", flush=True)

    skipped: list[str] = []
    succeeded: list[str] = []
    failed: list[str] = []

    for ticker_idx, ticker in enumerate(tickers, start=1):
        output = args.cache_root / f"{ticker}.pt"
        print(f"[{ticker_idx}/{total}] {ticker}", flush=True)
        if _has_current_cache(output):
            print(f"  skip: cache already exists at {output}", flush=True)
            skipped.append(ticker)
            continue
        if output.exists():
            print("  regenerate: cache uses old preprocessing", flush=True)
        partial_file = args.cache_root / f"{ticker}.partial.pt"
        if partial_file.exists():
            print(f"  partial checkpoint found — will resume mid-stock", flush=True)
        try:
            embed_ticker(
                ticker,
                dataset_root=args.dataset_root,
                cache_root=args.cache_root,
                summarizer=summarizer,
                partial_save_interval=args.partial_save_interval,
            )
            succeeded.append(ticker)
        except Exception:
            print(
                f"  ERROR processing {ticker} — skipping and continuing:\n"
                + traceback.format_exc(),
                flush=True,
            )
            failed.append(ticker)

    # Final summary
    print("\n" + "=" * 60, flush=True)
    print(
        f"Embedding run complete{shard_label}.\n"
        f"  Succeeded : {len(succeeded):>4}  {succeeded[:5]}{'...' if len(succeeded) > 5 else ''}\n"
        f"  Skipped   : {len(skipped):>4}  (cache already existed)\n"
        f"  Failed    : {len(failed):>4}  {failed if failed else ''}",
        flush=True,
    )
    if failed:
        print(
            "\nFailed tickers — retry individually with:",
            flush=True,
        )
        for t in failed:
            print(f"  python prepare_embeddings.py --ticker {t} --device cuda", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
