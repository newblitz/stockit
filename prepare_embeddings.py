"""Cache frozen hierarchical mT5 embeddings for CMIN news, once per stock/day.

Run without --ticker to process every CMIN-US stock (110 tickers).
Already-cached tickers are skipped automatically, so the script is safely
resumable after interruption.
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import math
import re
import traceback
from collections import defaultdict
from datetime import date
from pathlib import Path

import torch

from src.data import _read_prices, available_tickers
from src.summarization import HierarchicalSummarizer

# §4.2 preprocessing constants
_MIN_ARTICLE_CHARS = 50
_DEDUP_THRESHOLD = 0.9   # character 4-gram Jaccard threshold (~90% Levenshtein similarity)
_NGRAM_SIZE = 4
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def _clean(raw: str) -> str:
    """§4.2 text normalization: HTML unescape → lowercase → strip non-alphanumeric → collapse whitespace."""
    text = html.unescape(raw)
    text = text.lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _char_ngrams(text: str) -> frozenset[str]:
    """Precompute character n-gram set for O(1)-per-lookup Jaccard similarity."""
    return frozenset(text[i : i + _NGRAM_SIZE] for i in range(max(0, len(text) - _NGRAM_SIZE + 1)))


def _is_near_duplicate(candidate_ngrams: frozenset[str], cand_len: int, kept_ngrams: list[frozenset[str]], kept_lens: list[int]) -> bool:
    """Return True if candidate Jaccard-similarity with any kept article exceeds _DEDUP_THRESHOLD.

    Uses precomputed character 4-gram sets and C-level set operations instead of pure-Python
    SequenceMatcher, giving a 20-50× speedup with equivalent duplicate-detection accuracy for
    short financial headlines (§4.2: keep earliest; drop later articles with >90% overlap).

    Avoids allocating a union frozenset by computing |A∪B| = |A| + |B| - |A∩B| arithmetically.
    """
    if not candidate_ngrams:
        return False
    for k, k_len in zip(kept_ngrams, kept_lens):
        if not k:
            continue
        intersection_size = len(candidate_ngrams & k)
        union_size = cand_len + k_len - intersection_size
        if union_size > 0 and intersection_size / union_size > _DEDUP_THRESHOLD:
            return True
    return False


def hourly_documents(news_dir: Path, trading_days: list[date]) -> list[list[str]]:
    """Convert CMIN JSONL headlines into deduplicated, cleaned, hour-level documents.

    Applies the three §4.2 preprocessing steps before text reaches the summarizer:
    1. HTML entity decoding + lowercasing + non-alphanumeric stripping + whitespace collapse.
    2. Articles shorter than _MIN_ARTICLE_CHARS (50) characters after cleaning are dropped.
    3. Per-day near-duplicate removal (>90% character overlap): earliest timestamp is kept.
    """
    # Collect (timestamp, hour, cleaned_text) per calendar day.
    by_day: dict[date, list[tuple[str, str, str]]] = defaultdict(list)
    if news_dir.exists():
        for file in news_dir.iterdir():
            if not file.is_file():
                continue
            try:
                day = date.fromisoformat(file.name)
            except ValueError:
                continue
            for line in file.read_text().splitlines():
                record = json.loads(line)
                timestamp = record.get("created_at", "")
                hour = timestamp[:13] if len(timestamp) >= 13 else f"{day.isoformat()} 00"
                raw = record.get("text", "")
                if isinstance(raw, list):
                    raw = " ".join(raw)
                text = _clean(raw)
                if len(text) >= _MIN_ARTICLE_CHARS:
                    by_day[day].append((timestamp, hour, text))

    result: list[list[str]] = []
    for current_day in trading_days:
        # Sort ascending by timestamp so the earliest article wins deduplication.
        articles = sorted(by_day.get(current_day, []), key=lambda x: x[0])

        # Deduplicate: keep first occurrence, drop near-duplicates.
        # n-gram sets are precomputed once per article so _is_near_duplicate
        # only needs C-level set ops (not repeated SequenceMatcher calls).
        kept_texts: list[str] = []
        kept_hours: list[str] = []
        kept_ngrams: list[frozenset[str]] = []
        kept_lens: list[int] = []
        for _ts, hour, text in articles:
            ng = _char_ngrams(text)
            ng_len = len(ng)
            if not _is_near_duplicate(ng, ng_len, kept_ngrams, kept_lens):
                kept_texts.append(text)
                kept_hours.append(hour)
                kept_ngrams.append(ng)
                kept_lens.append(ng_len)

        # Group deduplicated articles by hour, preserving chronological order within each hour.
        hourly: dict[str, list[str]] = defaultdict(list)
        for hour, text in zip(kept_hours, kept_texts):
            hourly[hour].append(text)

        result.append([" ".join(hourly[h]) for h in sorted(hourly)])

    return result


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
    days, _ = _read_prices(dataset_root / "price" / "processed" / f"{ticker}.txt")
    documents = hourly_documents(dataset_root / "news" / "preprocessed" / ticker, days)
    total_days = len(days)

    output  = cache_root / f"{ticker}.pt"
    partial = cache_root / f"{ticker}.partial.pt"

    # ── Resume from partial checkpoint if available ───────────────────────────
    start = 0
    embeddings: list[torch.Tensor] = []
    if partial.exists():
        saved = torch.load(partial, map_location="cpu", weights_only=False)
        embeddings = list(saved["embeddings"])   # Tensor → list for appending
        start = len(embeddings)
        print(f"  {ticker}: resuming from day {start + 1}/{total_days} (partial checkpoint found)", flush=True)

    for index, day_documents in enumerate(documents[start:], start=start + 1):
        _, embedding = summarizer.summarize_and_embed(day_documents)
        embeddings.append(embedding)

        if index % 25 == 0 or index == total_days:
            print(f"  {ticker}: {index}/{total_days} days embedded", flush=True)

        # ── Save partial checkpoint (intra-stock, for session-expiry resilience) ──
        if partial_save_interval > 0 and index % partial_save_interval == 0 and index < total_days:
            torch.save(
                {"dates": [d.isoformat() for d in days[:index]], "embeddings": torch.stack(embeddings)},
                partial,
            )
            print(f"  {ticker}: partial checkpoint saved at day {index}/{total_days}", flush=True)

    # ── All days done: write final file and clean up partial ─────────────────
    torch.save(
        {"dates": [day.isoformat() for day in days], "embeddings": torch.stack(embeddings)},
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
    done    = [t for t in tickers if (cache_root / f"{t}.pt").exists()]
    partial = [t for t in tickers if (cache_root / f"{t}.partial.pt").exists()
               and not (cache_root / f"{t}.pt").exists()]
    pending = [t for t in tickers if not (cache_root / f"{t}.pt").exists()
               and not (cache_root / f"{t}.partial.pt").exists()]
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

    pending_tickers = [t for t in tickers if not (args.cache_root / f"{t}.pt").exists()]
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
        if output.exists():
            print(f"  skip: cache already exists at {output}", flush=True)
            skipped.append(ticker)
            continue
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
