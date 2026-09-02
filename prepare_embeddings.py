"""Cache frozen hierarchical mT5 embeddings for CMIN news, once per stock/day.

Run without --ticker to process every CMIN-US stock (110 tickers).
Already-cached tickers are skipped automatically, so the script is safely
resumable after interruption.
"""

from __future__ import annotations

import argparse
import json
import traceback
from collections import defaultdict
from datetime import date
from pathlib import Path

import torch

from src.data import _read_prices, available_tickers
from src.summarization import HierarchicalSummarizer


def hourly_documents(news_dir: Path, trading_days: list[date]) -> list[list[str]]:
    """Convert CMIN JSONL headlines into chronological hour-level documents."""
    by_day: dict[date, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
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
                text = record.get("text", "")
                if isinstance(text, list):
                    text = " ".join(text)
                if text:
                    by_day[day][hour].append(str(text))
    return [
        [" ".join(by_day[current_day][hour]) for hour in sorted(by_day[current_day])]
        for current_day in trading_days
    ]


def embed_ticker(
    ticker: str,
    *,
    dataset_root: Path,
    cache_root: Path,
    summarizer: HierarchicalSummarizer,
) -> None:
    """Build and save the embedding cache for a single ticker."""
    days, _ = _read_prices(dataset_root / "price" / "processed" / f"{ticker}.txt")
    documents = hourly_documents(dataset_root / "news" / "preprocessed" / ticker, days)
    embeddings = []
    for index, day_documents in enumerate(documents, start=1):
        _, embedding = summarizer.summarize_and_embed(day_documents)
        embeddings.append(embedding)
        if index % 25 == 0 or index == len(documents):
            print(f"  {ticker}: {index}/{len(documents)} days embedded", flush=True)
    output = cache_root / f"{ticker}.pt"
    torch.save(
        {"dates": [day.isoformat() for day in days], "embeddings": torch.stack(embeddings)},
        output,
    )
    print(f"  saved {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute frozen mT5 text embeddings for every CMIN-US ticker."
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
        help="Ticker(s) to process; may be repeated (default: all tickers).",
    )
    parser.add_argument(
        "--max-stocks",
        type=int,
        help="Process only the first N tickers alphabetically (useful for smoke tests).",
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

    tickers: list[str] = args.ticker or available_tickers(args.dataset_root)
    if args.max_stocks is not None:
        tickers = tickers[: args.max_stocks]

    total = len(tickers)
    args.cache_root.mkdir(parents=True, exist_ok=True)

    print(
        f"prepare_embeddings: {total} ticker(s) to process → cache at {args.cache_root}",
        flush=True,
    )
    print(f"Loading summarizer model '{args.model}' ...", flush=True)
    summarizer = HierarchicalSummarizer(args.model, device=args.device)
    print(f"Model loaded on device: {summarizer.device}\n", flush=True)

    skipped: list[str] = []
    succeeded: list[str] = []
    failed: list[str] = []

    for ticker_idx, ticker in enumerate(tickers, start=1):
        output = args.cache_root / f"{ticker}.pt"
        print(
            f"[{ticker_idx}/{total}] {ticker}",
            flush=True,
        )
        if output.exists():
            print(f"  skip: cache already exists at {output}", flush=True)
            skipped.append(ticker)
            continue
        try:
            embed_ticker(
                ticker,
                dataset_root=args.dataset_root,
                cache_root=args.cache_root,
                summarizer=summarizer,
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
        f"Embedding run complete.\n"
        f"  Succeeded : {len(succeeded):>4}  {succeeded[:5]}{'...' if len(succeeded) > 5 else ''}\n"
        f"  Skipped   : {len(skipped):>4}  (cache already existed)\n"
        f"  Failed    : {len(failed):>4}  {failed if failed else ''}",
        flush=True,
    )
    if failed:
        print(
            "\nFailed tickers (re-run with --ticker <TICKER> for each to retry):",
            flush=True,
        )
        for t in failed:
            print(f"  python prepare_embeddings.py --ticker {t}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
