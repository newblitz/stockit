"""Cache frozen hierarchical mT5 embeddings for CMIN news, once per stock/day."""

from __future__ import annotations

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/CMIN-Dataset-official/CMIN-US"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/cmin-us-mt5"))
    parser.add_argument("--ticker", action="append", help="Ticker to process; may be repeated")
    parser.add_argument("--max-stocks", type=int)
    parser.add_argument("--model", default="csebuetnlp/mT5_multilingual_XLSum")
    parser.add_argument("--device")
    args = parser.parse_args()

    tickers = args.ticker or available_tickers(args.dataset_root)
    if args.max_stocks is not None:
        tickers = tickers[: args.max_stocks]
    args.cache_root.mkdir(parents=True, exist_ok=True)
    summarizer = HierarchicalSummarizer(args.model, device=args.device)
    for ticker in tickers:
        output = args.cache_root / f"{ticker}.pt"
        if output.exists():
            print(f"skip {ticker}: cache exists", flush=True)
            continue
        days, _ = _read_prices(args.dataset_root / "price" / "processed" / f"{ticker}.txt")
        documents = hourly_documents(args.dataset_root / "news" / "preprocessed" / ticker, days)
        embeddings = []
        for index, day_documents in enumerate(documents, start=1):
            _, embedding = summarizer.summarize_and_embed(day_documents)
            embeddings.append(embedding)
            if index % 25 == 0 or index == len(documents):
                print(f"{ticker}: {index}/{len(documents)} days", flush=True)
        torch.save({"dates": [day.isoformat() for day in days], "embeddings": torch.stack(embeddings)}, output)
        print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
