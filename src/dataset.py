import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .technical_indicators import compute_price_features


class CMINDataset(Dataset):
    """CMIN dataset loader.

    Price data: TSV with columns [date, feat1..feat5, volume]
    News data:  Each date is a file containing JSON-lines:
                {"text": ["word1", ...], "created_at": "YYYY-MM-DD HH:MM:SS"}
    """

    def __init__(self, data_dir: str, dataset: str, tokenizer_name: str, seq_len: int, max_text_len: int, split: str):
        base = Path(data_dir) / dataset
        price_dir = base / "price" / "processed"
        news_dir = base / "news" / "preprocessed"

        self.seq_len = seq_len
        self.max_text_len = max_text_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Load all stocks
        self.stocks = sorted([f.stem for f in price_dir.glob("*.txt")])

        # Split dates
        train_end = "2020-06-30"
        val_end = "2020-12-31"
        if split == "train":
            self.date_range = ("2018-01-01", train_end)
        elif split == "val":
            self.date_range = (train_end, val_end)
        else:
            self.date_range = (val_end, "2021-12-31")

        # Load price data for all stocks
        self.price_data = {}
        self.all_dates = set()
        for stock in self.stocks:
            pf = price_dir / f"{stock}.txt"
            if not pf.exists():
                continue
            raw = self._load_price(pf)
            if raw is None:
                continue
            features = compute_price_features(raw)
            dates = [r[0] for r in raw]
            self.price_data[stock] = dict(zip(dates, features))
            self.all_dates.update(dates)

        self.all_dates = sorted(self.all_dates)
        self.stocks = [s for s in self.stocks if s in self.price_data]

        # Filter dates to split range
        self.all_dates = [d for d in self.all_dates if self.date_range[0] <= d <= self.date_range[1]]

        # Preload news (map of stock -> date -> text)
        self.news_dir = news_dir
        self.news_cache = {}

        # Build samples: for each (stock, date) with enough history
        self.samples = []
        for stock in self.stocks:
            stock_dates = sorted(self.price_data[stock].keys())
            stock_dates_in_range = [d for d in stock_dates if d in set(self.all_dates)]
            for i in range(seq_len, len(stock_dates_in_range)):
                window_dates = stock_dates_in_range[i - seq_len : i]
                next_idx = i
                if next_idx >= len(stock_dates_in_range):
                    continue
                next_date = stock_dates_in_range[next_idx]
                next_feat = self.price_data[stock].get(next_date)
                if next_feat is None:
                    continue
                label = 1.0 if next_feat[4] > 0 else 0.0
                self.samples.append({
                    "stock": stock,
                    "window_dates": window_dates,
                    "label": label,
                })

        print(f"[{split}] {len(self.samples)} samples, {len(self.stocks)} stocks")

    @staticmethod
    def _load_price(path: Path):
        rows = []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 6:
                    parts = line.strip().split()
                if len(parts) < 6:
                    continue
                try:
                    date = parts[0]
                    vals = [float(x) for x in parts[1:]]
                    rows.append([date] + vals)
                except ValueError:
                    continue
        if not rows:
            return None
        return rows

    def _load_news(self, stock: str, date: str) -> str:
        """Load and concatenate all news headlines for a stock on a given date.

        Each date is a file (not a directory) containing JSON-lines.
        """
        key = (stock, date)
        if key in self.news_cache:
            return self.news_cache[key]

        fpath = self.news_dir / stock / date
        if not fpath.exists():
            self.news_cache[key] = ""
            return ""

        texts = []
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if "text" in obj:
                            words = obj["text"]
                            if isinstance(words, list):
                                texts.append(" ".join(str(w) for w in words))
                            else:
                                texts.append(str(words))
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass

        if not texts:
            self.news_cache[key] = ""
            return ""

        # Deduplicate
        unique = []
        seen = set()
        for t in texts:
            normalized = re.sub(r"[^a-z0-9]", "", t.lower())
            if normalized not in seen:
                seen.add(normalized)
                unique.append(t)

        result = " ".join(unique)
        self.news_cache[key] = result
        return result

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        stock = s["stock"]

        all_news = []
        for d in s["window_dates"]:
            news = self._load_news(stock, d)
            if news:
                all_news.append(news)

        combined_text = " ".join(all_news) if all_news else "no news available"

        enc = self.tokenizer(
            combined_text,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        price_seq = np.stack([self.price_data[stock][d] for d in s["window_dates"]])

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "price": torch.tensor(price_seq, dtype=torch.float32),
            "label": torch.tensor(s["label"], dtype=torch.float32),
        }
