"""CMIN dataset parsing and chronological window construction."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


Split = Literal["train", "val", "test"]
SPLIT_RANGES = {
    "train": (date(2018, 1, 1), date(2020, 6, 30)),
    "val": (date(2020, 7, 1), date(2020, 12, 31)),
    "test": (date(2021, 1, 1), date(2021, 12, 31)),
}


@dataclass(frozen=True)
class StockSeries:
    ticker: str
    dates: list[date]
    features: np.ndarray  # [trading_days, 6]
    movements: np.ndarray  # raw next-day-label source, before normalization
    embeddings: Tensor  # [trading_days, text_embedding_dim]


def available_tickers(dataset_root: str | Path) -> list[str]:
    return sorted(path.stem for path in Path(dataset_root, "price", "processed").glob("*.txt"))


def _read_prices(path: Path) -> tuple[list[date], np.ndarray]:
    rows: list[list[float]] = []
    dates: list[date] = []
    for line in path.read_text().splitlines():
        fields = line.split("\t")
        if len(fields) != 7:
            raise ValueError(f"expected date plus six values in {path}; got {len(fields)}")
        dates.append(date.fromisoformat(fields[0]))
        rows.append([float(value) for value in fields[1:]])
    return dates, np.asarray(rows, dtype=np.float32)


def _normalize_per_stock(dates: list[date], features: np.ndarray) -> np.ndarray:
    """Z-score every feature using only the paper's training interval."""
    train_stop = SPLIT_RANGES["train"][1]
    mask = np.asarray([current <= train_stop for current in dates])
    train = features[mask]
    mean, std = train.mean(axis=0), train.std(axis=0)
    return (features - mean) / np.maximum(std, 1e-6)


def load_stock_series(dataset_root: str | Path, cache_root: str | Path, ticker: str) -> StockSeries:
    dataset_root, cache_root = Path(dataset_root), Path(cache_root)
    dates, features = _read_prices(dataset_root / "price" / "processed" / f"{ticker}.txt")
    cache_file = cache_root / f"{ticker}.pt"
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Missing text cache {cache_file}. Run `python prepare_embeddings.py --ticker {ticker}` first."
        )
    payload = torch.load(cache_file, map_location="cpu", weights_only=False)
    cached_dates = [date.fromisoformat(value) for value in payload["dates"]]
    if cached_dates != dates:
        raise ValueError(f"cached dates do not match prices for {ticker}")
    return StockSeries(
        ticker,
        dates,
        _normalize_per_stock(dates, features),
        features[:, 0].copy(),
        payload["embeddings"].float(),
    )


class CMINWindowDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """30-day samples with next-trading-day movement labels (Eq. 17).

    Tickers whose embedding cache (``.pt``) has not yet been built are
    automatically skipped with a warning so that training can proceed on
    the subset of stocks that have already been embedded.  Run
    ``prepare_embeddings.py`` (without ``--ticker``) to build caches for
    all 110 CMIN-US tickers and then retrain on the full dataset.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        cache_root: str | Path,
        split: Split,
        *,
        seq_len: int = 30,
        max_stocks: int | None = None,
    ) -> None:
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        tickers = available_tickers(dataset_root)
        if max_stocks is not None:
            tickers = tickers[:max_stocks]
        if not tickers:
            raise FileNotFoundError(f"no processed price files under {dataset_root}")
        start, end = SPLIT_RANGES[split]
        self.samples: list[tuple[Tensor, Tensor, Tensor]] = []
        self.price_dim: int | None = None
        self.text_embedding_dim: int | None = None
        loaded_tickers: list[str] = []
        skipped_tickers: list[str] = []
        for ticker in tickers:
            cache_file = Path(cache_root) / f"{ticker}.pt"
            if not cache_file.exists():
                skipped_tickers.append(ticker)
                continue
            try:
                series = load_stock_series(dataset_root, cache_root, ticker)
            except Exception as exc:
                warnings.warn(
                    f"Skipping {ticker}: failed to load — {exc}",
                    stacklevel=2,
                )
                skipped_tickers.append(ticker)
                continue
            self.price_dim = series.features.shape[1]
            self.text_embedding_dim = series.embeddings.shape[1]
            loaded_tickers.append(ticker)
            # i is the final observed day; its next day supplies the target.
            for i in range(seq_len - 1, len(series.dates) - 1):
                target_day = series.dates[i + 1]
                if start <= target_day <= end:
                    price_window = torch.from_numpy(series.features[i - seq_len + 1 : i + 1])
                    text_window = series.embeddings[i - seq_len + 1 : i + 1]
                    # Column zero is the supplied close-to-close movement percentage.
                    label = torch.tensor([float(series.movements[i + 1] > 0)], dtype=torch.float32)
                    self.samples.append((price_window, text_window, label))
        if skipped_tickers:
            warnings.warn(
                f"{len(skipped_tickers)} ticker(s) skipped (missing embedding cache): "
                f"{skipped_tickers[:10]}{'...' if len(skipped_tickers) > 10 else ''}. "
                "Run `python prepare_embeddings.py` to build all caches before training "
                "on the full dataset.",
                stacklevel=2,
            )
        print(
            f"CMINWindowDataset [{split}]: loaded {len(loaded_tickers)} tickers, "
            f"skipped {len(skipped_tickers)}, total {split} samples: {len(self.samples):,}",
            flush=True,
        )
        if not self.samples:
            raise ValueError(
                f"no {split} samples were built. "
                "Ensure embedding caches exist for at least one ticker "
                "(run `python prepare_embeddings.py`)."
            )
        assert self.price_dim is not None and self.text_embedding_dim is not None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.samples[index]
