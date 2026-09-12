"""CMIN dataset parsing and chronological window construction."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
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

# Bump this whenever an embedding-affecting cleaning rule changes.  A cache
# produced with an older rule must never be mixed into a paper-faithful run.
PREPROCESSING_VERSION = "paper-4.2-v3-close-time-levenshtein"
MAX_MISSING_FRACTION = 0.05
MAX_FORWARD_FILL_DAYS = 3
RETURN_COLUMNS = slice(0, 5)


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


@lru_cache(maxsize=None)
def market_trading_days(dataset_root: str) -> tuple[date, ...]:
    """Use the union of observed dates as the market calendar.

    This avoids treating weekends as missing trading days while still exposing a
    date absent from one stock but present for its market peers.
    """
    root = Path(dataset_root)
    days: set[date] = set()
    for ticker in available_tickers(root):
        ticker_days, _ = _read_prices(root / "price" / "processed" / f"{ticker}.txt")
        days.update(ticker_days)
    return tuple(sorted(days))


def _forward_fill_row(previous: np.ndarray) -> np.ndarray:
    """Represent a carried-forward close in CMIN's return-based format."""
    filled = previous.copy()
    filled[RETURN_COLUMNS] = 0.0
    # Volume is not a close, but carrying its last observation is the least
    # invasive forward-fill and keeps the feature finite.
    return filled


def _fill_short_missing_periods(
    dates: list[date], features: np.ndarray, expected_days: tuple[date, ...]
) -> tuple[list[date], np.ndarray, int]:
    by_day = {day: features[index] for index, day in enumerate(dates)}
    missing = [day for day in expected_days if day not in by_day]
    missing_fraction = len(missing) / max(len(expected_days), 1)
    if missing_fraction > MAX_MISSING_FRACTION:
        raise ValueError(
            f"{len(missing)}/{len(expected_days)} market trading days missing "
            f"({missing_fraction:.2%} > {MAX_MISSING_FRACTION:.0%})"
        )

    filled_dates: list[date] = []
    filled_rows: list[np.ndarray] = []
    index = 0
    while index < len(expected_days):
        current = expected_days[index]
        if current in by_day:
            filled_dates.append(current)
            filled_rows.append(by_day[current])
            index += 1
            continue
        end = index
        while end < len(expected_days) and expected_days[end] not in by_day:
            end += 1
        gap = expected_days[index:end]
        if len(gap) > MAX_FORWARD_FILL_DAYS or not filled_rows:
            raise ValueError(f"unfillable missing period of {len(gap)} trading days")
        previous = filled_rows[-1]
        for missing_day in gap:
            previous = _forward_fill_row(previous)
            filled_dates.append(missing_day)
            filled_rows.append(previous)
        index = end
    return filled_dates, np.stack(filled_rows).astype(np.float32), len(missing)


def _truncate_returns_from_training(dates: list[date], values: np.ndarray) -> np.ndarray:
    """Clip each return field to its training-only mean ± 3 standard deviations."""
    train_stop = SPLIT_RANGES["train"][1]
    mask = np.asarray([current <= train_stop for current in dates])
    train_returns = values[mask, RETURN_COLUMNS]
    mean = train_returns.mean(axis=0)
    std = train_returns.std(axis=0)
    lower, upper = mean - 3 * std, mean + 3 * std
    clipped = values.copy()
    clipped[:, RETURN_COLUMNS] = np.clip(clipped[:, RETURN_COLUMNS], lower, upper)
    return clipped


def prepared_prices(
    dataset_root: str | Path, ticker: str
) -> tuple[list[date], np.ndarray, np.ndarray]:
    """Apply every §4.2 numerical preprocessing operation for one stock.

    Missing-data eligibility and return clipping happen before feature creation;
    z-score statistics are applied later, using the training interval only.
    """
    root = Path(dataset_root)
    raw_dates, raw_values = _read_prices(root / "price" / "processed" / f"{ticker}.txt")
    dates, values, _ = _fill_short_missing_periods(
        raw_dates, raw_values, market_trading_days(str(root.resolve()))
    )
    values = _truncate_returns_from_training(dates, values)
    # Preserve the original six processed CMIN fields as model input.
    return dates, values, values[:, 0].copy()


def _normalize_per_stock(dates: list[date], features: np.ndarray) -> np.ndarray:
    """Z-score every feature using only the paper's training interval."""
    train_stop = SPLIT_RANGES["train"][1]
    mask = np.asarray([current <= train_stop for current in dates])
    train = features[mask]
    mean, std = train.mean(axis=0), train.std(axis=0)
    return (features - mean) / np.maximum(std, 1e-6)


def load_stock_series(dataset_root: str | Path, cache_root: str | Path, ticker: str) -> StockSeries:
    dataset_root, cache_root = Path(dataset_root), Path(cache_root)
    dates, features, movements = prepared_prices(dataset_root, ticker)
    cache_file = cache_root / f"{ticker}.pt"
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Missing text cache {cache_file}. Run `python prepare_embeddings.py --ticker {ticker}` first."
        )
    payload = torch.load(cache_file, map_location="cpu", weights_only=False)
    cache_version = payload.get("preprocessing_version")
    if cache_version not in (None, PREPROCESSING_VERSION):
        raise ValueError(
            "embedding cache uses different preprocessing; rerun prepare_embeddings.py"
        )
    cached_dates = [date.fromisoformat(value) for value in payload["dates"]]
    if cached_dates != dates:
        raise ValueError(f"cached dates do not match prices for {ticker}")
    return StockSeries(
        ticker,
        dates,
        _normalize_per_stock(dates, features),
        movements,
        payload["embeddings"].float(),
    )


class CMINWindowDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
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
        self.samples: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        # Aligned with ``samples`` for chronological per-stock backtests.
        self.sample_tickers: list[str] = []
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
                window_start = series.dates[i - seq_len + 1]
                # §4.2 requires chronological splits with zero temporal overlap:
                # every input day, as well as the next-day target, belongs to the
                # selected split.  In particular, validation/test windows may not
                # consume the tail of the preceding partition.
                if start <= window_start and start <= target_day <= end:
                    price_window = torch.from_numpy(series.features[i - seq_len + 1 : i + 1])
                    text_window = series.embeddings[i - seq_len + 1 : i + 1]
                    # Column zero is the supplied close-to-close movement percentage.
                    label = torch.tensor([float(series.movements[i + 1] > 0)], dtype=torch.float32)
                    # Retain the clipped next-day log return for paper §4.6.1's
                    # post-selection trading evaluation; it is never model input.
                    next_return = torch.tensor([series.movements[i + 1]], dtype=torch.float32)
                    self.samples.append((price_window, text_window, label, next_return))
                    self.sample_tickers.append(ticker)
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

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.samples[index]
