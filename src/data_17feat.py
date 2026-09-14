"""Parallel 17-feature ("Price+TA+Text") dataset pipeline (paper §4.6).

Mirrors ``src/data.py`` exactly for every rule that does not depend on
price-feature dimensionality: forward-fill / >5%-missing exclusion /
3-sigma clipping (all via ``src.data.prepared_prices``), the market
calendar, the chronological train/val/test splits (identical
``SPLIT_RANGES``), and the cached mT5 text embeddings.

Text embeddings do **not** depend on the price-feature representation, so
the existing ``prepare_embeddings.py`` caches are reused completely
unchanged -- there is no separate "17-feature" embedding step to run.

The only difference from ``src/data.py`` is that the six raw CMIN price
columns are expanded to the paper's 17-dimensional Price+TA+Text
representation (via ``src.technical_indicators_17feat``) before per-stock
z-score normalization.

Kept as a fully separate module (rather than edited into ``src/data.py``)
so the 6-feature pipeline's code, caches, and checkpoints are never
touched by this file.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data import (
    PREPROCESSING_VERSION,
    SPLIT_RANGES,
    Split,
    _normalize_per_stock,
    available_tickers,
    prepared_prices,
)
from src.technical_indicators_17feat import NUM_TOTAL_COLUMNS, compute_17_price_features


@dataclass(frozen=True)
class StockSeries17Feat:
    ticker: str
    dates: list[date]
    features: np.ndarray  # [trading_days, 17]
    movements: np.ndarray  # raw next-day-label source, before normalization
    embeddings: Tensor  # [trading_days, text_embedding_dim]


def load_stock_series_17feat(
    dataset_root: str | Path, cache_root: str | Path, ticker: str
) -> StockSeries17Feat:
    dataset_root, cache_root = Path(dataset_root), Path(cache_root)
    dates, six_col_features, movements = prepared_prices(dataset_root, ticker)
    features17 = compute_17_price_features(six_col_features)

    cache_file = cache_root / f"{ticker}.pt"
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Missing text cache {cache_file}. Run `python prepare_embeddings.py --ticker {ticker}` "
            "first (the same cache used by the 6-feature pipeline is reused unchanged)."
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
    return StockSeries17Feat(
        ticker,
        dates,
        _normalize_per_stock(dates, features17),
        movements,
        payload["embeddings"].float(),
    )


class CMINWindowDataset17Feat(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """17-feature counterpart of ``src.data.CMINWindowDataset``.

    Identical windowing, chronological-split, and no-leakage rules; only
    the price-feature width differs (17 vs. 6).
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
                series = load_stock_series_17feat(dataset_root, cache_root, ticker)
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
                # §4.2 requires chronological splits with zero temporal overlap.
                if start <= window_start and start <= target_day <= end:
                    price_window = torch.from_numpy(series.features[i - seq_len + 1 : i + 1])
                    text_window = series.embeddings[i - seq_len + 1 : i + 1]
                    label = torch.tensor([float(series.movements[i + 1] > 0)], dtype=torch.float32)
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
            f"CMINWindowDataset17Feat [{split}]: loaded {len(loaded_tickers)} tickers, "
            f"skipped {len(skipped_tickers)}, total {split} samples: {len(self.samples):,}",
            flush=True,
        )
        if not self.samples:
            raise ValueError(
                f"no {split} samples were built. "
                "Ensure embedding caches exist for at least one ticker "
                "(run `python prepare_embeddings.py`)."
            )
        assert self.price_dim == NUM_TOTAL_COLUMNS
        assert self.text_embedding_dim is not None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.samples[index]
