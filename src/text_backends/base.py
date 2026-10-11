"""Common frozen-component interfaces for the text-processing path (replace.md).

The downstream architecture (``src/model.py``), training loop (``train.py``),
and price pipeline (``src/data.py``) only ever consume a cached per-day
embedding vector. They are intentionally agnostic to which model produced it.
These two abstract classes let the hourly/daily summarizer and the daily text
embedder be swapped independently and frozen (no gradients, run once offline).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class Summarizer(ABC):
    """Frozen two-stage summarizer: hourly news -> hourly summary -> daily synopsis."""

    @abstractmethod
    def summarize_hourly(self, texts: Sequence[str], *, ticker: str | None = None) -> list[str]:
        """Summarize independent hour-level documents.

        ``texts`` are already-cleaned, deduplicated, hour-bucketed document
        strings (see ``src/news_preprocessing.hourly_documents``). Returns one
        summary per input text, in the same order. Callers should omit empty
        hours from ``texts`` rather than relying on the implementation to
        special-case them (see Algorithm 1 edge cases in replace.md Step 2).
        """

    @abstractmethod
    def summarize_daily(
        self, hourly_summaries_per_day: Sequence[Sequence[str]], *, ticker: str | None = None
    ) -> list[str]:
        """Combine each day's chronological hourly summaries into one daily synopsis.

        ``hourly_summaries_per_day`` is a batch: one entry per trading day,
        each entry a chronologically ordered list of that day's hourly
        summaries (may be empty for a newsless day). Returns one daily
        synopsis string per day, in the same order. Batching the whole set of
        days together (rather than one day at a time) lets vLLM-backed
        implementations use efficient batched generation.
        """


class Embedder(ABC):
    """Frozen dense text encoder: ``Embed(S_day) -> R^dim`` (Algorithm 1)."""

    dim: int

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode daily synopsis strings into dense vectors, shape ``(len(texts), self.dim)``."""
