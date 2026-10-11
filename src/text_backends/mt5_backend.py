"""Thin ``Summarizer`` / ``Embedder`` wrappers around the existing frozen mT5 code.

This module does not reimplement mT5 inference: it adapts the existing
``src.summarization.HierarchicalSummarizer`` (the original Algorithm-1
implementation, untouched) to the new common interfaces so the mT5 path keeps
working as the reproducible baseline (config A in replace.md Step 7) while
``QwenSummarizer`` / ``BGEM3Embedder`` are plugged in through the same factory.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from src.summarization import HierarchicalSummarizer

from .base import Embedder, Summarizer

DEFAULT_MT5_MODEL = "csebuetnlp/mT5_multilingual_XLSum"

# These mirror the previous default prompts baked into
# HierarchicalSummarizer.summarize_and_embed, kept verbatim for baseline
# reproducibility.
_HOURLY_PROMPT_MT5 = "Summarize the market-moving facts in this hourly financial news."
_DAILY_PROMPT_MT5 = "Summarize the market-moving facts in these chronological hourly summaries."


class MT5Summarizer(Summarizer):
    """Wraps the frozen two-stage mT5 XL-Sum summarizer (unchanged weights/logic)."""

    def __init__(
        self,
        model_name: str = DEFAULT_MT5_MODEL,
        *,
        max_length: int = 512,
        summary_max_new_tokens: int = 64,
        device: str | torch.device | None = None,
    ) -> None:
        self.impl = HierarchicalSummarizer(
            model_name,
            max_length=max_length,
            summary_max_new_tokens=summary_max_new_tokens,
            device=device,
        )

    def summarize_hourly(self, texts: Sequence[str], *, ticker: str | None = None) -> list[str]:
        texts = list(texts)
        if not texts:
            return []
        return self.impl._summarize(texts, _HOURLY_PROMPT_MT5)

    def summarize_daily(
        self, hourly_summaries_per_day: Sequence[Sequence[str]], *, ticker: str | None = None
    ) -> list[str]:
        days = list(hourly_summaries_per_day)
        if not days:
            return []
        # Empty day -> empty string, matching the original
        # ``if not hourly_documents: hourly_documents = [""]`` fallback.
        joined = [" ".join(day) if day else "" for day in days]
        return self.impl._summarize(joined, _DAILY_PROMPT_MT5)


class MT5Embedder(Embedder):
    """Mean-pooled frozen mT5 encoder embedding of the daily synopsis (Algorithm 1's ``Embed``)."""

    dim = 768

    def __init__(
        self,
        summarizer: MT5Summarizer | None = None,
        *,
        model_name: str = DEFAULT_MT5_MODEL,
        max_length: int = 512,
        device: str | torch.device | None = None,
    ) -> None:
        # Reuse the already-loaded mT5 model/tokenizer when paired with
        # MT5Summarizer (config A) instead of loading a second copy.
        self.impl = summarizer.impl if summarizer is not None else HierarchicalSummarizer(
            model_name, max_length=max_length, device=device
        )

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors: list[np.ndarray] = []
        for text in texts:
            tokens = self.impl.tokenizer(
                text or "", return_tensors="pt", truncation=True, max_length=self.impl.max_length
            ).to(self.impl.device)
            states = self.impl.model.get_encoder()(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            embedding = (states * mask).sum(1) / mask.sum(1).clamp_min(1)
            vectors.append(embedding.squeeze(0).cpu().numpy())
        return np.stack(vectors).astype(np.float32)
