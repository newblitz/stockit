"""Frozen two-stage mT5 hierarchical text summarization from Algorithm 1."""

from __future__ import annotations

from collections.abc import Sequence

import torch


class HierarchicalSummarizer:
    """Hourly mT5 summaries -> daily mT5 summary -> frozen encoder embedding.

    Transformers is imported only when this class is constructed, leaving the
    neural architecture usable in environments that only install PyTorch.
    ``hourly_documents`` must already be chronologically ordered.
    """

    def __init__(
        self,
        model_name: str = "csebuetnlp/mT5_multilingual_XLSum",
        max_length: int = 512,
        summary_max_new_tokens: int = 64,
        device: str | torch.device | None = None,
    ) -> None:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError("Install `transformers` to use HierarchicalSummarizer.") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
        self.model.eval().requires_grad_(False)
        self.max_length = max_length
        self.summary_max_new_tokens = summary_max_new_tokens
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

    @torch.inference_mode()
    def _summarize(self, texts: Sequence[str], prompt: str) -> list[str]:
        inputs = [f"{prompt}\n{text}" for text in texts]
        tokens = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
        ).to(self.device)
        generated = self.model.generate(**tokens, max_new_tokens=self.summary_max_new_tokens)
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    @torch.inference_mode()
    def summarize_and_embed(
        self,
        hourly_documents: Sequence[str],
        *,
        hourly_prompt: str = "Summarize the market-moving facts in this hourly financial news.",
        daily_prompt: str = "Summarize the market-moving facts in these chronological hourly summaries.",
    ) -> tuple[str, torch.Tensor]:
        """Apply Algorithm 1 and return ``(daily_summary, embedding[d_model])``."""
        if not hourly_documents:
            hourly_documents = [""]
        hourly_summaries = self._summarize(hourly_documents, hourly_prompt)
        daily_summary = self._summarize([" ".join(hourly_summaries)], daily_prompt)[0]
        tokens = self.tokenizer(
            daily_summary, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)
        # Mean pooling the frozen encoder's last hidden states gives one daily vector.
        states = self.model.get_encoder()(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1)
        embedding = (states * mask).sum(1) / mask.sum(1).clamp_min(1)
        return daily_summary, embedding.squeeze(0).cpu()
