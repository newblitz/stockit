"""Select a Summarizer / Embedder implementation from ``config.py`` switches."""

from __future__ import annotations

from .base import Embedder, Summarizer
from .mt5_backend import DEFAULT_MT5_MODEL, MT5Embedder, MT5Summarizer

SUMMARIZER_CHOICES = ("mt5_xlsum", "qwen3_8b")
EMBEDDER_CHOICES = ("mt5_encoder", "bge_m3")

# Fixed embedding widths per embedder backend; used to size Eq. (1)'s
# Linear(d -> d_p) layer before any cache file exists (e.g. in tests or when
# constructing a model ahead of `prepare_embeddings.py`/`build_text_cache.py`).
EMBEDDER_DIMS = {"mt5_encoder": 768, "bge_m3": 1024}


def embedder_dim(name: str) -> int:
    try:
        return EMBEDDER_DIMS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown embedder '{name}'. Choices: {EMBEDDER_CHOICES}") from exc


def build_summarizer(config: object, *, device: str | None = None) -> Summarizer:
    """Build the frozen summarizer selected by ``config.summarizer``."""
    name = getattr(config, "summarizer", "mt5_xlsum")
    if name == "mt5_xlsum":
        model_name = getattr(config, "summarizer_model_name", None) or DEFAULT_MT5_MODEL
        return MT5Summarizer(model_name=model_name, device=device)
    if name == "qwen3_8b":
        from .qwen_backend import DEFAULT_QWEN_MODEL, QwenSummarizer

        model_name = getattr(config, "summarizer_model_name", None) or DEFAULT_QWEN_MODEL
        return QwenSummarizer(model_name=model_name, device=device)
    raise ValueError(f"Unknown summarizer '{name}'. Choices: {SUMMARIZER_CHOICES}")


def build_embedder(
    config: object, *, summarizer: Summarizer | None = None, device: str | None = None
) -> Embedder:
    """Build the frozen embedder selected by ``config.embedder``.

    If ``summarizer`` is an already-loaded ``MT5Summarizer`` and
    ``config.embedder == "mt5_encoder"``, the embedder reuses its model
    instead of loading mT5 a second time (this is the config-A baseline,
    where one frozen mT5 model performs both roles, exactly as in the
    original ``HierarchicalSummarizer.summarize_and_embed``).
    """
    name = getattr(config, "embedder", "mt5_encoder")
    if name == "mt5_encoder":
        shared = summarizer if isinstance(summarizer, MT5Summarizer) else None
        return MT5Embedder(shared, device=device)
    if name == "bge_m3":
        from .bge_backend import BGEM3Embedder, DEFAULT_BGE_MODEL

        model_name = getattr(config, "embedder_model_name", None) or DEFAULT_BGE_MODEL
        return BGEM3Embedder(model_name=model_name, device=device)
    raise ValueError(f"Unknown embedder '{name}'. Choices: {EMBEDDER_CHOICES}")
