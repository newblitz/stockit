"""Pluggable frozen summarizer / embedder backends (see replace.md).

Use :func:`build_summarizer` / :func:`build_embedder` to construct the
backend selected by ``config.summarizer`` / ``config.embedder`` without
importing every optional dependency (vLLM, FlagEmbedding, ...) up front.
"""

from .base import Embedder, Summarizer
from .factory import build_embedder, build_summarizer, embedder_dim

__all__ = [
    "Summarizer",
    "Embedder",
    "build_summarizer",
    "build_embedder",
    "embedder_dim",
]
