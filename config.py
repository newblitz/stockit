from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # Data
    dataset: str = "CMIN-US"
    data_dir: str = "data"

    # Hyperparameters (Table 2)
    seq_len: int = 30
    patch_len: int = 10
    stride: int = 5
    d_model: int = 128
    d_ff: int = 256
    n_heads: int = 16
    n_layers: int = 4
    n_classes: int = 1
    dropout: float = 0.2

    # Training
    lr: float = 1e-4
    lr_decay: float = 1e-4
    lr_decay_epoch: int = 5
    epochs: int = 20
    batch_size: int = 16
    patience: int = 5

    # LLM (legacy single-model field; kept for backward compatibility with
    # prepare_embeddings.py and any script that still reads config.llm_model).
    llm_model: str = "csebuetnlp/mT5_multilingual_XLSum"
    max_text_len: int = 512
    # mT5-base encoder hidden width; change this if cached text embeddings use
    # a different LLM or pooling representation. This is only a default for
    # building a model before any cache exists -- src/data.py always infers
    # the real width from the cache files at training time (see README
    # "Tensor shapes"), so this value never needs to be kept in sync by hand.
    text_embedding_dim: int = 768

    # Frozen text-processing backends (see replace.md). These are read by
    # `scripts/build_text_cache.py` (and its factory in `src/text_backends/`).
    # `prepare_embeddings.py` is untouched and always uses mT5 regardless of
    # this config, so README's canonical, already-cached mT5 pipeline stays
    # reproducible no matter what these default to. Set `--summarizer
    # mt5_xlsum --embedder mt5_encoder` on `scripts/build_text_cache.py` for
    # the config-A mT5 baseline (replace.md Step 7).
    summarizer: str = "qwen3_8b"           # {"mt5_xlsum", "qwen3_8b"}
    embedder: str = "bge_m3"               # {"mt5_encoder", "bge_m3"}
    # None -> each backend's own default model name (see src/text_backends/
    # */_backend.py DEFAULT_*_MODEL). Left as None rather than a fixed string
    # so switching `summarizer`/`embedder` never silently carries over a
    # mismatched model name (e.g. a Qwen repo id while summarizer=mt5_xlsum).
    summarizer_model_name: str | None = None
    embedder_model_name: str | None = None

    # Price features
    # Original CMIN processed price inputs: movement, OHLC returns, and volume.
    price_dim: int = 6

    # Co-attention fusion layers
    n_fusion_layers: int = 2

    # Device
    device: str = "cuda"

    @property
    def text_cache_subdir(self) -> str:
        """Namespace for a text cache built by scripts/build_text_cache.py.

        Keeps runs for different summarizer/embedder choices from overwriting
        each other, e.g. 'mt5_xlsum_mt5_encoder', 'qwen3_8b_bge_m3'.
        """
        return f"{self.summarizer}_{self.embedder}"

    @property
    def resolved_text_embedding_dim(self) -> int:
        """The embedding width implied by config.embedder, independent of any cache file."""
        from src.text_backends import embedder_dim

        return embedder_dim(self.embedder)
