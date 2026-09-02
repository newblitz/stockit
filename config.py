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
    epochs: int = 100
    batch_size: int = 16
    # Set patience to 0 to disable early stopping; otherwise stop after this many
    # epochs without validation-MCC improvement.
    patience: int = 0

    # LLM
    llm_model: str = "csebuetnlp/mT5_multilingual_XLSum"
    max_text_len: int = 512
    # mT5-base encoder hidden width; change this if cached text embeddings use
    # a different LLM or pooling representation.
    text_embedding_dim: int = 768

    # Price features
    # The official CMIN processed files contain movement, OHLC returns, and
    # volume: six numerical columns after the date.
    price_dim: int = 6

    # Co-attention fusion layers
    n_fusion_layers: int = 2

    # Device
    device: str = "cuda"
