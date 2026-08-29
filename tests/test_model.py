import torch

from config import Config
from src.model import HierarchicalCoAttentionStockPredictor


def test_paper_model_shapes_and_backward() -> None:
    model = HierarchicalCoAttentionStockPredictor(
        text_embedding_dim=64,
        seq_len=30,
        patch_len=10,
        stride=5,
        price_dim=17,
        d_model=32,
        d_ff=64,
        n_heads=4,
        n_layers=1,
        n_fusion_layers=2,
    )
    prices = torch.randn(2, 30, 17)
    text = torch.randn(2, 30, 64)
    logits = model(prices, text)
    assert logits.shape == (2, 1)
    assert model.price_encoder.num_patches == 6  # floor((30 - 10) / 5) + 2
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).backward()


def test_construct_from_repository_config() -> None:
    model = HierarchicalCoAttentionStockPredictor.from_config(Config())
    assert model.classifier.out_features == 1
