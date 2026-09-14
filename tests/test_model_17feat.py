import torch

from src.model import HierarchicalCoAttentionStockPredictor


def test_17_feature_model_shapes_and_backward() -> None:
    """Same shape/backward smoke test as tests/test_model.py, but for the
    paper's full 17-dimensional Price+TA+Text configuration."""
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
    logits.sum().backward()
