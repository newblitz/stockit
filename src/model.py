"""PyTorch implementation of the model described in Zhang et al. (2026).

The trainable model consumes a window of price features and the corresponding
daily text embeddings.  The paper keeps the mT5 summarizer frozen; therefore
summarization/embedding is intentionally outside this module (see
``HierarchicalSummarizer`` in :mod:`src.summarization`).

Tensor convention
-----------------
``prices``: ``[batch, sequence_length, price_dim]``
``text_embeddings``: ``[batch, sequence_length, text_embedding_dim]``
``logits``: ``[batch, n_classes]``
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PriceTextAlignment(nn.Module):
    """Equations (1)--(3): project text and let price query text."""

    def __init__(self, text_embedding_dim: int, price_dim: int, dropout: float) -> None:
        super().__init__()
        self.text_projection = nn.Sequential(
            nn.Linear(text_embedding_dim, price_dim), nn.Dropout(dropout)
        )
        self.q = nn.Linear(price_dim, price_dim, bias=False)
        self.k = nn.Linear(price_dim, price_dim, bias=False)
        self.v = nn.Linear(price_dim, price_dim, bias=False)
        self.scale = math.sqrt(price_dim)

    def forward(self, prices: Tensor, text_embeddings: Tensor) -> Tensor:
        text = self.text_projection(text_embeddings)
        q, k, v = self.q(prices), self.k(text), self.v(text)
        weights = (q @ k.transpose(-2, -1) / self.scale).softmax(dim=-1)
        return weights @ v


class SingleChannelPatchTransformer(nn.Module):
    """Equations (4)--(7), with each price feature treated as one channel.

    The feature/channel axis is folded into the batch axis before the
    Transformer.  Right replication padding by ``stride`` gives exactly
    ``floor((L - P) / S) + 2`` patches, as specified in Equation (4).
    """

    def __init__(
        self,
        *,
        seq_len: int,
        patch_len: int,
        stride: int,
        price_dim: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if patch_len > seq_len:
            raise ValueError("patch_len must not exceed seq_len")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.price_dim = price_dim
        self.num_patches = (seq_len - patch_len) // stride + 2
        self.patch_projection = nn.Linear(patch_len, d_model)
        # Shared over samples and channels; this is the position term in (5).
        self.position = nn.Parameter(torch.zeros(1, 1, self.num_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Encode ``[B, L, price_dim]`` to ``[B*price_dim, N, d_model]``."""
        if x.ndim != 3:
            raise ValueError("expected a [batch, sequence, feature] tensor")
        batch, length, features = x.shape
        if length != self.seq_len or features != self.price_dim:
            raise ValueError(
                f"expected [B, {self.seq_len}, {self.price_dim}], got {tuple(x.shape)}"
            )
        # [B, C, L] -> right replication pad -> [B, C, N, P]
        channels = x.transpose(1, 2)
        padded = torch.cat((channels, channels[..., -1:].expand(-1, -1, self.stride)), dim=-1)
        patches = padded.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        if patches.size(-2) != self.num_patches:  # defensive check for Eq. (4)
            raise RuntimeError("unexpected patch count")
        encoded = self.patch_projection(patches) + self.position
        encoded = self.dropout(encoded)
        encoded = encoded.reshape(batch * self.price_dim, self.num_patches, -1)
        return self.encoder(encoded)


class BidirectionalCoAttention(nn.Module):
    """Equations (8)--(13): text-to-price and price-to-text co-attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.text_to_price = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.price_to_text = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.fuse = nn.Linear(2 * d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, text: Tensor, price: Tensor) -> Tensor:
        # Stream 1 (Eq. 8--9): text queries price.
        text_to_price, _ = self.text_to_price(text, price, price, need_weights=False)
        # Stream 2 (Eq. 10--11): price queries text.
        price_to_text, _ = self.price_to_text(price, text, text, need_weights=False)
        fused = self.fuse(torch.cat((text_to_price, price_to_text), dim=-1))
        z1 = self.norm1(price + self.dropout(fused))
        return self.norm2(z1 + self.dropout(self.ffn(z1)))


class HierarchicalCoAttentionStockPredictor(nn.Module):
    """Full trainable trend-prediction architecture from Sections 3.3--3.5.

    ``forward`` returns logits so training should use :class:`BCEWithLogitsLoss`.
    Call :meth:`predict_proba` for the paper's sigmoid probability in Eq. (15).
    """

    def __init__(
        self,
        *,
        text_embedding_dim: int,
        seq_len: int = 30,
        patch_len: int = 10,
        stride: int = 5,
        price_dim: int = 17,
        d_model: int = 128,
        d_ff: int = 256,
        n_heads: int = 16,
        n_layers: int = 4,
        n_fusion_layers: int = 2,
        n_classes: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.price_dim = price_dim
        self.alignment = PriceTextAlignment(text_embedding_dim, price_dim, dropout)
        self.text_encoder = SingleChannelPatchTransformer(
            seq_len=seq_len, patch_len=patch_len, stride=stride, price_dim=price_dim,
            d_model=d_model, d_ff=d_ff, n_heads=n_heads, n_layers=n_layers, dropout=dropout,
        )
        self.price_encoder = SingleChannelPatchTransformer(
            seq_len=seq_len, patch_len=patch_len, stride=stride, price_dim=price_dim,
            d_model=d_model, d_ff=d_ff, n_heads=n_heads, n_layers=n_layers, dropout=dropout,
        )
        self.co_attention = nn.ModuleList(
            BidirectionalCoAttention(d_model, n_heads, d_ff, dropout)
            for _ in range(n_fusion_layers)
        )
        n_patches = self.price_encoder.num_patches
        # Eq. (14) flattens channel, patch, and hidden dimensions per sample.
        self.classifier = nn.Linear(price_dim * n_patches * d_model, n_classes)

    @classmethod
    def from_config(cls, config: object) -> "HierarchicalCoAttentionStockPredictor":
        """Build the model from the repository's ``Config`` dataclass."""
        return cls(
            text_embedding_dim=config.text_embedding_dim,
            seq_len=config.seq_len,
            patch_len=config.patch_len,
            stride=config.stride,
            price_dim=config.price_dim,
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            n_fusion_layers=config.n_fusion_layers,
            n_classes=config.n_classes,
            dropout=config.dropout,
        )

    def forward(self, prices: Tensor, text_embeddings: Tensor) -> Tensor:
        if prices.shape[:2] != text_embeddings.shape[:2]:
            raise ValueError("price and text inputs must have matching batch and sequence axes")
        aligned_text = self.alignment(prices, text_embeddings)
        text = self.text_encoder(aligned_text)
        price = self.price_encoder(prices)
        for block in self.co_attention:
            price = block(text, price)
        batch = prices.size(0)
        features = price.reshape(batch, self.price_dim, price.size(1), price.size(2)).flatten(1)
        return self.classifier(features)

    def predict_proba(self, prices: Tensor, text_embeddings: Tensor) -> Tensor:
        return torch.sigmoid(self(prices, text_embeddings))
