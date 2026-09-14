"""Config for the paper's full 17-dimensional "Price+TA+Text" model (§4.6, §4.8).

Kept entirely separate from ``config.py`` (which remains the canonical
6-dimensional "Price+Text" ablation config used by ``train.py``) so the two
model variants never share settings, checkpoints, logs, or TensorBoard runs,
and so this file can never accidentally change the 6-feature model's
behaviour.
"""

from dataclasses import dataclass

from config import Config


@dataclass
class Config17Feature(Config):
    """Same Table 2 / §4.8 hyperparameters as ``Config``, but with the
    paper's full 17-dimensional price representation (§4.6: the 6 raw CMIN
    columns plus 11 technical indicators, see
    ``src/technical_indicators_17feat.py``) instead of the 6-dimensional
    "Price+Text" ablation.

    §4.8's grid search reports ``(M=2, H=16)`` as the CMIN-US
    accuracy-optimal co-attention configuration *for the full model*
    (67.01% ACC — the paper's headline result). Those are already
    ``Config``'s defaults (``n_heads=16``, ``n_fusion_layers=2``), so they
    carry over here unchanged.
    """

    # Price features
    # Paper §4.6 "Price+TA+Text": 6 raw CMIN columns (movement, four OHLC
    # returns, volume) + 11 technical indicators = 17.
    price_dim: int = 17
