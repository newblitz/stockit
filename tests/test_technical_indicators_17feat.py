import numpy as np
import pytest

from src.technical_indicators_17feat import NUM_TOTAL_COLUMNS, compute_17_price_features


def test_output_shape_and_passthrough_columns() -> None:
    rng = np.random.default_rng(0)
    raw = rng.normal(scale=0.01, size=(120, 6)).astype(np.float32)
    raw[:, 5] = rng.integers(1_000, 1_000_000, size=120)  # volume column

    features = compute_17_price_features(raw)

    assert features.shape == (120, NUM_TOTAL_COLUMNS)
    # The first 6 columns must be the original CMIN fields, unmodified.
    np.testing.assert_allclose(features[:, :6], raw, rtol=1e-5, atol=1e-5)
    assert np.isfinite(features).all()


def test_rejects_wrong_input_width() -> None:
    with pytest.raises(ValueError):
        compute_17_price_features(np.zeros((10, 5), dtype=np.float32))


def test_rsi_column_is_bounded_zero_one() -> None:
    rng = np.random.default_rng(1)
    raw = rng.normal(scale=0.02, size=(60, 6)).astype(np.float32)
    raw[:, 5] = 1.0
    features = compute_17_price_features(raw)
    rsi_column = features[:, 6]
    assert rsi_column.min() >= 0.0 and rsi_column.max() <= 1.0


def test_deterministic_for_same_input() -> None:
    rng = np.random.default_rng(2)
    raw = rng.normal(scale=0.015, size=(80, 6)).astype(np.float32)
    raw[:, 5] = np.abs(raw[:, 5]) * 1_000_000
    first = compute_17_price_features(raw)
    second = compute_17_price_features(raw)
    np.testing.assert_array_equal(first, second)
