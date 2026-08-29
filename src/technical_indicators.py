import numpy as np


def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI (Relative Strength Index)."""
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = _ema(gain, period)
    avg_loss = _ema(loss, period)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi / 100.0  # normalize to [0, 1]


def compute_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, histogram."""
    fast_ema = _ema(prices, fast)
    slow_ema = _ema(prices, slow)
    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(prices: np.ndarray, period: int = 20, num_std: float = 2.0):
    """Bollinger Bands (upper, middle, lower)."""
    middle = _sma(prices, period)
    std = _rolling_std(prices, period)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return _ema(tr, period)


def compute_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On-Balance Volume."""
    delta = np.diff(close, prepend=close[0])
    direction = np.sign(delta)
    obv = np.cumsum(direction * volume)
    return obv / (np.abs(obv).max() + 1e-8)  # normalize


def compute_ma(prices: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    return _sma(prices, period)


def compute_price_features(raw: np.ndarray) -> np.ndarray:
    """Compute 17-dimensional feature vector from CMIN raw data.

    raw: (T, 6) -> columns [date, feat1, feat2, feat3, feat4, feat5, volume]
    The 5 feature columns are log-return-like normalized values.
    We reconstruct approximate prices by cumulative sum (treating as log returns),
    then compute technical indicators.

    Returns: (T, 17)
    """
    feats = raw[:, 1:6]  # (T, 5) - OHLCV log returns
    volume = raw[:, 6] if raw.shape[1] > 6 else np.ones(raw.shape[0])

    # Reconstruct approximate prices from close returns (column 4, index 4)
    close_returns = feats[:, 4]
    close_prices = np.exp(np.cumsum(close_returns))
    close_prices[0] = 1.0

    # Use close as proxy for OHL (since we don't have raw OHLCV)
    high_prices = close_prices * (1 + np.abs(feats[:, 1]))
    low_prices = close_prices * (1 - np.abs(feats[:, 2]))

    # Compute indicators
    rsi = compute_rsi(close_prices)
    macd_l, macd_s, macd_h = compute_macd(close_prices)
    bb_u, bb_m, bb_l = compute_bollinger(close_prices)
    atr = compute_atr(high_prices, low_prices, close_prices)
    obv = compute_obv(close_prices, volume)
    ma5 = compute_ma(close_prices, 5)
    ma10 = compute_ma(close_prices, 10)
    ma20 = compute_ma(close_prices, 20)

    # Stack all 17 features
    features = np.stack([
        feats[:, 0],  # 1. feat1 (open-like)
        feats[:, 1],  # 2. feat2 (high-like)
        feats[:, 2],  # 3. feat3 (low-like)
        feats[:, 3],  # 4. feat4
        feats[:, 4],  # 5. feat5 (close return)
        rsi,          # 6. RSI
        macd_l,       # 7. MACD line
        macd_s,       # 8. MACD signal
        macd_h,       # 9. MACD histogram
        bb_u,         # 10. BB upper
        bb_m,         # 11. BB middle
        bb_l,         # 12. BB lower
        atr,          # 13. ATR
        obv,          # 14. OBV
        ma5,          # 15. MA5
        ma10,         # 16. MA10
        ma20,         # 17. MA20
    ], axis=-1)  # (T, 17)

    return features.astype(np.float32)


# --- Helpers ---

def _ema(data: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1)
    result = np.empty_like(data)
    result[0] = data[0]
    for t in range(1, len(data)):
        result[t] = alpha * data[t] + (1 - alpha) * result[t - 1]
    return result


def _sma(data: np.ndarray, period: int) -> np.ndarray:
    result = np.empty_like(data)
    for t in range(len(data)):
        start = max(0, t - period + 1)
        result[t] = np.mean(data[start : t + 1])
    return result


def _rolling_std(data: np.ndarray, period: int) -> np.ndarray:
    result = np.zeros_like(data)
    for t in range(len(data)):
        start = max(0, t - period + 1)
        result[t] = np.std(data[start : t + 1], ddof=0)
    return result
