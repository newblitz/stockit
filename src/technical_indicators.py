import numpy as np


def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI (Relative Strength Index)."""
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = _ema(gain, period)
    avg_loss = _ema(loss, period)
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 100.0), where=avg_loss != 0)
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
    """Return the paper's six CMIN price fields plus eleven TA features.

    ``raw`` is the six numeric columns in a processed CMIN row: daily movement,
    four OHLC log returns, and volume.  The close-return column is column four;
    its cumulative exponential gives a stable relative close series for indicators.
    """
    if raw.ndim != 2 or raw.shape[1] != 6:
        raise ValueError("expected CMIN price data shaped [days, 6]")
    returns = raw[:, :5]
    volume = raw[:, 5]

    # Reconstruct approximate prices from close returns (column 4, index 4)
    close_returns = returns[:, 4]
    close_prices = np.exp(np.cumsum(close_returns))
    close_prices[0] = 1.0

    # Use close as proxy for OHL (since we don't have raw OHLCV)
    high_prices = close_prices * np.exp(np.maximum(returns[:, 1], 0.0))
    low_prices = close_prices * np.exp(np.minimum(returns[:, 2], 0.0))

    # Compute indicators
    rsi = compute_rsi(close_prices)
    macd_l, macd_s, macd_h = compute_macd(close_prices)
    bb_u, _bb_m, bb_l = compute_bollinger(close_prices)
    atr = compute_atr(high_prices, low_prices, close_prices)
    obv = compute_obv(close_prices, volume)
    ma5 = compute_ma(close_prices, 5)
    ma10 = compute_ma(close_prices, 10)
    ma20 = compute_ma(close_prices, 20)

    # Stack all 17 features
    features = np.stack([
        raw[:, 0],    # 1. movement / close-to-close return
        raw[:, 1],    # 2--5. OHLC returns
        raw[:, 2],
        raw[:, 3],
        raw[:, 4],
        raw[:, 5],    # 6. volume
        rsi,          # 7. RSI
        macd_l,       # 8. MACD line
        macd_s,       # 9. MACD signal
        macd_h,       # 10. MACD histogram
        bb_u,         # 11. BB upper
        bb_l,         # 12. BB lower (middle duplicates MA20 below)
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
