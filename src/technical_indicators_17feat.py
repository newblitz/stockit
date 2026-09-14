"""17-dimensional "Price+TA+Text" price representation (paper §4.6, §4.8).

This is the feature builder for the *separate* 17-feature model pipeline
(``config_17feat.py`` / ``src/data_17feat.py`` / ``train_17feat.py``). It is
intentionally independent of ``src/technical_indicators.py`` (legacy, dead
code kept only for the already-obsolete ``src/dataset.py``) and must never be
imported by, or merged into, the canonical 6-feature pipeline
(``config.py`` / ``src/data.py`` / ``train.py``).

Paper composition (§4.6, Ablation study)
-----------------------------------------
    "Expanding from Close to six-dimensional OHLCV ... Incorporating 11
    technical indicators, denoted as TA, further boosts performance ...
    Combined with the 6-dimensional OHLCV features, this yields a
    17-dimensional price representation spanning momentum (returns, RSI,
    MACD), volatility (Bollinger Bands, ATR), trend (moving averages), and
    volume (OBV) dimensions."

This module resolves that description into exactly 17 columns:

    0     movement            (raw CMIN column 0; ~= close-to-close return)
    1-4   open/high/low/close returns (raw CMIN columns 1-4)
    5     volume              (raw CMIN column 5)
    6     RSI(14)
    7-9   MACD line, signal, histogram (12, 26, 9)
    10-11 Bollinger upper/lower band (20, 2 sigma)  [middle omitted: ~MA20]
    12    ATR(14)
    13    OBV
    14-16 MA5, MA10, MA20

Columns 0-5 are the paper's 6-dim OHLCV block (unchanged, identical to the
6-feature model's price input); columns 6-16 are the paper's "11 technical
indicators" (RSI=1, MACD=3, Bollinger=2, ATR=1, OBV=1, MAs=3 -> 11 total),
matching the paper's stated categories: momentum={RSI, MACD}, volatility=
{Bollinger, ATR}, trend={MAs}, volume={OBV}. "returns" in the paper's
category list refers to the raw OHLC return columns already present in
slots 1-4, not an additional feature.

Documented ambiguities / assumptions
-------------------------------------
``paper.md`` does not specify any of the following; these are best-effort,
standard-practice choices, made explicit here so they can be revisited if
the authors' source code or a more precise dataset spec ever surfaces:

1. **Indicator periods.** RSI(14), MACD(12, 26, 9), Bollinger(20, 2 sigma),
   ATR(14), and MA(5, 10, 20) are conventional technical-analysis defaults,
   not values given in the paper.

2. **Reconstructing a price level from returns.** CMIN's processed rows
   give *log returns*, not absolute price levels. To compute price-level
   indicators (RSI/MACD/Bollinger/ATR/MAs) at all, a relative "close level"
   series is reconstructed once per stock as the cumulative product of
   ``exp(close_return)``, anchored at 1.0 on the stock's first available
   trading day. This is a stand-in for the unavailable true price level
   (and, over ~1000 trading days, is a compounding index rather than a
   literal share price), not a reconstruction of the stock's actual price.

3. **OHLC column order.** The four return columns (raw indices 1-4) are
   assumed ordered Open, High, Low, Close. This is inferred, not
   documented, from: (a) it is the conventional OHLC reading order, and
   (b) empirically, raw column 0 ("movement") and raw column 4 are nearly
   numerically identical in the processed CMIN-US files (e.g. AAPL's first
   two rows differ only past the 4th significant digit), consistent with
   column 4 being the (unadjusted) close return and column 0 being an
   independently supplied close-to-close movement figure. A prior version
   of this feature builder (``src/technical_indicators.py``, now legacy)
   used columns 1 and 2 as High/Low proxies; this module instead uses
   columns 2 and 3, matching the OHLC order above. If the true column
   order differs, only the derived High/Low-dependent indicators (ATR,
   Bollinger via the high/low-nudged level) are affected — the raw
   OHLC-return columns themselves (slots 1-4) are unaffected either way,
   since they are passed through unchanged regardless of their true label.

4. **Daily high/low levels.** Needed only as inputs to ATR (and, loosely,
   as the volatility context Bollinger/ATR are meant to capture). Each
   day's high/low level is approximated as
   ``close_level[t] * exp(max(high_return[t], 0))`` and
   ``close_level[t] * exp(min(low_return[t], 0))`` respectively — the
   reconstructed close level for that day, nudged by that day's own
   high/low return. This keeps high >= close >= low by construction but
   is an approximation, not a reconstruction of true intraday OHLC prices.

5. **Scale.** Values are left on their natural scale here. Per-stock
   z-score normalization (training-period statistics only, identical to
   the 6-feature pipeline) is applied afterwards by
   ``src/data_17feat.py``, so the differing raw scales of, e.g., OBV vs.
   RSI are immaterial to the model's actual input.
"""

from __future__ import annotations

import numpy as np

# Raw column layout produced by ``src.data.prepared_prices()``: six values
# per trading day, identical to the paper's §4.2 processed CMIN row
# ("movement, four OHLC returns, volume").
_MOVEMENT = 0
_OPEN_RET, _HIGH_RET, _LOW_RET, _CLOSE_RET = 1, 2, 3, 4
_VOLUME = 5

NUM_RAW_COLUMNS = 6
NUM_TA_COLUMNS = 11
NUM_TOTAL_COLUMNS = NUM_RAW_COLUMNS + NUM_TA_COLUMNS  # 17


def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI (Relative Strength Index), normalized to [0, 1]."""
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = _ema(gain, period)
    avg_loss = _ema(loss, period)
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 100.0), where=avg_loss != 0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi / 100.0


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
    """On-Balance Volume, normalized by its own max magnitude."""
    delta = np.diff(close, prepend=close[0])
    direction = np.sign(delta)
    obv = np.cumsum(direction * volume)
    return obv / (np.abs(obv).max() + 1e-8)


def compute_ma(prices: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    return _sma(prices, period)


def compute_17_price_features(raw: np.ndarray) -> np.ndarray:
    """Expand the paper's 6-dim processed CMIN row into the full 17-dim
    "Price+TA+Text" representation (§4.6).

    ``raw`` must be shaped ``[days, 6]``: the same array returned as the
    second element of ``src.data.prepared_prices()`` (i.e. already
    forward-filled and 3-sigma-clipped per §4.2 -- this function does not
    repeat that preprocessing).

    Returns a ``[days, 17]`` float32 array whose first 6 columns are
    ``raw`` unchanged and whose remaining 11 columns are the technical
    indicators described in this module's docstring.
    """
    if raw.ndim != 2 or raw.shape[1] != NUM_RAW_COLUMNS:
        raise ValueError(f"expected CMIN price data shaped [days, {NUM_RAW_COLUMNS}]; got {raw.shape}")

    close_level = np.exp(np.cumsum(raw[:, _CLOSE_RET]))
    close_level[0] = 1.0
    high_level = close_level * np.exp(np.maximum(raw[:, _HIGH_RET], 0.0))
    low_level = close_level * np.exp(np.minimum(raw[:, _LOW_RET], 0.0))
    volume = raw[:, _VOLUME]

    rsi = compute_rsi(close_level)
    macd_line, macd_signal, macd_hist = compute_macd(close_level)
    bb_upper, _bb_mid, bb_lower = compute_bollinger(close_level)
    atr = compute_atr(high_level, low_level, close_level)
    obv = compute_obv(close_level, volume)
    ma5 = compute_ma(close_level, 5)
    ma10 = compute_ma(close_level, 10)
    ma20 = compute_ma(close_level, 20)

    technical = np.stack(
        [rsi, macd_line, macd_signal, macd_hist, bb_upper, bb_lower, atr, obv, ma5, ma10, ma20],
        axis=-1,
    )
    features = np.concatenate([raw, technical], axis=-1)
    assert features.shape[1] == NUM_TOTAL_COLUMNS
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
