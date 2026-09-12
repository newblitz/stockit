"""Paper-style held-out evaluation, statistical aggregation, and backtesting."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def classification_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predicted = probabilities.reshape(-1) >= 0.5
    truth = labels.reshape(-1).astype(bool)
    tp = int(np.sum(predicted & truth)); tn = int(np.sum(~predicted & ~truth))
    fp = int(np.sum(predicted & ~truth)); fn = int(np.sum(~predicted & truth))
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "mcc": 0.0 if denominator == 0 else (tp * tn - fp * fn) / math.sqrt(denominator),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def simulate_long_only(
    probabilities: np.ndarray,
    next_log_returns: np.ndarray,
    *,
    threshold: float = 0.6,
    round_trip_cost: float = 0.001,
    min_holding_days: int = 3,
) -> dict[str, float]:
    """Paper §4.6.1 long/cash backtest with confidence-proportional sizing.

    Samples are evaluated independently in dataset order, so this is a benchmark
    of the paper's stated policy, not a portfolio with cross-stock capital
    constraints.  Costs are split evenly on entry and exit.
    """
    probs = probabilities.reshape(-1)
    returns = np.expm1(next_log_returns.reshape(-1))
    equity = 1.0
    daily: list[float] = []
    holding = 0
    position = 0.0
    wins = 0
    trades = 0
    for probability, market_return in zip(probs, returns):
        desired = float(np.clip((probability - threshold) / (1 - threshold), 0.0, 1.0))
        if holding > 0:
            holding -= 1
            desired = position
        entering = position == 0.0 and desired > 0.0
        exiting = position > 0.0 and desired == 0.0
        pnl = position * market_return
        if entering or exiting:
            pnl -= round_trip_cost / 2
        if entering:
            holding = max(min_holding_days - 1, 0)
            trades += 1
        if position > 0.0 and pnl > 0.0:
            wins += 1
        equity *= 1 + pnl
        daily.append(pnl)
        position = desired
    values = np.asarray(daily, dtype=float)
    annual_return = equity ** (252 / max(len(values), 1)) - 1
    volatility = values.std(ddof=1) * math.sqrt(252) if len(values) > 1 else 0.0
    sharpe = 0.0 if volatility == 0 else values.mean() * 252 / volatility
    curve = np.cumprod(1 + values)
    drawdown = curve / np.maximum.accumulate(curve) - 1 if len(curve) else np.zeros(1)
    return {
        "annual_return": float(annual_return), "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()), "win_rate": wins / max(trades, 1),
        "trades": float(trades), "final_equity": float(equity),
    }


def aggregate_runs(results: Sequence[dict[str, float]], metric_names: Sequence[str]) -> dict[str, float]:
    if not results:
        raise ValueError("at least one result is required")
    output: dict[str, float] = {"runs": float(len(results))}
    for name in metric_names:
        values = np.asarray([result[name] for result in results], dtype=float)
        output[f"{name}_mean"] = float(values.mean())
        output[f"{name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return output


def paired_t_test(left: Sequence[float], right: Sequence[float]) -> dict[str, float]:
    """Two-sided paired t-test; SciPy is deliberately required for valid p-values."""
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("paired tests require equally sized samples with at least two runs")
    try:
        from scipy.stats import ttest_rel
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install scipy to run statistical tests: pip install scipy") from exc
    statistic, pvalue = ttest_rel(left, right)
    return {"t_statistic": float(statistic), "p_value": float(pvalue), "n_pairs": float(len(left))}
