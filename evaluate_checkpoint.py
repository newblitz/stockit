"""Evaluate one selected checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from src.data import CMINWindowDataset
from src.evaluation import aggregate_runs, classification_metrics, simulate_long_only
from src.model import HierarchicalCoAttentionStockPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a paper-model checkpoint once on CMIN test data.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/CMIN-Dataset-official/CMIN-US"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/cmin-us-mt5"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--latency-runs", type=int, default=1_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_config = saved.get("config", {})
    config = Config(**saved_config)
    dataset = CMINWindowDataset(args.dataset_root, args.cache_root, "test", seq_len=config.seq_len)
    if dataset.price_dim != config.price_dim or dataset.text_embedding_dim != config.text_embedding_dim:
        raise ValueError("checkpoint, price features, and embedding cache dimensions must match exactly")
    device = torch.device(args.device)
    model = HierarchicalCoAttentionStockPredictor.from_config(config).to(device)
    model.load_state_dict(saved["model_state_dict"]); model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size)
    probabilities: list[np.ndarray] = []; labels: list[np.ndarray] = []; returns: list[np.ndarray] = []
    with torch.inference_mode():
        for price, text, label, next_return in loader:
            probabilities.append(model.predict_proba(price.to(device), text.to(device)).cpu().numpy())
            labels.append(label.numpy()); returns.append(next_return.numpy())
    probs, truth, next_returns = np.concatenate(probabilities), np.concatenate(labels), np.concatenate(returns)
    first_price, first_text, _, _ = next(iter(loader))
    first_price, first_text = first_price.to(device), first_text.to(device)
    with torch.inference_mode():
        for _ in range(10): model(first_price, first_text)
        if device.type == "cuda": torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(args.latency_runs): model(first_price, first_text)
        if device.type == "cuda": torch.cuda.synchronize()
    # Test data are ordered by ticker then time.  Backtest each stock's actual
    # timeline independently instead of accidentally joining one ticker's last
    # session to another ticker's first session.
    per_stock = []
    ticker_array = np.asarray(dataset.sample_tickers)
    for ticker in np.unique(ticker_array):
        indices = np.flatnonzero(ticker_array == ticker)
        per_stock.append(simulate_long_only(probs[indices], next_returns[indices]))
    trading = aggregate_runs(per_stock, ("annual_return", "sharpe", "max_drawdown", "win_rate"))
    result = {**classification_metrics(probs, truth), **trading,
              "latency_ms_per_batch": (time.perf_counter() - started) * 1_000 / args.latency_runs,
              "latency_batch_size": float(len(first_price)), "test_samples": float(len(dataset))}
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
