"""Causal logistic-regression diagnostic for the cached Qwen3 -> BGE-M3 data.

This deliberately does not use ``CMINWindowDataset``: this is a one-day
diagnostic, whereas the model dataset contains 30-day windows.  It uses the
same raw CMIN TSVs, ticker universe, split boundaries, and raw-movement label
definition as ``src.data``.  A row at day t contains only information at or
before t and predicts the supplied close-to-close movement at t+1.

Example:
    python logistic_text_diagnostic.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


SPLITS = {
    "train": (date(2018, 1, 1), date(2020, 6, 30)),
    "val": (date(2020, 7, 1), date(2020, 12, 31)),
    "test": (date(2021, 1, 1), date(2021, 12, 31)),
}


@dataclass
class Rows:
    ticker: np.ndarray
    news_date: np.ndarray
    label_date: np.ndarray
    label: np.ndarray
    price: np.ndarray
    embedding: np.ndarray
    has_news: np.ndarray


def read_prices(path: Path) -> tuple[list[date], np.ndarray]:
    dates, rows = [], []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 7:
            raise ValueError(f"{path}:{line_no}: expected date plus six TSV values")
        dates.append(date.fromisoformat(fields[0]))
        rows.append([float(x) for x in fields[1:]])
    values = np.asarray(rows, dtype=np.float64)
    if not len(dates) or not np.isfinite(values).all():
        raise ValueError(f"{path}: empty or non-finite price data")
    return dates, values


def news_days(path: Path) -> set[str]:
    """Dates with at least one source-news hourly record, not a placeholder."""
    if not path.exists():
        raise ValueError(f"missing inspectable text cache {path}; cannot identify news rows for PCA")
    result = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("kind") == "hourly":
            result.add(record["date"])
    return result


def stationary_price_features(values: np.ndarray, t: int) -> np.ndarray:
    """Returns and trailing (past-and-current) statistics; never accesses t+1."""
    # The official fields are movement, four OHLC log returns, and volume.
    returns = values[:, :5]
    log_volume = np.log(np.maximum(values[:, 5], 1.0))
    volume_change = np.diff(log_volume, prepend=log_volume[0])

    def trailing(x: np.ndarray, width: int, statistic: str) -> float:
        window = x[max(0, t - width + 1) : t + 1]
        return float(window.mean() if statistic == "mean" else window.std(ddof=0))

    current = returns[t]
    # Return levels, absolute returns, volume change, and short/medium trailing
    # return and volume statistics are all stationary transformations.
    features = list(current) + list(np.abs(current)) + [float(volume_change[t])]
    for width in (5, 20):
        features.extend([
            trailing(returns[:, 0], width, "mean"),
            trailing(returns[:, 0], width, "std"),
            trailing(returns[:, 4], width, "mean"),
            trailing(returns[:, 4], width, "std"),
            trailing(volume_change, width, "mean"),
            trailing(volume_change, width, "std"),
        ])
    return np.asarray(features, dtype=np.float64)


def load_rows(dataset_root: Path, cache_root: Path) -> Rows:
    price_files = sorted((dataset_root / "price" / "processed").glob("*.txt"))
    cache_files = {p.stem: p for p in cache_root.glob("*.pt")}
    tickers = [p.stem for p in price_files]
    if not price_files:
        raise ValueError(f"no CMIN price files under {dataset_root}")
    if set(tickers) != set(cache_files):
        missing_cache = sorted(set(tickers) - set(cache_files))
        extra_cache = sorted(set(cache_files) - set(tickers))
        raise ValueError(f"ticker universe mismatch: missing caches={missing_cache[:10]}, extra caches={extra_cache[:10]}")

    ticker_col, news_col, label_day_col, labels, prices, embeddings, news_mask = ([] for _ in range(7))
    embedding_dim = None
    for price_file in price_files:
        ticker = price_file.stem
        dates, values = read_prices(price_file)
        payload = torch.load(cache_files[ticker], map_location="cpu", weights_only=False)
        if not {"dates", "embeddings"}.issubset(payload):
            raise ValueError(f"{cache_files[ticker]}: expected dates and embeddings")
        cache_dates = [date.fromisoformat(x) for x in payload["dates"]]
        embedding = np.asarray(payload["embeddings"], dtype=np.float64)
        # This is the key fail-fast alignment check: no nearest-date matching.
        if cache_dates != dates or embedding.shape[0] != len(dates):
            raise ValueError(f"{ticker}: cache dates do not exactly match price dates; stopping")
        if embedding.ndim != 2 or not np.isfinite(embedding).all():
            raise ValueError(f"{ticker}: invalid embedding matrix")
        if embedding_dim is None:
            embedding_dim = embedding.shape[1]
        elif embedding.shape[1] != embedding_dim:
            raise ValueError(f"{ticker}: embedding width differs from other tickers")
        source_news_days = news_days(cache_root / "texts" / f"{ticker}.jsonl")

        # t is the information/news/price date.  The raw movement at t+1 is
        # the repository's exact next-day close-direction label source.
        for t in range(len(dates) - 1):
            feature_day, target_day = dates[t], dates[t + 1]
            if not any(start <= feature_day and target_day <= end for start, end in SPLITS.values()):
                continue
            ticker_col.append(ticker)
            news_col.append(feature_day.isoformat())
            label_day_col.append(target_day.isoformat())
            labels.append(int(values[t + 1, 0] > 0.0))
            prices.append(stationary_price_features(values, t))
            embeddings.append(embedding[t])
            news_mask.append(feature_day.isoformat() in source_news_days)

    return Rows(*(np.asarray(x) for x in (ticker_col, news_col, label_day_col, labels, prices, embeddings, news_mask)))


def split_mask(rows: Rows, split: str) -> np.ndarray:
    start, end = SPLITS[split]
    days = np.asarray([date.fromisoformat(d) for d in rows.news_date])
    target_days = np.asarray([date.fromisoformat(d) for d in rows.label_date])
    return np.asarray([(start <= d <= end and start <= y <= end) for d, y in zip(days, target_days)])


def report_metrics(y: torch.Tensor, prediction: torch.Tensor) -> tuple[float, float, float]:
    """GPU metric calculation, returned only as three small Python scalars."""
    y, prediction = y.bool(), prediction.bool()
    tp = (y & prediction).sum().float()
    tn = (~y & ~prediction).sum().float()
    fp = (~y & prediction).sum().float()
    fn = (y & ~prediction).sum().float()
    total = tp + tn + fp + fn
    acc = (tp + tn) / total
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).sqrt()
    mcc = torch.where(denominator > 0, (tp * tn - fp * fn) / denominator, torch.zeros_like(denominator))
    return float(acc), float(mcc), float(prediction.float().mean())


def shards(values: np.ndarray, devices: list[torch.device], dtype=torch.float32) -> list[torch.Tensor]:
    """The only CPU work here is transfer from disk/RAM to the two GPUs."""
    chunks = np.array_split(values, len(devices))
    return [torch.as_tensor(chunk, dtype=dtype, device=device) for chunk, device in zip(chunks, devices)]


def gpu_scale(train: np.ndarray, *others: np.ndarray, devices: list[torch.device]) -> tuple[np.ndarray, ...]:
    """Fit StandardScaler-equivalent statistics on train across both GPUs."""
    train_parts = shards(train, devices)
    count = sum(part.shape[0] for part in train_parts)
    total = sum((part.sum(0).to(devices[0]) for part in train_parts))
    mean = total / count
    square_total = sum(((part * part).sum(0).to(devices[0]) for part in train_parts))
    std = (square_total / count - mean.square()).clamp_min(1e-12).sqrt()
    out = []
    for values in (train, *others):
        parts = shards(values, devices)
        # Scaling happens on each GPU.  Moving the compact transformed matrix
        # to host lets the next experiment reuse a single, simple interface.
        out.append(torch.cat([((part - mean.to(part.device)) / std.to(part.device)).cpu() for part in parts]).numpy())
    return tuple(out)


def dual_gpu_pca(train_embedding: np.ndarray, all_embedding: np.ndarray, devices: list[torch.device]) -> np.ndarray:
    """Exact covariance PCA(32), with row shards on both GPUs and a GPU reduction."""
    train_parts = shards(train_embedding, devices)
    count = sum(part.shape[0] for part in train_parts)
    mean = sum((part.sum(0).to(devices[0]) for part in train_parts)) / count
    covariance = sum((((part - mean.to(part.device)).T @ (part - mean.to(part.device))).to(devices[0]) for part in train_parts))
    # eigh is on GPU 0; the expensive 70k x 1024 covariance construction used both GPUs.
    _values, vectors = torch.linalg.eigh(covariance / (count - 1))
    components = vectors[:, -32:]
    result = []
    for part in shards(all_embedding, devices):
        result.append(((part - mean.to(part.device)) @ components.to(part.device)).cpu())
    return torch.cat(result).numpy()


def dual_gpu_fit_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, *, c: float,
    seed: int, devices: list[torch.device], epochs: int, learning_rate: float,
) -> np.ndarray:
    """Synchronous two-GPU logistic regression (both replicas receive averaged gradients)."""
    torch.manual_seed(seed)
    X_parts, y_parts = shards(X_train, devices), shards(y_train[:, None], devices)
    weights = [torch.zeros(X_train.shape[1], 1, device=d, requires_grad=True) for d in devices]
    biases = [torch.zeros(1, device=d, requires_grad=True) for d in devices]
    optimizers = [torch.optim.Adam([w, b], lr=learning_rate) for w, b in zip(weights, biases)]
    for _ in range(epochs):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        for X, y, w, b in zip(X_parts, y_parts, weights, biases):
            loss = F.binary_cross_entropy_with_logits(X @ w + b, y) + 0.5 * w.square().sum() / c
            loss.backward()
        # Average exact full-batch gradients, then apply identical Adam steps on both devices.
        grad_w = sum(w.grad.to(devices[0]) for w in weights) / len(weights)
        grad_b = sum(b.grad.to(devices[0]) for b in biases) / len(biases)
        for w, b in zip(weights, biases):
            w.grad.copy_(grad_w.to(w.device))
            b.grad.copy_(grad_b.to(b.device))
        for optimizer in optimizers:
            optimizer.step()
        with torch.no_grad():
            for w, b in zip(weights[1:], biases[1:]):
                w.copy_(weights[0].to(w.device))
                b.copy_(biases[0].to(b.device))
    probabilities = []
    for X in shards(X_eval, devices):
        probabilities.append(torch.sigmoid(X @ weights[0].to(X.device) + biases[0].to(X.device)).flatten().cpu())
    return torch.cat(probabilities).numpy()


def choose_logistic(X_train, y_train, X_val, y_val, seed: int, devices, epochs, learning_rate):
    """Select C/threshold using already train-fitted scaled data on the GPUs."""
    y_val_gpu = torch.as_tensor(y_val, device=devices[0])
    choices = []
    for c in (0.001, 0.01, 0.1, 1.0, 10.0):
        probability = dual_gpu_fit_predict(X_train, y_train, X_val, c=c, seed=seed, devices=devices, epochs=epochs, learning_rate=learning_rate)
        probability_gpu = torch.as_tensor(probability, device=devices[0])
        for threshold in np.arange(0.30, 0.701, 0.01):
            acc, mcc, _ = report_metrics(y_val_gpu, probability_gpu >= threshold)
            choices.append((mcc, acc, -abs(threshold - 0.5), c, threshold))
    return max(choices, key=lambda x: x[:5])


def causal_time_shuffle(rows: Rows, seed: int) -> np.ndarray:
    """Randomly remap each row to an embedding from its own stock's past.

    Sampling is split-local and only from dates <= t, so this control cannot
    import a future embedding or cross a train/validation/test boundary.
    """
    rng = np.random.default_rng(seed)
    result = np.empty_like(rows.embedding)
    for split in SPLITS:
        mask = split_mask(rows, split)
        for ticker in np.unique(rows.ticker[mask]):
            indices = np.flatnonzero(mask & (rows.ticker == ticker))
            # independent past draws break day-to-day pairing while remaining causal
            for offset, index in enumerate(indices):
                result[index] = rows.embedding[indices[rng.integers(offset + 1)]]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/CMIN-Dataset/CMIN-US"))
    parser.add_argument("--cache-root", type=Path, default=Path("new_cache/qwen3_8b_bge_m3"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", nargs=2, default=["cuda:0", "cuda:1"], metavar=("GPU0", "GPU1"),
                        help="Two Kaggle CUDA devices; CPU fallback is intentionally unsupported.")
    parser.add_argument("--epochs", type=int, default=500, help="Full-batch Adam steps per C candidate.")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("This diagnostic is configured for a Kaggle two-GPU session; enable two GPUs and rerun.")
    devices = [torch.device(name) for name in args.devices]
    if any(device.type != "cuda" or device.index is None or device.index >= torch.cuda.device_count() for device in devices):
        raise ValueError(f"invalid CUDA device pair: {args.devices}")

    rows = load_rows(args.dataset_root, args.cache_root)
    masks = {name: split_mask(rows, name) for name in SPLITS}
    if sum(mask.sum() for mask in masks.values()) != len(rows.label):
        raise AssertionError("a row fell outside the chronological splits")

    print("Alignment check (news/price are day t; label is the next trading day):")
    for index in np.linspace(0, len(rows.label) - 1, num=5, dtype=int):
        print(f"  {rows.ticker[index]:7s}  news={rows.news_date[index]}  price={rows.news_date[index]}  label={rows.label_date[index]}")
    print("\nClass balance:")
    for split, mask in masks.items():
        print(f"  {split:5s} n={mask.sum():6d}  positive={rows.label[mask].mean():.4f}  news-present={rows.has_news[mask].mean():.4f}")

    train, val, test = (masks[x] for x in ("train", "val", "test"))
    if rows.has_news[train].sum() < 32:
        raise ValueError("fewer than 32 train rows with source news; cannot fit requested PCA")
    print("\nFitting PCA(32) on train rows with source news only across two GPUs; fitting all scalers/C/thresholds on train/val only.")
    text = dual_gpu_pca(rows.embedding[train & rows.has_news], rows.embedding, devices)
    shuffled_text = dual_gpu_pca(rows.embedding[train & rows.has_news], causal_time_shuffle(rows, args.seed), devices)

    feature_sets = {
        "price-only": rows.price,
        "text-only": text,
        "text shuffled": shuffled_text,
        "price + text": np.hstack((rows.price, text)),
    }
    results = []
    majority = int(rows.label[train].mean() >= 0.5)
    y_val_gpu = torch.as_tensor(rows.label[val], device=devices[0])
    y_test_gpu = torch.as_tensor(rows.label[test], device=devices[0])
    results.append(("majority-class", *report_metrics(y_val_gpu, torch.full((val.sum(),), bool(majority), device=devices[0])), *report_metrics(y_test_gpu, torch.full((test.sum(),), bool(majority), device=devices[0])), "train majority"))

    for name, X in feature_sets.items():
        X_train, X_val, X_test = gpu_scale(X[train], X[val], X[test], devices=devices)
        _mcc, _acc, _tie, c, threshold = choose_logistic(X_train, rows.label[train], X_val, rows.label[val], args.seed, devices, args.epochs, args.learning_rate)
        # Refit only the validation-selected configuration once, then make one
        # combined held-out prediction pass.  Test probabilities are sliced out
        # only after C and the threshold have been fixed.
        held_out_probability = dual_gpu_fit_predict(X_train, rows.label[train], np.vstack((X_val, X_test)), c=c, seed=args.seed, devices=devices, epochs=args.epochs, learning_rate=args.learning_rate)
        val_probability, test_probability = held_out_probability[:len(X_val)], held_out_probability[len(X_val):]
        val_prediction = torch.as_tensor(val_probability >= threshold, device=devices[0])
        # The selected model makes its single held-out test prediction here.
        test_prediction = torch.as_tensor(test_probability >= threshold, device=devices[0])
        results.append((name, *report_metrics(y_val_gpu, val_prediction), *report_metrics(y_test_gpu, test_prediction), f"C={c:g}, thr={threshold:.2f}"))

    print("\n| experiment | val ACC | val MCC | val +rate | test ACC | test MCC | test +rate | selection |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for name, va, vm, vp, ta, tm, tp, selection in results:
        print(f"| {name} | {va:.4f} | {vm:.4f} | {vp:.4f} | {ta:.4f} | {tm:.4f} | {tp:.4f} | {selection} |")


if __name__ == "__main__":
    main()
