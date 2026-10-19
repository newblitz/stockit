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
import gc
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

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
    price_date: np.ndarray
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


def load_rows(dataset_root: Path, cache_root: Path, *, same_day: bool = False) -> Rows:
    price_files = sorted((dataset_root / "price" / "processed").glob("*.txt"))
    cache_files = {p.stem: p for p in cache_root.glob("*.pt")}
    tickers = [p.stem for p in price_files]
    if not price_files:
        raise ValueError(f"no CMIN price files under {dataset_root}")
    if set(tickers) != set(cache_files):
        missing_cache = sorted(set(tickers) - set(cache_files))
        extra_cache = sorted(set(cache_files) - set(tickers))
        raise ValueError(f"ticker universe mismatch: missing caches={missing_cache[:10]}, extra caches={extra_cache[:10]}")

    ticker_col, news_col, price_day_col, label_day_col, labels, prices, embeddings, news_mask = ([] for _ in range(8))
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
        start_index = 1 if same_day else 0
        stop_index = len(dates) if same_day else len(dates) - 1
        for t in range(start_index, stop_index):
            # Next-day: news/price t -> label t+1.  Same-day control: news t
            # -> label t, while price features stop at t-1.
            feature_day = dates[t]
            price_day = dates[t - 1] if same_day else dates[t]
            target_day = dates[t] if same_day else dates[t + 1]
            if not any(start <= price_day and feature_day <= end and target_day <= end for start, end in SPLITS.values()):
                continue
            ticker_col.append(ticker)
            news_col.append(feature_day.isoformat())
            price_day_col.append(price_day.isoformat())
            label_day_col.append(target_day.isoformat())
            labels.append(int(values[t if same_day else t + 1, 0] > 0.0))
            prices.append(stationary_price_features(values, t - 1 if same_day else t))
            embeddings.append(embedding[t])
            news_mask.append(feature_day.isoformat() in source_news_days)

    return Rows(*(np.asarray(x) for x in (ticker_col, news_col, price_day_col, label_day_col, labels, prices, embeddings, news_mask)))


def split_mask(rows: Rows, split: str) -> np.ndarray:
    start, end = SPLITS[split]
    days = np.asarray([date.fromisoformat(d) for d in rows.news_date])
    price_days = np.asarray([date.fromisoformat(d) for d in rows.price_date])
    target_days = np.asarray([date.fromisoformat(d) for d in rows.label_date])
    return np.asarray([(start <= p <= end and start <= d <= end and start <= y <= end) for p, d, y in zip(price_days, days, target_days)])


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


def global_day_shuffle(rows: Rows, seed: int) -> np.ndarray:
    """Permute complete (ticker, day) embedding observations globally.

    Each split is shuffled independently, but every source embedding may come
    from any ticker and any date in that split.  This deliberately destroys
    both temporal pairing and stock identity; no source row is reused.
    """
    rng = np.random.default_rng(seed)
    result = np.empty_like(rows.embedding)
    for split in SPLITS:
        mask = split_mask(rows, split)
        indices = np.flatnonzero(mask)
        result[indices] = rows.embedding[rng.permutation(indices)]
    return result


def timestamp_bucket_cache(
    dataset_root: Path, cache_root: Path, bucket: str, devices: list[torch.device]
) -> dict[str, np.ndarray]:
    """Load or create Qwen-summary/BGE embeddings split at the NYSE close.

    The daily cache cannot be split retrospectively: it contains only one
    post-Qwen BGE vector per day.  This function therefore rebuilds separate
    daily synopses from the timestamped source JSONL and persists them under
    ``timestamp_buckets/{before_close,after_close}`` for subsequent runs.
    """
    if bucket not in {"before_close", "after_close"}:
        raise ValueError(bucket)
    output = cache_root / "timestamp_buckets" / f"{bucket}.pt"
    if output.exists():
        payload = torch.load(output, map_location="cpu", weights_only=False)
        if payload.get("bucket") == bucket:
            return {ticker: np.asarray(value, dtype=np.float64) for ticker, value in payload["embeddings"].items()}

    # Source CMIN-US timestamps are documented and used elsewhere in this
    # repository as UTC; convert them to America/New_York before the 16:00
    # regular-session cutoff.  We do not shift an after-close item here: its
    # local publication date t is paired with the next-day label t+1.
    from config import Config
    from src.news_preprocessing import clean_text, is_near_duplicate
    from src.text_backends import build_embedder, build_summarizer

    exchange_tz, close = ZoneInfo("America/New_York"), time(16, 0)
    price_files = sorted((dataset_root / "price" / "processed").glob("*.txt"))
    prepared: dict[str, tuple[list[str], list[list[str]]]] = {}
    for price_file in price_files:
        ticker = price_file.stem
        days, _ = read_prices(price_file)
        by_day: dict[date, dict[str, list[str]]] = {}
        news_dir = dataset_root / "news" / "preprocessed" / ticker
        for path in news_dir.glob("*") if news_dir.exists() else []:
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                try:
                    published = datetime.fromisoformat(record["created_at"]).replace(tzinfo=timezone.utc).astimezone(exchange_tz)
                except (KeyError, TypeError, ValueError):
                    continue
                is_before = published.time() <= close
                if (bucket == "before_close") != is_before or published.date() not in days:
                    continue
                raw = record.get("text", "")
                text = clean_text(" ".join(raw) if isinstance(raw, list) else str(raw))
                if len(text) < 50:
                    continue
                by_day.setdefault(published.date(), {}).setdefault(published.strftime("%H"), []).append(text)
        docs = []
        for day in days:
            hour_texts = []
            kept: list[str] = []
            for hour in sorted(by_day.get(day, {})):
                unique = []
                for text in by_day[day][hour]:
                    if not is_near_duplicate(text, kept):
                        kept.append(text)
                        unique.append(text)
                if unique:
                    hour_texts.append(" ".join(unique))
            docs.append(hour_texts)
        prepared[ticker] = ([d.isoformat() for d in days], docs)

    # Qwen's vLLM backend tensor-parallelizes over both Kaggle GPUs.  Release
    # it before BGE-M3 is initialized, avoiding two large models competing for
    # the same memory.
    config = Config()
    summarizer = build_summarizer(config, device=str(devices[0]))
    synopses: dict[str, list[str]] = {}
    for ticker, (_dates, per_day_docs) in prepared.items():
        flat = [document for docs in per_day_docs for document in docs]
        hourly = summarizer.summarize_hourly(flat, ticker=ticker) if flat else []
        cursor, per_day_hourly = 0, []
        for docs in per_day_docs:
            per_day_hourly.append(hourly[cursor : cursor + len(docs)])
            cursor += len(docs)
        synopses[ticker] = summarizer.summarize_daily(per_day_hourly, ticker=ticker)
    del summarizer
    gc.collect()
    for device in devices:
        torch.cuda.empty_cache()

    # BGE-M3 is frozen.  It uses one GPU at a time; Qwen above used both GPUs
    # for the costly summary generation.  Cached output makes this a one-time job.
    embedder = build_embedder(config, device=str(devices[0]))
    embeddings = {ticker: embedder.encode(texts) for ticker, texts in synopses.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"bucket": bucket, "dates": {ticker: dates for ticker, (dates, _docs) in prepared.items()}, "embeddings": embeddings}, output)
    return {ticker: np.asarray(value, dtype=np.float64) for ticker, value in embeddings.items()}


def bucket_embeddings_for_rows(rows: Rows, cache: dict[str, np.ndarray], dataset_root: Path) -> np.ndarray:
    """Join a timestamp-bucket cache to rows by exact ticker/date, failing closed."""
    positions = {}
    for price_file in (dataset_root / "price" / "processed").glob("*.txt"):
        dates, _ = read_prices(price_file)
        positions[price_file.stem] = {day.isoformat(): i for i, day in enumerate(dates)}
    output = []
    for ticker, news_day in zip(rows.ticker, rows.news_date):
        if ticker not in cache or news_day not in positions[ticker]:
            raise ValueError(f"timestamp bucket cache does not align with {ticker} {news_day}")
        output.append(cache[ticker][positions[ticker][news_day]])
    return np.asarray(output, dtype=np.float64)


def has_publication_timestamps(dataset_root: Path) -> bool:
    """Check raw source records before attempting the costly bucket rebuild."""
    for path in (dataset_root / "news" / "preprocessed").glob("*/*"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines()[:10]:
            try:
                if json.loads(line).get("created_at"):
                    return True
            except json.JSONDecodeError:
                continue
    return False


def lightgbm_gpu_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray, str]:
    """Two-GPU Dask LightGBM with validation-MCC early stopping.

    Kaggle's stock LightGBM wheel is often CPU-only.  This intentionally fails
    rather than silently using CPU; install a CUDA-enabled LightGBM/Dask-CUDA
    image before running if those imports are unavailable.
    """
    try:
        import dask.array as da
        from dask.distributed import Client
        from dask_cuda import LocalCUDACluster
        import lightgbm as lgb
        from lightgbm.dask import DaskLGBMClassifier
    except ImportError as exc:
        raise RuntimeError("LightGBM GPU diagnostic requires dask-cuda and a CUDA-enabled lightgbm build in Kaggle") from exc

    def mcc_metric(y_true, probability):
        # LightGBM invokes this only on the validation fold for early stopping.
        prediction = np.asarray(probability) >= 0.5
        truth = np.asarray(y_true).astype(bool)
        tp, tn = np.sum(truth & prediction), np.sum(~truth & ~prediction)
        fp, fn = np.sum(~truth & prediction), np.sum(truth & ~prediction)
        denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        return "mcc", 0.0 if denom == 0 else (tp * tn - fp * fn) / denom, True

    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES="0,1")
    client = Client(cluster)
    try:
        chunks = max(1, len(X_train) // 2)
        train_x = da.from_array(X_train, chunks=(chunks, X_train.shape[1]))
        train_y = da.from_array(y_train, chunks=(chunks,))
        val_x = da.from_array(X_val, chunks=(max(1, len(X_val) // 2), X_val.shape[1]))
        val_y = da.from_array(y_val, chunks=(max(1, len(X_val) // 2),))
        model = DaskLGBMClassifier(
            objective="binary", metric="None", device_type="gpu", tree_learner="data",
            n_estimators=2000, learning_rate=0.03, num_leaves=31, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, random_state=seed, n_jobs=2,
        )
        model.fit(train_x, train_y, eval_set=[(val_x, val_y)], eval_metric=mcc_metric,
                  callbacks=[lgb.early_stopping(80, first_metric_only=True, verbose=False)])
        val_probability = model.predict_proba(val_x)[:, 1].compute()
        test_x = da.from_array(X_test, chunks=(max(1, len(X_test) // 2), X_test.shape[1]))
        test_probability = model.predict_proba(test_x)[:, 1].compute()
        return np.asarray(val_probability), np.asarray(test_probability), f"best_iter={model.best_iteration_}"
    finally:
        client.close()
        cluster.close()


def select_threshold(y: np.ndarray, probability: np.ndarray, device: torch.device) -> float:
    truth, scores = torch.as_tensor(y, device=device), torch.as_tensor(probability, device=device)
    choices = []
    for threshold in np.arange(0.30, 0.701, 0.01):
        acc, mcc, _ = report_metrics(truth, scores >= threshold)
        choices.append((mcc, acc, -abs(threshold - 0.5), float(threshold)))
    return max(choices)[3]


def logistic_result(name, X, rows: Rows, masks, seed, devices, epochs, learning_rate):
    train, val, test = (masks[key] for key in ("train", "val", "test"))
    X_train, X_val, X_test = gpu_scale(X[train], X[val], X[test], devices=devices)
    _mcc, _acc, _tie, c, threshold = choose_logistic(
        X_train, rows.label[train], X_val, rows.label[val], seed, devices, epochs, learning_rate
    )
    held_out = dual_gpu_fit_predict(
        X_train, rows.label[train], np.vstack((X_val, X_test)), c=c, seed=seed,
        devices=devices, epochs=epochs, learning_rate=learning_rate,
    )
    val_probability, test_probability = held_out[:val.sum()], held_out[val.sum():]
    y_val, y_test = torch.as_tensor(rows.label[val], device=devices[0]), torch.as_tensor(rows.label[test], device=devices[0])
    return (name, *report_metrics(y_val, torch.as_tensor(val_probability >= threshold, device=devices[0])),
            *report_metrics(y_test, torch.as_tensor(test_probability >= threshold, device=devices[0])),
            f"seed={seed}; C={c:g}, thr={threshold:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/CMIN-Dataset/CMIN-US"))
    parser.add_argument("--cache-root", type=Path, default=Path("new_cache/qwen3_8b_bge_m3"))
    parser.add_argument("--seeds", type=int, nargs=3, default=[13, 42, 2026],
                        help="Exactly three seeds; reported separately for every experiment.")
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

    def describe(label: str, rows: Rows) -> dict[str, np.ndarray]:
        masks = {name: split_mask(rows, name) for name in SPLITS}
        if sum(mask.sum() for mask in masks.values()) != len(rows.label):
            raise AssertionError(f"{label}: a row fell outside the chronological splits")
        print(f"\n{label} alignment (news=t, price feature date, label date):")
        for index in np.linspace(0, len(rows.label) - 1, num=5, dtype=int):
            print(f"  {rows.ticker[index]:7s}  news={rows.news_date[index]}  price={rows.price_date[index]}  label={rows.label_date[index]}")
        print("Class balance:")
        for split, mask in masks.items():
            print(f"  {split:5s} n={mask.sum():6d} positive={rows.label[mask].mean():.4f} news-present={rows.has_news[mask].mean():.4f}")
        return masks

    next_rows, same_rows = load_rows(args.dataset_root, args.cache_root), load_rows(args.dataset_root, args.cache_root, same_day=True)
    next_masks, same_masks = describe("Next-day", next_rows), describe("Same-day control", same_rows)

    def pca_text(rows: Rows, masks: dict[str, np.ndarray]) -> np.ndarray:
        train = masks["train"]
        if rows.has_news[train].sum() < 32:
            raise ValueError("fewer than 32 train rows with source news; cannot fit requested PCA")
        return dual_gpu_pca(rows.embedding[train & rows.has_news], rows.embedding, devices)

    print("\nPCA(32), scaling, C and thresholds use train/validation only. Test predictions are made after selection.")
    next_text, same_text = pca_text(next_rows, next_masks), pca_text(same_rows, same_masks)
    results = []
    for seed in args.seeds:
        # Original next-day panel: requested ablations plus a GPU LightGBM model.
        train, val, test = (next_masks[k] for k in ("train", "val", "test"))
        majority = int(next_rows.label[train].mean() >= 0.5)
        results.append(("next-day majority-class", *report_metrics(torch.as_tensor(next_rows.label[val], device=devices[0]), torch.full((val.sum(),), bool(majority), device=devices[0])), *report_metrics(torch.as_tensor(next_rows.label[test], device=devices[0]), torch.full((test.sum(),), bool(majority), device=devices[0])), f"seed={seed}; train majority"))
        results.extend([
            logistic_result("next-day price-only", next_rows.price, next_rows, next_masks, seed, devices, args.epochs, args.learning_rate),
            logistic_result("next-day text-only", next_text, next_rows, next_masks, seed, devices, args.epochs, args.learning_rate),
            logistic_result("next-day shuffled text", dual_gpu_pca(next_rows.embedding[next_masks["train"] & next_rows.has_news], global_day_shuffle(next_rows, seed), devices), next_rows, next_masks, seed, devices, args.epochs, args.learning_rate),
            logistic_result("next-day price + text", np.hstack((next_rows.price, next_text)), next_rows, next_masks, seed, devices, args.epochs, args.learning_rate),
            logistic_result("same-day price-only (t-1)", same_rows.price, same_rows, same_masks, seed, devices, args.epochs, args.learning_rate),
            logistic_result("same-day text-only", same_text, same_rows, same_masks, seed, devices, args.epochs, args.learning_rate),
            logistic_result("same-day shuffled text", dual_gpu_pca(same_rows.embedding[same_masks["train"] & same_rows.has_news], global_day_shuffle(same_rows, seed), devices), same_rows, same_masks, seed, devices, args.epochs, args.learning_rate),
        ])
        lgb_val, lgb_test, detail = lightgbm_gpu_predict(np.hstack((next_rows.price, next_text))[train], next_rows.label[train], np.hstack((next_rows.price, next_text))[val], next_rows.label[val], np.hstack((next_rows.price, next_text))[test], seed=seed)
        threshold = select_threshold(next_rows.label[val], lgb_val, devices[0])
        results.append(("next-day LightGBM price + text", *report_metrics(torch.as_tensor(next_rows.label[val], device=devices[0]), torch.as_tensor(lgb_val >= threshold, device=devices[0])), *report_metrics(torch.as_tensor(next_rows.label[test], device=devices[0]), torch.as_tensor(lgb_test >= threshold, device=devices[0])), f"seed={seed}; {detail}; thr={threshold:.2f}"))

    if has_publication_timestamps(args.dataset_root):
        # Timestamped CMIN source records are available.  Rebuild bucket-specific
        # Qwen-summary/BGE caches once, then run next-day text-only diagnostics.
        for bucket in ("before_close", "after_close"):
            bucket_text = bucket_embeddings_for_rows(next_rows, timestamp_bucket_cache(args.dataset_root, args.cache_root, bucket, devices), args.dataset_root)
            bucket_pca = dual_gpu_pca(bucket_text[next_masks["train"] & next_rows.has_news], bucket_text, devices)
            for seed in args.seeds:
                results.append(logistic_result(f"next-day text-only ({bucket})", bucket_pca, next_rows, next_masks, seed, devices, args.epochs, args.learning_rate))
    else:
        print("\nPublication timestamps are unavailable; skipping before-close/after-close text experiments.")

    print("\n| experiment | val ACC | val MCC | val +rate | test ACC | test MCC | test +rate | selection |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for name, va, vm, vp, ta, tm, tp, selection in results:
        print(f"| {name} | {va:.4f} | {vm:.4f} | {vp:.4f} | {ta:.4f} | {tm:.4f} | {tp:.4f} | {selection} |")


if __name__ == "__main__":
    main()
