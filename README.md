# Hierarchical Text–Price Co-Attention for Stock Trend Prediction

This repository is a PyTorch implementation of the model described in
[`paper/paper.md`](paper/paper.md), *Integrative stock price trend prediction
via hierarchical LLM text processing and patch-based transformer with
co-attention* (Zhang, Dong, and Xu, 2026). It is set up to train on the public
[CMIN Dataset](https://github.com/BigRoddy/CMIN-Dataset), starting with
CMIN-US.

This document is the handoff document for future agents and contributors.
Read it before changing the training path: there are older, incompatible
files in `src/` (documented below), and the root-level pipeline is the
authoritative one.

## What is implemented

The canonical pipeline follows the paper in four stages:

```text
CMIN news JSONL                  CMIN processed price TSV
       |                                     |
hourly mT5 summaries                    per-stock z-score
       |                                     |
daily mT5 summary                         30-day window
       |                                     |
daily frozen mT5 embedding                 |
       +----------> price-to-text alignment+
                                      |
                  two single-channel patch Transformers
                                      |
                 stacked bidirectional co-attention
                                      |
             flatten -> linear classifier -> next-day up probability
```

The frozen summarizer is executed once when building an embedding cache, not
inside every training epoch. The neural encoder, co-attention, and classifier
are the trainable parts.

## Canonical files

| File | Purpose |
| --- | --- |
| `config.py` | Paper-default hyperparameters: 30 days, patch 10, stride 5, `d_model=128`, `d_ff=256`, 16 heads, 4 encoder layers, 2 co-attention blocks, dropout 0.2, 20 epochs. |
| `src/model.py` | Trainable architecture. `HierarchicalCoAttentionStockPredictor.forward()` returns logits, while `predict_proba()` applies sigmoid. |
| `src/summarization.py` | Frozen two-stage mT5 XL-Sum summarizer and mean-pooled encoder embedding extractor. |
| `src/data.py` | Official CMIN TSV reader, per-stock train-only standardization, chronological splits, and 30-day windows. |
| `prepare_embeddings.py` | Precomputes and saves a `[trading_days, 768]` embedding cache per ticker. |
| `train.py` | The **authoritative training entry point**: train/validation loop, metrics, logs, and checkpoints. |
| `tests/test_model.py` | Shape and backward-pass smoke tests. |
| `paper/paper.md` | Local copy of the paper and the source of architecture/hyperparameter decisions. |

### Legacy files — do not use for the current run

`main.py`, `src/dataset.py`, `src/technical_indicators.py`, and `src/train.py`
belong to an older, incompatible approach. They expect a `StockModel` class
and compute 17 engineered indicators, but `src/model.py` now exports
`HierarchicalCoAttentionStockPredictor` and the canonical CMIN pipeline uses
the official six numerical price columns. As written, `python main.py` is not
the supported command and may fail. Do not mix its data loader or checkpoints
with the root-level `train.py` pipeline.

## Repository file structure

```text
major/
├── README.md                    # Project overview and handoff notes (this file)
├── HOW_TO_RUN.md                # Step-by-step run instructions (local + Kaggle)
├── requirements.txt             # Python dependencies
├── config.py                    # Paper-default hyperparameters
│
├── prepare_embeddings.py        # Step 1: build frozen mT5 text-embedding caches
├── train.py                     # Step 2: train/validate the co-attention model
├── main.py                      # LEGACY — do not use
│
├── src/
│   ├── __init__.py
│   ├── model.py                 # HierarchicalCoAttentionStockPredictor (canonical)
│   ├── data.py                  # CMIN loader, splits, 30-day windows (canonical)
│   ├── summarization.py         # HierarchicalSummarizer / Algorithm 1 (canonical)
│   ├── dataset.py               # LEGACY — old data loader
│   ├── train.py                 # LEGACY — old training loop
│   └── technical_indicators.py  # LEGACY — 17 engineered indicators
│
├── tests/
│   └── test_model.py            # Shape and backward-pass smoke tests
│
├── paper/
│   ├── paper.md                 # Local copy of the paper
│   └── fig*.png                 # Paper figures
│
├── data/
│   ├── CMIN-Dataset-official/   # Official CMIN download (default dataset root)
│   │   └── CMIN-US/
│   │       ├── price/
│   │       │   ├── processed/   # <TICKER>.txt — one row per trading day
│   │       │   └── raw/         # Raw price CSVs (not used by train.py)
│   │       └── news/
│   │           ├── preprocessed/# <TICKER>/<YYYY-MM-DD> — JSONL news per day
│   │           └── raw/         # Raw news CSVs (not used by train.py)
│   └── cache/
│       └── cmin-us-mt5/         # Precomputed text embeddings (created by prepare_embeddings.py)
│           └── <TICKER>.pt      # {dates, embeddings [trading_days, 768]}
│
├── checkpoints/
│   └── cmin-us/                 # Training outputs (created by train.py)
│       ├── best.pt              # Best by validation MCC (primary checkpoint)
│       ├── best_accuracy.pt     # Best by validation accuracy
│       ├── last.pt              # Most recent epoch
│       ├── history.json         # Per-epoch metrics
│       ├── history.csv          # Same metrics, spreadsheet-friendly
│       └── metrics.json         # Final epoch summary
│
└── logs/                        # Optional training/embedding logs (--log-file)
```

Paths under `data/`, `checkpoints/`, and `logs/` are created at runtime and may
be empty in a fresh clone. Override dataset, cache, and checkpoint locations with
`--dataset-root`, `--cache-root`, and `--checkpoint-dir` (see
[`HOW_TO_RUN.md`](HOW_TO_RUN.md)).

## Model mapping to the paper

`src/model.py` implements the following paper components.

| Paper section/equations | Code | Result |
| --- | --- | --- |
| §3.2, Algorithm 1 | `HierarchicalSummarizer` | Hourly news → hourly summaries → daily summary → frozen daily embedding. |
| §3.3, Eq. 1–3 | `PriceTextAlignment` | Text is linearly projected to price width; price queries text through scaled dot-product cross-attention. |
| §3.3, Eq. 4–7 | `SingleChannelPatchTransformer` | Each price feature becomes a separate channel; overlap patches are projected, position-embedded, and self-attended. |
| §3.4, Eq. 8–13 | `BidirectionalCoAttention` | Text queries price and price queries text. Their outputs are fused, residual-connected to price, normalized, and passed through an FFN. |
| §3.5, Eq. 14–16 | classifier in `HierarchicalCoAttentionStockPredictor` | Feature, patch, and channel dimensions are flattened into one binary logit. Training uses `BCEWithLogitsLoss`. |

### Tensor shapes

For a batch of `B` samples with the CMIN-US processed files:

```text
prices           [B, 30, 6]
text_embeddings  [B, 30, 768]
aligned text     [B, 30, 6]
patch streams    [B * 6, 6, 128]  # six patches: floor((30 - 10) / 5) + 2
logits           [B, 1]
labels           [B, 1]
```

`text_embedding_dim` is inferred from cache files at training time, so a
different cached text encoder can be used as long as all ticker caches have
the same embedding width.

## Data layout and semantics

The complete official CMIN-US download is expected here:

```text
data/CMIN-Dataset-official/CMIN-US/
├── price/processed/<TICKER>.txt
└── news/preprocessed/<TICKER>/<YYYY-MM-DD>
```

The official processed price files have a date and six numerical values. The
first value is the supplied close-to-close movement percentage and is used to
make the next-day label:

```text
label = 1 if next_day_raw_movement > 0 else 0
```

It is important that labels use the **raw**, unstandardized movement; this is
handled in `src/data.py`. All six input price columns are z-scored separately
for each stock using only dates through 2020-06-30. This prevents test-period
normalization leakage.

The target-day chronological split is:

| Split | Target date range |
| --- | --- |
| Train | 2018-01-01 to 2020-06-30 |
| Validation | 2020-07-01 to 2020-12-31 |
| Test | 2021-01-01 to 2021-12-31 |

A validation/test window can contain earlier historical days, but its label is
only included in the named target range. This preserves useful lookback
context without leaking future observations.

## Setup

Use Python 3.10+ and install dependencies:

```bash
python -m pip install -r requirements.txt
```

The summarizer needs internet access the first time it downloads
`csebuetnlp/mT5_multilingual_XLSum`. This is the valid public Hugging Face
identifier. The paper/config’s earlier `...-base-cased` form was invalid and
has been corrected. An `HF_TOKEN` is optional, but avoids Hugging Face
anonymous rate limits.

If CMIN-US is not already present, fetch the official public dataset without
overwriting another local copy:

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/BigRoddy/CMIN-Dataset.git data/CMIN-Dataset-official
git -C data/CMIN-Dataset-official sparse-checkout set CMIN-US
```

CMIN-CN is not yet wired as a one-command default, although the loader accepts
any root with the same `price/processed` and `news/preprocessed` layout. To
use it, sparsely fetch `CMIN-CN` and supply explicit `--dataset-root`,
`--cache-root`, and `--checkpoint-dir` paths.

## Runbook

### 1. Verify code before a long job

```bash
python -m pytest -q
```

Expected result: the model shape/backward tests pass.

### 2. Build text-embedding caches

Embedding creation is the slowest stage because it invokes a frozen mT5 model
twice per trading day. It creates a resumable per-ticker artifact:

```text
data/cache/cmin-us-mt5/AAPL.pt
```

Start with one ticker:

```bash
python prepare_embeddings.py --ticker AAPL --device cpu
```

On a CUDA-capable machine, omit `--device cpu` or specify `--device cuda`.
To cache every CMIN-US ticker, run:

```bash
python prepare_embeddings.py
```

Existing cache files are skipped. Do not delete a cache unless you deliberately
want to regenerate it with different prompts/model settings.

### 3. Train

An AAPL-only smoke/training run (the first ticker alphabetically) is useful to
verify the complete workflow:

```bash
python train.py --max-stocks 1 --log-file logs/cmin-us-aapl-training.log
```

After text caches have been prepared for all 110 tickers, run the full CMIN-US
experiment:

```bash
python train.py --log-file logs/cmin-us-training.log
```

Useful overrides:

```bash
python train.py --epochs 5 --batch-size 32 --device cpu --max-stocks 1
```

### 4. Monitor an active run

```bash
tail -f logs/cmin-us-aapl-training.log
ls -lh checkpoints/cmin-us
```

Epoch output includes elapsed time, learning rate, train/validation loss,
accuracy, and Matthews correlation coefficient (MCC). Example:

```text
Epoch 003/20 | 24.8s | train loss 0.6812, acc 0.5530, MCC 0.1021 |
val loss 0.6694, acc 0.5897, MCC 0.1782 | lr 1.00e-04 |
saved best.pt (MCC), saved best_accuracy.pt
```

If a prior agent started background preparation/training, first inspect it
instead of launching a duplicate job:

```bash
ps -ax | rg 'prepare_embeddings|train.py'
```

## Training artifacts and checkpoint policy

The root training script creates the supplied checkpoint directory (default:
`checkpoints/cmin-us`) and writes:

| Artifact | Meaning |
| --- | --- |
| `last.pt` | Latest completed epoch; overwritten each epoch. |
| `best.pt` | Model selected by highest **validation MCC**. This is the primary best checkpoint because MCC is a paper metric and is robust to class imbalance. |
| `best_accuracy.pt` | Model selected by highest validation accuracy. |
| `history.json` | Full list of epoch metric dictionaries, rewritten after every epoch. |
| `history.csv` | Spreadsheet-friendly version of the same full history. |
| `metrics.json` | Final epoch summary. |

Every `.pt` checkpoint contains `model_state_dict`, `optimizer_state_dict`,
the serialized `Config`, the epoch summary, and metric history. Restore the
primary checkpoint as follows (construct with the same dimensions first):

```python
import torch
from src.model import HierarchicalCoAttentionStockPredictor

checkpoint = torch.load("checkpoints/cmin-us/best.pt", map_location="cpu", weights_only=False)
model = HierarchicalCoAttentionStockPredictor(
    text_embedding_dim=checkpoint["config"]["text_embedding_dim"],
    price_dim=checkpoint["config"]["price_dim"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

When restoring a model trained from data, prefer reading the dimensions from a
cache file/dataset or the checkpoint’s saved configuration. If an experiment
used a non-default architecture, recreate it with all matching arguments
(`seq_len`, patch settings, widths, heads, layers, and dropout) before loading.

## Important implementation choices and limitations

- The paper describes equations and hyperparameters but does not publish the
  authors’ source implementation. This code is a faithful, documented PyTorch
  interpretation, not a byte-for-byte official reproduction.
- `SingleChannelPatchTransformer` uses right replication padding by `stride`.
  This produces the paper’s patch-count formula: `floor((L - P) / S) + 2`.
- The paper does not prescribe generation length; summarization uses a bounded
  `summary_max_new_tokens=64` to make cache creation tractable.
- mT5 embedding generation on CPU is expensive. A GPU is strongly recommended
  for full CMIN-US/CMIN-CN preprocessing. Caches are deliberately separate
  from source data and can be reused between neural-model experiments.
- The root `train.py` currently builds train and validation datasets. It does
  not yet load/evaluate the test split after selecting `best.pt`; add that only
  after the training/checkpoint workflow is stable, so the held-out test set is
  not accidentally used for model selection.
- CMIN-US is the active default. CMIN-CN requires downloading its separate
  directory and building a separate text cache/checkpoint path.

## Current handoff checklist

Before making changes, a future agent should:

1. Read this README and `paper/paper.md`.
2. Use root-level `prepare_embeddings.py` and `train.py`, not the legacy
   `main.py` path.
3. Check for active embedding/training processes before starting new ones.
4. Preserve `data/CMIN-Dataset-official` and cached `.pt` files; they are
   large, expensive to regenerate, and may have been created by a prior agent.
5. Keep validation MCC as the primary checkpoint-selection criterion unless a
   deliberate experiment changes the protocol.
