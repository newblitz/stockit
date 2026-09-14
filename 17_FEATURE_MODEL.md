# 17-feature ("Price+TA+Text") model — parallel pipeline

This is the paper's **full model** from §4.6/§4.8 (`Price+TA+Text`, 17-dim
price input): 6 raw CMIN columns (movement, 4 OHLC returns, volume) + 11
technical indicators (RSI, MACD×3, Bollinger×2, ATR, OBV, MA5/10/20).
It reports the paper's headline result: 67.01% ACC / 0.346 MCC on CMIN-US.

It lives entirely alongside the existing 6-feature ("Price+Text") pipeline
and **never modifies, imports destructively, or overwrites** any of its
files. Both models can be trained, checkpointed, and compared independently.

## New files (17-feature pipeline)

| File | Mirrors | Purpose |
|---|---|---|
| `config_17feat.py` | `config.py` | `Config17Feature` — same Table 2/§4.8 hyperparameters, `price_dim=17` |
| `src/technical_indicators_17feat.py` | — | Builds the 17-dim feature vector from the 6 raw CMIN columns |
| `src/data_17feat.py` | `src/data.py` | `CMINWindowDataset17Feat` — same splits/leakage rules, 17-dim price windows |
| `train_17feat.py` | `train.py` | Training entry point; imports loss/metric/checkpoint helpers from `train.py` unchanged |
| `evaluate_checkpoint_17feat.py` | `evaluate_checkpoint.py` | Held-out test evaluation for a 17-feature checkpoint |
| `run_ten_seeds_17feat.py` | `run_ten_seeds.py` | Multi-seed runner, calls `train_17feat.py` |
| `tests/test_technical_indicators_17feat.py`, `tests/test_model_17feat.py` | — | Shape/sanity coverage for the new code |

**Nothing else changed.** `config.py`, `src/model.py`, `src/data.py`,
`train.py`, `evaluate_checkpoint.py`, `run_ten_seeds.py`,
`prepare_embeddings.py`, and every existing checkpoint remain byte-for-byte
untouched.

## No new embedding step needed

The mT5 text embeddings cached by `prepare_embeddings.py` are pure text
embeddings — completely independent of how many price features are used.
**The existing `data/cache/cmin-us-mt5/*.pt` caches are reused unchanged.**
`src/data_17feat.py` loads the exact same cache files as `src/data.py`; it
only expands the *price* side (6 → 17 columns) before z-score normalization.
You do not need to re-run or duplicate `prepare_embeddings.py` for this.

## Where the model architecture code is shared

`src/model.py`'s `HierarchicalCoAttentionStockPredictor` was already
generic in `price_dim` (never hardcoded to 6), so it is reused as-is for
both pipelines — no duplicate model file exists or is needed. Only the
*input width* differs between the two configs.

## Resolving the paper's ambiguity (§4.6's 17-dim description)

`paper.md` says: *"Combined with the 6-dimensional OHLCV features, this
yields a 17-dimensional price representation spanning momentum (returns,
RSI, MACD), volatility (Bollinger Bands, ATR), trend (moving averages), and
volume (OBV) dimensions."* It does not give exact formulas, periods, or
column semantics. Full documented reasoning is in the module docstring of
`src/technical_indicators_17feat.py`; summary:

- **11 TA columns** = RSI(1) + MACD line/signal/histogram(3) + Bollinger
  upper/lower(2, middle omitted since it ≈ MA20) + ATR(1) + OBV(1) +
  MA5/MA10/MA20(3) = 11, matching the paper's stated category breakdown and
  count exactly.
- **Indicator periods** (RSI 14, MACD 12/26/9, Bollinger 20/2σ, ATR 14, MAs
  5/10/20) are standard technical-analysis defaults — the paper does not
  specify these.
- **Price-level reconstruction**: CMIN gives log returns, not price levels.
  A relative "close level" is reconstructed per stock as the cumulative
  product of `exp(close_return)` (anchored at 1.0) purely so
  RSI/MACD/Bollinger/ATR/MAs have something to operate on. This is an
  approximation, not the stock's real price.
- **OHLC column order**: assumed Open, High, Low, Close for raw columns
  1–4, based on (a) conventional reading order and (b) an empirical check
  that raw column 0 ("movement") and raw column 4 are nearly numerically
  identical in the real CMIN-US files — consistent with column 4 being the
  close return. (An earlier, legacy, unused file guessed a different
  column mapping for the High/Low proxies; this new file corrects that.)
- **Daily high/low levels** for ATR are approximated from the reconstructed
  close level nudged by that day's own high/low return, not true intraday
  OHLC prices.

These are documented, best-effort choices — flag them if the authors'
source code or a stricter dataset spec ever becomes available.

## Commands

Embeddings are already built and are reused as-is, so you can go straight
to training.

**Smoke test (tiny, CPU, before committing to a full GPU run):**

```bash
python train_17feat.py --max-stocks 3 --epochs 1 --device cpu \
  --checkpoint-dir checkpoints/_smoke_17feat --tensorboard-dir "" --checkpoint-interval 0
```

**Full training run:**

```bash
python train_17feat.py \
  --dataset-root data/CMIN-Dataset-official/CMIN-US \
  --cache-root data/cache/cmin-us-mt5 \
  --checkpoint-dir checkpoints/cmin-us-17feat \
  --device cuda \
  --log-file logs/cmin-us-17feat-training.log
```

**Resume (training in pieces, e.g. across Kaggle sessions):**

```bash
python train_17feat.py \
  --checkpoint-dir checkpoints/cmin-us-17feat \
  --device cuda --resume \
  --log-file logs/cmin-us-17feat-training.log
```

(`last.pt` every epoch, `periodic/epoch_NNN.pt` every `--checkpoint-interval`
epochs (default 5), `best.pt`/`best_accuracy.pt` on improvement — identical
mechanics to `train.py`; see the "Kaggle" answer in this conversation for
`kaggle_persist.py` save/restore commands, unchanged for this pipeline.)

**Evaluate a checkpoint:**

```bash
python evaluate_checkpoint_17feat.py --checkpoint checkpoints/cmin-us-17feat/best.pt --device cuda
```

**Ten-seed comparison run (for comparing against the 6-feature model's
`run_ten_seeds.py` output):**

```bash
python run_ten_seeds_17feat.py --output-root experiments/proposed-17feat
```
