# How to Run

Step-by-step instructions for the **canonical** training pipeline described in
[`README.md`](README.md) and [`paper/paper.md`](paper/paper.md).

Use the root-level scripts only:

- `prepare_embeddings.py` — Step 1: build frozen mT5 text-embedding caches
- `train.py` — Step 2: train the co-attention model

Do **not** use `main.py` or `src/train.py`; those belong to an older,
incompatible pipeline.

---

## Pipeline overview

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

1. **Prepare embeddings** (slow, run once per ticker) → `data/cache/.../<TICKER>.pt`
2. **Train** (fast once caches exist) → `checkpoints/.../best.pt`

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Internet on first run (downloads `csebuetnlp/mT5_multilingual_XLSum` from Hugging Face)
- GPU strongly recommended for embedding preparation

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Optional: set a Hugging Face token to avoid anonymous rate limits.

```bash
export HF_TOKEN=your_hf_token_here
```

---

## Expected dataset layout

The loader expects a **dataset root** directory with this structure:

```text
<dataset-root>/
├── price/processed/<TICKER>.txt
└── news/preprocessed/<TICKER>/<YYYY-MM-DD>
```

Each price file has one row per trading day: date plus six numeric columns
(tab-separated). Labels are derived from the **raw** first column (next-day
movement), not the z-scored values.

Default local path:

```text
data/CMIN-Dataset-official/CMIN-US/
```

### Download CMIN-US (local machine)

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/BigRoddy/CMIN-Dataset.git data/CMIN-Dataset-official
git -C data/CMIN-Dataset-official sparse-checkout set CMIN-US
```

For CMIN-CN:

```bash
git -C data/CMIN-Dataset-official sparse-checkout add CMIN-CN
```

---

## Step 0: Verify the code

```bash
python -m pytest -q
```

Expected: all tests pass.

---

## Step 1: Build text-embedding caches

`prepare_embeddings.py` implements **Algorithm 1** from the paper:

- group news into hourly documents
- mT5 summarizes each hour
- mT5 summarizes hourly outputs into one daily summary
- frozen mT5 encoder produces a 768-d vector per trading day
- save one `.pt` file per ticker

Output format per file:

```text
data/cache/cmin-us-mt5/AAPL.pt
```

### Smoke test (one ticker)

```bash
python prepare_embeddings.py --ticker AAPL --device cuda
```

On CPU only:

```bash
python prepare_embeddings.py --ticker AAPL --device cpu
```

### Full CMIN-US run (all tickers)

```bash
python prepare_embeddings.py --device cuda
```

### Resume behavior

If `AAPL.pt` already exists, the script prints `skip AAPL: cache exists` and
continues. Delete a cache file only when you intentionally want to regenerate it.

### `prepare_embeddings.py` options

| Flag | Default | Description |
| --- | --- | --- |
| `--dataset-root` | `data/CMIN-Dataset-official/CMIN-US` | Root folder with `price/processed` and `news/preprocessed` |
| `--cache-root` | `data/cache/cmin-us-mt5` | Where per-ticker `.pt` caches are written |
| `--ticker` | all tickers | Process specific ticker(s); repeatable, e.g. `--ticker AAPL --ticker MSFT` |
| `--max-stocks` | none | Process only the first N tickers alphabetically |
| `--model` | `csebuetnlp/mT5_multilingual_XLSum` | Hugging Face summarizer model |
| `--device` | `cuda` if available else `cpu` | Device for mT5 inference |

---

## Step 2: Train the model

Training reads price windows from `--dataset-root` and text embeddings from
`--cache-root`. It does **not** run mT5 during training.

### Smoke test (first ticker only)

```bash
python train.py --max-stocks 1 --log-file logs/cmin-us-aapl-training.log
```

### Full CMIN-US training

```bash
python train.py --log-file logs/cmin-us-training.log
```

### Useful overrides

```bash
python train.py \
  --epochs 20 \
  --batch-size 16 \
  --device cuda \
  --seed 42 \
  --log-file logs/cmin-us-training.log
```

### `train.py` options

| Flag | Default | Description |
| --- | --- | --- |
| `--dataset-root` | `data/CMIN-Dataset-official/CMIN-US` | CMIN dataset root |
| `--cache-root` | `data/cache/cmin-us-mt5` | Precomputed text-embedding cache directory |
| `--checkpoint-dir` | `checkpoints/cmin-us` | Where checkpoints and metrics are saved |
| `--epochs` | `20` | Training epochs |
| `--batch-size` | `16` | Mini-batch size |
| `--max-stocks` | all cached tickers | Limit to first N tickers (smoke tests) |
| `--device` | `cuda` if available else `cpu` | Training device |
| `--seed` | `42` | Random seed |
| `--log-file` | none | Mirror epoch output to a log file |

### Monitor training

```bash
tail -f logs/cmin-us-training.log
ls -lh checkpoints/cmin-us
```

### Checkpoints written

| File | Meaning |
| --- | --- |
| `best.pt` | Best model by **validation MCC** (primary metric) |
| `best_accuracy.pt` | Best model by validation accuracy |
| `last.pt` | Most recent epoch |
| `history.json` / `history.csv` | Per-epoch metrics |
| `metrics.json` | Final epoch summary |

---

## Custom dataset paths (local)

Set the same `--dataset-root` and `--cache-root` in **both** scripts.

### CMIN-US with custom locations

```bash
DATASET_ROOT=/path/to/CMIN-US
CACHE_ROOT=/path/to/cache/cmin-us-mt5
CHECKPOINT_DIR=/path/to/checkpoints/cmin-us

python prepare_embeddings.py \
  --dataset-root "$DATASET_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --device cuda

python train.py \
  --dataset-root "$DATASET_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --log-file /path/to/logs/cmin-us-training.log
```

### CMIN-CN example

```bash
DATASET_ROOT=data/CMIN-Dataset-official/CMIN-CN
CACHE_ROOT=data/cache/cmin-cn-mt5
CHECKPOINT_DIR=checkpoints/cmin-cn

python prepare_embeddings.py \
  --dataset-root "$DATASET_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --device cuda

python train.py \
  --dataset-root "$DATASET_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --log-file logs/cmin-cn-training.log
```

---

## Kaggle notebook setup

Kaggle mounts input datasets under `/kaggle/input/` and writes persistent
notebook output to `/kaggle/working/`. Use explicit paths for dataset, cache,
checkpoints, and logs.

### 1. Add the dataset in Kaggle

1. Upload or attach the CMIN dataset to your notebook.
2. Note the mount path shown in the notebook, for example:
   - `/kaggle/input/cmin-dataset/CMIN-US`
   - or `/kaggle/input/cmin-dataset-official/CMIN-US`

The path must contain:

```text
price/processed/
news/preprocessed/
```

### 2. Install dependencies

```python
!pip install -q -r /kaggle/input/your-code-repo/requirements.txt
```

If the repo is the notebook itself:

```python
!pip install -q torch transformers sentencepiece numpy
```

### 3. Set Kaggle paths

```python
import os
from pathlib import Path

# Change this to your actual Kaggle input path
DATASET_ROOT = Path("/kaggle/input/cmin-dataset/CMIN-US")

# Writable paths inside the notebook
CACHE_ROOT = Path("/kaggle/working/cache/cmin-us-mt5")
CHECKPOINT_DIR = Path("/kaggle/working/checkpoints/cmin-us")
LOG_DIR = Path("/kaggle/working/logs")

CACHE_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
```

### 4. Optional Hugging Face token

```python
import os
os.environ["HF_TOKEN"] = "your_hf_token_here"
```

### 5. Smoke test on Kaggle (one ticker)

```python
!python prepare_embeddings.py \
  --dataset-root {DATASET_ROOT} \
  --cache-root {CACHE_ROOT} \
  --ticker AAPL \
  --device cuda
```

```python
!python train.py \
  --dataset-root {DATASET_ROOT} \
  --cache-root {CACHE_ROOT} \
  --checkpoint-dir {CHECKPOINT_DIR} \
  --max-stocks 1 \
  --device cuda \
  --log-file {LOG_DIR}/cmin-us-aapl-training.log
```

### 6. Full Kaggle run

```python
!python prepare_embeddings.py \
  --dataset-root {DATASET_ROOT} \
  --cache-root {CACHE_ROOT} \
  --device cuda
```

```python
!python train.py \
  --dataset-root {DATASET_ROOT} \
  --cache-root {CACHE_ROOT} \
  --checkpoint-dir {CHECKPOINT_DIR} \
  --device cuda \
  --log-file {LOG_DIR}/cmin-us-training.log
```

### 7. Save outputs from Kaggle

Checkpoints and caches live under `/kaggle/working/`. Download them from the
notebook output panel or zip them:

```python
!zip -r /kaggle/working/cmin-us-artifacts.zip \
  /kaggle/working/checkpoints/cmin-us \
  /kaggle/working/cache/cmin-us-mt5
```

### Kaggle tips

- Turn on the GPU accelerator for both embedding prep and training.
- Embedding preparation is the slowest step; run it once and reuse the cache.
- If you restart the notebook, `/kaggle/working/` may be cleared unless you
  save artifacts to a Kaggle Dataset or download them.
- Use the **same** `--dataset-root` and `--cache-root` in both scripts.
- If your Kaggle dataset root is one level higher, point `--dataset-root` at
  the folder that directly contains `price/` and `news/`.

Example when the notebook sees:

```text
/kaggle/input/cmin-dataset/
└── CMIN-US/
    ├── price/processed/
    └── news/preprocessed/
```

then use:

```bash
--dataset-root /kaggle/input/cmin-dataset/CMIN-US
```

Example when the notebook sees:

```text
/kaggle/input/cmin-dataset/
├── price/processed/
└── news/preprocessed/
```

then use:

```bash
--dataset-root /kaggle/input/cmin-dataset
```

---

## End-to-end quick reference

### Local default paths

```bash
# 0) tests
python -m pytest -q

# 1) embeddings
python prepare_embeddings.py --ticker AAPL --device cuda
python prepare_embeddings.py --device cuda

# 2) train
python train.py --max-stocks 1 --log-file logs/cmin-us-aapl-training.log
python train.py --log-file logs/cmin-us-training.log
```

### Custom / Kaggle template

```bash
python prepare_embeddings.py \
  --dataset-root <DATASET_ROOT> \
  --cache-root <CACHE_ROOT> \
  --device cuda

python train.py \
  --dataset-root <DATASET_ROOT> \
  --cache-root <CACHE_ROOT> \
  --checkpoint-dir <CHECKPOINT_DIR> \
  --log-file <LOG_FILE>
```

Replace the placeholders with your local paths or Kaggle `/kaggle/input/...`
and `/kaggle/working/...` paths.

---

## Troubleshooting

| Problem | Likely cause | Fix |
| --- | --- | --- |
| `Missing text cache ... Run prepare_embeddings.py` | Step 1 not finished for that ticker | Run `prepare_embeddings.py` with matching `--dataset-root` and `--cache-root` |
| `cached dates do not match prices` | Cache regenerated against different price files | Delete the ticker `.pt` and rebuild the cache |
| Embedding step extremely slow | Running mT5 on CPU | Use `--device cuda` on a GPU machine |
| Hugging Face download errors | Rate limits or no internet | Set `HF_TOKEN`, retry, or pre-download the model |
| `python main.py` fails | Legacy entry point | Use `train.py` instead |
| Training starts but MCC stays near 0 | Too few stocks, smoke run only, or cache/model mismatch | Run full ticker set with canonical caches |

Check for an already-running job before starting a duplicate:

```bash
ps -ax | grep -E 'prepare_embeddings|train.py' | grep -v grep
```

---

## Data splits (paper protocol)

| Split | Target date range |
| --- | --- |
| Train | 2018-01-01 to 2020-06-30 |
| Validation | 2020-07-01 to 2020-12-31 |
| Test | 2021-01-01 to 2021-12-31 |

`train.py` currently trains on train and validates on validation. Test-set
evaluation after selecting `best.pt` is not yet wired into the default script.
