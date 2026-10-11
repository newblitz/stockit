# Changes: pluggable Qwen3 / BGE-M3 text backends

Scope: replace.md — swap the frozen summarizer (mT5 XL-Sum -> optionally
Qwen3-8B) and the frozen daily text embedder (mT5 encoder -> optionally
BGE-M3) behind a config switch, without touching the downstream architecture,
training loop, data splits, or price preprocessing.

> Note on filename: the deliverable in replace.md is named `CHANGES.md`, but
> this repository already has a `changes.md` (a prior session's audit log) and
> this filesystem is case-insensitive, so `CHANGES.md` is the same file as
> `changes.md`. This document is named `CHANGES_TEXT_BACKENDS.md` instead to
> avoid overwriting that file; `README.md` links here.

## Default backend: Qwen3-8B + BGE-M3

`config.py`'s `Config.summarizer` / `Config.embedder` default to
`"qwen3_8b"` / `"bge_m3"`, per replace.md Step 1 and explicit user
confirmation. This is what `scripts/build_text_cache.py` uses when no
`--summarizer`/`--embedder` flags are passed.

This default is safe for the existing canonical pipeline because
`prepare_embeddings.py` does not read `Config.summarizer`/`Config.embedder`
at all -- its own CLI always defaults to mT5 (`--model
csebuetnlp/mT5_multilingual_XLSum`), independent of `config.py`. So
README's already-built `data/cache/cmin-us-mt5/` caches and any checkpoint
trained from them stay exactly as reproducible as before; only the *new*
`scripts/build_text_cache.py` script's default changed. The mT5 baseline
(replace.md's config A) remains one explicit flag away:

```bash
python scripts/build_text_cache.py --summarizer mt5_xlsum --embedder mt5_encoder ...
```

`summarizer_model_name` / `embedder_model_name` default to `None` rather than
a fixed model-name string, so each backend always resolves its own correct
default repo id (`src/text_backends/*_backend.py`'s `DEFAULT_*_MODEL`)
regardless of which `summarizer`/`embedder` is selected -- switching
`summarizer` from `"qwen3_8b"` to `"mt5_xlsum"` (or back) never silently
carries over a mismatched model name from the other backend.

## Files added

| File | Purpose |
| --- | --- |
| `src/news_preprocessing.py` | §4.2 news cleaning/dedup/hour-bucketing/session-mapping, factored out of `prepare_embeddings.py` so both it and the new cache builder share one implementation (no behavior change; `prepare_embeddings.py` now imports these functions instead of defining its own copies). |
| `src/text_backends/base.py` | `Summarizer` / `Embedder` abstract interfaces (replace.md Step 0/1). |
| `src/text_backends/mt5_backend.py` | `MT5Summarizer` / `MT5Embedder`: thin wrappers around the existing, untouched `src/summarization.HierarchicalSummarizer` so the mT5 path is reachable through the same interface as the new backends (baseline reproducibility, replace.md Step 6.6). |
| `src/text_backends/qwen_backend.py` | `QwenSummarizer`: Qwen3-8B, non-thinking mode (`enable_thinking=False`, defensive `<think>` stripping), vLLM-preferred with a batched `transformers` fallback, greedy decoding (replace.md Step 2). |
| `src/text_backends/bge_backend.py` | `BGEM3Embedder`: BGE-M3 dense embeddings via `FlagEmbedding.BGEM3FlagModel` (fallback: `sentence-transformers`), L2-normalized, dim 1024 (replace.md Step 3). |
| `src/text_backends/factory.py` | `build_summarizer(config)` / `build_embedder(config, summarizer=...)` / `embedder_dim(name)`, selecting a backend from `config.py`'s switches. |
| `src/text_backends/__init__.py` | Re-exports the above. |
| `prompts.py` | `HOURLY_PROMPT_QWEN` / `DAILY_PROMPT_QWEN` templates (replace.md Step 2; verbatim from the task spec). |
| `scripts/build_text_cache.py` | Resumable cache builder: hourly + daily synopses to inspectable JSONL, daily embeddings to a `{dates, embeddings, preprocessing_version}` `.pt` file **in the same format `src/data.py` already reads** (replace.md Step 4). Supports `--stage {all,summarize,embed}` to unload the summarizer before loading the embedder. |
| `CHANGES_TEXT_BACKENDS.md` | This file. |

## Files modified

| File | Change |
| --- | --- |
| `prepare_embeddings.py` | No behavior change. News-preprocessing helpers (`_clean`, `_has_current_cache` stays local, `levenshtein_distance`, `_is_near_duplicate`, `_market_session`, `_available_trading_day`, `hourly_documents`) are now imported from `src/news_preprocessing.py` instead of being defined inline. All existing names/imports used by `tests/test_preprocessing.py` are preserved. |
| `config.py` | Added `summarizer`, `embedder` fields, defaulting to `"qwen3_8b"`/`"bge_m3"`, plus `summarizer_model_name`/`embedder_model_name` (default `None` -> each backend's own default repo id) and `Config.text_cache_subdir` / `Config.resolved_text_embedding_dim` helpers. `text_embedding_dim` (768) is untouched and documented as a pre-cache fallback only — `src/data.py` already infers the real width from cache files, so no downstream dimension code needed to change (replace.md Step 5 requirement was already satisfied by the existing design). |
| `requirements.txt` | Added optional `vllm`, `accelerate`, `FlagEmbedding`, `sentence-transformers` (replace.md Step 8). Not required for `prepare_embeddings.py`'s own mT5 pipeline. |
| `README.md` | Added one new section, "Alternative frozen text backends (Qwen3 + BGE-M3)", linking here. No existing section was changed. |

## Files explicitly NOT modified

`src/model.py`, `src/data.py`, `train.py`, `src/summarization.py`,
`prepare_embeddings.py`'s embedding loop/CLI, loss function, data splits, and
price preprocessing are all unchanged, per replace.md's constraints.
`src/summarization.py` keeps working standalone, and `prepare_embeddings.py`
is still the tool for the original single-model mT5 cache.

## How the embedding-dimension handoff works (Step 5)

`src/data.py`'s `CMINWindowDataset` already sets
`self.text_embedding_dim = series.embeddings.shape[1]` from each ticker's
cache file, and `train.py` builds the model with
`text_embedding_dim=train_set.text_embedding_dim`. Because
`scripts/build_text_cache.py` writes the exact same `.pt` schema as
`prepare_embeddings.py` (`{dates, embeddings, preprocessing_version}`), a
BGE-M3 cache's 1024-wide embeddings flow into
`PriceTextAlignment`'s `Linear(d -> d_p)` (Eq. 1) automatically — no code
change was needed in `src/model.py` or `src/data.py`.

## Commands

### Build a cache

```bash
# Config A (baseline, same effective pipeline as prepare_embeddings.py,
# via the generic interface, written to a separately namespaced cache dir):
python scripts/build_text_cache.py \
  --dataset-root data/CMIN-Dataset-official/CMIN-US \
  --summarizer mt5_xlsum --embedder mt5_encoder \
  --device cuda
# -> data/cache/mt5_xlsum_mt5_encoder/<TICKER>.pt

# Config B: mT5-XLSum summarizer + BGE-M3 embedder
python scripts/build_text_cache.py \
  --dataset-root data/CMIN-Dataset-official/CMIN-US \
  --summarizer mt5_xlsum --embedder bge_m3 \
  --device cuda
# -> data/cache/mt5_xlsum_bge_m3/<TICKER>.pt

# Config C: Qwen3-8B summarizer + BGE-M3 embedder (the default; flags shown
# explicitly here for clarity, but --summarizer/--embedder may be omitted)
python scripts/build_text_cache.py \
  --dataset-root data/CMIN-Dataset-official/CMIN-US \
  --summarizer qwen3_8b --embedder bge_m3 \
  --device cuda
# -> data/cache/qwen3_8b_bge_m3/<TICKER>.pt

# Run stages separately to control GPU residency (e.g. free Qwen3's 24GB
# before loading BGE-M3 on a smaller GPU):
python scripts/build_text_cache.py --summarizer qwen3_8b --embedder bge_m3 --stage summarize --device cuda
python scripts/build_text_cache.py --summarizer qwen3_8b --embedder bge_m3 --stage embed --device cuda

# Smoke test: one ticker
python scripts/build_text_cache.py --ticker AAPL --summarizer qwen3_8b --embedder bge_m3 --device cuda

# Progress check (no model load)
python scripts/build_text_cache.py --summarizer qwen3_8b --embedder bge_m3 --status
```

If vLLM is unavailable (e.g. no CUDA), `QwenSummarizer` automatically falls
back to batched Hugging Face `transformers` generation; pass `--no-vllm` to
force that path explicitly.

### Train / evaluate with a given cache

`train.py` is unmodified: point it at the matching `--cache-root` and it
picks up the embedding width automatically.

```bash
# Config A
python train.py --cache-root data/cache/mt5_xlsum_mt5_encoder --checkpoint-dir checkpoints/cmin-us-a --log-file logs/cmin-us-a.log

# Config B
python train.py --cache-root data/cache/mt5_xlsum_bge_m3 --checkpoint-dir checkpoints/cmin-us-b --log-file logs/cmin-us-b.log

# Config C
python train.py --cache-root data/cache/qwen3_8b_bge_m3 --checkpoint-dir checkpoints/cmin-us-c --log-file logs/cmin-us-c.log

# Evaluate a checkpoint (unmodified evaluate_checkpoint.py also needs the matching --cache-root)
python evaluate_checkpoint.py --checkpoint checkpoints/cmin-us-c/best.pt --cache-root data/cache/qwen3_8b_bge_m3
```

For the ten-seed comparison across configs A/B/C (replace.md Step 7), run
`run_ten_seeds.py --cache-root <cache> --output-root experiments/<config>`
for each config and compare the resulting summaries (`--compare-summary`
already supports paired t-tests between two seed summaries).

## Verification performed in this environment

This sandbox has no GPU, no network access to Hugging Face Hub/vLLM/PyPI by
default, and does not have `vllm` / `FlagEmbedding` / `sentence-transformers`
installed, so the Qwen3-8B and BGE-M3 backends could not be executed live
here. What was verified:

1. `python -m pytest -q` — all 12 existing tests pass after the
   `prepare_embeddings.py` refactor (no behavior change).
2. End-to-end smoke test of `scripts/build_text_cache.py` with
   `summarizer=mt5_xlsum, embedder=mt5_encoder` (config A) against a trimmed
   real AAPL slice, using the mT5 model already cached locally
   (`HF_HUB_OFFLINE=1`, CPU):
   - Stage `summarize` writes hourly + daily JSONL records, including the
     "empty day" (no news) fallback matching the original
     `HierarchicalSummarizer` behavior.
   - Stage `embed` (run standalone, i.e. loading its own mT5 instance rather
     than reusing a live `QwenSummarizer`/`MT5Summarizer` object) produces a
     `<TICKER>.pt` with the exact `{dates, embeddings[N,768], preprocessing_version}`
     schema `src/data.py` expects.
   - `src.data.load_stock_series(...)` loads that cache directly (no code
     changes), and a `HierarchicalCoAttentionStockPredictor` forward/
     `predict_proba` pass runs on it successfully.
   - Re-running the full `--stage all` command a second time, after both
     artifacts already existed, skipped all work (resumability), including
     after a fix for a bug caught during this testing: the embedding stage
     originally re-embedded every day on every run because it never checked
     whether a *complete* final `.pt` already covered all expected dates
     (only the intra-run `.partial.pt` was checked). Fixed in
     `scripts/build_text_cache.py`'s `embed_ticker` via a new
     `_has_current_embeddings` check, mirroring `prepare_embeddings.py`'s own
     `_has_current_cache` pattern.
   - Appending one new trading day and news file, then re-running, correctly
     summarized only the new day (JSONL resumability), consistent with
     `prepare_embeddings.py`'s existing convention that the final embedding
     tensor is rebuilt from scratch whenever the cached date list changes
     (no partial checkpoint exists once a prior run completed).
3. Qwen3-8B / BGE-M3 backends were reviewed for correctness against
   replace.md's spec (non-thinking chat template, think-tag stripping,
   greedy decoding, dense-only BGE-M3 encoding, L2 normalization) but not
   executed, since neither model nor the `vllm`/`FlagEmbedding` packages are
   available in this environment. Running Step 6's full sanity checklist
   (shape/NaN/determinism/eyeball checks) and Step 7's ten-seed A/B/C
   comparison is the recommended next step on a GPU machine with network
   access, using the commands above.
