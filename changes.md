# Change log and paper-alignment audit

Date: 2026-09-10 (price-feature count corrected 2026-09-12)

This document records changes made in this working session to align the
canonical root pipeline with `paper/paper.md`. It is an implementation audit,
not a claim that the paper's reported results have been reproduced.

**Correction (2026-09-12):** an earlier draft of this session switched the
active model to the paper's full 17-dimensional "Price+TA+Text" ablation
configuration (Table in §4.6: 6 processed price columns + 11 technical
indicators). That switch has been reverted at the user's request. The
canonical pipeline (`config.py`, `src/model.py`, `src/data.py`, `train.py`,
and `tests/`) now uses, and has been verified to use, the original
**6-dimensional** price representation (the paper's "Price+Text" row: daily
movement, four OHLC log returns, and volume). `src/technical_indicators.py`'s
17-column feature builder is legacy, dead code used only by the already-obsolete
`src/dataset.py`/`src/train.py` pair; it is not invoked by the canonical
pipeline and does not affect `price_dim`. The bullets below that mention a
17-feature configuration describe that reverted, no-longer-active state and
are kept only for history.

## Problems addressed

The starting pipeline had several data-integrity and reproducibility problems:

- it did not implement the paper's short-gap imputation, >5% missing-data
  exclusion, 3-sigma return truncation, or Levenshtein-based text deduplication;
- it used all articles attached to a calendar day, creating a potential
  post-market-close news look-ahead path;
- normalization was applied inconsistently across splits;
- validation/test samples could consume input days from the preceding split;
- the repository had no reusable multi-seed, statistical-test, backtest, or
  latency-evaluation utilities.

## Changes made

### Data, text, and embedding preparation

- `src/data.py`
  - Adds a market calendar from the union of stock dates, forward-fills gaps of
    at most three market sessions, and excludes a stock when more than 5% of
    market sessions are missing.
  - Represents a forward-filled price observation as zero OHLC-return movement
    with the preceding volume carried forward.
  - Applies per-stock training-period mean ± 3 standard-deviation clipping to
    return columns before technical-feature construction.
  - Uses training-period statistics only for feature standardization.
  - Creates 30-day samples only when every input day and the target day are in
    the selected chronological split; validation/test no longer borrow training
    or validation history.
  - Retains next-day return metadata solely for post-selection backtesting; it
    is not an input feature.

- `src/technical_indicators.py` *(reverted, see correction note above)*
  - An earlier draft of this session corrected this builder to output 17
    columns (6 processed price fields + 11 technical indicators) so the
    canonical pipeline could match the paper's full "Price+TA+Text" model.
    That switch was reverted; the file is legacy dead code again, used only
    by the obsolete `src/dataset.py`, and does not affect the active 6-feature
    `price_dim`.

- `prepare_embeddings.py`
  - Replaces character 4-gram Jaccard duplicate detection with exact normalized
    character-level Levenshtein similarity; later articles above 90% similarity
    are dropped and the earliest is retained.
  - Keeps the paper's cleaning rules: HTML unescaping, lowercasing,
    non-alphanumeric removal, whitespace collapse, and a 50-character minimum.
  - Groups cleaned articles by hour, summarizes hourly content before daily
    content, and stores preprocessing provenance in every cache.
  - Introduces a market-close availability rule. Source timestamps are assumed
    UTC, converted to America/New_York for CMIN-US or Asia/Shanghai for CMIN-CN,
    and articles after the regular close are assigned to the next trading day.
  - Bumps the cache version to `paper-4.2-v3-close-time-levenshtein`; old caches
    are deliberately not accepted.

### Model/training configuration and entry points

- `config.py`
  - Changes defaults to Table 2 values: 20 epochs and patience 5.
  - *(Reverted)* An earlier draft set the full-model price dimension to 17;
    `price_dim` is back to the original 6 at the user's request.

- `train.py`
  - Uses sample-weighted evaluation loss, validation-MCC checkpoint selection,
    and a single final held-out test evaluation after model selection.
  - Adds the specified five-epoch learning-rate schedule and deterministic CUDA
    seed settings.
  - Adds fail-fast checkpoint geometry validation. A six-feature checkpoint is
    rejected before `load_state_dict` rather than producing a shape error or an
    invalid comparison.

- `main.py`
  - Is now a compatibility entry point that delegates to the canonical root
    `train.py` implementation instead of the obsolete trainer.

### Evaluation and documentation

- `src/evaluation.py` (new)
  - Provides ACC/MCC, ten-run mean/std aggregation, paired t-tests (via SciPy),
    and a documented long/cash policy simulator.

- `evaluate_checkpoint.py` (new)
  - Evaluates one compatible checkpoint on the held-out test split, reports
    classification metrics, per-stock averaged trading metrics, and inference
    latency.

- `run_ten_seeds.py` (new)
  - Trains independent seed directories (default seeds 0–9), aggregates test
    metrics, and can run paired t-tests against a supplied baseline/ablation
    summary with matched seeds.

- `requirements.txt`
  - Adds SciPy for statistically valid paired t-tests.

- `README.md`
  - Updates the checkpoint/test-evaluation and cache/time-availability notes.

- `tests/test_preprocessing.py` (new)
  - Covers forward fill, missing-stock exclusion, training-only clipping,
    Levenshtein deduplication, and after-close/weekend news reassignment.

## Verification performed

- `python -m pytest -q` completed successfully: **7 passed**.
- Python compilation checks passed for the modified scripts and modules.
- `git diff --check` passed.
- Inspection confirmed that the existing `best.pt`, `best_accuracy.pt`, and
  `last.pt` (when present) declare `price_dim=6`, matching the active
  6-feature pipeline. No checkpoints were present in `checkpoints/` at the
  time of the 2026-09-12 revert.
- Re-verified after the 2026-09-12 revert: `python -m pytest -q` passes
  (7 tests), and a live smoke test loading real CMIN-US price/cache data
  (`src/data.py`) confirms `StockSeries.features` has shape `(days, 6)`,
  `CMINWindowDataset.price_dim == 6`, and a forward/backward pass through
  `HierarchicalCoAttentionStockPredictor` succeeds with 6-dimensional price
  windows.

## Required operational reset

The caches and checkpoints currently present in the repository are not valid
artifacts for this revised pipeline. Regenerate embeddings and train into a new
checkpoint directory before using evaluation commands:

```bash
python prepare_embeddings.py --device cuda
python run_ten_seeds.py --output-root experiments/proposed
```

## Remaining gaps relative to `paper/paper.md`

These are deliberate, known gaps. They must not be described as completed or
used to claim reproduction of Tables 3–7.

1. **Named baselines are absent.** ALSTM, StockNet, Adv-LSTM, DTML, CMIN,
   LLMFactor, and CausalStock are not implemented or run. The multi-seed runner
   can compare results only after real baseline implementations produce matched
   seed summaries.
2. **Ablation variants are absent.** Text-only, close-only, OHLCV-only,
   price+TA, direct-embedding, single-stage-summary, concatenation, and
   one-way-cross-attention variants required for Tables 4–6 are not implemented
   as selectable models.
3. **No paper result has been reproduced yet.** Ten actual seed runs, their
   mean/std values, paired t-tests, and the reported ACC/MCC tables have not
   been generated. The scripts are infrastructure only until fresh caches and
   experiments are run.
4. **Trading evaluation is an approximation, not Table 7 reproduction.** The
   implementation averages independent per-stock long/cash paths. It does not
   implement the paper's stated $100,000 cross-stock portfolio, aggregate
   capital allocation, or every reported risk metric (for example Calmar).
5. **Latency has not been measured on the paper's RTX 5080 protocol.** The
   evaluator measures a configurable number of local forward passes; a result
   from another device, precision mode, batch size, or cache state is not
   comparable to Table 7.
6. **CMIN-CN is not operationally validated.** The code has a China close-time
   policy, but the repository currently contains CMIN-US data/caches only. The
   300-stock CMIN-CN pipeline, its separate paths, and its M=4/H=4 setting have
   not been exercised end to end.
7. **Some paper details are underspecified.** The paper does not supply exact
   technical-indicator formulas/periods, LLM generation settings, or embedding
   pooling. The implementation uses standard indicators, `max_new_tokens=64`,
   and masked mean pooling; these are documented choices, not verified author
   code.
8. **Timestamp timezone is an explicit assumption.** CMIN timestamp strings do
   not carry an offset in the checked files. Treating them as UTC is reasonable
   for Yahoo data but must be verified against dataset provenance before making
   a strict causal claim.
9. **Dataset coverage needs an empirical audit after regeneration.** The local
   CMIN-US price files observed here end on 2021-12-23 rather than the paper's
   stated 2021-12-31. The final retained-stock count and truncation rate have
   not yet been checked against the paper's reported 108 stocks and <0.8%.
10. **Legacy modules remain.** `src/dataset.py` and `src/train.py` are obsolete
    and are not the canonical route. The root scripts should be used until the
    legacy modules are removed or converted to compatible wrappers.

## Scope note

This log excludes unrelated working-tree changes and generated binary cache
files that existed or were produced outside this implementation work (for
example `.DS_Store` files and untracked embedding-cache shards).
