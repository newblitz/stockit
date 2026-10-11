# Task: Replace the Summarizer and Embedder in the Existing Pipeline

## Goal

The repo already implements the full pipeline (hierarchical summarization → embedding → alignment → patching → co-attention → classifier). **Do not rewrite it.** Only swap two frozen components:

| Component | Current | New |
|---|---|---|
| Summarizer (hourly + daily) | `mT5_multilingual_XLSum` | **Qwen3-8B** (instruct / non-thinking mode) |
| Embedder (`Embed(S_day)`) | mT5 encoder embedding | **BGE-M3** (dense embeddings) |

Everything downstream (alignment, patching, co-attention, classifier, training loop, data splits, metrics, hyperparameters in Table 2) must stay unchanged, except for the input embedding dimension described in Step 5.

Both models are frozen. They are run once offline, their outputs are cached to disk, and the downstream model trains from the cache.

---

## Step 0: Locate the existing code

Find and read the following in the repo before changing anything:

1. The module implementing Algorithm 1 (hourly summarization → concat → daily summarization → `Embed`). Likely names: `hierarchical_summary`, `summarizer`, `text_encoder`, or similar.
2. Where the mT5 tokenizer and model are loaded and called.
3. Where the resulting embedding `X_e` (shape `B × L × d`) is consumed. The first consumer is the `Linear(d → d_p)` + Dropout layer (Eq. 1).
4. How embeddings are currently stored or loaded (precomputed file, on-the-fly, dataset class).
5. The config system (argparse, yaml, or constants) so the new options can be added consistently.

Report a short summary of these locations before editing.

---

## Step 1: Add a config switch (keep the old path working)

Add config options so the baseline remains reproducible:

```
summarizer: {mt5_xlsum | qwen3_8b}        # default: qwen3_8b
embedder:   {mt5_encoder | bge_m3}        # default: bge_m3
summarizer_model_name: "Qwen/Qwen3-8B"
embedder_model_name:   "BAAI/bge-m3"
embed_dim: auto from embedder             # mt5-base=768, bge_m3=1024
cache_dir: ./cache/<summarizer>_<embedder>/
```

Do not delete the mT5 code. Put each backend behind a common interface:

```python
class Summarizer:
    def summarize_hourly(self, texts: list[str]) -> list[str]: ...
    def summarize_daily(self, hourly_summaries: list[str]) -> list[str]: ...

class Embedder:
    dim: int
    def encode(self, texts: list[str]) -> np.ndarray: ...   # (N, dim)
```

Implement `MT5Summarizer` / `MT5Embedder` as wrappers around the existing code, and add `QwenSummarizer` / `BGEM3Embedder` as new classes. The factory selects by config.

---

## Step 2: Implement `QwenSummarizer`

**Model:** `Qwen/Qwen3-8B`, run in non-thinking mode.
- Use `tokenizer.apply_chat_template(..., enable_thinking=False)`.
- Strip any `<think>...</think>` block if one still appears.

**Inference engine:** use **vLLM** for batched generation (preferred). Fall back to HF `transformers` with batching if vLLM is unavailable.
- dtype: `bfloat16` (or `float16`) on a 24 GB GPU.
- If VRAM is limited, use an AWQ/GPTQ 4-bit build or switch to `Qwen/Qwen3-4B` via the config.
- Greedy decoding for determinism: `temperature=0`, `max_tokens=256` for hourly, `max_tokens=384` for daily.
- Set `max_model_len` to at least 8192.

**Hourly step:** group raw news by stock and one-hour interval exactly as the current code does (keep the existing segmentation logic and the existing text cleaning). Remove the mT5 512-token truncation for this path. If an hourly bucket exceeds the context limit, truncate to the model's `max_model_len` minus the prompt and output budget.

**Prompts (put them in a separate `prompts.py` or yaml so they are easy to edit):**

Hourly prompt:
```
You are a financial news analyst. Below is the news about {ticker_or_company}
published during one hour. Summarize the key factors in this information that
could affect the stock price. Be concise (at most 3 sentences). Preserve concrete
facts such as figures, guidance, and named events.

News:
{documents}

Summary:
```

Daily prompt:
```
Below are chronologically ordered hourly summaries of news about {ticker_or_company}
for one trading day. Combine them into a single daily synopsis covering the key
factors that caused the stock price movement today. Be concise (at most 5 sentences).

Hourly summaries:
{concat_hourly}

Daily synopsis:
```

These mirror the prompts shown in Fig. 1 of the paper. If the repo already has prompt templates, reuse their structure and only adapt the wording for an instruction-tuned model.

**Edge cases:**
- Empty hour: skip the LLM call and omit it from the concatenation.
- Empty day (no news at all): follow whatever behavior the current code uses (zero vector, previous day's embedding, or a placeholder). Do not invent a new behavior.
- Daily step input: concatenate the hourly summaries in chronological order, as in Algorithm 1.

---

## Step 3: Implement `BGEM3Embedder`

**Model:** `BAAI/bge-m3`, dense embeddings only.

- Use `FlagEmbedding.BGEM3FlagModel(..., use_fp16=True)` or `sentence-transformers`. Either is fine; pick what is already in `requirements`, otherwise `FlagEmbedding`.
- Embedding dimension: **1024**.
- `max_length`: 512 is enough for a daily synopsis (the model supports up to 8192).
- L2-normalize the output. This is the model's default for the dense vectors.
- No instruction prefix is needed for BGE-M3.
- Batch size 32–64, with fp16 on GPU.

Input: the daily synopsis text `S_day`. Output: one vector `x_e ∈ R^1024` per (stock, day). This replaces `Embed(S_day)` in Algorithm 1.

---

## Step 4: Caching (mandatory)

Both models are frozen, so everything is precomputed once.

1. Write a standalone script, e.g. `scripts/build_text_cache.py`, which:
   - iterates over all stocks and dates for both datasets (CMIN-US and CMIN-CN);
   - runs hourly summarization, then daily summarization, then embedding;
   - saves hourly summaries and daily synopses as JSONL, keyed by `(dataset, stock, date)` (and hour for hourly), so the text can be inspected;
   - saves daily embeddings as `.npy` / `.pt`, keyed by `(dataset, stock, date)`.
2. Make it **resumable**: skip keys that already exist in the cache, and flush to disk every N items so a crash does not lose progress.
3. Run the summarization stage and the embedding stage as separate passes. The summarizer should be unloaded from the GPU before the embedder is loaded.
4. The cache directory is namespaced by `<summarizer>_<embedder>` so runs do not overwrite each other.
5. The training dataset class loads embeddings from this cache and must no longer call any LLM during training.

---

## Step 5: Fix the embedding dimension downstream

The only downstream change allowed.

- The layer in Eq. (1), `Linear(d → d_p)` followed by Dropout, takes `d` as its input size. mT5-base used `d = 768`; BGE-M3 uses `d = 1024`.
- Read `d` from the embedder config (`embedder.dim`) instead of hard-coding it, then check every place the old value appears (model constructor, dataset, reshape calls, checkpoint loading).
- Do not change `d_p`, patch length, stride, `d_model`, `d_ff`, number of layers, heads, dropout, learning rate, batch size, or epochs.

---

## Step 6: Tests and sanity checks

Run these before launching full experiments:

1. **Smoke test on one stock and one month.** Run the full cache build and then one training epoch. It must complete without shape errors.
2. **Shape check.** The dataset must yield `X_e` of shape `(L=30, d=1024)` per sample (or `B × L × 1024` per batch).
3. **Inspect 10–20 outputs by eye.** Print the hourly summaries and the daily synopsis next to the raw news and confirm they are coherent, in the right language for CMIN-CN (Chinese news summarized in Chinese or English consistently; choose one and keep it fixed), and not truncated mid-sentence.
4. **Embedding check.** Verify no NaNs or infs, norms are about 1 (after normalization), and different days give different vectors.
5. **Determinism.** Re-running the cache build on the same input must give the same summaries (greedy decoding) and the same embeddings.
6. **Baseline preserved.** With `summarizer=mt5_xlsum, embedder=mt5_encoder`, the original pipeline must still reproduce its previous numbers within seed noise.

---

## Step 7: Experiment configurations to run

Use the existing training and evaluation scripts, the same splits, and the same 10 seeds as the paper. Only the switches change:

| Config | Summarizer | Embedder |
|---|---|---|
| A (baseline) | mT5-XLSum | mT5 encoder |
| B | mT5-XLSum | BGE-M3 |
| C | Qwen3-8B | BGE-M3 |

Report ACC and MCC (mean ± std) per dataset in the same format as Tables 3–6.

---

## Step 8: Dependencies

Add to `requirements.txt` (pin versions that match the existing environment where possible):

```
vllm
transformers
FlagEmbedding
sentence-transformers   # only if used instead of FlagEmbedding
accelerate
```

Document the GPU requirement in the README: about 24 GB for Qwen3-8B in bf16, or use `Qwen3-4B` / a 4-bit build on smaller GPUs. BGE-M3 needs less than 4 GB.

---

## Deliverables

1. The new `Summarizer` / `Embedder` interfaces and the Qwen3 and BGE-M3 implementations.
2. `prompts.py` (or yaml) with the hourly and daily prompts.
3. `scripts/build_text_cache.py`, resumable.
4. Config switches and the `d` dimension fix.
5. A short `CHANGES.md` listing every file modified and the commands to (a) build the cache and (b) train and evaluate for configs A, B, and C.

## Constraints

- Do not modify the downstream architecture, loss, splits, preprocessing of price data, or evaluation code.
- Do not retrain or fine-tune either new model. Both are frozen.
- Keep changes minimal and confined to the text-processing path.
