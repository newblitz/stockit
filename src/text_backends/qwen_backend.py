"""Frozen Qwen3 instruct-mode hierarchical summarizer (replace.md Step 2).

Prefers vLLM for batched generation; falls back to batched Hugging Face
``transformers`` generation when vLLM is not installed or fails to initialize
(e.g. no compatible GPU). Both paths run in "non-thinking" mode and strip any
``<think>...</think>`` block defensively.

T4 GPU notes
------------
* Use ``Qwen/Qwen3-8B-AWQ`` (AWQ 4-bit, ~4.5 GB) rather than the base fp16
  model (~16 GB) so vLLM has enough VRAM left for the KV cache.
* T4 does not support bfloat16 — dtype is hardcoded to float16.
* ``enforce_eager=True`` disables CUDA-graph capture, which is unreliable on
  T4 and causes hangs / assertion errors with some vLLM versions.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from prompts import DAILY_PROMPT_QWEN, HOURLY_PROMPT_QWEN

from .base import Summarizer

DEFAULT_QWEN_MODEL = "Qwen/Qwen3-8B"
# AWQ-quantized variant (~4.5 GB on GPU).  Recommended for T4 / small-VRAM
# GPUs when using vLLM; the base fp16 model (~16 GB) does not leave enough
# room for a useful KV cache on a 16 GB card.
DEFAULT_QWEN_AWQ_MODEL = "Qwen/Qwen3-8B-AWQ"
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Remove a leftover <think>...</think> block even in non-thinking mode."""
    return _THINK_RE.sub("", text).strip()


class QwenSummarizer(Summarizer):
    def __init__(
        self,
        model_name: str = DEFAULT_QWEN_MODEL,
        *,
        dtype: str = "float16",  # T4 does not support bfloat16; use float16
        max_model_len: int = 8192,
        hourly_max_tokens: int = 256,
        daily_max_tokens: int = 384,
        device: str | None = None,
        use_vllm: bool = True,
        gpu_memory_utilization: float = 0.85,  # leave headroom for KV cache on T4
        hf_batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.hourly_max_tokens = hourly_max_tokens
        self.daily_max_tokens = daily_max_tokens
        # Only used by the transformers fallback; vLLM batches everything
        # it's given in one call and schedules it internally.
        self.hf_batch_size = hf_batch_size
        self._backend: str | None = None
        self._tokenizer = None
        self._engine = None
        self._sampling_params_cls = None
        self._hf_model = None
        self._hf_device = None

        if use_vllm:
            try:
                self._init_vllm(gpu_memory_utilization)
            except Exception as exc:  # pragma: no cover - depends on optional package/GPU
                print(
                    f"[QwenSummarizer] vLLM unavailable ({exc!r}); "
                    "falling back to transformers batched generation (much slower -- "
                    "install `vllm` and, for an AWQ model, `autoawq` to avoid this).",
                    flush=True,
                )
        if self._backend is None:
            self._init_transformers(device)

    def _init_vllm(self, gpu_memory_utilization: float) -> None:
        from transformers import AutoConfig, AutoTokenizer
        from vllm import LLM, SamplingParams

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        remote_config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        quantization_config = getattr(remote_config, "quantization_config", None)
        if isinstance(quantization_config, dict):
            quant_method = quantization_config.get("quant_method")
        else:
            quant_method = getattr(quantization_config, "quant_method", None)

        llm_kwargs: dict = {
            "model": self.model_name,
            "dtype": "float16",
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
            "enforce_eager": True,
        }
        if quant_method:
            llm_kwargs["quantization"] = quant_method
        self._engine = LLM(**llm_kwargs)
        self._sampling_params_cls = SamplingParams
        self._backend = "vllm"

    def _init_transformers(self, device: str | None) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # Left padding is required for correct batched causal-LM generation:
        # with right padding, every sequence's "next token" position differs,
        # which silently produces garbage continuations for the shorter ones.
        self._tokenizer.padding_side = "left"

        # T4 does not support bfloat16, so float16 is used for compute dtype.
        torch_dtype = torch.float16

        # AWQ/GPTQ-quantized checkpoints (e.g. Qwen3-8B-AWQ) already declare
        # their own quantization_config in config.json. Passing a
        # BitsAndBytesConfig on top of an already-quantized checkpoint raises
        # "model is quantized with AwqConfig but you are passing a
        # BitsAndBytesConfig". Query the model's actual remote config instead
        # of guessing quantization from the repo name string, since a
        # name-based heuristic misses renamed/local/mirrored AWQ checkpoints.
        remote_config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        already_quantized = getattr(remote_config, "quantization_config", None) is not None

        load_kwargs: dict = {
            "device_map": "auto",
            "dtype": torch_dtype,  # `dtype` replaces the deprecated `torch_dtype` kwarg
            "trust_remote_code": True,
        }
        if not already_quantized:
            # Base / fp16 model: apply 4-bit NF4 via BitsAndBytes to fit on T4.
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )

        self._hf_model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
        self._hf_model.eval().requires_grad_(False)

        # Keep a device reference for placing input tensors; the model itself
        # is already placed by device_map="auto", so do not call .to() on it.
        self._hf_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._backend = "hf"

    def _to_chat_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _generate(self, prompts: list[str], max_tokens: int) -> list[str]:
        if not prompts:
            return []
        chat_prompts = [self._to_chat_prompt(prompt) for prompt in prompts]
        if self._backend == "vllm":
            sampling = self._sampling_params_cls(temperature=0.0, max_tokens=max_tokens)
            outputs = self._engine.generate(chat_prompts, sampling)
            raw_texts = [output.outputs[0].text for output in outputs]
        else:
            # Batch hf_batch_size prompts per generate() call instead of one
            # at a time: this is the main speed fix for the transformers
            # fallback path (a per-prompt loop never uses the GPU's batch
            # parallelism, so it is roughly hf_batch_size times slower).
            import torch

            raw_texts = []
            budget = max(self.max_model_len - max_tokens, 1)
            for start in range(0, len(chat_prompts), self.hf_batch_size):
                batch = chat_prompts[start : start + self.hf_batch_size]
                tokens = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=budget,
                    padding=True,
                ).to(self._hf_device)
                with torch.inference_mode():
                    generated = self._hf_model.generate(
                        **tokens,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        pad_token_id=self._tokenizer.pad_token_id,
                    )
                input_len = tokens["input_ids"].shape[1]
                for row in generated:
                    new_tokens = row[input_len:]
                    raw_texts.append(self._tokenizer.decode(new_tokens, skip_special_tokens=True))
        return [_strip_think(text) for text in raw_texts]

    def summarize_hourly(self, texts: Sequence[str], *, ticker: str | None = None) -> list[str]:
        texts = list(texts)
        if not texts:
            return []
        company = ticker or "the company"
        prompts = [
            HOURLY_PROMPT_QWEN.format(ticker_or_company=company, documents=text) for text in texts
        ]
        return self._generate(prompts, self.hourly_max_tokens)

    def summarize_daily(
        self, hourly_summaries_per_day: Sequence[Sequence[str]], *, ticker: str | None = None
    ) -> list[str]:
        days = list(hourly_summaries_per_day)
        if not days:
            return []
        company = ticker or "the company"
        prompts = []
        for day in days:
            concat_hourly = "\n".join(day) if day else ""
            prompts.append(DAILY_PROMPT_QWEN.format(ticker_or_company=company, concat_hourly=concat_hourly))
        return self._generate(prompts, self.daily_max_tokens)
