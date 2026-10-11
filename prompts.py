"""Prompt templates for the frozen LLM text pipeline (Algorithm 1 / Fig. 1).

These are plain ``str.format`` templates (not f-strings) so they can be
inspected, logged, and edited without touching Python code. They are used
by ``src/text_backends/qwen_backend.py``. The mT5 XL-Sum backend keeps its
own short, non-chat-style prompts (see ``src/summarization.py`` and
``src/text_backends/mt5_backend.py``) because mT5 XL-Sum is a plain seq2seq
summarizer, not an instruction-tuned chat model.
"""

from __future__ import annotations

# Hourly step (Algorithm 1, line "hourly summary <- Summarize(hour documents)").
# {ticker_or_company}: stock ticker or company name for this batch of news.
# {documents}: the cleaned, concatenated news text published during one hour.
HOURLY_PROMPT_QWEN = """You are a financial news analyst. Below is the news about {ticker_or_company}
published during one hour. Summarize the key factors in this information that
could affect the stock price. Be concise (at most 3 sentences). Preserve concrete
facts such as figures, guidance, and named events.

News:
{documents}

Summary:"""

# Daily step (Algorithm 1, line "daily summary <- Summarize(concat(hourly summaries))").
# {ticker_or_company}: stock ticker or company name for this trading day.
# {concat_hourly}: chronologically ordered hourly summaries, newline-joined.
DAILY_PROMPT_QWEN = """Below are chronologically ordered hourly summaries of news about {ticker_or_company}
for one trading day. Combine them into a single daily synopsis covering the key
factors that caused the stock price movement today. Be concise (at most 5 sentences).

Hourly summaries:
{concat_hourly}

Daily synopsis:"""
