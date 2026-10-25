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
HOURLY_PROMPT_QWEN = """You are a financial news analyst. Summarize in English the information below about
{ticker_or_company} from the past hour. Capture every factor that could affect the
stock price.

Rules:

- Use only what the text states. Do not add outside knowledge.
- Keep all concrete details: figures, percentages, dollar amounts, guidance,
  estimates versus actuals, deals, regulatory actions, analyst rating or price
  target changes, product and management events, and named parties.
- For each factor, say whether it is positive, negative, or neutral for
  {ticker_or_company}.
- Merge duplicate reports of the same event into one point.
- Ignore content unrelated to {ticker_or_company}'s business or share price.
- Match length to content: one or two sentences if there is little news, more if
  there are several distinct events. Do not leave out a material fact to stay short,
  and do not pad.
-Even if the news is minor or only loosely related, write one sentence stating what it reports and how relevant it is to {ticker_or_company}’s stock (high, medium, or low)
- Output only the summary.

Information:
{documents}"""

# Daily step (Algorithm 1, line "daily summary <- Summarize(concat(hourly summaries))").
# {ticker_or_company}: stock ticker or company name for this trading day.
# {concat_hourly}: chronologically ordered hourly summaries, newline-joined.
DAILY_PROMPT_QWEN = """You are a financial news analyst. Below is a compilation of the hourly summaries
for {ticker_or_company} for one trading day, in chronological order. Write a daily
synopsis in English of the factors likely to influence the stock price.

Rules:
- Use only what the summaries state. Do not add outside knowledge.
- Ignore "No material news" entries and merge repeated items.
- Cover every material event. Put the most significant first, and Preserve the most important figures,
  magnitude, and guidance.
- State whether each main factor is positive, negative, or neutral for
  {ticker_or_company}. If the day is mixed, say so.
- If a later summary updates an earlier one, report the latest status.
- Match length to content: a short paragraph on a quiet day, a longer one on a
  busy day. Do not drop a material fact to stay short, and do not pad.
-If the day's items are minor, write a short synopsis of what was reported and state that the overall impact is low.
- Output only the synopsis.

Hourly summaries:
{concat_hourly}"""
