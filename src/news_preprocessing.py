"""§4.2 CMIN news preprocessing: cleaning, dedup, hour bucketing, session mapping.

This module factors out the dataset-agnostic news-preprocessing logic that was
originally embedded in ``prepare_embeddings.py`` so it can be shared by the new
``scripts/build_text_cache.py`` (Qwen3 / BGE-M3 backends) without duplicating
or diverging from the paper-faithful cleaning rules. ``prepare_embeddings.py``
still owns the mT5-specific embedding loop; it imports these helpers instead
of defining its own copies.
"""

from __future__ import annotations

import html
import json
import re
from bisect import bisect_left
from collections import defaultdict
from datetime import date, datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# §4.2 preprocessing constants
MIN_ARTICLE_CHARS = 50
DEDUP_THRESHOLD = 0.9
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")
US_CLOSE = time(16, 0)
CN_CLOSE = time(15, 0)


def clean_text(raw: str) -> str:
    """§4.2 text normalization: HTML unescape → lowercase → strip non-alphanumeric → collapse whitespace."""
    text = html.unescape(raw)
    text = text.lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def levenshtein_distance(left: str, right: str) -> int:
    """Exact character-level Levenshtein distance using O(min(n, m)) memory."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1, previous[right_index] + 1,
                previous[right_index - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def is_near_duplicate(candidate: str, kept: list[str]) -> bool:
    """Use the §4.2 >90% normalized Levenshtein-overlap rule."""
    for earlier in kept:
        scale = max(len(candidate), len(earlier))
        similarity = 1.0 if scale == 0 else 1 - levenshtein_distance(candidate, earlier) / scale
        if similarity > DEDUP_THRESHOLD:
            return True
    return False


def market_session(news_dir: Path) -> tuple[ZoneInfo, time]:
    """Return the documented market-local cutoff for a CMIN news directory."""
    if "CMIN-CN" in str(news_dir):
        return ZoneInfo("Asia/Shanghai"), CN_CLOSE
    return ZoneInfo("America/New_York"), US_CLOSE


def available_trading_day(
    timestamp: str, trading_days: list[date], market_tz: ZoneInfo, close_time: time
) -> date | None:
    """Map UTC source timestamps to the first session where news is available.

    CMIN's timestamps have no explicit offset.  The Yahoo-news CMIN-US source
    is treated as UTC; it is converted to the exchange timezone, and an item
    published after the regular-session close is assigned to the next trading
    day.  This prevents it from entering an earlier same-day decision.
    """
    try:
        published = datetime.fromisoformat(timestamp).replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    local = published.astimezone(market_tz)
    candidate = local.date()
    if local.timetz().replace(tzinfo=None) > close_time:
        candidate = date.fromordinal(candidate.toordinal() + 1)
    index = bisect_left(trading_days, candidate)
    return trading_days[index] if index < len(trading_days) else None


def hourly_documents(news_dir: Path, trading_days: list[date]) -> list[list[str]]:
    """Convert CMIN JSONL headlines into deduplicated, cleaned, hour-level documents.

    Applies the three §4.2 preprocessing steps before text reaches the summarizer:
    1. HTML entity decoding + lowercasing + non-alphanumeric stripping + whitespace collapse.
    2. Articles shorter than MIN_ARTICLE_CHARS (50) characters after cleaning are dropped.
    3. Per-day near-duplicate removal (>90% normalized Levenshtein overlap): earliest is kept.

    Returns one entry per trading day; each entry is the list of chronologically
    ordered hour-level document strings for that day (may be empty).
    """
    # Collect under the first market session where the article is available.
    by_day: dict[date, list[tuple[str, str, str]]] = defaultdict(list)
    market_tz, close_time = market_session(news_dir)
    if news_dir.exists():
        for file in news_dir.iterdir():
            if not file.is_file():
                continue
            try:
                date.fromisoformat(file.name)
            except ValueError:
                continue
            for line in file.read_text().splitlines():
                record = json.loads(line)
                timestamp = record.get("created_at", "")
                available_day = available_trading_day(timestamp, trading_days, market_tz, close_time)
                if available_day is None:
                    continue
                hour = timestamp[:13] if len(timestamp) >= 13 else f"{available_day.isoformat()} 00"
                raw = record.get("text", "")
                if isinstance(raw, list):
                    raw = " ".join(raw)
                text = clean_text(raw)
                if len(text) >= MIN_ARTICLE_CHARS:
                    by_day[available_day].append((timestamp, hour, text))

    result: list[list[str]] = []
    for current_day in trading_days:
        # Sort ascending by timestamp so the earliest article wins deduplication.
        articles = sorted(by_day.get(current_day, []), key=lambda x: x[0])

        # Deduplicate: keep first occurrence, drop near-duplicates.
        kept_texts: list[str] = []
        kept_hours: list[str] = []
        for _ts, hour, text in articles:
            if not is_near_duplicate(text, kept_texts):
                kept_texts.append(text)
                kept_hours.append(hour)

        # Group deduplicated articles by hour, preserving chronological order within each hour.
        hourly: dict[str, list[str]] = defaultdict(list)
        for hour, text in zip(kept_hours, kept_texts):
            hourly[hour].append(text)

        result.append([" ".join(hourly[h]) for h in sorted(hourly)])

    return result
