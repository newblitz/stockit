from datetime import date, time
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from src.data import (
    SPLIT_RANGES,
    _fill_short_missing_periods,
    _truncate_returns_from_training,
)
from prepare_embeddings import _available_trading_day, _is_near_duplicate, levenshtein_distance


def test_short_missing_period_is_forward_filled_as_flat_return() -> None:
    expected = tuple(date(2020, 1, day) for day in range(1, 22))
    dates = [day for day in expected if day != date(2020, 1, 3)]
    rows = np.tile(np.array([0.1, 0.2, 0.3, 0.4, 0.1, 100.0], dtype=np.float32), (20, 1))

    output_dates, output_rows, missing = _fill_short_missing_periods(dates, rows, expected)

    assert output_dates == list(expected)
    assert missing == 1
    np.testing.assert_array_equal(output_rows[2, :5], np.zeros(5))
    assert output_rows[2, 5] == 100.0


def test_stock_over_five_percent_missing_is_excluded() -> None:
    expected = tuple(date(2020, 1, day) for day in range(1, 22))
    rows = np.ones((19, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="missing"):
        _fill_short_missing_periods(list(expected[:19]), rows, expected)


def test_three_sigma_limits_are_fit_only_on_training_rows() -> None:
    dates = [date(2020, 6, 25), date(2020, 6, 26), date(2020, 6, 29),
             date(2020, 6, 30), date(2020, 7, 1)]
    values = np.zeros((5, 6), dtype=np.float32)
    values[:4, :5] = np.array([[0.0], [0.0], [0.0], [1.0]], dtype=np.float32)
    values[4, :5] = 100.0

    clipped = _truncate_returns_from_training(dates, values)
    # Training mean=.25 and std=sqrt(.1875), so the validation extreme is clipped
    # with training statistics rather than statistics fitted on all five rows.
    assert clipped[4, 0] < 2.0
    assert SPLIT_RANGES["train"][1] == date(2020, 6, 30)


def test_near_duplicate_rule_is_exact_levenshtein_not_ngram_jaccard() -> None:
    original = "earnings beat expectations revenue rises strongly"
    near_copy = "earnings beat expectations revenue rises strong"
    assert levenshtein_distance("kitten", "sitting") == 3
    assert _is_near_duplicate(near_copy, [original])
    assert not _is_near_duplicate("unrelated central bank announcement", [original])


def test_after_close_news_moves_to_the_next_trading_session() -> None:
    sessions = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]
    # 22:00 UTC is 17:00 in New York during January: after the 16:00 close.
    assert _available_trading_day("2020-01-02 22:00:00", sessions, ZoneInfo("America/New_York"), time(16)) == date(2020, 1, 3)
    # Friday after close is available on the following Monday, not Friday.
    assert _available_trading_day("2020-01-03 22:00:00", sessions, ZoneInfo("America/New_York"), time(16)) == date(2020, 1, 6)
