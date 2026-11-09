"""Each quality rule fires on its fault and only there."""

from __future__ import annotations

from datetime import date

import pytest

from tradingagents_quant.data import quality
from tradingagents_quant.data.normalize import normalize_vendor_frame
from tradingagents_quant.data.quality import Issue, check_calendar, check_series, worst

from .fixtures import FETCHED_AT, linear_closes, trading_days, yahoo_frame

DAYS = trading_days("2024-01-02", 30)
AS_OF = date.fromisoformat(DAYS[-1])


def _series(frame, symbol="SPY"):
    return normalize_vendor_frame(symbol, symbol, frame, fetched_at=FETCHED_AT)


def _rules(issues, severity=None):
    return sorted({i.rule for i in issues if severity is None or i.severity == severity})


def _check(frame):
    return check_series(_series(frame), as_of=AS_OF, max_stale_days=10)


def test_a_clean_series_has_no_issue():
    issues = _check(yahoo_frame(DAYS, linear_closes(30)))
    assert issues == [] and worst(issues) == "ok"


def test_a_non_positive_price_is_an_error():
    frame = yahoo_frame(DAYS, linear_closes(30))
    frame.iloc[5, frame.columns.get_loc("Low")] = 0.0
    assert _rules(_check(frame), "error") == ["non_positive_price"]


def test_material_ohlc_inconsistency_is_an_error_rounding_a_warning():
    frame = yahoo_frame(DAYS, linear_closes(30))
    frame.iloc[3, frame.columns.get_loc("High")] = frame.iloc[3]["Close"] * 0.95  # 5% below close
    frame.iloc[7, frame.columns.get_loc("High")] = frame.iloc[7]["Close"] * (1 - 1e-4)
    issues = _check(frame)
    assert _rules(issues, "error") == ["ohlc_inconsistent"]
    assert _rules(issues, "warning") == ["ohlc_rounding"]


def test_volume_rules():
    frame = yahoo_frame(DAYS, linear_closes(30))
    frame.iloc[2, frame.columns.get_loc("Volume")] = -5
    frame.iloc[4, frame.columns.get_loc("Volume")] = 0
    issues = _check(frame)
    assert _rules(issues, "error") == ["negative_volume"]
    assert _rules(issues, "warning") == ["zero_volume"]


def test_an_extreme_move_is_flagged_unless_a_split_explains_it():
    closes = linear_closes(30)
    closes[10] = closes[9] * 0.5
    closes[11:] = [c * 0.5 for c in closes[11:]]
    flagged = _check(yahoo_frame(DAYS, closes))
    explained = _check(yahoo_frame(DAYS, closes, splits={DAYS[10]: 2.0}))
    assert "extreme_move" in _rules(flagged, "warning")
    assert "extreme_move" not in _rules(explained)


def test_a_stale_series_is_an_error():
    issues = check_series(_series(yahoo_frame(DAYS, linear_closes(30))),
                          as_of=date(2024, 3, 31), max_stale_days=10)
    assert _rules(issues, "error") == ["stale_series"]


def test_dropped_rows_are_reported_with_their_severity():
    frame = yahoo_frame(DAYS, linear_closes(30))
    frame.iloc[4, frame.columns.get_loc("Close")] = float("nan")
    conflicting = frame.iloc[[6]].copy()
    conflicting["Close"] = 1.0
    import pandas as pd

    issues = _check(pd.concat([frame, conflicting]))
    assert _rules(issues, "warning") == ["missing_close"]
    assert _rules(issues, "error") == ["duplicate_conflicting"]


def test_many_hits_are_capped_with_a_summary():
    frame = yahoo_frame(DAYS, linear_closes(30), volume=0)
    hits = [i for i in _check(frame) if i.rule == "zero_volume"]
    assert len(hits) == quality.MAX_PER_RULE + 1
    assert "30 dates in total" in hits[-1].detail


def test_calendar_gaps_and_extra_dates():
    calendar = _series(yahoo_frame(DAYS, linear_closes(30)), "SPY")
    gappy = _series(yahoo_frame(DAYS[:10] + DAYS[11:], linear_closes(29)), "QQQ")
    extra = _series(yahoo_frame([*DAYS[:5], "2024-01-06"], linear_closes(6)), "TLT")  # a Saturday
    issues = check_calendar({"SPY": calendar, "QQQ": gappy, "TLT": extra}, "SPY")
    assert [(i.symbol, i.rule, i.date) for i in issues] == [
        ("QQQ", "calendar_gap", DAYS[10]),
        ("TLT", "calendar_extra", "2024-01-06"),
    ]


def test_without_the_calendar_symbol_gaps_are_unchecked_and_said_so():
    qqq = _series(yahoo_frame(DAYS, linear_closes(30)), "QQQ")
    issues = check_calendar({"QQQ": qqq}, "SPY")
    assert [(i.symbol, i.rule, i.severity) for i in issues] == [
        ("QQQ", "calendar_unavailable", "warning")
    ]


def test_worst_orders_the_severities():
    info = Issue("X", "r", "info", "")
    warn = Issue("X", "r", "warning", "")
    err = Issue("X", "r", "error", "")
    assert worst([info]) == "ok"
    assert worst([info, warn]) == "warning"
    assert worst([warn, err, info]) == "error"
    with pytest.raises(ValueError):
        Issue("X", "r", "fatal", "")
