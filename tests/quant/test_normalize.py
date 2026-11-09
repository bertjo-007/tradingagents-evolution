"""Normalization keeps what was reported, drops what cannot be stored, fills nothing."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pandas as pd

from tradingagents_quant.data.normalize import BAR_COLUMNS, normalize_vendor_frame

from .fixtures import FETCHED_AT, trading_days, yahoo_frame


def _norm(frame, **kw):
    return normalize_vendor_frame("SPY", "SPY", frame, fetched_at=kw.pop("fetched_at", FETCHED_AT), **kw)


def test_canonical_columns_dates_and_values():
    days = trading_days("2024-01-02", 3)
    s = _norm(yahoo_frame(days, [100, 101, 102]), currency="USD")
    assert list(s.bars.columns) == list(BAR_COLUMNS)
    assert list(s.bars.index) == days
    assert s.bars.loc[days[1], "close"] == 101.0
    assert s.bars.loc[days[1], "adj_close"] == 101 * 0.98
    assert s.currency == "USD" and s.dropped == ()


def test_a_positive_offset_market_keeps_its_local_date():
    # A Tokyo bar stamped 00:00 JST is 15:00 UTC the day before; converting to
    # UTC would move it back a day.
    s = _norm(yahoo_frame(["2024-03-05"], [100.0], tz="Asia/Tokyo"))
    assert list(s.bars.index) == ["2024-03-05"]


def test_the_unsettled_session_is_left_out():
    days = ["2024-05-06", "2024-05-07"]
    fetched = datetime(2024, 5, 7, 15, 0, tzinfo=UTC)  # 11:00 in New York, market open
    s = _norm(yahoo_frame(days, [100, 101]), fetched_at=fetched)
    assert list(s.bars.index) == ["2024-05-06"]
    assert [(d.date, d.reason) for d in s.dropped] == [("2024-05-07", "unsettled_session")]


def test_a_bar_without_close_is_dropped_and_recorded_not_filled():
    days = trading_days("2024-01-02", 3)
    frame = yahoo_frame(days, [100, 101, 102])
    frame.iloc[1, frame.columns.get_loc("Close")] = float("nan")
    frame.iloc[2, frame.columns.get_loc("High")] = float("nan")
    s = _norm(frame)
    assert list(s.bars.index) == [days[0], days[2]]
    assert [(d.date, d.reason) for d in s.dropped] == [(days[1], "missing_close")]
    assert math.isnan(s.bars.loc[days[2], "high"])  # reported as missing, kept as missing


def test_duplicates_identical_and_conflicting():
    days = trading_days("2024-01-02", 2)
    frame = yahoo_frame(days, [100, 101])
    twice = pd.concat([frame, frame.iloc[[0]]])
    conflicting = frame.iloc[[1]].copy()
    conflicting["Close"] = 999.0
    s = _norm(pd.concat([twice, conflicting]))
    reasons = sorted(d.reason for d in s.dropped)
    assert reasons == ["duplicate_conflicting", "duplicate_identical"]
    assert s.bars.loc[days[1], "close"] == 101.0  # the first report is kept


def test_actions_are_the_nonzero_dividends_and_splits():
    days = trading_days("2024-01-02", 4)
    s = _norm(yahoo_frame(days, [100] * 4, dividends={days[1]: 0.5}, splits={days[3]: 2.0}))
    assert s.actions.to_dict("records") == [
        {"date": days[1], "kind": "dividend", "value": 0.5},
        {"date": days[3], "kind": "split", "value": 2.0},
    ]


def test_a_missing_provider_column_becomes_missing_values():
    days = trading_days("2024-01-02", 2)
    frame = yahoo_frame(days, [100, 101]).drop(columns=["Adj Close"])
    s = _norm(frame)
    assert s.bars["adj_close"].isna().all()


def test_an_empty_frame_gives_an_empty_series():
    s = _norm(pd.DataFrame())
    assert s.bars.empty and s.actions.empty
