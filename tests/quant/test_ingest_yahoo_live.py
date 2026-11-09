"""Against the real Yahoo Finance. Runs only on request: pytest -m integration."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tradingagents_quant.data.ingest_yahoo import fetch_yahoo
from tradingagents_quant.data.normalize import normalize_vendor_frame
from tradingagents_quant.data.quality import check_series, reference_date

pytestmark = pytest.mark.integration


def test_a_month_of_spy_arrives_complete_and_clean():
    raw = fetch_yahoo("SPY", "2024-01-02", "2024-01-31")
    series = normalize_vendor_frame("SPY", raw.canonical, raw.frame,
                                    fetched_at=datetime.now(UTC), currency=raw.currency)
    assert raw.currency == "USD"
    assert len(series.bars) == 21  # NYSE sessions in January 2024 from the 2nd
    assert series.first_date == "2024-01-02" and series.last_date == "2024-01-31"
    issues = check_series(series, as_of=reference_date(datetime.now(UTC).date(), "2024-01-31"),
                          max_stale_days=10)
    assert [i for i in issues if i.severity == "error"] == []


def test_dividends_and_splits_are_captured():
    raw = fetch_yahoo("QQQ", "2000-03-01", "2000-12-31")
    series = normalize_vendor_frame("QQQ", raw.canonical, raw.frame, fetched_at=datetime.now(UTC))
    splits = series.actions[series.actions["kind"] == "split"]
    assert splits.to_dict("records") == [{"date": "2000-03-20", "kind": "split", "value": 2.0}]
