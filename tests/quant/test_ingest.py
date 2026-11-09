"""The ingest pipeline, with an injected provider: no network, no vendor library."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from tradingagents_quant.config import QuantConfig
from tradingagents_quant.data.ingest import compare_series, ingest
from tradingagents_quant.data.normalize import normalize_vendor_frame
from tradingagents_quant.data.snapshot import SnapshotStore
from tradingagents_quant.errors import DataUnavailableError, IngestError, NoDataError
from tradingagents_quant.store.db import Database

from .fixtures import FETCHED_AT, FakeProvider, linear_closes, trading_days, yahoo_frame

DAYS = trading_days("2024-01-02", 30)
# The morning after the last fixture session: every bar is settled.
_NEXT = date.fromisoformat(DAYS[-1]) + timedelta(days=1)
NOW = datetime(_NEXT.year, _NEXT.month, _NEXT.day, 12, tzinfo=UTC)
CONFIG = QuantConfig.model_validate({"universe": {"symbols": ["SPY", "QQQ", "TLT"]}})


@pytest.fixture
def store():
    db = Database(":memory:")
    db.migrate()
    return SnapshotStore(db)


def _run(store, provider, symbols=("SPY", "QQQ", "TLT"), end=None):
    return ingest(store, symbols=symbols, start=DAYS[0], end=end, fetch=provider,
                  source="fixture", source_version="fixture 1", config=CONFIG, now=NOW)


def test_a_failing_symbol_is_recorded_and_gets_no_bars(store):
    provider = FakeProvider({
        "SPY": yahoo_frame(DAYS, linear_closes(30)),
        "QQQ": yahoo_frame(DAYS, linear_closes(30, 300)),
        "TLT": DataUnavailableError("throttled"),
    })
    result = _run(store, provider)
    snap = result.snapshot
    assert snap.status == "partial"
    assert snap.symbols() == ("QQQ", "SPY")
    assert [(f.symbol, f.error_type) for f in result.failures] == [("TLT", "DataUnavailableError")]
    assert store.db.query_one("SELECT COUNT(*) FROM bars WHERE symbol = 'TLT'")[0] == 0
    event = store.db.query_one("SELECT level, message FROM system_events ORDER BY id DESC")
    assert event["level"] == "warning" and snap.snapshot_id in event["message"]


def test_when_everything_fails_no_snapshot_is_written(store):
    provider = FakeProvider({"SPY": NoDataError("unknown"), "QQQ": DataUnavailableError("down")})
    with pytest.raises(IngestError, match="no snapshot written"):
        _run(store, provider, symbols=("SPY", "QQQ"))
    assert store.list() == []
    assert store.db.query_one("SELECT level FROM system_events")[0] == "error"


def test_a_bug_is_not_mistaken_for_a_data_failure(store):
    provider = FakeProvider({"SPY": KeyError("bug in an adapter")})
    with pytest.raises(KeyError):
        _run(store, provider, symbols=("SPY",))


def test_bars_past_the_requested_end_are_not_kept(store):
    provider = FakeProvider({"SPY": yahoo_frame(DAYS, linear_closes(30))})
    snap = _run(store, provider, symbols=("SPY",), end=DAYS[19]).snapshot
    assert snap.series_info("SPY").last_date == DAYS[19]
    assert snap.requested_end == DAYS[19]


def test_calendar_gaps_are_found_across_series(store):
    provider = FakeProvider({
        "SPY": yahoo_frame(DAYS, linear_closes(30)),
        "QQQ": yahoo_frame(DAYS[:5] + DAYS[6:], linear_closes(29)),
    })
    result = _run(store, provider, symbols=("SPY", "QQQ"))
    gaps = [(i.symbol, i.date) for i in result.issues if i.rule == "calendar_gap"]
    assert gaps == [("QQQ", DAYS[5])]
    assert result.snapshot.series_info("QQQ").quality == "warning"


def test_staleness_is_judged_against_the_day_before_the_fetch(store):
    late = datetime(2024, 4, 30, 12, tzinfo=UTC)
    provider = FakeProvider({"SPY": yahoo_frame(DAYS, linear_closes(30))})
    result = ingest(store, symbols=("SPY",), start=DAYS[0], end=None, fetch=provider,
                    source="fixture", source_version="fixture 1", config=CONFIG, now=late)
    assert result.snapshot.series_info("SPY").quality == "error"
    assert [i.rule for i in result.issues if i.severity == "error"] == ["stale_series"]


def test_drift_shows_restated_missing_and_new_values(store):
    snap = _run(store, FakeProvider({"SPY": yahoo_frame(DAYS, linear_closes(30))}),
                symbols=("SPY",)).snapshot
    stored = store.bars(snap.snapshot_id, "SPY")

    same = normalize_vendor_frame("SPY", "SPY", yahoo_frame(DAYS, linear_closes(30)), fetched_at=FETCHED_AT)
    assert compare_series("SPY", stored, same).clean

    restated = yahoo_frame(DAYS[:3] + DAYS[4:], linear_closes(30)[:3] + linear_closes(30)[4:])
    restated.iloc[0, restated.columns.get_loc("Adj Close")] = 1.0  # a new dividend adjustment
    drift = compare_series(
        "SPY", stored, normalize_vendor_frame("SPY", "SPY", restated, fetched_at=FETCHED_AT)
    )
    assert drift.missing_dates == (DAYS[3],)
    assert [(d, c) for d, c, *_ in drift.changed] == [(DAYS[0], "adj_close")]
    assert not drift.clean
