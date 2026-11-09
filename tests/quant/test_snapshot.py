"""Snapshots round-trip exactly, are sealed by the database, and prove their integrity."""

from __future__ import annotations

import math
import sqlite3

import numpy as np
import pytest

from tradingagents_quant.config import QuantConfig
from tradingagents_quant.data.normalize import normalize_vendor_frame
from tradingagents_quant.data.quality import Issue
from tradingagents_quant.data.snapshot import Failure, SnapshotStore
from tradingagents_quant.errors import SnapshotError
from tradingagents_quant.store.db import Database

from .fixtures import FETCHED_AT, trading_days, yahoo_frame

DAYS = trading_days("2024-01-02", 20)
# Values with no short decimal form, so a lossy round trip would show.
CLOSES = [100 / 3 + i * math.pi for i in range(20)]


def _series(symbol="SPY", closes=CLOSES, **kw):
    return normalize_vendor_frame(symbol, symbol, yahoo_frame(DAYS, closes, **kw), fetched_at=FETCHED_AT)


@pytest.fixture
def store():
    db = Database(":memory:")
    db.migrate()
    return SnapshotStore(db)


def _create(store, series, issues=(), failures=(), created_at="2026-11-09T10:00:00.000000Z"):
    return store.create(series=series, issues=list(issues), failures=list(failures),
                        source="test", source_version="fixture 1", requested_start=DAYS[0],
                        requested_end=None, config=QuantConfig(), created_at=created_at)


def test_bars_and_actions_round_trip_bit_for_bit(store):
    s = _series(dividends={DAYS[3]: 0.123456789})
    s.bars.iloc[5, s.bars.columns.get_loc("high")] = np.nan  # a value the provider did not report
    snap = _create(store, [s])
    back = store.bars(snap.snapshot_id, "SPY")
    assert [d.strftime("%Y-%m-%d") for d in back.index] == DAYS
    for column in s.bars.columns:
        for a, b in zip(s.bars[column], back[column], strict=True):
            assert (math.isnan(a) and math.isnan(b)) or a == b
    assert store.actions(snap.snapshot_id, "SPY").to_dict("records") == [
        {"date": DAYS[3], "kind": "dividend", "value": 0.123456789}
    ]


def test_the_end_date_is_a_point_in_time_cutoff(store):
    snap = _create(store, [_series()])
    cut = store.bars(snap.snapshot_id, "SPY", end=DAYS[9])
    assert cut.index[-1].strftime("%Y-%m-%d") == DAYS[9] and len(cut) == 10
    assert len(store.bars(snap.snapshot_id, "SPY", start=DAYS[15])) == 5


def test_identical_data_hashes_identically_and_one_changed_value_does_not(store):
    a = _create(store, [_series()], created_at="2026-11-09T10:00:00.000000Z")
    b = _create(store, [_series()], created_at="2026-11-09T11:00:00.000000Z")
    changed = list(CLOSES)
    changed[7] = math.nextafter(changed[7], math.inf)  # the smallest possible change
    c = _create(store, [_series(closes=changed)], created_at="2026-11-09T12:00:00.000000Z")
    assert a.content_hash == b.content_hash and a.snapshot_id != b.snapshot_id
    assert a.content_hash != c.content_hash
    assert a.series_info("SPY").sha256 != c.series_info("SPY").sha256


def test_status_ids_quality_and_failures(store):
    issues = [Issue("QQQ", "zero_volume", "warning", "volume 0", DAYS[2])]
    snap = _create(store, [_series("SPY"), _series("QQQ")], issues,
                   [Failure("TLT", "DataUnavailableError", "throttled")])
    assert snap.snapshot_id.startswith("snap-20261109T100000Z-")
    assert snap.status == "partial"
    assert {s.symbol: s.quality for s in snap.series} == {"QQQ": "warning", "SPY": "ok"}
    assert store.failures(snap.snapshot_id) == [Failure("TLT", "DataUnavailableError", "throttled")]
    assert store.issues(snap.snapshot_id, "QQQ") == issues
    fingerprint = store.db.query_one("SELECT fingerprint FROM config_versions")[0]
    assert fingerprint == snap.config_fingerprint == QuantConfig().fingerprint()


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE bars SET close = 1 WHERE symbol = 'SPY'",
        "DELETE FROM bars WHERE symbol = 'SPY'",
        "INSERT INTO bars (snapshot_id, symbol, date, close) VALUES ('{sid}', 'SPY', '2099-01-01', 1)",
        "UPDATE corporate_actions SET value = 9",
        "DELETE FROM snapshot_series",
        "INSERT INTO quality_issues (snapshot_id, symbol, rule, severity, detail) "
        "VALUES ('{sid}', 'SPY', 'x', 'info', 'late')",
        "UPDATE snapshots SET status = 'complete'",
        "DELETE FROM snapshots",
    ],
)
def test_a_sealed_snapshot_cannot_be_changed(store, statement):
    snap = _create(store, [_series(dividends={DAYS[1]: 0.1})])
    with pytest.raises(sqlite3.DatabaseError, match="sealed|immutable"):
        store.db.execute(statement.format(sid=snap.snapshot_id))


def test_integrity_holds_and_tampering_is_found(store):
    snap = _create(store, [_series("SPY"), _series("QQQ")])
    assert store.verify_integrity(snap.snapshot_id) == {}
    # Tampering has to go around the database's own guard first.
    store.db.execute("DROP TRIGGER bars_no_update")
    store.db.execute(
        "UPDATE bars SET close = close + 0.01 WHERE symbol = 'QQQ' AND date = ?", (DAYS[4],)
    )
    assert set(store.verify_integrity(snap.snapshot_id)) == {"QQQ"}


def test_the_same_series_twice_is_refused_not_merged(store):
    with pytest.raises(SnapshotError, match="more than once: SPY"):
        _create(store, [_series("SPY"), _series("SPY", closes=[1.0] * 20)])


def test_a_create_that_fails_midway_writes_nothing(store):
    # The duplicate failure violates a key only after all bars are written.
    failures = [Failure("TLT", "DataUnavailableError", "a"), Failure("TLT", "NoDataError", "b")]
    with pytest.raises(sqlite3.IntegrityError):
        _create(store, [_series("SPY")], failures=failures)
    assert store.db.query_one("SELECT COUNT(*) FROM snapshots")[0] == 0
    assert store.db.query_one("SELECT COUNT(*) FROM bars")[0] == 0


def test_create_refuses_nothing_and_empty_series(store):
    with pytest.raises(SnapshotError):
        _create(store, [])
    empty = normalize_vendor_frame("SPY", "SPY", yahoo_frame([], []), fetched_at=FETCHED_AT)
    with pytest.raises(SnapshotError, match="without bars"):
        _create(store, [empty])


def test_latest_and_resolve(store):
    with pytest.raises(SnapshotError, match="taq ingest"):
        store.resolve(None)
    first = _create(store, [_series()], created_at="2026-11-09T10:00:00.000000Z")
    second = _create(store, [_series()], created_at="2026-11-10T10:00:00.000000Z")
    assert store.latest().snapshot_id == second.snapshot_id
    assert store.resolve(first.snapshot_id).snapshot_id == first.snapshot_id
    assert [s.snapshot_id for s in store.list()] == [first.snapshot_id, second.snapshot_id]
    with pytest.raises(SnapshotError):
        store.get("snap-nope")
