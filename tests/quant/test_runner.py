"""Backtests from a snapshot: data checks, warm-up, benchmark, storage."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

import pytest

from tradingagents_quant.config import QuantConfig
from tradingagents_quant.data.ingest import ingest
from tradingagents_quant.data.snapshot import SnapshotStore
from tradingagents_quant.errors import DataError
from tradingagents_quant.sim import persist
from tradingagents_quant.sim.runner import SURVIVORSHIP_NOTE, run_backtest
from tradingagents_quant.store.db import Database
from tradingagents_quant.strategy.spec import StrategySpec

from .fixtures import FakeProvider, synthetic_bars, yahoo_frame

BARS = synthetic_bars(700, drift=0.0004, seed=9)
DAYS = [d.strftime("%Y-%m-%d") for d in BARS.index]
CONFIG = QuantConfig.model_validate({"universe": {"symbols": ["SPY", "BAD"]}})
SPEC = StrategySpec(name="cross", rule="sma_cross", params={"fast": 20, "slow": 100},
                    symbols=("SPY", "BAD"))


@pytest.fixture
def store():
    db = Database(":memory:")
    db.migrate()
    store = SnapshotStore(db)
    good = yahoo_frame(DAYS, list(BARS["close"]), dividends={DAYS[100]: 0.4, DAYS[400]: 0.4})
    bad = yahoo_frame(DAYS, list(BARS["close"]))
    bad.iloc[50, bad.columns.get_loc("Low")] = -1.0  # a data error
    ingest(store, symbols=("SPY", "BAD"), start=DAYS[0], end=DAYS[-1],
           fetch=FakeProvider({"SPY": good, "BAD": bad}), source="fixture",
           source_version="fixture 1", config=CONFIG, now=datetime(2030, 1, 1, tzinfo=UTC))
    return store


def test_a_backtest_runs_against_buy_and_hold(store):
    run = run_backtest(store, SPEC, CONFIG, symbol="SPY")
    assert run.metrics["bars"] == 700 and run.benchmark_metrics["bars"] == 700
    assert run.benchmark_spec.rule == "buy_and_hold"
    assert "total_return_vs_benchmark" in run.comparison
    assert run.benchmark_metrics["exposure"] > 0.99
    assert run.metrics["dividends"] >= 0 and run.benchmark_metrics["dividends"] > 0
    assert SURVIVORSHIP_NOTE in run.notes
    assert set(run.regimes)  # returns split by regime


def test_a_series_with_data_errors_is_refused(store):
    with pytest.raises(DataError, match="non_positive_price"):
        run_backtest(store, SPEC, CONFIG, symbol="BAD")


def test_the_window_starts_warm(store):
    start = DAYS[300]
    run = run_backtest(store, SPEC, CONFIG, symbol="SPY", start=start)
    assert run.metrics["start"] == start
    # SMA(100) needed 100 bars of history; it had them before the window.
    assert run.result.target.iloc[0] in (0.0, 1.0)
    assert len(run.result.equity) == 400


def test_the_end_is_a_cutoff(store):
    run = run_backtest(store, SPEC, CONFIG, symbol="SPY", end=DAYS[499])
    assert run.metrics["end"] == DAYS[499]


def test_refusals(store):
    with pytest.raises(ValueError, match="not an instrument"):
        run_backtest(store, SPEC, CONFIG, symbol="QQQ")
    with pytest.raises(DataError, match="fewer than two bars"):
        run_backtest(store, SPEC, CONFIG, symbol="SPY", start=DAYS[-1])
    off = QuantConfig.model_validate({"modes": {"backtesting_mode": False}})
    with pytest.raises(DataError, match="backtesting_mode"):
        run_backtest(store, SPEC, off, symbol="SPY")


def test_a_stored_run_is_traceable_and_append_only(store):
    run = run_backtest(store, SPEC, CONFIG, symbol="SPY")
    run_id = persist.save(store.db, run)
    db = store.db
    row = db.query_one("SELECT * FROM backtest_runs WHERE run_id = ?", (run_id,))
    assert row["strategy_version"] == SPEC.version_hash()
    assert row["snapshot_id"] == run.snapshot_id and row["series_sha256"] == run.series_sha256
    assert row["config_fingerprint"] == CONFIG.fingerprint() and row["code_version"]
    stored = json.loads(row["metrics_json"])
    assert stored["total_return"] == run.metrics["total_return"]
    child = db.query_one("SELECT kind, parent_run_id FROM backtest_runs WHERE parent_run_id = ?",
                         (run_id,))
    assert tuple(child) == ("benchmark", run_id)
    assert db.query_one("SELECT COUNT(*) FROM backtest_trades WHERE run_id = ?", (run_id,))[0] == \
        len(run.result.trades)
    series = json.loads(db.query_one("SELECT series_json FROM backtest_equity WHERE run_id = ?",
                                     (run_id,))[0])
    assert len(series) == 700 and series[-1][1] == pytest.approx(run.metrics["final_equity"])
    spec_json = json.loads(db.query_one("SELECT spec_json FROM strategy_versions WHERE version_hash = ?",
                                        (SPEC.version_hash(),))[0])
    assert StrategySpec.model_validate(spec_json) == SPEC  # the spec can be rebuilt exactly
    for statement in ("UPDATE backtest_runs SET metrics_json = '{}'", "DELETE FROM backtest_trades",
                      "UPDATE strategy_versions SET name = 'x'", "DELETE FROM backtest_equity"):
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            db.execute(statement)
