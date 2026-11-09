"""Paper trading: the backtest engine stepped day by day, with state saved in
between; a ledger that shows tampering; suspension, splits and halts."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from tradingagents_quant import PAPER_LABEL
from tradingagents_quant.config import QuantConfig
from tradingagents_quant.data.ingest import ingest
from tradingagents_quant.data.snapshot import SnapshotStore
from tradingagents_quant.paper import ledger
from tradingagents_quant.paper.account import PaperError, get_account, latest_state, open_account
from tradingagents_quant.paper.runner import resume, run_account, summary
from tradingagents_quant.risk.engine import MarketHistory, RiskEngine
from tradingagents_quant.sim.engine import simulate_portfolio
from tradingagents_quant.sim.execution import CostModel
from tradingagents_quant.sim.runner import portfolio_targets, prepare
from tradingagents_quant.store.db import Database
from tradingagents_quant.strategy.spec import ExitSpec, StrategySpec

from .fixtures import FakeProvider, synthetic_bars, yahoo_frame

N = 600
DAYS = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2015-01-05", periods=N)]
CONFIG = QuantConfig.model_validate({
    "universe": {"symbols": ["SPY", "TLT"]},
    "risk": {"max_position_weight": 0.6, "max_drawdown": 0.3},
})
SPEC = StrategySpec(name="trend", rule="trend_following", params={"window": 30},
                    symbols=("SPY", "TLT"), exits=ExitSpec(stop_loss="percent", stop_value=0.06))


def _frames(n=N, spy=None, tlt=None):
    spy = synthetic_bars(n, seed=1, drift=0.0005)["close"] if spy is None else spy
    tlt = synthetic_bars(n, seed=2, drift=0.0001)["close"] if tlt is None else tlt
    days = DAYS[:n]
    return {"SPY": yahoo_frame(days, list(spy), dividends={days[100]: 0.5}),
            "TLT": yahoo_frame(days, list(tlt))}


def _ingest(store, frames, end):
    ingest(store, symbols=tuple(frames), start=DAYS[0], end=end, fetch=FakeProvider(frames),
           source="fixture", source_version="fixture 1", config=CONFIG,
           now=datetime(2030, 1, 1, tzinfo=UTC))


@pytest.fixture
def store():
    db = Database(":memory:")
    db.migrate()
    s = SnapshotStore(db)
    _ingest(s, _frames(), DAYS[-1])
    return s


def _fills(store, account):
    return [(e["date"], e["symbol"], e["side"], e["shares"], e["price"], e["commission"], e["reason"])
            for e in ledger.entries(store.db, account.account_id, ("fill",))]


def test_day_by_day_paper_trading_makes_the_backtests_fills(store):
    start = DAYS[200]
    acc = open_account(store, name="replay", spec=SPEC, config=CONFIG, start=start)
    cuts = [DAYS[i] for i in (230, 231, 232, 233, 234, 235, 280, 333, 334, 401, 470, 540, N - 1)]
    cuts += [DAYS[i] for i in range(541, 560)]  # a run of single days: state saved and reloaded
    for until in sorted(set(cuts)):
        run_account(store, acc, CONFIG, until=until)

    # The same days, once, through the backtest loop with the same risk engine.
    prep = prepare(store, None, SPEC.symbols, CONFIG)
    targets, labels = portfolio_targets(SPEC, prep.markets)
    calendar = prep.calendar[prep.calendar >= pd.Timestamp(start)]
    inside = {s: m.loc[m.index.isin(calendar)] for s, m in prep.markets.items()}
    costs = CostModel.from_settings(CONFIG.costs)
    risk = RiskEngine(CONFIG.risk, MarketHistory.from_markets(prep.markets), costs)
    backtest = simulate_portfolio(
        inside, {s: targets[s].loc[inside[s].index] for s in inside}, costs=costs,
        initial_cash=CONFIG.backtest.initial_cash, exits=SPEC.exits,
        rebalance_band=CONFIG.backtest.rebalance_band, calendar=calendar,
        entry_regimes={s: labels[s].loc[inside[s].index] for s in inside},
        order_filter=risk.as_filter())
    expected = [(f.date.strftime("%Y-%m-%d"), f.symbol, f.side, f.shares, f.price, f.commission,
                 f.reason) for f in backtest.fills]
    assert len(expected) > 10, "the scenario should trade"
    assert _fills(store, acc) == expected
    assert summary(store, acc)["equity"] == pytest.approx(float(backtest.equity.iloc[-1]), rel=1e-12)
    runs = store.db.query_one("SELECT COUNT(*) FROM paper_states WHERE account_id = ?",
                              (acc.account_id,))[0]
    assert runs == len(set(cuts))  # every state kept


def test_every_entry_is_labelled_and_the_chain_is_intact(store):
    acc = open_account(store, name="labels", spec=SPEC, config=CONFIG, start=DAYS[200])
    run_account(store, acc, CONFIG, until=DAYS[400])
    entries = ledger.entries(store.db, acc.account_id)
    assert {e["kind"] for e in entries} >= {"opened", "fill", "day", "risk"}
    assert all(e["label"] == PAPER_LABEL for e in entries if e["kind"] in ("fill", "day", "opened"))
    assert ledger.verify(store.db, acc.account_id) == []


def test_the_ledger_refuses_changes_and_shows_changes_made_around_it(store):
    acc = open_account(store, name="tamper", spec=SPEC, config=CONFIG, start=DAYS[200])
    run_account(store, acc, CONFIG, until=DAYS[300])
    db = store.db
    for statement in ("UPDATE paper_ledger SET payload_json = '{}'", "DELETE FROM paper_ledger",
                      "DELETE FROM paper_states", "UPDATE paper_accounts SET initial_cash = 1e9"):
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            db.execute(statement)
    db.execute("DROP TRIGGER paper_ledger_no_update")
    db.execute("UPDATE paper_ledger SET payload_json = replace(payload_json, '\"shares\":', "
               "'\"shares\":1') WHERE account_id = ? AND kind = 'fill' AND seq = "
               "(SELECT MIN(seq) FROM paper_ledger WHERE kind = 'fill')", (acc.account_id,))
    problems = ledger.verify(db, acc.account_id)
    assert problems and "does not match its hash" in problems[0]


def test_a_deep_drawdown_suspends_new_positions_but_not_exits(store):
    db = Database(":memory:")
    db.migrate()
    s = SnapshotStore(db)
    # A gap no exit can avoid: from 160 to 100 overnight, then a recovery.
    rise = list(np.linspace(100, 160, 300))
    crash = [100.0] * 20
    recovery = list(np.linspace(100, 170, 280))
    spy = pd.Series(rise + crash + recovery)
    _ingest(s, _frames(spy=spy), DAYS[-1])
    spec = StrategySpec(name="trend", rule="trend_following", params={"window": 10},
                        symbols=("SPY",))
    cfg = QuantConfig.model_validate({"universe": {"symbols": ["SPY"]},
                                      "risk": {"max_drawdown": 0.10, "max_position_weight": 1.0,
                                               "max_correlated_weight": 1.0}})
    acc = open_account(s, name="crash", spec=spec, config=cfg, start=DAYS[200])
    run = run_account(s, acc, cfg)
    assert run.status == "suspended"
    entries = ledger.entries(db, acc.account_id)
    suspended_at = next(e["date"] for e in entries if e["kind"] == "status")
    later_buys = [e for e in entries if e["kind"] == "fill" and e["side"] == "buy"
                  and e["date"] > suspended_at]
    refusals = [e for e in entries if e["kind"] == "rejection" and e["date"] > suspended_at]
    assert later_buys == []  # the recovery is not bought while suspended
    assert any("suspended" in e["detail"] for e in refusals)
    with pytest.raises(PaperError, match="reason"):
        resume(s, acc, reason=" ")
    resume(s, acc, reason="reviewed the crash, limits unchanged")
    assert latest_state(db, acc.account_id)["status"] == "active"


def test_a_split_after_the_last_run_is_converted_not_mistaken_for_a_crash():
    db = Database(":memory:")
    db.migrate()
    s = SnapshotStore(db)
    spy = synthetic_bars(N, seed=1, drift=0.002, vol=0.004)["close"]  # steadily up: held
    _ingest(s, _frames(spy=spy), DAYS[400])
    spec = StrategySpec(name="trend", rule="trend_following", params={"window": 30},
                        symbols=("SPY",))
    cfg = QuantConfig.model_validate({"universe": {"symbols": ["SPY"]},
                                      "risk": {"max_position_weight": 1.0,
                                               "max_correlated_weight": 1.0}})
    acc = open_account(s, name="split", spec=spec, config=cfg, start=DAYS[300])
    run_account(s, acc, cfg)
    before = summary(s, acc)
    held = before["positions"]["SPY"]["shares"]
    assert held > 0
    # A 2:1 split on day 402: the traded price halves from then on, and the
    # provider divides the history before it by 2 as well — the whole series
    # now sits at half the level, continuous across the split.
    split_at = 402
    frames = _frames(spy=spy / 2.0)
    frames["SPY"].iloc[split_at, frames["SPY"].columns.get_loc("Stock Splits")] = 2.0
    _ingest(s, frames, DAYS[450])
    run_account(s, acc, cfg)
    entries = ledger.entries(db, acc.account_id, ("rebase",))
    assert entries and entries[0]["factor"] == pytest.approx(2.0)
    after = summary(s, acc)
    assert after["status"] == "active"
    # No jump in equity from the change of basis: 50 days on a steady rise.
    assert 0.9 < after["equity"] / before["equity"] < 1.25


def test_an_unexplained_change_in_the_data_halts_the_account():
    db = Database(":memory:")
    db.migrate()
    s = SnapshotStore(db)
    spy = synthetic_bars(N, seed=1, drift=0.002, vol=0.004)["close"]
    _ingest(s, _frames(spy=spy), DAYS[400])
    spec = StrategySpec(name="trend", rule="trend_following", params={"window": 30},
                        symbols=("SPY",))
    cfg = QuantConfig.model_validate({"universe": {"symbols": ["SPY"]},
                                      "risk": {"max_position_weight": 1.0,
                                               "max_correlated_weight": 1.0}})
    acc = open_account(s, name="halt", spec=spec, config=cfg, start=DAYS[300])
    run_account(s, acc, cfg)
    _ingest(s, _frames(spy=spy / 1.5), DAYS[450])  # every price changed, no split
    run = run_account(s, acc, cfg)
    assert run.status == "halted" and "no split explains" in run.notes[0]
    again = run_account(s, acc, cfg)
    assert again.status == "halted" and again.days == []


def test_accounts_need_paper_mode_a_unique_name_and_positive_cash(store):
    off = QuantConfig.model_validate({"modes": {"paper_trading_mode": False}})
    with pytest.raises(PaperError, match="paper_trading_mode"):
        open_account(store, name="x", spec=SPEC, config=off)
    open_account(store, name="x", spec=SPEC, config=CONFIG)
    with pytest.raises(PaperError, match="exists"):
        open_account(store, name="x", spec=SPEC, config=CONFIG)
    with pytest.raises(PaperError, match="positive"):
        open_account(store, name="y", spec=SPEC, config=CONFIG, initial_cash=-5)
    assert get_account(store.db, "x").spec == SPEC


def test_a_second_run_without_new_data_changes_nothing(store):
    acc = open_account(store, name="idle", spec=SPEC, config=CONFIG, start=DAYS[200])
    first = run_account(store, acc, CONFIG)
    second = run_account(store, acc, CONFIG)
    assert first.days and second.days == [] and "no new trading day" in second.notes[0]
