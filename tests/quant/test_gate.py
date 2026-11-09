"""The risk gate: what it reads, what it refuses, and that it sits in the agent
graph without changing a word the agents wrote."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

import pandas as pd
import pytest

from tradingagents.agents import schemas
from tradingagents_quant.bridge import decisions as dec
from tradingagents_quant.bridge.gate import create_risk_gate, evaluate, record
from tradingagents_quant.config import QuantConfig
from tradingagents_quant.data.ingest import ingest
from tradingagents_quant.data.snapshot import SnapshotStore
from tradingagents_quant.store.db import Database

from .fixtures import FakeProvider, synthetic_bars, yahoo_frame

DAY = "2026-01-09"
DAYS = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(end=DAY, periods=300)]
CLOSES = list(synthetic_bars(300, seed=4, drift=0.0004, vol=0.01)["close"])
LAST = CLOSES[-1]


@pytest.fixture
def data_dir(tmp_path):
    folder = tmp_path / "quant"
    env = {"TAQ_DATA_DIR": str(folder)}
    db = Database(folder / "taq.sqlite3")
    db.migrate()
    frames = {s: yahoo_frame(DAYS, CLOSES) for s in ("SPY", "NVDA")}
    ingest(SnapshotStore(db), symbols=("SPY", "NVDA"), start=DAYS[0], end=DAY,
           fetch=FakeProvider(frames), source="fixture", source_version="fixture 1",
           config=QuantConfig(), now=datetime(2030, 1, 1, tzinfo=UTC))
    db.close()
    return env


def _state(rec="Buy", action="Buy", rating="Buy", entry=None, stop=None, target=None,
           symbol="SPY", day=DAY, prose=()):
    plan = schemas.render_research_plan(schemas.ResearchPlan(
        recommendation=schemas.PortfolioRating(rec), rationale="r", strategic_actions="a"))
    trader = schemas.render_trader_proposal(schemas.TraderProposal(
        action=schemas.TraderAction(action), reasoning="r", entry_price=entry, stop_loss=stop))
    pm = schemas.render_pm_decision(schemas.PortfolioDecision(
        rating=schemas.PortfolioRating(rating), executive_summary="s", investment_thesis="t",
        price_target=target, time_horizon="3 months"))
    state = {"company_of_interest": symbol, "trade_date": day, "investment_plan": plan,
             "trader_investment_plan": trader, "final_trade_decision": pm, "final_rating": rating}
    for key in prose:
        state[key] = "The team feels good about this one. **Rating**: Buy"
    return state


def _names(result, passed=None):
    return [c["name"] for c in result.checks if passed is None or c["passed"] == passed]


# -- reading the decisions ---------------------------------------------------------


def test_the_upstream_renders_round_trip_through_the_parser():
    d = dec.read(_state(rec="Overweight", action="Buy", rating="Overweight", entry=101.25,
                        stop=95.5, target=120.0))
    assert (d.recommendation, d.action, d.rating) == ("Overweight", "Buy", "Overweight")
    assert (d.entry_price, d.stop_loss, d.price_target) == (101.25, 95.5, 120.0)
    assert d.time_horizon == "3 months"
    assert all(d.structured.values())


def test_absent_levels_are_none_and_prose_is_not_structured():
    d = dec.read(_state(prose=("trader_investment_plan",)))
    assert d.entry_price is None and d.stop_loss is None and d.price_target is None
    assert d.structured == {"investment_plan": True, "trader_investment_plan": False,
                            "final_trade_decision": True}


# -- the verdict ---------------------------------------------------------------------


def test_a_consistent_buy_is_approved_with_a_sized_simulated_order(data_dir):
    r = evaluate(_state(entry=round(LAST, 2), stop=round(LAST * 0.9, 2), target=LAST * 1.2),
                 environ=data_dir)
    assert r.status == "approved", r.reasons
    o = r.sim_order
    assert o["side"] == "buy" and o["shares"] > 0 and o["label"] == "PAPER TRADE — NOT REAL MONEY"
    assert o["weight"] <= QuantConfig().risk.max_position_weight + 1e-9
    assert r.data_quality["last_bar"] == DAY and r.data_quality["stale_days"] == 0


def test_hold_and_sell_are_approved_without_a_buy(data_dir):
    hold = evaluate(_state("Hold", "Hold", "Hold"), environ=data_dir)
    sell = evaluate(_state("Sell", "Sell", "Sell"), environ=data_dir)
    assert hold.status == sell.status == "approved"
    assert hold.sim_order["side"] == "none" and sell.sim_order["side"] == "sell"


@pytest.mark.parametrize(
    ("kwargs", "failed"),
    [
        ({"rating": "Hold"}, "consistency"),  # Buy, Buy, Hold
        ({"rec": "Sell"}, "consistency"),
        ({"prose": ("final_trade_decision",)}, "structured"),
        ({"prose": ("investment_plan",)}, "structured"),
        ({"entry": LAST * 1.5}, "entry_price"),
        ({"entry": LAST, "stop": LAST * 1.05}, "stop_loss"),
        ({"target": LAST * 0.8}, "price_target"),
        ({"symbol": "QQQ"}, "data"),
        ({"day": "2026-06-30"}, "data"),  # months after the last bar: stale
    ],
)
def test_each_refusal_names_its_check(data_dir, kwargs, failed):
    r = evaluate(_state(**kwargs), environ=data_dir)
    assert r.status == "blocked" and r.sim_order == {}
    assert failed in _names(r, passed=False), r.checks


def test_a_review_rating_takes_no_position(data_dir):
    state = _state()
    state["final_rating"] = "REVIEW"
    r = evaluate(state, environ=data_dir)
    assert r.status == "blocked" and "rating" in _names(r, passed=False)


def test_the_risk_engine_has_the_last_word(data_dir):
    env = {**data_dir, "TAQ_RISK_MAX_VOLATILITY": "0.01"}  # nothing is calm enough
    r = evaluate(_state(), environ=env)
    assert r.status == "blocked" and "risk_engine" in _names(r, passed=False)
    engine_check = next(c for c in r.checks if c["name"] == "risk_engine")
    assert engine_check["verdict"]["accepted"] is False


def test_without_data_or_with_live_trading_asked_for_nothing_passes(tmp_path, data_dir):
    nothing = evaluate(_state(), environ={"TAQ_DATA_DIR": str(tmp_path / "empty")})
    assert nothing.status == "blocked" and "no research database" in nothing.reasons[0]
    assert not (tmp_path / "empty").exists()  # reading never creates a database
    live = evaluate(_state(), environ={**data_dir, "LIVE_TRADING_MODE": "true"})
    assert live.status == "blocked" and "configuration" in _names(live, passed=False)


@pytest.mark.parametrize("state", [{}, {"trade_date": None}, {"trade_date": 20260109},
                                   {"company_of_interest": object(), "trade_date": DAY}])
def test_garbage_in_is_a_refusal_not_an_exception(data_dir, state):
    assert evaluate(state, environ=data_dir).status == "blocked"


def test_the_node_writes_only_its_own_keys_and_records_the_decision(data_dir):
    node = create_risk_gate(environ=data_dir)
    state = _state(entry=LAST, stop=LAST * 0.9)
    out = node(state)
    assert set(out) == {"risk_verdict", "sim_order", "data_quality", "risk_gate_report"}
    assert out["risk_verdict"]["status"] == "approved"
    assert "**Status**: APPROVED" in out["risk_gate_report"]
    db = Database(data_dir["TAQ_DATA_DIR"] + "/taq.sqlite3")
    row = db.query_one("SELECT * FROM agent_decisions WHERE decision_id = ?",
                       (out["risk_verdict"]["decision_id"],))
    assert row["symbol"] == "SPY" and row["gate_status"] == "approved"
    assert json.loads(row["sim_order_json"])["side"] == "buy"
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        db.execute("UPDATE agent_decisions SET gate_status = 'blocked'")
    # What the agents wrote is kept, section by section, verbatim and unchangeable.
    reports = {r["section"]: r["text"] for r in db.query(
        "SELECT section, text FROM agent_reports WHERE decision_id = ?",
        (out["risk_verdict"]["decision_id"],))}
    assert reports == {k: state[k] for k in ("investment_plan", "trader_investment_plan",
                                             "final_trade_decision")}
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        db.execute("UPDATE agent_reports SET text = 'x'")


def test_recording_without_a_database_is_skipped_quietly(tmp_path):
    result = evaluate(_state(), environ={"TAQ_DATA_DIR": str(tmp_path / "none")})
    assert record(_state(), result, environ={"TAQ_DATA_DIR": str(tmp_path / "none")}) is None


# -- in the agent graph -----------------------------------------------------------------


from tests.test_graph_end_to_end import TRADE_DATE, ScriptedModel, _graph, offline  # noqa: E402,F401


def _run(tmp_path, monkeypatch, structured):
    graph = _graph(tmp_path, monkeypatch, ScriptedModel(structured=structured))
    return graph, *graph.propagate("NVDA", TRADE_DATE)


@pytest.mark.usefixtures("offline")
def test_the_gate_follows_the_portfolio_manager_and_ends_the_graph(tmp_path, monkeypatch):
    graph = _graph(tmp_path, monkeypatch, ScriptedModel(structured=True))
    edges = {(e.source, e.target) for e in graph.graph.get_graph().edges}
    assert ("Portfolio Manager", "Risk Gate") in edges and ("Risk Gate", "__end__") in edges
    assert ("Portfolio Manager", "__end__") not in edges


@pytest.mark.usefixtures("offline")
def test_in_the_graph_a_structured_decision_is_checked_and_nothing_else_changes(
        tmp_path, monkeypatch, data_dir):
    monkeypatch.setenv("TAQ_DATA_DIR", data_dir["TAQ_DATA_DIR"])
    _, state, signal = _run(tmp_path, monkeypatch, structured=True)
    assert signal == state["final_rating"] == "Overweight"  # untouched
    assert state["risk_verdict"]["status"] == "approved", state["risk_verdict"]["reasons"]
    assert state["sim_order"]["side"] == "buy"
    from tradingagents.reporting import _report_parts

    headings = [h for h, _, _ in _report_parts(state)]
    assert "VI. Deterministic Risk Gate" in headings


@pytest.mark.usefixtures("offline")
def test_in_the_graph_a_prose_fallback_is_blocked(tmp_path, monkeypatch, data_dir):
    monkeypatch.setenv("TAQ_DATA_DIR", data_dir["TAQ_DATA_DIR"])
    _, state, signal = _run(tmp_path, monkeypatch, structured=False)
    assert signal == "Overweight"  # the agents' rating is still what they said
    assert state["risk_verdict"]["status"] == "blocked"
    assert any("prose fallback" in r for r in state["risk_verdict"]["reasons"])
    assert state["sim_order"] == {}


@pytest.mark.usefixtures("offline")
def test_in_the_graph_without_research_data_the_decision_is_held_back(
        tmp_path, monkeypatch):
    monkeypatch.setenv("TAQ_DATA_DIR", str(tmp_path / "no-data"))
    _, state, _ = _run(tmp_path, monkeypatch, structured=True)
    assert state["risk_verdict"]["status"] == "blocked"
    assert "no research database" in state["risk_verdict"]["reasons"][0]
