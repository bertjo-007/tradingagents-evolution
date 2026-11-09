"""The dashboard's read-only API, against a data directory filled by the real
commands: every value it serves must be one the system stored or computed."""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from tradingagents.agents import schemas  # noqa: E402
from tradingagents_quant import PAPER_LABEL, cli  # noqa: E402
from tradingagents_quant.api.app import STATIC, create_app  # noqa: E402
from tradingagents_quant.bridge.gate import evaluate, record  # noqa: E402
from tradingagents_quant.data.snapshot import SnapshotStore  # noqa: E402
from tradingagents_quant.store.db import Database  # noqa: E402

from .fixtures import FakeProvider, synthetic_bars, yahoo_frame  # noqa: E402

SECRET = "sk-test-0123456789-never-shown"
HOSTILE = '<img src=x onerror="alert(1)"> **Rating**: Buy'
_CONNECT = socket.socket.connect  # read at import, before any test's network guard


@pytest.fixture(autouse=True)
def _loopback_only(monkeypatch):
    """asyncio on Windows builds its self-pipe over a loopback connection, and
    the TestClient runs an event loop. Loopback is let through; every other
    address still meets the suite's guard."""
    guard = socket.socket.connect

    def connect(sock, address, *args):
        if isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1"):
            return _CONNECT(sock, address, *args)
        return guard(sock, address, *args)

    monkeypatch.setattr(socket.socket, "connect", connect)


def _no_nan(text: str):
    def refuse(name):
        raise AssertionError(f"{name} in a JSON response")
    return json.loads(text, parse_constant=refuse)


def _state(symbol, day, *, entry, stop, rating="Buy"):
    plan = schemas.render_research_plan(schemas.ResearchPlan(
        recommendation=schemas.PortfolioRating(rating), rationale="r", strategic_actions="a"))
    trader = schemas.render_trader_proposal(schemas.TraderProposal(
        action=schemas.TraderAction("Buy" if rating == "Buy" else "Hold"), reasoning="r",
        entry_price=entry, stop_loss=stop))
    pm = schemas.render_pm_decision(schemas.PortfolioDecision(
        rating=schemas.PortfolioRating(rating), executive_summary="s", investment_thesis="t",
        time_horizon="3 months"))
    return {"company_of_interest": symbol, "trade_date": day, "investment_plan": plan,
            "trader_investment_plan": trader, "final_trade_decision": pm, "final_rating": rating,
            "market_report": HOSTILE,
            "investment_debate_state": {"bull_history": "Bull: momentum.",
                                        "bear_history": "Bear: valuation."},
            "risk_debate_state": {"history": "Risk debate."}}


def seed(folder: Path) -> dict:
    """Fill ``folder`` the way a user would: ingest, backtest, walk-forward,
    two paper accounts (one refused by the risk engine), two agent decisions."""
    spy = synthetic_bars(2300, drift=0.0004, seed=12)
    days = [d.strftime("%Y-%m-%d") for d in spy.index]
    tlt = synthetic_bars(2300, drift=0.0001, seed=3)
    provider = FakeProvider({"SPY": yahoo_frame(days, list(spy["close"]),
                                                dividends={days[400]: 1.1}),
                             "TLT": yahoo_frame(days, list(tlt["close"]))})
    env = {"TAQ_DATA_DIR": str(folder), "TAQ_RISK_REQUIRE_STOP": "true"}
    runner = CliRunner()

    def run(*args):
        result = runner.invoke(cli.app, list(args), env=env)
        assert result.exit_code == 0, result.output
        return result.output

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli, "_provider", lambda: (provider, "fixture", "fixture 1"))
        run("ingest", "--symbols", "SPY,TLT", "--end", days[-1])
    run("backtest", "--strategy", "sma_cross", "--param", "fast=20", "--param", "slow=100",
        "--start", days[300])
    run("walkforward", "--strategy", "sma_cross", "--grid", "fast=10,20", "--grid",
        "slow=60,100", "--train-years", "3", "--validation-years", "1", "--test-years", "1",
        "--min-trades", "0")
    run("paper", "open", "--name", "trend", "--strategy", "trend_following", "--symbol",
        "SPY,TLT", "--param", "window=30", "--fraction", "0.3", "--stop", "percent:0.08",
        "--start", days[2000], "--cash", "50000")
    run("paper", "open", "--name", "no-stop", "--strategy", "trend_following", "--param",
        "window=30", "--start", days[2000])
    run("paper", "run")
    last = float(spy["close"].iloc[-1])
    approved = evaluate(_state("SPY", days[-1], entry=last, stop=last * 0.92), environ=env)
    blocked_state = _state("SPY", days[-1], entry=last, stop=last * 1.05)
    blocked = evaluate(blocked_state, environ=env)
    assert (approved.status, blocked.status) == ("approved", "blocked")
    ids = {"approved": record(_state("SPY", days[-1], entry=last, stop=last * 0.92), approved,
                              environ=env),
           "blocked": record(blocked_state, blocked, environ=env)}
    return {"env": env, "days": days, "last": last, "decisions": ids}


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:  # the per-test network guard is not up yet
        mp.setattr(socket.socket, "connect", _refuse_network)
        mp.setattr(socket, "getaddrinfo", _refuse_network)
        folder = tmp_path_factory.mktemp("dashboard") / "data"
        info = seed(folder)
    env = {**info["env"], "GOOGLE_API_KEY": SECRET}
    info["client"] = TestClient(create_app(environ=env))
    info["db_file"] = folder / "taq.sqlite3"
    return info


def _refuse_network(*args, **kwargs):
    raise OSError("network disabled in tests")


@pytest.fixture
def empty(tmp_path):
    env = {"TAQ_DATA_DIR": str(tmp_path / "none"), "GOOGLE_API_KEY": SECRET}
    return TestClient(create_app(environ=env))


def get(client, path, status=200):
    response = client.get(path)
    assert response.status_code == status, (path, response.text)
    return _no_nan(response.text) if status == 200 else None


# -- nothing stored: nothing shown ---------------------------------------------------


def test_without_a_database_every_view_is_empty_and_says_so(empty):
    s = get(empty, "/api/status")
    assert s["database"] is False and s["snapshot"] is None and s["market"] is None
    assert s["counts"] == {} and s["paper"] == []
    assert s["research_tasks"] == [] and "no scheduled research loop" in s["research_tasks_note"]
    assert s["modes"]["live_trading_mode"] is False
    assert s["modes"]["live_order_execution"] == "disabled"
    for path in ("/api/strategies", "/api/backtests", "/api/experiments", "/api/paper"):
        assert get(empty, path) == []
    assert get(empty, "/api/snapshot") == {}
    risk = get(empty, "/api/risk")
    assert risk["refusals"] == [] and risk["accounts"] == [] and risk["gate_refusals"] == []
    agents = get(empty, "/api/agents")
    assert agents["decisions"] == [] and agents["usage"] == [] and agents["cache_entries"] == 0
    assert all(h["latest"] is None for h in get(empty, "/api/hypotheses"))
    for path in ("/api/market/SPY", "/api/series/SPY", "/api/backtests/bt-x",
                 "/api/experiments/wf-x", "/api/paper/x", "/api/agents/ad-x"):
        get(empty, path, status=404)


def test_the_empty_state_holds_no_number_that_is_not_configuration(empty):
    """Outside the configuration and the schema, an empty system has no figure to show."""
    s = get(empty, "/api/status")
    numbers = []

    def walk(value, where):
        if isinstance(value, bool):
            return
        if isinstance(value, int | float):
            numbers.append(where)
        elif isinstance(value, dict):
            for k, v in value.items():
                walk(v, f"{where}.{k}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, f"{where}[{i}]")

    walk(s, "status")
    walk(get(empty, "/api/paper"), "paper")
    walk(get(empty, "/api/backtests"), "backtests")
    assert numbers == []


# -- stored data: shown as stored -------------------------------------------------------


def test_status_counts_what_the_database_holds(seeded):
    s = get(seeded["client"], "/api/status")
    assert s["database"] is True and s["schema_current"] is True
    assert s["counts"] == {"backtests": 1, "experiments": 1, "validated": s["counts"]["validated"],
                           "paper_accounts": 2, "agent_decisions": 2}
    assert s["snapshot"]["bars"] == 2 * 2300
    assert {x["symbol"] for x in s["snapshot"]["series"]} == {"SPY", "TLT"}
    m = s["market"]
    assert m["symbol"] == "SPY" and m["as_of"] == seeded["days"][-1]
    assert m["close"] == pytest.approx(seeded["last"])
    assert {"sma_50", "sma_200", "rsi_14", "atr_14", "hv_20", "dmi_14_adx"} <= set(m["indicators"])
    assert {a["name"] for a in s["paper"]} == {"trend", "no-stop"}


def test_series_are_the_snapshot_closes_thinned_but_ending_on_the_last_bar(seeded):
    ser = get(seeded["client"], "/api/series/spy")
    assert ser["symbol"] == "SPY" and len(ser["points"]) <= 1500
    assert ser["points"][0][0] == seeded["days"][0]
    assert ser["points"][-1] == [seeded["days"][-1], pytest.approx(seeded["last"])]
    assert get(seeded["client"], "/api/market/TLT")["symbol"] == "TLT"
    get(seeded["client"], "/api/market/QQQ", status=404)


def test_a_backtest_comes_with_its_benchmarks_trades_and_provenance(seeded):
    client = seeded["client"]
    (row,) = get(client, "/api/backtests")
    assert row["strategy"] == "sma_cross" and row["metrics"]["start"] == seeded["days"][300]
    b = get(client, f"/api/backtests/{row['run_id']}")
    assert b["label"] == PAPER_LABEL and b["snapshot_id"] and len(b["series_sha256"]) == 64
    assert b["equity"][-1][0] == seeded["days"][-1]
    assert b["equity"][-1][1] == pytest.approx(b["metrics"]["final_equity"])
    names = {x["name"] for x in b["benchmarks"]}
    assert "buy_and_hold" in names
    for x in b["benchmarks"]:
        assert x["equity"] and x["equity"][-1][0] == seeded["days"][-1]
    closed = [t for t in b["trades"] if t["exit_reason"] != "open"]
    assert len(closed) == b["metrics"]["trades"] and len(b["trades"]) - len(closed) <= 1
    for key in ("entry_date", "exit_date", "exit_reason", "holding_bars", "return_pct", "pnl"):
        assert key in b["trades"][0]
    assert b["regimes"]


def test_an_experiment_shows_its_folds_design_and_counted_trials(seeded):
    client = seeded["client"]
    (row,) = get(client, "/api/experiments")
    assert row["status"] == "complete" and row["verdict"] in (
        "validated", "failed_validation", "inconclusive")
    e = get(client, f"/api/experiments/{row['experiment_id']}")
    folds = e["summary"]["folds"]
    assert folds and all({"n", "periods", "chosen", "test", "benchmark"} <= set(f) for f in folds)
    assert e["design"]["design"]["rule"] == "sma_cross"
    assert sum(e["trials"].values()) > 0
    assert row["p_higher_sharpe"] == e["summary"]["deflated"]["p_deflated"]


def test_strategy_versions_count_their_runs(seeded):
    versions = get(seeded["client"], "/api/strategies")
    sma = [v for v in versions if v["rule"] == "sma_cross"]
    assert len({v["version"] for v in versions}) == len(versions)
    assert sum(v["backtests"] for v in sma) == 1 and sum(v["trials"] for v in sma) > 0
    assert sum(v["paper_accounts"] for v in versions) == 2
    rules = {r["rule"] for r in get(seeded["client"], "/api/rules")}
    assert {"buy_and_hold", "sma_cross", "trend_following"} <= rules


def test_paper_accounts_match_their_ledger(seeded):
    client = seeded["client"]
    db = Database(seeded["db_file"], read_only=True)
    try:
        store = SnapshotStore(db)
        from tradingagents_quant.paper.account import get_account
        from tradingagents_quant.paper.runner import summary

        expected = summary(store, get_account(db, "trend"))
    finally:
        db.close()
    a = get(client, "/api/paper/trend")
    assert a["label"] == PAPER_LABEL and a["ledger_intact"] is True
    assert a["equity"] == pytest.approx(expected["equity"]) and a["as_of"] == seeded["days"][-1]
    assert a["equity_curve"][-1] == [seeded["days"][-1], pytest.approx(expected["equity"])]
    assert any(e["kind"] == "fill" for e in a["events"])
    assert any(e["kind"] == "opened" for e in a["events"])
    assert all(e["label"] == PAPER_LABEL for e in a["events"])


def test_risk_shows_the_limits_the_refusals_and_the_gate(seeded):
    r = get(seeded["client"], "/api/risk")
    assert r["limits"]["require_stop"] is True  # the configuration the dashboard was given
    refused = [x for x in r["refusals"] if x["account"] == "no-stop"]
    assert refused and all(not x["accepted"] for x in refused)
    assert {a["name"] for a in r["accounts"]} == {"trend", "no-stop"}
    (gate,) = r["gate_refusals"]
    assert gate["decision_id"] == seeded["decisions"]["blocked"] and gate["reasons"]


def test_agent_decisions_carry_what_the_agents_wrote_verbatim(seeded):
    client = seeded["client"]
    a = get(client, "/api/agents")
    assert {d["gate"] for d in a["decisions"]} == {"approved", "blocked"}
    assert a["settings"]["llm_provider"] == "google"
    d = get(client, f"/api/agents/{seeded['decisions']['approved']}")
    assert d["gate"] == "approved" and d["sim_order"]["side"] == "buy"
    assert d["sim_order"]["label"] == PAPER_LABEL
    assert d["reports"]["bull_history"] == "Bull: momentum."
    assert d["reports"]["bear_history"] == "Bear: valuation."
    assert d["reports"]["risk_debate"] == "Risk debate."
    # Stored and served unchanged; the page escapes it (app.js: esc).
    assert d["reports"]["market_report"] == HOSTILE
    assert {c["name"] for c in d["verdict"]["checks"]}


# -- the boundaries ------------------------------------------------------------------


def test_settings_say_whether_a_key_is_set_never_what_it_is(seeded, empty):
    for client in (seeded["client"], empty):
        s = get(client, "/api/settings")
        assert s["credentials"]["GOOGLE_API_KEY"] is True
        assert s["credentials"]["OPENAI_API_KEY"] is False
        for path in ("/api/settings", "/api/status", "/api/agents", "/api/risk"):
            assert SECRET not in client.get(path).text


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_the_api_refuses_every_write(seeded, method):
    for path in ("/api/status", "/api/paper/trend", "/api/backtests", "/"):
        response = getattr(seeded["client"], method)(path)
        assert response.status_code == 405, (method, path)


def test_reading_changes_nothing_in_the_database(seeded):
    before = hashlib.sha256(seeded["db_file"].read_bytes()).hexdigest()
    client = seeded["client"]
    for path in ("/api/status", "/api/snapshot", "/api/strategies", "/api/backtests",
                 "/api/experiments", "/api/paper", "/api/paper/trend", "/api/risk",
                 "/api/agents", "/api/settings", "/api/hypotheses", "/api/series/SPY"):
        get(client, path)
    assert hashlib.sha256(seeded["db_file"].read_bytes()).hexdigest() == before
    assert client.get("/api/status").headers["cache-control"] == "no-store"


def test_the_page_loads_nothing_from_elsewhere_and_carries_the_label():
    for name in ("index.html", "app.js", "app.css"):
        text = (STATIC / name).read_text(encoding="utf-8")
        assert "http://" not in text and "https://" not in text, name
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert f'const PAPER = "{PAPER_LABEL}"' in js


def test_the_page_and_its_files_are_served(seeded):
    client = seeded["client"]
    page = client.get("/")
    assert page.status_code == 200 and '<script type="module" src="/static/app.js">' in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200


# -- taq dashboard ----------------------------------------------------------------------


@pytest.fixture
def served(monkeypatch, tmp_path):
    calls = []
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append((app, kw)))
    runner = CliRunner()

    def invoke(*args):
        return runner.invoke(cli.app, ["dashboard", *args],
                             env={"TAQ_DATA_DIR": str(tmp_path / "data")})

    invoke.calls = calls
    return invoke


def test_the_dashboard_listens_on_loopback_by_default(served):
    result = served()
    assert result.exit_code == 0, result.output
    ((app, kw),) = served.calls
    assert kw["host"] == "127.0.0.1" and kw["port"] == 8000
    assert PAPER_LABEL in result.output and "http://127.0.0.1:8000/" in result.output


def test_the_dashboard_refuses_another_address_unless_asked(served):
    refused = served("--host", "0.0.0.0")
    assert refused.exit_code == 2 and "no login" in refused.output and not served.calls
    assert served("--host", "0.0.0.0", "--allow-remote").exit_code == 0
    assert served.calls[0][1]["host"] == "0.0.0.0"


def test_the_dashboard_brings_an_old_schema_up_to_date(served, tmp_path):
    import sqlite3

    from tradingagents_quant.store.migrations import MIGRATIONS

    folder = tmp_path / "data"
    folder.mkdir()
    conn = sqlite3.connect(folder / "taq.sqlite3")
    conn.executescript("BEGIN;\n" + MIGRATIONS[0] + "\nPRAGMA user_version = 1;\nCOMMIT;")
    conn.close()
    assert served().exit_code == 0
    db = Database(folder / "taq.sqlite3")
    try:
        from tradingagents_quant.store.migrations import SCHEMA_VERSION

        assert db.schema_version == SCHEMA_VERSION
    finally:
        db.close()


def test_the_dashboard_does_not_create_a_database(served, tmp_path):
    assert served().exit_code == 0
    assert not (tmp_path / "data" / "taq.sqlite3").exists()
