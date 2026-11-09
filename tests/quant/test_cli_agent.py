"""taq agent: a confirmation before money is spent, and a recorded run that
replays exactly and for free — with scripted models, offline vendors."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest
from typer.testing import CliRunner

from tests.test_graph_end_to_end import (  # noqa: F401
    STRUCTURED,
    TRADE_DATE,
    ScriptedModel,
    _Client,
    offline,
)
from tradingagents.graph import trading_graph
from tradingagents_quant import cli
from tradingagents_quant.config import QuantConfig
from tradingagents_quant.data.ingest import ingest
from tradingagents_quant.data.snapshot import SnapshotStore
from tradingagents_quant.store.db import Database

from .fixtures import FakeProvider, synthetic_bars, yahoo_frame


@pytest.fixture
def env(tmp_path, monkeypatch):
    folder = tmp_path / "quant"
    monkeypatch.setenv("TAQ_DATA_DIR", str(folder))
    db = Database(folder / "taq.sqlite3")
    db.migrate()
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(end=TRADE_DATE, periods=300)]
    frame = yahoo_frame(days, list(synthetic_bars(300, seed=4, drift=0.0004)["close"]))
    ingest(SnapshotStore(db), symbols=("NVDA",), start=days[0], end=TRADE_DATE,
           fetch=FakeProvider({"NVDA": frame}), source="fixture", source_version="fixture 1",
           config=QuantConfig(), now=datetime(2030, 1, 1, tzinfo=UTC))
    db.close()
    return {"TAQ_DATA_DIR": str(folder)}


def _agent(env, *args):
    return CliRunner().invoke(cli.app, ["agent", "--ticker", "NVDA", "--date", TRADE_DATE,
                                        "--no-save", *args], env=env)


def test_a_run_that_costs_money_needs_a_yes(env):
    result = _agent(env)
    assert result.exit_code == 2 and "costs money" in result.output and "--yes" in result.output


@pytest.mark.usefixtures("offline")
def test_a_recorded_run_replays_identically_and_free(env, monkeypatch):
    model = ScriptedModel(structured=True)
    monkeypatch.setattr(trading_graph, "create_tier_client", lambda config, tier, **k: _Client(model))
    recorded = _agent(env, "--cache", "record", "--yes")
    assert recorded.exit_code == 0, recorded.output
    assert "agents' rating Overweight" in recorded.output
    assert "risk gate: APPROVED" in recorded.output
    assert "decision recorded as ad-" in recorded.output
    calls_after_recording = len(model.calls)

    replayed = _agent(env, "--cache", "replay")  # no --yes needed: it costs nothing
    assert replayed.exit_code == 0, replayed.output
    assert "agents' rating Overweight" in replayed.output and "risk gate: APPROVED" in replayed.output
    line = next(x for x in replayed.output.splitlines() if x.startswith("model calls"))
    total, cached = line.split(": ")[1].split(" (")[0], line.split("(")[1].split(" from")[0]
    assert total == cached and "0 tokens charged" in line
    # The model itself was not asked once. What still counted up are the
    # scripted structured answers, which bypass the model call (and so the cache).
    assert len(model.calls) - calls_after_recording == len(STRUCTURED)


@pytest.mark.usefixtures("offline")
def test_replaying_what_was_never_recorded_stops_the_run(env, monkeypatch):
    model = ScriptedModel(structured=True)
    monkeypatch.setattr(trading_graph, "create_tier_client", lambda config, tier, **k: _Client(model))
    result = _agent(env, "--cache", "replay")
    assert result.exit_code == 2 and "ReplayMiss" in result.output
