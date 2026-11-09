"""taq ingest / snapshots / report / verify, end to end against a temporary data directory."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from tradingagents_quant import cli
from tradingagents_quant.errors import DataUnavailableError

from .fixtures import FakeProvider, linear_closes, trading_days, yahoo_frame

DAYS = trading_days("2024-01-02", 30)


@pytest.fixture
def run(tmp_path, monkeypatch):
    provider = FakeProvider({
        "SPY": yahoo_frame(DAYS, linear_closes(30)),
        "QQQ": yahoo_frame(DAYS, linear_closes(30, 300), volume=0),
        "TLT": DataUnavailableError("throttled"),
    })
    monkeypatch.setattr(cli, "_provider", lambda: (provider, "fixture", "fixture 1"))
    runner = CliRunner()
    env = {"TAQ_DATA_DIR": str(tmp_path / "data")}

    def invoke(*args):
        return runner.invoke(cli.app, list(args), env=env)

    invoke.provider = provider
    return invoke


def test_ingest_report_and_verify(run):
    result = run("ingest", "--symbols", "SPY,QQQ,TLT", "--end", DAYS[-1])
    assert result.exit_code == 0, result.output
    assert "(partial)" in result.output
    assert "FAILED   : TLT: DataUnavailableError" in result.output

    listing = run("snapshots")
    assert listing.exit_code == 0 and "partial" in listing.output and "2 series" in listing.output

    report = run("report", "--details")
    assert report.exit_code == 0, report.output
    assert "QQQ" in report.output and "zero_volume" in report.output
    assert "FAILED  TLT" in report.output

    verified = run("verify")
    assert verified.exit_code == 0 and "integrity ok" in verified.output


def test_verify_drift_reports_a_restatement(run):
    assert run("ingest", "--symbols", "SPY", "--end", DAYS[-1]).exit_code == 0
    restated = yahoo_frame(DAYS, linear_closes(30))
    restated.iloc[2, restated.columns.get_loc("Close")] += 1.0
    run.provider.frames["SPY"] = restated
    result = run("verify", "--drift")
    assert result.exit_code == 0, result.output
    assert "SPY: 1 values changed" in result.output
    assert "1 of 1 series restated" in result.output


def test_regime_shows_the_latest_label_and_indicator_values(run):
    from .fixtures import synthetic_bars

    bars = synthetic_bars(400, drift=0.003, vol=0.006)
    days = [d.strftime("%Y-%m-%d") for d in bars.index]
    run.provider.frames["SPY"] = yahoo_frame(days, list(bars["close"]))
    assert run("ingest", "--symbols", "SPY", "--end", days[-1]).exit_code == 0
    result = run("regime", "--symbol", "spy")
    assert result.exit_code == 0, result.output
    assert f"SPY on {days[-1]}" in result.output and "trend strong_up" in result.output
    assert "rsi_14" in result.output and "atr_14" in result.output


def test_regime_of_a_symbol_not_in_the_snapshot_fails_clearly(run):
    assert run("ingest", "--symbols", "SPY", "--end", DAYS[-1]).exit_code == 0
    result = run("regime", "--symbol", "NOPE")
    assert result.exit_code == 2 and "not in snapshot" in result.output


def test_report_without_a_snapshot_says_what_to_do(run):
    result = run("report")
    assert result.exit_code == 2
    assert "taq ingest" in result.output


def test_ingest_with_nothing_usable_fails_loudly(run):
    result = run("ingest", "--symbols", "TLT")
    assert result.exit_code == 2
    assert "no snapshot written" in result.output
