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


def _long_spy(run, n=600):
    from .fixtures import synthetic_bars

    bars = synthetic_bars(n, drift=0.0004, seed=12)
    days = [d.strftime("%Y-%m-%d") for d in bars.index]
    run.provider.frames["SPY"] = yahoo_frame(days, list(bars["close"]))
    assert run("ingest", "--symbols", "SPY", "--end", days[-1]).exit_code == 0
    return days


def test_backtest_prints_strategy_against_buy_and_hold_and_stores_it(run):
    days = _long_spy(run)
    result = run("backtest", "--strategy", "sma_cross", "--param", "fast=20", "--param",
                 "slow=100", "--start", days[150])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "PAPER TRADE — NOT REAL MONEY" in out and "buy & hold" in out
    assert f"{days[150]}..{days[-1]}" in out and "Survivorship bias" in out
    assert "stored as bt-" in out


def test_backtest_with_stops_and_vol_sizing(run):
    _long_spy(run)
    result = run("backtest", "--param", "fast=10", "--param", "slow=50", "--sizing", "vol_target",
                 "--fraction", "0.8", "--stop", "atr:3", "--take", "percent:0.15", "--no-save")
    assert result.exit_code == 0, result.output
    assert "not stored" in result.output


def test_backtest_of_several_instruments_shows_every_benchmark(run):
    from .fixtures import synthetic_bars

    days = _long_spy(run)
    run.provider.frames["TLT"] = yahoo_frame(days, list(synthetic_bars(600, seed=3)["close"]))
    assert run("ingest", "--symbols", "SPY,TLT", "--end", days[-1]).exit_code == 0
    result = run("backtest", "--strategy", "trend_following", "--param", "window=50",
                 "--symbol", "SPY,TLT", "--no-save")
    assert result.exit_code == 0, result.output
    header = next(line for line in result.output.splitlines() if "strategy" in line and "cash" in line)
    for column in ("buy & hold", "SMA 50/200", "cash"):
        assert column in header
    assert "avg exposure" in result.output


@pytest.mark.parametrize(
    "args",
    [["--param", "fast=300", "--param", "slow=100"], ["--param", "nonsense"],
     ["--stop", "trailing:3"], ["--strategy", "magic"], ["--symbol", "QQQ"]],
)
def test_backtest_refuses_bad_input(run, args):
    _long_spy(run)
    result = run("backtest", *args)
    assert result.exit_code == 2, result.output


def test_walkforward_records_an_experiment_and_refuses_silent_reuse(run, tmp_path):
    _long_spy(run, n=2300)
    args = ["walkforward", "--strategy", "sma_cross", "--grid", "fast=10,20", "--grid",
            "slow=60,100", "--train-years", "3", "--validation-years", "1", "--test-years", "1",
            "--min-trades", "0"]
    first = run(*args)
    assert first.exit_code == 0, first.output
    assert "walk-forward validation" in first.output and "VERDICT:" in first.output
    assert "recorded as wf-" in first.output and "higher Sharpe than buy & hold" in first.output
    listing = run("experiments")
    assert listing.exit_code == 0 and "sma_cross" in listing.output
    again = run(*args)
    assert again.exit_code == 2 and "earlier experiment" in again.output
    assert run(*args, "--allow-reuse").exit_code == 0

    path = tmp_path / "H-X.toml"
    path.write_text(
        '[hypothesis]\nid = "H-X"\nstatement = "test"\ncreated = "2026-11-09"\n[design]\n'
        'rule = "trend_following"\nsymbols = ["SPY"]\ngrid = {window = [50, 100]}\n'
        'train_years = 3\nvalidation_years = 1\ntest_years = 1\n[criteria]\nmin_oos_trades = 0\n'
        'max_variants = 2\n', encoding="utf-8")
    # Another rule, the same SPY test periods: still a second look at them.
    refused = run("walkforward", "--hypothesis", str(path))
    assert refused.exit_code == 2 and "earlier experiment" in refused.output
    hyp = run("walkforward", "--hypothesis", str(path), "--allow-reuse")
    assert hyp.exit_code == 0, hyp.output
    assert "H-X (pre-registered 2026-11-09): test" in hyp.output
    assert "reused from 2 earlier experiment(s)" in hyp.output
    assert "counts 3 trials" in hyp.output


def test_paper_trading_from_the_command_line(run):
    days = _long_spy(run, n=600)
    opened = run("paper", "open", "--name", "demo", "--strategy", "trend_following",
                 "--param", "window=30", "--start", days[300], "--cash", "50000")
    assert opened.exit_code == 0, opened.output
    assert "PAPER TRADE — NOT REAL MONEY" in opened.output and "50,000" in opened.output
    first = run("paper", "run")
    assert first.exit_code == 0, first.output
    assert f"demo: {days[300]}..{days[-1]}" in first.output and "buy" in first.output
    status = run("paper", "status", "--name", "demo")
    assert status.exit_code == 0 and "simulated equity" in status.output
    history = run("paper", "history", "--name", "demo", "--limit", "5000")
    assert history.exit_code == 0 and "opened with 50,000 simulated cash" in history.output
    assert run("paper", "verify").output.strip() == "demo: ledger intact"
    assert "demo" in run("paper", "list").output
    assert "no new day" in run("paper", "run").output
    assert run("paper", "open", "--name", "demo").exit_code == 2  # the name is taken
    assert run("paper", "resume", "--name", "demo", "--reason", "x").exit_code == 2  # not suspended


def test_report_without_a_snapshot_says_what_to_do(run):
    result = run("report")
    assert result.exit_code == 2
    assert "taq ingest" in result.output


def test_ingest_with_nothing_usable_fails_loudly(run):
    result = run("ingest", "--symbols", "TLT")
    assert result.exit_code == 2
    assert "no snapshot written" in result.output
