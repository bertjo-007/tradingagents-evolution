"""Command line of the research core: ``taq`` (or ``python -m tradingagents_quant.cli``)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from typing import Annotated

import typer

from tradingagents_quant import PAPER_LABEL, __version__, paths
from tradingagents_quant.config import config_file, load_config
from tradingagents_quant.errors import ConfigError, QuantError

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Deterministic research core: data, backtests, paper trading. No live trading.",
)


def _utf8_output() -> None:
    """Write UTF-8 even when the output is redirected.

    Python on Windows encodes redirected output in the ANSI code page, which
    turns the dash of the paper-trade label into a byte no UTF-8 reader shows.
    The label must arrive intact wherever the output goes.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if (stream.encoding or "").lower().replace("-", "") != "utf8":
                stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError):
            pass  # a stream that cannot be reconfigured (a test runner's buffer)


@app.callback()
def _main() -> None:
    """Runs before every command."""
    _utf8_output()


def _fail(exc: Exception) -> None:
    typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=2)


def _open():
    from tradingagents_quant.data.snapshot import SnapshotStore
    from tradingagents_quant.store.db import Database

    cfg = load_config()
    db = Database.open_default()
    return cfg, db, SnapshotStore(db)


def _provider():
    """The snapshot source: (fetch, source name, source version). Tests replace it."""
    from tradingagents_quant.data import ingest_yahoo

    return ingest_yahoo.fetch_yahoo, ingest_yahoo.SOURCE, ingest_yahoo.source_version()


@app.command("config")
def show_config() -> None:
    """Print the effective configuration, where it came from, and its fingerprint."""
    try:
        cfg = load_config()
        home = paths.data_home()
        source = config_file(os.environ, home)
    except ConfigError as exc:
        _fail(exc)
        return
    typer.echo(f"tradingagents_quant {__version__}")
    typer.echo(f"data directory : {home}")
    typer.echo(f"database       : {paths.db_path(home)}")
    typer.echo(f"config file    : {source or '(none, defaults)'}")
    typer.echo(f"fingerprint    : {cfg.fingerprint()}")
    typer.echo(f"orders         : {PAPER_LABEL}")
    typer.echo(json.dumps(cfg.to_dict(), indent=2))


@app.command("ingest")
def ingest_command(
    symbols: str = typer.Option("", help="Comma-separated symbols; default: the configured universe."),
    start: str = typer.Option("", help="First date (YYYY-MM-DD); default: data.history_start."),
    end: str = typer.Option("", help="Last date (YYYY-MM-DD, inclusive); default: last settled day."),
) -> None:
    """Fetch history from the provider and seal it as a new immutable snapshot."""
    from tradingagents_quant.data.ingest import ingest

    try:
        cfg, db, store = _open()
        wanted = (
            tuple(s.strip().upper() for s in symbols.split(",") if s.strip())
            or cfg.universe.ingest_symbols()
        )
        fetch, source, version = _provider()
        result = ingest(
            store, symbols=wanted, start=start or cfg.data.history_start, end=end or None,
            fetch=fetch, source=source, source_version=version, config=cfg,
        )
    except QuantError as exc:
        _fail(exc)
        return
    snap = result.snapshot
    counts = Counter(i.severity for i in result.issues)
    typer.echo(f"snapshot {snap.snapshot_id} ({snap.status})")
    typer.echo(f"  series   : {len(snap.series)}   bars: {sum(s.n_bars for s in snap.series)}")
    typer.echo(f"  issues   : {counts['error']} error, {counts['warning']} warning, {counts['info']} info")
    for f in result.failures:
        typer.secho(f"  FAILED   : {f.symbol}: {f.error_type}: {f.detail}", fg=typer.colors.YELLOW)
    typer.echo("  details  : taq report")


@app.command("snapshots")
def list_snapshots() -> None:
    """List the sealed snapshots, oldest first."""
    try:
        _, _, store = _open()
        snaps = store.list()
    except QuantError as exc:
        _fail(exc)
        return
    if not snaps:
        typer.echo("no snapshot yet; run `taq ingest`")
    for s in snaps:
        typer.echo(f"{s.snapshot_id}  {s.status:<8}  {len(s.series):>3} series  "
                   f"{s.source_version}  config {s.config_fingerprint or '-'}")


@app.command("report")
def report(
    snapshot: str = typer.Option("", help="Snapshot id; default: the latest."),
    details: bool = typer.Option(False, help="List every error and warning."),
) -> None:
    """Per-series quality report of a snapshot."""
    try:
        _, _, store = _open()
        snap = store.resolve(snapshot or None)
        issues = store.issues(snap.snapshot_id)
        failures = store.failures(snap.snapshot_id)
    except QuantError as exc:
        _fail(exc)
        return
    typer.echo(f"snapshot {snap.snapshot_id} ({snap.status}), created {snap.created_at}")
    typer.echo(f"source {snap.source} / {snap.source_version}, config {snap.config_fingerprint}")
    typer.echo(f"{'SYMBOL':<8}{'FIRST':<12}{'LAST':<12}{'BARS':>6}  {'QUALITY':<8}{'ERR':>4}{'WARN':>5}{'INFO':>5}")
    per = {s.symbol: Counter() for s in snap.series}
    for i in issues:
        per.setdefault(i.symbol, Counter())[i.severity] += 1
    for s in snap.series:
        c = per[s.symbol]
        typer.echo(f"{s.symbol:<8}{s.first_date:<12}{s.last_date:<12}{s.n_bars:>6}  {s.quality:<8}"
                   f"{c['error']:>4}{c['warning']:>5}{c['info']:>5}")
    for f in failures:
        typer.secho(f"FAILED  {f.symbol}: {f.error_type}: {f.detail}", fg=typer.colors.YELLOW)
    if details:
        for i in issues:
            if i.severity != "info":
                typer.echo(f"  {i.severity:<7} {i.symbol:<6} {i.date or '':<10} {i.rule}: {i.detail}")


@app.command("verify")
def verify(
    snapshot: str = typer.Option("", help="Snapshot id; default: the latest."),
    drift: bool = typer.Option(False, help="Also re-fetch and compare with the provider (network)."),
) -> None:
    """Re-hash the stored data; with --drift, show what the provider has restated since."""
    from tradingagents_quant.data.ingest import compare_series
    from tradingagents_quant.data.normalize import normalize_vendor_frame

    try:
        _, _, store = _open()
        snap = store.resolve(snapshot or None)
        broken = store.verify_integrity(snap.snapshot_id)
    except QuantError as exc:
        _fail(exc)
        return
    if broken:
        for symbol, (stored, now) in broken.items():
            typer.secho(f"CORRUPT {symbol}: stored {stored[:12]} != recomputed {now[:12]}",
                        fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.echo(f"integrity ok: {snap.snapshot_id}, {len(snap.series)} series match their hashes")
    if not drift:
        return
    from datetime import UTC, datetime

    fetch, _, _ = _provider()
    restated = 0
    for s in snap.series:
        try:
            raw = fetch(s.symbol, snap.requested_start, s.last_date)
        except QuantError as exc:
            typer.secho(f"  {s.symbol}: not re-fetched ({exc})", fg=typer.colors.YELLOW)
            continue
        fresh = normalize_vendor_frame(s.symbol, raw.canonical, raw.frame,
                                       fetched_at=datetime.now(UTC), currency=raw.currency)
        result = compare_series(s.symbol, store.bars(snap.snapshot_id, s.symbol), fresh)
        if result.clean:
            typer.echo(f"  {s.symbol}: unchanged")
            continue
        restated += 1
        first = result.changed[0] if result.changed else None
        typer.secho(
            f"  {s.symbol}: {len(result.changed)} values changed, {len(result.missing_dates)} dates "
            f"gone, {len(result.new_dates)} new" + (f"; first: {first[0]} {first[1]} "
                                                    f"{first[2]} -> {first[3]}" if first else ""),
            fg=typer.colors.YELLOW,
        )
    typer.echo(f"drift: {restated} of {len(snap.series)} series restated by the provider since "
               f"{snap.created_at}. The snapshot itself is unchanged.")


def _parse_params(items: list[str]) -> dict[str, float | int]:
    params: dict[str, float | int] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise typer.BadParameter(f"--param expects name=value, got {item!r}")
        number = float(value)
        params[key.strip()] = int(number) if number.is_integer() else number
    return params


def _parse_level(text: str) -> tuple[str, float]:
    if not text:
        return "none", 0.0
    kind, sep, value = text.partition(":")
    if not sep or kind not in ("percent", "atr"):
        raise typer.BadParameter(f"expected percent:<fraction> or atr:<multiple>, got {text!r}")
    return kind, float(value)


_BENCH_LABELS = {"buy_and_hold": "buy & hold", "index": "index", "sma_baseline": "SMA 50/200",
                 "cash": "cash"}


def _fmt(value, kind: str = "pct") -> str:
    if value is None:
        return "n/a"
    if kind == "pct":
        return f"{value:+.2%}"
    if kind == "share":
        return f"{value:.2%}"
    if kind == "num":
        return f"{value:.2f}"
    return f"{value:,.0f}"


@app.command("backtest")
def backtest_command(
    strategy: str = typer.Option("sma_cross", help="Rule name (see strategy library)."),
    symbol: str = typer.Option("SPY", help="Instrument(s) in the snapshot, comma-separated."),
    start: str = typer.Option("", help="First day of the test window (YYYY-MM-DD)."),
    end: str = typer.Option("", help="Last day of the test window (YYYY-MM-DD)."),
    param: Annotated[list[str] | None, typer.Option(
        help="Rule parameter, e.g. --param fast=50 (repeatable).")] = None,
    sizing: str = typer.Option("fixed_fraction", help="fixed_fraction | vol_target | atr_risk"),
    fraction: float = typer.Option(1.0, help="Largest weight of equity (0-1)."),
    stop: str = typer.Option("", help="Stop loss: percent:0.1 or atr:3"),
    take: str = typer.Option("", help="Take profit: percent:0.2 or atr:6"),
    max_hold: int = typer.Option(0, help="Exit after this many bars (0: no limit)."),
    signal_prices: str = typer.Option("total_return", help="total_return | price"),
    snapshot: str = typer.Option("", help="Snapshot id; default: the latest."),
    save: bool = typer.Option(True, help="Store the run in the database."),
) -> None:
    """Backtest a strategy on one or more instruments against buy and hold, the index,
    a moving-average baseline and cash. Simulated, not real money."""
    from tradingagents_quant.sim import persist
    from tradingagents_quant.sim.runner import run_backtest
    from tradingagents_quant.strategy.spec import ExitSpec, SizingSpec, StrategySpec

    try:
        cfg, db, store = _open()
        stop_kind, stop_value = _parse_level(stop)
        take_kind, take_value = _parse_level(take)
        spec = StrategySpec(
            name=strategy, rule=strategy, params=_parse_params(param or []), symbols=symbol,
            signal_prices=signal_prices, sizing=SizingSpec(method=sizing, fraction=fraction),
            exits=ExitSpec(stop_loss=stop_kind, stop_value=stop_value, take_profit=take_kind,
                           take_value=take_value, max_holding_bars=max_hold or None),
            benchmark=cfg.universe.benchmark,
        )
        run = run_backtest(store, spec, cfg, snapshot_id=snapshot or None,
                           start=start or None, end=end or None)
        run_id = persist.save(db, run) if save else None
    except (QuantError, ValueError) as exc:
        _fail(exc)
        return
    s = run.metrics
    columns = [("strategy", s)] + [(_BENCH_LABELS.get(b.name, b.name), b.metrics)
                                   for b in run.benchmarks]
    typer.echo(f"{PAPER_LABEL} — simulated backtest")
    typer.echo(f"{spec.name} on {run.symbol}, {s['start']}..{s['end']} ({s['bars']} bars), "
               f"snapshot {run.snapshot_id}, strategy version {spec.version_hash()}")
    typer.echo(f"{'':<16}" + "".join(f"{name:>13}" for name, _ in columns))
    rows = [("total return", "total_return", "pct"), ("CAGR", "cagr", "pct"),
            ("volatility", "volatility", "share"), ("Sharpe", "sharpe", "num"),
            ("Sortino", "sortino", "num"), ("max drawdown", "max_drawdown", "pct"),
            ("days invested", "exposure", "share"), ("avg exposure", "avg_gross_exposure", "share"),
            ("final equity", "final_equity", "cash"), ("costs", "total_costs", "cash"),
            ("dividends", "dividends", "cash")]
    for label, key, kind in rows:
        typer.echo(f"{label:<16}" + "".join(f"{_fmt(mt.get(key), kind):>13}" for _, mt in columns))
    typer.echo(f"trades {s['trades']} closed{' + 1 open' if s['open_trade'] else ''}, "
               f"win rate {_fmt(s['win_rate'])}, profit factor {_fmt(s['profit_factor'], 'num')}, "
               f"avg holding {_fmt(s['avg_holding_bars'], 'num')} bars, exits {s['exit_reasons']}")
    if s.get("annualization_note"):
        typer.echo(f"note: {s['annualization_note']}")
    invested = {k: v for k, v in run.regimes.items() if v["exposure"]}
    if invested:
        typer.echo("by regime (return while invested in it):")
        for label, r in sorted(invested.items(), key=lambda kv: -kv[1]["bars"])[:8]:
            typer.echo(f"  {label:<24} {r['share_of_bars']:>6.1%} of bars, exposure "
                       f"{r['exposure']:>6.1%}, compounded {_fmt(r['compounded_return'])}")
    for w in run.data_warnings[:5]:
        typer.secho(f"data warning: {w}", fg=typer.colors.YELLOW)
    for note in run.notes:
        typer.echo(f"note: {note}")
    typer.echo(f"stored as {run_id}" if run_id else "not stored (--no-save)")


def _parse_grid(items: list[str]) -> dict[str, list[float | int]]:
    out: dict[str, list[float | int]] = {}
    for item in items:
        key, sep, values = item.partition("=")
        if not sep or not values:
            raise typer.BadParameter(f"--grid expects name=v1,v2,..., got {item!r}")
        numbers = [float(v) for v in values.split(",") if v.strip()]
        out[key.strip()] = [int(x) if x.is_integer() else x for x in numbers]
    return out


@app.command("walkforward")
def walkforward_command(
    hypothesis: str = typer.Option("", help="Pre-registered hypothesis file (TOML)."),
    strategy: str = typer.Option("trend_following", help="Rule (without --hypothesis)."),
    symbol: str = typer.Option("SPY", help="Instrument(s), comma-separated."),
    grid: Annotated[list[str] | None, typer.Option(
        help="Parameter values to search, e.g. --grid window=100,150,200 (repeatable).")] = None,
    fixed: Annotated[list[str] | None, typer.Option(
        help="Parameter held fixed, e.g. --fixed rsi_window=2 (repeatable).")] = None,
    start: str = typer.Option("", help="First day of the history to use."),
    end: str = typer.Option("", help="Last day of the history to use."),
    train_years: float = typer.Option(6.0),
    validation_years: float = typer.Option(2.0),
    test_years: float = typer.Option(2.0),
    embargo: int = typer.Option(21, help="Bars left out between segments."),
    anchored: bool = typer.Option(False, help="Train from the first bar each time."),
    objective: str = typer.Option("sharpe", help="sharpe | sortino | cagr | calmar"),
    top_k: int = typer.Option(3, help="Finalists from training tried on validation."),
    min_trades: int = typer.Option(15, help="Out-of-sample trades needed for a verdict."),
    allow_reuse: bool = typer.Option(False, help="Knowingly reuse test periods of earlier "
                                                 "experiments (counted as trials)."),
    snapshot: str = typer.Option("", help="Snapshot id; default: the latest."),
    save: bool = typer.Option(True, help="Record the experiment in the database."),
) -> None:
    """Walk-forward validation: optimize on train, choose on validation, judge on test."""
    from tradingagents_quant.sim.walkforward import (
        WalkForwardDesign,
        load_hypothesis,
        run_walkforward,
    )

    try:
        cfg, _, store = _open()
        hyp = load_hypothesis(hypothesis) if hypothesis else None
        design = hyp.design if hyp else WalkForwardDesign(
            rule=strategy, symbols=symbol, grid=_parse_grid(grid or []),
            fixed_params=_parse_params(fixed or []), start=start or None, end=end or None,
            train_years=train_years, validation_years=validation_years, test_years=test_years,
            embargo_bars=embargo, anchored=anchored, objective=objective, top_k=top_k,
            min_oos_trades=min_trades)
        report = run_walkforward(store, design, cfg, snapshot_id=snapshot or None,
                                 allow_reuse=allow_reuse, hypothesis=hyp, save=save)
    except (QuantError, ValueError) as exc:
        _fail(exc)
        return
    d = report.design
    typer.echo(f"{PAPER_LABEL} — walk-forward validation")
    if hyp:
        typer.echo(f"{hyp.id} (pre-registered {hyp.created}): {hyp.statement}")
    typer.echo(f"{d.rule} on {','.join(d.symbols)}: {len(report.folds)} folds, train "
               f"{d.train_years:g}y / validation {d.validation_years:g}y / test {d.test_years:g}y, "
               f"embargo {d.embargo_bars} bars, objective {d.objective}, "
               f"{len(d.candidates())} variant(s)")
    typer.echo(f"{'FOLD':<5}{'TEST PERIOD':<25}{'CHOSEN':<28}{'TRAIN':>7}{'VALID':>7}{'TEST':>7}"
               f"{'B&H':>7}")

    def sr(x):
        return "n/a" if x is None or not math.isfinite(x) else f"{x:.2f}"

    for f in report.folds:
        chosen = ", ".join(f"{k}={v:g}" for k, v in f.chosen.items()) or "(defaults)"
        start_t, end_t = f.periods["test"]
        typer.echo(f"{f.n:<5}{start_t + '..' + end_t:<25}{chosen:<28}{sr(f.train_objective):>7}"
                   f"{sr(f.validation_objective):>7}{sr(f.test.get('sharpe')):>7}"
                   f"{sr(f.benchmark.get('sharpe')):>7}")
    o, b = report.oos, report.benchmark_oos
    typer.echo(f"out of sample, {o.get('years', 0):g} years stitched:{'strategy':>14}{'buy & hold':>14}")
    for label, key, kind in (("  total return", "total_return", "pct"), ("  CAGR", "cagr", "pct"),
                             ("  Sharpe", "sharpe", "num"), ("  Sortino", "sortino", "num"),
                             ("  max drawdown", "max_drawdown", "pct")):
        typer.echo(f"{label:<36}{_fmt(o.get(key), kind):>14}{_fmt(b.get(key), kind):>14}")
    dsr = report.deflated
    typer.echo(f"higher Sharpe than buy & hold? probability {sr(dsr.get('p'))}, after "
               f"{dsr.get('trials')} look(s) at these test periods {sr(dsr.get('p_deflated'))} "
               f"(Jobson-Korkie/Memmel test)")
    typer.echo(f"for information: information ratio {sr(dsr.get('information_ratio'))} "
               f"(edge in return), P(strategy Sharpe > 0) {sr(dsr.get('psr_vs_zero'))} "
               "(says nothing about an edge; buy and hold has one too)")
    typer.echo(f"out-of-sample trades: {report.oos_trades}")
    colour = typer.colors.GREEN if report.verdict == "validated" else typer.colors.YELLOW
    typer.secho(f"VERDICT: {report.verdict}", fg=colour, bold=True)
    for reason in report.reasons:
        typer.echo(f"  - {reason}")
    for note in report.notes:
        typer.echo(f"note: {note}")
    typer.echo(f"recorded as {report.experiment_id}" if report.experiment_id else "not recorded")


@app.command("experiments")
def experiments_command(limit: int = typer.Option(20, help="How many, newest first.")) -> None:
    """The experiment register: every walk-forward run and its verdict."""
    try:
        _, db, _ = _open()
        rows = db.query(
            "SELECT experiment_id, created_at, hypothesis_id, rule, symbols, status, verdict, "
            "json_extract(summary_json, '$.oos.sharpe') AS oos, "
            "json_extract(summary_json, '$.benchmark_oos.sharpe') AS bh, "
            "json_extract(summary_json, '$.deflated.p_deflated') AS p, "
            "json_extract(summary_json, '$.deflated.deflated_sharpe') AS old, "
            "json_extract(summary_json, '$.deflated.test') AS test "
            "FROM experiments ORDER BY created_at DESC LIMIT ?", (limit,))
    except QuantError as exc:
        _fail(exc)
        return
    if not rows:
        typer.echo("no experiment yet; run `taq walkforward`")
    for r in rows:
        if r["status"] != "complete":
            nums = "failed"
        else:
            # Runs before a7b03f7 used superseded tests (see docs/VALIDIERUNG.md);
            # they are shown as what they were, not re-labelled.
            if r["p"] is not None:
                test = f"P(higher Sharpe) {_fmt(r['p'], 'num')}"
            else:
                test = f"superseded test ({r['test'] or 'Sharpe vs zero'}) {_fmt(r['old'], 'num')}"
            nums = f"OOS Sharpe {_fmt(r['oos'], 'num')} vs B&H {_fmt(r['bh'], 'num')}, {test}"
        typer.echo(f"{r['experiment_id']}  {r['hypothesis_id'] or '-':<6} {r['rule']:<18} "
                   f"{r['symbols'][:30]:<31} {r['verdict'] or r['status']:<18} {nums}")


def _agent_graph_config(cfg) -> dict:
    """The agent framework's configuration from [agents]."""
    import copy

    from tradingagents.default_config import DEFAULT_CONFIG

    graph_config = copy.deepcopy(DEFAULT_CONFIG)
    graph_config.update(
        llm_provider=cfg.agents.llm_provider, quick_think_llm=cfg.agents.quick_model,
        deep_think_llm=cfg.agents.deep_model, max_debate_rounds=cfg.agents.max_debate_rounds,
        max_risk_discuss_rounds=cfg.agents.max_risk_discuss_rounds)
    return graph_config


@app.command("agent")
def agent_command(
    ticker: str = typer.Option(..., help="Instrument (must be in a snapshot for the risk gate)."),
    date: str = typer.Option(..., help="Analysis date YYYY-MM-DD; data is served as of it."),
    analysts: str = typer.Option("", help="Comma-separated (default: [agents].analysts)."),
    cache: str = typer.Option("", help="record | replay | off (default: [agents].llm_cache)."),
    yes: bool = typer.Option(False, "--yes", help="Confirm a run that calls the model (costs money)."),
    save: bool = typer.Option(True, help="Write the report tree."),
) -> None:
    """Run the multi-agent analysis; the deterministic risk gate checks its decision."""
    from tradingagents_quant.bridge import llm

    try:
        cfg, _, _ = _open()
        mode = cache or cfg.agents.llm_cache
        if mode not in ("record", "replay", "off"):
            raise ConfigError(f"--cache must be record, replay or off, not {mode!r}")
        if mode != "replay" and not yes:
            raise ConfigError(
                f"this run calls {cfg.agents.llm_provider} ({cfg.agents.quick_model} / "
                f"{cfg.agents.deep_model}) and costs money: up to {cfg.agents.max_tokens_per_run:,} "
                "tokens. Add --yes to run it, or --cache replay to repeat a recorded run for free.")
        notes = []
        if (cfg.agents.llm_provider == "google" and not os.environ.get("GOOGLE_API_KEY")
                and os.environ.get("GEMINI_API_KEY")):
            os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]  # this process only
            notes.append("GOOGLE_API_KEY taken from GEMINI_API_KEY for this run (not stored)")
        db_path = paths.db_path(paths.data_home())
        run_id = "ag-" + hashlib.sha256(f"{ticker}|{date}|{time.time_ns()}".encode()).hexdigest()[:10]
        cache_obj, budget = llm.install(db_path, mode=mode, run_id=run_id,
                                        max_run=cfg.agents.max_tokens_per_run,
                                        max_day=cfg.agents.max_tokens_per_day)
    except QuantError as exc:
        _fail(exc)
        return
    chosen = tuple(a.strip() for a in analysts.split(",") if a.strip()) or cfg.agents.analysts
    try:
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        graph = TradingAgentsGraph(selected_analysts=chosen, config=_agent_graph_config(cfg),
                                   callbacks=[budget])
        state, rating = graph.propagate(ticker.upper(), date)
        report_dir = graph.save_reports(state, ticker.upper()) if save else None
    except Exception as exc:  # a model, budget or replay stop ends the run, not the program
        _fail(f"the agent run stopped: {type(exc).__name__}: {exc}")
        return
    finally:
        llm.uninstall()
    verdict, order = state.get("risk_verdict") or {}, state.get("sim_order") or {}
    typer.echo(f"{PAPER_LABEL} — agent analysis, checked by the deterministic risk gate")
    typer.echo(f"{ticker.upper()} on {date}: agents' rating {rating}")
    colour = typer.colors.GREEN if verdict.get("status") == "approved" else typer.colors.YELLOW
    typer.secho(f"risk gate: {str(verdict.get('status', 'missing')).upper()}", fg=colour, bold=True)
    for reason in verdict.get("reasons", []):
        typer.echo(f"  - {reason}")
    if order:
        typer.echo("simulated order: " + ", ".join(f"{k} {v}" for k, v in order.items()
                                                   if k != "label"))
    typer.echo(f"model calls: {budget.calls} ({budget.cached_calls} from the cache), "
               f"{budget.run_tokens:,} tokens charged to the budget")
    if cache_obj is not None:
        typer.echo(f"cache ({mode}): {cache_obj.hits} hits, {cache_obj.misses} misses, "
                   f"{cache_obj.stored} new answers stored")
    if verdict.get("decision_id"):
        typer.echo(f"decision recorded as {verdict['decision_id']}")
    if report_dir:
        typer.echo(f"reports: {report_dir}")
    for note in notes:
        typer.echo(f"note: {note}")


paper_app = typer.Typer(no_args_is_help=True, help=f"Paper trading. {PAPER_LABEL}.")
app.add_typer(paper_app, name="paper")


def _paper_spec(strategy, symbol, param, sizing, fraction, stop, take, max_hold, signal_prices,
                benchmark):
    from tradingagents_quant.strategy.spec import ExitSpec, SizingSpec, StrategySpec

    stop_kind, stop_value = _parse_level(stop)
    take_kind, take_value = _parse_level(take)
    return StrategySpec(
        name=strategy, rule=strategy, params=_parse_params(param or []), symbols=symbol,
        signal_prices=signal_prices, sizing=SizingSpec(method=sizing, fraction=fraction),
        exits=ExitSpec(stop_loss=stop_kind, stop_value=stop_value, take_profit=take_kind,
                       take_value=take_value, max_holding_bars=max_hold or None),
        benchmark=benchmark)


@paper_app.command("open")
def paper_open(
    name: str = typer.Option(..., help="A name for the account."),
    strategy: str = typer.Option("trend_following", help="Rule."),
    symbol: str = typer.Option("SPY", help="Instrument(s), comma-separated."),
    param: Annotated[list[str] | None, typer.Option(help="Rule parameter name=value.")] = None,
    sizing: str = typer.Option("fixed_fraction"),
    fraction: float = typer.Option(1.0),
    stop: str = typer.Option("", help="percent:0.1 or atr:3"),
    take: str = typer.Option("", help="percent:0.2 or atr:6"),
    max_hold: int = typer.Option(0),
    signal_prices: str = typer.Option("total_return"),
    cash: float = typer.Option(0.0, help="Simulated starting cash (default: backtest.initial_cash)."),
    start: str = typer.Option("", help="First decision day (default: the latest trading day)."),
) -> None:
    """Open a paper account for one strategy version."""
    from tradingagents_quant.paper.account import open_account

    try:
        cfg, _, store = _open()
        spec = _paper_spec(strategy, symbol, param, sizing, fraction, stop, take, max_hold,
                           signal_prices, cfg.universe.benchmark)
        acc = open_account(store, name=name, spec=spec, config=cfg, initial_cash=cash or None,
                           start=start or None)
    except (QuantError, ValueError) as exc:
        _fail(exc)
        return
    typer.echo(f"{PAPER_LABEL}")
    typer.echo(f"opened {acc.name} ({acc.account_id}): {spec.name} on {','.join(acc.symbols)}, "
               f"simulated cash {acc.initial_cash:,.0f}, first decision at the close of "
               f"{acc.start_date}, strategy version {spec.version_hash()}")
    typer.echo("next: `taq ingest` for fresh data, then `taq paper run`")


@paper_app.command("run")
def paper_run(
    name: str = typer.Option("", help="One account (default: all)."),
    until: str = typer.Option("", help="Process up to this day only (default: latest)."),
) -> None:
    """Advance accounts over the trading days they have not seen yet."""
    from tradingagents_quant.paper.account import get_account, list_accounts
    from tradingagents_quant.paper.runner import run_account

    try:
        cfg, db, store = _open()
        accounts = [get_account(db, name)] if name else list_accounts(db)
        if not accounts:
            typer.echo("no paper account yet; `taq paper open --name ...`")
            return
        runs = [run_account(store, a, cfg, until=until or None) for a in accounts]
    except (QuantError, ValueError) as exc:
        _fail(exc)
        return
    typer.echo(PAPER_LABEL)
    for r in runs:
        span = f"{r.days[0]}..{r.days[-1]} ({len(r.days)} days)" if r.days else "no new day"
        typer.echo(f"{r.account.name}: {span}, status {r.status}"
                   + (f", equity {r.equity:,.2f}" if r.days else ""))
        for f in r.fills:
            typer.echo(f"  {f['date']} {f['side']:<4} {f['shares']:g} {f['symbol']} at "
                       f"{f['price']:.4f} ({f['reason']}), fee {f['commission']:.2f}")
        for x in r.rejections:
            typer.secho(f"  {x['date']} {x['symbol']}: {x['reason']} — {x['detail']}",
                        fg=typer.colors.YELLOW)
        if r.risk_decisions:
            typer.echo(f"  risk engine: {r.risk_decisions} decision(s), {r.risk_refusals} refused")
        for note in r.notes:
            typer.echo(f"  note: {note}")


@paper_app.command("status")
def paper_status(name: str = typer.Option("", help="One account (default: all).")) -> None:
    """Positions, cash and equity of paper accounts."""
    from tradingagents_quant.paper.account import get_account, list_accounts
    from tradingagents_quant.paper.runner import summary

    try:
        _, db, store = _open()
        accounts = [get_account(db, name)] if name else list_accounts(db)
    except QuantError as exc:
        _fail(exc)
        return
    typer.echo(PAPER_LABEL)
    if not accounts:
        typer.echo("no paper account yet; `taq paper open --name ...`")
    for a in accounts:
        s = summary(store, a)
        pnl = s["equity"] / a.initial_cash - 1.0
        peak = s.get("peak_equity") or s["equity"]
        typer.echo(f"{a.name} [{s['status']}] {a.spec.name} on {','.join(a.symbols)}, as of "
                   f"{s['as_of'] or '(not run yet)'}")
        typer.echo(f"  simulated equity {s['equity']:,.2f} ({pnl:+.2%} since start), cash "
                   f"{s['cash']:,.2f}, drawdown {1 - s['equity'] / peak:.2%}")
        for sym, p in s["positions"].items():
            stop = f", stop {p['stop']:.2f}" if p["stop"] else ""
            typer.echo(f"  {sym:<6} {p['shares']:>10g} sh  close {p['last_close']:.2f}  value "
                       f"{p['value']:,.0f}  weight {p['weight']:.1%}{stop}")
        for sym, o in s.get("pending", {}).items():
            typer.echo(f"  pending: {o['side']} {o['shares']:g} {sym} at the next open ({o['reason']})")


@paper_app.command("history")
def paper_history(name: str = typer.Option(..., help="Account."),
                  limit: int = typer.Option(30, help="Latest entries.")) -> None:
    """Fills, refusals and status changes of an account, from its ledger."""
    from tradingagents_quant.paper import ledger
    from tradingagents_quant.paper.account import get_account

    try:
        _, db, _ = _open()
        acc = get_account(db, name)
        rows = ledger.entries(db, acc.account_id,
                              ("opened", "fill", "rejection", "status", "rebase", "dividend"), limit)
    except QuantError as exc:
        _fail(exc)
        return
    typer.echo(PAPER_LABEL)
    for e in rows:
        if e["kind"] == "fill":
            text = f"{e['side']} {e['shares']:g} {e['symbol']} at {e['price']:.4f} ({e['reason']})"
        elif e["kind"] == "rejection":
            text = f"{e['symbol']}: {e['reason']} — {e['detail']}"
        elif e["kind"] == "dividend":
            text = f"dividends {e['amount']:,.2f}"
        elif e["kind"] == "opened":
            text = f"opened with {e['initial_cash']:,.0f} simulated cash"
        else:
            text = json.dumps({k: v for k, v in e.items()
                               if k not in ("seq", "recorded_at", "kind", "date")})
        typer.echo(f"{e['seq']:>5} {e['date'] or '':<10} {e['kind']:<9} {text}")


@paper_app.command("verify")
def paper_verify(name: str = typer.Option("", help="One account (default: all).")) -> None:
    """Check every ledger's hash chain."""
    from tradingagents_quant.paper import ledger
    from tradingagents_quant.paper.account import get_account, list_accounts

    try:
        _, db, _ = _open()
        accounts = [get_account(db, name)] if name else list_accounts(db)
    except QuantError as exc:
        _fail(exc)
        return
    broken = False
    for a in accounts:
        problems = ledger.verify(db, a.account_id)
        if problems:
            broken = True
            for p in problems:
                typer.secho(f"{a.name}: {p}", fg=typer.colors.RED)
        else:
            typer.echo(f"{a.name}: ledger intact")
    if broken:
        raise typer.Exit(code=1)


@paper_app.command("resume")
def paper_resume(name: str = typer.Option(..., help="Account."),
                 reason: str = typer.Option(..., help="Why it may trade again.")) -> None:
    """Lift a suspension after a review; the reason goes into the ledger."""
    from tradingagents_quant.paper.account import get_account
    from tradingagents_quant.paper.runner import resume

    try:
        _, db, store = _open()
        resume(store, get_account(db, name), reason=reason)
    except QuantError as exc:
        _fail(exc)
        return
    typer.echo(f"{name} resumed: {reason}")


@paper_app.command("list")
def paper_list() -> None:
    """All paper accounts."""
    from tradingagents_quant.paper.account import latest_state, list_accounts

    try:
        _, db, _ = _open()
        accounts = list_accounts(db)
    except QuantError as exc:
        _fail(exc)
        return
    for a in accounts:
        st = latest_state(db, a.account_id)
        typer.echo(f"{a.name:<20} {a.account_id}  {a.spec.name} on {','.join(a.symbols)}  "
                   f"{(st or {}).get('status', 'active')}  as of {(st or {}).get('as_of', '-')}")


@app.command("regime")
def regime_command(
    symbol: str = typer.Option("SPY", help="Symbol in the snapshot."),
    snapshot: str = typer.Option("", help="Snapshot id; default: the latest."),
    years: int = typer.Option(5, help="How many recent calendar years to break down."),
) -> None:
    """Current market regime of a symbol and its regime shares per year."""
    from tradingagents_quant import regime
    from tradingagents_quant.indicators import registry

    try:
        _, _, store = _open()
        snap = store.resolve(snapshot or None)
        bars = store.bars(snap.snapshot_id, snap.series_info(symbol.upper()).symbol)
    except QuantError as exc:
        _fail(exc)
        return
    labels = regime.classify(bars)
    last = labels.index[-1]
    typer.echo(f"{symbol.upper()} on {last.date()} (snapshot {snap.snapshot_id}): "
               f"trend {labels['trend'].iloc[-1]}, volatility {labels['volatility'].iloc[-1]}")
    shown = [(k, {}) for k, s in registry.REGISTRY.items() if s.shown_to_agents]
    values = registry.values_at(registry.compute_many(bars, shown), last)
    typer.echo("  " + ", ".join(f"{k} {v:.4g}" if v is not None else f"{k} n/a"
                                for k, v in values.items()))
    typer.echo(f"{'YEAR':<6}" + "".join(f"{t:>12}" for t in regime.TRENDS[:-1]))
    for year in sorted({d.year for d in labels.index})[-years:]:
        shares = labels.loc[str(year), "trend"].value_counts(normalize=True)
        typer.echo(f"{year:<6}" + "".join(f"{shares.get(t, 0):>12.0%}" for t in regime.TRENDS[:-1]))


def _loopback(host: str) -> bool:
    import ipaddress

    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@app.command("dashboard")
def dashboard_command(
    host: str = typer.Option("127.0.0.1", help="Address to listen on."),
    port: int = typer.Option(8000, help="Port."),
    allow_remote: bool = typer.Option(
        False, "--allow-remote", help="Listen on an address other than loopback. The dashboard "
                                      "has no login: anyone who reaches it sees everything."),
) -> None:
    """Serve the read-only research dashboard (needs the [dashboard] extra)."""
    if not _loopback(host) and not allow_remote:
        _fail(ConfigError(f"{host} is not a loopback address and the dashboard has no login; "
                          "pass --allow-remote to listen there anyway"))
        return
    try:
        import uvicorn

        from tradingagents_quant.api.app import create_app
    except ImportError:
        _fail(QuantError('the dashboard needs its extra: pip install -e ".[dashboard]"'))
        return
    from tradingagents_quant.store.db import Database

    try:
        db_file = paths.db_path(paths.data_home())
        if db_file.is_file():  # the views read the current schema; no file is created here
            with Database(db_file) as db:
                db.migrate()
        web = create_app()
    except QuantError as exc:
        _fail(exc)
        return
    typer.echo(PAPER_LABEL)
    typer.echo(f"dashboard: http://{host}:{port}/ (read-only, Ctrl+C stops it)")
    uvicorn.run(web, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    app()
