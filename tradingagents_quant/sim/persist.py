"""Store backtest runs. Append-only: a stored result is never changed."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from tradingagents_quant.config import QuantConfig
from tradingagents_quant.provenance import code_version
from tradingagents_quant.sim.runner import BacktestRun
from tradingagents_quant.sim.types import PortfolioResult, SimulationResult
from tradingagents_quant.store.db import Database, utc_now
from tradingagents_quant.strategy.spec import StrategySpec

Result = SimulationResult | PortfolioResult


def _ensure_strategy(conn, spec: StrategySpec, now: str) -> str:
    version = spec.version_hash()
    conn.execute(
        "INSERT OR IGNORE INTO strategy_versions (version_hash, name, rule, spec_json, first_seen) "
        "VALUES (?, ?, ?, ?, ?)",
        (version, spec.name, spec.rule, json.dumps(spec.to_dict(), sort_keys=True), now),
    )
    return version


def _ensure_config(conn, config: QuantConfig, now: str) -> str:
    fingerprint = config.fingerprint()
    conn.execute(
        "INSERT OR IGNORE INTO config_versions (fingerprint, config_json, first_seen) VALUES (?, ?, ?)",
        (fingerprint, json.dumps(config.to_dict(), sort_keys=True), now),
    )
    return fingerprint


def _run_id(*parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:10]
    return f"bt-{parts[0].replace('-', '').replace(':', '').split('.')[0]}-{digest}"


def _insert(conn, *, run_id: str, now: str, kind: str, version: str, run: BacktestRun,
            result: Result, metrics: dict[str, Any], fingerprint: str,
            parent: str | None, experiment_id: str | None, regimes: dict | None,
            notes: list[str] | None, symbol: str | None = None, sha: str | None = None) -> None:
    conn.execute(
        "INSERT INTO backtest_runs (run_id, created_at, kind, strategy_version, symbol, snapshot_id, "
        "series_sha256, start_date, end_date, config_fingerprint, code_version, parent_run_id, "
        "experiment_id, metrics_json, regimes_json, notes_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, now, kind, version, symbol or run.symbol, run.snapshot_id,
         sha or run.series_sha256,
         metrics["start"], metrics["end"], fingerprint, code_version(), parent, experiment_id,
         json.dumps(metrics, sort_keys=True), json.dumps(regimes, sort_keys=True) if regimes else None,
         json.dumps(notes) if notes else None),
    )
    rows = []
    for n, t in enumerate(result.trades):
        held, peak = 0.0, 0.0
        for f in t.fills:
            held += f.shares if f.side == "buy" else -f.shares
            peak = max(peak, held)
        rows.append((run_id, n, t.entry_date.strftime("%Y-%m-%d"),
                     t.exit_date.strftime("%Y-%m-%d") if t.exit_date is not None else None,
                     t.entry_price, t.exit_price, peak, t.pnl, t.return_pct, t.holding_bars,
                     t.exit_reason, t.entry_regime, t.dividends))
    conn.executemany(
        "INSERT INTO backtest_trades (run_id, n, entry_date, exit_date, entry_price, exit_price, "
        "max_shares, pnl, return_pct, holding_bars, exit_reason, entry_regime, dividends) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    invested = (result.positions.sum(axis=1) if isinstance(result, PortfolioResult)
                else result.shares)  # shares held (summed over instruments for a portfolio)
    series = [[d.strftime("%Y-%m-%d"), float(e), float(c), float(s)]
              for d, e, c, s in zip(result.equity.index, result.equity, result.cash, invested,
                                    strict=True)]
    conn.execute("INSERT INTO backtest_equity (run_id, series_json) VALUES (?, ?)",
                 (run_id, json.dumps(series)))


def save(db: Database, run: BacktestRun, *, kind: str = "backtest",
         experiment_id: str | None = None) -> str:
    """Store the run (and its benchmark, linked as a child) in one transaction."""
    now = utc_now()
    with db.transaction() as conn:
        version = _ensure_strategy(conn, run.spec, now)
        fingerprint = _ensure_config(conn, run.config, now)
        run_id = _run_id(now, version, run.symbol, run.snapshot_id, kind)
        _insert(conn, run_id=run_id, now=now, kind=kind, version=version, run=run,
                result=run.result, metrics={**run.metrics, "comparison": run.comparison},
                fingerprint=fingerprint, parent=None, experiment_id=experiment_id,
                regimes=run.regimes, notes=run.notes + run.data_warnings)
        for b in run.benchmarks:
            b_version = _ensure_strategy(conn, b.spec, now)
            _insert(conn, run_id=_run_id(now, b_version, ",".join(b.symbols), run_id, b.name),
                    now=now, kind="benchmark", version=b_version, run=run, result=b.result,
                    metrics={**b.metrics, "benchmark": b.name}, fingerprint=fingerprint,
                    parent=run_id, experiment_id=experiment_id, regimes=None, notes=None,
                    symbol=",".join(b.symbols), sha=b.series_sha256)
    return run_id
