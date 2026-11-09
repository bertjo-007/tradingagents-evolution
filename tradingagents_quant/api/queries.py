"""What the dashboard shows, read from the database and nothing else.

Every function returns plain data built from stored rows or from computations
on stored snapshots. Where nothing is stored, the answer is empty and says so;
no value is ever filled in for display.
"""

from __future__ import annotations

import json
import math
import os
import tomllib
from pathlib import Path
from typing import Any

from tradingagents_quant import PAPER_LABEL, __version__, regime
from tradingagents_quant.config import QuantConfig
from tradingagents_quant.data.snapshot import SnapshotStore
from tradingagents_quant.errors import QuantError, SnapshotError
from tradingagents_quant.indicators import registry
from tradingagents_quant.paper import ledger
from tradingagents_quant.store.db import Database
from tradingagents_quant.store.migrations import SCHEMA_VERSION

REPO = Path(__file__).resolve().parents[2]
MAX_POINTS = 1500
KEY_VARIABLES = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                 "ALPHA_VANTAGE_API_KEY", "FRED_API_KEY")


def _finite(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(v) for v in value]
    return value


def _loads(text: str | None) -> Any:
    return json.loads(text) if text else None


def thin(points: list[list[Any]], limit: int = MAX_POINTS) -> list[list[Any]]:
    """At most ``limit`` points, evenly spaced, the last always kept."""
    if len(points) <= limit:
        return points
    step = math.ceil(len(points) / limit)
    out = points[::step]
    if out[-1] is not points[-1]:
        out.append(points[-1])
    return out


def _has(db: Database, table: str) -> bool:
    return db.query_one("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                        (table,)) is not None


# -- overview ----------------------------------------------------------------------


def status(db: Database | None, cfg: QuantConfig, home: Path) -> dict[str, Any]:
    from tradingagents_quant.provenance import code_version

    out: dict[str, Any] = {
        "label": PAPER_LABEL, "version": __version__, "code_version": code_version(),
        "config_fingerprint": cfg.fingerprint(), "modes": cfg.modes.model_dump(),
        "data_dir": str(home), "database": db is not None, "snapshot": None,
        "counts": {}, "market": None, "paper": [],
        "research_tasks": [], "research_tasks_note": "no scheduled research loop runs yet",
    }
    if db is None:
        return out
    out["schema_version"] = db.schema_version
    out["schema_current"] = db.schema_version == SCHEMA_VERSION
    store = SnapshotStore(db)
    snap = store.latest()
    if snap is not None:
        out["snapshot"] = snapshot_summary(store, snap.snapshot_id)
        out["market"] = market(store, snap.snapshot_id, cfg.universe.benchmark)
    counts = {}
    for name, sql in (
        ("backtests", "SELECT COUNT(*) FROM backtest_runs WHERE kind = 'backtest'"),
        ("experiments", "SELECT COUNT(*) FROM experiments WHERE status = 'complete'"),
        ("validated", "SELECT COUNT(*) FROM experiments WHERE verdict = 'validated'"),
        ("paper_accounts", "SELECT COUNT(*) FROM paper_accounts"),
        ("agent_decisions", "SELECT COUNT(*) FROM agent_decisions"),
    ):
        try:
            counts[name] = db.query_one(sql)[0]
        except Exception:
            counts[name] = None
    out["counts"] = counts
    out["paper"] = [{k: a[k] for k in ("name", "status", "equity", "initial_cash", "drawdown",
                                        "as_of")} for a in paper_accounts(db, store)]
    return _finite(out)


def snapshot_summary(store: SnapshotStore, snapshot_id: str) -> dict[str, Any]:
    info = store.get(snapshot_id)
    quality: dict[str, int] = {}
    for s in info.series:
        quality[s.quality] = quality.get(s.quality, 0) + 1
    return {"snapshot_id": info.snapshot_id, "created_at": info.created_at,
            "source": info.source_version, "status": info.status,
            "series": [{"symbol": s.symbol, "first": s.first_date, "last": s.last_date,
                        "bars": s.n_bars, "quality": s.quality} for s in info.series],
            "bars": sum(s.n_bars for s in info.series), "quality": quality,
            "failures": [f.__dict__ for f in store.failures(snapshot_id)]}


def market(store: SnapshotStore, snapshot_id: str, symbol: str) -> dict[str, Any] | None:
    """Regime and indicator values of one instrument at its last bar."""
    try:
        bars = store.bars(snapshot_id, store.get(snapshot_id).series_info(symbol).symbol)
    except SnapshotError:
        return None
    if bars.empty:
        return None
    labels = regime.classify(bars)
    shown = [("sma", {"n": 50}), ("sma", {"n": 200}), ("ema", {"n": 10}), ("rsi", {}),
             ("macd", {}), ("boll", {}), ("atr", {}), ("hv", {}), ("dmi", {})]
    values = registry.values_at(registry.compute_many(bars, shown), bars.index[-1])
    years = {}
    for year in sorted({d.year for d in labels.index})[-6:]:
        shares = labels.loc[str(year), "trend"].value_counts(normalize=True)
        years[str(year)] = {k: float(v) for k, v in shares.items()}
    return _finite({"symbol": symbol, "as_of": bars.index[-1].strftime("%Y-%m-%d"),
                    "close": float(bars["close"].iloc[-1]), "trend": labels["trend"].iloc[-1],
                    "volatility": labels["volatility"].iloc[-1], "indicators": values,
                    "trend_by_year": years})


def series(store: SnapshotStore, snapshot_id: str, symbol: str) -> dict[str, Any]:
    bars = store.bars(snapshot_id, symbol)
    points = [[d.strftime("%Y-%m-%d"), float(c)] for d, c in zip(bars.index, bars["close"],
                                                                 strict=True)]
    return {"symbol": symbol, "snapshot_id": snapshot_id, "points": thin(points)}


# -- strategies and backtests ----------------------------------------------------------


def rules() -> list[dict[str, Any]]:
    from tradingagents_quant.strategy.library import RULES

    return [{"rule": r.key, "family": r.family, "defaults": dict(r.defaults),
             "bounds": {k: list(v) for k, v in r.bounds.items()},
             "entry": r.entry.format(**r.defaults), "exit": r.exit.format(**r.defaults),
             "required": list(r.required), "holding": r.holding} for r in RULES.values()]


def strategies(db: Database) -> list[dict[str, Any]]:
    rows = db.query(
        "SELECT v.version_hash, v.name, v.rule, v.spec_json, v.first_seen, "
        "(SELECT COUNT(*) FROM backtest_runs b WHERE b.strategy_version = v.version_hash "
        " AND b.kind = 'backtest') AS backtests, "
        "(SELECT COUNT(*) FROM experiment_trials t WHERE t.strategy_version = v.version_hash) "
        " AS trials, "
        "(SELECT COUNT(*) FROM paper_accounts p WHERE p.strategy_version = v.version_hash) "
        " AS paper FROM strategy_versions v ORDER BY v.first_seen DESC")
    return [{"version": r["version_hash"], "name": r["name"], "rule": r["rule"],
             "spec": _loads(r["spec_json"]), "first_seen": r["first_seen"],
             "backtests": r["backtests"], "trials": r["trials"], "paper_accounts": r["paper"]}
            for r in rows]


_HEADLINE = ("total_return", "cagr", "sharpe", "sortino", "max_drawdown", "trades", "exposure",
             "total_costs", "start", "end", "bars")


def backtests(db: Database, limit: int = 200) -> list[dict[str, Any]]:
    rows = db.query(
        "SELECT b.run_id, b.created_at, b.symbol, b.start_date, b.end_date, b.metrics_json, "
        "v.name, v.rule, b.strategy_version FROM backtest_runs b "
        "JOIN strategy_versions v ON v.version_hash = b.strategy_version "
        "WHERE b.kind = 'backtest' ORDER BY b.created_at DESC LIMIT ?", (limit,))
    out = []
    for r in rows:
        m = _loads(r["metrics_json"]) or {}
        out.append({"run_id": r["run_id"], "created_at": r["created_at"], "symbol": r["symbol"],
                    "strategy": r["name"], "rule": r["rule"], "version": r["strategy_version"],
                    "metrics": {k: m.get(k) for k in _HEADLINE},
                    "vs_benchmark": (m.get("comparison") or {})})
    return _finite(out)


def backtest(db: Database, run_id: str) -> dict[str, Any] | None:
    row = db.query_one(
        "SELECT b.*, v.name, v.spec_json FROM backtest_runs b "
        "JOIN strategy_versions v ON v.version_hash = b.strategy_version WHERE b.run_id = ?",
        (run_id,))
    if row is None:
        return None

    def equity(rid: str) -> list[list[Any]]:
        e = db.query_one("SELECT series_json FROM backtest_equity WHERE run_id = ?", (rid,))
        return thin([p[:2] for p in json.loads(e[0])]) if e else []

    trades = [dict(t) for t in db.query(
        "SELECT * FROM backtest_trades WHERE run_id = ? ORDER BY n", (run_id,))]
    children = db.query(
        "SELECT run_id, symbol, metrics_json FROM backtest_runs WHERE parent_run_id = ? "
        "ORDER BY run_id", (run_id,))
    benchmarks = []
    for c in children:
        m = _loads(c["metrics_json"]) or {}
        benchmarks.append({"run_id": c["run_id"], "name": m.get("benchmark", "buy_and_hold"),
                           "symbol": c["symbol"], "metrics": {k: m.get(k) for k in _HEADLINE},
                           "equity": equity(c["run_id"])})
    return _finite({
        "run_id": run_id, "created_at": row["created_at"], "symbol": row["symbol"],
        "strategy": row["name"], "spec": _loads(row["spec_json"]),
        "version": row["strategy_version"], "snapshot_id": row["snapshot_id"],
        "series_sha256": row["series_sha256"], "config_fingerprint": row["config_fingerprint"],
        "code_version": row["code_version"], "metrics": _loads(row["metrics_json"]),
        "regimes": _loads(row["regimes_json"]), "notes": _loads(row["notes_json"]) or [],
        "trades": trades, "equity": equity(run_id), "benchmarks": benchmarks,
        "label": PAPER_LABEL,
    })


def experiments(db: Database) -> list[dict[str, Any]]:
    rows = db.query("SELECT * FROM experiments ORDER BY created_at DESC")
    out = []
    for r in rows:
        s = _loads(r["summary_json"]) or {}
        d = s.get("deflated") or {}
        out.append({
            "experiment_id": r["experiment_id"], "created_at": r["created_at"],
            "hypothesis": r["hypothesis_id"], "rule": r["rule"], "symbols": r["symbols"],
            "status": r["status"], "verdict": r["verdict"], "error": r["error"],
            "oos_sharpe": (s.get("oos") or {}).get("sharpe"),
            "benchmark_sharpe": (s.get("benchmark_oos") or {}).get("sharpe"),
            "p_higher_sharpe": d.get("p_deflated"), "test": d.get("test") or "Sharpe vs zero",
            "superseded_test": d.get("p_deflated") is None,
            "oos_trades": s.get("oos_trades"), "reasons": s.get("reasons") or [],
        })
    return _finite(out)


def experiment(db: Database, experiment_id: str) -> dict[str, Any] | None:
    r = db.query_one("SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,))
    if r is None:
        return None
    trials = db.query("SELECT segment, COUNT(*) AS n FROM experiment_trials WHERE "
                      "experiment_id = ? GROUP BY segment", (experiment_id,))
    return _finite({"experiment_id": experiment_id, "created_at": r["created_at"],
                    "verdict": r["verdict"], "status": r["status"],
                    "design": _loads(r["design_json"]), "summary": _loads(r["summary_json"]),
                    "trials": {t["segment"]: t["n"] for t in trials},
                    "hypothesis_sha256": r["hypothesis_sha256"]})


def hypotheses(db: Database | None) -> list[dict[str, Any]]:
    out = []
    for path in sorted((REPO / "docs" / "hypothesen").glob("H-*.toml")):
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        head, design = raw.get("hypothesis", {}), raw.get("design", {})
        latest = None
        if db is not None:
            row = db.query_one(
                "SELECT experiment_id, verdict, summary_json FROM experiments WHERE "
                "hypothesis_id = ? AND status = 'complete' ORDER BY created_at DESC LIMIT 1",
                (head.get("id"),))
            if row is not None:
                s = _loads(row["summary_json"]) or {}
                latest = {"experiment_id": row["experiment_id"], "verdict": row["verdict"],
                          "p_higher_sharpe": (s.get("deflated") or {}).get("p_deflated"),
                          "oos_sharpe": (s.get("oos") or {}).get("sharpe"),
                          "benchmark_sharpe": (s.get("benchmark_oos") or {}).get("sharpe")}
        out.append({"id": head.get("id"), "statement": head.get("statement"),
                    "created": head.get("created"), "rule": design.get("rule"),
                    "symbols": design.get("symbols"), "grid": design.get("grid"),
                    "criteria": raw.get("criteria", {}), "latest": latest})
    return _finite(out)


# -- paper trading and risk --------------------------------------------------------------


def paper_accounts(db: Database, store: SnapshotStore) -> list[dict[str, Any]]:
    from tradingagents_quant.paper.account import list_accounts
    from tradingagents_quant.paper.runner import summary

    out = []
    for a in list_accounts(db):
        s = summary(store, a)
        peak = s.get("peak_equity") or s["equity"]
        out.append({"name": a.name, "account_id": a.account_id, "strategy": a.spec.name,
                    "version": a.spec.version_hash(), "symbols": list(a.symbols),
                    "status": s["status"], "as_of": s["as_of"], "equity": s["equity"],
                    "cash": s["cash"], "initial_cash": a.initial_cash,
                    "drawdown": 1.0 - s["equity"] / peak if peak else None,
                    "positions": s["positions"], "pending": s.get("pending", {}),
                    "label": PAPER_LABEL})
    return _finite(out)


def paper_account(db: Database, store: SnapshotStore, name: str) -> dict[str, Any] | None:
    from tradingagents_quant.paper.account import PaperError, get_account

    try:
        a = get_account(db, name)
    except PaperError:
        return None
    base = next(x for x in paper_accounts(db, store) if x["name"] == name)
    days = ledger.entries(db, a.account_id, ("day",))
    equity = thin([[e["date"], e["equity"]] for e in days])
    events = ledger.entries(db, a.account_id, ("fill", "rejection", "status", "rebase",
                                               "dividend", "opened"))
    risk = ledger.entries(db, a.account_id, ("risk",), limit=200)
    return _finite({**base, "equity_curve": equity, "events": events[-300:],
                    "risk_decisions": risk, "ledger_intact": ledger.verify(db, a.account_id) == [],
                    "spec": a.spec.to_dict()})


def risk(db: Database | None, cfg: QuantConfig) -> dict[str, Any]:
    out: dict[str, Any] = {"limits": cfg.risk.model_dump(), "refusals": [], "accounts": [],
                           "gate_refusals": []}
    if db is None:
        return out
    store = SnapshotStore(db)
    for a in paper_accounts(db, store):
        weights = [p["weight"] for p in a["positions"].values() if p.get("weight") is not None]
        gross = sum(p["value"] for p in a["positions"].values() if p.get("value") is not None)
        out["accounts"].append({"name": a["name"], "status": a["status"],
                                "drawdown": a["drawdown"],
                                "gross_exposure": gross / a["equity"] if a["equity"] else None,
                                "largest_position": max(weights) if weights else 0.0,
                                "positions": len(a["positions"])})
    rows = db.query("SELECT p.name, l.date, l.payload_json FROM paper_ledger l "
                    "JOIN paper_accounts p ON p.account_id = l.account_id WHERE l.kind = 'risk' "
                    "AND json_extract(l.payload_json, '$.accepted') = 0 "
                    "ORDER BY l.date DESC, l.seq DESC LIMIT 100")
    out["refusals"] = [{"account": r["name"], "date": r["date"], **json.loads(r["payload_json"])}
                       for r in rows]
    gate = db.query("SELECT decision_id, symbol, trade_date, final_rating, verdict_json "
                    "FROM agent_decisions WHERE gate_status = 'blocked' "
                    "ORDER BY created_at DESC LIMIT 100")
    out["gate_refusals"] = [{"decision_id": r["decision_id"], "symbol": r["symbol"],
                             "date": r["trade_date"], "rating": r["final_rating"],
                             "reasons": (_loads(r["verdict_json"]) or {}).get("reasons", [])}
                            for r in gate]
    return _finite(out)


# -- agents and settings -----------------------------------------------------------------


def agents(db: Database | None, cfg: QuantConfig) -> dict[str, Any]:
    out: dict[str, Any] = {"settings": cfg.agents.model_dump(), "decisions": [], "usage": [],
                           "cache_entries": 0}
    if db is None:
        return out
    rows = db.query("SELECT * FROM agent_decisions ORDER BY created_at DESC LIMIT 100")
    out["decisions"] = [{"decision_id": r["decision_id"], "created_at": r["created_at"],
                         "symbol": r["symbol"], "trade_date": r["trade_date"],
                         "rating": r["final_rating"], "gate": r["gate_status"],
                         "reasons": (_loads(r["verdict_json"]) or {}).get("reasons", []),
                         "decisions": _loads(r["decisions_json"]),
                         "sim_order": _loads(r["sim_order_json"])} for r in rows]
    usage = db.query("SELECT day, COUNT(*) AS calls, SUM(cached) AS cached, "
                     "SUM(CASE WHEN cached = 0 THEN input_tokens + output_tokens ELSE 0 END) "
                     "AS tokens FROM llm_calls GROUP BY day ORDER BY day DESC LIMIT 60")
    out["usage"] = [dict(u) for u in usage]
    out["cache_entries"] = db.query_one("SELECT COUNT(*) FROM llm_cache")[0]
    return _finite(out)


def agent_decision(db: Database, decision_id: str) -> dict[str, Any] | None:
    r = db.query_one("SELECT * FROM agent_decisions WHERE decision_id = ?", (decision_id,))
    if r is None:
        return None
    reports = {x["section"]: x["text"] for x in db.query(
        "SELECT section, text FROM agent_reports WHERE decision_id = ?", (decision_id,))} \
        if _has(db, "agent_reports") else {}
    return _finite({"decision_id": decision_id, "created_at": r["created_at"],
                    "symbol": r["symbol"], "trade_date": r["trade_date"],
                    "rating": r["final_rating"], "gate": r["gate_status"],
                    "verdict": _loads(r["verdict_json"]), "sim_order": _loads(r["sim_order_json"]),
                    "data_quality": _loads(r["data_quality_json"]),
                    "decisions": _loads(r["decisions_json"]), "reports": reports,
                    "reports_sha256": r["reports_sha256"]})


def settings(cfg: QuantConfig, home: Path, environ: dict[str, str] | None = None) -> dict[str, Any]:
    """The configuration, and for each credential only whether it is set."""
    environ = dict(os.environ if environ is None else environ)
    return {"config": cfg.to_dict(), "fingerprint": cfg.fingerprint(), "data_dir": str(home),
            "credentials": {name: bool(environ.get(name)) for name in KEY_VARIABLES},
            "data_provider": {"prices": "Yahoo Finance via snapshots (taq ingest)",
                              "agents": "the agent framework's vendor router "
                                        "(tradingagents/default_config.py data_vendors)"},
            "research_frequency": "manual (`taq ingest`, `taq paper run`)",
            "label": PAPER_LABEL}


def open_store(home: Path) -> tuple[Database | None, SnapshotStore | None]:
    from tradingagents_quant import paths

    path = paths.db_path(home)
    if not path.is_file():
        return None, None
    try:
        db = Database(path, read_only=True)
    except Exception as exc:  # pragma: no cover - a locked or damaged file
        raise QuantError(f"cannot open {path}: {exc}") from exc
    return db, SnapshotStore(db)
