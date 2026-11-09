"""The Risk Gate: a deterministic node after the Portfolio Manager.

The agents argue in prose and end with a rating. Before anything comes of
it, this node checks — in plain code, without a model — that

1. there is research data for the instrument and the date (the system's own
   snapshot, point in time, not stale, without data errors);
2. the three decisions were structured answers, not a fallback to prose;
3. they agree: the Research Manager's recommendation, the Trader's action
   and the Portfolio Manager's rating point the same way;
4. the agents' numbers fit the market: the entry near the last close, the
   stop below the entry, the price target on the side the rating says;
5. the deterministic risk engine accepts the simulated order, and at what size.

Its verdict and the simulated order go into new state fields (``risk_verdict``,
``sim_order``, ``data_quality``, ``risk_gate_report``). It changes nothing the
agents wrote: ``final_rating`` and ``final_trade_decision`` stay as they are,
and a refusal acts where orders are placed, not by rewriting the rating. It
calls no model and no data vendor, and it never raises — any failure is a
refusal with its reason.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from tradingagents_quant import PAPER_LABEL
from tradingagents_quant.bridge import decisions as dec

GATE_VERSION = "v1"
REPORT_KEYS = ("market_report", "sentiment_report", "news_report", "fundamentals_report",
               "investment_plan", "trader_investment_plan", "final_trade_decision")


@dataclass
class GateResult:
    status: str  # approved | blocked
    reasons: list[str]
    checks: list[dict[str, Any]]
    sim_order: dict[str, Any]
    data_quality: dict[str, Any]
    decisions: dict[str, Any]
    snapshot_id: str | None = None
    notes: list[str] = field(default_factory=list)

    def verdict(self) -> dict[str, Any]:
        return {"status": self.status, "reasons": self.reasons, "checks": self.checks,
                "gate_version": GATE_VERSION, "snapshot_id": self.snapshot_id,
                "notes": self.notes, "label": PAPER_LABEL}

    def report(self, symbol: str, trade_date: str) -> str:
        lines = [f"**Status**: {self.status.upper()}", "",
                 f"Deterministic check of the agents' decision on {symbol} for {trade_date} "
                 f"(gate {GATE_VERSION}; no model involved). {PAPER_LABEL}.", ""]
        for c in self.checks:
            mark = "pass" if c["passed"] else "FAIL"
            lines.append(f"- {mark} — {c['name']}: {c['detail']}")
        if self.sim_order:
            o = self.sim_order
            lines += ["", "**Simulated order**: " + ", ".join(f"{k} {v}" for k, v in o.items()
                                                                if k != "label")]
        for n in self.notes:
            lines += ["", f"Note: {n}"]
        return "\n".join(lines)


class _Checks:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, name: str, passed: bool, detail: str, **extra: Any) -> bool:
        self.items.append({"name": name, "passed": bool(passed), "detail": detail, **extra})
        return passed

    @property
    def failed(self) -> list[str]:
        return [f"{c['name']}: {c['detail']}" for c in self.items if not c["passed"]]


def _blocked(checks: _Checks, decisions: dec.Decisions, quality: dict[str, Any],
             snapshot_id: str | None = None, notes: list[str] | None = None) -> GateResult:
    return GateResult("blocked", checks.failed, checks.items, {}, quality, decisions.to_dict(),
                      snapshot_id, notes or [])


def evaluate(state: Mapping[str, Any], *, environ: Mapping[str, str] | None = None) -> GateResult:
    """The gate's judgement of a finished agent state. Never raises."""
    checks = _Checks()
    decisions = dec.Decisions()
    try:
        return _evaluate(dict(state), environ if environ is not None else os.environ, checks)
    except Exception as exc:  # the gate must never end the agent run
        checks.add("gate", False, f"internal error, decision refused: {type(exc).__name__}: {exc}")
        return _blocked(checks, decisions, {})


def _evaluate(state: dict[str, Any], environ: Mapping[str, str], checks: _Checks) -> GateResult:
    from tradingagents_quant import paths
    from tradingagents_quant.config import load_config
    from tradingagents_quant.data.snapshot import SnapshotStore
    from tradingagents_quant.errors import QuantError
    from tradingagents_quant.store.db import Database

    decisions = dec.read(state)
    symbol = str(state.get("company_of_interest") or "").upper()
    trade_date = str(state.get("trade_date") or "")
    quality: dict[str, Any] = {"symbol": symbol, "trade_date": trade_date}

    try:
        config = load_config(environ=environ)
    except QuantError as exc:
        checks.add("configuration", False, str(exc))
        return _blocked(checks, decisions, quality)
    try:
        day = date.fromisoformat(trade_date)
    except ValueError:
        checks.add("data", False, f"trade date {trade_date!r} is not YYYY-MM-DD")
        return _blocked(checks, decisions, quality)

    # 1. The system's own data, point in time.
    db_file = paths.db_path(paths.data_home(environ))
    if not db_file.is_file():
        checks.add("data", False, "no research database (run `taq ingest`): nothing to check "
                                  "the agents' numbers against")
        return _blocked(checks, decisions, quality)
    db = Database(db_file, read_only=True)
    try:
        store = SnapshotStore(db)
        row = db.query_one(
            "SELECT s.snapshot_id FROM snapshot_series ss JOIN snapshots s "
            "ON s.snapshot_id = ss.snapshot_id WHERE ss.symbol = ? AND s.status <> 'building' "
            "ORDER BY s.created_at DESC LIMIT 1", (symbol,))
        if row is None:
            checks.add("data", False, f"{symbol} is in no snapshot (run `taq ingest --symbols "
                                      f"{symbol}`)")
            return _blocked(checks, decisions, quality)
        snapshot_id = row[0]
        info = store.get(snapshot_id).series_info(symbol)
        bars = store.bars(snapshot_id, symbol, end=trade_date)
        quality.update({"snapshot_id": snapshot_id, "series_quality": info.quality,
                        "source": store.get(snapshot_id).source_version})
        if info.quality == "error":
            checks.add("data", False, f"{symbol} has data errors in snapshot {snapshot_id}")
            return _blocked(checks, decisions, quality, snapshot_id)
        if bars.empty:
            checks.add("data", False, f"no bar of {symbol} on or before {trade_date}")
            return _blocked(checks, decisions, quality, snapshot_id)
        last_day = bars.index[-1].date()
        stale = (day - last_day).days
        last_close = float(bars["close"].iloc[-1])
        quality.update({"last_bar": last_day.isoformat(), "last_close": last_close,
                        "stale_days": stale})
        if not checks.add("data", stale <= config.data.max_stale_days,
                          f"last bar {last_day} ({stale} days before {trade_date}), "
                          f"close {last_close:.4f}, series quality {info.quality}"):
            return _blocked(checks, decisions, quality, snapshot_id)

        # 2. Structured answers, not prose.
        for key, ok in decisions.structured.items():
            checks.add("structured", ok, f"{dec.STAGE_NAMES[key]}: "
                       + ("structured answer" if ok else "no structured answer (prose fallback); "
                          "its figures cannot be checked"))
        rating = decisions.rating or "REVIEW"
        checks.add("rating", rating in dec.RATING_DIRECTION,
                   f"final rating {rating}" + ("" if rating in dec.RATING_DIRECTION
                                               else " — no position is taken on a review"))
        if checks.failed:
            return _blocked(checks, decisions, quality, snapshot_id)

        # 3. The three decisions point the same way.
        directions = {
            "Research Manager": dec.RATING_DIRECTION.get(decisions.recommendation or ""),
            "Trader": dec.ACTION_DIRECTION.get(decisions.action or ""),
            "Portfolio Manager": dec.RATING_DIRECTION.get(rating),
        }
        known = {k: v for k, v in directions.items() if v is not None}
        checks.add("consistency", len(known) == 3 and len(set(known.values())) == 1,
                   f"recommendation {decisions.recommendation}, action {decisions.action}, "
                   f"rating {rating}")
        direction = directions["Portfolio Manager"]

        # 4. The agents' numbers against the market.
        limit = config.agents.max_entry_deviation
        if decisions.entry_price is not None:
            dev = decisions.entry_price / last_close - 1.0
            checks.add("entry_price", abs(dev) <= limit,
                       f"entry {decisions.entry_price:g} is {dev:+.1%} from the last close "
                       f"(allowed ±{limit:.0%})")
        if decisions.stop_loss is not None and direction == 1:
            reference = decisions.entry_price or last_close
            checks.add("stop_loss", decisions.stop_loss < reference,
                       f"stop {decisions.stop_loss:g} against entry {reference:g}")
        if decisions.price_target is not None and direction != 0:
            right_side = (decisions.price_target > last_close if direction == 1
                          else decisions.price_target < last_close)
            checks.add("price_target", right_side,
                       f"target {decisions.price_target:g} against the last close "
                       f"{last_close:g} for a {rating} rating")
        if checks.failed:
            return _blocked(checks, decisions, quality, snapshot_id)

        # 5. The risk engine on a simulated order.
        order = _simulated_order(store, snapshot_id, symbol, bars, last_close, rating, direction,
                                 decisions, config, checks)
        if checks.failed:
            return _blocked(checks, decisions, quality, snapshot_id)
        return GateResult("approved", [], checks.items, order, quality, decisions.to_dict(),
                          snapshot_id)
    finally:
        db.close()


# Share of the largest permitted position each rating asks for.
RATING_WEIGHT = {"Buy": 1.0, "Overweight": 0.5}


def _simulated_order(store, snapshot_id, symbol, bars, last_close, rating, direction, decisions,
                     config, checks) -> dict[str, Any]:
    from tradingagents_quant.risk.engine import MarketHistory, RiskEngine
    from tradingagents_quant.sim.engine import Order, OrderContext
    from tradingagents_quant.sim.execution import CostModel
    from tradingagents_quant.sim.prices import market_frame
    from tradingagents_quant.sim.sizing import shares_for

    if direction == 0:
        checks.add("order", True, "Hold: no change to a position")
        return {"side": "none", "reason": "hold", "label": PAPER_LABEL}
    if direction == -1:
        checks.add("order", True, f"{rating}: reduce or close an existing position; selling "
                                  "is never held back by the risk engine")
        return {"side": "sell", "reason": rating.lower(), "applies_to": "an existing position",
                "label": PAPER_LABEL}

    costs = CostModel.from_settings(config.costs)
    equity = config.backtest.initial_cash
    market, _ = market_frame(bars, store.actions(snapshot_id, symbol))
    weight = RATING_WEIGHT[rating] * config.risk.max_position_weight
    wanted = shares_for(weight, equity, last_close, costs)
    day = bars.index[-1]
    engine = RiskEngine(config.risk, MarketHistory.from_markets({symbol: market}), costs)
    verdict = engine.evaluate(OrderContext(
        day, symbol, Order("buy", wanted, "entry", day.strftime("%Y-%m-%d"), math.nan, "agent"),
        last_close, equity, equity, equity, equity, {}, decisions.stop_loss))
    checks.add("risk_engine", verdict.accepted,
               f"{verdict.approved:g} of {wanted:g} shares approved"
               + (f" ({'; '.join(verdict.reasons)})" if verdict.reasons else ""),
               verdict=verdict.to_dict())
    if not verdict.accepted:
        return {}
    return {"side": "buy", "shares": verdict.approved, "reference_price": last_close,
            "weight": round(verdict.approved * last_close / equity, 6),
            "reference_equity": equity, "stop": decisions.stop_loss,
            "price_target": decisions.price_target, "fill": "next open",
            "label": PAPER_LABEL}


def reports_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in REPORT_KEYS:
        digest.update(f"{key}\n{state.get(key) or ''}\n".encode())
    return digest.hexdigest()


def record(state: Mapping[str, Any], result: GateResult, *,
           environ: Mapping[str, str] | None = None) -> str | None:
    """Store the decision when a research database exists. Returns the id, or
    None when there is nowhere to store it. Never raises."""
    try:
        from tradingagents_quant import paths
        from tradingagents_quant.config import load_config
        from tradingagents_quant.provenance import code_version
        from tradingagents_quant.store.db import Database, utc_now

        environ = environ if environ is not None else os.environ
        db_file = paths.db_path(paths.data_home(environ))
        if not db_file.is_file():
            return None
        now = utc_now()
        symbol = str(state.get("company_of_interest") or "").upper()
        trade_date = str(state.get("trade_date") or "")
        sha = reports_sha256(state)
        decision_id = "ad-" + hashlib.sha256(f"{now}|{symbol}|{sha}".encode()).hexdigest()[:12]
        fingerprint = load_config(environ=environ).fingerprint()
        db = Database(db_file)
        try:
            db.migrate()
            with db.transaction() as conn:
                conn.execute(
                    "INSERT INTO agent_decisions (decision_id, created_at, symbol, trade_date, "
                    "final_rating, gate_status, snapshot_id, reports_sha256, decisions_json, "
                    "verdict_json, sim_order_json, data_quality_json, config_fingerprint, "
                    "code_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (decision_id, now, symbol, trade_date, state.get("final_rating"), result.status,
                     result.snapshot_id, sha, json.dumps(result.decisions, default=str),
                     json.dumps(result.verdict(), default=str),
                     json.dumps(result.sim_order, default=str),
                     json.dumps(result.data_quality, default=str), fingerprint, code_version()))
        finally:
            db.close()
        return decision_id
    except Exception:
        return None


def create_risk_gate(environ: Mapping[str, str] | None = None):
    """The LangGraph node: reads the finished agent state, writes only its own keys."""

    def risk_gate(state: Mapping[str, Any]) -> dict[str, Any]:
        result = evaluate(state, environ=environ)
        decision_id = record(state, result, environ=environ)
        verdict = result.verdict()
        verdict["decision_id"] = decision_id
        symbol = str(state.get("company_of_interest") or "").upper()
        return {
            "risk_verdict": verdict,
            "sim_order": result.sim_order,
            "data_quality": result.data_quality,
            "risk_gate_report": result.report(symbol, str(state.get("trade_date") or "")),
        }

    return risk_gate
