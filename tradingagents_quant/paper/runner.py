"""Step a paper account over the trading days it has not seen yet.

The same ``Book`` the backtest steps, with the same targets
(``sim.runner.portfolio_targets``), the same costs and exits, and the risk
engine as the filter on every buy. ``tests/quant/test_paper.py`` replays a
history day by day through this runner and requires the very fills a backtest
over the same days makes.

Status:

- ``active`` — trading.
- ``suspended`` — the drawdown reached ``risk.max_drawdown``. Exits, stops
  and targets keep working; no new buy is placed until ``resume`` (a person's
  decision, recorded in the ledger).
- ``halted`` — the data no longer agrees with the account (a price changed
  without a split to explain it). Nothing is processed until the cause is
  understood: computing on with a wrong basis would corrupt the record.

A split after the last run (the provider has since divided the history by
its factor) is not an inconsistency: the account's shares, prices and levels
are converted to the new basis, and the ledger says so.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from tradingagents_quant import PAPER_LABEL
from tradingagents_quant.config import QuantConfig
from tradingagents_quant.data.snapshot import SnapshotStore
from tradingagents_quant.paper import ledger
from tradingagents_quant.paper.account import PaperAccount, PaperError, latest_state
from tradingagents_quant.risk.engine import MarketHistory, RiskEngine
from tradingagents_quant.sim.engine import (
    Book,
    FilterDecision,
    Order,
    OrderContext,
    bars_for,
    portfolio_weights,
)
from tradingagents_quant.sim.execution import CostModel
from tradingagents_quant.sim.runner import portfolio_targets, prepare
from tradingagents_quant.store.db import utc_now


@dataclass
class PaperRun:
    account: PaperAccount
    status: str
    days: list[str] = field(default_factory=list)
    fills: list[dict[str, Any]] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    risk_decisions: int = 0
    risk_refusals: int = 0
    equity: float = math.nan
    cash: float = math.nan
    notes: list[str] = field(default_factory=list)
    label: str = PAPER_LABEL


def _rebase(book: Book, markets: dict[str, pd.DataFrame], actions: dict[str, pd.DataFrame],
            as_of: str) -> tuple[list[dict[str, Any]], str | None]:
    """Bring held positions onto the basis of the new data. Returns the
    rebase events and, if a change cannot be explained, why the account
    must halt."""
    events = []
    day = pd.Timestamp(as_of)
    for s, p in book.pos.items():
        if p.shares <= 0:
            continue
        m = markets[s]
        if day not in m.index:
            return events, f"{s}: the new data has no bar on {as_of}, the last day processed"
        ratio = p.last_close / float(m.at[day, "close"])
        if abs(ratio - 1.0) <= 1e-9:
            continue
        acts = actions[s]
        splits = acts[(acts["kind"] == "split") & (acts["date"] > as_of)]["value"]
        factor = float(splits.prod()) if len(splits) else 1.0
        if abs(ratio / factor - 1.0) > 1e-6:
            return events, (f"{s}: the close of {as_of} changed by a factor of {ratio:.6f}, "
                            f"which no split explains (splits since: {factor:g})")
        p.shares *= ratio
        p.last_close /= ratio
        p.stop = p.stop / ratio if p.stop is not None else None
        p.take = p.take / ratio if p.take is not None else None
        if p.pending is not None:
            o = p.pending
            p.pending = Order(o.side, o.shares * ratio, o.reason, o.decided, o.atr / ratio, o.regime)
        if p.trade is not None:
            p.trade.entry_price /= ratio
        events.append({"symbol": s, "factor": ratio, "reason": "split after the last run",
                       "shares_now": p.shares})
    return events, None


def run_account(
    store: SnapshotStore, account: PaperAccount, config: QuantConfig, *,
    snapshot_id: str | None = None, until: str | None = None,
) -> PaperRun:
    if not config.modes.paper_trading_mode:
        raise PaperError("paper_trading_mode is off in the configuration")
    db = store.db
    latest = latest_state(db, account.account_id)
    status = latest["status"] if latest else "active"
    run = PaperRun(account, status)
    if status == "halted":
        run.notes.append("account halted; see its ledger for the reason")
        return run

    spec = account.spec
    symbols = tuple(sorted(account.symbols))
    prep = prepare(store, snapshot_id, account.symbols, config, end=until)
    markets = prep.markets
    actions = {s: store.actions(prep.snapshot_id, s) for s in symbols}
    calendar = prep.calendar
    days = (calendar[calendar >= pd.Timestamp(account.start_date)] if latest is None
            else calendar[calendar > pd.Timestamp(latest["as_of"])])
    if not len(days):
        run.notes.append("no new trading day since the last run")
        return run

    costs = CostModel.from_settings(config.costs)
    exits = dict.fromkeys(symbols, spec.exits)
    band = config.backtest.rebalance_band
    if latest is None:
        book = Book(symbols, costs, exits, account.initial_cash, band)
    else:
        book = Book.from_state(latest["state"], symbols=symbols, costs=costs, exits=exits,
                               rebalance_band=band, reinvest_dividends=False)
    events: list[tuple[str, str | None, dict[str, Any]]] = []

    if latest is not None:
        rebased, problem = _rebase(book, markets, actions, latest["as_of"])
        if problem:
            status = "halted"
            events.append(("status", latest["as_of"], {"status": status, "reason": problem}))
            _save(store, account, prep.snapshot_id, latest["as_of"], status,
                  latest["state"], events)
            run.status = status
            run.notes.append(problem)
            return run
        events += [("rebase", latest["as_of"], e) for e in rebased]

    targets, labels = portfolio_targets(spec, markets)
    weights = portfolio_weights(pd.DataFrame({s: targets[s].reindex(calendar) for s in symbols}))
    bars = bars_for(markets, symbols, days, weights, exits, labels)
    risk = RiskEngine(config.risk, MarketHistory.from_markets(markets), costs).as_filter()
    current = {"status": status}

    def guarded(ctx: OrderContext) -> FilterDecision:
        if current["status"] != "active":
            return FilterDecision(0.0, (f"account {current['status']}: no new positions",))
        return risk(ctx)

    book.order_filter = guarded
    for day, bar in zip(days, bars, strict=True):
        date = day.strftime("%Y-%m-%d")
        n_fills, n_rej, n_risk, div0 = (len(book.fills), len(book.rejections),
                                        len(book.filter_records), book.dividends)
        book.step(day, bar)
        for f in book.fills[n_fills:]:
            payload = {**asdict(f), "date": date}
            events.append(("fill", date, payload))
            run.fills.append(payload)
        for r in book.rejections[n_rej:]:
            payload = {**asdict(r), "date": date}
            events.append(("rejection", date, payload))
            run.rejections.append(payload)
        for _, _, verdict in book.filter_records[n_risk:]:
            run.risk_decisions += 1
            if verdict is not None:
                run.risk_refusals += int(not verdict.accepted)
                events.append(("risk", date, verdict.to_dict()))
        if book.dividends > div0:
            events.append(("dividend", date, {"amount": book.dividends - div0}))
        drawdown = 1.0 - book.equity / book.peak_equity if book.peak_equity > 0 else 0.0
        events.append(("day", date, {
            "label": PAPER_LABEL, "equity": book.equity, "cash": book.cash, "drawdown": drawdown,
            "positions": {s: p.shares for s, p in book.pos.items() if p.shares > 0},
        }))
        if current["status"] == "active" and drawdown >= config.risk.max_drawdown:
            current["status"] = "suspended"
            events.append(("status", date, {
                "status": "suspended",
                "reason": f"drawdown {drawdown:.1%} reached the limit "
                          f"{config.risk.max_drawdown:.0%}; exits continue, no new positions "
                          "until resumed"}))
        run.days.append(date)

    run.status = current["status"]
    run.equity, run.cash = book.equity, book.cash
    _save(store, account, prep.snapshot_id, run.days[-1], run.status, book.state(), events)
    return run


def _save(store: SnapshotStore, account: PaperAccount, snapshot_id: str, as_of: str, status: str,
          state: dict[str, Any], events: list[tuple[str, str | None, dict[str, Any]]]) -> None:
    """Ledger entries and the new state in one transaction: all or nothing."""
    with store.db.transaction() as conn:
        seq, digest = ledger.head(conn, account.account_id)
        for kind, date, payload in events:
            seq, digest = ledger.append(conn, account.account_id, kind, date, payload)
        conn.execute(
            "INSERT INTO paper_states (account_id, run_at, as_of, snapshot_id, status, ledger_seq, "
            "ledger_hash, state_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (account.account_id, utc_now(), as_of, snapshot_id, status, seq, digest,
             json.dumps(ledger._clean(state), sort_keys=True, allow_nan=False)))


def resume(store: SnapshotStore, account: PaperAccount, *, reason: str) -> None:
    """Lift a suspension. A person's decision, so it needs a reason, and the
    ledger keeps it. A halted account cannot be resumed this way."""
    latest = latest_state(store.db, account.account_id)
    if latest is None or latest["status"] != "suspended":
        raise PaperError(f"{account.name} is not suspended")
    if not reason.strip():
        raise PaperError("resuming needs a reason")
    _save(store, account, latest["snapshot_id"], latest["as_of"], "active", latest["state"],
          [("status", latest["as_of"], {"status": "active", "reason": reason.strip()})])


def summary(store: SnapshotStore, account: PaperAccount) -> dict[str, Any]:
    """Positions, cash and equity from the latest state."""
    latest = latest_state(store.db, account.account_id)
    if latest is None:
        return {"status": "active", "as_of": None, "equity": account.initial_cash,
                "cash": account.initial_cash, "positions": {}, "label": PAPER_LABEL}
    state = latest["state"]
    positions = {}
    for s, p in state["positions"].items():
        if p["shares"] > 0:
            value = p["shares"] * (p["last_close"] or math.nan)
            positions[s] = {"shares": p["shares"], "last_close": p["last_close"], "value": value,
                            "weight": value / state["equity"] if state["equity"] else math.nan,
                            "stop": p["stop"], "take": p["take"],
                            "entry_price": (p["trade"] or {}).get("entry_price")}
    return {"status": latest["status"], "as_of": latest["as_of"], "equity": state["equity"],
            "cash": state["cash"], "peak_equity": state["peak_equity"], "positions": positions,
            "initial_cash": account.initial_cash, "label": PAPER_LABEL,
            "pending": {s: p["pending"] for s, p in state["positions"].items() if p["pending"]}}
