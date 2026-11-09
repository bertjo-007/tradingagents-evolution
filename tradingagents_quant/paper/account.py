"""Paper accounts: opening one, finding it, reading its latest state."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from tradingagents_quant import PAPER_LABEL
from tradingagents_quant.config import QuantConfig
from tradingagents_quant.data.snapshot import SnapshotStore
from tradingagents_quant.errors import QuantError
from tradingagents_quant.paper import ledger
from tradingagents_quant.sim.runner import prepare
from tradingagents_quant.store.db import Database, utc_now
from tradingagents_quant.strategy.spec import StrategySpec


class PaperError(QuantError):
    """A paper account cannot be opened, found or run as asked."""


@dataclass(frozen=True)
class PaperAccount:
    account_id: str
    name: str
    created_at: str
    spec: StrategySpec
    symbols: tuple[str, ...]
    initial_cash: float
    start_date: str
    config_fingerprint: str | None
    label: str = PAPER_LABEL


def _from_row(db: Database, row: sqlite3.Row) -> PaperAccount:
    spec_row = db.query_one("SELECT spec_json FROM strategy_versions WHERE version_hash = ?",
                            (row["strategy_version"],))
    spec = StrategySpec.model_validate(json.loads(spec_row[0]))
    return PaperAccount(row["account_id"], row["name"], row["created_at"], spec,
                        tuple(row["symbols"].split(",")), float(row["initial_cash"]),
                        row["start_date"], row["config_fingerprint"], row["label"])


def open_account(
    store: SnapshotStore, *, name: str, spec: StrategySpec, config: QuantConfig,
    initial_cash: float | None = None, snapshot_id: str | None = None, start: str | None = None,
) -> PaperAccount:
    """A new account. It decides for the first time at the close of ``start``
    (default: the latest trading day in the snapshot) and trades from the
    next open on."""
    if not config.modes.paper_trading_mode:
        raise PaperError("paper_trading_mode is off in the configuration")
    name = name.strip()
    if not name:
        raise PaperError("an account needs a name")
    db = store.db
    if db.query_one("SELECT 1 FROM paper_accounts WHERE name = ?", (name,)):
        raise PaperError(f"a paper account named {name!r} exists already")
    prep = prepare(store, snapshot_id, spec.symbols, config)
    calendar = prep.calendar
    if start:
        calendar = calendar[calendar >= start]
        if not len(calendar):
            raise PaperError(f"no trading day on or after {start} in the snapshot")
        first = calendar[0]
    else:
        first = calendar[-1]
    cash = float(initial_cash if initial_cash is not None else config.backtest.initial_cash)
    if cash <= 0:
        raise PaperError("simulated starting cash must be positive")
    now = utc_now()
    account_id = "pa-" + hashlib.sha256(f"{name}|{now}".encode()).hexdigest()[:10]
    version = spec.version_hash()
    with db.transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO strategy_versions (version_hash, name, rule, spec_json, "
            "first_seen) VALUES (?, ?, ?, ?, ?)",
            (version, spec.name, spec.rule, json.dumps(spec.to_dict(), sort_keys=True), now))
        conn.execute(
            "INSERT OR IGNORE INTO config_versions (fingerprint, config_json, first_seen) "
            "VALUES (?, ?, ?)",
            (config.fingerprint(), json.dumps(config.to_dict(), sort_keys=True), now))
        conn.execute(
            "INSERT INTO paper_accounts (account_id, name, created_at, strategy_version, symbols, "
            "initial_cash, start_date, config_fingerprint, label) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (account_id, name, now, version, ",".join(spec.symbols), cash,
             first.strftime("%Y-%m-%d"), config.fingerprint(), PAPER_LABEL))
        ledger.append(conn, account_id, "opened", first.strftime("%Y-%m-%d"), {
            "label": PAPER_LABEL, "name": name, "initial_cash": cash, "strategy_version": version,
            "spec": spec.to_dict(), "snapshot_id": prep.snapshot_id,
            "config_fingerprint": config.fingerprint(),
        })
    return get_account(db, name)


def get_account(db: Database, name: str) -> PaperAccount:
    row = db.query_one("SELECT * FROM paper_accounts WHERE name = ?", (name,))
    if row is None:
        raise PaperError(f"no paper account named {name!r}; `taq paper list` shows them")
    return _from_row(db, row)


def list_accounts(db: Database) -> list[PaperAccount]:
    return [_from_row(db, r) for r in db.query("SELECT * FROM paper_accounts ORDER BY created_at")]


def latest_state(db: Database, account_id: str) -> dict[str, Any] | None:
    row = db.query_one("SELECT * FROM paper_states WHERE account_id = ? ORDER BY run_at DESC "
                       "LIMIT 1", (account_id,))
    if row is None:
        return None
    return {"as_of": row["as_of"], "run_at": row["run_at"], "status": row["status"],
            "snapshot_id": row["snapshot_id"], "state": json.loads(row["state_json"])}
