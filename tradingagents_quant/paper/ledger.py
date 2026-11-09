"""The account ledger: append-only, each entry chained to the previous one.

``hash = sha256(prev_hash | seq | recorded_at | kind | date | payload)``. The
database refuses updates and deletes (triggers); the chain makes a change that
went around the database visible to ``verify``.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from typing import Any

from tradingagents_quant.store.db import Database, utc_now

GENESIS = "0" * 64


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    return value


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(_clean(payload), sort_keys=True, separators=(",", ":"), allow_nan=False,
                      default=str)


def entry_hash(prev: str, seq: int, recorded_at: str, kind: str, date: str | None,
               payload_json: str) -> str:
    text = "|".join([prev, str(seq), recorded_at, kind, date or "", payload_json])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def head(conn: sqlite3.Connection, account_id: str) -> tuple[int, str]:
    row = conn.execute("SELECT seq, hash FROM paper_ledger WHERE account_id = ? "
                       "ORDER BY seq DESC LIMIT 1", (account_id,)).fetchone()
    return (int(row[0]), str(row[1])) if row else (-1, GENESIS)


def append(conn: sqlite3.Connection, account_id: str, kind: str, date: str | None,
           payload: dict[str, Any]) -> tuple[int, str]:
    seq, prev = head(conn, account_id)
    seq += 1
    recorded_at = utc_now()
    body = _canonical(payload)
    digest = entry_hash(prev, seq, recorded_at, kind, date, body)
    conn.execute(
        "INSERT INTO paper_ledger (account_id, seq, recorded_at, kind, date, payload_json, "
        "prev_hash, hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (account_id, seq, recorded_at, kind, date, body, prev, digest))
    return seq, digest


def verify(db: Database, account_id: str) -> list[str]:
    """Problems in the chain; an empty list means intact."""
    problems, prev, expected = [], GENESIS, 0
    for r in db.query("SELECT * FROM paper_ledger WHERE account_id = ? ORDER BY seq",
                      (account_id,)):
        if r["seq"] != expected:
            problems.append(f"entry {expected} missing (found {r['seq']})")
            expected = r["seq"]
        if r["prev_hash"] != prev:
            problems.append(f"entry {r['seq']}: link to the previous entry broken")
        recomputed = entry_hash(r["prev_hash"], r["seq"], r["recorded_at"], r["kind"], r["date"],
                                r["payload_json"])
        if recomputed != r["hash"]:
            problems.append(f"entry {r['seq']} ({r['kind']}): content does not match its hash")
        prev, expected = r["hash"], expected + 1
    return problems


def entries(db: Database, account_id: str, kinds: tuple[str, ...] | None = None,
            limit: int | None = None) -> list[dict[str, Any]]:
    sql = "SELECT seq, recorded_at, kind, date, payload_json FROM paper_ledger WHERE account_id = ?"
    params: list[Any] = [account_id]
    if kinds:
        sql += f" AND kind IN ({','.join('?' * len(kinds))})"
        params += list(kinds)
    sql += " ORDER BY seq DESC" + (" LIMIT ?" if limit else "")
    if limit:
        params.append(limit)
    rows = db.query(sql, params)
    return [{"seq": r["seq"], "recorded_at": r["recorded_at"], "kind": r["kind"], "date": r["date"],
             **json.loads(r["payload_json"])} for r in reversed(rows)]
