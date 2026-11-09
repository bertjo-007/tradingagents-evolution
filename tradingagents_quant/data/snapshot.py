"""Immutable, hashed snapshots of market data in SQLite.

A snapshot is written in one transaction and sealed; the database refuses any
later change (see ``store/migrations.py``). Each series carries a SHA-256 over
its exact values, the snapshot a hash over all series, so ``verify_integrity``
can prove that what a backtest reads is what was fetched.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd

from tradingagents_quant.config import QuantConfig
from tradingagents_quant.data.normalize import BAR_COLUMNS, PriceSeries
from tradingagents_quant.data.quality import Issue, worst
from tradingagents_quant.errors import SnapshotError
from tradingagents_quant.store.db import Database, utc_now


@dataclass(frozen=True)
class Failure:
    symbol: str
    error_type: str
    detail: str


@dataclass(frozen=True)
class SeriesInfo:
    symbol: str
    canonical_symbol: str
    currency: str | None
    first_date: str
    last_date: str
    n_bars: int
    sha256: str
    quality: str


@dataclass(frozen=True)
class SnapshotInfo:
    snapshot_id: str
    created_at: str
    source: str
    source_version: str
    status: str
    content_hash: str
    config_fingerprint: str | None
    requested_start: str
    requested_end: str | None
    series: tuple[SeriesInfo, ...]

    def symbols(self) -> tuple[str, ...]:
        return tuple(s.symbol for s in self.series)

    def series_info(self, symbol: str) -> SeriesInfo:
        for s in self.series:
            if s.symbol == symbol:
                return s
        raise SnapshotError(f"{symbol} is not in snapshot {self.snapshot_id}")


def _num(value: float | None) -> str:
    """Exact text of a stored number: repr round-trips an IEEE double."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return repr(float(value))


def series_hash(
    bars: Iterable[Sequence[object]], actions: Iterable[Sequence[object]]
) -> str:
    """SHA-256 over bars (date + BAR_COLUMNS) and actions (date, kind, value),
    each in date order. Computed the same way on write and on verify."""
    digest = hashlib.sha256()
    for row in bars:
        digest.update(
            (",".join([str(row[0]), *(_num(v) for v in row[1:])]) + "\n").encode()  # type: ignore[arg-type]
        )
    digest.update(b"--actions--\n")
    for day, kind, value in actions:
        digest.update(f"{day},{kind},{_num(value)}\n".encode())  # type: ignore[arg-type]
    return digest.hexdigest()


def _bar_rows(series: PriceSeries) -> list[tuple]:
    rows = []
    for day, values in zip(series.bars.index, series.bars.itertuples(index=False), strict=True):
        rows.append((str(day), *(None if math.isnan(v) else float(v) for v in values)))
    return rows


def _action_rows(series: PriceSeries) -> list[tuple]:
    return [(str(r.date), str(r.kind), float(r.value)) for r in series.actions.itertuples(index=False)]


def content_hash(series_hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for symbol in sorted(series_hashes):
        digest.update(f"{symbol}:{series_hashes[symbol]}\n".encode())
    return digest.hexdigest()


class SnapshotStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    # -- write --------------------------------------------------------------

    def create(
        self,
        *,
        series: Sequence[PriceSeries],
        issues: Sequence[Issue],
        failures: Sequence[Failure],
        source: str,
        source_version: str,
        requested_start: str,
        requested_end: str | None,
        config: QuantConfig | None,
        created_at: str | None = None,
    ) -> SnapshotInfo:
        if not series:
            raise SnapshotError("a snapshot needs at least one series")
        empty = [s.symbol for s in series if s.bars.empty]
        if empty:
            raise SnapshotError(f"series without bars cannot be stored: {', '.join(empty)}")
        symbols = [s.symbol for s in series]
        twice = sorted({s for s in symbols if symbols.count(s) > 1})
        if twice:
            raise SnapshotError(f"series given more than once: {', '.join(twice)}")
        created_at = created_at or utc_now()

        prepared = {}
        for s in series:
            bars, actions = _bar_rows(s), _action_rows(s)
            prepared[s.symbol] = (s, bars, actions, series_hash(bars, actions))
        snapshot_hash = content_hash({k: v[3] for k, v in prepared.items()})
        stamp = created_at.replace("-", "").replace(":", "").split(".")[0] + "Z"
        snapshot_id = f"snap-{stamp}-{snapshot_hash[:8]}"
        status = "partial" if failures else "complete"
        issues_by_symbol: dict[str, list[Issue]] = {}
        for issue in issues:
            issues_by_symbol.setdefault(issue.symbol, []).append(issue)

        with self.db.transaction() as conn:
            fingerprint = None
            if config is not None:
                fingerprint = config.fingerprint()
                conn.execute(
                    "INSERT OR IGNORE INTO config_versions (fingerprint, config_json, first_seen) "
                    "VALUES (?, ?, ?)",
                    (fingerprint, json.dumps(config.to_dict(), sort_keys=True), created_at),
                )
            conn.execute(
                "INSERT INTO snapshots (snapshot_id, created_at, source, source_version, "
                "config_fingerprint, requested_start, requested_end, status, content_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'building', ?)",
                (snapshot_id, created_at, source, source_version, fingerprint,
                 requested_start, requested_end, snapshot_hash),
            )
            for symbol, (s, bars, actions, digest) in prepared.items():
                conn.execute(
                    "INSERT INTO snapshot_series (snapshot_id, symbol, canonical_symbol, currency, "
                    "first_date, last_date, n_bars, sha256, quality) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (snapshot_id, symbol, s.canonical, s.currency, s.first_date, s.last_date,
                     len(bars), digest, worst(issues_by_symbol.get(symbol, []))),
                )
                conn.executemany(
                    "INSERT INTO bars (snapshot_id, symbol, date, open, high, low, close, adj_close, "
                    "volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(snapshot_id, symbol, *row) for row in bars],
                )
                conn.executemany(
                    "INSERT INTO corporate_actions (snapshot_id, symbol, date, kind, value) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [(snapshot_id, symbol, *row) for row in actions],
                )
            conn.executemany(
                "INSERT INTO quality_issues (snapshot_id, symbol, date, rule, severity, detail) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(snapshot_id, i.symbol, i.date, i.rule, i.severity, i.detail) for i in issues],
            )
            conn.executemany(
                "INSERT INTO ingest_failures (snapshot_id, symbol, error_type, detail) "
                "VALUES (?, ?, ?, ?)",
                [(snapshot_id, f.symbol, f.error_type, f.detail) for f in failures],
            )
            # Sealing is the last write; from here on the triggers refuse changes.
            conn.execute(
                "UPDATE snapshots SET status = ? WHERE snapshot_id = ?", (status, snapshot_id)
            )
        return self.get(snapshot_id)

    # -- read ---------------------------------------------------------------

    def get(self, snapshot_id: str) -> SnapshotInfo:
        row = self.db.query_one("SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,))
        if row is None:
            raise SnapshotError(f"no snapshot {snapshot_id!r}")
        series = tuple(
            SeriesInfo(r["symbol"], r["canonical_symbol"], r["currency"], r["first_date"],
                       r["last_date"], r["n_bars"], r["sha256"], r["quality"])
            for r in self.db.query(
                "SELECT * FROM snapshot_series WHERE snapshot_id = ? ORDER BY symbol", (snapshot_id,)
            )
        )
        return SnapshotInfo(row["snapshot_id"], row["created_at"], row["source"],
                            row["source_version"], row["status"], row["content_hash"],
                            row["config_fingerprint"], row["requested_start"],
                            row["requested_end"], series)

    def list(self) -> list[SnapshotInfo]:
        rows = self.db.query(
            "SELECT snapshot_id FROM snapshots WHERE status <> 'building' ORDER BY created_at"
        )
        return [self.get(r["snapshot_id"]) for r in rows]

    def latest(self) -> SnapshotInfo | None:
        row = self.db.query_one(
            "SELECT snapshot_id FROM snapshots WHERE status <> 'building' "
            "ORDER BY created_at DESC LIMIT 1"
        )
        return self.get(row["snapshot_id"]) if row else None

    def resolve(self, snapshot_id: str | None) -> SnapshotInfo:
        """The named snapshot, or the latest one if none is named."""
        if snapshot_id:
            return self.get(snapshot_id)
        latest = self.latest()
        if latest is None:
            raise SnapshotError("no snapshot yet; run `taq ingest` first")
        return latest

    def bars(
        self, snapshot_id: str, symbol: str, *, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        """Bars of one series, dates ``start <= date <= end`` (both inclusive).

        ``end`` is the point-in-time cutoff: a backtest deciding on day ``t``
        asks with ``end=t`` and cannot see a later bar.
        """
        sql = ("SELECT date, open, high, low, close, adj_close, volume FROM bars "
               "WHERE snapshot_id = ? AND symbol = ?")
        params: list[object] = [snapshot_id, symbol]
        if start:
            sql += " AND date >= ?"
            params.append(start)
        if end:
            sql += " AND date <= ?"
            params.append(end)
        rows = self.db.query(sql + " ORDER BY date", params)
        frame = pd.DataFrame([tuple(r) for r in rows], columns=["date", *BAR_COLUMNS])
        frame[list(BAR_COLUMNS)] = frame[list(BAR_COLUMNS)].astype(float)
        return frame.set_index(pd.DatetimeIndex(pd.to_datetime(frame.pop("date")), name="date"))

    def actions(self, snapshot_id: str, symbol: str) -> pd.DataFrame:
        rows = self.db.query(
            "SELECT date, kind, value FROM corporate_actions WHERE snapshot_id = ? AND symbol = ? "
            "ORDER BY date, kind",
            (snapshot_id, symbol),
        )
        return pd.DataFrame([tuple(r) for r in rows], columns=["date", "kind", "value"])

    def issues(self, snapshot_id: str, symbol: str | None = None) -> list[Issue]:
        sql = "SELECT symbol, rule, severity, detail, date FROM quality_issues WHERE snapshot_id = ?"
        params: list[object] = [snapshot_id]
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol)
        rows = self.db.query(sql + " ORDER BY symbol, id", params)
        return [Issue(r["symbol"], r["rule"], r["severity"], r["detail"], r["date"]) for r in rows]

    def failures(self, snapshot_id: str) -> list[Failure]:
        rows = self.db.query(
            "SELECT symbol, error_type, detail FROM ingest_failures WHERE snapshot_id = ? "
            "ORDER BY symbol",
            (snapshot_id,),
        )
        return [Failure(r["symbol"], r["error_type"], r["detail"]) for r in rows]

    # -- verify -------------------------------------------------------------

    def verify_integrity(self, snapshot_id: str) -> dict[str, tuple[str, str]]:
        """Re-hash every stored series; return ``{symbol: (stored, recomputed)}``
        for each that no longer matches. Empty means intact."""
        info = self.get(snapshot_id)
        mismatches = {}
        for s in info.series:
            bars = self.db.query(
                "SELECT date, open, high, low, close, adj_close, volume FROM bars "
                "WHERE snapshot_id = ? AND symbol = ? ORDER BY date",
                (snapshot_id, s.symbol),
            )
            actions = self.db.query(
                "SELECT date, kind, value FROM corporate_actions WHERE snapshot_id = ? AND symbol = ? "
                "ORDER BY date, kind",
                (snapshot_id, s.symbol),
            )
            recomputed = series_hash([tuple(r) for r in bars], [tuple(r) for r in actions])
            if recomputed != s.sha256 or len(bars) != s.n_bars:
                mismatches[s.symbol] = (s.sha256, recomputed)
        recomputed_content = content_hash({s.symbol: s.sha256 for s in info.series})
        if recomputed_content != info.content_hash:
            mismatches["<snapshot>"] = (info.content_hash, recomputed_content)
        return mismatches
