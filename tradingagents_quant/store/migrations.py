"""Schema migrations, applied in order. Migration ``i`` takes the database from
``user_version == i`` to ``i + 1``. Never edit a migration that has shipped;
append a new one.
"""

from __future__ import annotations

_V1_SNAPSHOTS = """
-- Every configuration a stored result was produced under, by fingerprint.
CREATE TABLE config_versions (
    fingerprint  TEXT PRIMARY KEY,
    config_json  TEXT NOT NULL,
    first_seen   TEXT NOT NULL
);

-- A snapshot is an immutable copy of market data as a provider reported it at
-- one moment. Backtests read snapshots only, never a live provider, so a result
-- can be reproduced after the provider restates its history.
CREATE TABLE snapshots (
    snapshot_id         TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,              -- UTC, ISO 8601
    source              TEXT NOT NULL,              -- e.g. 'yahoo'
    source_version      TEXT NOT NULL,              -- e.g. 'yfinance 1.8.0'
    config_fingerprint  TEXT REFERENCES config_versions(fingerprint),
    requested_start     TEXT NOT NULL,
    requested_end       TEXT,                       -- NULL: up to the fetch date
    status              TEXT NOT NULL CHECK (status IN ('building', 'complete', 'partial')),
    content_hash        TEXT
);

CREATE TABLE snapshot_series (
    snapshot_id       TEXT NOT NULL REFERENCES snapshots(snapshot_id),
    symbol            TEXT NOT NULL,
    canonical_symbol  TEXT NOT NULL,
    currency          TEXT,
    first_date        TEXT NOT NULL,
    last_date         TEXT NOT NULL,
    n_bars            INTEGER NOT NULL,
    sha256            TEXT NOT NULL,
    quality           TEXT NOT NULL CHECK (quality IN ('ok', 'warning', 'error')),
    PRIMARY KEY (snapshot_id, symbol)
) WITHOUT ROWID;

-- Daily bars as reported. close is split-adjusted (Yahoo's convention),
-- adj_close is split- and dividend-adjusted. Nothing here is filled in: a value
-- the provider did not report is NULL, a bar without a close is not stored.
CREATE TABLE bars (
    snapshot_id  TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    date         TEXT NOT NULL,                     -- exchange-local YYYY-MM-DD
    open         REAL,
    high         REAL,
    low          REAL,
    close        REAL NOT NULL,
    adj_close    REAL,
    volume       REAL,
    PRIMARY KEY (snapshot_id, symbol, date),
    FOREIGN KEY (snapshot_id, symbol) REFERENCES snapshot_series(snapshot_id, symbol)
) WITHOUT ROWID;

CREATE TABLE corporate_actions (
    snapshot_id  TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    date         TEXT NOT NULL,
    kind         TEXT NOT NULL CHECK (kind IN ('dividend', 'split', 'capital_gain')),
    value        REAL NOT NULL,
    PRIMARY KEY (snapshot_id, symbol, date, kind),
    FOREIGN KEY (snapshot_id, symbol) REFERENCES snapshot_series(snapshot_id, symbol)
) WITHOUT ROWID;

CREATE TABLE quality_issues (
    id           INTEGER PRIMARY KEY,
    snapshot_id  TEXT NOT NULL REFERENCES snapshots(snapshot_id),
    symbol       TEXT NOT NULL,
    date         TEXT,
    rule         TEXT NOT NULL,
    severity     TEXT NOT NULL CHECK (severity IN ('error', 'warning', 'info')),
    detail       TEXT NOT NULL
);
CREATE INDEX quality_issues_by_series ON quality_issues(snapshot_id, symbol);

-- Symbols the provider could not deliver. Recorded, never filled in.
CREATE TABLE ingest_failures (
    snapshot_id  TEXT NOT NULL REFERENCES snapshots(snapshot_id),
    symbol       TEXT NOT NULL,
    error_type   TEXT NOT NULL,
    detail       TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, symbol)
) WITHOUT ROWID;

CREATE TABLE system_events (
    id            INTEGER PRIMARY KEY,
    ts            TEXT NOT NULL,
    component     TEXT NOT NULL,
    level         TEXT NOT NULL CHECK (level IN ('debug', 'info', 'warning', 'error')),
    message       TEXT NOT NULL,
    context_json  TEXT
);

-- Immutability, enforced by the database rather than by convention. Rows of a
-- snapshot are written only while it is building, inside one transaction, and
-- never changed or removed afterwards.
CREATE TRIGGER snapshots_sealed BEFORE UPDATE ON snapshots
WHEN OLD.status <> 'building'
BEGIN SELECT RAISE(ABORT, 'a sealed snapshot cannot change'); END;

CREATE TRIGGER snapshots_kept BEFORE DELETE ON snapshots
WHEN OLD.status <> 'building'
BEGIN SELECT RAISE(ABORT, 'a sealed snapshot cannot be deleted'); END;
"""


def _sealed_child(table: str) -> str:
    return f"""
CREATE TRIGGER {table}_insert_while_building BEFORE INSERT ON {table}
WHEN (SELECT status FROM snapshots WHERE snapshot_id = NEW.snapshot_id) IS NOT 'building'
BEGIN SELECT RAISE(ABORT, 'snapshot is sealed: {table} cannot be added'); END;

CREATE TRIGGER {table}_no_update BEFORE UPDATE ON {table}
BEGIN SELECT RAISE(ABORT, 'snapshot data is immutable: {table}'); END;

CREATE TRIGGER {table}_no_delete BEFORE DELETE ON {table}
WHEN (SELECT status FROM snapshots WHERE snapshot_id = OLD.snapshot_id) IS NOT 'building'
BEGIN SELECT RAISE(ABORT, 'snapshot data is immutable: {table}'); END;
"""


_V1_TRIGGERS = "".join(
    _sealed_child(t)
    for t in ("snapshot_series", "bars", "corporate_actions", "quality_issues", "ingest_failures")
)

MIGRATIONS: tuple[str, ...] = (_V1_SNAPSHOTS + _V1_TRIGGERS,)

SCHEMA_VERSION = len(MIGRATIONS)
