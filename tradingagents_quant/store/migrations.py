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

_V2_BACKTESTS = """
-- Every strategy version that produced a stored result, by hash of its spec.
CREATE TABLE strategy_versions (
    version_hash  TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    rule          TEXT NOT NULL,
    spec_json     TEXT NOT NULL,
    first_seen    TEXT NOT NULL
);

-- A backtest is traceable to the strategy version, the exact data (snapshot
-- and series hash), the configuration and the code that produced it.
CREATE TABLE backtest_runs (
    run_id              TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,
    kind                TEXT NOT NULL CHECK (kind IN ('backtest', 'benchmark', 'walkforward', 'optimization')),
    strategy_version    TEXT NOT NULL REFERENCES strategy_versions(version_hash),
    symbol              TEXT NOT NULL,
    snapshot_id         TEXT NOT NULL REFERENCES snapshots(snapshot_id),
    series_sha256       TEXT NOT NULL,
    start_date          TEXT NOT NULL,
    end_date            TEXT NOT NULL,
    config_fingerprint  TEXT REFERENCES config_versions(fingerprint),
    code_version        TEXT NOT NULL,
    parent_run_id       TEXT REFERENCES backtest_runs(run_id),
    experiment_id       TEXT,
    metrics_json        TEXT NOT NULL,
    regimes_json        TEXT,
    notes_json          TEXT
);
CREATE INDEX backtest_runs_by_strategy ON backtest_runs(strategy_version, symbol);

CREATE TABLE backtest_trades (
    run_id        TEXT NOT NULL REFERENCES backtest_runs(run_id),
    n             INTEGER NOT NULL,
    entry_date    TEXT NOT NULL,
    exit_date     TEXT,
    entry_price   REAL NOT NULL,
    exit_price    REAL,
    max_shares    REAL NOT NULL,
    pnl           REAL NOT NULL,
    return_pct    REAL NOT NULL,
    holding_bars  INTEGER NOT NULL,
    exit_reason   TEXT NOT NULL,
    entry_regime  TEXT NOT NULL,
    dividends     REAL NOT NULL,
    PRIMARY KEY (run_id, n)
) WITHOUT ROWID;

CREATE TABLE backtest_equity (
    run_id       TEXT PRIMARY KEY REFERENCES backtest_runs(run_id),
    series_json  TEXT NOT NULL      -- [[date, equity, cash, shares], ...]
) WITHOUT ROWID;
"""


def _append_only(table: str) -> str:
    return f"""
CREATE TRIGGER {table}_no_update BEFORE UPDATE ON {table}
BEGIN SELECT RAISE(ABORT, 'results are append-only: {table}'); END;
CREATE TRIGGER {table}_no_delete BEFORE DELETE ON {table}
BEGIN SELECT RAISE(ABORT, 'results are append-only: {table}'); END;
"""


_V2_TRIGGERS = "".join(
    _append_only(t)
    for t in ("strategy_versions", "backtest_runs", "backtest_trades", "backtest_equity")
)

_V3_EXPERIMENTS = """
-- One walk-forward experiment: a design (rule, grid, windows, criteria),
-- optionally pre-registered as a hypothesis, and its verdict.
CREATE TABLE experiments (
    experiment_id       TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,
    kind                TEXT NOT NULL CHECK (kind IN ('walkforward')),
    status              TEXT NOT NULL CHECK (status IN ('complete', 'failed')),
    hypothesis_id       TEXT,
    hypothesis_sha256   TEXT,
    rule                TEXT NOT NULL,
    symbols             TEXT NOT NULL,
    snapshot_id         TEXT NOT NULL REFERENCES snapshots(snapshot_id),
    config_fingerprint  TEXT REFERENCES config_versions(fingerprint),
    code_version        TEXT NOT NULL,
    design_json         TEXT NOT NULL,
    summary_json        TEXT,
    verdict             TEXT CHECK (verdict IN ('validated', 'failed_validation', 'inconclusive')),
    error               TEXT
);

-- Every parameter set evaluated on every segment, selected or not: the honest
-- number of trials is what the deflated Sharpe ratio needs.
CREATE TABLE experiment_trials (
    experiment_id     TEXT NOT NULL REFERENCES experiments(experiment_id),
    trial_n           INTEGER NOT NULL,
    fold              INTEGER NOT NULL,
    segment           TEXT NOT NULL CHECK (segment IN ('train', 'validation', 'test', 'benchmark')),
    strategy_version  TEXT NOT NULL REFERENCES strategy_versions(version_hash),
    params_json       TEXT NOT NULL,
    start_date        TEXT NOT NULL,
    end_date          TEXT NOT NULL,
    objective         REAL,
    selected          INTEGER NOT NULL DEFAULT 0,
    metrics_json      TEXT NOT NULL,
    PRIMARY KEY (experiment_id, trial_n)
) WITHOUT ROWID;

-- The out-of-sample periods an experiment looked at. A later experiment on
-- the same instruments and overlapping periods is told so, and has to say
-- that it reuses them on purpose.
CREATE TABLE experiment_test_periods (
    experiment_id  TEXT NOT NULL REFERENCES experiments(experiment_id),
    fold           INTEGER NOT NULL,
    symbols        TEXT NOT NULL,
    start_date     TEXT NOT NULL,
    end_date       TEXT NOT NULL,
    PRIMARY KEY (experiment_id, fold)
) WITHOUT ROWID;
CREATE INDEX experiment_test_periods_by_symbols ON experiment_test_periods(symbols);
"""

_V3_TRIGGERS = "".join(
    _append_only(t) for t in ("experiments", "experiment_trials", "experiment_test_periods")
)

_V4_PAPER = """
-- A paper-trading account: one strategy version, simulated cash. Nothing in
-- here is or represents real money.
CREATE TABLE paper_accounts (
    account_id          TEXT PRIMARY KEY,
    name                TEXT NOT NULL UNIQUE,
    created_at          TEXT NOT NULL,
    strategy_version    TEXT NOT NULL REFERENCES strategy_versions(version_hash),
    symbols             TEXT NOT NULL,
    initial_cash        REAL NOT NULL,
    start_date          TEXT NOT NULL,
    config_fingerprint  TEXT REFERENCES config_versions(fingerprint),
    label               TEXT NOT NULL
);

-- Everything that happens to an account, in order, each entry chained to the
-- one before by its hash: an entry changed or removed later breaks the chain.
CREATE TABLE paper_ledger (
    account_id    TEXT NOT NULL REFERENCES paper_accounts(account_id),
    seq           INTEGER NOT NULL,
    recorded_at   TEXT NOT NULL,
    kind          TEXT NOT NULL CHECK (kind IN ('opened', 'fill', 'rejection', 'risk', 'day',
                                                'dividend', 'status', 'rebase', 'note')),
    date          TEXT,
    payload_json  TEXT NOT NULL,
    prev_hash     TEXT NOT NULL,
    hash          TEXT NOT NULL,
    PRIMARY KEY (account_id, seq)
) WITHOUT ROWID;

-- The account's state after each run, to continue from. Every one is kept.
CREATE TABLE paper_states (
    account_id   TEXT NOT NULL REFERENCES paper_accounts(account_id),
    run_at       TEXT NOT NULL,
    as_of        TEXT NOT NULL,
    snapshot_id  TEXT NOT NULL REFERENCES snapshots(snapshot_id),
    status       TEXT NOT NULL CHECK (status IN ('active', 'suspended', 'halted')),
    ledger_seq   INTEGER NOT NULL,
    ledger_hash  TEXT NOT NULL,
    state_json   TEXT NOT NULL,
    PRIMARY KEY (account_id, run_at)
) WITHOUT ROWID;
"""

_V4_TRIGGERS = "".join(_append_only(t) for t in ("paper_accounts", "paper_ledger", "paper_states"))

MIGRATIONS: tuple[str, ...] = (
    _V1_SNAPSHOTS + _V1_TRIGGERS,
    _V2_BACKTESTS + _V2_TRIGGERS,
    _V3_EXPERIMENTS + _V3_TRIGGERS,
    _V4_PAPER + _V4_TRIGGERS,
)

SCHEMA_VERSION = len(MIGRATIONS)
