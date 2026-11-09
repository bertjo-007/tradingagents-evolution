"""The SQLite connection: pragmas, migrations, transactions."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tradingagents_quant import paths
from tradingagents_quant.errors import SchemaError
from tradingagents_quant.store.migrations import MIGRATIONS, SCHEMA_VERSION


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Database:
    """One connection, in autocommit mode; ``transaction()`` groups writes."""

    def __init__(self, path: Path | str, *, read_only: bool = False) -> None:
        self.path = path
        memory = str(path) == ":memory:"
        if read_only:
            # Never creates a file, never migrates, never writes.
            uri = Path(path).resolve().as_uri() + "?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=10.0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA busy_timeout = 10000")
            return
        if not memory:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: no implicit BEGIN; transactions are explicit.
        self._conn = sqlite3.connect(str(path), isolation_level=None, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 10000")
        if not memory:
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")

    @classmethod
    def open_default(cls) -> Database:
        """The database in the configured data directory, migrated."""
        db = cls(paths.db_path(paths.data_home()))
        db.migrate()
        return db

    # -- schema -------------------------------------------------------------

    @property
    def schema_version(self) -> int:
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    def migrate(self) -> int:
        """Apply the migrations this database has not seen; return the version."""
        current = self.schema_version
        if current > SCHEMA_VERSION:
            raise SchemaError(
                f"database schema version {current} is newer than this code "
                f"({SCHEMA_VERSION}); update the code instead of opening it"
            )
        for version in range(current, SCHEMA_VERSION):
            script = f"BEGIN;\n{MIGRATIONS[version]}\nPRAGMA user_version = {version + 1};\nCOMMIT;"
            try:
                self._conn.executescript(script)
            except sqlite3.Error as exc:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise SchemaError(f"migration to version {version + 1} failed: {exc}") from exc
        return self.schema_version

    # -- access -------------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """All-or-nothing: BEGIN IMMEDIATE, COMMIT on success, ROLLBACK on error."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return self._conn.execute(sql, tuple(params)).fetchall()

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        return self._conn.execute(sql, tuple(params)).fetchone()

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, tuple(params))

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def log_event(
        self, component: str, level: str, message: str, context: dict[str, Any] | None = None
    ) -> None:
        self._conn.execute(
            "INSERT INTO system_events (ts, component, level, message, context_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (utc_now(), component, level, message, json.dumps(context) if context else None),
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
