"""The database: migrations by user_version, pragmas, refusal of a newer schema."""

from __future__ import annotations

import pytest

from tradingagents_quant.errors import SchemaError
from tradingagents_quant.store.db import Database
from tradingagents_quant.store.migrations import SCHEMA_VERSION


def test_migration_from_an_empty_database_to_the_current_version():
    db = Database(":memory:")
    assert db.schema_version == 0
    assert db.migrate() == SCHEMA_VERSION
    tables = {r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"snapshots", "snapshot_series", "bars", "corporate_actions", "quality_issues",
            "ingest_failures", "system_events", "config_versions"} <= tables


def test_migrating_twice_changes_nothing():
    db = Database(":memory:")
    db.migrate()
    before = db.query("SELECT sql FROM sqlite_master ORDER BY name")
    assert db.migrate() == SCHEMA_VERSION
    assert [r["sql"] for r in db.query("SELECT sql FROM sqlite_master ORDER BY name")] == [
        r["sql"] for r in before
    ]


def test_a_database_newer_than_the_code_is_refused(tmp_path):
    path = tmp_path / "taq.sqlite3"
    db = Database(path)
    db.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    db.close()
    with pytest.raises(SchemaError, match="newer"):
        Database(path).migrate()


def test_a_file_database_uses_wal_and_foreign_keys(tmp_path):
    db = Database(tmp_path / "sub" / "taq.sqlite3")
    assert db.query_one("PRAGMA journal_mode")[0] == "wal"
    assert db.query_one("PRAGMA foreign_keys")[0] == 1


def test_a_failed_transaction_leaves_nothing(tmp_path):
    db = Database(":memory:")
    db.migrate()
    with pytest.raises(RuntimeError), db.transaction() as conn:
        conn.execute("INSERT INTO system_events (ts, component, level, message) "
                     "VALUES ('t', 'test', 'info', 'm')")
        raise RuntimeError("boom")
    assert db.query_one("SELECT COUNT(*) FROM system_events")[0] == 0


def test_events_are_logged():
    db = Database(":memory:")
    db.migrate()
    db.log_event("test", "warning", "something", {"n": 1})
    row = db.query_one("SELECT component, level, message, context_json FROM system_events")
    assert tuple(row) == ("test", "warning", "something", '{"n": 1}')
