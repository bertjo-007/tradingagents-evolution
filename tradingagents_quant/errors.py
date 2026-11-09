"""Error types of the research core.

Own types rather than ``tradingagents.dataflows.errors``: importing anything from
``tradingagents`` runs its ``load_dotenv`` (see the package docstring).
"""

from __future__ import annotations


class QuantError(Exception):
    """Base for every failure the research core reports on purpose."""


class ConfigError(QuantError, ValueError):
    """A configuration file or variable is malformed or out of range."""


class LiveTradingForbiddenError(ConfigError):
    """Something asked for live trading or live order execution.

    This system runs research, backtests and paper trading only. The request is
    refused at startup instead of being ignored, so a misconfiguration is seen.
    """


class DataError(QuantError):
    """Market data could not be obtained or is unfit for use."""


class DataUnavailableError(DataError):
    """The provider could not answer: unreachable, throttled, failed.

    Says nothing about the instrument. Nothing is filled in for it.
    """


class NoDataError(DataError):
    """The provider answered that it has no data for the instrument."""


class IngestError(DataError):
    """An ingest produced nothing usable, so no snapshot was written."""


class SnapshotError(QuantError):
    """A snapshot is missing, or its stored data no longer matches its hash."""


class SchemaError(QuantError):
    """The database schema is newer than this code, or a migration failed."""


class UnsafeStorageLocationError(ConfigError):
    """The database or snapshot directory lies inside a synced folder.

    SQLite keeps ``-wal``/``-shm`` files beside the database; a sync client that
    locks or uploads them mid-write causes ``database is locked`` errors and, in
    the worst case, a corrupt database.
    """
