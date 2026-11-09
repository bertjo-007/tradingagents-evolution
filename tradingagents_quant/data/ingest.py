"""Fetch, normalize, check and seal a snapshot. The provider is injected, so the
whole pipeline runs offline in tests.

Only expected data failures (``DataUnavailableError``, ``NoDataError``) are
recorded per symbol; any other exception is a bug and propagates. A symbol that
fails is listed in the snapshot's failures and gets no bars — never invented
ones, and never another provider's without saying so.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from tradingagents_quant.config import QuantConfig
from tradingagents_quant.data.normalize import BAR_COLUMNS, PriceSeries, normalize_vendor_frame
from tradingagents_quant.data.quality import Issue, check_calendar, check_series, reference_date
from tradingagents_quant.data.snapshot import Failure, SnapshotInfo, SnapshotStore
from tradingagents_quant.errors import DataError, IngestError, NoDataError


@dataclass(frozen=True)
class RawFetch:
    canonical: str
    frame: pd.DataFrame
    currency: str | None = None


# symbol, start (inclusive), end (inclusive or None) -> RawFetch
Fetcher = Callable[[str, str, str | None], RawFetch]


@dataclass
class IngestResult:
    snapshot: SnapshotInfo
    issues: list[Issue] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)


def fetch_all(
    symbols: Sequence[str],
    *,
    start: str,
    end: str | None,
    fetch: Fetcher,
    fetched_at: datetime,
) -> tuple[dict[str, PriceSeries], list[Failure]]:
    series: dict[str, PriceSeries] = {}
    failures: list[Failure] = []
    for symbol in symbols:
        try:
            raw = fetch(symbol, start, end)
        except DataError as exc:
            failures.append(Failure(symbol, type(exc).__name__, str(exc)))
            continue
        normalized = normalize_vendor_frame(
            symbol, raw.canonical, raw.frame, fetched_at=fetched_at, currency=raw.currency
        )
        if end is not None:
            normalized = _clip_end(normalized, end)
        if normalized.bars.empty:
            failures.append(Failure(symbol, NoDataError.__name__, "no settled bar with a close"))
            continue
        series[symbol] = normalized
    return series, failures


def _clip_end(series: PriceSeries, end: str) -> PriceSeries:
    """Providers sometimes return a bar past the requested end; it is not kept."""
    bars = series.bars[series.bars.index <= end]
    actions = series.actions[series.actions["date"] <= end].reset_index(drop=True)
    return PriceSeries(series.symbol, series.canonical, series.currency, bars, actions,
                       tuple(d for d in series.dropped if d.date <= end))


def ingest(
    store: SnapshotStore,
    *,
    symbols: Sequence[str],
    start: str,
    end: str | None,
    fetch: Fetcher,
    source: str,
    source_version: str,
    config: QuantConfig,
    now: datetime | None = None,
) -> IngestResult:
    now = now or datetime.now(UTC)
    series, failures = fetch_all(symbols, start=start, end=end, fetch=fetch, fetched_at=now)
    if not series:
        details = "; ".join(f"{f.symbol}: {f.detail}" for f in failures) or "no symbols"
        store.db.log_event("ingest", "error", "ingest produced no series", {"failures": details})
        raise IngestError(f"no series could be ingested, no snapshot written ({details})")

    as_of = reference_date(now.date(), end)
    issues: list[Issue] = []
    for s in series.values():
        issues += check_series(s, as_of=as_of, max_stale_days=config.data.max_stale_days)
    issues += check_calendar(series, config.universe.calendar_symbol)

    snapshot = store.create(
        series=list(series.values()),
        issues=issues,
        failures=failures,
        source=source,
        source_version=source_version,
        requested_start=start,
        requested_end=end,
        config=config,
    )
    level = "warning" if failures else "info"
    store.db.log_event("ingest", level, f"snapshot {snapshot.snapshot_id} sealed",
                       {"series": len(series), "failures": [f.symbol for f in failures]})
    return IngestResult(snapshot, issues, failures)


# -- drift ------------------------------------------------------------------


@dataclass(frozen=True)
class Drift:
    symbol: str
    missing_dates: tuple[str, ...]  # in the snapshot, gone at the provider
    new_dates: tuple[str, ...]  # at the provider, not in the snapshot
    changed: tuple[tuple[str, str, float | None, float | None], ...]  # date, column, old, new

    @property
    def clean(self) -> bool:
        return not (self.missing_dates or self.new_dates or self.changed)


# Yahoo recomputes adj_close on every request in float32: two fetches minutes
# apart differ by up to 2.2e-6 relative (measured 2026-11-09 over 108,412 bars;
# open/high/low/close/volume were bit-identical). A real restatement is far
# larger: a new dividend moves adjusted history by ~1e-3, a split by 2x.
DRIFT_TOLERANCE = 1e-5


def _same(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) <= DRIFT_TOLERANCE * max(abs(a), abs(b), 1e-12)


def compare_series(symbol: str, stored: pd.DataFrame, fresh: PriceSeries) -> Drift:
    """How the provider's current history differs from the snapshot, over the
    snapshot's date range. A restated close (a new split or dividend
    adjustment) shows here as changed values."""
    old = stored.copy()
    old.index = [d.strftime("%Y-%m-%d") for d in old.index]
    new = fresh.bars[(fresh.bars.index >= old.index[0]) & (fresh.bars.index <= old.index[-1])]
    missing = tuple(sorted(set(old.index) - set(new.index)))
    added = tuple(sorted(set(new.index) - set(old.index)))
    changed = []
    for day in sorted(set(old.index) & set(new.index)):
        for column in BAR_COLUMNS:
            a, b = float(old.at[day, column]), float(new.at[day, column])
            if not _same(a, b):
                changed.append((day, column, None if math.isnan(a) else a, None if math.isnan(b) else b))
    return Drift(symbol, missing, added, tuple(changed))
