"""Turn a provider's daily frame into the canonical series, without judging it.

Canonical form: one row per exchange-local date (``YYYY-MM-DD``), sorted,
columns ``open high low close adj_close volume`` as floats. Nothing is filled
in. What cannot be stored (a bar without a close, a duplicate date, an unsettled
session) is left out *and recorded* in ``dropped``, so the quality rules can
report it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import pandas as pd

BAR_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")

# Provider column names, by canonical name.
_VENDOR_COLUMNS = {
    "open": ("Open", "open"),
    "high": ("High", "high"),
    "low": ("Low", "low"),
    "close": ("Close", "close"),
    "adj_close": ("Adj Close", "adj_close", "Adj_Close", "adjclose"),
    "volume": ("Volume", "volume"),
}
_ACTION_COLUMNS = {
    "dividend": ("Dividends",),
    "split": ("Stock Splits",),
    "capital_gain": ("Capital Gains",),
}


@dataclass(frozen=True)
class Dropped:
    date: str
    reason: str  # missing_close | duplicate_identical | duplicate_conflicting | unsettled_session
    detail: str = ""


@dataclass(frozen=True)
class PriceSeries:
    symbol: str
    canonical: str
    currency: str | None
    bars: pd.DataFrame  # index: str dates, columns BAR_COLUMNS
    actions: pd.DataFrame  # columns: date, kind, value
    dropped: tuple[Dropped, ...] = field(default_factory=tuple)

    @property
    def first_date(self) -> str:
        return str(self.bars.index[0])

    @property
    def last_date(self) -> str:
        return str(self.bars.index[-1])


def _column(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series | None:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return None


def _exchange_dates(index: pd.Index) -> list[str]:
    """Each timestamp as the exchange-local calendar date it belongs to.

    tz_localize(None) keeps the local wall clock; converting to UTC instead
    would move a positive-offset market's bars to the previous day.
    """
    if isinstance(index, pd.DatetimeIndex):
        local = index.tz_localize(None) if index.tz is not None else index
        return [ts.strftime("%Y-%m-%d") for ts in local.normalize()]
    out = []
    for value in index:
        ts = pd.Timestamp(value)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        out.append(ts.normalize().strftime("%Y-%m-%d"))
    return out


def exchange_today(index: pd.Index, now: datetime) -> date:
    """Today's date where the exchange is, judged by the frame's own timezone."""
    tz = getattr(index, "tz", None)
    moment = pd.Timestamp(now if now.tzinfo else now.replace(tzinfo=UTC))
    return (moment.tz_convert(tz) if tz is not None else moment.tz_convert(UTC)).date()


def _float_or_nan(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def normalize_vendor_frame(
    symbol: str,
    canonical: str,
    frame: pd.DataFrame,
    *,
    fetched_at: datetime,
    currency: str | None = None,
) -> PriceSeries:
    """Canonicalize a provider frame (index: timestamps, yfinance-style columns).

    ``fetched_at`` decides which session is still unsettled: a bar dated today
    or later in the exchange's timezone is left out, because its close may be
    an intraday price and a snapshot must never hold one.
    """
    if frame is None or frame.empty:
        empty = pd.DataFrame(columns=list(BAR_COLUMNS), dtype=float)
        empty.index.name = "date"
        return PriceSeries(symbol, canonical, currency, empty, _empty_actions())

    dates = _exchange_dates(frame.index)
    today = exchange_today(frame.index, fetched_at).isoformat()

    columns = {}
    for name, candidates in _VENDOR_COLUMNS.items():
        series = _column(frame, candidates)
        columns[name] = (
            [math.nan] * len(frame) if series is None else [_float_or_nan(v) for v in series]
        )

    dropped: list[Dropped] = []
    kept: dict[str, tuple[float, ...]] = {}
    for i, day in enumerate(dates):
        row = tuple(columns[c][i] for c in BAR_COLUMNS)
        if day >= today:
            dropped.append(Dropped(day, "unsettled_session", f"session of {day} not settled at fetch"))
            continue
        if math.isnan(row[BAR_COLUMNS.index("close")]):
            dropped.append(Dropped(day, "missing_close", "provider reported no close"))
            continue
        if day in kept:
            same = all(
                (math.isnan(a) and math.isnan(b)) or a == b for a, b in zip(kept[day], row, strict=True)
            )
            if same:
                dropped.append(Dropped(day, "duplicate_identical", "same bar reported twice"))
            else:
                dropped.append(
                    Dropped(day, "duplicate_conflicting", f"second bar differs: {row} vs {kept[day]}")
                )
            continue  # the first report is kept
        kept[day] = row

    ordered = sorted(kept)
    bars = pd.DataFrame([kept[d] for d in ordered], index=pd.Index(ordered, name="date"),
                        columns=list(BAR_COLUMNS), dtype=float)

    actions = _extract_actions(frame, dates, today)
    return PriceSeries(symbol, canonical, currency, bars, actions, tuple(dropped))


def _empty_actions() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.Series(dtype=str), "kind": pd.Series(dtype=str),
                         "value": pd.Series(dtype=float)})


def _extract_actions(frame: pd.DataFrame, dates: list[str], today: str) -> pd.DataFrame:
    rows = []
    for kind, candidates in _ACTION_COLUMNS.items():
        series = _column(frame, candidates)
        if series is None:
            continue
        seen = set()
        for day, value in zip(dates, series, strict=True):
            number = _float_or_nan(value)
            if day >= today or math.isnan(number) or number == 0.0 or day in seen:
                continue
            seen.add(day)
            rows.append({"date": day, "kind": kind, "value": number})
    if not rows:
        return _empty_actions()
    return pd.DataFrame(rows).sort_values(["date", "kind"], ignore_index=True)
