"""Quality rules for a canonical price series. They report; they never repair.

Severity:
- ``error``   the series is unfit for a backtest; the engine refuses it.
- ``warning`` usable, but the report shows it next to every result.
- ``info``    expected and harmless (an unsettled session left out).

A rule that fires on many dates stores the first ``MAX_PER_RULE`` and one
summary issue, so a broken series does not flood the database.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from tradingagents_quant.data.normalize import PriceSeries

SEVERITIES = ("info", "warning", "error")
MAX_PER_RULE = 25

# A high below the open/close/low (or a low above them) by more than this
# fraction of the close is material: stops and ranges computed from it would be
# wrong. Below it, it is rounding noise in the provider's data.
OHLC_MATERIAL = 0.01
OHLC_NOISE = 1e-6
# A day-over-day close move larger than this, on a day without a split, is
# flagged for a look: for ETFs it is almost always a data error.
EXTREME_MOVE = 0.25


@dataclass(frozen=True)
class Issue:
    symbol: str
    rule: str
    severity: str
    detail: str
    date: str | None = None

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity {self.severity!r}")


def worst(issues: Iterable[Issue]) -> str:
    """'error', 'warning' or 'ok' — the quality of a series with these issues."""
    severities = {i.severity for i in issues}
    if "error" in severities:
        return "error"
    if "warning" in severities:
        return "warning"
    return "ok"


def _capped(symbol: str, rule: str, severity: str, hits: list[tuple[str, str]]) -> list[Issue]:
    issues = [Issue(symbol, rule, severity, detail, day) for day, detail in hits[:MAX_PER_RULE]]
    if len(hits) > MAX_PER_RULE:
        issues.append(
            Issue(symbol, rule, severity, f"{len(hits)} dates in total; first {MAX_PER_RULE} listed")
        )
    return issues


_DROPPED_SEVERITY = {
    "unsettled_session": "info",
    "duplicate_identical": "info",
    "missing_close": "warning",
    "duplicate_conflicting": "error",
}


def check_series(series: PriceSeries, *, as_of: date, max_stale_days: int) -> list[Issue]:
    """Rules that need only the series itself."""
    symbol = series.symbol
    issues: list[Issue] = []

    by_reason: dict[str, list[tuple[str, str]]] = {}
    for d in series.dropped:
        by_reason.setdefault(d.reason, []).append((d.date, d.detail or d.reason))
    for reason, hits in by_reason.items():
        issues += _capped(symbol, reason, _DROPPED_SEVERITY.get(reason, "warning"), hits)

    bars = series.bars
    if bars.empty:
        issues.append(Issue(symbol, "empty_series", "error", "no settled bar with a close"))
        return issues

    o, h, low, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    adj, vol = bars["adj_close"].to_numpy(), bars["volume"].to_numpy()
    days = [str(d) for d in bars.index]

    non_positive, material, noise = [], [], []
    for i, day in enumerate(days):
        prices = [p for p in (o[i], h[i], low[i], c[i]) if not math.isnan(p)]
        if any(p <= 0 for p in prices):
            non_positive.append((day, f"non-positive price in {prices}"))
            continue
        if math.isnan(h[i]) or math.isnan(low[i]):
            continue
        others = [p for p in (o[i], c[i]) if not math.isnan(p)]
        excess = max(
            max([*others, low[i]]) - h[i],
            low[i] - min([*others, h[i]]),
            0.0,
        ) / c[i]
        if excess > OHLC_MATERIAL:
            material.append((day, f"high/low inconsistent by {excess:.2%} of close"))
        elif excess > OHLC_NOISE:
            noise.append((day, f"high/low inconsistent by {excess:.4%} of close"))
    issues += _capped(symbol, "non_positive_price", "error", non_positive)
    issues += _capped(symbol, "ohlc_inconsistent", "error", material)
    issues += _capped(symbol, "ohlc_rounding", "warning", noise)

    negative = [(d, f"volume {v}") for d, v in zip(days, vol, strict=True) if v < 0]
    zero = [(d, "volume 0") for d, v in zip(days, vol, strict=True) if v == 0]
    issues += _capped(symbol, "negative_volume", "error", negative)
    issues += _capped(symbol, "zero_volume", "warning", zero)

    missing_adj = [(d, "no adjusted close") for d, a in zip(days, adj, strict=True) if math.isnan(a)]
    issues += _capped(symbol, "missing_adj_close", "warning", missing_adj)
    above = [
        (d, f"adj_close {a} above close {x}")
        for d, a, x in zip(days, adj, c, strict=True)
        if not math.isnan(a) and a > x * (1 + 1e-4)
    ]
    issues += _capped(symbol, "adj_close_above_close", "warning", above)

    split_days = set(series.actions.loc[series.actions["kind"] == "split", "date"])
    moves = []
    for i in range(1, len(days)):
        change = c[i] / c[i - 1] - 1.0
        if abs(change) > EXTREME_MOVE and days[i] not in split_days:
            moves.append((days[i], f"close moved {change:+.1%} without a split"))
    issues += _capped(symbol, "extreme_move", "warning", moves)

    stale_by = (as_of - date.fromisoformat(days[-1])).days
    if stale_by > max_stale_days:
        issues.append(
            Issue(symbol, "stale_series", "error",
                  f"latest bar {days[-1]} is {stale_by} days before {as_of.isoformat()}", days[-1])
        )
    return issues


def check_calendar(series: Mapping[str, PriceSeries], calendar_symbol: str) -> list[Issue]:
    """Gaps and extra dates against the calendar symbol's trading days.

    A gap is a trading day, between a series' first and last bar, on which it
    has no bar. It is reported, not filled: carrying the previous close forward
    would put an old price under a new date.
    """
    calendar = series.get(calendar_symbol)
    if calendar is None or calendar.bars.empty:
        return [
            Issue(s, "calendar_unavailable", "warning",
                  f"calendar symbol {calendar_symbol} not in this snapshot; gaps unchecked")
            for s in series if s != calendar_symbol
        ]
    trading = pd.Index(calendar.bars.index)
    issues: list[Issue] = []
    for symbol, s in series.items():
        if symbol == calendar_symbol or s.bars.empty:
            continue
        own = pd.Index(s.bars.index)
        window = trading[(trading >= own[0]) & (trading <= min(own[-1], trading[-1]))]
        gaps = [(d, f"no bar on trading day {d}") for d in window.difference(own)]
        extra_days = own[(own >= trading[0]) & (own <= trading[-1])].difference(trading)
        extra = [(d, f"bar on {d}, which is no trading day of {calendar_symbol}") for d in extra_days]
        issues += _capped(symbol, "calendar_gap", "warning", gaps)
        issues += _capped(symbol, "calendar_extra", "warning", extra)
    return issues


def reference_date(fetched_on: date, requested_end: str | None) -> date:
    """The date staleness is judged against: the requested end, else the day
    before the fetch (today's session is never in a snapshot)."""
    if requested_end:
        return date.fromisoformat(requested_end)
    return fetched_on - timedelta(days=1)
