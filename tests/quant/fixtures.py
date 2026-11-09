"""Deterministic test data shaped like a yfinance ``history(auto_adjust=False,
actions=True)`` frame. Clearly synthetic: never used outside tests."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import pandas as pd

from tradingagents_quant.data.ingest import RawFetch

# A fetch moment well after every fixture date, so no bar counts as unsettled.
FETCHED_AT = datetime(2030, 1, 1, 12, 0, tzinfo=UTC)


def trading_days(start: str, n: int) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=n)]


def yahoo_frame(
    dates: Sequence[str],
    closes: Sequence[float],
    *,
    tz: str = "America/New_York",
    spread: float = 0.01,
    volume: float = 1_000_000.0,
    dividends: dict[str, float] | None = None,
    splits: dict[str, float] | None = None,
) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(d).tz_localize(tz) for d in dates], name="Date")
    closes = [float(c) for c in closes]
    frame = pd.DataFrame(
        {
            "Open": closes,
            "High": [c * (1 + spread) for c in closes],
            "Low": [c * (1 - spread) for c in closes],
            "Close": closes,
            "Adj Close": [c * 0.98 for c in closes],
            "Volume": [volume] * len(closes),
            "Dividends": [0.0] * len(closes),
            "Stock Splits": [0.0] * len(closes),
        },
        index=index,
    )
    for day, value in (dividends or {}).items():
        frame.loc[pd.Timestamp(day).tz_localize(tz), "Dividends"] = value
    for day, value in (splits or {}).items():
        frame.loc[pd.Timestamp(day).tz_localize(tz), "Stock Splits"] = value
    return frame


def linear_closes(n: int, start: float = 100.0, step: float = 0.5) -> list[float]:
    return [start + step * i for i in range(n)]


class FakeProvider:
    """A fetcher over fixed frames; a symbol mapped to an exception raises it."""

    def __init__(self, frames: dict[str, pd.DataFrame | Exception]) -> None:
        self.frames = frames
        self.calls: list[tuple[str, str, str | None]] = []

    def __call__(self, symbol: str, start: str, end: str | None) -> RawFetch:
        self.calls.append((symbol, start, end))
        item = self.frames[symbol]
        if isinstance(item, Exception):
            raise item
        return RawFetch(symbol, item, "USD")
