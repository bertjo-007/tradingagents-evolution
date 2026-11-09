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


def synthetic_bars(
    n: int, *, seed: int = 7, drift: float = 0.0003, vol: float = 0.012, start: str = "2010-01-04",
    vol_path: Sequence[float] | None = None, mean_reversion: float | None = None,
) -> pd.DataFrame:
    """A synthetic daily OHLCV random walk in the canonical bar layout.

    numpy's legacy RandomState keeps its stream stable across releases (the
    newer Generator does not promise that), so these numbers never change.
    ``vol_path`` gives a per-bar volatility instead of the constant ``vol``.
    ``mean_reversion`` (0 < phi < 1) makes the log price an AR(1) around a
    fixed level instead of a walk: a range-bound market. A driftless random
    walk is *not* one — it trends by chance, and a regime classifier rightly
    says so.
    """
    import numpy as np

    rng = np.random.RandomState(seed)
    sigma = np.asarray(vol_path, dtype=float) if vol_path is not None else np.full(n, vol)
    shocks = sigma * rng.standard_normal(n)
    if mean_reversion is None:
        log_price = np.cumsum(drift + shocks)
    else:
        log_price = np.zeros(n)
        for t in range(1, n):
            log_price[t] = mean_reversion * log_price[t - 1] + shocks[t]
    close = 100.0 * np.exp(log_price)
    # As in real bars, a session opens near the previous close (a small
    # overnight gap) and its wicks extend the open-close range.
    previous = np.concatenate([[100.0], close[:-1]])
    open_ = previous * np.exp(sigma * 0.2 * rng.standard_normal(n))
    wick = np.abs(sigma * 0.5 * rng.standard_normal((2, n)))
    high = np.maximum(open_, close) * (1 + wick[0])
    low = np.minimum(open_, close) * (1 - wick[1])
    volume = np.round(1e6 * np.exp(0.3 * rng.standard_normal(n)))
    index = pd.DatetimeIndex(pd.bdate_range(start, periods=n), name="date")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "adj_close": close,
         "volume": volume},
        index=index,
    )


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
