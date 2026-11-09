"""Prices the simulator trades at, and prices strategies decide on.

- **Execution** uses quoted prices: ``open high low close`` as stored
  (split-adjusted, not dividend-adjusted). A holder on an ex-date receives the
  dividend in cash.
- **Signals** may use a total-return index instead, so an ex-date drop is not
  mistaken for a sell-off. It is built forward from the first bar —
  ``tr[t] = tr[t-1] * (close[t] + dividend[t]) / close[t-1]`` — and therefore
  reads no later bar. (Back-adjusting from the latest bar, as Yahoo's
  ``adj_close`` does, would let a dividend paid in 2020 change a 2010 price:
  look-ahead.) Yahoo's ``adj_close`` is not used at all; it also differs between
  fetches (see docs/DATEN.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def market_frame(bars: pd.DataFrame, actions: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Quoted OHLCV plus ``dividend`` (cash per share on its ex-date) and the
    total-return columns ``tr_open tr_high tr_low tr_close``. Returns the frame
    and notes about anything that had to be moved."""
    frame = bars[["open", "high", "low", "close", "volume"]].astype(float).copy()
    notes: list[str] = []
    dividend = pd.Series(0.0, index=frame.index)
    if not actions.empty:
        divs = actions[actions["kind"] == "dividend"]
        for day, value in zip(pd.to_datetime(divs["date"]), divs["value"], strict=True):
            if day in dividend.index:
                dividend[day] += float(value)
                continue
            later = dividend.index[dividend.index > day]
            if len(later):
                dividend[later[0]] += float(value)
                notes.append(f"dividend of {day.date()} is on no bar; credited on {later[0].date()}")
            elif day > dividend.index[-1]:
                pass  # after the data: not part of this window
    frame["dividend"] = dividend

    close = frame["close"]
    growth = (close + dividend) / close.shift(1)
    growth.iloc[0] = 1.0
    tr_close = close.iloc[0] * growth.cumprod()
    factor = tr_close / close
    for col in ("open", "high", "low"):
        frame[f"tr_{col}"] = frame[col] * factor
    frame["tr_close"] = tr_close
    return frame, notes


def signal_frame(market: pd.DataFrame, basis: str) -> pd.DataFrame:
    """The bars a rule sees: total-return or quoted prices, same column names."""
    if basis == "price":
        return market[["open", "high", "low", "close", "volume"]].copy()
    if basis != "total_return":
        raise ValueError(f"unknown signal price basis {basis!r}")
    out = market[["tr_open", "tr_high", "tr_low", "tr_close", "volume"]].copy()
    out.columns = ["open", "high", "low", "close", "volume"]
    return out


def daily_returns(equity: pd.Series) -> pd.Series:
    returns = equity / equity.shift(1) - 1.0
    return returns.iloc[1:].replace([np.inf, -np.inf], np.nan)
