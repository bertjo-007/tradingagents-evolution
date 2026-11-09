"""Market regime per bar, from past data only.

Two independent axes, because a strategy may care about one and not the other:

- **trend**: ``strong_up``, ``up``, ``strong_down``, ``down``, ``sideways``,
  ``uncertain``.

  *Structure* says whether there is a trend: the price on one side of its
  200-bar average and the 50-bar average at least ``min_separation`` daily
  standard deviations away from the 200 on the same side. Measured in the
  series' own volatility, so one threshold serves a bond ETF and a tech ETF.

  *Strength* (Wilder's ADX, direction from the DI lines) says how forceful it
  is: a structured trend with ADX >= ``adx_strong`` is ``strong_up``/
  ``strong_down``, otherwise ``up``/``down``. Without structure, a low ADX is
  ``sideways``; anything else is ``uncertain``.

  Why not ADX alone: a quiet, steady advance has small daily ranges and so a
  low ADX. On SPY, ADX alone called 2017 (+18.5 %, one of the calmest bull
  years) "sideways" on 56 % of its days. Low ADX means weak directional
  pressure, not the absence of a trend.
- **volatility**: ``high``, ``normal``, ``low`` — today's 20-bar volatility as
  a percentile of its own last 252 values, so a bond ETF and a tech ETF are
  each judged against themselves.

Before enough history exists the label is ``unknown``: the classifier does not
guess. Every backtest and simulated trade records the regime it ran in, so a
strategy's result can be split by regime instead of averaged over them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from tradingagents_quant.indicators import core

TRENDS = ("strong_up", "up", "strong_down", "down", "sideways", "uncertain", "unknown")
VOLATILITIES = ("high", "normal", "low", "unknown")


@dataclass(frozen=True)
class RegimeRules:
    long_window: int = 200
    short_window: int = 50
    adx_window: int = 14
    adx_strong: float = 25.0  # at or above: a forceful trend
    adx_flat: float = 20.0  # below, without structure: sideways
    # Distance of the short from the long average, in daily standard deviations
    # of the close, from which the averages describe a trend rather than noise.
    min_separation: float = 2.0
    vol_window: int = 20
    vol_lookback: int = 252
    vol_high: float = 0.80  # percentile of the own past at or above: high
    vol_low: float = 0.20  # at or below: low

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @property
    def warmup(self) -> int:
        """Bars before both axes can be classified."""
        return max(self.long_window, 2 * self.adx_window, self.vol_window + self.vol_lookback)


def classify(bars: pd.DataFrame, rules: RegimeRules | None = None) -> pd.DataFrame:
    """Columns ``trend``, ``volatility``, ``regime`` (``trend/volatility``)."""
    rules = rules or RegimeRules()
    close, high, low = bars["close"], bars["high"], bars["low"]

    long_ma = core.sma(close, rules.long_window)
    short_ma = core.sma(close, rules.short_window)
    dmi = core.dmi(high, low, close, rules.adx_window)
    adx, pdi, mdi = dmi["adx"], dmi["plus_di"], dmi["minus_di"]

    hv = core.historical_volatility(close, rules.vol_window)
    daily_vol = hv / np.sqrt(252)
    separation = ((short_ma - long_ma) / long_ma) / daily_vol.where(daily_vol > 0)

    known = long_ma.notna() & short_ma.notna() & adx.notna() & separation.notna()
    rising = (close > long_ma) & (separation >= rules.min_separation)
    falling = (close < long_ma) & (separation <= -rules.min_separation)
    forceful = adx >= rules.adx_strong
    trend = np.select(
        [
            ~known,
            rising & forceful & (pdi > mdi),
            rising,
            falling & forceful & (mdi > pdi),
            falling,
            adx < rules.adx_flat,
        ],
        ["unknown", "strong_up", "up", "strong_down", "down", "sideways"],
        default="uncertain",
    )

    # Percentile of today's value among the last vol_lookback values, today
    # included: rank(pct=True) over a trailing window reads no later bar.
    pct = hv.rolling(rules.vol_lookback, min_periods=rules.vol_lookback).rank(pct=True)
    volatility = np.select(
        [pct.isna(), pct >= rules.vol_high, pct <= rules.vol_low],
        ["unknown", "high", "low"], default="normal",
    )

    out = pd.DataFrame({"trend": trend, "volatility": volatility}, index=bars.index)
    out["regime"] = out["trend"] + "/" + out["volatility"]
    return out


def summarize(labels: pd.DataFrame) -> dict[str, float]:
    """Share of bars per combined regime, unknown included."""
    return labels["regime"].value_counts(normalize=True).sort_index().to_dict()
