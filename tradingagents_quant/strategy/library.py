"""Registered trading rules. A rule maps signal bars to a signal in [0, 1] per
bar, decided at that bar's close from bars up to it.

A rule may use only causal computations (the indicator registry, shifts into
the past, forward loops). ``tests/quant/test_engine_leakage.py`` runs every rule
through a truncation test: a backtest cut at day t must make the same decisions
and fills up to t. A rule added here without being added there fails the suite.

Families, as the brief lists them: trend following, moving-average crossover,
momentum, mean reversion, volatility, breakout, regime-dependent — plus
buy-and-hold and cash as baselines.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradingagents_quant.indicators import core


@dataclass(frozen=True)
class Rule:
    key: str
    func: Callable[..., pd.Series]
    defaults: Mapping[str, float | int]
    bounds: Mapping[str, tuple[float, float]]
    entry: str  # str.format over the resolved parameters
    exit: str
    required: tuple[str, ...]
    holding: tuple[int, int] | None = None
    family: str = ""
    constraints: tuple[Callable[[Mapping[str, float | int]], str | None], ...] = field(default=())

    def check_params(self, params: Mapping[str, float | int]) -> None:
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"rule {self.key} has no parameter(s) {sorted(unknown)}")
        resolved = {**self.defaults, **params}
        for name, value in resolved.items():
            low, high = self.bounds[name]
            if not low <= value <= high:
                raise ValueError(f"{self.key}.{name}={value} outside [{low}, {high}]")
        for constraint in self.constraints:
            problem = constraint(resolved)
            if problem:
                raise ValueError(f"{self.key}: {problem}")

    def signal(self, bars: pd.DataFrame, params: Mapping[str, float | int]) -> pd.Series:
        self.check_params(params)
        resolved = {**self.defaults, **params}
        out = self.func(bars, **resolved).astype(float)
        if not out.index.equals(bars.index):
            raise ValueError(f"rule {self.key} returned a signal on a different index")
        bad = out.dropna()
        if ((bad < 0) | (bad > 1)).any():
            raise ValueError(f"rule {self.key} returned a signal outside [0, 1]")
        return out


def hold_between(enter: pd.Series, leave: pd.Series, known: pd.Series) -> pd.Series:
    """1 from a bar where ``enter`` holds until a bar where ``leave`` holds.

    A forward loop over the bars, so a bar's state depends only on earlier
    bars. NaN where the inputs are not known yet (warm-up); the state starts
    flat after it.
    """
    e, lv, k = enter.to_numpy(dtype=bool), leave.to_numpy(dtype=bool), known.to_numpy(dtype=bool)
    out = np.full(len(e), np.nan)
    state = 0.0
    for i in range(len(e)):
        if not k[i]:
            continue
        if state == 0.0 and e[i]:
            state = 1.0
        elif state == 1.0 and lv[i]:
            state = 0.0
        out[i] = state
    return pd.Series(out, index=enter.index)


# -- rules ---------------------------------------------------------------------


def _buy_and_hold(bars: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=bars.index)


def _cash(bars: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=bars.index)


def _sma_cross(bars: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    fast_ma, slow_ma = core.sma(bars["close"], int(fast)), core.sma(bars["close"], int(slow))
    on = (fast_ma > slow_ma).astype(float)
    return on.where(slow_ma.notna())  # NaN in warm-up: no position, and visibly so


def _trend_following(bars: pd.DataFrame, window: int) -> pd.Series:
    ma = core.sma(bars["close"], int(window))
    return (bars["close"] > ma).astype(float).where(ma.notna())


def _momentum(bars: pd.DataFrame, lookback: int, skip: int, threshold: float) -> pd.Series:
    mom = core.momentum(bars["close"], int(lookback), int(skip))
    return (mom > threshold).astype(float).where(mom.notna())


def _mean_reversion(bars: pd.DataFrame, rsi_window: int, entry: float, exit: float,
                    trend_window: int) -> pd.Series:
    close = bars["close"]
    rsi = core.rsi(close, int(rsi_window))
    known = rsi.notna()
    enter = rsi < entry
    if trend_window > 0:
        # Buy dips only in an uptrend: below its long average a falling price
        # is more often a trend than a dip.
        trend = core.sma(close, int(trend_window))
        known &= trend.notna()
        enter &= close > trend
    return hold_between(enter, rsi > exit, known)


def _breakout(bars: pd.DataFrame, entry_window: int, exit_window: int) -> pd.Series:
    # Yesterday's channels: today's high must not confirm itself.
    upper = core.donchian(bars["high"], bars["low"], int(entry_window))["upper"].shift(1)
    lower = core.donchian(bars["high"], bars["low"], int(exit_window))["lower"].shift(1)
    known = upper.notna() & lower.notna()
    return hold_between(bars["close"] > upper, bars["close"] < lower, known)


def _volatility_filter(bars: pd.DataFrame, window: int, lookback: int, quantile: float) -> pd.Series:
    hv = core.historical_volatility(bars["close"], int(window))
    threshold = hv.rolling(int(lookback), min_periods=int(lookback)).quantile(quantile)
    return (hv <= threshold).astype(float).where(threshold.notna())


def _regime_switch(bars: pd.DataFrame, avoid_high_vol: int) -> pd.Series:
    from tradingagents_quant import regime

    labels = regime.classify(bars)
    known = labels["trend"] != "unknown"
    on = labels["trend"].isin(["up", "strong_up"])
    if avoid_high_vol:
        known &= labels["volatility"] != "unknown"
        on &= labels["volatility"] != "high"
    return on.astype(float).where(known)


def _fast_below_slow(p: Mapping[str, float | int]) -> str | None:
    return None if p["fast"] < p["slow"] else "fast window must be shorter than slow window"


def _exit_shorter(p: Mapping[str, float | int]) -> str | None:
    return None if p["exit_window"] < p["entry_window"] else "exit window must be shorter than entry"


def _exit_above_entry(p: Mapping[str, float | int]) -> str | None:
    return None if p["exit"] > p["entry"] else "RSI exit level must lie above the entry level"


def _skip_below_lookback(p: Mapping[str, float | int]) -> str | None:
    return None if p["skip"] < p["lookback"] else "skip must be shorter than lookback"


RULES: dict[str, Rule] = {
    r.key: r
    for r in (
        Rule("buy_and_hold", _buy_and_hold, {}, {}, family="baseline",
             entry="buy at the first opportunity", exit="never (held to the end)",
             required=("close",)),
        Rule("cash", _cash, {}, {}, family="baseline",
             entry="never", exit="never (zero-return baseline; idle cash earns no interest)",
             required=("close",)),
        Rule("sma_cross", _sma_cross, {"fast": 50, "slow": 200},
             {"fast": (2, 250), "slow": (5, 400)}, family="moving-average crossover",
             entry="long while SMA({fast}) is above SMA({slow}) of the close",
             exit="flat when SMA({fast}) falls to or below SMA({slow})",
             required=("close",), holding=(20, 500), constraints=(_fast_below_slow,)),
        Rule("trend_following", _trend_following, {"window": 200}, {"window": (10, 400)},
             family="trend following",
             entry="long while the close is above its SMA({window})",
             exit="flat when the close falls to or below its SMA({window})",
             required=("close",), holding=(5, 250)),
        Rule("momentum", _momentum, {"lookback": 252, "skip": 21, "threshold": 0.0},
             {"lookback": (20, 504), "skip": (0, 63), "threshold": (-0.5, 0.5)},
             family="momentum",
             entry="long while the return from {lookback} to {skip} bars ago exceeds {threshold:.1%}",
             exit="flat when that return falls to or below {threshold:.1%}",
             required=("close",), holding=(20, 250), constraints=(_skip_below_lookback,)),
        Rule("mean_reversion", _mean_reversion,
             {"rsi_window": 2, "entry": 10.0, "exit": 70.0, "trend_window": 200},
             {"rsi_window": (2, 30), "entry": (1, 50), "exit": (30, 99), "trend_window": (0, 400)},
             family="mean reversion",
             entry="buy when RSI({rsi_window}) drops below {entry:g} while the close is above its "
                   "SMA({trend_window}) (no trend filter if 0)",
             exit="sell when RSI({rsi_window}) rises above {exit:g}",
             required=("close",), holding=(1, 20), constraints=(_exit_above_entry,)),
        Rule("breakout", _breakout, {"entry_window": 55, "exit_window": 20},
             {"entry_window": (5, 252), "exit_window": (2, 126)}, family="breakout",
             entry="buy when the close exceeds the highest high of the previous {entry_window} bars",
             exit="sell when the close falls below the lowest low of the previous {exit_window} bars",
             required=("high", "low", "close"), holding=(10, 200), constraints=(_exit_shorter,)),
        Rule("volatility_filter", _volatility_filter,
             {"window": 20, "lookback": 252, "quantile": 0.5},
             {"window": (5, 126), "lookback": (63, 1260), "quantile": (0.05, 0.95)},
             family="volatility",
             entry="long while {window}-bar volatility is at or below its {quantile:.0%} quantile "
                   "of the last {lookback} bars",
             exit="flat when volatility rises above that quantile",
             required=("close",), holding=(5, 120)),
        Rule("regime_switch", _regime_switch, {"avoid_high_vol": 0}, {"avoid_high_vol": (0, 1)},
             family="regime-dependent",
             entry="long while the trend regime is up or strong_up "
                   "(and, if avoid_high_vol=1, volatility is not high)",
             exit="flat in any other regime",
             required=("high", "low", "close"), holding=(10, 250)),
    )
}
