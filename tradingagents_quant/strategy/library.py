"""Registered trading rules. A rule maps signal bars to a signal in [0, 1] per
bar, decided at that bar's close from bars up to it.

A rule may use only causal computations (the indicator registry, shifts into
the past). ``tests/quant/test_engine_leakage.py`` runs every rule through a
truncation test: a backtest cut at day t must produce the same fills up to t.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

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


def _buy_and_hold(bars: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=bars.index)


def _sma_cross(bars: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    fast_ma, slow_ma = core.sma(bars["close"], int(fast)), core.sma(bars["close"], int(slow))
    on = (fast_ma > slow_ma).astype(float)
    return on.where(slow_ma.notna())  # NaN in warm-up: no position, and visibly so


def _fast_below_slow(p: Mapping[str, float | int]) -> str | None:
    return None if p["fast"] < p["slow"] else "fast window must be shorter than slow window"


RULES: dict[str, Rule] = {
    r.key: r
    for r in (
        Rule(
            "buy_and_hold", _buy_and_hold, {}, {},
            entry="buy at the first opportunity", exit="never (held to the end)",
            required=("close",),
        ),
        Rule(
            "sma_cross", _sma_cross, {"fast": 50, "slow": 200},
            {"fast": (2, 250), "slow": (5, 400)},
            entry="long while SMA({fast}) is above SMA({slow}) of the close",
            exit="flat when SMA({fast}) falls to or below SMA({slow})",
            required=("close",), holding=(20, 500), constraints=(_fast_below_slow,),
        ),
    )
}
