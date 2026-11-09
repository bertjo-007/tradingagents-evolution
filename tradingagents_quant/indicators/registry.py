"""The one place indicators are looked up: their inputs, defaults and column names.

Strategies (sim/strategy) and the agent bridge read indicator values through
``compute``/``values_at`` only, so every consumer sees the same numbers, and an
agent is handed computed values instead of being asked to produce them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from tradingagents_quant.indicators import core


@dataclass(frozen=True)
class IndicatorSpec:
    key: str
    func: Callable[..., pd.Series | pd.DataFrame]
    inputs: tuple[str, ...]  # bar columns, in the order the function takes them
    defaults: Mapping[str, int | float]
    description: str
    shown_to_agents: bool = False  # in the upstream agents' stockstats whitelist

    def params(self, overrides: Mapping[str, Any]) -> dict[str, int | float]:
        unknown = set(overrides) - set(self.defaults)
        if unknown:
            raise ValueError(f"{self.key} has no parameter(s) {sorted(unknown)}")
        return {**self.defaults, **overrides}

    def prefix(self, params: Mapping[str, int | float]) -> str:
        if not params:
            return self.key
        return "_".join([self.key, *(f"{params[k]:g}" for k in self.defaults)])


def _spec(key, func, inputs, defaults, description, shown=False) -> IndicatorSpec:
    return IndicatorSpec(key, func, tuple(inputs), dict(defaults), description, shown)


_SPECS = (
    _spec("sma", core.sma, ["close"], {"n": 20}, "simple moving average of the close", True),
    _spec("ema", core.ema, ["close"], {"n": 10}, "exponential moving average of the close", True),
    _spec("rsi", core.rsi, ["close"], {"n": 14}, "relative strength index, 0-100", True),
    _spec("macd", core.macd, ["close"], {"fast": 12, "slow": 26, "signal": 9},
          "MACD line, signal line, histogram", True),
    _spec("boll", core.bollinger, ["close"], {"n": 20, "k": 2.0},
          "Bollinger bands: mid, upper, lower, %B, bandwidth", True),
    _spec("atr", core.atr, ["high", "low", "close"], {"n": 14}, "average true range", True),
    _spec("vwma", core.vwma, ["high", "low", "close", "volume"], {"n": 20},
          "volume-weighted moving average of the typical price", True),
    _spec("hv", core.historical_volatility, ["close"], {"n": 20},
          "annualized volatility of log returns"),
    _spec("roc", core.roc, ["close"], {"n": 20}, "rate of change over n bars"),
    _spec("mom", core.momentum, ["close"], {"lookback": 252, "skip": 21},
          "return from lookback to skip bars ago (12-1 momentum)"),
    _spec("obv", core.obv, ["close", "volume"], {}, "on-balance volume (relative level)"),
    _spec("volratio", core.volume_ratio, ["volume"], {"n": 20}, "volume over its n-bar average"),
    _spec("dmi", core.dmi, ["high", "low", "close"], {"n": 14},
          "+DI, -DI and Wilder's ADX (trend strength)"),
    _spec("donchian", core.donchian, ["high", "low"], {"n": 20},
          "highest high / lowest low of the last n bars"),
)

REGISTRY: dict[str, IndicatorSpec] = {s.key: s for s in _SPECS}


def get(key: str) -> IndicatorSpec:
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown indicator {key!r}; known: {', '.join(sorted(REGISTRY))}") from None


def compute(bars: pd.DataFrame, key: str, **overrides: Any) -> pd.DataFrame:
    """One indicator over ``bars``, as a frame of named columns (``rsi_14``,
    ``macd_12_26_9_signal``, ...)."""
    spec = get(key)
    params = spec.params(overrides)
    missing = [c for c in spec.inputs if c not in bars.columns]
    if missing:
        raise ValueError(f"{key} needs bar column(s) {missing}")
    result = spec.func(*(bars[c] for c in spec.inputs), **params)
    prefix = spec.prefix(params)
    if isinstance(result, pd.Series):
        return result.rename(prefix).to_frame()
    return result.add_prefix(prefix + "_")


def compute_many(
    bars: pd.DataFrame, requests: Iterable[tuple[str, Mapping[str, Any]]]
) -> pd.DataFrame:
    frames = [compute(bars, key, **dict(params)) for key, params in requests]
    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=bars.index)


def values_at(frame: pd.DataFrame, when: Any) -> dict[str, float | None]:
    """Indicator values on or before ``when``: the latest row not after it.
    NaN (warm-up, missing input) becomes None, which a reader must treat as
    "not available", never as zero."""
    eligible = frame.loc[frame.index <= pd.Timestamp(when)]
    if eligible.empty:
        return dict.fromkeys(frame.columns)
    row = eligible.iloc[-1]
    return {c: (None if math.isnan(v) else float(v)) for c, v in row.items()}
