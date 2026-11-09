"""From a rule's signal in [0, 1] to a target weight of equity, per bar.

Every method reads only bars up to the one it sizes. This is the one place
position size is computed; the risk engine (M6) calls it rather than keeping a
second version of the arithmetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingagents_quant.indicators import core
from tradingagents_quant.sim.execution import CostModel
from tradingagents_quant.strategy.spec import SizingSpec


def shares_for(weight: float, equity: float, price: float, costs: CostModel) -> float:
    """Shares that make ``weight`` of ``equity`` at ``price``, allowing for the
    adverse fill. The one formula the engine and the risk engine both use."""
    if weight <= 0 or equity <= 0 or not price > 0:
        return 0.0
    return costs.round_shares(weight * equity / (price * (1.0 + costs.adverse)))


def target_weights(signal: pd.Series, market: pd.DataFrame, sizing: SizingSpec) -> pd.Series:
    """Target weight per bar. NaN signal (warm-up) is weight 0."""
    on = signal.fillna(0.0).clip(0.0, 1.0)
    if sizing.method == "fixed_fraction":
        return on * sizing.fraction
    close = market["close"]
    if sizing.method == "vol_target":
        vol = core.historical_volatility(close, sizing.vol_window)
        scale = (sizing.target_vol / vol.where(vol > 0)).clip(upper=sizing.fraction)
    else:  # atr_risk
        atr = core.atr(market["high"], market["low"], close, sizing.atr_window)
        risk_weight = sizing.risk_per_trade * close / (sizing.atr_multiple * atr.where(atr > 0))
        scale = risk_weight.clip(upper=sizing.fraction)
    # Without a volatility estimate yet there is no size: not a guess, zero.
    return (on * scale).fillna(0.0).replace([np.inf, -np.inf], 0.0)
