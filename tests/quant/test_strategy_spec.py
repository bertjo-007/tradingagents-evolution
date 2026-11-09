"""Strategy specifications: versioned by content, validated, described in full."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from tradingagents_quant.sim.sizing import target_weights
from tradingagents_quant.strategy.library import RULES
from tradingagents_quant.strategy.spec import ExitSpec, SizingSpec, StrategySpec

from .fixtures import synthetic_bars


def _spec(**kw):
    base = {"name": "x", "rule": "sma_cross", "params": {"fast": 20, "slow": 100},
            "symbols": ("spy",)}
    return StrategySpec(**{**base, **kw})


def test_the_version_follows_every_field():
    a = _spec()
    assert a.version_hash() == _spec().version_hash()
    assert a.symbols == ("SPY",)
    for change in ({"params": {"fast": 21, "slow": 100}},
                   {"sizing": SizingSpec(fraction=0.5)},
                   {"exits": ExitSpec(stop_loss="percent", stop_value=0.1)},
                   {"signal_prices": "price"}, {"notes": "changed"}):
        assert _spec(**change).version_hash() != a.version_hash(), change


def test_a_spec_cannot_be_changed_in_place():
    with pytest.raises(ValidationError):
        _spec().params = {"fast": 5}


@pytest.mark.parametrize(
    "kw",
    [{"rule": "magic"}, {"params": {"fast": 200, "slow": 100}}, {"params": {"window": 3}},
     {"params": {"fast": 1, "slow": 100}}, {"symbols": ()}, {"direction": "short"},
     {"exits": {"stop_loss": "percent", "stop_value": 0}},
     {"exits": {"stop_loss": "percent", "stop_value": 1.0}},
     {"sizing": {"fraction": 1.5}}],
)
def test_invalid_specs_are_refused(kw):
    with pytest.raises((ValidationError, ValueError)):
        _spec(**kw)


def test_the_description_names_everything_the_brief_asks_for():
    d = _spec(exits=ExitSpec(stop_loss="atr", stop_value=3, max_holding_bars=60),
              sizing=SizingSpec(method="vol_target", target_vol=0.12)).describe()
    for key in ("instruments", "timeframe", "entry", "exit", "position_sizing", "risk_limits",
                "required_data", "benchmark", "expected_holding_bars"):
        assert d[key], key
    assert "SMA(20)" in d["entry"] and "3 x ATR(14)" in d["exit"] and "12%" in d["position_sizing"]


def test_rules_return_a_signal_in_range_on_the_same_index():
    bars = synthetic_bars(300)
    for key, rule in RULES.items():
        signal = rule.signal(bars, {})
        assert signal.index.equals(bars.index), key
        valid = signal.dropna()
        assert ((valid >= 0) & (valid <= 1)).all(), key


def test_sma_cross_is_flat_and_unknown_during_warm_up():
    signal = RULES["sma_cross"].signal(synthetic_bars(300), {"fast": 10, "slow": 50})
    assert signal.iloc[:49].isna().all() and signal.iloc[49:].notna().all()


# -- sizing ----------------------------------------------------------------------


def _bars_with_vol(vol):
    return synthetic_bars(300, vol=vol, seed=4)


def test_fixed_fraction_scales_the_signal():
    bars = synthetic_bars(50)
    signal = pd.Series([np.nan] * 10 + [1.0] * 40, index=bars.index)
    w = target_weights(signal, bars, SizingSpec(fraction=0.6))
    assert (w.iloc[:10] == 0).all() and (w.iloc[10:] == 0.6).all()


def test_vol_target_holds_less_of_a_wilder_market_and_respects_the_cap():
    spec = SizingSpec(method="vol_target", target_vol=0.10, fraction=1.0)
    calm, wild = _bars_with_vol(0.004), _bars_with_vol(0.03)
    on = pd.Series(1.0, index=calm.index)
    w_calm = target_weights(on, calm, spec).iloc[-1]
    w_wild = target_weights(on, wild, spec).iloc[-1]
    assert w_calm == 1.0  # target / tiny vol is capped at the fraction
    assert 0.1 < w_wild < 0.3  # ~10 % / ~48 % annualized
    assert (target_weights(on, wild, spec).iloc[:20] == 0).all()  # no estimate yet: no size


def test_atr_risk_by_hand():
    idx = pd.bdate_range("2024-01-01", periods=30)
    bars = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=idx)
    spec = SizingSpec(method="atr_risk", risk_per_trade=0.01, atr_multiple=2.0, atr_window=14,
                      fraction=1.0)
    w = target_weights(pd.Series(1.0, index=idx), bars, spec)
    # ATR 2; a 2-ATR move (4) on a 100 price is 4 %; 1 % risk -> 25 % weight.
    assert w.iloc[-1] == pytest.approx(0.25)
