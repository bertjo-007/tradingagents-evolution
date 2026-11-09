"""Each rule does what its description says, on data built to show it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents_quant.strategy.library import RULES, hold_between
from tradingagents_quant.strategy.spec import StrategySpec

from .fixtures import synthetic_bars


def _bars(closes, spread=0.005):
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range("2020-01-01", periods=len(closes))
    return pd.DataFrame({"open": closes, "high": closes * (1 + spread), "low": closes * (1 - spread),
                         "close": closes, "volume": 1e6}, index=idx)


def _signal(key, bars, **params):
    return RULES[key].signal(bars, params)


def test_hold_between_enters_stays_and_leaves():
    idx = pd.RangeIndex(7)
    enter = pd.Series([0, 1, 0, 0, 1, 0, 1], index=idx).astype(bool)
    leave = pd.Series([0, 0, 0, 1, 0, 1, 0], index=idx).astype(bool)
    known = pd.Series([0, 1, 1, 1, 1, 1, 1], index=idx).astype(bool)
    out = hold_between(enter, leave, known)
    assert np.isnan(out.iloc[0])
    assert list(out.iloc[1:]) == [1, 1, 0, 1, 0, 1]


def test_trend_following_is_long_above_the_average_only():
    closes = [100.0] * 30 + [110.0] * 10 + [90.0] * 10
    s = _signal("trend_following", _bars(closes), window=20)
    assert s.iloc[:19].isna().all()
    assert s.iloc[29] == 0.0  # flat price equals its average: not above
    assert s.iloc[32] == 1.0 and s.iloc[45] == 0.0


def test_momentum_follows_the_sign_of_the_skipped_return():
    up = _signal("momentum", _bars(np.linspace(100, 150, 80)), lookback=40, skip=5)
    down = _signal("momentum", _bars(np.linspace(150, 100, 80)), lookback=40, skip=5)
    assert up.iloc[:40].isna().all() and (up.iloc[40:] == 1).all()
    assert (down.iloc[40:] == 0).all()


def test_mean_reversion_buys_a_dip_in_an_uptrend_and_sells_the_recovery():
    closes = list(np.linspace(100, 130, 60)) + [124, 119, 116] + list(np.linspace(118, 132, 10))
    s = _signal("mean_reversion", _bars(closes), rsi_window=2, entry=10, exit=70, trend_window=50)
    dip = 62  # third down day
    assert s.iloc[dip] == 1.0
    recovered = next(i for i in range(dip + 1, len(closes)) if s.iloc[i] == 0.0)
    assert closes[recovered] > closes[dip]


def test_mean_reversion_ignores_a_dip_below_the_trend():
    closes = list(np.linspace(130, 100, 60)) + [96, 92, 90] + [91] * 5
    s = _signal("mean_reversion", _bars(closes), rsi_window=2, entry=10, exit=70, trend_window=50)
    assert (s.dropna() == 0).all()


def test_breakout_enters_on_a_new_high_and_exits_on_a_new_low():
    closes = [100.0] * 30 + [101, 103, 105, 107] + [106] * 5 + [95, 94]
    s = _signal("breakout", _bars(closes, spread=0.001), entry_window=20, exit_window=10)
    assert s.iloc[29] == 0.0
    assert s.iloc[30] == 1.0  # 101 > the previous 20 highs (100.1)
    assert s.iloc[38] == 1.0  # holding through the pause
    assert s.iloc[39] == 0.0  # 95 < the previous 10 lows


def test_volatility_filter_steps_aside_when_volatility_rises():
    calm = [0.004] * 300
    bars = synthetic_bars(340, vol_path=calm + [0.03] * 40, seed=8)
    s = _signal("volatility_filter", bars, window=20, lookback=252, quantile=0.5)
    assert s.iloc[-1] == 0.0
    assert s.iloc[:271].isna().all()


def test_regime_switch_holds_only_in_up_regimes():
    up = _signal("regime_switch", synthetic_bars(500, drift=0.003, vol=0.006), avoid_high_vol=0)
    down = _signal("regime_switch", synthetic_bars(500, drift=-0.003, vol=0.006), avoid_high_vol=0)
    assert up.iloc[-50:].mean() > 0.9 and down.iloc[-50:].mean() == 0.0


def test_cash_never_holds_and_buy_and_hold_always_does():
    bars = synthetic_bars(50)
    assert (_signal("cash", bars) == 0).all() and (_signal("buy_and_hold", bars) == 1).all()


@pytest.mark.parametrize(
    ("key", "params"),
    [("breakout", {"entry_window": 20, "exit_window": 20}),
     ("mean_reversion", {"entry": 50, "exit": 40}),
     ("momentum", {"lookback": 20, "skip": 20}),
     ("regime_switch", {"avoid_high_vol": 2})],
)
def test_contradictory_parameters_are_refused(key, params):
    with pytest.raises(ValueError):
        StrategySpec(name="x", rule=key, params=params, symbols=("SPY",))


def test_every_rule_describes_itself_completely():
    for key, rule in RULES.items():
        spec = StrategySpec(name=key, rule=key, symbols=("SPY",))
        described = spec.describe()
        assert described["entry"] and described["exit"], key
        assert rule.family, key
