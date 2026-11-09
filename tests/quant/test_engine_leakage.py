"""Look-ahead, tested three ways.

1. **Truncation**: a backtest on data cut at day t makes exactly the same
   *decisions* and fills up to t as one on the full data. Comparing fills
   alone is not enough: a rule that peeks one day ahead corrupts only the
   decision at the cut, whose fill would come the morning after it, so fills
   up to t still agree. (Found by planting exactly that leak; see
   ``test_a_one_day_peek_is_caught``.)
2. **Oracle**: a strategy that is *given* tomorrow's information must look
   absurdly good. If it did not, the engine would be blurring timing in a way
   that could also hide a real leak.
3. **Delayed oracle**: the same oracle one day late must lose its edge,
   showing that the engine executes exactly when the timing model says.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents_quant.config import QuantConfig
from tradingagents_quant.sim import metrics
from tradingagents_quant.sim.engine import simulate
from tradingagents_quant.sim.execution import CostModel
from tradingagents_quant.sim.prices import market_frame
from tradingagents_quant.sim.runner import simulate_spec
from tradingagents_quant.strategy.library import RULES
from tradingagents_quant.strategy.spec import ExitSpec, SizingSpec, StrategySpec

from .fixtures import synthetic_bars

CONFIG = QuantConfig()


def _market(n=900, seed=21, drift=0.0004):
    bars = synthetic_bars(n, seed=seed, drift=drift, vol=0.012)
    dates = [d.strftime("%Y-%m-%d") for d in bars.index[::63]]
    actions = pd.DataFrame({"date": dates, "kind": "dividend", "value": 0.35})
    return market_frame(bars, actions)[0]


SPECS = [
    StrategySpec(name="bh", rule="buy_and_hold", symbols=("X",)),
    StrategySpec(name="cross", rule="sma_cross", params={"fast": 10, "slow": 40}, symbols=("X",)),
    StrategySpec(name="cross-vol", rule="sma_cross", params={"fast": 20, "slow": 60}, symbols=("X",),
                 sizing=SizingSpec(method="vol_target", target_vol=0.1, fraction=1.0)),
    StrategySpec(name="cross-stops", rule="sma_cross", params={"fast": 10, "slow": 30},
                 symbols=("X",), sizing=SizingSpec(method="atr_risk", risk_per_trade=0.01),
                 exits=ExitSpec(stop_loss="atr", stop_value=2.5, take_profit="percent",
                                take_value=0.08, max_holding_bars=40)),
]


def test_every_rule_is_covered_by_the_truncation_test():
    assert {s.rule for s in SPECS} == set(RULES) - {"leaky"}


def _truncation_differences(spec, market, cuts=(80, 150, 333, 512, 700, 899)) -> list[str]:
    full = simulate_spec(spec, "X", market, CONFIG, market.index, [])
    problems = []
    for t in cuts:
        cut_market = market.iloc[: t + 1]
        cut = simulate_spec(spec, "X", cut_market, CONFIG, cut_market.index, [])
        day = market.index[t]
        if not np.array_equal(full.target.iloc[: t + 1].to_numpy(), cut.target.to_numpy()):
            problems.append(f"decisions differ up to bar {t}")
        if [f for f in full.fills if f.date <= day] != cut.fills:
            problems.append(f"fills differ up to bar {t}")
        if not np.array_equal(full.equity.iloc[: t + 1].to_numpy(), cut.equity.to_numpy()):
            problems.append(f"equity differs up to bar {t}")
        if full.rejections[: len(cut.rejections)] != cut.rejections:
            problems.append(f"rejections differ up to bar {t}")
    return problems


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_a_backtest_cut_at_t_decides_and_fills_the_same_up_to_t(spec):
    market = _market()
    assert simulate_spec(spec, "X", market, CONFIG, market.index, []).fills, "should trade"
    assert _truncation_differences(spec, market) == []


def test_a_one_day_peek_is_caught(monkeypatch):
    """The test above must fail for the smallest real leak: tomorrow's close."""
    from tradingagents_quant.indicators import core
    from tradingagents_quant.strategy import library

    def peeks(bars, fast, slow):
        tomorrow = bars["close"].shift(-1)
        on = (core.sma(tomorrow, int(fast)) > core.sma(tomorrow, int(slow))).astype(float)
        return on.where(tomorrow.notna())

    leaky = library.Rule("leaky", peeks, {"fast": 10, "slow": 40},
                         {"fast": (2, 250), "slow": (5, 400)}, "", "", ("close",))
    monkeypatch.setitem(library.RULES, "leaky", leaky)
    spec = StrategySpec(name="leak", rule="leaky", symbols=("X",))
    problems = _truncation_differences(spec, _market())
    assert any("decisions differ" in p for p in problems)


def _sharpe(result):
    return metrics.compute(result, initial_cash=100_000.0, min_annualization_bars=252)["sharpe"]


def _oracle_weights(market: pd.DataFrame) -> pd.Series:
    # Decided at close t, a position is held from open t+1 to open t+2. The
    # oracle cheats: it knows whether that holding period will gain.
    gain = market["open"].shift(-2) > market["open"].shift(-1)
    return gain.astype(float)


def test_an_oracle_that_cheats_looks_absurdly_good():
    market = _market(n=2000, drift=0.0)
    oracle = simulate("X", market, _oracle_weights(market), costs=CostModel.free(),
                      initial_cash=100_000.0, rebalance_band=0.0)
    assert _sharpe(oracle) > 8  # no honest daily strategy gets near this


def test_the_same_oracle_one_day_late_has_no_edge():
    market = _market(n=2000, drift=0.0)
    late = _oracle_weights(market).shift(1).fillna(0.0)
    delayed = simulate("X", market, late, costs=CostModel.free(), initial_cash=100_000.0,
                       rebalance_band=0.0)
    assert abs(_sharpe(delayed)) < 1.2  # within noise of zero for ~8 years of a driftless walk


def test_costs_eat_into_a_high_turnover_strategy():
    market = _market(n=2000, drift=0.0)
    oracle = _oracle_weights(market)
    free = simulate("X", market, oracle, costs=CostModel.free(), initial_cash=100_000.0,
                    rebalance_band=0.0)
    costly = simulate("X", market, oracle, costs=CostModel(spread_bps=10, slippage_bps=10),
                      initial_cash=100_000.0, rebalance_band=0.0)
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]
    assert costly.commissions + costly.friction > 0
