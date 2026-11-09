"""The risk engine: one refusal or cut per limit, unknown risk refused, never
an order made larger, and the same share arithmetic as the engine."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from tradingagents_quant.config import RiskSettings
from tradingagents_quant.risk.engine import MarketHistory, RiskEngine
from tradingagents_quant.sim.engine import Order, OrderContext, simulate, simulate_portfolio
from tradingagents_quant.sim.execution import CostModel
from tradingagents_quant.sim.sizing import shares_for

from .fixtures import synthetic_bars

FREE = CostModel.free()
BARS = synthetic_bars(300, seed=21, vol=0.01)
BARS["volume"] = 1e6
DAY = BARS.index[-1]
PRICE = float(BARS["close"].iloc[-1])


def engine(limits=None, markets=None, costs=FREE):
    markets = markets or {"A": BARS}
    return RiskEngine(limits or RiskSettings(), MarketHistory.from_markets(markets), costs)


def ctx(shares=100.0, *, symbol="A", equity=1e6, cash=1e6, peak=None, prev=None, holdings=None,
        stop=None, reason="entry", price=PRICE, day=DAY):
    return OrderContext(day, symbol, Order("buy", shares, reason, day.strftime("%Y-%m-%d"), 1.0),
                        price, equity, cash, peak or equity, prev or equity, holdings or {}, stop)


def by_name(verdict):
    return {c.name: c for c in verdict.checks}


def test_an_ordinary_order_passes_unchanged():
    v = engine().evaluate(ctx(100))
    assert v.accepted and v.approved == 100 and v.reasons == ()
    assert all(c.passed for c in v.checks)


def test_a_position_above_the_weight_limit_is_shrunk_to_it():
    limits = RiskSettings(max_position_weight=0.10)
    v = engine(limits).evaluate(ctx(10_000))  # far above 10 % of 1e6
    assert v.approved == np.floor(0.10 * 1e6 / PRICE)
    assert any("position_weight" in r for r in v.reasons)


def test_gross_exposure_counts_what_is_already_held():
    other = synthetic_bars(300, seed=5)
    other["volume"] = 1e6
    limits = RiskSettings(max_gross_exposure=0.5, max_position_weight=1.0,
                          correlation_threshold=0.99)
    held = {"B": (400_000 / float(other["close"].iloc[-1]), float(other["close"].iloc[-1]))}
    v = engine(limits, {"A": BARS, "B": other}).evaluate(ctx(5_000, holdings=held))
    assert v.approved == np.floor(100_000 / PRICE)  # 0.5 x 1e6 - 400 000


def test_a_new_position_beyond_the_count_is_refused():
    markets = dict.fromkeys("ABC", BARS)
    holdings = {"B": (1.0, PRICE), "C": (1.0, PRICE)}
    limits = RiskSettings(max_open_positions=2, correlation_threshold=1.0)
    v = engine(limits, markets).evaluate(ctx(10, holdings=holdings))
    assert not v.accepted and not by_name(v)["open_positions"].passed


@pytest.mark.parametrize(("peak", "prev", "check"), [(1.5e6, 1e6, "drawdown"), (1e6, 1.1e6, "daily_loss")])
def test_an_account_in_trouble_takes_no_new_risk(peak, prev, check):
    v = engine().evaluate(ctx(10, peak=peak, prev=prev))
    assert not v.accepted and not by_name(v)[check].passed


def test_a_too_volatile_instrument_is_refused():
    wild = synthetic_bars(300, seed=3, vol=0.08)
    wild["volume"] = 1e6
    v = engine(markets={"A": wild}).evaluate(ctx(10, price=float(wild["close"].iloc[-1])))
    assert not v.accepted and not by_name(v)["volatility"].passed


def test_an_illiquid_instrument_is_refused_and_a_big_order_cut_to_the_volume():
    thin = BARS.copy()
    thin["volume"] = 100.0  # ~10 000 USD a day
    assert not engine(markets={"A": thin}).evaluate(ctx(10)).accepted
    # An account large enough that the market, not the account, is the limit;
    # a lone position is its own correlated group, so that cap is lifted too.
    v = engine(RiskSettings(max_position_weight=1.0, max_correlated_weight=1.0)).evaluate(
        ctx(50_000, equity=1e8, cash=1e8))
    adv = float((BARS["close"] * BARS["volume"]).tail(20).mean())
    assert v.approved == np.floor(0.01 * adv / PRICE)


def test_correlated_positions_share_one_limit():
    twin = BARS.copy()  # perfectly correlated with A
    held = {"T": (300_000 / PRICE, PRICE)}
    limits = RiskSettings(max_correlated_weight=0.4, max_position_weight=1.0)
    v = engine(limits, {"A": BARS, "T": twin}).evaluate(ctx(5_000, holdings=held))
    assert v.approved == np.floor(100_000 / PRICE)
    assert by_name(v)["correlation"].value > 0.4


def test_risk_that_cannot_be_computed_is_refused():
    short = BARS.iloc[:10]  # no 20-bar volatility, no average volume yet
    v = engine(markets={"A": short}).evaluate(ctx(10, day=short.index[-1],
                                                  price=float(short["close"].iloc[-1])))
    assert not v.accepted
    assert not by_name(v)["volatility"].passed and not by_name(v)["liquidity"].passed
    unknown = engine().evaluate(ctx(10, symbol="NOPE"))
    assert not unknown.accepted
    assert not engine().evaluate(ctx(10, price=float("nan"))).accepted


def test_a_correlation_that_cannot_be_computed_is_refused():
    young = BARS.iloc[-30:]  # 30 bars: no 60-bar correlation with it
    v = engine(markets={"A": BARS, "Y": young}).evaluate(ctx(10, holdings={"Y": (1.0, PRICE)}))
    assert not v.accepted and not by_name(v)["correlation"].passed


def test_stops():
    assert not engine(RiskSettings(require_stop=True)).evaluate(ctx(10)).accepted
    assert engine(RiskSettings(require_stop=True)).evaluate(ctx(10, stop=PRICE * 0.9)).accepted
    assert not engine().evaluate(ctx(10, stop=PRICE * 1.01)).accepted  # stop above the price
    # 2 % of 1e6 at risk, 10 per share to the stop: at most 2 000 shares.
    v = engine(RiskSettings(max_position_weight=1.0, max_adv_fraction=1.0)).evaluate(
        ctx(5_000, stop=PRICE - 10))
    assert v.approved == 2_000


def test_the_engine_never_enlarges_an_order_and_its_verdict_cannot_be_changed():
    v = engine().evaluate(ctx(3))
    assert v.approved == 3
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.approved = 10**6  # type: ignore[misc]


def test_the_engine_has_no_door_for_an_outside_assessment():
    import inspect

    params = set(inspect.signature(RiskEngine.evaluate).parameters)
    assert params == {"self", "ctx"}  # no confidence, no override, no agent opinion


def test_sizing_is_the_engines_arithmetic():
    costs = CostModel(spread_bps=4, slippage_bps=2)
    risk = engine(costs=costs)
    for w, eq, px in [(0.3, 1e5, 123.45), (1.0, 5e4, 9.99), (0.07, 2e6, 431.0)]:
        assert risk.size(w, eq, px) == shares_for(w, eq, px, costs)


def test_in_a_run_the_filter_holds_the_position_limit():
    market = BARS.copy()
    market["dividend"] = 0.0
    limits = RiskSettings(max_position_weight=0.3)
    risk = engine(limits, {"A": market})
    target = pd.Series(1.0, index=market.index)
    r = simulate("A", market, target, costs=FREE, initial_cash=1e6, order_filter=risk.as_filter())
    first = r.fills[0]
    # The limit holds at the decision price; a gap at the next open moves the
    # filled value with it, as it moves the cash a buy needs.
    decided = market.index[market.index.get_loc(first.date) - 1]
    assert first.shares * market.loc[decided, "close"] <= 0.3 * 1e6
    assert any(x.reason == "risk_reduced" for x in r.rejections)


def test_an_unfiltered_run_orders_exactly_what_the_risk_engine_would_size():
    market = BARS.copy()
    market["dividend"] = 0.0
    target = pd.Series(np.where(np.arange(300) % 40 < 20, 0.6, 0.0), index=market.index)
    costs = CostModel(spread_bps=4, slippage_bps=2)
    r = simulate_portfolio({"A": market}, {"A": target}, costs=costs, initial_cash=1e6)
    risk = engine(costs=costs)
    entries = [f for f in r.fills if f.reason == "entry"]
    assert entries
    for f in entries:
        decided = market.index[market.index.get_loc(f.date) - 1]
        equity = float(r.equity.loc[decided])
        assert f.shares <= risk.size(0.6, equity, float(market.loc[decided, "close"]))
