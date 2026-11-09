"""The simulation loop in closed form: every expected number is worked out by
hand from the bars below, not read off the code."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents_quant.sim.engine import simulate
from tradingagents_quant.sim.execution import CostModel
from tradingagents_quant.strategy.spec import ExitSpec

from .fixtures import synthetic_bars

FREE = CostModel.free()


def market(rows, dividends=None):
    """rows: (open, high, low, close) per bar."""
    index = pd.bdate_range("2024-01-01", periods=len(rows))
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index, dtype=float)
    frame["volume"] = 1e6
    frame["dividend"] = 0.0
    for i, value in (dividends or {}).items():
        frame.iloc[i, frame.columns.get_loc("dividend")] = value
    return frame


def run(mkt, weights, *, costs=FREE, cash=1000.0, **kw):
    return simulate("TEST", mkt, pd.Series(weights, index=mkt.index, dtype=float), costs=costs,
                    initial_cash=cash, **kw)


FLAT = [(10, 10, 10, 10)] * 3


def test_a_round_trip_without_costs():
    mkt = market([(10, 10, 10, 10), (10, 11, 9, 11), (11, 12, 10, 12), (12, 12, 11, 12),
                  (12, 13, 12, 13), (13, 13, 13, 13)])
    r = run(mkt, [1, 1, 1, 0, 0, 0])
    # Decided at close 0 (100 shares at 10), filled at open 1; decided flat at
    # close 3, sold at open 4 for 12.
    assert [(f.date, f.side, f.shares, f.price) for f in r.fills] == [
        (mkt.index[1], "buy", 100, 10.0), (mkt.index[4], "sell", 100, 12.0)]
    assert list(r.equity) == [1000, 1100, 1200, 1200, 1200, 1200]
    assert r.trades[0].pnl == 200 and r.trades[0].holding_bars == 3
    assert r.trades[0].exit_reason == "exit_signal"


def test_a_round_trip_with_spread_slippage_and_commission():
    mkt = market([(10, 10, 10, 10), (10, 10, 10, 10), (12, 12, 12, 12), (12, 12, 12, 12)])
    costs = CostModel(commission_per_share=0.01, commission_min=1.0, commission_max_pct=0.01,
                      spread_bps=10, slippage_bps=5)  # adverse 0.1 %
    r = run(mkt, [1, 0, 0, 0], costs=costs)
    # Size at close 0: floor(1000 / (10 x 1.001)) = 99. Buy at 10.01, fee 1.
    # cash 1000 - 990.99 - 1 = 8.01. Sell at open 2: 12 x 0.999 = 11.988, fee 1.
    buy, sell = r.fills
    assert (buy.shares, buy.price, buy.commission) == (99, pytest.approx(10.01), 1.0)
    assert sell.price == pytest.approx(11.988)
    assert r.equity.iloc[-1] == pytest.approx(8.01 + 99 * 11.988 - 1.0)
    assert r.commissions == 2.0
    assert r.friction == pytest.approx(99 * 0.01 + 99 * 0.012)


def test_the_dividend_goes_to_whoever_held_at_the_previous_close():
    mkt = market([(10, 10, 10, 10)] * 5, dividends={2: 0.5, 4: 0.25})
    r = run(mkt, [1, 1, 1, 0, 0])
    # 100 shares held from the open of bar 1, sold at the open of bar 4. Both
    # ex-dates (bars 2 and 4) find them held at the previous close, so both
    # dividends are theirs; selling on the ex-date's open does not forfeit it.
    assert r.dividends == pytest.approx(100 * 0.5 + 100 * 0.25)
    late = market([(10, 10, 10, 10)] * 3, dividends={1: 0.5})
    assert run(late, [1, 1, 1]).dividends == 0.0  # bought at the ex-date's open: not entitled


def test_a_stop_inside_the_bar_fills_at_the_stop():
    mkt = market([(10, 10, 10, 10), (10, 10, 10, 10), (9.5, 9.8, 8.5, 9.0), (9, 9, 9, 9)])
    r = run(mkt, [1, 1, 1, 1], exits=ExitSpec(stop_loss="percent", stop_value=0.1))
    assert r.trades[0].exit_reason == "stop_loss"
    assert r.fills[-1].price == pytest.approx(9.0) and r.fills[-1].date == mkt.index[2]


def test_a_gap_through_the_stop_fills_at_the_worse_open():
    mkt = market([(10, 10, 10, 10), (10, 10, 10, 10), (8.0, 8.5, 7.5, 8.2), (8, 8, 8, 8)])
    r = run(mkt, [1, 1, 1, 1], exits=ExitSpec(stop_loss="percent", stop_value=0.1))
    assert r.fills[-1].price == 8.0 and r.trades[0].exit_reason == "stop_loss"


def test_when_one_bar_reaches_stop_and_target_the_stop_is_assumed():
    mkt = market([(10, 10, 10, 10), (10, 10, 10, 10), (10, 12.5, 8.5, 11), (11, 11, 11, 11)])
    exits = ExitSpec(stop_loss="percent", stop_value=0.1, take_profit="percent", take_value=0.2)
    r = run(mkt, [1, 1, 1, 1], exits=exits)
    assert r.trades[0].exit_reason == "stop_loss" and r.fills[-1].price == pytest.approx(9.0)


def test_a_gap_through_the_target_fills_at_the_better_open():
    mkt = market([(10, 10, 10, 10), (10, 10, 10, 10), (13, 13.5, 12.8, 13), (13, 13, 13, 13)])
    r = run(mkt, [1, 1, 1, 1], exits=ExitSpec(take_profit="percent", take_value=0.2))
    assert r.trades[0].exit_reason == "take_profit" and r.fills[-1].price == 13.0


def test_the_stop_is_live_on_the_entry_bar():
    mkt = market([(10, 10, 10, 10), (10, 10.2, 8.9, 9.5), (9.5, 9.5, 9.5, 9.5)])
    r = run(mkt, [1, 1, 1], exits=ExitSpec(stop_loss="percent", stop_value=0.1))
    assert r.trades[0].exit_date == mkt.index[1] and r.trades[0].holding_bars == 0


def test_after_a_stop_the_signal_must_reset_before_a_new_entry():
    rows = [(10, 10, 10, 10), (10, 10, 10, 10), (9, 9, 8.5, 9)] + [(9, 9, 9, 9)] * 5
    mkt = market(rows)
    exits = ExitSpec(stop_loss="percent", stop_value=0.1)
    waits = run(mkt, [1, 1, 1, 1, 0, 1, 1, 1], exits=exits)
    buys = [f.date for f in waits.fills if f.side == "buy"]
    assert buys == [mkt.index[1], mkt.index[6]]  # not on bar 3: the signal was still on
    eager = run(mkt, [1, 1, 1, 1, 1, 1, 1, 1],
                exits=ExitSpec(stop_loss="percent", stop_value=0.1, reentry="immediate"))
    assert [f.date for f in eager.fills if f.side == "buy"][1] == mkt.index[3]


def test_a_gap_up_shrinks_the_order_to_the_cash_and_says_so():
    mkt = market([(10, 10, 10, 10), (11, 11, 11, 11), (11, 11, 11, 11)])
    r = run(mkt, [1, 1, 1])
    assert r.fills[0].shares == 90  # 100 decided at 10; 1000 buys 90 at 11
    assert [x.reason for x in r.rejections] == ["reduced_for_cash"]
    assert r.cash.min() >= 0


def test_a_bar_without_open_fills_nothing_and_the_order_is_retried():
    mkt = market([(10, 10, 10, 10), (np.nan, 10, 10, 10), (10, 10, 10, 10), (10, 10, 10, 10)])
    r = run(mkt, [1, 1, 1, 1])
    assert [x.reason for x in r.rejections] == ["no_open_price"]
    assert r.fills[0].date == mkt.index[2]


def test_a_time_stop_exits_after_the_holding_limit():
    mkt = market([(10, 10, 10, 10)] * 8)
    r = run(mkt, [1] * 8, exits=ExitSpec(max_holding_bars=3))
    t = r.trades[0]
    assert t.exit_reason == "time_stop" and t.holding_bars == 3 and t.exit_date == mkt.index[4]


def test_small_target_changes_inside_the_band_do_not_trade():
    mkt = market([(10, 10, 10, 10)] * 5)
    r = run(mkt, [1.0, 0.9, 0.9, 0.5, 0.5], rebalance_band=0.2)
    assert [(f.side, f.shares, f.reason) for f in r.fills] == [
        ("buy", 100, "entry"), ("sell", 50, "rebalance")]


def test_an_atr_stop_that_cannot_be_computed_refuses_the_entry():
    mkt = market([(10, 11, 9, 10)] * 20)
    r = run(mkt, [1] * 20, exits=ExitSpec(stop_loss="atr", stop_value=3, atr_window=14))
    assert r.rejections[0].reason == "stop_not_computable"
    assert r.fills[0].date == mkt.index[14]  # ATR(14) first exists at bar 13


def test_invalid_targets_are_refused():
    mkt = market(FLAT)
    with pytest.raises(ValueError, match="long-only"):
        run(mkt, [1, 1.5, 1])
    with pytest.raises(ValueError, match="long-only"):
        run(mkt, [0, -1, 0])
    with pytest.raises(ValueError, match="index"):
        simulate("T", mkt, pd.Series([1, 1, 1]), costs=FREE, initial_cash=1.0)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_accounting_closes_for_random_strategies(seed):
    """Cash never negative; equity = cash + shares x close on every bar; the
    trades' P&L adds up to the change in equity, to the cent."""
    bars = synthetic_bars(400, seed=seed)
    bars["dividend"] = 0.0
    bars.iloc[::63, bars.columns.get_loc("dividend")] = 0.4
    rng = np.random.RandomState(seed)
    weights = np.where(rng.rand(400) < 0.5, 0.0, rng.choice([0.3, 0.6, 1.0], 400))
    costs = CostModel(commission_per_share=0.005, commission_min=1.0, spread_bps=3, slippage_bps=2)
    exits = ExitSpec(stop_loss="percent", stop_value=0.03, take_profit="percent", take_value=0.05)
    r = simulate("RND", bars, pd.Series(weights, index=bars.index), costs=costs,
                 initial_cash=50_000.0, exits=exits)
    assert r.cash.min() >= -1e-9
    np.testing.assert_allclose(r.equity, r.cash + r.shares * bars["close"], rtol=0, atol=1e-6)
    assert sum(t.pnl for t in r.trades) == pytest.approx(r.equity.iloc[-1] - 50_000.0, abs=1e-6)
    assert all(f.label == "PAPER TRADE — NOT REAL MONEY" for f in r.fills)
