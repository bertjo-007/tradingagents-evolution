"""Several instruments, one cash account: limits, order of fills, calendars,
dividend reinvestment and an accounting that closes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents_quant.sim.engine import portfolio_weights, simulate, simulate_portfolio
from tradingagents_quant.sim.execution import CostModel
from tradingagents_quant.strategy.spec import ExitSpec

from .fixtures import synthetic_bars

FREE = CostModel.free()


def mk(closes, start="2024-01-01", dividends=None, dates=None):
    closes = [float(x) for x in closes]
    index = pd.DatetimeIndex(dates) if dates is not None else pd.bdate_range(start, periods=len(closes))
    frame = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes,
                          "volume": 1e6, "dividend": 0.0}, index=index)
    for i, v in (dividends or {}).items():
        frame.iloc[i, frame.columns.get_loc("dividend")] = v
    return frame


def run(markets, weights, cash=1000.0, **kw):
    targets = {s: pd.Series(w, index=markets[s].index, dtype=float) for s, w in weights.items()}
    return simulate_portfolio(markets, targets, costs=kw.pop("costs", FREE), initial_cash=cash, **kw)


def test_two_instruments_share_the_cash():
    a, b = mk([10] * 4), mk([20] * 4)
    r = run({"A": a, "B": b}, {"A": [0.5] * 4, "B": [0.5] * 4})
    first = {(f.symbol, f.shares) for f in r.fills if f.date == a.index[1]}
    assert first == {("A", 50), ("B", 25)}  # 500 each
    assert r.cash.iloc[-1] == 0 and r.equity.iloc[-1] == 1000


def test_the_gross_limit_scales_all_weights_down_in_proportion():
    w = portfolio_weights(pd.DataFrame({"A": [0.8, 0.3], "B": [0.8, np.nan]}), max_gross=1.0)
    assert list(w["A"]) == [0.5, 0.3] and w["B"].iloc[0] == 0.5 and np.isnan(w["B"].iloc[1])
    capped = portfolio_weights(pd.DataFrame({"A": [0.9], "B": [0.1]}), max_weight=0.6)
    assert capped.iloc[0].tolist() == [0.6, 0.1]


def test_sells_fill_before_buys_so_a_switch_is_funded():
    a, b = mk([10] * 5), mk([10] * 5)
    r = run({"A": a, "B": b}, {"A": [1, 1, 0, 0, 0], "B": [0, 0, 1, 1, 1]})
    day = a.index[3]
    assert [(f.symbol, f.side) for f in r.fills if f.date == day] == [("A", "sell"), ("B", "buy")]
    assert [f.shares for f in r.fills if f.date == day] == [100, 100]


def test_competing_buys_shrink_in_proportion_when_cash_is_short():
    # Decided at 10 each (50 + 50 shares), both open at 12: cash buys ~41 + 41.
    a, b = mk([10, 12, 12]), mk([10, 12, 12])
    r = run({"A": a, "B": b}, {"A": [0.5] * 3, "B": [0.5] * 3})
    bought = {f.symbol: f.shares for f in r.fills}
    assert bought == {"A": 41, "B": 41}
    assert {x.reason for x in r.rejections} == {"reduced_for_cash"}
    assert r.cash.min() >= 0


def test_an_instrument_is_not_traded_before_its_first_bar():
    a = mk([10] * 6)
    late = mk([10] * 3, dates=a.index[3:])
    r = run({"A": a, "L": late}, {"A": [0.5] * 6, "L": [0.5] * 3})
    assert min(f.date for f in r.fills if f.symbol == "L") == a.index[4]  # decided on its first bar
    assert not [x for x in r.rejections if x.symbol == "L"]


def test_a_held_instrument_without_a_bar_keeps_its_last_close_and_the_notes_say_so():
    a = mk([10] * 6)
    gappy = mk([20, 20, 20, 24, 24], dates=a.index.delete(3))
    r = run({"A": a, "G": gappy}, {"A": [0.5] * 6, "G": [0.5] * 5})
    assert r.market_value.loc[a.index[3], "G"] == r.positions.loc[a.index[3], "G"] * 20
    assert any("G had calendar days without a bar" in n for n in r.notes)


def test_an_order_for_a_day_without_a_bar_is_rejected_and_redecided():
    a = mk([10] * 6)
    gappy = mk([20] * 5, dates=a.index.delete(1))
    r = run({"A": a, "G": gappy}, {"A": [0.0] * 6, "G": [0.5] * 5})
    assert [(x.symbol, x.reason) for x in r.rejections] == [("G", "no_bar")]
    assert min(f.date for f in r.fills if f.symbol == "G") == a.index[3]


def test_a_single_member_portfolio_is_the_single_instrument_simulation():
    bars = synthetic_bars(300, seed=13)
    bars["dividend"] = 0.0
    weights = pd.Series(np.where(np.arange(300) % 50 < 30, 1.0, 0.0), index=bars.index)
    costs = CostModel(commission_per_share=0.01, spread_bps=4, slippage_bps=2)
    exits = ExitSpec(stop_loss="percent", stop_value=0.04)
    one = simulate("X", bars, weights, costs=costs, initial_cash=1e5, exits=exits)
    many = simulate_portfolio({"X": bars}, {"X": weights}, costs=costs, initial_cash=1e5,
                              exits=exits, calendar=bars.index)
    assert one.fills == many.fills
    np.testing.assert_array_equal(one.equity.to_numpy(), many.equity.to_numpy())


@pytest.mark.parametrize("seed", [5, 6])
def test_portfolio_accounting_closes(seed):
    rng = np.random.RandomState(seed)
    markets, targets = {}, {}
    for k, sym in enumerate("ABCD"):
        bars = synthetic_bars(350 - 40 * k, seed=seed * 10 + k, start="2015-01-05")
        bars.index = pd.bdate_range("2015-01-05", periods=350)[40 * k:]  # staggered inceptions
        bars["dividend"] = 0.0
        bars.iloc[::60, bars.columns.get_loc("dividend")] = 0.3
        markets[sym] = bars
        targets[sym] = pd.Series(np.where(rng.rand(len(bars)) < 0.4, 0.0, 0.5), index=bars.index)
    costs = CostModel(commission_per_share=0.005, commission_min=1.0, spread_bps=3, slippage_bps=2)
    r = simulate_portfolio(markets, targets, costs=costs, initial_cash=200_000.0, max_gross=1.0,
                           exits=ExitSpec(stop_loss="percent", stop_value=0.03))
    assert r.cash.min() >= -1e-9
    assert (r.weights.sum(axis=1) <= 1.0 + 1e-12).all()
    np.testing.assert_allclose(r.equity, r.cash + r.market_value.sum(axis=1), atol=1e-6)
    assert sum(t.pnl for t in r.trades) == pytest.approx(r.equity.iloc[-1] - 200_000.0, abs=1e-6)


# -- dividend reinvestment -------------------------------------------------------


def test_dividends_are_reinvested_in_whole_shares_when_asked():
    mkt = mk([10] * 8, dividends={2: 0.6, 5: 0.6})
    target = pd.Series(1.0, index=mkt.index)
    kept = simulate("X", mkt, target, costs=FREE, initial_cash=1000.0, rebalance_band=1e9)
    drip = simulate("X", mkt, target, costs=FREE, initial_cash=1000.0, rebalance_band=1e9,
                    reinvest_dividends=True)
    assert kept.shares.iloc[-1] == 100 and kept.cash.iloc[-1] == pytest.approx(120.0)
    # 100 shares x 0.60 = 60 -> 6 shares at the next open; then 106 x 0.60 = 63.6 -> 6 more.
    reinvest = [f for f in drip.fills if f.reason == "dividend_reinvest"]
    assert [(f.date, f.shares) for f in reinvest] == [(mkt.index[3], 6), (mkt.index[6], 6)]
    assert drip.shares.iloc[-1] == 112 and drip.cash.iloc[-1] == pytest.approx(3.6)


def test_buy_and_hold_with_reinvestment_tracks_the_total_return_index():
    from tradingagents_quant.sim.prices import market_frame

    bars = synthetic_bars(1000, seed=31, drift=0.0003)
    dates = [d.strftime("%Y-%m-%d") for d in bars.index[30::63]]
    market = market_frame(bars, pd.DataFrame({"date": dates, "kind": "dividend", "value": 0.5}))[0]
    r = simulate("X", market, pd.Series(1.0, index=market.index), costs=FREE, initial_cash=1e6,
                 rebalance_band=1e9, reinvest_dividends=True)
    tr = market["tr_close"]
    expected = tr.iloc[-1] / tr.iloc[1] * market["close"].iloc[1] / market["open"].iloc[1]
    assert r.equity.iloc[-1] / 1e6 == pytest.approx(expected, rel=2e-3)
