"""Performance figures against hand-computed values."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tradingagents_quant.sim import metrics
from tradingagents_quant.sim.types import Fill, SimulationResult, Trade


def _result(equity, shares=None, trades=(), fills=(), regimes=None, dividends=0.0):
    idx = pd.bdate_range("2024-01-01", periods=len(equity))
    eq = pd.Series(equity, index=idx, dtype=float)
    sh = pd.Series(shares if shares is not None else [1.0] * len(equity), index=idx, dtype=float)
    return SimulationResult(fills=list(fills), symbol="T", equity=eq, cash=eq * 0, shares=sh,
                            target=sh * 0, trades=list(trades), dividends=dividends,
                            regimes=regimes.set_index(idx) if regimes is not None else None)


def _trade(pnl, bars=5):
    day = pd.Timestamp("2024-01-01")
    buy = Fill(day, "buy", 1, 100, 100, 0, "entry")
    sell = Fill(day, "sell", 1, 100 + pnl, 100 + pnl, 0, "exit_signal")
    return Trade(day, 100, "up/normal", [buy, sell], exit_date=day, exit_price=100 + pnl,
                 exit_reason="exit_signal", holding_bars=bars)


def test_returns_drawdown_and_total_return_by_hand():
    r = _result([110, 99, 108.9])
    out = metrics.compute(r, initial_cash=100.0, min_annualization_bars=2)
    assert out["total_return"] == pytest.approx(0.089)
    assert out["max_drawdown"] == pytest.approx(99 / 110 - 1)
    assert out["longest_drawdown_bars"] == 2  # 99 and 108.9 are both below the 110 peak
    returns = np.array([0.1, -0.1, 0.1])
    expected_sharpe = returns.mean() / returns.std(ddof=1) * math.sqrt(252)
    assert out["sharpe"] == pytest.approx(expected_sharpe)
    downside = math.sqrt((0 + 0.01 + 0) / 3)
    assert out["sortino"] == pytest.approx(returns.mean() / downside * math.sqrt(252))


def test_no_annualized_figure_from_a_short_window():
    out = metrics.compute(_result([101, 102, 103]), initial_cash=100.0, min_annualization_bars=252)
    for key in ("cagr", "volatility", "sharpe", "sortino", "turnover_per_year"):
        assert out[key] is None
    assert "not reported" in out["annualization_note"]
    assert out["total_return"] == pytest.approx(0.03)


def test_cagr_over_exactly_one_year():
    equity = list(np.linspace(100.5, 110.0, 252))
    out = metrics.compute(_result(equity), initial_cash=100.0, min_annualization_bars=252)
    assert out["cagr"] == pytest.approx(0.10)


def test_trade_statistics():
    trades = [_trade(10), _trade(-5), _trade(20), _trade(-5)]
    out = metrics.compute(_result([100, 100, 100], trades=trades), initial_cash=100.0)
    assert out["trades"] == 4 and out["win_rate"] == 0.5
    assert out["profit_factor"] == pytest.approx(30 / 10)
    assert out["avg_win"] == 15 and out["avg_loss"] == -5
    assert out["exit_reasons"] == {"exit_signal": 4}


def test_no_profit_factor_without_a_loss_and_no_ratios_without_trades():
    out = metrics.compute(_result([100, 101], trades=[_trade(1)]), initial_cash=100.0)
    assert out["profit_factor"] is None  # not infinity dressed up as a number
    empty = metrics.compute(_result([100, 100]), initial_cash=100.0)
    assert empty["win_rate"] is None and empty["avg_holding_bars"] is None


def test_exposure_and_turnover():
    fills = [Fill(pd.Timestamp("2024-01-01"), "buy", 10, 10, 10, 0, "entry")]
    out = metrics.compute(_result([100, 100, 100, 100], shares=[0, 10, 10, 0], fills=fills),
                          initial_cash=100.0)
    assert out["exposure"] == 0.5
    assert out["turnover"] == pytest.approx(1.0)  # 100 traded over mean equity 100


def test_a_constant_equity_has_no_sharpe():
    out = metrics.compute(_result([100.0] * 300), initial_cash=100.0)
    assert out["sharpe"] is None and out["volatility"] is None


def test_regime_returns_belong_to_the_label_known_while_holding():
    labels = pd.DataFrame({"regime": ["a", "b", "b", "a"]})
    r = _result([100, 110, 121, 121], shares=[1, 1, 1, 1], regimes=labels)
    split = metrics.by_regime(r, initial_cash=100.0)
    # Return of bar t goes to the label of bar t-1: bar 1 (+10 %) to "a",
    # bar 2 (+10 %) to "b", bar 3 (0 %) to "b"; bar 0 to "unknown".
    assert split["a"]["compounded_return"] == pytest.approx(0.10)
    assert split["b"]["compounded_return"] == pytest.approx(0.10)
    assert split["unknown"]["bars"] == 1


def test_comparison_is_strategy_minus_benchmark():
    diff = metrics.compare({"total_return": 0.5, "sharpe": None, "max_drawdown": -0.2},
                           {"total_return": 0.7, "sharpe": 0.9, "max_drawdown": -0.5})
    assert diff["total_return_vs_benchmark"] == pytest.approx(-0.2)
    assert diff["sharpe_vs_benchmark"] is None
    assert diff["max_drawdown_vs_benchmark"] == pytest.approx(0.3)
