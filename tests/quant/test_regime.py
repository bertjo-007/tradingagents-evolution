"""Regimes: recognized in clear synthetic markets, unknown before warm-up, causal."""

from __future__ import annotations

import numpy as np
import pytest

from tradingagents_quant.regime import TRENDS, VOLATILITIES, RegimeRules, classify, summarize

from .fixtures import synthetic_bars

RULES = RegimeRules()
N = RULES.warmup + 200


def test_a_steady_rise_is_a_strong_uptrend():
    labels = classify(synthetic_bars(N, drift=0.003, vol=0.006))
    assert (labels["trend"].iloc[-50:] == "strong_up").mean() > 0.8


def test_a_steady_fall_is_a_strong_downtrend():
    labels = classify(synthetic_bars(N, drift=-0.003, vol=0.006))
    assert (labels["trend"].iloc[-50:] == "strong_down").mean() > 0.8


def test_a_quiet_steady_advance_is_an_uptrend_not_sideways():
    # Small daily ranges keep ADX low; ADX alone would call this sideways
    # (it called SPY's calm 2017 sideways on 56 % of its days).
    labels = classify(synthetic_bars(N, drift=0.0008, vol=0.004, seed=5))
    tail = labels["trend"].iloc[-100:]
    assert tail.isin(["up", "strong_up"]).mean() > 0.8
    assert (tail == "sideways").mean() == 0.0


def test_a_range_bound_market_is_not_a_strong_trend():
    labels = classify(synthetic_bars(N, vol=0.01, mean_reversion=0.8, seed=11))
    tail = labels["trend"].iloc[-100:]
    assert tail.isin(["sideways", "uncertain"]).mean() > 0.9
    assert (tail == "sideways").mean() > 0.5


def test_volatility_is_judged_against_the_own_past():
    calm_then_wild = [0.005] * (N - 30) + [0.03] * 30
    wild_then_calm = [0.03] * (N - 30) + [0.004] * 30
    assert classify(synthetic_bars(N, vol_path=calm_then_wild))["volatility"].iloc[-1] == "high"
    assert classify(synthetic_bars(N, vol_path=wild_then_calm))["volatility"].iloc[-1] == "low"


def test_nothing_is_guessed_before_warm_up():
    labels = classify(synthetic_bars(N))
    assert (labels["trend"].iloc[: RULES.long_window - 1] == "unknown").all()
    assert (labels["volatility"].iloc[: RULES.vol_window + RULES.vol_lookback - 1] == "unknown").all()
    assert set(labels["trend"]) <= set(TRENDS) and set(labels["volatility"]) <= set(VOLATILITIES)


@pytest.mark.parametrize("t", [150, 199, 260, 271, 272, 300, N - 1])
def test_a_label_never_depends_on_a_later_bar(t):
    bars = synthetic_bars(N, seed=3)
    full = classify(bars)
    cut = classify(bars.iloc[: t + 1])
    assert tuple(full.iloc[t]) == tuple(cut.iloc[-1])


def test_summary_shares_add_up():
    shares = summarize(classify(synthetic_bars(N)))
    assert sum(shares.values()) == pytest.approx(1.0)
    assert np.all(np.array(list(shares.values())) > 0)
