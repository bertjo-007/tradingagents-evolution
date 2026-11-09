"""The deflated Sharpe ratio against independent references: closed forms, a
Monte Carlo of the expected maximum, and calibration under the null."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest

from tradingagents_quant.sim.deflate import (
    deflated_sharpe,
    expected_max_sharpe,
    moments,
    probabilistic_sharpe,
)


def test_psr_for_normal_returns_is_a_plain_z_test():
    # skew 0, kurtosis 3: the variance term is 1 + SR^2 / 2.
    sr, t = 0.05, 1001
    expected = NormalDist().cdf(sr * math.sqrt(t - 1) / math.sqrt(1 + sr**2 / 2))
    assert probabilistic_sharpe(sr, 0.0, t) == pytest.approx(expected)
    assert probabilistic_sharpe(0.0, 0.0, t) == pytest.approx(0.5)


def test_fat_tails_and_negative_skew_make_a_sharpe_less_certain():
    base = probabilistic_sharpe(0.05, 0.0, 1000)
    assert probabilistic_sharpe(0.05, 0.0, 1000, skew=-1.0, kurtosis=8.0) < base


def test_expected_maximum_matches_a_monte_carlo_of_pure_noise():
    rng = np.random.RandomState(1)
    for n in (10, 100, 1000):
        simulated = rng.standard_normal((4000, n)).max(axis=1).mean()
        assert expected_max_sharpe(n, 1.0) == pytest.approx(simulated, abs=0.06), n
    assert expected_max_sharpe(1, 1.0) == 0.0


def test_more_trials_raise_the_bar():
    values = [expected_max_sharpe(n, 0.01) for n in (2, 10, 100, 1000)]
    assert values == sorted(values) and values[0] > 0


def test_moments_of_a_known_series():
    sr, skew, kurt, t = moments([0.01, -0.01, 0.02, -0.02, 0.0])
    assert t == 5 and sr == pytest.approx(0.0) and skew == pytest.approx(0.0)
    assert kurt == pytest.approx(1.7)  # symmetric, light-tailed five points
    with pytest.raises(ValueError):
        moments([0.01, 0.01, 0.01])


def test_deflation_is_calibrated_under_the_null():
    """Pick the best of 20 worthless strategies, 300 times over. A naive test
    calls the winner significant most of the time; the deflated one about as
    often as its 5 % level allows."""
    rng = np.random.RandomState(7)
    naive = deflated_hits = 0
    runs, trials, days = 300, 20, 750
    for _ in range(runs):
        returns = rng.standard_normal((trials, days)) * 0.01
        sharpes = returns.mean(axis=1) / returns.std(axis=1, ddof=1)
        best = returns[int(np.argmax(sharpes))]
        result = deflated_sharpe(best, n_trials=trials, trial_sharpes=sharpes)
        naive += result["psr_vs_zero"] > 0.95
        deflated_hits += result["deflated_sharpe"] > 0.95
    assert naive / runs > 0.4
    assert deflated_hits / runs < 0.10


def test_a_real_edge_survives_deflation():
    rng = np.random.RandomState(3)
    edge = rng.standard_normal(2520) * 0.01 + 0.0012  # ~1.9 annualized Sharpe, 10 years
    result = deflated_sharpe(edge, n_trials=50)
    assert result["deflated_sharpe"] > 0.99
    assert result["sharpe_annualized"] == pytest.approx(1.9, abs=0.3)
