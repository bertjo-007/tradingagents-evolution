"""How much of a Sharpe ratio survives the number of things that were tried.

Bailey, D. H. and López de Prado, M. (2014), "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality",
Journal of Portfolio Management 40(5).

- ``probabilistic_sharpe``: the probability that the true Sharpe exceeds a
  threshold, given the observed one, the sample length, and the skewness and
  kurtosis of the returns (fat tails make a Sharpe less certain).
- ``expected_max_sharpe``: the Sharpe the best of N worthless strategies is
  expected to reach by luck alone.
- ``deflated_sharpe``: the probabilistic Sharpe against that luck threshold.

All Sharpe ratios here are **per period** (daily), not annualized; the caller
converts. Only the standard library's NormalDist is used (no scipy: see
docs/ARCHITEKTUR.md on native builds).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist

import numpy as np

EULER_GAMMA = 0.5772156649015329
_N = NormalDist()


def moments(returns: Sequence[float]) -> tuple[float, float, float, int]:
    """(Sharpe per period, skewness, kurtosis — not excess, sample size)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    t = len(r)
    if t < 3:
        raise ValueError("need at least three returns")
    std = r.std(ddof=1)
    if std == 0:
        raise ValueError("returns have no variation")
    z = (r - r.mean()) / r.std(ddof=0)
    return float(r.mean() / std), float(np.mean(z**3)), float(np.mean(z**4)), t


def probabilistic_sharpe(
    sharpe: float, threshold: float, t: int, skew: float = 0.0, kurtosis: float = 3.0
) -> float:
    """P(true Sharpe > threshold). Sharpe and threshold per period."""
    if t < 2:
        raise ValueError("need at least two observations")
    variance = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if variance <= 0:
        return math.nan
    return _N.cdf((sharpe - threshold) * math.sqrt(t - 1) / math.sqrt(variance))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum Sharpe among ``n_trials`` strategies with true Sharpe
    zero, whose estimated Sharpe ratios vary with ``sharpe_variance``."""
    if n_trials < 1:
        raise ValueError("at least one trial")
    if n_trials == 1 or sharpe_variance <= 0:
        return 0.0
    n = float(n_trials)
    return math.sqrt(sharpe_variance) * (
        (1.0 - EULER_GAMMA) * _N.inv_cdf(1.0 - 1.0 / n)
        + EULER_GAMMA * _N.inv_cdf(1.0 - 1.0 / (n * math.e))
    )


def sharpe_difference(
    returns_a: Sequence[float], returns_b: Sequence[float], *, n_trials: int = 1
) -> dict[str, float | None]:
    """Is the Sharpe ratio of ``a`` higher than that of ``b``, measured on the
    same days? The Jobson-Korkie test with Memmel's (2003) correction, for two
    correlated return series:

        theta = (2 (1 - rho) + (SR_a^2 + SR_b^2 - 2 SR_a SR_b rho^2) / 2) / T
        z     = (SR_a - SR_b) / sqrt(theta)

    ``p`` = Phi(z) is the probability that ``a``'s true Sharpe is the higher.
    With ``n_trials`` looks at the same data the claim has to survive all of
    them (Sidak): ``p_deflated`` = p ** n_trials.

    This tests what the hypotheses claim — a higher Sharpe ratio. (An
    information ratio would test a higher *return*, and would reject a
    strategy that is better per unit of risk while holding less.) The test
    assumes roughly normal returns; fat tails make it somewhat optimistic.
    """
    a = np.asarray(returns_a, dtype=float)
    b = np.asarray(returns_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("the two return series must cover the same days")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    t = len(a)
    if t < 3:
        raise ValueError("need at least three returns")
    sd_a, sd_b = a.std(ddof=1), b.std(ddof=1)
    if sd_a == 0 or sd_b == 0:
        raise ValueError("a return series has no variation")
    sr_a, sr_b = a.mean() / sd_a, b.mean() / sd_b
    rho = float(np.corrcoef(a, b)[0, 1])
    theta = (2.0 * (1.0 - rho) + 0.5 * (sr_a**2 + sr_b**2 - 2.0 * sr_a * sr_b * rho**2)) / t
    if theta <= 0:  # identical series: no difference can be measured
        z, p = 0.0, 0.5
    else:
        z = float((sr_a - sr_b) / math.sqrt(theta))
        p = _N.cdf(z)
    return {
        "sharpe_a_annualized": float(sr_a * math.sqrt(252)),
        "sharpe_b_annualized": float(sr_b * math.sqrt(252)),
        "correlation": rho,
        "observations": t,
        "z": z,
        "p": p,
        "trials": n_trials,
        "p_deflated": p**n_trials,
    }


def deflated_sharpe(
    returns: Sequence[float], *, n_trials: int, trial_sharpes: Sequence[float] | None = None,
) -> dict[str, float]:
    """Deflated Sharpe of the selected strategy's ``returns``.

    ``trial_sharpes``: per-period Sharpe ratios of all trials, for their
    variance. Without them (or with one trial) the variance is the sampling
    variance of a Sharpe under the null, 1 / (T - 1).
    """
    sharpe, skew, kurt, t = moments(returns)
    if trial_sharpes is not None and len(trial_sharpes) > 1:
        variance = float(np.var(np.asarray(trial_sharpes, dtype=float), ddof=1))
    else:
        variance = 1.0 / (t - 1)
    threshold = expected_max_sharpe(n_trials, variance)
    return {
        "sharpe_per_period": sharpe,
        "sharpe_annualized": sharpe * math.sqrt(252),
        "skewness": skew,
        "kurtosis": kurt,
        "observations": t,
        "trials": n_trials,
        "luck_threshold_annualized": threshold * math.sqrt(252),
        "psr_vs_zero": probabilistic_sharpe(sharpe, 0.0, t, skew, kurt),
        "deflated_sharpe": probabilistic_sharpe(sharpe, threshold, t, skew, kurt),
    }
