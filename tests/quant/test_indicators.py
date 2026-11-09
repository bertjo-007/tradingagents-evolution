"""Indicators: causal by truncation, equal to stockstats where the agents see
them, right on hand-computed values, and honest about missing input."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tradingagents_quant.indicators import core, registry

from .fixtures import synthetic_bars

BARS = synthetic_bars(600)

# Small windows keep the truncation test fast and put many bars past warm-up.
_CAUSAL_PARAMS = {
    "sma": {"n": 10}, "ema": {"n": 10}, "rsi": {"n": 14},
    "macd": {"fast": 5, "slow": 13, "signal": 4}, "boll": {"n": 20, "k": 2.0},
    "atr": {"n": 14}, "vwma": {"n": 10}, "hv": {"n": 20}, "roc": {"n": 5},
    "mom": {"lookback": 60, "skip": 5}, "obv": {}, "volratio": {"n": 10},
    "dmi": {"n": 14}, "donchian": {"n": 20},
}


def test_every_registered_indicator_is_covered_by_the_causality_test():
    assert set(_CAUSAL_PARAMS) == set(registry.REGISTRY)


@pytest.mark.parametrize("key", sorted(_CAUSAL_PARAMS))
def test_a_value_never_depends_on_a_later_bar(key):
    """ind(bars[:t+1]) at t is bit-identical to ind(bars) at t."""
    full = registry.compute(BARS, key, **_CAUSAL_PARAMS[key])
    for t in [0, 1, 5, 13, 19, 27, 40, 59, 60, 61, 120, 299, 450, 599]:
        cut = registry.compute(BARS.iloc[: t + 1], key, **_CAUSAL_PARAMS[key])
        a, b = full.iloc[t].to_numpy(), cut.iloc[-1].to_numpy()
        assert np.array_equal(a, b, equal_nan=True), f"{key} at bar {t}: {a} != {b}"


# -- parity with stockstats, which computes these numbers for the agents ------


@pytest.fixture(scope="module")
def stockstats_frame():
    from stockstats import wrap

    frame = BARS[["open", "high", "low", "close", "volume"]].copy()
    frame.index.name = "date"
    return wrap(frame)


def _close(ours: pd.Series, theirs: pd.Series, *, start: int, rel: float) -> None:
    a, b = ours.iloc[start:].to_numpy(), np.asarray(theirs)[start:]
    assert not np.isnan(a).any(), "warm-up reaches into the compared range"
    np.testing.assert_allclose(a, b, rtol=rel, atol=0)


@pytest.mark.parametrize(
    ("ours", "column", "start"),
    [
        (lambda: core.sma(BARS["close"], 50), "close_50_sma", 49),
        (lambda: core.sma(BARS["close"], 200), "close_200_sma", 199),
        (lambda: core.ema(BARS["close"], 10), "close_10_ema", 9),
        (lambda: core.rsi(BARS["close"], 14), "rsi_14", 14),
        (lambda: core.macd(BARS["close"])["macd"], "macd", 25),
        (lambda: core.macd(BARS["close"])["signal"], "macds", 33),
        (lambda: core.macd(BARS["close"])["hist"], "macdh", 33),
        (lambda: core.bollinger(BARS["close"])["mid"], "boll", 19),
        (lambda: core.bollinger(BARS["close"])["upper"], "boll_ub", 19),
        (lambda: core.bollinger(BARS["close"])["lower"], "boll_lb", 19),
        (lambda: core.vwma(BARS["high"], BARS["low"], BARS["close"], BARS["volume"], 20),
         "vwma_20", 19),
    ],
)
def test_the_indicators_shown_to_agents_equal_stockstats(stockstats_frame, ours, column, start):
    _close(ours(), stockstats_frame[column], start=start, rel=1e-12)


def test_atr_equals_stockstats_from_the_end_of_warm_up(stockstats_frame):
    # Both take the first bar's true range as its own high - low (stockstats
    # sets the missing previous close to the first close), so they agree exactly.
    ours = core.atr(BARS["high"], BARS["low"], BARS["close"], 14)
    _close(ours, stockstats_frame["atr_14"], start=13, rel=1e-12)


# -- hand-computed values ------------------------------------------------------


def _s(values) -> pd.Series:
    return pd.Series(values, dtype=float, index=pd.bdate_range("2024-01-01", periods=len(values)))


def _eq(series, expected):
    np.testing.assert_allclose(series.to_numpy(), np.asarray(expected, dtype=float), rtol=1e-12,
                               equal_nan=True)


def test_sma_and_ema_by_hand():
    _eq(core.sma(_s([1, 2, 3, 4, 5]), 3), [np.nan, np.nan, 2, 3, 4])
    # span 3: alpha 0.5, adjusted weights 1, 0.5, 0.25 -> (3 + 1 + 0.25) / 1.75
    _eq(core.ema(_s([1, 2, 3]), 3), [np.nan, np.nan, 4.25 / 1.75])


def test_rsi_extremes():
    assert core.rsi(_s(range(1, 40)), 14).iloc[-1] == 100.0
    assert core.rsi(_s(range(40, 1, -1)), 14).iloc[-1] == 0.0
    assert core.rsi(_s([10.0] * 30), 14).iloc[-1] == 50.0  # no movement: neutral
    assert core.rsi(_s(range(1, 40)), 14).iloc[:14].isna().all()


def test_true_range_takes_the_gap_into_account():
    high, low, close = _s([11, 10]), _s([9, 8]), _s([12, 9])
    # Bar 2: high-low 2, |10-12| 2, |8-12| 4.
    _eq(core.true_range(high, low, close), [2, 4])


def test_a_missing_high_makes_the_range_unknown_not_smaller():
    high, low, close = _s([11, np.nan, 12]), _s([9, 9, 10]), _s([10, 10, 11])
    tr = core.true_range(high, low, close)
    assert math.isnan(tr.iloc[1]) and tr.iloc[2] == 2.0


def test_volatility_of_constant_growth_is_zero():
    close = _s(100 * np.exp(0.01 * np.arange(30)))
    assert core.historical_volatility(close, 20).iloc[-1] == pytest.approx(0.0, abs=1e-12)


def test_rate_of_change_momentum_and_obv():
    _eq(core.roc(_s([100, 110]), 1), [np.nan, 0.1])
    # lookback 3, skip 1 at t=3: close[2] / close[0] - 1 = 4 / 1 - 1
    assert core.momentum(_s([1, 2, 4, 8, 16]), 3, 1).iloc[3] == 3.0
    _eq(core.obv(_s([10, 11, 10, 10, 12]), _s([1, 2, 3, 4, 5])), [0, 2, -1, -1, 4])


def test_bollinger_of_a_constant_price_has_no_percent_b():
    bands = core.bollinger(_s([5.0] * 25), 20)
    assert bands["upper"].iloc[-1] == bands["lower"].iloc[-1] == 5.0
    assert math.isnan(bands["pct_b"].iloc[-1])  # undefined, not invented


def test_dmi_flat_and_trending():
    flat = core.dmi(_s([10.0] * 60), _s([10.0] * 60), _s([10.0] * 60), 14)
    assert flat["adx"].iloc[-1] == 0.0
    rising = np.arange(60, dtype=float)
    up = core.dmi(_s(rising + 1), _s(rising), _s(rising + 0.5), 14)
    assert up["plus_di"].iloc[-1] > 0 and up["minus_di"].iloc[-1] == 0.0
    assert up["adx"].iloc[-1] > 90
    assert up["adx"].iloc[: 2 * 14 - 1].isna().all()


def test_donchian_includes_today_so_a_breakout_must_shift():
    channel = core.donchian(_s([1, 3, 2, 5]), _s([0, 1, 1, 2]), 3)
    _eq(channel["upper"], [np.nan, np.nan, 3, 5])


@pytest.mark.parametrize(
    "call",
    [lambda: core.sma(_s([1, 2]), 0), lambda: core.macd(_s([1, 2]), 26, 12),
     lambda: core.momentum(_s([1, 2]), 5, 5), lambda: core.bollinger(_s([1, 2]), 1)],
)
def test_bad_parameters_are_refused(call):
    with pytest.raises(ValueError):
        call()


# -- registry --------------------------------------------------------------------


def test_registry_names_columns_by_parameters():
    assert list(registry.compute(BARS, "rsi").columns) == ["rsi_14"]
    assert list(registry.compute(BARS, "macd").columns) == [
        "macd_12_26_9_macd", "macd_12_26_9_signal", "macd_12_26_9_hist"]
    assert "boll_20_2_upper" in registry.compute(BARS, "boll").columns
    many = registry.compute_many(BARS, [("sma", {"n": 50}), ("sma", {"n": 200})])
    assert list(many.columns) == ["sma_50", "sma_200"]


def test_registry_refuses_what_it_cannot_compute():
    with pytest.raises(KeyError, match="unknown indicator"):
        registry.compute(BARS, "magic")
    with pytest.raises(ValueError, match="no parameter"):
        registry.compute(BARS, "rsi", window=14)
    with pytest.raises(ValueError, match="needs bar column"):
        registry.compute(BARS[["close"]], "atr")


def test_values_at_reads_no_later_row_and_reports_missing_as_none():
    frame = registry.compute(BARS, "sma", n=50)
    day = BARS.index[100]
    assert registry.values_at(frame, day) == {"sma_50": float(frame.iloc[100, 0])}
    weekend = day + pd.Timedelta(days=1) if day.weekday() < 4 else day
    assert registry.values_at(frame, weekend)["sma_50"] in (
        float(frame.iloc[100, 0]), float(frame.iloc[101, 0]))
    assert registry.values_at(frame, BARS.index[10]) == {"sma_50": None}
    assert registry.values_at(frame, "1990-01-01") == {"sma_50": None}


def test_the_agent_flag_matches_the_upstream_whitelist():
    shown = {k for k, s in registry.REGISTRY.items() if s.shown_to_agents}
    assert shown == {"sma", "ema", "rsi", "macd", "boll", "atr", "vwma"}
