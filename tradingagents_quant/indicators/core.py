"""Indicator formulas.

Conventions, chosen so the deterministic core and the agent layer agree:
the indicators the agents are shown (upstream ``vendors/yahoo/market.py``
whitelist: SMA, EMA, MACD, RSI, Bollinger, ATR, VWMA) use the formulas of
``stockstats``, which computes them for the agents:

- EMA: ``ewm(span=n, adjust=True)``.
- Wilder smoothing (RSI, ATR, DMI): ``ewm(alpha=1/n, adjust=True)``.
- Bollinger: population of the last ``n`` closes, ``std`` with ``ddof=1``.
- RSI is 50 where average gain and loss are both zero (no movement).

Two differences, both on purpose:

- **Warm-up is NaN.** stockstats reports values from the first bar, computed
  over a partial window; here a value appears only once its window is full.
  From there on the values equal stockstats' to 1e-12 relative
  (``tests/quant/test_indicators.py``).
- **ADX is Wilder's:** ``SMMA(DX, n)``. stockstats uses ``EMA(DX, 6)``. ADX is
  not shown to the agents, so nothing has to agree with it there.

No function fills a gap or reads a later bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _masked(series: pd.Series, first_valid: int) -> pd.Series:
    """NaN for the first ``first_valid`` positions (the warm-up)."""
    out = series.astype(float).copy()
    if first_valid > 0:
        out.iloc[:first_valid] = np.nan
    return out


def _check_window(n: int) -> None:
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"window must be a positive integer, got {n!r}")


def smma(series: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothed moving average, stockstats' convention (unmasked)."""
    return series.ewm(alpha=1.0 / n, adjust=True, ignore_na=False).mean()


def sma(close: pd.Series, n: int) -> pd.Series:
    _check_window(n)
    return close.rolling(n, min_periods=n).mean()


def ema(close: pd.Series, n: int) -> pd.Series:
    _check_window(n)
    return _masked(close.ewm(span=n, adjust=True, ignore_na=False).mean(), n - 1)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    _check_window(n)
    diff = close.diff()
    gain = diff.clip(lower=0.0)
    loss = (-diff).clip(lower=0.0)
    # The first bar has no change; stockstats counts it as zero movement.
    gain.iloc[:1] = 0.0
    loss.iloc[:1] = 0.0
    avg_gain, avg_loss = smma(gain, n), smma(loss, n)
    total = avg_gain + avg_loss
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(total.to_numpy() != 0, 100.0 * avg_gain / total, 50.0)
    return _masked(pd.Series(values, index=close.index), n)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    for w in (fast, slow, signal):
        _check_window(w)
    if fast >= slow:
        raise ValueError(f"fast window {fast} must be shorter than slow window {slow}")
    line = (
        close.ewm(span=fast, adjust=True).mean() - close.ewm(span=slow, adjust=True).mean()
    )
    sig = line.ewm(span=signal, adjust=True).mean()
    first_line, first_signal = slow - 1, slow + signal - 2
    return pd.DataFrame(
        {
            "macd": _masked(line, first_line),
            "signal": _masked(sig, first_signal),
            "hist": _masked(line - sig, first_signal),
        },
        index=close.index,
    )


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    _check_window(n)
    if n < 2:
        raise ValueError("Bollinger bands need a window of at least 2")
    mid = close.rolling(n, min_periods=n).mean()
    std = close.rolling(n, min_periods=n).std(ddof=1)
    upper, lower = mid + k * std, mid - k * std
    width = upper - lower
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_b = (close - lower) / width.where(width != 0)
    return pd.DataFrame(
        {"mid": mid, "upper": upper, "lower": lower, "pct_b": pct_b, "bandwidth": width / mid},
        index=close.index,
    )


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """max(high - low, |high - prev close|, |low - prev close|); the first bar,
    without a previous close, is its own range high - low."""
    prev = close.shift(1)
    ranges = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1)
    # skipna covers only the first bar's missing previous close. A missing high
    # or low makes the range unknown, not the range of what is left.
    tr = ranges.max(axis=1, skipna=True)
    tr[high.isna() | low.isna()] = np.nan
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    _check_window(n)
    return _masked(smma(true_range(high, low, close), n), n - 1)


def historical_volatility(close: pd.Series, n: int = 20, periods_per_year: int = 252) -> pd.Series:
    """Annualized standard deviation (ddof=1) of the last ``n`` log returns."""
    _check_window(n)
    if n < 2:
        raise ValueError("volatility needs at least 2 returns")
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(n, min_periods=n).std(ddof=1) * np.sqrt(periods_per_year)


def roc(close: pd.Series, n: int) -> pd.Series:
    """Rate of change over ``n`` bars: close / close n bars ago - 1."""
    _check_window(n)
    return close / close.shift(n) - 1.0


def momentum(close: pd.Series, lookback: int = 252, skip: int = 21) -> pd.Series:
    """Return from ``lookback`` to ``skip`` bars ago (the classic 12-1 month
    momentum): it leaves out the last month, which tends to reverse."""
    _check_window(lookback)
    if not 0 <= skip < lookback:
        raise ValueError(f"skip {skip} must be in [0, lookback)")
    return close.shift(skip) / close.shift(lookback) - 1.0


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-balance volume from the first bar (level is relative to the start;
    compare its changes, not its value across different start dates)."""
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume.fillna(0.0)).cumsum()


def volume_ratio(volume: pd.Series, n: int = 20) -> pd.Series:
    """Volume relative to its average over the last ``n`` bars."""
    _check_window(n)
    avg = volume.rolling(n, min_periods=n).mean()
    return volume / avg.where(avg != 0)


def vwma(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 20
) -> pd.Series:
    """Volume-weighted average of the typical price (high + low + close) / 3,
    as stockstats computes it. NaN where the window traded no volume (stockstats
    reports 0 there, which would read as a price)."""
    _check_window(n)
    typical = (high + low + close) / 3.0
    num = (typical * volume).rolling(n, min_periods=n).sum()
    den = volume.rolling(n, min_periods=n).sum()
    return num / den.where(den != 0)


def dmi(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.DataFrame:
    """+DI, -DI and Wilder's ADX (trend strength, 0-100, direction-free)."""
    _check_window(n)
    up = high.diff()
    down = -low.diff()
    plus = np.where((up > down) & (up > 0), up, 0.0)
    minus = np.where((down > up) & (down > 0), down, 0.0)
    # The first bar has no previous one: no move. Later, a missing high or low
    # is an unknown move, not a zero one.
    unknown = (up.isna() | down.isna()).to_numpy(copy=True)  # writable under copy-on-write
    unknown[:1] = False
    plus[unknown] = np.nan
    minus[unknown] = np.nan
    tr_s = smma(true_range(high, low, close), n).to_numpy()
    pdm_s = smma(pd.Series(plus, index=close.index), n).to_numpy()
    mdm_s = smma(pd.Series(minus, index=close.index), n).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        # No range at all (a constant price) is no directional movement: DI 0.
        plus_di = np.where(tr_s == 0, 0.0, 100.0 * pdm_s / tr_s)
        minus_di = np.where(tr_s == 0, 0.0, 100.0 * mdm_s / tr_s)
        total = plus_di + minus_di
        dx = np.where(total == 0, 0.0, 100.0 * np.abs(plus_di - minus_di) / total)
    dx[np.isnan(total)] = np.nan
    plus_di = pd.Series(plus_di, index=close.index)
    minus_di = pd.Series(minus_di, index=close.index)
    adx = smma(pd.Series(dx, index=close.index), n)
    return pd.DataFrame(
        {
            "plus_di": _masked(plus_di, n),
            "minus_di": _masked(minus_di, n),
            "adx": _masked(adx, 2 * n - 1),
        },
        index=close.index,
    )


def donchian(high: pd.Series, low: pd.Series, n: int = 20) -> pd.DataFrame:
    """Highest high and lowest low of the last ``n`` bars *including* today.

    A breakout rule compares today's close with yesterday's channel
    (``shift(1)``); comparing with today's channel would let today's high
    confirm itself.
    """
    _check_window(n)
    return pd.DataFrame(
        {
            "upper": high.rolling(n, min_periods=n).max(),
            "lower": low.rolling(n, min_periods=n).min(),
        },
        index=high.index,
    )
