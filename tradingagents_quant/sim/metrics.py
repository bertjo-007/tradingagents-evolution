"""Performance figures of a simulation.

Annualized figures (CAGR, volatility, Sharpe, Sortino, annual turnover) are
``None`` when the window is shorter than ``min_annualization_bars``: a Sharpe
ratio from three months of data says more about the three months than about
the strategy. Ratios with a zero denominator are ``None`` as well, never a
large number standing in for infinity.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from tradingagents_quant.sim.types import SimulationResult

PERIODS_PER_YEAR = 252


def _round(value: float | None, digits: int = 10) -> float | None:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return round(float(value), digits)


def equity_returns(equity: pd.Series, initial_cash: float) -> pd.Series:
    """Daily returns, the first measured against the starting cash."""
    base = pd.concat([pd.Series([initial_cash]), equity.reset_index(drop=True)], ignore_index=True)
    returns = (base / base.shift(1) - 1.0).iloc[1:]
    returns.index = equity.index
    return returns


def drawdown(equity: pd.Series, initial_cash: float) -> pd.Series:
    peak = np.maximum.accumulate(np.concatenate([[initial_cash], equity.to_numpy()]))[1:]
    return pd.Series(equity.to_numpy() / peak - 1.0, index=equity.index)


def _longest_underwater(dd: pd.Series) -> int:
    longest = current = 0
    for value in dd.to_numpy():
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return longest


def compute(
    result: SimulationResult,
    *,
    initial_cash: float,
    risk_free_rate: float = 0.0,
    min_annualization_bars: int = 252,
) -> dict[str, Any]:
    equity = result.equity
    n = len(equity)
    years = n / PERIODS_PER_YEAR
    annualize = n >= min_annualization_bars
    returns = equity_returns(equity, initial_cash)
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / PERIODS_PER_YEAR) - 1.0
    excess = returns - rf_daily

    std = float(returns.std(ddof=1)) if n > 1 else math.nan
    downside = float(np.sqrt(np.mean(np.minimum(excess.to_numpy(), 0.0) ** 2))) if n else math.nan
    dd = drawdown(equity, initial_cash)

    closed = [t for t in result.trades if t.closed]
    wins = [t.pnl for t in closed if t.pnl > 0]
    losses = [t.pnl for t in closed if t.pnl <= 0]
    traded = sum(f.shares * f.price for f in result.fills)
    mean_equity = float(equity.mean())

    final = float(equity.iloc[-1])
    metrics: dict[str, Any] = {
        "start": equity.index[0].strftime("%Y-%m-%d"),
        "end": equity.index[-1].strftime("%Y-%m-%d"),
        "bars": n,
        "years": _round(years, 4),
        "initial_cash": initial_cash,
        "final_equity": _round(final, 6),
        "total_return": _round(final / initial_cash - 1.0),
        "cagr": _round((final / initial_cash) ** (1.0 / years) - 1.0) if annualize and final > 0 else None,
        "volatility": _round(std * math.sqrt(PERIODS_PER_YEAR)) if annualize and std > 0 else None,
        "sharpe": (_round(float(excess.mean()) / std * math.sqrt(PERIODS_PER_YEAR))
                   if annualize and std > 0 else None),
        "sortino": (_round(float(excess.mean()) / downside * math.sqrt(PERIODS_PER_YEAR))
                    if annualize and downside > 0 else None),
        "max_drawdown": _round(float(dd.min())),
        "max_drawdown_date": dd.idxmin().strftime("%Y-%m-%d") if float(dd.min()) < 0 else None,
        "longest_drawdown_bars": _longest_underwater(dd),
        "trades": len(closed),
        "open_trade": any(not t.closed for t in result.trades),
        "win_rate": _round(len(wins) / len(closed)) if closed else None,
        "profit_factor": _round(sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else None,
        "avg_win": _round(float(np.mean(wins)), 6) if wins else None,
        "avg_loss": _round(float(np.mean(losses)), 6) if losses else None,
        "avg_trade_return": _round(float(np.mean([t.return_pct for t in closed]))) if closed else None,
        "avg_holding_bars": _round(float(np.mean([t.holding_bars for t in closed])), 4) if closed else None,
        "exposure": _round(float((result.shares > 0).mean())),
        "turnover": _round(traded / mean_equity) if mean_equity > 0 else None,
        "turnover_per_year": (_round(traded / mean_equity / years)
                              if annualize and mean_equity > 0 else None),
        "commissions": _round(result.commissions, 6),
        "spread_and_slippage": _round(result.friction, 6),
        "total_costs": _round(result.commissions + result.friction, 6),
        "dividends": _round(result.dividends, 6),
        "rejections": len(result.rejections),
        "exit_reasons": {r: sum(1 for t in closed if t.exit_reason == r)
                         for r in sorted({t.exit_reason for t in closed})},
    }
    if not annualize:
        metrics["annualization_note"] = (
            f"{n} bars < {min_annualization_bars}: annualized figures not reported")
    return metrics


def compare(strategy: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
    """Strategy minus benchmark, for the figures where a difference means something."""
    out: dict[str, Any] = {}
    for key in ("total_return", "cagr", "volatility", "sharpe", "sortino", "max_drawdown"):
        a, b = strategy.get(key), benchmark.get(key)
        out[f"{key}_vs_benchmark"] = _round(a - b) if a is not None and b is not None else None
    return out


def by_regime(result: SimulationResult, initial_cash: float) -> dict[str, dict[str, Any]]:
    """Returns split by the regime that was known while the position was held:
    the return of bar t belongs to the label of bar t-1."""
    if result.regimes is None:
        return {}
    returns = equity_returns(result.equity, initial_cash)
    label = result.regimes["regime"].shift(1).fillna("unknown")
    held = (result.shares.shift(1).fillna(0.0) > 0)
    out = {}
    for regime in sorted(set(label)):
        mask = label == regime
        r = returns[mask]
        out[regime] = {
            "bars": int(mask.sum()),
            "share_of_bars": _round(float(mask.mean()), 6),
            "exposure": _round(float(held[mask].mean()), 6),
            "mean_daily_return": _round(float(r.mean())) if len(r) else None,
            "compounded_return": _round(float(np.prod(1.0 + r.to_numpy()) - 1.0)) if len(r) else None,
        }
    return out
