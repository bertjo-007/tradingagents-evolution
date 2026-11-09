"""Deterministic risk checks on every new buy order.

Each check computes a value from point-in-time data (nothing after the order's
decision date) and compares it with a limit from ``config.RiskSettings``:

| check              | refuses or shrinks when                                     |
|--------------------|-------------------------------------------------------------|
| data               | no usable price or history for the instrument at the date   |
| drawdown           | the account is further below its peak than ``max_drawdown`` |
| daily_loss         | the last day lost more than ``max_daily_loss``              |
| open_positions     | a new position would exceed ``max_open_positions``          |
| volatility         | the instrument's 20-bar volatility exceeds ``max_volatility`` |
| liquidity          | its 20-bar average traded value is below ``min_dollar_volume``; the order is shrunk to ``max_adv_fraction`` of it |
| position_weight    | the position would exceed ``max_position_weight`` (shrunk)  |
| gross_exposure     | all positions would exceed ``max_gross_exposure`` (shrunk)  |
| correlation        | positions correlated above ``correlation_threshold`` would together exceed ``max_correlated_weight`` (shrunk) |
| stop               | ``require_stop`` and the order has no stop; a stop at or above the price |
| risk_per_trade     | the loss at the stop would exceed ``max_risk_per_trade`` (shrunk) |

A value that cannot be computed fails its check: no trade is the safe answer
to an unknown risk. The engine can refuse or shrink an order, never enlarge
it, and it takes no argument through which an assessment — an agent's, or
anyone's — could loosen a limit. Orders that reduce a position never come here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tradingagents_quant.config import RiskSettings
from tradingagents_quant.indicators import core
from tradingagents_quant.sim.engine import FilterDecision, OrderContext
from tradingagents_quant.sim.execution import CostModel
from tradingagents_quant.sim.sizing import shares_for


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    value: float | None
    limit: float | None
    detail: str
    max_shares: float | None = None  # a size limit: the most this check allows

    def to_dict(self) -> dict[str, Any]:
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                for k, v in asdict(self).items()}


@dataclass(frozen=True)
class Verdict:
    date: str
    symbol: str
    reason: str  # the order's reason: entry | rebalance | dividend_reinvest
    requested: float
    approved: float
    checks: tuple[Check, ...]
    limits_fingerprint: str
    reasons: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.approved > 0

    def to_dict(self) -> dict[str, Any]:
        return {"date": self.date, "symbol": self.symbol, "reason": self.reason,
                "requested": self.requested, "approved": self.approved,
                "accepted": self.accepted, "reasons": list(self.reasons),
                "limits_fingerprint": self.limits_fingerprint,
                "checks": [c.to_dict() for c in self.checks]}


@dataclass
class MarketHistory:
    """Per-instrument series the checks read, computed causally once."""

    close: dict[str, pd.Series] = field(default_factory=dict)
    volatility: dict[str, pd.Series] = field(default_factory=dict)
    dollar_volume: dict[str, pd.Series] = field(default_factory=dict)
    returns: dict[str, pd.Series] = field(default_factory=dict)

    @classmethod
    def from_markets(cls, markets: Mapping[str, pd.DataFrame], *, vol_window: int = 20,
                     liquidity_window: int = 20) -> MarketHistory:
        h = cls()
        for s, m in markets.items():
            close = m["close"].astype(float)
            h.close[s] = close
            h.volatility[s] = core.historical_volatility(close, vol_window)
            traded = close * m["volume"].astype(float)
            h.dollar_volume[s] = traded.rolling(liquidity_window, min_periods=liquidity_window).mean()
            h.returns[s] = close / close.shift(1) - 1.0
        return h

    @staticmethod
    def _at(series: pd.Series | None, day: pd.Timestamp) -> float:
        if series is None:
            return math.nan
        upto = series.loc[:day]
        return float(upto.iloc[-1]) if len(upto) else math.nan

    def vol(self, s: str, day: pd.Timestamp) -> float:
        return self._at(self.volatility.get(s), day)

    def adv(self, s: str, day: pd.Timestamp) -> float:
        return self._at(self.dollar_volume.get(s), day)

    def correlation(self, a: str, b: str, day: pd.Timestamp, window: int) -> float:
        ra, rb = self.returns.get(a), self.returns.get(b)
        if ra is None or rb is None:
            return math.nan
        joined = pd.concat([ra.loc[:day], rb.loc[:day]], axis=1, join="inner").dropna().tail(window)
        if len(joined) < window:
            return math.nan
        return float(np.corrcoef(joined.iloc[:, 0], joined.iloc[:, 1])[0, 1])


def _limits_fingerprint(limits: RiskSettings) -> str:
    canonical = json.dumps(limits.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


class RiskEngine:
    def __init__(self, limits: RiskSettings, history: MarketHistory, costs: CostModel) -> None:
        self.limits = limits
        self.history = history
        self.costs = costs
        self.fingerprint = _limits_fingerprint(limits)

    def size(self, weight: float, equity: float, price: float) -> float:
        """The same share arithmetic as the engine (``sim.sizing.shares_for``)."""
        return shares_for(weight, equity, price, self.costs)

    def _shares_within(self, budget_value: float, price: float) -> float:
        return max(0.0, self.costs.round_shares(budget_value / (price * (1.0 + self.costs.adverse))))

    def evaluate(self, ctx: OrderContext) -> Verdict:
        L, s, day = self.limits, ctx.symbol, ctx.date
        wanted, price, equity = float(ctx.order.shares), float(ctx.price), float(ctx.equity)
        checks: list[Check] = []

        if not (math.isfinite(price) and price > 0 and math.isfinite(equity) and equity > 0):
            checks.append(Check("data", False, None, None,
                                f"no usable price ({price}) or equity ({equity})"))
            return self._verdict(ctx, checks)
        order_value = wanted * price * (1.0 + self.costs.adverse)
        held_shares, _ = ctx.holdings.get(s, (0.0, price))
        held_value = held_shares * price
        others = {k: v for k, v in ctx.holdings.items() if k != s}
        gross_value = held_value + sum(q * c for q, c in others.values() if math.isfinite(c))
        is_new = held_shares == 0
        is_entry_like = ctx.order.reason != "dividend_reinvest"

        # Account state: stop adding risk to an account in trouble.
        dd = 1.0 - equity / ctx.peak_equity if ctx.peak_equity > 0 else math.nan
        checks.append(Check("drawdown", math.isfinite(dd) and dd < L.max_drawdown, dd,
                            L.max_drawdown, f"account {dd:.1%} below its peak"
                            if math.isfinite(dd) else "drawdown not computable"))
        loss = 1.0 - equity / ctx.previous_equity if ctx.previous_equity > 0 else math.nan
        checks.append(Check("daily_loss", math.isfinite(loss) and loss < L.max_daily_loss, loss,
                            L.max_daily_loss, f"last day's change {-loss:+.1%}"
                            if math.isfinite(loss) else "daily change not computable"))
        if is_new:
            n_open = len(ctx.holdings)
            checks.append(Check("open_positions", n_open < L.max_open_positions, float(n_open),
                                float(L.max_open_positions), f"{n_open} positions open"))

        # The instrument.
        vol = self.history.vol(s, day)
        checks.append(Check("volatility", math.isfinite(vol) and vol <= L.max_volatility, vol,
                            L.max_volatility, f"20-bar volatility {vol:.1%}"
                            if math.isfinite(vol) else "volatility not computable"))
        adv = self.history.adv(s, day)
        if not math.isfinite(adv):
            checks.append(Check("liquidity", False, None, L.min_dollar_volume,
                                "average traded value not computable"))
        else:
            cap = self._shares_within(L.max_adv_fraction * adv, price)
            checks.append(Check("liquidity", adv >= L.min_dollar_volume, adv, L.min_dollar_volume,
                                f"20-bar average traded value {adv:,.0f}", max_shares=cap))

        # The portfolio.
        checks.append(Check(
            "position_weight", True, (held_value + order_value) / equity, L.max_position_weight,
            "position after the order",
            max_shares=self._shares_within(L.max_position_weight * equity - held_value, price)))
        checks.append(Check(
            "gross_exposure", True, (gross_value + order_value) / equity, L.max_gross_exposure,
            "all positions after the order",
            max_shares=self._shares_within(L.max_gross_exposure * equity - gross_value, price)))
        correlated_value, unknown = held_value, []
        for other, (q, c) in others.items():
            rho = self.history.correlation(s, other, day, L.correlation_window)
            if not math.isfinite(rho):
                unknown.append(other)
            elif rho >= L.correlation_threshold:
                correlated_value += q * c
        if unknown:
            checks.append(Check("correlation", False, None, L.correlation_threshold,
                                f"correlation with {', '.join(sorted(unknown))} not computable"))
        else:
            checks.append(Check(
                "correlation", True, (correlated_value + order_value) / equity,
                L.max_correlated_weight, "correlated positions after the order",
                max_shares=self._shares_within(L.max_correlated_weight * equity - correlated_value,
                                               price)))

        # The trade itself.
        if ctx.stop is None:
            checks.append(Check("stop", not (L.require_stop and is_entry_like), None, None,
                                "no stop level: the loss of this trade is not bounded"
                                if not L.require_stop else "a stop is required"))
        elif not ctx.stop < price:
            checks.append(Check("stop", False, ctx.stop, price, f"stop {ctx.stop:.4f} is not "
                                f"below the price {price:.4f}"))
        else:
            per_share = price * (1.0 + self.costs.adverse) - ctx.stop
            cap = self.costs.round_shares(L.max_risk_per_trade * equity / per_share)
            checks.append(Check("risk_per_trade", True, wanted * per_share / equity,
                                L.max_risk_per_trade, "loss at the stop", max_shares=cap))
        return self._verdict(ctx, checks)

    def _verdict(self, ctx: OrderContext, checks: list[Check]) -> Verdict:
        wanted = float(ctx.order.shares)
        failed = [c for c in checks if not c.passed]
        approved = 0.0
        reasons = [f"{c.name}: {c.detail}" for c in failed]
        if not failed:
            caps = [c.max_shares for c in checks if c.max_shares is not None]
            approved = min([wanted, *caps])
            approved = max(0.0, self.costs.round_shares(approved))
            for c in checks:
                if c.max_shares is not None and c.max_shares < wanted:
                    reasons.append(f"{c.name}: at most {c.max_shares:g} shares ({c.detail})")
            if approved <= 0:
                reasons.append("no size fits every limit")
        return Verdict(ctx.date.strftime("%Y-%m-%d"), ctx.symbol, ctx.order.reason, wanted,
                       approved, tuple(checks), self.fingerprint, tuple(reasons))

    def as_filter(self):
        """The engine's order filter: the verdict decides what may be ordered."""

        def decide(ctx: OrderContext) -> FilterDecision:
            v = self.evaluate(ctx)
            return FilterDecision(v.approved, v.reasons, v)

        return decide
