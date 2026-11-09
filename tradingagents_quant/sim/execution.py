"""Fill prices and commissions of simulated orders.

A buy fills above its reference price and a sell below it, by half the spread
plus slippage. Commission: per share with a minimum and a cap (as a per-share
broker charges), plus an optional percentage. All configurable; the defaults
in ``config.CostSettings`` are deliberately not optimistic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tradingagents_quant.config import CostSettings


@dataclass(frozen=True)
class CostModel:
    commission_per_share: float = 0.005
    commission_min: float = 1.0
    commission_max_pct: float = 0.01
    commission_bps: float = 0.0
    spread_bps: float = 2.0
    slippage_bps: float = 2.0
    fractional_shares: bool = False

    @classmethod
    def from_settings(cls, s: CostSettings) -> CostModel:
        return cls(s.commission_per_share, s.commission_min, s.commission_max_pct,
                   s.commission_bps, s.spread_bps, s.slippage_bps, s.fractional_shares)

    @classmethod
    def free(cls) -> CostModel:
        """No costs at all — for tests and for showing what costs take away."""
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)

    @property
    def adverse(self) -> float:
        """Fraction a fill moves against the order: half spread + slippage."""
        return (self.spread_bps / 2.0 + self.slippage_bps) / 1e4

    def fill_price(self, side: str, reference: float) -> float:
        if side == "buy":
            return reference * (1.0 + self.adverse)
        if side == "sell":
            return reference * (1.0 - self.adverse)
        raise ValueError(f"unknown side {side!r}")

    def commission(self, shares: float, price: float) -> float:
        if shares <= 0:
            return 0.0
        value = shares * price
        fee = max(self.commission_min, shares * self.commission_per_share)
        fee += value * self.commission_bps / 1e4
        if self.commission_max_pct > 0:
            fee = min(fee, value * self.commission_max_pct)
        return fee

    def round_shares(self, shares: float) -> float:
        if shares <= 0:
            return 0.0
        if self.fractional_shares:
            return math.floor(shares * 1e6) / 1e6
        return float(math.floor(shares))

    def affordable(self, cash: float, price: float, wanted: float) -> float:
        """The most of ``wanted`` shares that cash pays for, commission included."""
        if price <= 0 or cash <= 0:
            return 0.0
        shares = self.round_shares(min(wanted, cash / price))
        step = 1e-6 if self.fractional_shares else 1.0
        while shares > 0 and shares * price + self.commission(shares, price) > cash:
            shares = self.round_shares(shares - step)
        return shares
