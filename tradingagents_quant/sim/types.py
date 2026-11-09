"""Records the simulator produces."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from tradingagents_quant import PAPER_LABEL

EXIT_REASONS = ("exit_signal", "stop_loss", "take_profit", "time_stop", "open")


@dataclass(frozen=True)
class Fill:
    date: pd.Timestamp
    side: str  # buy | sell
    shares: float
    reference: float  # the price the order referred to (open, stop or target level)
    price: float  # what was paid or received per share, spread and slippage included
    commission: float
    reason: str  # entry | rebalance | exit_signal | stop_loss | take_profit | time_stop
    label: str = PAPER_LABEL

    @property
    def cash_flow(self) -> float:
        value = self.shares * self.price
        return (-value if self.side == "buy" else value) - self.commission

    @property
    def friction(self) -> float:
        """Spread and slippage paid on this fill, in cash."""
        return abs(self.price - self.reference) * self.shares


@dataclass(frozen=True)
class Rejection:
    date: pd.Timestamp
    reason: str  # insufficient_cash | no_open_price
    detail: str


@dataclass
class Trade:
    """A round trip: from flat to a position and back to flat."""

    entry_date: pd.Timestamp
    entry_price: float
    entry_regime: str
    fills: list[Fill] = field(default_factory=list)
    dividends: float = 0.0
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str = "open"
    holding_bars: int = 0
    mark_value: float = 0.0  # market value of an open trade at the last close

    @property
    def invested(self) -> float:
        return sum(f.shares * f.price + f.commission for f in self.fills if f.side == "buy")

    @property
    def pnl(self) -> float:
        return sum(f.cash_flow for f in self.fills) + self.dividends + self.mark_value

    @property
    def return_pct(self) -> float:
        return self.pnl / self.invested if self.invested else 0.0

    @property
    def closed(self) -> bool:
        return self.exit_date is not None


@dataclass
class SimulationResult:
    symbol: str
    equity: pd.Series
    cash: pd.Series
    shares: pd.Series
    target: pd.Series
    fills: list[Fill]
    trades: list[Trade]
    rejections: list[Rejection]
    dividends: float
    notes: list[str]
    regimes: pd.DataFrame | None = None
    label: str = PAPER_LABEL

    @property
    def commissions(self) -> float:
        return sum(f.commission for f in self.fills)

    @property
    def friction(self) -> float:
        return sum(f.friction for f in self.fills)
