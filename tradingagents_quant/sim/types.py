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
    symbol: str = ""
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
    reason: str  # insufficient_cash | reduced_for_cash | no_open_price | no_bar | stop_not_computable
    detail: str
    symbol: str = ""


@dataclass
class Trade:
    """A round trip in one instrument: from flat to a position and back to flat."""

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
    symbol: str = ""

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
class _Costs:
    fills: list[Fill]

    @property
    def commissions(self) -> float:
        return sum(f.commission for f in self.fills)

    @property
    def friction(self) -> float:
        return sum(f.friction for f in self.fills)


@dataclass
class SimulationResult(_Costs):
    """One instrument."""

    symbol: str = ""
    equity: pd.Series = field(default_factory=pd.Series)
    cash: pd.Series = field(default_factory=pd.Series)
    shares: pd.Series = field(default_factory=pd.Series)
    target: pd.Series = field(default_factory=pd.Series)
    trades: list[Trade] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    dividends: float = 0.0
    notes: list[str] = field(default_factory=list)
    regimes: pd.DataFrame | None = None
    market_value: pd.Series | None = None
    label: str = PAPER_LABEL

    @property
    def invested(self) -> pd.Series:
        return self.shares > 0

    @property
    def gross_exposure(self) -> pd.Series | None:
        if self.market_value is None:
            return None
        return self.market_value / self.equity


@dataclass
class PortfolioResult(_Costs):
    """Several instruments sharing one cash account."""

    symbols: tuple[str, ...] = ()
    equity: pd.Series = field(default_factory=pd.Series)
    cash: pd.Series = field(default_factory=pd.Series)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)  # shares per symbol
    market_value: pd.DataFrame = field(default_factory=pd.DataFrame)
    weights: pd.DataFrame = field(default_factory=pd.DataFrame)  # targets after portfolio limits
    trades: list[Trade] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    dividends: float = 0.0
    notes: list[str] = field(default_factory=list)
    regimes: pd.DataFrame | None = None  # of the calendar symbol: the market's regime
    label: str = PAPER_LABEL

    @property
    def invested(self) -> pd.Series:
        return (self.positions > 0).any(axis=1)

    @property
    def gross_exposure(self) -> pd.Series:
        return self.market_value.sum(axis=1) / self.equity

    def single(self, symbol: str, target: pd.Series | None = None) -> SimulationResult:
        """This result as the single-instrument record (for one-symbol runs)."""
        return SimulationResult(
            fills=self.fills, symbol=symbol, equity=self.equity, cash=self.cash,
            shares=self.positions[symbol].rename("shares"),
            target=(self.weights[symbol] if target is None else target).rename("target"),
            trades=self.trades, rejections=self.rejections, dividends=self.dividends,
            notes=self.notes, regimes=self.regimes, market_value=self.market_value[symbol],
        )
