"""The simulation loop, for one instrument or several sharing one cash account.

Timing, which is what keeps it free of look-ahead:

1. **Morning of bar t** — a holder at the close of t-1 receives that day's
   dividend (ex-date entitlement). Then the orders decided at the close of t-1
   fill at the **open of t**, plus half spread and slippage: sells first, so
   their proceeds can pay for the buys.
2. **During bar t** — stop and target levels are checked against the bar's low
   and high. A position held overnight that opens through its stop fills at
   the (worse) open. If one bar reaches both stop and target, the stop is
   assumed: daily bars do not tell which came first, and the simulator does not
   pick the flattering order.
3. **Close of bar t** — equity is marked at the close. The target weights
   decided from bars up to t become the orders for the open of t+1. Sizes use
   the close of t, the last price known when deciding; if the cash does not
   cover the buys at the next open, they shrink (in proportion, when several
   compete) and the cut is recorded.

Portfolio limits: the weights of all instruments together never exceed
``max_gross`` (scaled down in proportion), no single one exceeds
``max_weight``. An instrument without a bar on a calendar day (before its first
trading day, or a gap) is neither traded nor decided on that day; a position
in it keeps its last close for valuation, which the notes say.

The loop is a ``Book`` stepped one day at a time. A backtest steps it over
history; paper trading (``paper/``) steps the same book over each new day and
stores its state in between — one engine, not two. An optional
``order_filter`` sees every new buy order before it is placed: the
deterministic risk engine, which may refuse or shrink it. Orders that reduce
risk (sells) are never filtered.

Nothing in the loop reads a bar after the current one. ``tests/quant/
test_engine_leakage.py`` checks that by truncation, by an oracle that is
allowed to cheat (it must look absurdly good) and by delaying that oracle a
day (its edge must vanish).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tradingagents_quant.indicators import core
from tradingagents_quant.sim.execution import CostModel
from tradingagents_quant.sim.sizing import shares_for
from tradingagents_quant.sim.types import Fill, PortfolioResult, Rejection, SimulationResult, Trade
from tradingagents_quant.strategy.spec import ExitSpec


@dataclass(frozen=True)
class Order:
    side: str
    shares: float
    reason: str
    decided: str  # date of the decision, YYYY-MM-DD
    atr: float  # ATR at the decision, for ATR-based exit levels
    regime: str = "unknown"  # regime at the decision, recorded on the trade it opens


@dataclass(frozen=True)
class OrderContext:
    """What the order filter (the risk engine) is shown about a new buy order."""

    date: pd.Timestamp
    symbol: str
    order: Order
    price: float  # close at the decision
    equity: float
    cash: float
    peak_equity: float
    previous_equity: float
    holdings: Mapping[str, tuple[float, float]]  # symbol -> (shares, last close)
    stop: float | None  # where the stop would sit for an entry at this price


@dataclass(frozen=True)
class FilterDecision:
    shares: float  # what may be ordered: 0 refuses, less than asked shrinks
    reasons: tuple[str, ...] = ()
    record: Any = None  # the full verdict, for the audit trail


OrderFilter = Callable[[OrderContext], FilterDecision]


@dataclass
class _Position:
    shares: float = 0.0
    trade: Trade | None = None
    stop: float | None = None
    take: float | None = None
    stopped_out: bool = False
    bars_held: int = 0
    last_close: float = math.nan
    pending: Order | None = None
    dividend_cash: float = 0.0  # received, not yet reinvested (reinvest_dividends only)


@dataclass
class Bar:
    """One calendar day across the book's instruments (arrays in symbol order)."""

    has_bar: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    dividend: np.ndarray
    atr: np.ndarray
    regime: np.ndarray
    weight: np.ndarray  # target weight decided at this close (NaN: none)


def _levels(exits: ExitSpec, entry: float, atr: float) -> tuple[float | None, float | None]:
    stop = take = None
    if exits.stop_loss == "percent":
        stop = entry * (1.0 - exits.stop_value)
    elif exits.stop_loss == "atr":
        stop = entry - exits.stop_value * atr
    if exits.take_profit == "percent":
        take = entry * (1.0 + exits.take_value)
    elif exits.take_profit == "atr":
        take = entry + exits.take_value * atr
    return stop, take


@dataclass
class Book:
    """The state of a simulated account and the rules that move it one day on."""

    symbols: tuple[str, ...]
    costs: CostModel
    exits: dict[str, ExitSpec]
    cash: float
    rebalance_band: float = 0.2
    reinvest_dividends: bool = False
    order_filter: OrderFilter | None = None
    pos: dict[str, _Position] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    filter_records: list[tuple[pd.Timestamp, str, Any]] = field(default_factory=list)
    dividends: float = 0.0
    notes: list[str] = field(default_factory=list)
    carried: set[str] = field(default_factory=set)
    equity: float = math.nan
    peak_equity: float = math.nan
    previous_equity: float = math.nan

    def __post_init__(self) -> None:
        for s in self.symbols:
            self.pos.setdefault(s, _Position())
            self.exits.setdefault(s, ExitSpec())
        if math.isnan(self.equity):
            self.equity = self.peak_equity = self.previous_equity = self.cash

    # -- one day -------------------------------------------------------------

    def _fill(self, day: pd.Timestamp, s: str, side: str, qty: float, reference: float,
              reason: str) -> Fill:
        price = self.costs.fill_price(side, reference)
        f = Fill(day, side, qty, reference, price, self.costs.commission(qty, price), reason, s)
        self.cash += f.cash_flow
        p = self.pos[s]
        p.shares += qty if side == "buy" else -qty
        if p.shares <= 1e-12:
            p.shares = 0.0
        self.fills.append(f)
        return f

    def _close_trade(self, day: pd.Timestamp, s: str, f: Fill, reason: str) -> None:
        p = self.pos[s]
        assert p.trade is not None
        p.trade.exit_date, p.trade.exit_price, p.trade.exit_reason = day, f.price, reason
        p.trade.holding_bars = p.bars_held
        self.trades.append(p.trade)
        p.trade, p.stop, p.take = None, None, None

    def _buy(self, day: pd.Timestamp, s: str, order: Order, open_price: float, price: float,
             qty: float) -> None:
        if qty < order.shares:
            self.rejections.append(Rejection(
                day, "insufficient_cash" if qty == 0 else "reduced_for_cash",
                f"wanted {order.shares:g} shares at {price:.4f}, cash pays {qty:g}", s))
        if qty <= 0:
            return
        f = self._fill(day, s, "buy", qty, open_price, order.reason)
        p = self.pos[s]
        if order.reason == "dividend_reinvest":
            p.dividend_cash = max(0.0, p.dividend_cash + f.cash_flow)
        if p.trade is None:
            p.trade = Trade(day, f.price, order.regime, symbol=s)
            p.stop, p.take = _levels(self.exits[s], f.price, order.atr)
            p.bars_held = 0
        p.trade.fills.append(f)

    def step(self, day: pd.Timestamp, bar: Bar) -> float:
        """Advance the book by one calendar day; returns the closing equity."""
        symbols = self.symbols
        # 1. Dividends to whoever held at the previous close.
        for j, s in enumerate(symbols):
            p = self.pos[s]
            if bar.has_bar[j] and p.shares > 0 and bar.dividend[j] > 0:
                amount = p.shares * bar.dividend[j]
                self.cash += amount
                self.dividends += amount
                if self.reinvest_dividends:
                    p.dividend_cash += amount
                if p.trade is not None:
                    p.trade.dividends += amount
        held_overnight = {s: self.pos[s].shares > 0 for s in symbols}

        # 2. Orders decided at the previous close fill at this open: sells first.
        buys: list[tuple[str, Order, float, float]] = []
        for j, s in enumerate(symbols):
            p = self.pos[s]
            order, p.pending = p.pending, None
            if order is None:
                continue
            if not bar.has_bar[j]:
                self.rejections.append(Rejection(day, "no_bar", f"{order.side} not filled", s))
            elif math.isnan(bar.open[j]):
                self.rejections.append(Rejection(day, "no_open_price", f"{order.side} not filled",
                                                 s))
            elif order.side == "sell":
                qty = min(order.shares, p.shares)
                if qty > 0:
                    f = self._fill(day, s, "sell", qty, bar.open[j], order.reason)
                    assert p.trade is not None
                    p.trade.fills.append(f)
                    if p.shares == 0:
                        self._close_trade(day, s, f, order.reason)
            else:
                buys.append((s, order, bar.open[j], self.costs.fill_price("buy", bar.open[j])))
        if len(buys) == 1:
            s, order, open_price, price = buys[0]
            self._buy(day, s, order, open_price, price,
                      self.costs.affordable(self.cash, price, order.shares))
        elif buys:
            need = sum(q.shares * px + self.costs.commission(q.shares, px) for _, q, _, px in buys)
            factor = min(1.0, self.cash / need) if need > 0 else 0.0
            for s, order, open_price, price in buys:
                self._buy(day, s, order, open_price, price,
                          self.costs.affordable(self.cash, price, order.shares * factor))

        # 3. Stops and targets during the bar.
        for j, s in enumerate(symbols):
            p = self.pos[s]
            if not bar.has_bar[j] or p.shares <= 0 or (p.stop is None and p.take is None):
                continue
            o, h, lo = bar.open[j], bar.high[j], bar.low[j]
            reference = reason = None
            if math.isnan(h) or math.isnan(lo):
                self.notes.append(f"{day.date()} {s}: no high/low, stop and target not checked")
            elif held_overnight[s] and p.stop is not None and o <= p.stop:
                reference, reason = o, "stop_loss"  # gapped through the stop
            elif held_overnight[s] and p.take is not None and o >= p.take:
                reference, reason = o, "take_profit"  # gapped through the target
            elif p.stop is not None and lo <= p.stop:
                reference, reason = p.stop, "stop_loss"  # also when the target was hit
            elif p.take is not None and h >= p.take:
                reference, reason = p.take, "take_profit"
            if reason is not None:
                f = self._fill(day, s, "sell", p.shares, reference, reason)
                assert p.trade is not None
                p.trade.fills.append(f)
                self._close_trade(day, s, f, reason)
                p.stopped_out = self.exits[s].reentry == "after_signal_reset"

        # 4. Mark at the close.
        for j, s in enumerate(symbols):
            p = self.pos[s]
            if bar.has_bar[j]:
                p.last_close = bar.close[j]
                if p.shares > 0:
                    p.bars_held += 1
            elif p.shares > 0:
                self.carried.add(s)
        self.previous_equity = self.equity
        self.equity = self.cash + sum(p.shares * p.last_close for p in self.pos.values()
                                      if p.shares > 0)
        self.peak_equity = max(self.peak_equity, self.equity)

        # 5. Decide the orders for the next open.
        for j, s in enumerate(symbols):
            if not bar.has_bar[j]:
                continue
            p, ex = self.pos[s], self.exits[s]
            w = 0.0 if math.isnan(bar.weight[j]) else float(bar.weight[j])
            decided = day.strftime("%Y-%m-%d")
            atr, regime_now = float(bar.atr[j]), str(bar.regime[j])
            if p.stopped_out and p.shares == 0:
                if w <= 0:
                    p.stopped_out = False  # the signal has reset; the next entry may come
                w = 0.0
            if p.shares > 0 and ex.max_holding_bars and p.bars_held >= ex.max_holding_bars:
                p.pending = Order("sell", p.shares, "time_stop", decided, atr, regime_now)
                p.stopped_out = ex.reentry == "after_signal_reset"
            elif w <= 0 and p.shares > 0:
                p.pending = Order("sell", p.shares, "exit_signal", decided, atr, regime_now)
            elif w > 0:
                desired = shares_for(w, self.equity, bar.close[j], self.costs)
                if p.shares == 0 and desired > 0:
                    if "atr" in (ex.stop_loss, ex.take_profit) and math.isnan(atr):
                        # An exit level that cannot be computed is no trade (fail safe).
                        self.rejections.append(Rejection(day, "stop_not_computable",
                                                         "ATR not available yet; entry refused", s))
                    else:
                        self._place_buy(day, s, Order("buy", desired, "entry", decided, atr,
                                                      regime_now), bar.close[j])
                elif p.shares > 0 and abs(desired - p.shares) > self.rebalance_band * p.shares:
                    if desired > p.shares:
                        self._place_buy(day, s, Order("buy", desired - p.shares, "rebalance",
                                                      decided, atr, regime_now), bar.close[j])
                    else:
                        p.pending = Order("sell", p.shares - desired, "rebalance", decided, atr,
                                          regime_now)
            if (self.reinvest_dividends and p.pending is None and p.shares > 0
                    and p.dividend_cash > 0):
                qty = self.costs.round_shares(p.dividend_cash /
                                              (bar.close[j] * (1.0 + self.costs.adverse)))
                if qty > 0:
                    self._place_buy(day, s, Order("buy", qty, "dividend_reinvest", decided, atr,
                                                  regime_now), bar.close[j])
        return self.equity

    def _place_buy(self, day: pd.Timestamp, s: str, order: Order, price: float) -> None:
        """A buy order becomes pending — unless the order filter refuses it."""
        if self.order_filter is None:
            self.pos[s].pending = order
            return
        p = self.pos[s]
        stop = None
        if p.shares == 0:
            stop = _levels(self.exits[s], price * (1.0 + self.costs.adverse), order.atr)[0]
        elif p.stop is not None:
            stop = p.stop
        holdings = {sym: (q.shares, q.last_close) for sym, q in self.pos.items() if q.shares > 0}
        decision = self.order_filter(OrderContext(
            day, s, order, price, self.equity, self.cash, self.peak_equity, self.previous_equity,
            holdings, stop))
        self.filter_records.append((day, s, decision.record))
        allowed = min(order.shares, self.costs.round_shares(decision.shares))
        if allowed <= 0:
            self.rejections.append(Rejection(day, "risk_rejected",
                                             "; ".join(decision.reasons) or "refused", s))
            return
        if allowed < order.shares:
            self.rejections.append(Rejection(
                day, "risk_reduced",
                f"{order.shares:g} -> {allowed:g} shares: " + "; ".join(decision.reasons), s))
            order = Order(order.side, allowed, order.reason, order.decided, order.atr, order.regime)
        p.pending = order

    def finish(self) -> None:
        """Mark open trades at the last close (the book stays usable)."""
        for s in self.symbols:
            p = self.pos[s]
            if p.trade is not None:
                p.trade.holding_bars = p.bars_held
                p.trade.mark_value = p.shares * p.last_close

    def open_trades(self) -> list[Trade]:
        return [self.pos[s].trade for s in self.symbols if self.pos[s].trade is not None]

    # -- state ---------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """Everything needed to continue tomorrow (closed trades and fills are
        history and kept elsewhere; the open trade's fills travel with it)."""
        def fill(f: Fill) -> dict[str, Any]:
            d = asdict(f)
            d["date"] = f.date.strftime("%Y-%m-%d")
            return d

        positions = {}
        for s, p in self.pos.items():
            trade = None
            if p.trade is not None:
                trade = {**{k: v for k, v in asdict(p.trade).items() if k != "fills"},
                         "entry_date": p.trade.entry_date.strftime("%Y-%m-%d"),
                         "exit_date": None, "fills": [fill(f) for f in p.trade.fills]}
            positions[s] = {
                "shares": p.shares, "stop": p.stop, "take": p.take, "stopped_out": p.stopped_out,
                "bars_held": p.bars_held, "last_close": None if math.isnan(p.last_close)
                else p.last_close, "dividend_cash": p.dividend_cash,
                "pending": asdict(p.pending) if p.pending else None, "trade": trade,
            }
        return {"cash": self.cash, "dividends": self.dividends, "equity": self.equity,
                "peak_equity": self.peak_equity, "previous_equity": self.previous_equity,
                "positions": positions, "carried": sorted(self.carried)}

    @classmethod
    def from_state(cls, state: Mapping[str, Any], *, symbols: tuple[str, ...], costs: CostModel,
                   exits: dict[str, ExitSpec], rebalance_band: float, reinvest_dividends: bool,
                   order_filter: OrderFilter | None = None) -> Book:
        def fill(d: Mapping[str, Any]) -> Fill:
            return Fill(**{**d, "date": pd.Timestamp(d["date"])})

        book = cls(symbols, costs, dict(exits), float(state["cash"]), rebalance_band,
                   reinvest_dividends, order_filter, dividends=float(state["dividends"]),
                   carried=set(state.get("carried", [])), equity=float(state["equity"]),
                   peak_equity=float(state["peak_equity"]),
                   previous_equity=float(state["previous_equity"]))
        for s, d in state["positions"].items():
            trade = None
            if d["trade"] is not None:
                t = dict(d["trade"])
                t["entry_date"] = pd.Timestamp(t["entry_date"])
                t["fills"] = [fill(f) for f in t["fills"]]
                trade = Trade(**t)
            book.pos[s] = _Position(
                shares=float(d["shares"]), trade=trade, stop=d["stop"], take=d["take"],
                stopped_out=bool(d["stopped_out"]), bars_held=int(d["bars_held"]),
                last_close=math.nan if d["last_close"] is None else float(d["last_close"]),
                pending=Order(**d["pending"]) if d["pending"] else None,
                dividend_cash=float(d["dividend_cash"]))
        return book


# -- whole runs ------------------------------------------------------------------


def _validated(symbol: str, market: pd.DataFrame, target: pd.Series) -> pd.Series:
    if not target.index.equals(market.index):
        raise ValueError(f"{symbol}: target and market must share their index")
    weights = target.astype(float).fillna(0.0)
    if (weights < 0).any() or (weights > 1).any():
        raise ValueError(f"{symbol}: long-only, unlevered: target weights must lie in [0, 1]")
    return weights


def portfolio_weights(
    weights: pd.DataFrame, *, max_gross: float = 1.0, max_weight: float = 1.0
) -> pd.DataFrame:
    """Apply the portfolio limits row by row: scale all down in proportion when
    their sum exceeds ``max_gross``, then cap each at ``max_weight``. NaN (no
    bar that day) stays NaN."""
    gross = weights.sum(axis=1, skipna=True)
    scale = (max_gross / gross.where(gross > max_gross)).fillna(1.0)
    return weights.mul(scale, axis=0).clip(upper=max_weight)


def bars_for(
    markets: Mapping[str, pd.DataFrame], symbols: tuple[str, ...], calendar: pd.DatetimeIndex,
    weights: pd.DataFrame, exits_of: Mapping[str, ExitSpec],
    regime_of: Mapping[str, pd.DataFrame | None],
) -> list[Bar]:
    """The calendar as a list of day records for ``Book.step``."""
    n, k = len(calendar), len(symbols)
    has_bar = np.zeros((n, k), dtype=bool)
    o, h, lo, c, div, atr = (np.full((n, k), math.nan) for _ in range(6))
    labels = np.full((n, k), "unknown", dtype=object)
    for j, s in enumerate(symbols):
        m = markets[s].reindex(calendar)
        has_bar[:, j] = calendar.isin(markets[s].index)
        o[:, j], h[:, j], lo[:, j], c[:, j] = (m[col].to_numpy(dtype=float)
                                               for col in ("open", "high", "low", "close"))
        div[:, j] = m["dividend"].fillna(0.0).to_numpy() if "dividend" in m else 0.0
        if "atr" in (exits_of[s].stop_loss, exits_of[s].take_profit):
            own = markets[s]
            atr[:, j] = core.atr(own["high"], own["low"], own["close"],
                                 exits_of[s].atr_window).reindex(calendar).to_numpy()
        source = regime_of.get(s)
        if source is not None:
            labels[:, j] = source["regime"].reindex(calendar).fillna("unknown").to_numpy()
    W = weights.reindex(index=calendar, columns=list(symbols)).to_numpy()
    return [Bar(has_bar[i], o[i], h[i], lo[i], c[i], div[i], atr[i], labels[i], W[i])
            for i in range(n)]


def simulate_portfolio(
    markets: Mapping[str, pd.DataFrame],
    targets: Mapping[str, pd.Series],
    *,
    costs: CostModel,
    initial_cash: float,
    exits: ExitSpec | Mapping[str, ExitSpec] | None = None,
    rebalance_band: float = 0.2,
    calendar: pd.DatetimeIndex | None = None,
    max_gross: float = 1.0,
    max_weight: float = 1.0,
    regimes: pd.DataFrame | None = None,
    entry_regimes: Mapping[str, pd.DataFrame] | None = None,
    notes: list[str] | None = None,
    reinvest_dividends: bool = False,
    order_filter: OrderFilter | None = None,
) -> PortfolioResult:
    """Run instruments that share one cash account. ``markets``: quoted
    ``open high low close`` and ``dividend`` per symbol; ``targets``: weight of
    equity decided at each of that symbol's closes.

    ``reinvest_dividends``: buy more of the same instrument with its dividends
    at the next open (whole shares, so a small dividend waits until it buys
    one). For buy-and-hold benchmarks: without it their dividends would sit in
    cash for decades while a rebalancing strategy reinvests its own in passing,
    and the comparison would flatter the strategy.
    """
    symbols = tuple(sorted(markets))
    if not symbols:
        raise ValueError("no instruments to simulate")
    if set(targets) != set(symbols):
        raise ValueError("every instrument needs a target and no target may lack an instrument")
    if calendar is None:
        calendar = pd.DatetimeIndex(sorted(set().union(*(m.index for m in markets.values()))))
    if len(calendar) == 0:
        raise ValueError("no bars to simulate")
    exits_of = {s: (exits.get(s, ExitSpec()) if isinstance(exits, Mapping) else (exits or ExitSpec()))
                for s in symbols}
    raw = pd.DataFrame({s: _validated(s, markets[s], targets[s]).reindex(calendar) for s in symbols},
                       index=calendar)
    weights = portfolio_weights(raw, max_gross=max_gross, max_weight=max_weight)
    regime_of = {s: (entry_regimes or {}).get(s, regimes) for s in symbols}
    days = bars_for(markets, symbols, calendar, weights, exits_of, regime_of)

    book = Book(symbols, costs, dict(exits_of), float(initial_cash), rebalance_band,
                reinvest_dividends, order_filter, notes=list(notes or []))
    n, k = len(calendar), len(symbols)
    eq, cash_s = np.empty(n), np.empty(n)
    shares_s, value_s = np.zeros((n, k)), np.zeros((n, k))
    for i, day in enumerate(calendar):
        eq[i] = book.step(day, days[i])
        cash_s[i] = book.cash
        for j, s in enumerate(symbols):
            p = book.pos[s]
            shares_s[i, j] = p.shares
            value_s[i, j] = p.shares * p.last_close if p.shares > 0 else 0.0
    book.finish()
    for s in sorted(book.carried):
        book.notes.append(f"{s} had calendar days without a bar while held; valued at its last close")

    return PortfolioResult(
        fills=book.fills,
        symbols=symbols,
        equity=pd.Series(eq, index=calendar, name="equity"),
        cash=pd.Series(cash_s, index=calendar, name="cash"),
        positions=pd.DataFrame(shares_s, index=calendar, columns=list(symbols)),
        market_value=pd.DataFrame(value_s, index=calendar, columns=list(symbols)),
        weights=weights.fillna(0.0),
        trades=book.trades + book.open_trades(),
        rejections=book.rejections,
        dividends=book.dividends,
        notes=book.notes,
        regimes=regimes.reindex(calendar) if regimes is not None else None,
    )


def simulate(
    symbol: str,
    market: pd.DataFrame,
    target: pd.Series,
    *,
    costs: CostModel,
    initial_cash: float,
    exits: ExitSpec | None = None,
    rebalance_band: float = 0.2,
    regimes: pd.DataFrame | None = None,
    notes: list[str] | None = None,
    reinvest_dividends: bool = False,
    order_filter: OrderFilter | None = None,
) -> SimulationResult:
    """One instrument: the portfolio loop with a single member."""
    if len(market) == 0:
        raise ValueError("no bars to simulate")
    weights = _validated(symbol, market, target)
    result = simulate_portfolio(
        {symbol: market}, {symbol: target}, costs=costs, initial_cash=initial_cash, exits=exits,
        rebalance_band=rebalance_band, calendar=market.index, regimes=regimes, notes=notes,
        reinvest_dividends=reinvest_dividends, order_filter=order_filter,
    )
    return result.single(symbol, target=weights)
