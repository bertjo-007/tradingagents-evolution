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

Nothing in the loop reads a bar after the current one. ``tests/quant/
test_engine_leakage.py`` checks that by truncation, by an oracle that is
allowed to cheat (it must look absurdly good) and by delaying that oracle a
day (its edge must vanish).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradingagents_quant.indicators import core
from tradingagents_quant.sim.execution import CostModel
from tradingagents_quant.sim.types import Fill, PortfolioResult, Rejection, SimulationResult, Trade
from tradingagents_quant.strategy.spec import ExitSpec


@dataclass(frozen=True)
class _Order:
    side: str
    shares: float
    reason: str
    decided: int  # calendar index of the decision
    atr: float  # ATR at the decision, for ATR-based exit levels


class _Position:
    __slots__ = ("bars_held", "dividend_cash", "last_close", "pending", "shares", "stop",
                 "stopped_out", "take", "trade")

    def __init__(self) -> None:
        self.shares = 0.0
        self.trade: Trade | None = None
        self.stop: float | None = None
        self.take: float | None = None
        self.stopped_out = False
        self.bars_held = 0
        self.last_close = math.nan
        self.pending: _Order | None = None
        self.dividend_cash = 0.0  # received, not yet reinvested (reinvest_dividends only)


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

    n, k = len(calendar), len(symbols)
    raw = pd.DataFrame({s: _validated(s, markets[s], targets[s]).reindex(calendar) for s in symbols},
                       index=calendar)
    weights = portfolio_weights(raw, max_gross=max_gross, max_weight=max_weight)
    W = weights.to_numpy()

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
        source = (entry_regimes or {}).get(s)
        if source is None and regimes is not None:
            source = regimes
        if source is not None:
            labels[:, j] = source["regime"].reindex(calendar).fillna("unknown").to_numpy()

    notes = list(notes or [])
    cash = float(initial_cash)
    pos = {s: _Position() for s in symbols}
    fills: list[Fill] = []
    trades: list[Trade] = []
    rejections: list[Rejection] = []
    dividends = 0.0
    eq = np.empty(n)
    cash_s = np.empty(n)
    shares_s = np.zeros((n, k))
    value_s = np.zeros((n, k))
    carried: set[str] = set()

    def _fill(i: int, s: str, side: str, qty: float, reference: float, reason: str) -> Fill:
        nonlocal cash
        price = costs.fill_price(side, reference)
        f = Fill(calendar[i], side, qty, reference, price, costs.commission(qty, price), reason, s)
        cash += f.cash_flow
        p = pos[s]
        p.shares += qty if side == "buy" else -qty
        if p.shares <= 1e-12:
            p.shares = 0.0
        fills.append(f)
        return f

    def _close_trade(i: int, s: str, f: Fill, reason: str) -> None:
        p = pos[s]
        assert p.trade is not None
        p.trade.exit_date, p.trade.exit_price, p.trade.exit_reason = calendar[i], f.price, reason
        p.trade.holding_bars = p.bars_held
        trades.append(p.trade)
        p.trade, p.stop, p.take = None, None, None

    def _buy(i: int, j: int, s: str, order: _Order, price: float, qty: float) -> None:
        if qty < order.shares:
            rejections.append(Rejection(
                calendar[i], "insufficient_cash" if qty == 0 else "reduced_for_cash",
                f"wanted {order.shares:g} shares at {price:.4f}, cash pays {qty:g}", s))
        if qty <= 0:
            return
        f = _fill(i, s, "buy", qty, o[i, j], order.reason)
        p = pos[s]
        if order.reason == "dividend_reinvest":
            p.dividend_cash = max(0.0, p.dividend_cash + f.cash_flow)
        if p.trade is None:
            p.trade = Trade(calendar[i], f.price, str(labels[order.decided, j]), symbol=s)
            p.stop, p.take = _levels(exits_of[s], f.price, order.atr)
            p.bars_held = 0
        p.trade.fills.append(f)

    for i in range(n):
        # 1. Dividends to whoever held at the previous close.
        for j, s in enumerate(symbols):
            p = pos[s]
            if has_bar[i, j] and p.shares > 0 and div[i, j] > 0:
                amount = p.shares * div[i, j]
                cash += amount
                dividends += amount
                if reinvest_dividends:
                    p.dividend_cash += amount
                if p.trade is not None:
                    p.trade.dividends += amount
        held_overnight = {s: pos[s].shares > 0 for s in symbols}

        # 2. Orders decided at the previous close fill at this open: sells first.
        buys: list[tuple[int, str, _Order, float]] = []
        for j, s in enumerate(symbols):
            p = pos[s]
            order, p.pending = p.pending, None
            if order is None:
                continue
            if not has_bar[i, j]:
                rejections.append(Rejection(calendar[i], "no_bar", f"{order.side} not filled", s))
            elif math.isnan(o[i, j]):
                rejections.append(Rejection(calendar[i], "no_open_price", f"{order.side} not filled", s))
            elif order.side == "sell":
                qty = min(order.shares, p.shares)
                if qty > 0:
                    f = _fill(i, s, "sell", qty, o[i, j], order.reason)
                    assert p.trade is not None
                    p.trade.fills.append(f)
                    if p.shares == 0:
                        _close_trade(i, s, f, order.reason)
            else:
                buys.append((j, s, order, costs.fill_price("buy", o[i, j])))
        if len(buys) == 1:
            j, s, order, price = buys[0]
            _buy(i, j, s, order, price, costs.affordable(cash, price, order.shares))
        elif buys:
            need = sum(q.shares * px + costs.commission(q.shares, px) for _, _, q, px in buys)
            factor = min(1.0, cash / need) if need > 0 else 0.0
            for j, s, order, price in buys:
                _buy(i, j, s, order, price, costs.affordable(cash, price, order.shares * factor))

        # 3. Stops and targets during the bar.
        for j, s in enumerate(symbols):
            p = pos[s]
            if not has_bar[i, j] or p.shares <= 0 or (p.stop is None and p.take is None):
                continue
            reference = reason = None
            if math.isnan(h[i, j]) or math.isnan(lo[i, j]):
                notes.append(f"{calendar[i].date()} {s}: no high/low, stop and target not checked")
            elif held_overnight[s] and p.stop is not None and o[i, j] <= p.stop:
                reference, reason = o[i, j], "stop_loss"  # gapped through the stop
            elif held_overnight[s] and p.take is not None and o[i, j] >= p.take:
                reference, reason = o[i, j], "take_profit"  # gapped through the target
            elif p.stop is not None and lo[i, j] <= p.stop:
                reference, reason = p.stop, "stop_loss"  # also when the target was hit
            elif p.take is not None and h[i, j] >= p.take:
                reference, reason = p.take, "take_profit"
            if reason is not None:
                f = _fill(i, s, "sell", p.shares, reference, reason)
                assert p.trade is not None
                p.trade.fills.append(f)
                _close_trade(i, s, f, reason)
                p.stopped_out = exits_of[s].reentry == "after_signal_reset"

        # 4. Mark at the close.
        for j, s in enumerate(symbols):
            p = pos[s]
            if has_bar[i, j]:
                p.last_close = c[i, j]
                if p.shares > 0:
                    p.bars_held += 1
            elif p.shares > 0:
                carried.add(s)
            shares_s[i, j] = p.shares
            value_s[i, j] = p.shares * p.last_close if p.shares > 0 else 0.0
        eq[i] = cash + value_s[i].sum()
        cash_s[i] = cash

        # 5. Decide the orders for the next open.
        for j, s in enumerate(symbols):
            if not has_bar[i, j]:
                continue
            p, ex = pos[s], exits_of[s]
            w = 0.0 if math.isnan(W[i, j]) else float(W[i, j])
            if p.stopped_out and p.shares == 0:
                if w <= 0:
                    p.stopped_out = False  # the signal has reset; the next entry may come
                w = 0.0
            if p.shares > 0 and ex.max_holding_bars and p.bars_held >= ex.max_holding_bars:
                p.pending = _Order("sell", p.shares, "time_stop", i, atr[i, j])
                p.stopped_out = ex.reentry == "after_signal_reset"
            elif w <= 0 and p.shares > 0:
                p.pending = _Order("sell", p.shares, "exit_signal", i, atr[i, j])
            elif w > 0:
                desired = costs.round_shares(w * eq[i] / (c[i, j] * (1.0 + costs.adverse)))
                if p.shares == 0 and desired > 0:
                    if "atr" in (ex.stop_loss, ex.take_profit) and math.isnan(atr[i, j]):
                        # An exit level that cannot be computed is no trade (fail safe).
                        rejections.append(Rejection(calendar[i], "stop_not_computable",
                                                    "ATR not available yet; entry refused", s))
                    else:
                        p.pending = _Order("buy", desired, "entry", i, atr[i, j])
                elif p.shares > 0 and abs(desired - p.shares) > rebalance_band * p.shares:
                    side = "buy" if desired > p.shares else "sell"
                    p.pending = _Order(side, abs(desired - p.shares), "rebalance", i, atr[i, j])
            if reinvest_dividends and p.pending is None and p.shares > 0 and p.dividend_cash > 0:
                qty = costs.round_shares(p.dividend_cash / (c[i, j] * (1.0 + costs.adverse)))
                if qty > 0:
                    p.pending = _Order("buy", qty, "dividend_reinvest", i, atr[i, j])

    for s in symbols:
        p = pos[s]
        if p.trade is not None:
            p.trade.holding_bars = p.bars_held
            p.trade.mark_value = p.shares * p.last_close
            trades.append(p.trade)
    for s in sorted(carried):
        notes.append(f"{s} had calendar days without a bar while held; valued at its last close")

    return PortfolioResult(
        fills=fills,
        symbols=symbols,
        equity=pd.Series(eq, index=calendar, name="equity"),
        cash=pd.Series(cash_s, index=calendar, name="cash"),
        positions=pd.DataFrame(shares_s, index=calendar, columns=list(symbols)),
        market_value=pd.DataFrame(value_s, index=calendar, columns=list(symbols)),
        weights=weights.fillna(0.0),
        trades=trades,
        rejections=rejections,
        dividends=dividends,
        notes=notes,
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
) -> SimulationResult:
    """One instrument: the portfolio loop with a single member."""
    if len(market) == 0:
        raise ValueError("no bars to simulate")
    weights = _validated(symbol, market, target)
    result = simulate_portfolio(
        {symbol: market}, {symbol: target}, costs=costs, initial_cash=initial_cash, exits=exits,
        rebalance_band=rebalance_band, calendar=market.index, regimes=regimes, notes=notes,
        reinvest_dividends=reinvest_dividends,
    )
    return result.single(symbol, target=weights)
