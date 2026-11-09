"""The single-instrument simulation loop.

Timing, which is what keeps it free of look-ahead:

1. **Morning of bar t** — a holder at the close of t-1 receives that day's
   dividend (ex-date entitlement). Then the order decided at the close of t-1
   fills at the **open of t**, plus half spread and slippage.
2. **During bar t** — stop and target levels are checked against the bar's low
   and high. A position held overnight that opens through its stop fills at
   the (worse) open. If one bar reaches both stop and target, the stop is
   assumed: daily bars do not tell which came first, and the simulator does not
   pick the flattering order.
3. **Close of bar t** — equity is marked at the close. The target weight
   decided from bars up to t becomes the order for the open of t+1. Its size
   uses the close of t, the last price known when deciding; if the next open
   gaps up, the order shrinks to what the cash pays for and the cut is
   recorded.

Nothing in the loop reads a bar after the current one. ``tests/quant/
test_engine_leakage.py`` checks that by truncation, by an oracle that is
allowed to cheat (it must look absurdly good) and by delaying that oracle a
day (its edge must vanish).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradingagents_quant.indicators import core
from tradingagents_quant.sim.execution import CostModel
from tradingagents_quant.sim.types import Fill, Rejection, SimulationResult, Trade
from tradingagents_quant.strategy.spec import ExitSpec


@dataclass(frozen=True)
class _Order:
    side: str
    shares: float
    reason: str
    decided: int  # bar index of the decision
    atr: float  # ATR at the decision, for ATR-based exit levels


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
) -> SimulationResult:
    """Run one instrument. ``market``: quoted ``open high low close`` and
    ``dividend``; ``target``: weight of equity decided at each bar's close."""
    exits = exits or ExitSpec()
    if not target.index.equals(market.index):
        raise ValueError("target and market must share their index")
    if len(market) == 0:
        raise ValueError("no bars to simulate")
    weights = target.astype(float).fillna(0.0)
    if (weights < 0).any() or (weights > 1).any():
        raise ValueError("long-only, unlevered: target weights must lie in [0, 1]")

    dates = market.index
    o, h, lo, c = (market[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    div = market["dividend"].to_numpy(dtype=float) if "dividend" in market else np.zeros(len(c))
    needs_atr = "atr" in (exits.stop_loss, exits.take_profit)
    atr = (core.atr(market["high"], market["low"], market["close"], exits.atr_window).to_numpy()
           if needs_atr else np.full(len(c), math.nan))
    trend = regimes["regime"].reindex(dates).fillna("unknown").to_numpy() if regimes is not None \
        else np.full(len(c), "unknown", dtype=object)

    notes = list(notes or [])
    cash, shares = float(initial_cash), 0.0
    pending: _Order | None = None
    trade: Trade | None = None
    stop = take = None
    stopped_out = False
    bars_held = 0
    fills: list[Fill] = []
    trades: list[Trade] = []
    rejections: list[Rejection] = []
    dividends = 0.0
    eq = np.empty(len(c))
    cash_s = np.empty(len(c))
    shares_s = np.empty(len(c))

    def _fill(i: int, side: str, qty: float, reference: float, reason: str) -> Fill:
        nonlocal cash, shares
        price = costs.fill_price(side, reference)
        fee = costs.commission(qty, price)
        f = Fill(dates[i], side, qty, reference, price, fee, reason)
        cash += f.cash_flow
        shares += qty if side == "buy" else -qty
        fills.append(f)
        return f

    def _close_trade(i: int, f: Fill, reason: str) -> None:
        nonlocal trade, stop, take
        assert trade is not None
        trade.exit_date, trade.exit_price, trade.exit_reason = dates[i], f.price, reason
        trade.holding_bars = bars_held
        trades.append(trade)
        trade, stop, take = None, None, None

    for i in range(len(c)):
        # 1. Dividend to whoever held at the previous close.
        if shares > 0 and div[i] > 0:
            amount = shares * div[i]
            cash += amount
            dividends += amount
            if trade is not None:
                trade.dividends += amount
        held_overnight = shares > 0

        # 2. The order decided at the previous close fills at this open.
        if pending is not None:
            order, pending = pending, None
            if math.isnan(o[i]):
                rejections.append(Rejection(dates[i], "no_open_price", f"{order.side} not filled"))
            elif order.side == "buy":
                price = costs.fill_price("buy", o[i])
                qty = costs.affordable(cash, price, order.shares)
                if qty < order.shares:
                    rejections.append(Rejection(
                        dates[i], "insufficient_cash" if qty == 0 else "reduced_for_cash",
                        f"wanted {order.shares:g} shares at {price:.4f}, cash pays {qty:g}"))
                if qty > 0:
                    f = _fill(i, "buy", qty, o[i], order.reason)
                    if trade is None:
                        trade = Trade(dates[i], f.price, str(trend[order.decided]))
                        stop, take = _levels(exits, f.price, order.atr)
                        bars_held = 0
                    trade.fills.append(f)
            else:
                qty = min(order.shares, shares)
                if qty > 0:
                    f = _fill(i, "sell", qty, o[i], order.reason)
                    assert trade is not None
                    trade.fills.append(f)
                    if shares <= 1e-12:
                        shares = 0.0
                        _close_trade(i, f, order.reason)

        # 3. Stops and targets during the bar.
        if shares > 0 and (stop is not None or take is not None):
            reference = reason = None
            if math.isnan(h[i]) or math.isnan(lo[i]):
                notes.append(f"{dates[i].date()}: no high/low, stop and target not checked")
            elif held_overnight and stop is not None and o[i] <= stop:
                reference, reason = o[i], "stop_loss"  # gapped through the stop
            elif held_overnight and take is not None and o[i] >= take:
                reference, reason = o[i], "take_profit"  # gapped through the target
            elif stop is not None and lo[i] <= stop:
                reference, reason = stop, "stop_loss"  # also when the target was hit
            elif take is not None and h[i] >= take:
                reference, reason = take, "take_profit"
            if reason is not None:
                f = _fill(i, "sell", shares, reference, reason)
                assert trade is not None
                trade.fills.append(f)
                shares = 0.0
                _close_trade(i, f, reason)
                stopped_out = exits.reentry == "after_signal_reset"

        # 4. Mark at the close.
        if shares > 0:
            bars_held += 1
        eq[i] = cash + shares * c[i]
        cash_s[i], shares_s[i] = cash, shares

        # 5. Decide the order for the next open.
        w = float(weights.iloc[i])
        if stopped_out and shares == 0:
            if w <= 0:
                stopped_out = False  # the signal has reset; the next entry may come
            w = 0.0
        if shares > 0 and exits.max_holding_bars and bars_held >= exits.max_holding_bars:
            pending = _Order("sell", shares, "time_stop", i, atr[i])
            stopped_out = exits.reentry == "after_signal_reset"
        elif w <= 0 and shares > 0:
            pending = _Order("sell", shares, "exit_signal", i, atr[i])
        elif w > 0:
            desired = costs.round_shares(w * eq[i] / (c[i] * (1.0 + costs.adverse)))
            if shares == 0 and desired > 0:
                if needs_atr and math.isnan(atr[i]):
                    # An exit level that cannot be computed is no trade (fail safe).
                    rejections.append(Rejection(dates[i], "stop_not_computable",
                                                "ATR not available yet; entry refused"))
                else:
                    pending = _Order("buy", desired, "entry", i, atr[i])
            elif shares > 0 and abs(desired - shares) > rebalance_band * shares:
                side = "buy" if desired > shares else "sell"
                pending = _Order(side, abs(desired - shares), "rebalance", i, atr[i])

    if trade is not None:
        trade.holding_bars = bars_held
        trade.mark_value = shares * c[-1]
        trades.append(trade)

    return SimulationResult(
        symbol=symbol,
        equity=pd.Series(eq, index=dates, name="equity"),
        cash=pd.Series(cash_s, index=dates, name="cash"),
        shares=pd.Series(shares_s, index=dates, name="shares"),
        target=weights.rename("target"),
        fills=fills,
        trades=trades,
        rejections=rejections,
        dividends=dividends,
        notes=notes,
        regimes=regimes.reindex(dates) if regimes is not None else None,
    )
