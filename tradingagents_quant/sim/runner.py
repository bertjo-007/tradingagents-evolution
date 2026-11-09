"""A backtest from a snapshot: data checks, signal, sizing, simulation,
benchmarks, metrics, regime split. The only entry point the CLI and later
stages use, so every backtest passes the same checks.

One instrument or several. With N instruments each gets at most 1/N of
equity (its signal and sizing scaled by 1/N), so one active signal never
concentrates the whole account; the rest stays in cash.

Benchmarks, as the brief asks: buy and hold of the same instruments (equal
weight, never rebalanced, dividends reinvested in the instrument that paid
them), the market index (likewise), a simple moving-average strategy
on the index, and cash (zero return). A strategy is not a success because it
made money; it has to justify itself against these.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from tradingagents_quant import regime
from tradingagents_quant.config import QuantConfig
from tradingagents_quant.data.snapshot import SnapshotStore
from tradingagents_quant.errors import DataError, SnapshotError
from tradingagents_quant.sim import metrics as m
from tradingagents_quant.sim.engine import simulate, simulate_portfolio
from tradingagents_quant.sim.execution import CostModel
from tradingagents_quant.sim.prices import market_frame, signal_frame
from tradingagents_quant.sim.sizing import target_weights
from tradingagents_quant.sim.types import PortfolioResult, SimulationResult
from tradingagents_quant.strategy.library import RULES
from tradingagents_quant.strategy.spec import StrategySpec

SURVIVORSHIP_NOTE = (
    "Survivorship bias: the instruments are ones that still trade today. A result on "
    "them says nothing about instruments that were delisted."
)
NO_BAND = 1e9  # buy and hold: never rebalance

Result = SimulationResult | PortfolioResult


@dataclass
class Benchmark:
    name: str  # buy_and_hold | index | sma_baseline | cash
    spec: StrategySpec
    symbols: tuple[str, ...]
    series_sha256: str
    result: Result
    metrics: dict[str, Any]
    comparison: dict[str, Any]


@dataclass
class BacktestRun:
    spec: StrategySpec
    symbol: str  # the instruments, comma-joined
    snapshot_id: str
    series_sha256: str  # of the one series, or a hash over all of them
    config: QuantConfig
    result: Result
    metrics: dict[str, Any]
    regimes: dict[str, dict[str, Any]]
    symbols: tuple[str, ...] = ()
    benchmarks: list[Benchmark] = field(default_factory=list)
    data_warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def _primary(self) -> Benchmark | None:
        return next((b for b in self.benchmarks if b.name == "buy_and_hold"), None)

    @property
    def benchmark(self) -> Result | None:
        b = self._primary()
        return b.result if b else None

    @property
    def benchmark_spec(self) -> StrategySpec | None:
        b = self._primary()
        return b.spec if b else None

    @property
    def benchmark_metrics(self) -> dict[str, Any] | None:
        b = self._primary()
        return b.metrics if b else None

    @property
    def comparison(self) -> dict[str, Any]:
        b = self._primary()
        return b.comparison if b else {}


def load_market(
    store: SnapshotStore, snapshot_id: str, symbol: str, *, end: str | None
) -> tuple[pd.DataFrame, list[str], list[str], str]:
    """Market frame up to ``end``, refusing a series with data errors.
    Returns (market, warnings, notes, series sha256)."""
    info = store.get(snapshot_id).series_info(symbol)
    issues = store.issues(snapshot_id, symbol)
    if info.quality == "error":
        rules = sorted({i.rule for i in issues if i.severity == "error"})
        raise DataError(
            f"{symbol} has data errors in snapshot {snapshot_id} ({', '.join(rules)}); "
            "a backtest on it is refused. See `taq report --details`."
        )
    warnings = [f"{symbol} {i.rule}: {i.detail}" + (f" ({i.date})" if i.date else "")
                for i in issues if i.severity == "warning"]
    bars = store.bars(snapshot_id, symbol, end=end)
    actions = store.actions(snapshot_id, symbol)
    if end:
        actions = actions[actions["date"] <= end]
    market, notes = market_frame(bars, actions)
    return market, warnings, notes, info.sha256


def _targets(spec: StrategySpec, market: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Target weights and regime labels over the full history (warm-up included)."""
    signals = signal_frame(market, spec.signal_prices)
    signal = RULES[spec.rule].signal(signals, spec.resolved_params())
    return target_weights(signal, market, spec.sizing), regime.classify(signals)


def simulate_spec(
    spec: StrategySpec, symbol: str, market: pd.DataFrame, config: QuantConfig,
    window: pd.Index, notes: list[str], *, rebalance_band: float | None = None,
    reinvest_dividends: bool = False,
) -> SimulationResult:
    """One instrument. Signals and regimes see the full history before the
    window (warm-up); the simulation runs only inside it."""
    target, labels = _targets(spec, market)
    return simulate(
        symbol, market.loc[window], target.loc[window],
        costs=CostModel.from_settings(config.costs), initial_cash=config.backtest.initial_cash,
        exits=spec.exits,
        rebalance_band=config.backtest.rebalance_band if rebalance_band is None else rebalance_band,
        regimes=labels.loc[window], notes=notes, reinvest_dividends=reinvest_dividends,
    )


def simulate_portfolio_spec(
    spec: StrategySpec, markets: dict[str, pd.DataFrame], config: QuantConfig,
    calendar: pd.DatetimeIndex, market_regime: pd.DataFrame | None, notes: list[str], *,
    rebalance_band: float | None = None, reinvest_dividends: bool = False,
) -> PortfolioResult:
    n = len(markets)
    targets, inside, entry_regimes = {}, {}, {}
    for symbol, market in markets.items():
        target, labels = _targets(spec, market)
        days = market.index[market.index.isin(calendar)]
        inside[symbol] = market.loc[days]
        targets[symbol] = target.loc[days] / n
        entry_regimes[symbol] = labels.loc[days]
    return simulate_portfolio(
        inside, targets, costs=CostModel.from_settings(config.costs),
        initial_cash=config.backtest.initial_cash, exits=spec.exits,
        rebalance_band=config.backtest.rebalance_band if rebalance_band is None else rebalance_band,
        calendar=calendar, regimes=market_regime, entry_regimes=entry_regimes, notes=notes,
        reinvest_dividends=reinvest_dividends,
    )


def _combined_sha(shas: dict[str, str]) -> str:
    if len(shas) == 1:
        return next(iter(shas.values()))
    digest = hashlib.sha256()
    for symbol in sorted(shas):
        digest.update(f"{symbol}:{shas[symbol]}\n".encode())
    return digest.hexdigest()


def run_backtest(
    store: SnapshotStore,
    spec: StrategySpec,
    config: QuantConfig,
    *,
    symbol: str | None = None,
    symbols: tuple[str, ...] | None = None,
    snapshot_id: str | None = None,
    start: str | None = None,
    end: str | None = None,
    benchmark: bool = True,
) -> BacktestRun:
    if not config.modes.backtesting_mode:
        raise DataError("backtesting_mode is off in the configuration")
    chosen = tuple(s.upper() for s in ((symbol,) if symbol else (symbols or spec.symbols)))
    outside = [s for s in chosen if s not in spec.symbols]
    if outside:
        raise ValueError(f"{', '.join(outside)} is not an instrument of strategy {spec.name}")
    snap = store.resolve(snapshot_id)

    markets, warnings, notes, shas = {}, [], [], {}
    for s in chosen:
        markets[s], w, n_, shas[s] = load_market(store, snap.snapshot_id, s, end=end)
        warnings += w
        notes += n_

    # The calendar: the trading days of the calendar symbol inside the window.
    cal_symbol = config.universe.calendar_symbol
    try:
        if cal_symbol in markets:
            cal_market = markets[cal_symbol]
        else:
            cal_market = load_market(store, snap.snapshot_id, cal_symbol, end=end)[0]
        days = cal_market.index
    except (SnapshotError, DataError):
        cal_market = None
        days = pd.DatetimeIndex(sorted(set().union(*(mk.index for mk in markets.values()))))
        notes.append(f"calendar symbol {cal_symbol} unavailable; calendar is the union of the "
                     "instruments' days")
    window = days[days >= pd.Timestamp(start)] if start else days
    if end:
        window = window[window <= pd.Timestamp(end)]
    if len(chosen) == 1:
        window = markets[chosen[0]].index.intersection(window) if cal_market is not None else window
    if len(window) < 2:
        raise DataError(f"window {start}..{end} holds fewer than two bars of {', '.join(chosen)}")
    for s, mk in markets.items():
        if mk.index[0] > window[0]:
            notes.append(f"{s} starts trading on {mk.index[0].date()}, after the window opens; "
                         "no position before that")

    kwargs = {"initial_cash": config.backtest.initial_cash,
              "risk_free_rate": config.backtest.risk_free_rate,
              "min_annualization_bars": config.data.min_annualization_bars}
    market_regime = (regime.classify(signal_frame(cal_market, spec.signal_prices)).loc[window]
                     if cal_market is not None else None)
    if len(chosen) == 1:
        result: Result = simulate_spec(spec, chosen[0], markets[chosen[0]], config, window, notes)
    else:
        result = simulate_portfolio_spec(spec, markets, config, window, market_regime, notes)

    run = BacktestRun(
        spec=spec, symbol=",".join(chosen), symbols=chosen, snapshot_id=snap.snapshot_id,
        series_sha256=_combined_sha(shas), config=config, result=result,
        metrics=m.compute(result, **kwargs),
        regimes=m.by_regime(result, config.backtest.initial_cash),
        data_warnings=warnings, notes=[*result.notes, SURVIVORSHIP_NOTE],
    )
    if benchmark:
        run.benchmarks = _benchmarks(store, snap.snapshot_id, spec, config, chosen, markets, shas,
                                     window, market_regime, run.metrics, kwargs, end)
    return run


def _benchmarks(store, snapshot_id, spec, config, chosen, markets, shas, window, market_regime,
                strategy_metrics, kwargs, end) -> list[Benchmark]:
    out: list[Benchmark] = []
    basis = spec.signal_prices

    def add(name: str, bspec: StrategySpec, symbols: tuple[str, ...], sha: str, result: Result):
        metrics = m.compute(result, **kwargs)
        out.append(Benchmark(name, bspec, symbols, sha, result, metrics,
                             m.compare(strategy_metrics, metrics)))

    bh = StrategySpec(name=f"buy_and_hold {','.join(chosen)}", rule="buy_and_hold",
                      symbols=chosen, signal_prices=basis, benchmark=chosen[0])
    if len(chosen) == 1:
        res: Result = simulate_spec(bh, chosen[0], markets[chosen[0]], config, window, [],
                                    rebalance_band=NO_BAND, reinvest_dividends=True)
    else:
        res = simulate_portfolio_spec(bh, markets, config, window, market_regime, [],
                                      rebalance_band=NO_BAND, reinvest_dividends=True)
    add("buy_and_hold", bh, chosen, _combined_sha(shas), res)

    index = spec.benchmark.upper()
    try:
        index_market, _, _, index_sha = (
            (markets[index], [], [], shas[index]) if index in markets
            else load_market(store, snapshot_id, index, end=end))
    except (SnapshotError, DataError):
        index_market = None
    if index_market is not None:
        idx_window = index_market.index.intersection(window)
        if chosen != (index,):
            ix = StrategySpec(name=f"index {index}", rule="buy_and_hold", symbols=(index,),
                              signal_prices=basis, benchmark=index)
            add("index", ix, (index,), index_sha,
                simulate_spec(ix, index, index_market, config, idx_window, [],
                              rebalance_band=NO_BAND, reinvest_dividends=True))
        sma = StrategySpec(name=f"sma_baseline {index}", rule="sma_cross", symbols=(index,),
                           signal_prices=basis, benchmark=index)
        add("sma_baseline", sma, (index,), index_sha,
            simulate_spec(sma, index, index_market, config, idx_window, []))
    cash = StrategySpec(name="cash", rule="cash", symbols=(chosen[0],), benchmark=index)
    add("cash", cash, (chosen[0],), shas[chosen[0]],
        simulate_spec(cash, chosen[0], markets[chosen[0]], config,
                      markets[chosen[0]].index.intersection(window), []))
    return out
