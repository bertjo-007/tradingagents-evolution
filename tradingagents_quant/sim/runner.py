"""A backtest from a snapshot: data checks, signal, sizing, simulation,
benchmark, metrics, regime split. The only entry point the CLI and later
stages use, so every backtest passes the same checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from tradingagents_quant import regime
from tradingagents_quant.config import QuantConfig
from tradingagents_quant.data.snapshot import SnapshotStore
from tradingagents_quant.errors import DataError
from tradingagents_quant.sim import metrics as m
from tradingagents_quant.sim.engine import simulate
from tradingagents_quant.sim.execution import CostModel
from tradingagents_quant.sim.prices import market_frame, signal_frame
from tradingagents_quant.sim.sizing import target_weights
from tradingagents_quant.sim.types import SimulationResult
from tradingagents_quant.strategy.library import RULES
from tradingagents_quant.strategy.spec import StrategySpec

SURVIVORSHIP_NOTE = (
    "Survivorship bias: the instrument is one that still trades today. A result on "
    "it says nothing about instruments that were delisted."
)


@dataclass
class BacktestRun:
    spec: StrategySpec
    symbol: str
    snapshot_id: str
    series_sha256: str
    config: QuantConfig
    result: SimulationResult
    metrics: dict[str, Any]
    regimes: dict[str, dict[str, Any]]
    benchmark_spec: StrategySpec | None = None
    benchmark: SimulationResult | None = None
    benchmark_metrics: dict[str, Any] | None = None
    comparison: dict[str, Any] = field(default_factory=dict)
    data_warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


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
    warnings = [f"{i.rule}: {i.detail}" + (f" ({i.date})" if i.date else "")
                for i in issues if i.severity == "warning"]
    bars = store.bars(snapshot_id, symbol, end=end)
    actions = store.actions(snapshot_id, symbol)
    if end:
        actions = actions[actions["date"] <= end]
    market, notes = market_frame(bars, actions)
    return market, warnings, notes, info.sha256


def simulate_spec(
    spec: StrategySpec, symbol: str, market: pd.DataFrame, config: QuantConfig,
    window: pd.Index, notes: list[str],
) -> SimulationResult:
    signals = signal_frame(market, spec.signal_prices)
    # Signals and regimes see the full history before the window (warm-up);
    # the simulation runs only inside it.
    signal = RULES[spec.rule].signal(signals, spec.resolved_params())
    target = target_weights(signal, market, spec.sizing)
    labels = regime.classify(signals)
    return simulate(
        symbol,
        market.loc[window],
        target.loc[window],
        costs=CostModel.from_settings(config.costs),
        initial_cash=config.backtest.initial_cash,
        exits=spec.exits,
        rebalance_band=config.backtest.rebalance_band,
        regimes=labels.loc[window],
        notes=notes,
    )


def run_backtest(
    store: SnapshotStore,
    spec: StrategySpec,
    config: QuantConfig,
    *,
    symbol: str | None = None,
    snapshot_id: str | None = None,
    start: str | None = None,
    end: str | None = None,
    benchmark: bool = True,
) -> BacktestRun:
    if not config.modes.backtesting_mode:
        raise DataError("backtesting_mode is off in the configuration")
    symbol = (symbol or spec.symbols[0]).upper()
    if symbol not in spec.symbols:
        raise ValueError(f"{symbol} is not an instrument of strategy {spec.name}")
    snap = store.resolve(snapshot_id)
    market, warnings, notes, sha = load_market(store, snap.snapshot_id, symbol, end=end)
    window = market.index[market.index >= pd.Timestamp(start)] if start else market.index
    if len(window) < 2:
        raise DataError(f"window {start}..{end} holds fewer than two bars of {symbol}")

    result = simulate_spec(spec, symbol, market, config, window, list(notes))
    kwargs = {"initial_cash": config.backtest.initial_cash,
              "risk_free_rate": config.backtest.risk_free_rate,
              "min_annualization_bars": config.data.min_annualization_bars}
    run = BacktestRun(
        spec=spec, symbol=symbol, snapshot_id=snap.snapshot_id, series_sha256=sha, config=config,
        result=result, metrics=m.compute(result, **kwargs),
        regimes=m.by_regime(result, config.backtest.initial_cash),
        data_warnings=warnings, notes=[*notes, SURVIVORSHIP_NOTE],
    )
    if benchmark:
        bh = StrategySpec(name=f"buy_and_hold {symbol}", rule="buy_and_hold", symbols=(symbol,),
                          signal_prices=spec.signal_prices, benchmark=symbol)
        run.benchmark_spec = bh
        run.benchmark = simulate_spec(bh, symbol, market, config, window, [])
        run.benchmark_metrics = m.compute(run.benchmark, **kwargs)
        run.comparison = m.compare(run.metrics, run.benchmark_metrics)
    return run
