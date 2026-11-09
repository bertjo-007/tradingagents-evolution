"""Walk-forward validation: does a strategy hold up on data it was not fitted to?

Each fold has three consecutive segments, separated by an embargo (by default
21 bars, so a position or an indicator's memory cannot carry the end of one
segment into the start of the next):

    train ──embargo── validation ──embargo── test

- **train**: every parameter set of the grid is simulated; they are ranked by
  their neighbourhood-smoothed objective (``optimize.rank``), so a plateau
  beats a spike.
- **validation**: the best ``top_k`` of them are simulated; the best here is
  chosen. The test segment plays no part in any choice.
- **test**: the chosen set is simulated once, and buy and hold beside it.

Folds roll forward by the length of a test segment, so the test segments tile
the history without overlap and together form one out-of-sample record.
Every segment starts flat with fresh (simulated) cash, so both the strategy
and buy and hold pay an entry at the start of each.

The verdict follows criteria fixed before the run (``WalkForwardDesign``,
ideally from a pre-registered hypothesis file): out-of-sample Sharpe above
buy and hold's by a margin, a deflated Sharpe above a threshold, enough
trades. A strategy that shines only in training *fails validation*, and the
report says so.

Every evaluated parameter set is stored (``experiment_trials``): the honest
number of trials is what the deflated Sharpe ratio needs. And every test
period is remembered: a later experiment on the same instruments and
overlapping test periods must say that it reuses them, and the reuse raises
the bar its deflated Sharpe has to clear.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tradingagents_quant.config import QuantConfig
from tradingagents_quant.data.snapshot import SnapshotStore
from tradingagents_quant.errors import QuantError
from tradingagents_quant.provenance import code_version
from tradingagents_quant.sim import metrics as m, optimize
from tradingagents_quant.sim.deflate import deflated_sharpe
from tradingagents_quant.sim.runner import Prepared, evaluate, prepare
from tradingagents_quant.store.db import utc_now
from tradingagents_quant.strategy.library import RULES
from tradingagents_quant.strategy.spec import ExitSpec, SizingSpec, StrategySpec

BARS_PER_YEAR = 252


class ExperimentError(QuantError):
    """An experiment cannot run as designed (e.g. it would reuse test data)."""


class WalkForwardDesign(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: str
    symbols: tuple[str, ...]
    grid: dict[str, list[float | int]] = Field(default_factory=dict)
    fixed_params: dict[str, float | int] = Field(default_factory=dict)
    signal_prices: Literal["total_return", "price"] = "total_return"
    sizing: SizingSpec = SizingSpec()
    exits: ExitSpec = ExitSpec()
    start: str | None = None
    end: str | None = None
    train_years: float = Field(default=6.0, gt=0)
    validation_years: float = Field(default=2.0, gt=0)
    test_years: float = Field(default=2.0, gt=0)
    embargo_bars: int = Field(default=21, ge=0, le=252)
    anchored: bool = False  # train from the first bar every time (expanding window)
    objective: Literal["sharpe", "sortino", "cagr", "calmar"] = "sharpe"
    top_k: int = Field(default=3, ge=1)
    # Success criteria. Fixed before the run; ideally pre-registered.
    min_oos_trades: int = Field(default=15, ge=0)
    margin_vs_benchmark: float = 0.0
    min_deflated_sharpe: float = Field(default=0.95, ge=0, le=1)
    max_variants: int | None = Field(default=None, ge=1)

    @field_validator("symbols", mode="before")
    @classmethod
    def _symbols(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.split(",")
        return tuple(str(s).strip().upper() for s in value if str(s).strip())

    @model_validator(mode="after")
    def _consistent(self) -> WalkForwardDesign:
        if self.rule not in RULES:
            raise ValueError(f"unknown rule {self.rule!r}")
        if not self.symbols:
            raise ValueError("a design needs at least one symbol")
        known = set(RULES[self.rule].defaults)
        unknown = (set(self.grid) | set(self.fixed_params)) - known
        if unknown:
            raise ValueError(f"rule {self.rule} has no parameter(s) {sorted(unknown)}")
        both = set(self.grid) & set(self.fixed_params)
        if both:
            raise ValueError(f"parameter(s) both searched and fixed: {sorted(both)}")
        size = len(self.candidates())
        if self.max_variants is not None and size > self.max_variants:
            raise ValueError(f"the grid has {size} variants, more than the {self.max_variants} "
                             "this design allows")
        return self

    def candidates(self) -> list[optimize.Candidate]:
        return optimize.grid(self.grid) if self.grid else [optimize.Candidate((), {})]

    def spec_for(self, params: dict[str, float | int]) -> StrategySpec:
        return StrategySpec(
            name=f"{self.rule} walk-forward", rule=self.rule,
            params={**self.fixed_params, **params}, symbols=self.symbols,
            signal_prices=self.signal_prices, sizing=self.sizing, exits=self.exits,
        )

    def bars(self, years: float) -> int:
        return max(2, round(years * BARS_PER_YEAR))

    def fingerprint(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Hypothesis:
    """A pre-registered experiment: what is claimed, how it is tested, what
    counts as success — written down and committed before the run."""

    id: str
    statement: str
    created: str
    sha256: str
    path: str
    design: WalkForwardDesign


def load_hypothesis(path: str | Path) -> Hypothesis:
    raw_bytes = Path(path).read_bytes()
    try:
        raw = tomllib.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ExperimentError(f"hypothesis {path} is not valid UTF-8 TOML: {exc}") from exc
    head = raw.get("hypothesis", {})
    for key in ("id", "statement", "created"):
        if not head.get(key):
            raise ExperimentError(f"hypothesis {path} lacks [hypothesis].{key}")
    design = WalkForwardDesign.model_validate({**raw.get("design", {}), **raw.get("criteria", {})})
    return Hypothesis(str(head["id"]), str(head["statement"]), str(head["created"]),
                      hashlib.sha256(raw_bytes).hexdigest(), str(path), design)


@dataclass(frozen=True)
class Fold:
    n: int
    train: pd.DatetimeIndex
    validation: pd.DatetimeIndex
    test: pd.DatetimeIndex

    def periods(self) -> dict[str, tuple[str, str]]:
        return {name: (seg[0].strftime("%Y-%m-%d"), seg[-1].strftime("%Y-%m-%d"))
                for name, seg in (("train", self.train), ("validation", self.validation),
                                  ("test", self.test))}


def make_folds(calendar: pd.DatetimeIndex, design: WalkForwardDesign) -> list[Fold]:
    """Folds over the calendar. Rolling by default: each fold's train window has
    the same length and moves on by one test length; anchored: it grows."""
    tr, va, te = (design.bars(design.train_years), design.bars(design.validation_years),
                  design.bars(design.test_years))
    gap = design.embargo_bars
    folds, start, n = [], 0, 0
    while True:
        train_start = 0 if design.anchored else start
        train_end = start + tr
        val_start = train_end + gap
        test_start = val_start + va + gap
        test_end = test_start + te
        if test_end > len(calendar):
            break
        folds.append(Fold(n, calendar[train_start:train_end], calendar[val_start:val_start + va],
                          calendar[test_start:test_end]))
        n += 1
        start += te
    if not folds:
        need = tr + va + te + 2 * gap
        raise ExperimentError(f"{len(calendar)} bars cannot hold one fold of {need} bars; "
                              "shorten the segments or widen the window")
    return folds


# -- the run ---------------------------------------------------------------------


@dataclass
class Trial:
    fold: int
    segment: str
    spec: StrategySpec
    params: dict[str, float | int]
    start: str
    end: str
    metrics: dict[str, Any]
    objective: float
    selected: bool = False


@dataclass
class FoldReport:
    n: int
    periods: dict[str, tuple[str, str]]
    chosen: dict[str, float | int]
    train_objective: float
    train_smoothed: float
    validation_objective: float
    test: dict[str, Any]
    benchmark: dict[str, Any]
    candidates: int
    invalid: int
    in_sample_deflated: float | None


@dataclass
class WalkForwardReport:
    design: WalkForwardDesign
    snapshot_id: str
    folds: list[FoldReport]
    oos: dict[str, Any]
    benchmark_oos: dict[str, Any]
    deflated: dict[str, Any]
    verdict: str
    reasons: list[str]
    oos_trades: int
    reused_by: list[str]
    trials: list[Trial] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    hypothesis: Hypothesis | None = None
    experiment_id: str | None = None

    def summary(self) -> dict[str, Any]:
        return _finite({
            "folds": [dict(f.__dict__) for f in self.folds], "oos": self.oos,
            "benchmark_oos": self.benchmark_oos, "deflated": self.deflated,
            "verdict": self.verdict, "reasons": self.reasons, "oos_trades": self.oos_trades,
            "reused_by": self.reused_by, "notes": self.notes,
            "trials": len(self.trials),
        })


def _finite(value: Any) -> Any:
    """JSON has no infinity or NaN; an objective that could not be computed is null."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(v) for v in value]
    return value


def objective_of(metrics: dict[str, Any], name: str) -> float:
    if name == "calmar":
        cagr, dd = metrics.get("cagr"), metrics.get("max_drawdown")
        value = cagr / -dd if cagr is not None and dd is not None and dd < 0 else None
    else:
        value = metrics.get(name)
    return float(value) if value is not None and math.isfinite(value) else -math.inf


def decide(design: WalkForwardDesign, oos: dict[str, Any], bench: dict[str, Any],
           deflated: dict[str, Any], trades: int, train_sharpes: list[float]) -> tuple[str, list[str]]:
    """The verdict against the design's criteria, with every reason it failed."""
    sharpe, bench_sharpe = oos.get("sharpe"), bench.get("sharpe")
    if trades < design.min_oos_trades:
        return "inconclusive", [f"{trades} out-of-sample trades, fewer than the "
                                f"{design.min_oos_trades} the design requires"]
    if sharpe is None or bench_sharpe is None:
        return "inconclusive", ["the out-of-sample record is too short for a Sharpe ratio"]
    reasons = []
    if sharpe <= bench_sharpe + design.margin_vs_benchmark:
        reasons.append(f"out-of-sample Sharpe {sharpe:.2f} does not exceed buy and hold's "
                       f"{bench_sharpe:.2f} by the required {design.margin_vs_benchmark:.2f}")
    # The claim is "better than buy and hold", so the test is on the difference:
    # the daily active return (strategy minus buy and hold), deflated for the
    # number of looks at these test periods. A Sharpe above zero proves nothing
    # here — buy and hold itself has one.
    dsr = deflated.get("deflated_sharpe")
    if dsr is None or not math.isfinite(dsr):
        reasons.append("no measurable difference to buy and hold (active returns are flat)")
    elif dsr < design.min_deflated_sharpe:
        reasons.append(f"edge over buy and hold not significant: deflated information ratio "
                       f"{dsr:.3f}, below {design.min_deflated_sharpe:.2f}")
    finite = [s for s in train_sharpes if math.isfinite(s)]
    if finite:
        in_sample = sum(finite) / len(finite)
        if in_sample > 0 and sharpe < in_sample / 2:
            reasons.append(f"performance mostly in training: Sharpe {in_sample:.2f} in sample, "
                           f"{sharpe:.2f} out of sample")
    return ("validated" if not reasons else "failed_validation"), reasons


def _reused_by(store: SnapshotStore, symbols: str, folds: list[Fold]) -> list[str]:
    found: set[str] = set()
    for fold in folds:
        start, end = fold.periods()["test"]
        rows = store.db.query(
            "SELECT DISTINCT p.experiment_id FROM experiment_test_periods p "
            "JOIN experiments e ON e.experiment_id = p.experiment_id "
            "WHERE e.status = 'complete' AND p.symbols = ? AND p.start_date <= ? AND p.end_date >= ?",
            (symbols, end, start),
        )
        found |= {r[0] for r in rows}
    return sorted(found)


def _per_period_sharpe(metrics: dict[str, Any]) -> float:
    value = metrics.get("sharpe")
    return value / math.sqrt(BARS_PER_YEAR) if value is not None else math.nan


def run_walkforward(
    store: SnapshotStore,
    design: WalkForwardDesign,
    config: QuantConfig,
    *,
    snapshot_id: str | None = None,
    allow_reuse: bool = False,
    hypothesis: Hypothesis | None = None,
    save: bool = True,
) -> WalkForwardReport:
    if not config.modes.backtesting_mode:
        raise ExperimentError("backtesting_mode is off in the configuration")
    prep = prepare(store, snapshot_id, design.symbols, config, end=design.end)
    window = prep.window(design.start, design.end)
    folds = make_folds(window, design)
    symbols = ",".join(design.symbols)
    reused = _reused_by(store, symbols, folds)
    if reused and not allow_reuse:
        raise ExperimentError(
            f"the test periods overlap those of {len(reused)} earlier experiment(s) on {symbols} "
            f"({', '.join(reused[:5])}). Looking at the same out-of-sample data again makes it "
            "in-sample. Pass allow_reuse (--allow-reuse) to proceed knowingly; the reuse is "
            "recorded and raises the bar for the deflated Sharpe."
        )
    try:
        report = _run(prep, design, config, folds, reused)
    except Exception as exc:
        if save:
            _save_failure(store, prep, design, config, hypothesis, exc)
        raise
    report.hypothesis = hypothesis
    report.notes = [*prep.notes, *report.notes]
    if save:
        report.experiment_id = _save(store, prep, design, config, report)
    return report


def _run(prep: Prepared, design: WalkForwardDesign, config: QuantConfig, folds: list[Fold],
         reused: list[str]) -> WalkForwardReport:
    kwargs = {"initial_cash": config.backtest.initial_cash,
              "risk_free_rate": config.backtest.risk_free_rate,
              "min_annualization_bars": config.data.min_annualization_bars}
    candidates = design.candidates()
    trials: list[Trial] = []
    reports: list[FoldReport] = []
    oos_returns, bench_returns = [], []
    oos_trades = 0
    train_sharpes: list[float] = []
    bh_spec = StrategySpec(name="buy_and_hold walk-forward", rule="buy_and_hold",
                           symbols=design.symbols, signal_prices=design.signal_prices)

    def run_segment(spec, seg, fold_n, segment, params, buy_and_hold=False):
        result = evaluate(spec, prep, config, seg, buy_and_hold=buy_and_hold)
        metrics = m.compute(result, **kwargs)
        t = Trial(fold_n, segment, spec, params, metrics["start"], metrics["end"], metrics,
                  objective_of(metrics, design.objective))
        trials.append(t)
        return t, result

    for fold in folds:
        valid, invalid, results = [], 0, {}
        for cand in candidates:
            try:
                spec = design.spec_for(cand.params)
            except ValueError:
                invalid += 1  # e.g. fast >= slow: not a strategy, not a trial
                continue
            trial, result = run_segment(spec, fold.train, fold.n, "train", cand.params)
            valid.append(cand)
            results[cand.position] = (trial, result)
        if not valid:
            raise ExperimentError(f"fold {fold.n}: no valid parameter set in the grid")
        ranking = optimize.rank(valid, lambda c, r=results: r[c.position][0].objective)
        finalists = ranking[: design.top_k]
        best = None
        for cand, raw, smooth in finalists:
            trial, _ = run_segment(design.spec_for(cand.params), fold.validation, fold.n,
                                   "validation", cand.params)
            if best is None or trial.objective > best[3].objective:
                best = (cand, raw, smooth, trial)
        cand, raw, smooth, val_trial = best
        val_trial.selected = True
        results[cand.position][0].selected = True

        test_trial, test_result = run_segment(design.spec_for(cand.params), fold.test, fold.n,
                                              "test", cand.params)
        test_trial.selected = True
        bench_trial, bench_result = run_segment(bh_spec, fold.test, fold.n, "benchmark", {},
                                                buy_and_hold=True)
        oos_returns.append(m.equity_returns(test_result.equity, config.backtest.initial_cash))
        bench_returns.append(m.equity_returns(bench_result.equity, config.backtest.initial_cash))
        oos_trades += test_trial.metrics["trades"]

        # How much of the training Sharpe is the luck of choosing among the grid?
        train_trial, train_result = results[cand.position]
        train_sharpes.append(train_trial.metrics.get("sharpe") or math.nan)
        trial_sr = [_per_period_sharpe(results[c.position][0].metrics) for c in valid]
        trial_sr = [s for s in trial_sr if math.isfinite(s)]
        try:
            in_sample = deflated_sharpe(
                m.equity_returns(train_result.equity, config.backtest.initial_cash),
                n_trials=len(valid), trial_sharpes=trial_sr or None)["deflated_sharpe"]
        except ValueError:
            in_sample = None
        reports.append(FoldReport(
            fold.n, fold.periods(), dict(cand.params), raw, smooth, val_trial.objective,
            test_trial.metrics, bench_trial.metrics, len(valid), invalid, in_sample))

    oos = pd.concat(oos_returns)
    bench = pd.concat(bench_returns)
    oos_metrics = m.from_returns(oos, initial_cash=config.backtest.initial_cash,
                                 risk_free_rate=config.backtest.risk_free_rate,
                                 min_annualization_bars=config.data.min_annualization_bars)
    bench_metrics = m.from_returns(bench, initial_cash=config.backtest.initial_cash,
                                   risk_free_rate=config.backtest.risk_free_rate,
                                   min_annualization_bars=config.data.min_annualization_bars)
    # Each earlier look at these test periods was one more selection on them.
    n_trials = 1 + len(reused)
    # The edge: the daily difference to buy and hold on the same days. Its
    # deflated Sharpe (a deflated information ratio) is what the verdict uses.
    active = (oos - bench.reindex(oos.index)).dropna()
    try:
        deflated = deflated_sharpe(active.to_numpy(), n_trials=n_trials)
    except ValueError as exc:
        deflated = {"deflated_sharpe": None, "error": str(exc), "trials": n_trials}
    deflated["test"] = "active return vs buy and hold"
    try:
        absolute = deflated_sharpe(oos.to_numpy(), n_trials=n_trials)
        deflated["psr_vs_zero"] = absolute["psr_vs_zero"]
        deflated["strategy_sharpe_annualized"] = absolute["sharpe_annualized"]
    except ValueError:
        deflated["psr_vs_zero"] = None
    verdict, reasons = decide(design, oos_metrics, bench_metrics, deflated, oos_trades,
                              train_sharpes)
    notes = []
    if oos_trades < 100:
        notes.append(f"{oos_trades} out-of-sample trades: a strategy-level conclusion from fewer "
                     "than 100 trades has little statistical power")
    if reused:
        notes.append(f"test periods reused from {len(reused)} earlier experiment(s); the deflated "
                     f"Sharpe counts {n_trials} trials")
    return WalkForwardReport(design, prep.snapshot_id, reports, oos_metrics, bench_metrics,
                             deflated, verdict, reasons, oos_trades, reused, trials, notes)


def _experiment_id(now: str, design: WalkForwardDesign) -> str:
    stamp = now.replace("-", "").replace(":", "").split(".")[0]
    return f"wf-{stamp}-{design.fingerprint()[:8]}"


def _design_json(design: WalkForwardDesign, hypothesis: Hypothesis | None) -> str:
    payload = {"design": design.model_dump(mode="json")}
    if hypothesis is not None:
        payload["hypothesis"] = {"id": hypothesis.id, "statement": hypothesis.statement,
                                 "created": hypothesis.created, "path": hypothesis.path}
    return json.dumps(payload, sort_keys=True)


def _save(store: SnapshotStore, prep: Prepared, design: WalkForwardDesign, config: QuantConfig,
          report: WalkForwardReport) -> str:
    now = utc_now()
    experiment_id = _experiment_id(now, design)
    hyp = report.hypothesis
    with store.db.transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO config_versions (fingerprint, config_json, first_seen) "
            "VALUES (?, ?, ?)",
            (config.fingerprint(), json.dumps(config.to_dict(), sort_keys=True), now))
        conn.execute(
            "INSERT INTO experiments (experiment_id, created_at, kind, status, hypothesis_id, "
            "hypothesis_sha256, rule, symbols, snapshot_id, config_fingerprint, code_version, "
            "design_json, summary_json, verdict) VALUES (?, ?, 'walkforward', 'complete', ?, ?, ?, "
            "?, ?, ?, ?, ?, ?, ?)",
            (experiment_id, now, hyp.id if hyp else None, hyp.sha256 if hyp else None, design.rule,
             ",".join(design.symbols), prep.snapshot_id, config.fingerprint(), code_version(),
             _design_json(design, hyp), json.dumps(report.summary(), sort_keys=True, default=str,
                                                   allow_nan=False),
             report.verdict))
        for n, t in enumerate(report.trials):
            version = t.spec.version_hash()
            conn.execute(
                "INSERT OR IGNORE INTO strategy_versions (version_hash, name, rule, spec_json, "
                "first_seen) VALUES (?, ?, ?, ?, ?)",
                (version, t.spec.name, t.spec.rule, json.dumps(t.spec.to_dict(), sort_keys=True),
                 now))
            conn.execute(
                "INSERT INTO experiment_trials (experiment_id, trial_n, fold, segment, "
                "strategy_version, params_json, start_date, end_date, objective, selected, "
                "metrics_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (experiment_id, n, t.fold, t.segment, version, json.dumps(t.params, sort_keys=True),
                 t.start, t.end, t.objective if math.isfinite(t.objective) else None,
                 int(t.selected), json.dumps(_finite(t.metrics), sort_keys=True)))
        for f in report.folds:
            start, end = f.periods["test"]
            conn.execute(
                "INSERT INTO experiment_test_periods (experiment_id, fold, symbols, start_date, "
                "end_date) VALUES (?, ?, ?, ?, ?)",
                (experiment_id, f.n, ",".join(design.symbols), start, end))
    return experiment_id


def _save_failure(store, prep, design, config, hypothesis, exc) -> None:
    now = utc_now()
    with store.db.transaction() as conn:
        conn.execute(
            "INSERT INTO experiments (experiment_id, created_at, kind, status, hypothesis_id, "
            "hypothesis_sha256, rule, symbols, snapshot_id, config_fingerprint, code_version, "
            "design_json, error) VALUES (?, ?, 'walkforward', 'failed', ?, ?, ?, ?, ?, NULL, ?, "
            "?, ?)",
            (_experiment_id(now, design), now, hypothesis.id if hypothesis else None,
             hypothesis.sha256 if hypothesis else None, design.rule, ",".join(design.symbols),
             prep.snapshot_id, code_version(), _design_json(design, hypothesis),
             f"{type(exc).__name__}: {exc}"))
