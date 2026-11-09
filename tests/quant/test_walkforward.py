"""Walk-forward: folds that keep apart what they must, a choice the test data
cannot touch, a register that remembers what was looked at, and a verdict that
says when a strategy only worked in training."""

from __future__ import annotations

import math
import sqlite3
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from tradingagents_quant.config import QuantConfig
from tradingagents_quant.data.ingest import ingest
from tradingagents_quant.data.snapshot import SnapshotStore
from tradingagents_quant.sim import optimize, walkforward
from tradingagents_quant.sim.walkforward import (
    ExperimentError,
    WalkForwardDesign,
    decide,
    load_hypothesis,
    make_folds,
    run_walkforward,
)
from tradingagents_quant.store.db import Database

from .fixtures import FakeProvider, synthetic_bars, yahoo_frame

N_BARS = 2600
BARS = synthetic_bars(N_BARS, drift=0.0003, seed=17)
DAYS = [d.strftime("%Y-%m-%d") for d in BARS.index]
CONFIG = QuantConfig.model_validate({"universe": {"symbols": ["SPY"]}})
DESIGN = WalkForwardDesign(
    rule="sma_cross", symbols=("SPY",), grid={"fast": [10, 20, 40], "slow": [60, 100, 150]},
    train_years=3, validation_years=1, test_years=1, embargo_bars=10, top_k=2, min_oos_trades=0,
)


def _store(closes=None, dividends=None):
    db = Database(":memory:")
    db.migrate()
    store = SnapshotStore(db)
    frame = yahoo_frame(DAYS, list(BARS["close"] if closes is None else closes),
                        dividends=dividends or {DAYS[300]: 0.4})
    ingest(store, symbols=("SPY",), start=DAYS[0], end=DAYS[-1], fetch=FakeProvider({"SPY": frame}),
           source="fixture", source_version="fixture 1", config=CONFIG,
           now=datetime(2030, 1, 1, tzinfo=UTC))
    return store


# -- folds -------------------------------------------------------------------------


def test_rolling_folds_keep_segments_apart_and_tile_the_test_periods():
    calendar = BARS.index
    folds = make_folds(calendar, DESIGN)
    assert len(folds) >= 4
    for f in folds:
        assert len(f.train) == 756 and len(f.validation) == 252 and len(f.test) == 252
        # The embargo: exactly 10 bars between the segments, none shared.
        pos = {d: i for i, d in enumerate(calendar)}
        assert pos[f.validation[0]] - pos[f.train[-1]] == 11
        assert pos[f.test[0]] - pos[f.validation[-1]] == 11
        assert not (set(f.train) | set(f.validation)) & set(f.test)
    tests = [f.test for f in folds]
    for a, b in zip(tests, tests[1:], strict=False):
        assert a[-1] < b[0] and calendar.get_loc(b[0]) == calendar.get_loc(a[-1]) + 1


def test_anchored_folds_train_from_the_first_bar():
    folds = make_folds(BARS.index, DESIGN.model_copy(update={"anchored": True}))
    assert all(f.train[0] == BARS.index[0] for f in folds)
    assert [len(f.train) for f in folds] == sorted(len(f.train) for f in folds)


def test_too_little_history_for_one_fold_is_an_error():
    with pytest.raises(ExperimentError, match="cannot hold one fold"):
        make_folds(BARS.index[:900], DESIGN)


@pytest.mark.parametrize(
    "update",
    [{"grid": {"window": [3]}}, {"fixed_params": {"fast": 10}}, {"max_variants": 4},
     {"rule": "magic"}, {"symbols": ()}],
)
def test_inconsistent_designs_are_refused(update):
    with pytest.raises(ValueError):
        WalkForwardDesign.model_validate({**DESIGN.model_dump(), **update})


# -- optimizer ---------------------------------------------------------------------


def test_the_grid_is_complete_and_ordered():
    cands = optimize.grid({"slow": [100, 60], "fast": [20, 10]})
    assert [c.params for c in cands] == [
        {"fast": 10, "slow": 60}, {"fast": 10, "slow": 100},
        {"fast": 20, "slow": 60}, {"fast": 20, "slow": 100}]


def test_a_plateau_beats_a_spike():
    cands = optimize.grid({"x": [1, 2, 3, 4, 5, 6, 7]})
    # A lonely peak at x=2 (score 10, neighbours 0); a plateau of 6 around x=6.
    scores = {1: 0.0, 2: 10.0, 3: 0.0, 4: 1.0, 5: 6.0, 6: 6.0, 7: 6.0}
    ranked = optimize.rank(cands, lambda c: scores[c.params["x"]])
    assert ranked[0][0].params == {"x": 6}
    assert ranked[0][1] == 6.0 and ranked[0][2] == pytest.approx(6.0)
    spike = next(r for r in ranked if r[0].params == {"x": 2})
    assert spike[2] == pytest.approx(10 / 3)


def test_a_set_next_to_a_cliff_is_marked_down():
    cands = optimize.grid({"x": [1, 2, 3]})
    ranked = optimize.rank(cands, lambda c: {1: -math.inf, 2: 5.0, 3: 4.0}[c.params["x"]])
    assert ranked[0][0].params == {"x": 3}  # 2 sits beside an invalid set
    assert ranked[-1][2] == -math.inf


# -- the run -------------------------------------------------------------------------


def test_the_test_data_cannot_touch_the_choice():
    """Change everything after fold 0's validation: fold 0 must choose the same,
    with bit-identical training and validation scores."""
    folds = make_folds(BARS.index, DESIGN)
    cut = BARS.index.get_loc(folds[0].validation[-1])
    changed = BARS["close"].to_numpy().copy()
    changed[cut + 1:] *= np.exp(np.linspace(0, -0.8, len(changed) - cut - 1))  # a crash later on
    a = run_walkforward(_store(), DESIGN, CONFIG, save=False)
    b = run_walkforward(_store(changed), DESIGN, CONFIG, save=False)
    fa, fb = a.folds[0], b.folds[0]
    assert fa.chosen == fb.chosen
    assert fa.train_objective == fb.train_objective
    assert fa.validation_objective == fb.validation_objective
    assert fa.test != fb.test  # the test segment did see the change


def test_a_walk_forward_is_deterministic():
    a = run_walkforward(_store(), DESIGN, CONFIG, save=False)
    b = run_walkforward(_store(), DESIGN, CONFIG, save=False)
    assert [f.chosen for f in a.folds] == [f.chosen for f in b.folds]
    assert a.oos == b.oos and a.verdict == b.verdict


def test_an_overfitted_search_on_noise_is_not_validated():
    rng = np.random.RandomState(99)
    noise = 100 * np.exp(np.cumsum(0.012 * rng.standard_normal(N_BARS)))
    design = DESIGN.model_copy(update={
        "grid": {"fast": [5, 10, 15, 20, 30], "slow": [40, 60, 90, 120, 160]}, "min_oos_trades": 10})
    report = run_walkforward(_store(noise), design, CONFIG, save=False)
    assert report.verdict in ("failed_validation", "inconclusive")
    assert report.reasons


def test_the_register_stores_every_trial_and_refuses_silent_reuse():
    store = _store()
    first = run_walkforward(store, DESIGN, CONFIG)
    db = store.db
    folds = len(first.folds)
    count = db.query_one("SELECT COUNT(*) FROM experiment_trials WHERE experiment_id = ?",
                         (first.experiment_id,))[0]
    # Per fold: all 9 sets on train (every fast < every slow, so all valid),
    # top_k on validation, one test run, one buy-and-hold run.
    assert count == len(first.trials) == folds * (9 + DESIGN.top_k + 2)
    selected = db.query("SELECT segment FROM experiment_trials WHERE experiment_id = ? AND selected = 1",
                        (first.experiment_id,))
    assert sorted({r[0] for r in selected}) == ["test", "train", "validation"]
    periods = db.query_one("SELECT COUNT(*) FROM experiment_test_periods WHERE experiment_id = ?",
                           (first.experiment_id,))[0]
    assert periods == folds
    verdict = db.query_one("SELECT verdict, status FROM experiments WHERE experiment_id = ?",
                           (first.experiment_id,))
    assert tuple(verdict) == (first.verdict, "complete")

    with pytest.raises(ExperimentError, match="earlier experiment"):
        run_walkforward(store, DESIGN, CONFIG)
    second = run_walkforward(store, DESIGN, CONFIG, allow_reuse=True)
    assert second.reused_by == [first.experiment_id]
    assert second.deflated["trials"] == 2
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        db.execute("UPDATE experiments SET verdict = 'validated'")


def test_a_failed_run_is_recorded_as_failed(monkeypatch):
    store = _store()

    def broken(*args, **kwargs):
        raise RuntimeError("engine fell over")

    monkeypatch.setattr(walkforward, "evaluate", broken)
    with pytest.raises(RuntimeError):
        run_walkforward(store, DESIGN, CONFIG)
    row = store.db.query_one("SELECT status, error FROM experiments")
    assert row["status"] == "failed" and "engine fell over" in row["error"]


# -- verdict -------------------------------------------------------------------------


def test_the_verdict_names_every_reason():
    design = DESIGN.model_copy(update={"min_oos_trades": 10, "margin_vs_benchmark": 0.1})
    good = {"deflated_sharpe": 0.99}
    assert decide(design, {"sharpe": 0.9}, {"sharpe": 0.5}, good, 40, [1.0]) == ("validated", [])
    v, reasons = decide(design, {"sharpe": 0.55}, {"sharpe": 0.5}, {"deflated_sharpe": 0.7}, 40,
                        [2.0])
    assert v == "failed_validation" and len(reasons) == 3
    assert "mostly in training" in reasons[2]
    assert decide(design, {"sharpe": 2.0}, {"sharpe": 0.5}, good, 5, [])[0] == "inconclusive"
    assert decide(design, {"sharpe": None}, {"sharpe": 0.5}, good, 40, [])[0] == "inconclusive"


# -- hypotheses ------------------------------------------------------------------------


def test_a_hypothesis_file_is_loaded_with_its_hash(tmp_path):
    path = tmp_path / "H-T.toml"
    path.write_text(
        '[hypothesis]\nid = "H-T"\nstatement = "SMA timing beats holding"\ncreated = "2026-11-09"\n'
        '[design]\nrule = "sma_cross"\nsymbols = ["SPY"]\ngrid = {fast = [20, 50]}\n'
        'train_years = 3\nvalidation_years = 1\ntest_years = 1\n'
        '[criteria]\nmargin_vs_benchmark = 0.1\nmax_variants = 2\n', encoding="utf-8")
    h = load_hypothesis(path)
    assert h.id == "H-T" and len(h.sha256) == 64
    assert h.design.margin_vs_benchmark == 0.1 and h.design.grid == {"fast": [20, 50]}
    path.write_text('[hypothesis]\nid = "H-T"\n[design]\nrule = "sma_cross"\nsymbols = ["SPY"]\n',
                    encoding="utf-8")
    with pytest.raises(ExperimentError, match="statement"):
        load_hypothesis(path)


def test_every_committed_hypothesis_is_valid():
    from pathlib import Path

    folder = Path(__file__).resolve().parents[2] / "docs" / "hypothesen"
    files = sorted(folder.glob("H-*.toml"))
    assert files, "the pre-registered hypotheses live in docs/hypothesen/"
    for path in files:
        h = load_hypothesis(path)
        assert h.id == path.stem
        assert h.design.max_variants is not None, f"{h.id} must cap its variants in advance"
        assert len(h.design.candidates()) <= h.design.max_variants
    assert pd.Series([load_hypothesis(p).id for p in files]).is_unique
