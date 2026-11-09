"""Parameter search over a grid, chosen for robustness rather than the peak.

A parameter set is scored by the mean objective of itself and its grid
neighbours (one step away in one parameter). A set that is good only where it
stands — a spike, with poor neighbours — loses to one on a plateau. The brief
asks for exactly that: robust performance, not one narrow configuration.

The grid is enumerated in a fixed order, so a search is deterministic.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    position: tuple[int, ...]  # index into each parameter's value list
    params: dict[str, float | int]


def grid(values: Mapping[str, Sequence[float | int]]) -> list[Candidate]:
    """Every combination, in the order of the sorted parameter names."""
    names = sorted(values)
    lists = [sorted(set(values[n])) for n in names]
    if any(not v for v in lists):
        raise ValueError("every grid parameter needs at least one value")
    return [
        Candidate(pos, {n: lists[k][i] for k, (n, i) in enumerate(zip(names, pos, strict=True))})
        for pos in itertools.product(*(range(len(v)) for v in lists))
    ]


def neighbours(candidates: Sequence[Candidate]) -> dict[tuple[int, ...], list[tuple[int, ...]]]:
    present = {c.position for c in candidates}
    out = {}
    for c in candidates:
        near = []
        for axis in range(len(c.position)):
            for step in (-1, 1):
                other = list(c.position)
                other[axis] += step
                if tuple(other) in present:
                    near.append(tuple(other))
        out[c.position] = near
    return out


def smoothed(
    scores: Mapping[tuple[int, ...], float], near: Mapping[tuple[int, ...], list[tuple[int, ...]]]
) -> dict[tuple[int, ...], float]:
    """Mean score over each candidate and its neighbours. A candidate without a
    finite score stays -inf; a neighbour without one counts as the worst finite
    score present (a cliff next to a set is a warning about the set)."""
    finite = [s for s in scores.values() if math.isfinite(s)]
    floor = min(finite) if finite else -math.inf
    out = {}
    for pos, score in scores.items():
        if not math.isfinite(score):
            out[pos] = -math.inf
            continue
        values = [score] + [scores[n] if math.isfinite(scores[n]) else floor for n in near[pos]]
        out[pos] = sum(values) / len(values)
    return out


def rank(
    candidates: Sequence[Candidate], score: Callable[[Candidate], float]
) -> list[tuple[Candidate, float, float]]:
    """(candidate, raw score, smoothed score), best smoothed first; ties keep
    grid order."""
    raw = {c.position: score(c) for c in candidates}
    smooth = smoothed(raw, neighbours(candidates))
    order = sorted(range(len(candidates)), key=lambda i: (-smooth[candidates[i].position], i))
    return [(candidates[i], raw[candidates[i].position], smooth[candidates[i].position])
            for i in order]
