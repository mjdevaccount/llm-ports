"""Paired skill: how much a voice's probabilities beat a REFERENCE forecast on the same events.

`score.Score.brier_skill_score` measures skill over climatology, the constant forecast of the
sample's own hit rate. That is the wrong yardstick when a better reference exists for each event:
a symbol's base rate, a model's prior, yesterday's forecast. A voice that says "up 60%" in a tape
where the reference already says "up 60%" knows nothing the reference did not. Here the reference
is given per event, and skill is `1 - BS(fct) / BS(ref)` over the pairs both scored (> 0 beats the
reference, 0 ties it, < 0 loses to it).

The interval resamples whole CLUSTERS, never single pairs: events that share evidence (one day's
tape, overlapping horizons) are not independent, and an iid resample would be too narrow. The
percentile bootstrap recomputes both Brier sums in every resample, so the reference's own noise
is in the interval. Seeded, so a re-run over the same pairs is identical.

Invariants:
    - Pure, standard library only.
    - A pair is scored only when BOTH probabilities are valid; others are skipped.
    - Inputs of different lengths are not aligned, so nothing is scored.
    - A statistic the data cannot define is None, never a number.

Rationale: ADR-0013.
"""

from __future__ import annotations

import random
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from llm_ports.forecast._types import valid_p
from llm_ports.forecast.score import brier_score

#: Fewer clusters than this and the resampled distribution is too coarse to bound: no interval.
MIN_CLUSTERS = 5
#: Resamples behind each interval.
N_RESAMPLES = 2000


@dataclass(frozen=True)
class Skill:
    """How a voice scored against a reference forecast on the same events.

    Attributes:
        n: Pairs scored (both probabilities valid).
        clusters: Distinct clusters among them.
        brier_score: The voice's mean Brier score; None at n = 0.
        ref_brier_score: The reference's mean Brier score on the same pairs; None at n = 0.
        skill: `1 - brier_score / ref_brier_score`; None at n = 0 or when the reference is perfect.
        lo: Lower bound of the cluster-bootstrap interval on `skill`; None below MIN_CLUSTERS.
        hi: Upper bound, as `lo`.
    """

    n: int
    clusters: int
    brier_score: float | None
    ref_brier_score: float | None
    skill: float | None
    lo: float | None
    hi: float | None


def _ratio(f: float, r: float) -> float | None:
    return 1.0 - f / r if r > 0 else None


def _quantile(xs: list[float], q: float) -> float:
    i = q * (len(xs) - 1)
    lo = int(i)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def brier_skill(
    obs: Sequence[bool],
    fct: Sequence[float],
    ref: Sequence[float],
    clusters: Sequence[Hashable] | None = None,
    *,
    level: float = 0.95,
    n_resamples: int = N_RESAMPLES,
    seed: int = 0,
) -> Skill:
    """Score forecasts against a per-event reference forecast.

    Args:
        obs: Whether each event happened.
        fct: The voice's probabilities, aligned with `obs`.
        ref: The reference's probabilities for the same events.
        clusters: A cluster id per event (events sharing evidence share an id); None = each
            event is its own cluster.
        level: Two-sided confidence level of the interval, in (0, 1).
        n_resamples: Bootstrap resamples.
        seed: The generator seed.

    Returns:
        The `Skill`. Never raises on data; a sequence of a different length scores nothing.

    Raises:
        ValueError: `level` outside (0, 1) or `n_resamples` < 1.
    """
    if not 0.0 < level < 1.0 or n_resamples < 1:
        raise ValueError("brier_skill: level must be in (0, 1) and n_resamples >= 1")
    ids: Sequence[Hashable] = clusters if clusters is not None else range(len(obs))
    if not (len(obs) == len(fct) == len(ref) == len(ids)):
        return Skill(n=0, clusters=0, brier_score=None, ref_brier_score=None, skill=None, lo=None, hi=None)
    sums: dict[Hashable, list[float]] = {}
    for y, f, r, c in zip(obs, fct, ref, ids, strict=True):
        if not (valid_p(f) and valid_p(r)):
            continue
        acc = sums.setdefault(c, [0.0, 0.0, 0.0])
        acc[0] += brier_score(bool(y), float(f))
        acc[1] += brier_score(bool(y), float(r))
        acc[2] += 1.0
    n = int(sum(a[2] for a in sums.values()))
    if n == 0:
        return Skill(n=0, clusters=0, brier_score=None, ref_brier_score=None, skill=None, lo=None, hi=None)
    bf, br = sum(a[0] for a in sums.values()), sum(a[1] for a in sums.values())
    point = _ratio(bf, br)
    lo = hi = None
    cells = list(sums.values())
    if point is not None and len(cells) >= MIN_CLUSTERS:
        rng = random.Random(seed)
        draws: list[float] = []
        for _ in range(n_resamples):
            pick = [cells[rng.randrange(len(cells))] for _ in cells]
            s = _ratio(sum(a[0] for a in pick), sum(a[1] for a in pick))
            if s is not None:
                draws.append(s)
        if draws:
            draws.sort()
            tail = (1.0 - level) / 2.0
            lo, hi = _quantile(draws, tail), _quantile(draws, 1.0 - tail)
    return Skill(n=n, clusters=len(cells), brier_score=bf / n, ref_brier_score=br / n, skill=point, lo=lo, hi=hi)


__all__ = ["MIN_CLUSTERS", "N_RESAMPLES", "Skill", "brier_skill"]
