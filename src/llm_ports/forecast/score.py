"""Proper scores for binary forecasts: Brier, log score, skill against the base rate, reliability.

A proper scoring rule is minimised in expectation only by reporting the true probability, so a
voice cannot improve its score by hedging toward 0.5 or by overclaiming. Names and argument order
follow scoringrules: observation first, forecast second (`brier_score(obs, fct)`), and lower is
better.

Invariants:
    - Pure, standard library only.
    - A pair whose forecast is not a finite probability is skipped, never scored.
    - Observations and forecasts of different lengths are not aligned, so nothing is scored.
    - A statistic the data cannot define is None, never a number.

Rationale: ADR-0013.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from llm_ports.forecast._types import valid_p

#: Probabilities are clipped to `[EPS, 1 - EPS]` before a logarithm, so a confident miss costs a
#: large finite score rather than infinity.
EPS = 1e-6


def brier_score(obs: bool, fct: float) -> float:
    """Return the Brier score of one binary forecast, `(fct - obs) ** 2`; 0 is perfect.

    Args:
        obs: Whether the event happened.
        fct: The forecast probability.
    """
    return (fct - float(obs)) ** 2


def log_score(obs: bool, fct: float) -> float:
    """Return the log score of one binary forecast, `-ln(fct)` if it happened else `-ln(1 - fct)`.

    `fct` is clipped to `[EPS, 1 - EPS]`, so the score is finite.

    Args:
        obs: Whether the event happened.
        fct: The forecast probability.
    """
    q = min(max(fct, EPS), 1.0 - EPS)
    return -math.log(q if obs else 1.0 - q)


def _clean(obs: Sequence[bool], fct: Sequence[float]) -> list[tuple[bool, float]]:
    if len(obs) != len(fct):
        return []
    return [(bool(y), float(p)) for y, p in zip(obs, fct, strict=True) if valid_p(p)]


@dataclass(frozen=True)
class Score:
    """How a set of binary forecasts scored.

    Attributes:
        n: Pairs scored.
        mean_p: The mean forecast probability; None at n = 0.
        hit_rate: The share that happened (the base rate); None at n = 0.
        brier_score: The mean Brier score; None at n = 0.
        log_score: The mean log score; None at n = 0.
        brier_skill_score: `1 - brier_score / (hit_rate * (1 - hit_rate))`, skill over always
            forecasting the base rate (> 0 beats it); None when undefined (n = 0, or every
            outcome the same).
    """

    n: int
    mean_p: float | None
    hit_rate: float | None
    brier_score: float | None
    log_score: float | None
    brier_skill_score: float | None

    @property
    def overconfidence(self) -> float | None:
        """`mean_p - hit_rate`: positive when the voice claims more than it delivers; None at n = 0."""
        if self.mean_p is None or self.hit_rate is None:
            return None
        return self.mean_p - self.hit_rate


def score(obs: Sequence[bool], fct: Sequence[float]) -> Score:
    """Score binary forecasts.

    Args:
        obs: Whether each event happened.
        fct: The forecast probabilities, aligned with `obs`; a pair with an invalid forecast is
            skipped, and sequences of different lengths score nothing (n = 0).

    Returns:
        The `Score`. Never raises.
    """
    got = _clean(obs, fct)
    n = len(got)
    if n == 0:
        return Score(n=0, mean_p=None, hit_rate=None, brier_score=None, log_score=None, brier_skill_score=None)
    mean_p = sum(p for _, p in got) / n
    hit = sum(1 for y, _ in got if y) / n
    b = sum(brier_score(y, p) for y, p in got) / n
    ll = sum(log_score(y, p) for y, p in got) / n
    ref = hit * (1.0 - hit)
    return Score(
        n=n,
        mean_p=mean_p,
        hit_rate=hit,
        brier_score=b,
        log_score=ll,
        brier_skill_score=1.0 - b / ref if ref > 0 else None,
    )


@dataclass(frozen=True)
class Bin:
    """One bin of a reliability curve.

    Attributes:
        lo: The bin's lower edge (inclusive).
        hi: The bin's upper edge (exclusive, except the last bin, which includes 1.0).
        n: Forecasts in the bin.
        mean_p: Their mean probability.
        hit_rate: The share that happened; on a calibrated voice it tracks `mean_p`.
    """

    lo: float
    hi: float
    n: int
    mean_p: float
    hit_rate: float


def reliability(obs: Sequence[bool], fct: Sequence[float], bins: int = 10) -> tuple[Bin, ...]:
    """Return the reliability curve: forecasts grouped into equal-width probability bins.

    Args:
        obs: Whether each event happened.
        fct: The forecast probabilities, aligned with `obs`; a pair with an invalid forecast is
            skipped, and sequences of different lengths give an empty curve.
        bins: The number of equal-width bins over [0, 1]; values below 1 are read as 1.

    Returns:
        The non-empty bins, lowest first. Never raises.
    """
    k = max(int(bins), 1)
    grouped: list[list[tuple[bool, float]]] = [[] for _ in range(k)]
    for y, p in _clean(obs, fct):
        # The 1e-9 nudge keeps a value on an edge in its bin despite float error (0.57 * 100).
        grouped[min(math.floor(p * k + 1e-9), k - 1)].append((y, p))
    return tuple(
        Bin(
            lo=i / k,
            hi=(i + 1) / k,
            n=len(g),
            mean_p=sum(p for _, p in g) / len(g),
            hit_rate=sum(1 for y, _ in g if y) / len(g),
        )
        for i, g in enumerate(grouped)
        if g
    )


def ece(curve: Sequence[Bin]) -> float | None:
    """Return the expected calibration error of a reliability curve.

    The n-weighted mean of `|mean_p - hit_rate|` over the bins.

    Returns:
        The error, or None for an empty curve.
    """
    total = sum(b.n for b in curve)
    if total == 0:
        return None
    return sum(b.n * abs(b.mean_p - b.hit_rate) for b in curve) / total


__all__ = ["EPS", "Bin", "Score", "brier_score", "ece", "log_score", "reliability", "score"]
