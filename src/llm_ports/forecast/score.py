"""Proper scores for binary forecasts: Brier, log loss, skill against the base rate, reliability.

A proper scoring rule is minimised in expectation only by reporting the true probability, so a
voice cannot improve its score by hedging toward 0.5 or by overclaiming. Each function takes
`(p, outcome)` pairs, the forecast probability and whether the event happened.

Invariants:
    - Pure, standard library only.
    - A pair whose `p` is not a finite probability is skipped, never scored.
    - A statistic the data cannot define is None, never a number.

Rationale: ADR-0013.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from llm_ports.forecast._types import valid_p

#: `(forecast probability, the event happened)`.
Pair = tuple[float, bool]
#: Probabilities are clipped to `[EPS, 1 - EPS]` before a logarithm, so a confident miss costs a
#: large finite loss rather than infinity.
EPS = 1e-6


def brier(p: float, outcome: bool) -> float:
    """Return the Brier score of one binary forecast, `(p - outcome) ** 2`; 0 is perfect."""
    return (p - float(outcome)) ** 2


def log_loss(p: float, outcome: bool) -> float:
    """Return the log loss of one binary forecast, `-ln(p)` if it happened else `-ln(1 - p)`.

    `p` is clipped to `[EPS, 1 - EPS]`, so the loss is finite.
    """
    q = min(max(p, EPS), 1.0 - EPS)
    return -math.log(q if outcome else 1.0 - q)


def _clean(pairs: Iterable[Pair]) -> list[Pair]:
    return [(float(p), bool(y)) for p, y in pairs if valid_p(p)]


@dataclass(frozen=True)
class Score:
    """How a set of binary forecasts scored.

    Attributes:
        n: Pairs scored.
        mean_p: The mean forecast probability; None at n = 0.
        hit_rate: The share that happened (the base rate); None at n = 0.
        brier: The mean Brier score; None at n = 0.
        log_loss: The mean log loss; None at n = 0.
        brier_skill: `1 - brier / (hit_rate * (1 - hit_rate))`, skill over always forecasting the
            base rate (> 0 beats it); None when undefined (n = 0, or every outcome the same).
    """

    n: int
    mean_p: float | None
    hit_rate: float | None
    brier: float | None
    log_loss: float | None
    brier_skill: float | None

    @property
    def overconfidence(self) -> float | None:
        """`mean_p - hit_rate`: positive when the voice claims more than it delivers; None at n = 0."""
        if self.mean_p is None or self.hit_rate is None:
            return None
        return self.mean_p - self.hit_rate


def score(pairs: Iterable[Pair]) -> Score:
    """Score binary forecasts.

    Args:
        pairs: `(p, outcome)` pairs; a pair with an invalid `p` is skipped.

    Returns:
        The `Score`. Never raises.
    """
    got = _clean(pairs)
    n = len(got)
    if n == 0:
        return Score(n=0, mean_p=None, hit_rate=None, brier=None, log_loss=None, brier_skill=None)
    mean_p = sum(p for p, _ in got) / n
    hit = sum(1 for _, y in got if y) / n
    b = sum(brier(p, y) for p, y in got) / n
    ll = sum(log_loss(p, y) for p, y in got) / n
    ref = hit * (1.0 - hit)
    return Score(n=n, mean_p=mean_p, hit_rate=hit, brier=b, log_loss=ll, brier_skill=1.0 - b / ref if ref > 0 else None)


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


def reliability(pairs: Iterable[Pair], bins: int = 10) -> tuple[Bin, ...]:
    """Return the reliability curve: forecasts grouped into equal-width probability bins.

    Args:
        pairs: `(p, outcome)` pairs; a pair with an invalid `p` is skipped.
        bins: The number of equal-width bins over [0, 1]; values below 1 are read as 1.

    Returns:
        The non-empty bins, lowest first. Never raises.
    """
    k = max(int(bins), 1)
    grouped: list[list[Pair]] = [[] for _ in range(k)]
    for p, y in _clean(pairs):
        # The 1e-9 nudge keeps a value on an edge in its bin despite float error (0.57 * 100).
        grouped[min(math.floor(p * k + 1e-9), k - 1)].append((p, y))
    return tuple(
        Bin(
            lo=i / k,
            hi=(i + 1) / k,
            n=len(g),
            mean_p=sum(p for p, _ in g) / len(g),
            hit_rate=sum(1 for _, y in g if y) / len(g),
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


__all__ = ["EPS", "Bin", "Pair", "Score", "brier", "ece", "log_loss", "reliability", "score"]
