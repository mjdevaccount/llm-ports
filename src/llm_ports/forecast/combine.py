"""Combining voices: one probability from several forecasts of the same event.

- `Combiner`: the port. `combine` reads the voices' probabilities for one event; `learn` hears
  the outcome, so an online combiner can move its weights.
- `LogitPool`: the fixed baseline, a weighted mean in log-odds, optionally extremized (Satopää
  et al.: averaging forecasts that share information pulls the pool toward 0.5, and an alpha
  above 1 pushes it back). With no weights it is the equal-weight pool; it never learns.
- `logit` / `expit`: the log-odds transform and its inverse, clipped so a 0 or 1 stays finite.

Invariants:
    - Pure, standard library only; learned combiners live in adapters (`adapters.river`).
    - A combined probability is in (0, 1); no usable voice means None, never 0.5.

Rationale: ADR-0013.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from llm_ports.forecast._types import valid_p
from llm_ports.forecast.score import EPS


def logit(p: float, eps: float = EPS) -> float:
    """Return `ln(p / (1 - p))`, with `p` clipped to `[eps, 1 - eps]`."""
    q = min(max(float(p), eps), 1.0 - eps)
    return math.log(q / (1.0 - q))


def expit(x: float) -> float:
    """Return `1 / (1 + exp(-x))`, the inverse of `logit`, without overflow."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


class Combiner(Protocol):
    """Combines voices' probabilities for one event."""

    def combine(self, ps: Mapping[str, float]) -> float | None:
        """Return the combined probability.

        Args:
            ps: `voice -> probability` for the voices that spoke; a silent voice is absent.

        Returns:
            The combined probability, or None when no voice is usable or the combiner cannot
            combine yet. Never raises.
        """
        ...

    def learn(self, ps: Mapping[str, float], outcome: bool) -> None:
        """Hear the outcome of an event `combine` was asked about. Never raises."""
        ...


def _finite_real(v: object) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v)


@dataclass
class LogitPool:
    """A weighted mean in log-odds, extremized by `alpha`; a fixed baseline that never learns.

    Attributes:
        weights: `voice -> weight`; a voice absent here weighs 1.0. Weights are renormalised
            over the voices that spoke, and a voice whose weight is not finite and > 0 is ignored.
        alpha: The extremizing factor; 1.0 (default) leaves the pool as is. A non-finite
            `alpha` makes `combine` return None.
    """

    weights: Mapping[str, float] = field(default_factory=dict)
    alpha: float = 1.0

    def combine(self, ps: Mapping[str, float]) -> float | None:
        """Return the pooled probability, or None when no voice is usable. Never raises."""
        terms = [(self.weights.get(v, 1.0), logit(p)) for v, p in ps.items() if valid_p(p)]
        terms = [(float(w), x) for w, x in terms if _finite_real(w) and w > 0]
        total = sum(w for w, _ in terms)
        if not terms or total <= 0 or not _finite_real(self.alpha):
            return None
        return expit(float(self.alpha) * sum(w * x for w, x in terms) / total)

    def learn(self, ps: Mapping[str, float], outcome: bool) -> None:
        """Do nothing: the pool is fixed. Never raises."""
        del ps, outcome


__all__ = ["Combiner", "LogitPool", "expit", "logit"]
