"""Combining voices: one probability from several forecasts of the same event.

- `Combiner`: the port, River's classifier protocol. `predict_proba_one(x)` reads the voices'
  probabilities for one event (`x` is `voice -> probability`) and returns `{False: P(no),
  True: P(yes)}`; `learn_one(x, y)` hears the outcome, so an online combiner can move its
  weights. A River binary classifier satisfies it unwrapped.
- `LogitPool`: the fixed baseline, a weighted mean in log-odds, optionally extremized (Satopää
  et al.: averaging forecasts that share information pulls the pool toward 0.5, and an alpha
  above 1 pushes it back). With no weights it is the equal-weight pool; it never learns.
- `logit` / `expit`: the log-odds transform and its inverse, clipped so a 0 or 1 stays finite.

Invariants:
    - Pure, standard library only; learned combiners live in adapters (`adapters.river`).
    - A combined probability is in (0, 1); no usable voice is an empty dict, never 0.5, so
      `predict_proba_one(x).get(True)` is None.

Rationale: ADR-0013.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from llm_ports.forecast._types import finite_real, valid_p
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


def proba_one(p: float | None) -> dict[bool, float]:
    """Return `{False: 1 - p, True: p}`, River's `predict_proba_one` layout; `{}` for None."""
    return {} if p is None else {False: 1.0 - p, True: p}


class Combiner(Protocol):
    """Combines voices' probabilities for one event (River's classifier protocol)."""

    def predict_proba_one(self, x: Mapping[str, float]) -> dict[bool, float]:
        """Return the combined probability as `{False: P(no), True: P(yes)}`.

        Args:
            x: `voice -> probability` for the voices that spoke; a silent voice is absent.

        Returns:
            The two class probabilities, or `{}` when no voice is usable or the combiner cannot
            combine yet. Never raises.
        """
        ...

    def learn_one(self, x: Mapping[str, float], y: bool) -> None:
        """Hear the outcome `y` of an event `predict_proba_one` was asked about. Never raises."""
        ...


@dataclass
class LogitPool:
    """A weighted mean in log-odds, extremized by `alpha`; a fixed baseline that never learns.

    Attributes:
        weights: `voice -> weight`; a voice absent here weighs 1.0. Weights are renormalised
            over the voices that spoke, and a voice whose weight is not finite and > 0 is ignored.
        alpha: The extremizing factor; 1.0 (default) leaves the pool as is. A non-finite
            `alpha` makes `predict_proba_one` return `{}`.
    """

    weights: Mapping[str, float] = field(default_factory=dict)
    alpha: float = 1.0

    def predict_proba_one(self, x: Mapping[str, float]) -> dict[bool, float]:
        """Return the pooled `{False: P(no), True: P(yes)}`, or `{}` when no voice is usable. Never raises."""
        terms = [(self.weights.get(v, 1.0), logit(p)) for v, p in x.items() if valid_p(p)]
        terms = [(float(w), z) for w, z in terms if finite_real(w) and w > 0]
        total = sum(w for w, _ in terms)
        if not terms or total <= 0 or not finite_real(self.alpha):
            return {}
        return proba_one(expit(float(self.alpha) * sum(w * z for w, z in terms) / total))

    def learn_one(self, x: Mapping[str, float], y: bool) -> None:
        """Do nothing: the pool is fixed. Never raises."""
        del x, y


__all__ = ["Combiner", "LogitPool", "expit", "logit", "proba_one"]
