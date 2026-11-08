"""Combining voices: one probability from several forecasts of the same event.

- `Combiner`: the port, River's classifier protocol. `predict_proba_one(x)` reads the voices'
  probabilities for one event (`x` is `voice -> probability`) and returns `{False: P(no),
  True: P(yes)}`; `learn_one(x, y)` hears the outcome, so an online combiner can move its
  weights. A River binary classifier satisfies it unwrapped.
- `LogitPool`: the fixed baseline, a weighted mean in log-odds, optionally extremized (Satopää
  et al.: averaging forecasts that share information pulls the pool toward 0.5, and an alpha
  above 1 pushes it back). With no weights it is the equal-weight pool; it never learns.
- `PriorPool`: wraps any combiner with a prior voice: shrinks its log-odds toward the prior's,
  extremizes the result, and answers with the prior alone while the inner combiner abstains.
- `BestVoice`: the best-single-voice baseline, online: it answers with whichever speaking voice
  has the lowest running Brier score so far.
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
from llm_ports.forecast.score import EPS, brier_score


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


@dataclass
class PriorPool:
    """Shrink a combiner toward a prior voice, then extremize: `expit(alpha * z)`.

    `z = (1 - shrink) * logit(inner) + shrink * logit(prior)`, where `inner` is the wrapped
    combiner's probability and `prior` is `x[prior_key]`. The prior also reaches the inner
    combiner as an ordinary voice, so a learned combiner can weigh it. While the inner combiner
    abstains (`{}`, e.g. before its warmup), the pool answers with the prior alone, extremized.
    Without a prior in `x` it is the inner combiner, extremized.

    Attributes:
        inner: The wrapped combiner; it hears every outcome `learn_one` hears.
        prior_key: The voice in `x` that carries the prior.
        alpha: The extremizing factor; 1.0 leaves the pool as is.
        shrink: The weight on the prior's log-odds, in [0, 1]; 0 ignores it beyond the fallback.
            A non-finite `alpha` or a `shrink` outside [0, 1] makes `predict_proba_one` return `{}`.
    """

    inner: Combiner
    prior_key: str = "prior"
    alpha: float = 1.0
    shrink: float = 0.0

    def predict_proba_one(self, x: Mapping[str, float]) -> dict[bool, float]:
        """Return `{False: P(no), True: P(yes)}`, or `{}` when neither the inner combiner nor the prior answers. Never raises."""
        if not finite_real(self.alpha) or not finite_real(self.shrink) or not 0.0 <= self.shrink <= 1.0:
            return {}
        raw = x.get(self.prior_key)
        prior = float(raw) if raw is not None and valid_p(raw) else None
        got = self.inner.predict_proba_one(x).get(True)
        inner = float(got) if got is not None and valid_p(got) else None
        if inner is not None and prior is not None:
            w = float(self.shrink)
            z = (1.0 - w) * logit(inner) + w * logit(prior)
        elif inner is not None:
            z = logit(inner)
        elif prior is not None:
            z = logit(prior)
        else:
            return {}
        return proba_one(expit(float(self.alpha) * z))

    def learn_one(self, x: Mapping[str, float], y: bool) -> None:
        """Pass the outcome to the inner combiner. Never raises."""
        self.inner.learn_one(x, y)


@dataclass
class BestVoice:
    """The best single voice so far: answers with the speaking voice of lowest running Brier score.

    A voice is eligible once it has been heard on `min_n` outcomes; among the eligible voices in
    `x`, the one with the lowest mean Brier score answers (ties: the voice name, so the choice is
    deterministic). With no eligible voice it abstains (`{}`).

    Attributes:
        min_n: Outcomes a voice must have been scored on before it can answer.
        n: `voice -> outcomes it was scored on`.
        brier_sum: `voice -> sum of its Brier scores`.
    """

    min_n: int = 20
    n: dict[str, int] = field(default_factory=dict)
    brier_sum: dict[str, float] = field(default_factory=dict)

    def best(self, x: Mapping[str, float]) -> str | None:
        """Return the voice that would answer for `x`, or None. Never raises."""
        ok = [v for v, p in x.items() if valid_p(p) and self.n.get(v, 0) >= max(self.min_n, 1)]
        if not ok:
            return None
        return min(ok, key=lambda v: (self.brier_sum[v] / self.n[v], v))

    def predict_proba_one(self, x: Mapping[str, float]) -> dict[bool, float]:
        """Return the best eligible voice's `{False: P(no), True: P(yes)}`, or `{}`. Never raises."""
        v = self.best(x)
        return {} if v is None else proba_one(float(x[v]))

    def learn_one(self, x: Mapping[str, float], y: bool) -> None:
        """Score every usable voice in `x` against `y`. Never raises."""
        for v, p in x.items():
            if valid_p(p):
                self.n[v] = self.n.get(v, 0) + 1
                self.brier_sum[v] = self.brier_sum.get(v, 0.0) + brier_score(bool(y), float(p))


__all__ = ["BestVoice", "Combiner", "LogitPool", "PriorPool", "expit", "logit", "proba_one"]
