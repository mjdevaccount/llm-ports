"""A learned `Combiner` on River's online logistic regression (extra `[river]`).

`RiverLogitPool` regresses the outcome on each voice's log-odds, one feature per voice, plus a
silent flag per voice it has heard before and that did not speak this time: a voice's silence
can carry information. It learns one event at a time, so its weights follow drift without a
nightly refit. Until it has heard `warmup` outcomes it does not combine (None), so a caller falls
back to its baseline rather than read an untrained model.

Invariants:
    - Never raises across the port: a River failure leaves the state unchanged (`learn`) or
      returns None (`combine`).
    - A combined probability is in [0, 1].

Rationale: ADR-0013.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from llm_ports._extras import extra
from llm_ports.forecast._types import valid_p
from llm_ports.forecast.combine import logit

with extra("river"):
    from river import linear_model, optim

#: Outcomes heard before the pool combines.
WARMUP = 50


class RiverLogitPool:
    """An online logistic logit pool.

    Args:
        lr: The SGD learning rate.
        l2: The L2 penalty on the weights.
        warmup: Outcomes heard before `combine` answers; `WARMUP` by default.

    Attributes:
        seen: Outcomes heard so far.
        voices: Every voice heard so far.
    """

    def __init__(self, lr: float = 0.05, l2: float = 0.0, warmup: int = WARMUP) -> None:
        self._model: Any = linear_model.LogisticRegression(optimizer=optim.SGD(lr), l2=l2)
        self._warmup = warmup
        self.seen: int = 0
        self.voices: set[str] = set()

    def _features(self, ps: Mapping[str, float]) -> dict[str, float]:
        x = {f"voice:{v}": logit(p) for v, p in ps.items() if valid_p(p)}
        for v in self.voices:
            if f"voice:{v}" not in x:
                x[f"silent:{v}"] = 1.0
        return x

    def combine(self, ps: Mapping[str, float]) -> float | None:
        """Return the combined probability; None before warmup or with no usable voice. Never raises."""
        x = self._features(ps)
        if self.seen < self._warmup or not any(k.startswith("voice:") for k in x):
            return None
        try:
            p = float(self._model.predict_proba_one(x).get(True, math.nan))
        except (ValueError, ArithmeticError, KeyError):
            return None
        return p if valid_p(p) else None

    def learn(self, ps: Mapping[str, float], outcome: bool) -> None:
        """Update the weights on one resolved event; an event with no usable voice is ignored. Never raises."""
        x = self._features(ps)
        if not any(k.startswith("voice:") for k in x):
            return
        try:
            self._model.learn_one(x, bool(outcome))
        except (ValueError, ArithmeticError, KeyError):
            return
        self.seen += 1
        self.voices.update(v for v, p in ps.items() if valid_p(p))

    def weights(self) -> dict[str, float]:
        """Return the current feature weights (`voice:<v>`, `silent:<v>`) for a readout."""
        return {str(k): float(v) for k, v in self._model.weights.items()}


__all__ = ["WARMUP", "RiverLogitPool"]
