"""A learned `Combiner` on River's online logistic regression (extra `[river]`).

`RiverLogitPool` regresses the outcome on each voice's log-odds, one feature per voice, plus a
silent flag per voice it has heard before and that did not speak this time: a voice's silence
can carry information. It learns one event at a time, so its weights follow drift without a
nightly refit. Until it has heard `warmup` outcomes it does not combine (`{}`), so a caller falls
back to its baseline rather than read an untrained model. It speaks the `Combiner` protocol,
River's own: `predict_proba_one(x)` and `learn_one(x, y)`, with `x` as `voice -> probability`.

Invariants:
    - Never raises across the port: a River failure leaves the state unchanged (`learn_one`) or
      returns `{}` (`predict_proba_one`).
    - A combined probability is in [0, 1].

Rationale: ADR-0013.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from llm_ports._extras import extra
from llm_ports.forecast._types import valid_p
from llm_ports.forecast.combine import logit, proba_one

with extra("river"):
    from river import linear_model, optim

#: Outcomes heard before the pool combines.
WARMUP = 50
#: Voice features are log-odds clipped to `[FEATURE_EPS, 1 - FEATURE_EPS]` (about +-6.9), so one
#: voice saying 0 or 1 cannot swamp an unscaled linear model.
FEATURE_EPS = 1e-3


class RiverLogitPool:
    """An online logistic logit pool.

    Args:
        lr: The SGD learning rate.
        l2: The L2 penalty on the weights.
        warmup: Outcomes heard before `predict_proba_one` answers; `WARMUP` by default.

    Attributes:
        seen: Outcomes heard so far.
        failures: Updates River refused (state left unchanged); a rising count means the pool
            is not learning.
        voices: Every voice heard so far.
    """

    def __init__(self, lr: float = 0.05, l2: float = 0.0, warmup: int = WARMUP) -> None:
        self._model: Any = linear_model.LogisticRegression(optimizer=optim.SGD(lr), l2=l2)
        self._warmup = warmup
        self.seen: int = 0
        self.failures: int = 0
        self.voices: set[str] = set()

    def _features(self, x: Mapping[str, float]) -> dict[str, float]:
        f = {f"voice:{v}": logit(p, FEATURE_EPS) for v, p in x.items() if valid_p(p)}
        for v in self.voices:
            if f"voice:{v}" not in f:
                f[f"silent:{v}"] = 1.0
        return f

    def predict_proba_one(self, x: Mapping[str, float]) -> dict[bool, float]:
        """Return `{False: P(no), True: P(yes)}`; `{}` before warmup or with no usable voice. Never raises."""
        f = self._features(x)
        if self.seen < self._warmup or not any(k.startswith("voice:") for k in f):
            return {}
        try:
            p = float(self._model.predict_proba_one(f).get(True, math.nan))
        except (ValueError, ArithmeticError, KeyError):
            return {}
        return proba_one(p) if valid_p(p) else {}

    def learn_one(self, x: Mapping[str, float], y: bool) -> None:
        """Update the weights on one resolved event; an event with no usable voice is ignored. Never raises."""
        f = self._features(x)
        if not any(k.startswith("voice:") for k in f):
            return
        try:
            self._model.learn_one(f, bool(y))
        except (ValueError, ArithmeticError, KeyError):
            self.failures += 1
            return
        self.seen += 1
        self.voices.update(v for v, p in x.items() if valid_p(p))

    def weights(self) -> dict[str, float]:
        """Return the current feature weights (`voice:<v>`, `silent:<v>`) for a readout."""
        return {str(k): float(v) for k, v in self._model.weights.items()}


__all__ = ["FEATURE_EPS", "WARMUP", "RiverLogitPool"]
