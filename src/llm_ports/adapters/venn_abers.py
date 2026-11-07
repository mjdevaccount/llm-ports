"""A Venn-ABERS `Calibrator` (extra `[venn-abers]`), for voices with few resolved forecasts.

Venn-ABERS (Vovk, Petej and Fedorova) fits two isotonic regressions per test point, one assuming
each outcome, and returns the pair `(p0, p1)` that brackets the calibrated probability; its
validity guarantee holds at any sample size, so it is the calibrator for a small history. The
width of `(p0, p1)` says how much the history can say. `VennAbersCalibrator` is a scikit-learn
estimator: `fit(X, y)`, `predict_proba(X)` (the merged probability) and `predict_interval(X)`.

Invariants:
    - Never raises across the port: too few pairs, one outcome only, or a failed fit leave
      `model_` None and the reason in `insufficient_`, and both predictions return None; a
      failed prediction for one point returns its stated probability unchanged.
    - Every returned probability is in [0, 1].

Rationale: ADR-0013.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

from llm_ports._extras import extra
from llm_ports.forecast._estimator import Params
from llm_ports.forecast._types import valid_p
from llm_ports.forecast.calibrate import Insufficient, proba_rows, usable

with extra("venn-abers"):
    import numpy as np
    from venn_abers import VennAbers

#: Venn-ABERS is valid at any n, but one or two points say nothing; below this it is refused.
MIN_PAIRS = 10


def _two_class(p: Sequence[float]) -> Any:
    return np.array([[1.0 - float(x), float(x)] for x in p], dtype=float)


def _clip(v: float) -> float:
    return min(max(v, 0.0), 1.0)


class VennAbersCalibrator(Params):
    """Fits a Venn-ABERS map.

    Args:
        min_pairs: The floor of usable pairs; `MIN_PAIRS` by default.

    Attributes:
        min_pairs: As given.
        model_: The fitted `venn_abers.VennAbers`, or None when the last fit produced none.
        n_: The pairs the model was fitted on (0 with no model).
        insufficient_: Why the last fit produced no model, or None when it did.
    """

    model_: Any
    n_: int
    insufficient_: Insufficient | None

    def __init__(self, min_pairs: int = MIN_PAIRS) -> None:
        self.min_pairs = min_pairs

    def fit(self, X: Sequence[float], y: Sequence[bool]) -> Self:
        """Fit the map.

        Args:
            X: Stated probabilities.
            y: Whether each event happened, aligned with `X`.

        Returns:
            The estimator, with `model_` or `insufficient_` set. Never raises.
        """
        self.model_, self.n_ = None, 0
        got = usable(X, y, self.min_pairs)
        if isinstance(got, Insufficient):
            self.insufficient_ = got
            return self
        xs, ys = got
        try:
            model = VennAbers()
            model.fit(_two_class(xs), np.array([int(o) for o in ys]))
        except (ValueError, IndexError, ArithmeticError) as exc:
            self.insufficient_ = Insufficient(n=len(xs), need=self.min_pairs, reason=f"venn-abers fit failed: {exc}")
            return self
        self.model_, self.n_, self.insufficient_ = model, len(xs), None
        return self

    def _interval(self, p: float) -> tuple[float, float]:
        if not valid_p(p):
            return (0.5, 0.5)
        p = float(p)
        try:
            _, p0p1 = self.model_.predict_proba(_two_class([p]))
            lo, hi = float(p0p1[0][0]), float(p0p1[0][1])
        except (ValueError, IndexError, ArithmeticError):
            return (p, p)
        return (_clip(lo), _clip(hi))

    def _merged(self, p: float) -> float:
        if not valid_p(p):
            return 0.5
        p0, p1 = self._interval(p)
        denom = 1.0 - p0 + p1
        return float(p) if denom <= 0 else _clip(p1 / denom)

    def predict_interval(self, X: Sequence[float]) -> list[tuple[float, float]] | None:
        """Return `(p0, p1)` per stated probability, the bracket around its calibrated value.

        Returns:
            The brackets, None with no fitted model. A point with an invalid probability gets
            `(0.5, 0.5)`, a failed prediction `(p, p)`. Never raises.
        """
        if getattr(self, "model_", None) is None:
            return None
        return [self._interval(p) for p in X]

    def predict_proba(self, X: Sequence[float]) -> list[tuple[float, float]] | None:
        """Return `(P(no), P(yes))` per stated probability, `P(yes) = p1 / (1 - p0 + p1)`.

        Returns:
            The rows, None with no fitted model. Never raises.
        """
        if getattr(self, "model_", None) is None:
            return None
        return proba_rows([self._merged(p) for p in X])


__all__ = ["MIN_PAIRS", "VennAbersCalibrator"]
