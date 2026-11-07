"""A Venn-ABERS `Calibrator` (extra `[venn-abers]`), for voices with few resolved forecasts.

Venn-ABERS (Vovk, Petej and Fedorova) fits two isotonic regressions per test point, one assuming
each outcome, and returns the pair `(p0, p1)` that brackets the calibrated probability; its
validity guarantee holds at any sample size, so it is the calibrator for a small history. The
width of `(p0, p1)` says how much the history can say.

Invariants:
    - Never raises across the port: too few pairs, one outcome only, or a failed fit are
      `forecast.Insufficient`; a failed prediction returns the stated probability unchanged.
    - Every returned probability is in [0, 1].

Rationale: ADR-0013.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from llm_ports._extras import extra
from llm_ports.forecast._types import valid_p
from llm_ports.forecast.calibrate import Insufficient, usable

with extra("venn-abers"):
    import numpy as np
    from venn_abers import VennAbers

#: Venn-ABERS is valid at any n, but one or two points say nothing; below this it is refused.
MIN_PAIRS = 10


def _two_class(p: Sequence[float]) -> Any:
    return np.array([[1.0 - x, x] for x in p], dtype=float)


@dataclass(frozen=True)
class VennAbersFit:
    """A fitted Venn-ABERS map.

    Attributes:
        model: The fitted `venn_abers.VennAbers`.
        n: The pairs it was fitted on.
    """

    model: Any
    n: int

    def interval(self, p: float) -> tuple[float, float]:
        """Return `(p0, p1)`, the bracket around the calibrated probability for `p`.

        Returns:
            The bracket; `(0.5, 0.5)` for an invalid `p`, `(p, p)` for a failed prediction.
            Never raises.
        """
        if not valid_p(p):
            return (0.5, 0.5)
        try:
            _, p0p1 = self.model.predict_proba(_two_class([p]))
            lo, hi = float(p0p1[0][0]), float(p0p1[0][1])
        except (ValueError, IndexError, ArithmeticError):
            return (p, p)
        return (min(max(lo, 0.0), 1.0), min(max(hi, 0.0), 1.0))

    def __call__(self, p: float) -> float:
        """Return the merged calibrated probability `p1 / (1 - p0 + p1)`. Never raises."""
        if not valid_p(p):
            return 0.5
        p0, p1 = self.interval(p)
        denom = 1.0 - p0 + p1
        return p if denom <= 0 else min(max(p1 / denom, 0.0), 1.0)


@dataclass(frozen=True)
class VennAbersCalibrator:
    """Fits a Venn-ABERS map.

    Attributes:
        min_pairs: The floor of usable pairs; `MIN_PAIRS` by default.
    """

    min_pairs: int = MIN_PAIRS

    def fit(self, p: Sequence[float], outcomes: Sequence[bool]) -> VennAbersFit | Insufficient:
        """Fit the map.

        Args:
            p: Stated probabilities.
            outcomes: Whether each event happened, aligned with `p`.

        Returns:
            The fitted `VennAbersFit`, or `Insufficient`. Never raises.
        """
        got = usable(p, outcomes, self.min_pairs)
        if isinstance(got, Insufficient):
            return got
        xs, ys = got
        try:
            model = VennAbers()
            model.fit(_two_class(xs), np.array([int(y) for y in ys]))
        except (ValueError, IndexError, ArithmeticError) as exc:
            return Insufficient(n=len(xs), need=self.min_pairs, reason=f"venn-abers fit failed: {exc}")
        return VennAbersFit(model=model, n=len(xs))


__all__ = ["MIN_PAIRS", "VennAbersCalibrator", "VennAbersFit"]
