"""An isotonic `Calibrator` on scikit-learn (extra `[sklearn]`).

`IsotonicCalibrator.fit` runs `sklearn.isotonic.IsotonicRegression` and returns its knots as a
`forecast.PiecewiseLinear`, so the fitted map is applied and stored without scikit-learn.

Invariants:
    - Never raises across the port: too few pairs, one outcome only, or a failed fit are
      `forecast.Insufficient`.
    - The returned map is monotone non-decreasing and inside [0, 1].

Rationale: ADR-0013.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from llm_ports._extras import extra
from llm_ports.forecast.calibrate import Insufficient, PiecewiseLinear, usable

with extra("sklearn"):
    from sklearn.isotonic import IsotonicRegression

#: Isotonic regression overfits a handful of points; below this it is refused.
MIN_PAIRS = 30


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Fits a monotone calibration map by isotonic regression.

    Attributes:
        min_pairs: The floor of usable pairs; `MIN_PAIRS` by default.
    """

    min_pairs: int = MIN_PAIRS

    def fit(
        self, p: Sequence[float], outcomes: Sequence[bool], *, weights: Sequence[float] | None = None
    ) -> PiecewiseLinear | Insufficient:
        """Fit the map.

        Args:
            p: Stated probabilities.
            outcomes: Whether each event happened, aligned with `p`.
            weights: Optional sample weights aligned with `p`, to calibrate toward a reweighted
                population; a pair whose weight is not finite and > 0 is dropped. A `weights`
                whose length differs from `p` is `Insufficient`.

        Returns:
            The fitted `PiecewiseLinear`, or `Insufficient`. Never raises.
        """
        w: list[float] | None = None
        if weights is not None:
            if len(weights) != len(p):
                return Insufficient(n=0, need=self.min_pairs, reason="weights not aligned with p")
            keep = [i for i, v in enumerate(weights) if isinstance(v, int | float) and math.isfinite(v) and v > 0]
            p = [p[i] for i in keep]
            outcomes = [outcomes[i] for i in keep if i < len(outcomes)]
            w = [float(weights[i]) for i in keep]
        got = usable(p, outcomes, self.min_pairs)
        if isinstance(got, Insufficient):
            return got
        xs, ys = got
        if w is not None and len(xs) != len(w):
            return Insufficient(n=len(xs), need=self.min_pairs, reason="weights not aligned with usable pairs")
        try:
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
            iso.fit(xs, [float(y) for y in ys], sample_weight=w)
            knots_x = [float(v) for v in iso.X_thresholds_]
            knots_y = [min(max(float(v), 0.0), 1.0) for v in iso.y_thresholds_]
            # Defensive: X_thresholds_ is unique in current scikit-learn, but PiecewiseLinear needs
            # strictly increasing xs, so a repeated x keeps its last (largest) y.
            merged: dict[float, float] = {}
            for x, y in zip(knots_x, knots_y, strict=True):
                merged[x] = y
            kx = tuple(sorted(merged))
            return PiecewiseLinear(xs=kx, ys=tuple(merged[x] for x in kx), n=len(xs))
        except (ValueError, ArithmeticError) as exc:
            return Insufficient(n=len(xs), need=self.min_pairs, reason=f"isotonic fit failed: {exc}")


__all__ = ["MIN_PAIRS", "IsotonicCalibrator"]
