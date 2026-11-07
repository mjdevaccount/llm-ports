"""An isotonic `Calibrator` on scikit-learn (extra `[sklearn]`).

`IsotonicCalibrator` is a scikit-learn estimator: `fit(X, y, sample_weight=None)` runs
`sklearn.isotonic.IsotonicRegression` and keeps its knots as a `forecast.PiecewiseLinear` in
`map_`, so the fitted map is applied and stored without scikit-learn; `predict_proba(X)` reads it.

Invariants:
    - Never raises across the port: too few pairs, one outcome only, unaligned inputs or a failed
      fit leave `map_` None and the reason in `insufficient_`, and `predict_proba` returns None.
    - The fitted map is monotone non-decreasing and inside [0, 1].

Rationale: ADR-0013.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

from llm_ports._extras import extra
from llm_ports.forecast._estimator import Params
from llm_ports.forecast._types import finite_real, valid_p
from llm_ports.forecast.calibrate import Insufficient, PiecewiseLinear, usable

with extra("sklearn"):
    from sklearn.isotonic import IsotonicRegression

#: Isotonic regression overfits a handful of points; below this it is refused.
MIN_PAIRS = 30


class IsotonicCalibrator(Params):
    """Fits a monotone calibration map by isotonic regression.

    Args:
        min_pairs: The floor of usable pairs; `MIN_PAIRS` by default.

    Attributes:
        min_pairs: As given.
        map_: The fitted map, or None when the last fit produced none. Set by `fit`.
        insufficient_: Why the last fit produced no map, or None when it did. Set by `fit`.
    """

    map_: PiecewiseLinear | None
    insufficient_: Insufficient | None

    def __init__(self, min_pairs: int = MIN_PAIRS) -> None:
        self.min_pairs = min_pairs

    def _refuse(self, n: int, reason: str) -> Self:
        self.map_, self.insufficient_ = None, Insufficient(n=n, need=self.min_pairs, reason=reason)
        return self

    def fit(
        self,
        X: Sequence[float],
        y: Sequence[bool],
        sample_weight: Sequence[float] | None = None,
    ) -> Self:
        """Fit the map.

        Args:
            X: Stated probabilities.
            y: Whether each event happened, aligned with `X`.
            sample_weight: Optional weights aligned with `X`, to calibrate toward a reweighted
                population. A pair is dropped when its probability is invalid or its weight is
                not a finite real > 0. Inputs of different lengths are not aligned (a derived
                array that lost a row would misalign every pair after it), so nothing is fitted.

        Returns:
            The estimator, with `map_` or `insufficient_` set. Never raises.
        """
        w: list[float] | None = None
        if sample_weight is not None:
            if not len(sample_weight) == len(X) == len(y):
                return self._refuse(0, "sample_weight not aligned with X and y")
            # One positional walk over the triples (never index: a pandas Series indexes by label),
            # so `usable` below drops nothing more and the three lists stay aligned.
            kept = [
                (float(x), bool(o), float(v))
                for x, o, v in zip(X, y, sample_weight, strict=True)
                if valid_p(x) and finite_real(v) and v > 0
            ]
            X, y, w = [k[0] for k in kept], [k[1] for k in kept], [k[2] for k in kept]
        got = usable(X, y, self.min_pairs)
        if isinstance(got, Insufficient):
            self.map_, self.insufficient_ = None, got
            return self
        xs, ys = got
        try:
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
            iso.fit(xs, [float(o) for o in ys], sample_weight=w)
            knots_x = [float(v) for v in iso.X_thresholds_]
            knots_y = [min(max(float(v), 0.0), 1.0) for v in iso.y_thresholds_]
            # Defensive: X_thresholds_ is unique in current scikit-learn, but PiecewiseLinear needs
            # strictly increasing xs, so a repeated x keeps its last (largest) y.
            merged: dict[float, float] = {}
            for kx, ky in zip(knots_x, knots_y, strict=True):
                merged[kx] = ky
            xs_sorted = tuple(sorted(merged))
            self.map_ = PiecewiseLinear(xs=xs_sorted, ys=tuple(merged[k] for k in xs_sorted), n=len(xs))
        except (ValueError, ArithmeticError) as exc:
            return self._refuse(len(xs), f"isotonic fit failed: {exc}")
        self.insufficient_ = None
        return self

    def predict_proba(self, X: Sequence[float]) -> list[tuple[float, float]] | None:
        """Return `(P(no), P(yes))` per stated probability; None with no fitted map. Never raises."""
        m = getattr(self, "map_", None)
        return None if m is None else m.predict_proba(X)


__all__ = ["MIN_PAIRS", "IsotonicCalibrator"]
