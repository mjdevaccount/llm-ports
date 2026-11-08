"""Platt scaling: a two-parameter logistic calibration map, fitted and applied with the standard library.

`PlattCalibrator` is a scikit-learn estimator (the `Calibrator` port): `fit(X, y)` fits
`P(yes) = sigmoid(a * logit(p) + b)` by ridge-stabilised Newton steps and keeps the result as a
`Platt` in `map_`; `predict_proba(X)` reads it. Two parameters are the robust choice on a small
sample, where an isotonic or beta map over-fits the curve (Platt 1999; Niculescu-Mizil and
Caruana 2005). scikit-learn's `CalibratedClassifierCV(method="sigmoid")` fits the same curve but
needs the base classifier and the library; this one takes stated probabilities and runs anywhere.

A fitted slope `a <= 0` inverts the ranking of the stated probabilities: the voice is
anti-predictive, which is a failure to report, not a correction to apply. The fit is refused
(`insufficient_.reason` starts `inverted slope`) and the fitted values are kept in `fitted_` for
diagnosis. A calibration error such as ECE cannot see that inversion: a map compressed toward the
base rate scores well on ECE with no skill at all.

Invariants:
    - Standard library only.
    - Never raises across the port: too few pairs, one outcome only, a non-finite step or an
      inverted slope leave `map_` None and the reason in `insufficient_`.
    - A fitted `Platt` has a finite slope `a > 0` and a finite intercept, so it is monotone
      increasing and its output is inside (0, 1).

Rationale: ADR-0013.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

from llm_ports.forecast._estimator import Params
from llm_ports.forecast._types import finite_real, valid_p
from llm_ports.forecast.calibrate import Insufficient, proba_rows, usable

#: Below this many usable pairs a fit is refused.
MIN_PAIRS = 60
#: Logit clamp, so a stated 0 or 1 is representable; outputs are clamped to (EPS, 1 - EPS).
EPS = 1e-6


def _logit(p: float) -> float:
    p = min(1.0 - EPS, max(EPS, p))
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass(frozen=True)
class Platt:
    """A fitted Platt map: `p -> sigmoid(a * logit(p) + b)`.

    Attributes:
        a: Slope on the logit of the stated probability; finite and > 0.
        b: Intercept; finite.
        n: The pairs it was fitted on (0 when unknown).

    Raises:
        ValueError: At construction, for a slope that is not finite and > 0 or an intercept
            that is not finite.
    """

    a: float
    b: float
    n: int = 0

    def __post_init__(self) -> None:
        if not (finite_real(self.a) and self.a > 0):
            raise ValueError(f"Platt slope must be finite and > 0, got {self.a!r}")
        if not finite_real(self.b):
            raise ValueError(f"Platt intercept must be finite, got {self.b!r}")
        object.__setattr__(self, "a", float(self.a))
        object.__setattr__(self, "b", float(self.b))

    def __call__(self, p: float) -> float:
        """Return the calibrated probability for `p`; an invalid `p` maps to 0.5. Never raises."""
        if not valid_p(p):
            return 0.5
        out = _sigmoid(self.a * _logit(float(p)) + self.b)
        return min(1.0 - EPS, max(EPS, out))

    def predict(self, X: Sequence[float]) -> list[float]:
        """Return the calibrated probability per stated probability."""
        return [self(p) for p in X]

    def as_dict(self) -> dict[str, Any]:
        """Return the map as plain JSON-serialisable values."""
        return {"a": self.a, "b": self.b, "n": self.n}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Platt:
        """Rebuild a map from `as_dict` output.

        Raises:
            ValueError: For a missing key or values that fail the construction checks.
        """
        try:
            return cls(a=float(data["a"]), b=float(data["b"]), n=int(data.get("n", 0)))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"not a Platt dict: {exc}") from exc


def _loss(z: Sequence[float], y: Sequence[float], ridge: float, a: float, b: float) -> float:
    """The ridge-penalised logistic negative log-likelihood at `(a, b)`."""
    total = 0.5 * ridge * (a * a + b * b)
    for zi, yi in zip(z, y, strict=True):
        s = a * zi + b
        # log(1 + e^s) - y*s, computed without overflow
        total += (s if s > 0 else 0.0) + math.log1p(math.exp(-abs(s))) - yi * s
    return total


def _newton(z: Sequence[float], y: Sequence[float], ridge: float, max_iter: int, tol: float) -> tuple[float, float]:
    """Minimise the ridge-penalised logistic loss over `(a, b)` from `(1, 0)`; may return non-finite values.

    A full Newton step is taken whenever it lowers the loss; otherwise the step is halved until
    it does. Undamped Newton overshoots on a strongly inverted voice: from `(1, 0)` it ran to a
    slope near 7e5 of the WRONG sign, a map that would have passed the inversion refusal.
    """
    a, b = 1.0, 0.0
    cur = _loss(z, y, ridge, a, b)
    for _ in range(max_iter):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for zi, yi in zip(z, y, strict=True):
            mu = _sigmoid(a * zi + b)
            r = mu - yi
            w = mu * (1.0 - mu)
            g0 += r * zi
            g1 += r
            h00 += w * zi * zi
            h01 += w * zi
            h11 += w
        g0 += ridge * a
        g1 += ridge * b
        h00 += ridge
        h11 += ridge
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12 or not math.isfinite(det):
            break
        da = (h11 * g0 - h01 * g1) / det
        db = (h00 * g1 - h01 * g0) / det
        if not (math.isfinite(da) and math.isfinite(db)):
            break
        step = 1.0
        for _ in range(40):
            nxt = _loss(z, y, ridge, a - step * da, b - step * db)
            if nxt <= cur:
                break
            step *= 0.5
        else:
            break  # no descent along the Newton direction: at the optimum to float precision
        a -= step * da
        b -= step * db
        cur = nxt
        if abs(step * da) < tol and abs(step * db) < tol:
            break
    return a, b


class PlattCalibrator(Params):
    """Fits a Platt calibration map from stated probabilities and outcomes.

    Args:
        min_pairs: The floor of usable pairs; `MIN_PAIRS` by default.
        ridge: L2 penalty on `(a, b)`, which keeps a Newton step finite on separable or thin data.
        max_iter: Newton steps at most.
        tol: Stop once both parameter steps are below this.

    Attributes:
        map_: The fitted `Platt`, or None when the last fit produced none. Set by `fit`.
        insufficient_: Why the last fit produced no map, or None.
        fitted_: The `(a, b)` the last fit reached, kept even when refused; None before a fit
            or when there was too little data to fit.
    """

    map_: Platt | None
    insufficient_: Insufficient | None
    fitted_: tuple[float, float] | None

    def __init__(self, min_pairs: int = MIN_PAIRS, ridge: float = 1e-3, max_iter: int = 50, tol: float = 1e-8) -> None:
        self.min_pairs = min_pairs
        self.ridge = ridge
        self.max_iter = max_iter
        self.tol = tol
        self.map_ = None
        self.insufficient_ = None
        self.fitted_ = None

    def fit(self, X: Sequence[float], y: Sequence[bool]) -> Self:
        """Fit the map; never raises (the reason a fit produced no map lands in `insufficient_`).

        Returns:
            The estimator.
        """
        self.map_, self.insufficient_, self.fitted_ = None, None, None
        pairs = usable(X, y, self.min_pairs)
        if isinstance(pairs, Insufficient):
            self.insufficient_ = pairs
            return self
        xs, ys = pairs
        a, b = _newton([_logit(p) for p in xs], [1.0 if o else 0.0 for o in ys], self.ridge, self.max_iter, self.tol)
        n = len(xs)
        if not (math.isfinite(a) and math.isfinite(b)):
            self.insufficient_ = Insufficient(
                n=n, need=self.min_pairs, reason="the fit did not converge to finite values"
            )
            return self
        self.fitted_ = (a, b)
        if a <= 0:
            self.insufficient_ = Insufficient(
                n=n, need=self.min_pairs, reason=f"inverted slope a={a:.6g}: the voice ranks backwards"
            )
            return self
        self.map_ = Platt(a=a, b=b, n=n)
        return self

    def predict_proba(self, X: Sequence[float]) -> list[tuple[float, float]] | None:
        """Return `(P(no), P(yes))` per stated probability, or None with no usable fit."""
        if self.map_ is None:
            return None
        return proba_rows(self.map_.predict(X))


__all__ = ["EPS", "MIN_PAIRS", "Platt", "PlattCalibrator"]
