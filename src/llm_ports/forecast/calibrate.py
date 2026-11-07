"""Calibration: map a voice's stated probability to the probability it has earned.

- `Calibrator`: the port, the scikit-learn estimator protocol. `fit(X, y)` returns the
  estimator; `predict_proba(X)` returns one `(P(no), P(yes))` row per stated probability, the
  column order of a binary scikit-learn classifier. A scikit-learn, venn-abers or MAPIE
  calibrator satisfies it unwrapped.
- `Insufficient`: why a fit produced no map (too few resolved forecasts, one outcome only); a
  calibrator here records it in `insufficient_` and its `predict_proba` returns None.
- `PiecewiseLinear`: a monotone map held as knots, applied with the standard library and
  serialised as JSON, so a map fitted offline (an isotonic fit, `adapters.sklearn`) is applied
  where no fitting library is installed.

Invariants:
    - Pure, standard library only; the fitting libraries live in adapters.
    - A calibrated probability is always in [0, 1].

Rationale: ADR-0013.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, Self

from llm_ports.forecast._types import valid_p


@dataclass(frozen=True)
class Insufficient:
    """Too few resolved forecasts to fit a map.

    Attributes:
        n: Usable pairs given.
        need: The fitter's floor.
        reason: Why, in words (`n < need`, `one outcome only`).
    """

    n: int
    need: int
    reason: str


class Calibrator(Protocol):
    """Fits a calibration map from resolved forecasts (the scikit-learn estimator protocol)."""

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the constructor parameters by name."""
        ...

    def set_params(self, **params: Any) -> Self:
        """Set constructor parameters by name and return the estimator."""
        ...

    def fit(self, X: Sequence[float], y: Sequence[bool]) -> Self:
        """Fit a map from stated probabilities and what happened.

        Args:
            X: Stated probabilities.
            y: Whether each event happened, aligned with `X`.

        Returns:
            The estimator. A llm-ports calibrator never raises; too little data is recorded in
            `insufficient_`.
        """
        ...

    def predict_proba(self, X: Sequence[float]) -> list[tuple[float, float]] | None:
        """Return `(P(no), P(yes))` per stated probability, or None with no usable fit."""
        ...


def usable(X: Sequence[float], y: Sequence[bool], need: int) -> tuple[list[float], list[bool]] | Insufficient:
    """Return the aligned pairs a fitter may use, or why it may not.

    A pair with an invalid probability is dropped. A fit needs at least `need` pairs and both
    outcomes present.

    Args:
        X: Stated probabilities.
        y: Outcomes aligned with `X`; sequences of different lengths are not aligned, so
            nothing is usable.
        need: The fitter's floor.

    Returns:
        `(probabilities, outcomes)`, or `Insufficient`. Never raises.
    """
    if len(X) != len(y):
        return Insufficient(n=0, need=need, reason=f"X has {len(X)} items, y has {len(y)}")
    xs: list[float] = []
    ys: list[bool] = []
    for x, o in zip(X, y, strict=True):
        if valid_p(x):
            xs.append(float(x))
            ys.append(bool(o))
    if len(xs) < need:
        return Insufficient(n=len(xs), need=need, reason=f"{len(xs)} usable pairs < {need}")
    if all(ys) or not any(ys):
        return Insufficient(n=len(xs), need=need, reason="one outcome only")
    return xs, ys


def proba_rows(ps: Sequence[float]) -> list[tuple[float, float]]:
    """Return `(1 - p, p)` per probability: the two-column `predict_proba` layout."""
    return [(1.0 - p, p) for p in ps]


@dataclass(frozen=True)
class PiecewiseLinear:
    """A monotone non-decreasing calibration map held as knots.

    Between knots the map interpolates linearly; outside them it holds the end value. It is the
    fitted map, not a fitter: `IsotonicCalibrator.fit` returns one in `map_`.

    Attributes:
        xs: Knot inputs, strictly increasing, in [0, 1].
        ys: Knot outputs, non-decreasing, in [0, 1].
        n: The pairs it was fitted on (0 when unknown).

    Raises:
        ValueError: At construction, for no knots, unequal lengths, inputs not strictly
            increasing, outputs decreasing, or a knot outside [0, 1].
    """

    xs: tuple[float, ...]
    ys: tuple[float, ...]
    n: int = 0

    def __post_init__(self) -> None:
        if not self.xs or len(self.xs) != len(self.ys):
            raise ValueError("PiecewiseLinear needs equal, non-empty xs and ys")
        if not all(valid_p(v) for v in (*self.xs, *self.ys)):
            raise ValueError("PiecewiseLinear knots must be probabilities in [0, 1]")
        if any(b <= a for a, b in zip(self.xs, self.xs[1:], strict=False)):
            raise ValueError("PiecewiseLinear.xs must be strictly increasing")
        if any(b < a for a, b in zip(self.ys, self.ys[1:], strict=False)):
            raise ValueError("PiecewiseLinear.ys must be non-decreasing")
        # Knots are stored as float, so a call and `as_dict` never mix numeric types.
        object.__setattr__(self, "xs", tuple(float(x) for x in self.xs))
        object.__setattr__(self, "ys", tuple(float(y) for y in self.ys))

    def __call__(self, p: float) -> float:
        """Return the calibrated probability for `p`; an invalid `p` maps to 0.5, the neutral answer.

        Never raises.
        """
        if not valid_p(p):
            return 0.5
        p = float(p)
        i = bisect.bisect_right(self.xs, p)
        if i == 0:
            return self.ys[0]
        if i == len(self.xs):
            return self.ys[-1]
        x0, x1, y0, y1 = self.xs[i - 1], self.xs[i], self.ys[i - 1], self.ys[i]
        return y0 + (y1 - y0) * (p - x0) / (x1 - x0)

    def predict(self, X: Sequence[float]) -> list[float]:
        """Return the calibrated probability per stated probability. Never raises."""
        return [self(p) for p in X]

    def predict_proba(self, X: Sequence[float]) -> list[tuple[float, float]]:
        """Return `(P(no), P(yes))` per stated probability. Never raises."""
        return proba_rows(self.predict(X))

    def as_dict(self) -> dict[str, Any]:
        """Return the map as a JSON-ready dict; `from_dict` reads it back."""
        return {"xs": list(self.xs), "ys": list(self.ys), "n": self.n}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PiecewiseLinear:
        """Build a map from `as_dict` output.

        Raises:
            ValueError: As the constructor, or for a missing key.
        """
        try:
            return cls(
                xs=tuple(float(x) for x in data["xs"]), ys=tuple(float(y) for y in data["ys"]), n=int(data.get("n", 0))
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"not a PiecewiseLinear dict: {exc}") from exc


__all__ = ["Calibrator", "Insufficient", "PiecewiseLinear", "proba_rows", "usable"]
