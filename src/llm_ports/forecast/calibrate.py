"""Calibration: map a voice's stated probability to the probability it has earned.

- `Calibrator`: the port. `fit(p, outcomes)` returns a `Calibrated` map, or `Insufficient` when
  there are too few resolved forecasts to fit one.
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
from typing import Any, Protocol, runtime_checkable

from llm_ports.forecast._types import valid_p


@runtime_checkable
class Calibrated(Protocol):
    """A fitted calibration map."""

    def __call__(self, p: float) -> float:
        """Return the calibrated probability for a stated `p`, in [0, 1]. Never raises."""
        ...


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
    """Fits a calibration map from resolved forecasts."""

    def fit(self, p: Sequence[float], outcomes: Sequence[bool]) -> Calibrated | Insufficient:
        """Fit a map from stated probabilities and what happened.

        Args:
            p: Stated probabilities.
            outcomes: Whether each event happened, aligned with `p`.

        Returns:
            The fitted map, or `Insufficient`. Never raises.
        """
        ...


def usable(p: Sequence[float], outcomes: Sequence[bool], need: int) -> tuple[list[float], list[bool]] | Insufficient:
    """Return the aligned pairs a fitter may use, or why it may not.

    A pair with an invalid probability is dropped. A fit needs at least `need` pairs and both
    outcomes present.

    Args:
        p: Stated probabilities.
        outcomes: Outcomes aligned with `p`; extra items on either side are ignored.
        need: The fitter's floor.

    Returns:
        `(probabilities, outcomes)`, or `Insufficient`. Never raises.
    """
    xs: list[float] = []
    ys: list[bool] = []
    for x, y in zip(p, outcomes, strict=False):
        if valid_p(x):
            xs.append(float(x))
            ys.append(bool(y))
    if len(xs) < need:
        return Insufficient(n=len(xs), need=need, reason=f"{len(xs)} usable pairs < {need}")
    if all(ys) or not any(ys):
        return Insufficient(n=len(xs), need=need, reason="one outcome only")
    return xs, ys


@dataclass(frozen=True)
class PiecewiseLinear:
    """A monotone non-decreasing calibration map held as knots.

    Between knots the map interpolates linearly; outside them it holds the end value.

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

    def __call__(self, p: float) -> float:
        """Return the calibrated probability for `p`; an invalid `p` maps to the middle knot's output.

        Never raises.
        """
        if not valid_p(p):
            return self.ys[len(self.ys) // 2]
        i = bisect.bisect_right(self.xs, p)
        if i == 0:
            return self.ys[0]
        if i == len(self.xs):
            return self.ys[-1]
        x0, x1, y0, y1 = self.xs[i - 1], self.xs[i], self.ys[i - 1], self.ys[i]
        return y0 + (y1 - y0) * (p - x0) / (x1 - x0)

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


__all__ = ["Calibrated", "Calibrator", "Insufficient", "PiecewiseLinear", "usable"]
