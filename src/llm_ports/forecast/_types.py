"""The forecast values: what a voice said would happen, and what did.

Invariants:
    - Standard library only: this package imports nothing else from `llm_ports` and no vendor.
    - Every value is frozen.

Rationale: ADR-0013.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from datetime import datetime


def finite_real(v: object) -> bool:
    """Return True for a finite `numbers.Real` that is not a bool (a weight, a strength). Never raises."""
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v)


def valid_p(p: float) -> bool:
    """Return True when `p` is a finite real probability in [0, 1] (a bool is not a probability).

    Any `numbers.Real` counts (`int`, `float`, NumPy floating scalars, `Fraction`); `Decimal`
    is not a `numbers.Real` and is refused. A caller that accepts `p` reads it through `float`.
    Never raises.
    """
    return finite_real(p) and 0.0 <= float(p) <= 1.0


@dataclass(frozen=True)
class Forecast:
    """One gradeable statement a voice made about a future event.

    A forecast is a record of what was said, not an input to a decision: its probability is the
    voice's own (`p`), its verbal confidence is kept as written (`hedge`), and a calibrator or a
    `HedgeMap` turns either into a probability the voice has earned.

    Attributes:
        producer: Who asserted it (an author, a model role, an assembler).
        subject: What it is about (an instrument, an event key).
        kind: What shape of claim it is (`direction`, `level`, `range`, `event`, ...).
        asserted_at: When it was asserted; only text available then may back it.
        resolves_at: When the world settles it; never before `asserted_at`.
        evidence: The span the forecast was read from, verbatim.
        p: The probability the voice stated, or None when it stated only words.
        hedge: The verbal confidence as written (`likely`, `could`), or None.
        extracted_by: What transcribed it (a prompt pack and its sha), or None when the producer
            wrote it directly.

    Raises:
        ValueError: At construction, for a `p` outside [0, 1] or not finite, a `resolves_at`
            before `asserted_at` (or one naive and one aware), an empty `evidence`, `producer`, `subject` or `kind`, or a
            forecast with neither `p` nor `hedge`.
    """

    producer: str
    subject: str
    kind: str
    asserted_at: datetime
    resolves_at: datetime
    evidence: str
    p: float | None = None
    hedge: str | None = None
    extracted_by: str | None = None

    def __post_init__(self) -> None:
        if not (self.producer and self.subject and self.kind):
            raise ValueError("Forecast needs a producer, a subject and a kind")
        if not self.evidence.strip():
            raise ValueError("Forecast needs a non-empty evidence span")
        try:
            backwards = self.resolves_at < self.asserted_at
        except TypeError as exc:
            raise ValueError("Forecast.asserted_at and resolves_at must both be naive or both aware") from exc
        if backwards:
            raise ValueError("Forecast.resolves_at is before asserted_at")
        if self.p is None and not (self.hedge and self.hedge.strip()):
            raise ValueError("Forecast needs a probability or a hedge")
        if self.p is not None and not valid_p(self.p):
            raise ValueError(f"Forecast.p must be a finite probability in [0, 1], got {self.p!r}")


@dataclass(frozen=True)
class Resolved:
    """A forecast and what the world did.

    Attributes:
        forecast: The forecast.
        outcome: True when the forecast event happened.
        resolved_at: When the outcome was read.
    """

    forecast: Forecast
    outcome: bool
    resolved_at: datetime


__all__ = ["Forecast", "Resolved", "finite_real", "valid_p"]
