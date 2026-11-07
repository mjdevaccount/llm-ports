"""The forecast values: what a voice said would happen, and what did.

Field names follow forecasting-platform vocabulary (Metaculus, Good Judgment), schema.org's
`Claim` / `ClaimReview`, and W3C PROV for provenance. Where they name the same thing differently:

| field | Metaculus / GJ | schema.org | W3C PROV |
|---|---|---|---|
| `Forecast.author` | forecaster | `Claim.author` | `wasAttributedTo` |
| `Forecast.subject` | the question's subject | `Claim.about` | |
| `Forecast.kind` | `question.type` | | |
| `Forecast.forecast_time` | forecast time | `Claim.datePublished` | |
| `Forecast.resolve_time` | `scheduled_resolve_time` | | |
| `Forecast.text` | | `Claim.text` | |
| `Forecast.resolution_criteria` | `resolution_criteria` | | |
| `Forecast.probability` | probability | | |
| `Forecast.was_generated_by` | | | `wasGeneratedBy` |
| `Forecast.used` | | | `used` |
| `Resolved.claim_reviewed` | | `ClaimReview.claimReviewed` | |
| `Resolved.resolution` | `resolution` | `ClaimReview.reviewRating` | |
| `Resolved.actual_resolve_time` | `actual_resolve_time` | `ClaimReview.datePublished` | |

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
from typing import Literal, get_args

#: How a forecast resolved: it happened, it did not, or it can no longer be graded.
Resolution = Literal["yes", "no", "annulled"]


def finite_real(v: object) -> bool:
    """Return True for a finite `numbers.Real` that is not a bool (a weight, a strength). Never raises."""
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        return False
    try:
        return math.isfinite(v)
    except OverflowError:  # an int or Fraction past float range is not a usable number here
        return False


def valid_p(p: object) -> bool:
    """Return True when `p` is a finite real probability in [0, 1] (a bool is not a probability).

    Any `numbers.Real` counts (`int`, `float`, NumPy floating scalars, `Fraction`); `Decimal`
    is not a `numbers.Real` and is refused. A caller that accepts `p` reads it through `float`.
    Never raises.
    """
    return finite_real(p) and 0.0 <= float(p) <= 1.0  # type: ignore[arg-type]  # finite_real proved it Real


@dataclass(frozen=True)
class Forecast:
    """One gradeable statement a voice made about a future event.

    A forecast is a record of what was said, not an input to a decision: its probability is the
    voice's own (`probability`), its verbal confidence is kept as written (`hedge`), and a
    calibrator or a `HedgeMap` turns either into a probability the voice has earned.

    Attributes:
        author: Who asserted it (an author, a model role, an assembler).
        subject: What it is about (an instrument, an event key).
        kind: What shape of claim it is (`direction`, `level`, `range`, `event`, ...).
        forecast_time: When it was asserted; only text available then may back it.
        resolve_time: When the world settles it; never before `forecast_time`.
        text: The span the forecast was read from, verbatim.
        probability: The probability the voice stated, or None when it stated only words.
        hedge: The verbal confidence as written (`likely`, `could`), or None.
        resolution_criteria: How the outcome is read, or None when `kind` implies it.
        was_generated_by: What transcribed it (a prompt pack and its sha), or None when the
            author wrote it directly.
        used: What the transcription read (source document ids); empty when none are recorded.

    Raises:
        ValueError: At construction, for a `probability` outside [0, 1] or not finite, a
            `resolve_time` before `forecast_time` (or one naive and one aware), an empty `text`,
            `author`, `subject` or `kind`, or a forecast with neither `probability` nor `hedge`.
    """

    author: str
    subject: str
    kind: str
    forecast_time: datetime
    resolve_time: datetime
    text: str
    probability: float | None = None
    hedge: str | None = None
    resolution_criteria: str | None = None
    was_generated_by: str | None = None
    used: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not (self.author and self.subject and self.kind):
            raise ValueError("Forecast needs an author, a subject and a kind")
        if not self.text.strip():
            raise ValueError("Forecast needs a non-empty text span")
        try:
            backwards = self.resolve_time < self.forecast_time
        except TypeError as exc:
            raise ValueError("Forecast.forecast_time and resolve_time must both be naive or both aware") from exc
        if backwards:
            raise ValueError("Forecast.resolve_time is before forecast_time")
        if self.probability is None and not (self.hedge and self.hedge.strip()):
            raise ValueError("Forecast needs a probability or a hedge")
        if self.probability is not None and not valid_p(self.probability):
            raise ValueError(f"Forecast.probability must be a finite probability in [0, 1], got {self.probability!r}")
        object.__setattr__(self, "used", tuple(self.used))


@dataclass(frozen=True)
class Resolved:
    """A forecast and what the world did.

    Attributes:
        claim_reviewed: The forecast.
        resolution: `yes` when the forecast event happened, `no` when it did not, `annulled`
            when it can no longer be graded (the question was voided).
        actual_resolve_time: When the outcome was read.

    Raises:
        ValueError: At construction, for a `resolution` that is not `yes`, `no` or `annulled`.
    """

    claim_reviewed: Forecast
    resolution: Resolution
    actual_resolve_time: datetime

    def __post_init__(self) -> None:
        if self.resolution not in get_args(Resolution):
            raise ValueError(f"Resolved.resolution must be yes, no or annulled, got {self.resolution!r}")

    @property
    def outcome(self) -> bool | None:
        """True for `yes`, False for `no`, None for `annulled` (not gradeable)."""
        return None if self.resolution == "annulled" else self.resolution == "yes"


__all__ = ["Forecast", "Resolution", "Resolved", "finite_real", "valid_p"]
