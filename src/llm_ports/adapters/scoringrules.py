"""Proper scores beyond the binary case, on scoringrules (extra `[scoringrules]`).

`forecast.score` scores a binary forecast with the standard library. A range or a level is not
binary: a stated interval is scored by the interval score (width plus a penalty for each miss,
scaled by `2 / alpha`), and a stated distribution by the CRPS. Each function scores many
forecasts at once and returns one score per forecast; lower is better.

Invariants:
    - Never raises: inputs of unequal length, empty, or that scoringrules rejects give None.
    - A returned list is aligned with the inputs, and every score in it is finite.

Rationale: ADR-0013.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from llm_ports._extras import extra

with extra("scoringrules"):
    import numpy as np
    import scoringrules as sr


def _aligned(*cols: Sequence[float]) -> bool:
    return bool(cols[0]) and all(len(c) == len(cols[0]) for c in cols)


def _finite(values: object) -> list[float] | None:
    out = [float(v) for v in np.atleast_1d(np.asarray(values, dtype=float))]
    return out if all(math.isfinite(v) for v in out) else None


def brier_scores(p: Sequence[float], outcomes: Sequence[bool]) -> list[float] | None:
    """Return the Brier score of each binary forecast (a cross-check of `forecast.score.brier`).

    Returns:
        One score per forecast, or None for unusable input. Never raises.
    """
    if not _aligned(p, [float(y) for y in outcomes]):
        return None
    try:
        return _finite(sr.brier_score(np.array([int(y) for y in outcomes]), np.array(p, dtype=float)))
    except (ValueError, TypeError):
        return None


def interval_scores(
    observed: Sequence[float], lower: Sequence[float], upper: Sequence[float], alpha: float
) -> list[float] | None:
    """Return the interval score of each stated `[lower, upper]` range at level `1 - alpha`.

    Args:
        observed: What the world did.
        lower: Each range's lower end.
        upper: Each range's upper end.
        alpha: The range's stated miss rate, in (0, 1) (0.32 for a one-sigma range).

    Returns:
        One score per range, or None for unusable input or an alpha outside (0, 1). Never raises.
    """
    if not _aligned(observed, lower, upper) or not 0.0 < alpha < 1.0:
        return None
    try:
        return _finite(
            sr.interval_score(
                np.array(observed, dtype=float), np.array(lower, dtype=float), np.array(upper, dtype=float), alpha
            )
        )
    except (ValueError, TypeError):
        return None


def crps_normal(observed: Sequence[float], mu: Sequence[float], sigma: Sequence[float]) -> list[float] | None:
    """Return the CRPS of each stated normal distribution `N(mu, sigma)` at what was observed.

    Returns:
        One score per forecast, or None for unusable input or a sigma <= 0. Never raises.
    """
    if not _aligned(observed, mu, sigma) or any(s <= 0 for s in sigma):
        return None
    try:
        return _finite(
            sr.crps_normal(np.array(observed, dtype=float), np.array(mu, dtype=float), np.array(sigma, dtype=float))
        )
    except (ValueError, TypeError):
        return None


__all__ = ["brier_scores", "crps_normal", "interval_scores"]
