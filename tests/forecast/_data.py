"""Shared forecast test data."""

from __future__ import annotations

import random


def overconfident(n: int, seed: int = 3) -> tuple[list[float], list[bool]]:
    """A voice whose stated p overclaims: the truth is p**2 (0.68 stated hits about 0.46)."""
    rng = random.Random(seed)
    ps = [rng.random() for _ in range(n)]
    return ps, [rng.random() < p * p for p in ps]
