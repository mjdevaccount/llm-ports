"""Verbal confidence, learned per source: what "likely" from one voice has meant.

A voice that writes "likely" does not state a number, and two voices mean different things by
the same word. `HedgeMap` learns, from resolved forecasts, the probability each `(source, hedge)`
has earned. A source with few resolved uses of a word is shrunk toward the word's pooled rate
across every source (beta-binomial shrinkage), so one lucky call does not make "could" mean 0.9.

The prior for a word is `Beta(m * s, (1 - m) * s)`: `m` is the word's pooled hit rate, `s` its
strength in pseudo-observations. `s` is estimated by the method of moments from how much the
sources disagree beyond binomial noise (Kleinman's weighted estimator). It falls back to
`DEFAULT_STRENGTH` when it cannot be estimated (fewer than two sources, a pooled rate of 0 or 1,
or sources that each used the word once), and to `MAX_STRENGTH` (near-full pooling) when the
sources agree within noise.

Invariants:
    - Pure, standard library only.
    - Hedge text is compared normalised: case-folded, whitespace collapsed.
    - A word never seen resolved has no probability (None), never a guess.

Rationale: ADR-0013.
"""

from __future__ import annotations

import numbers
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from llm_ports.forecast._types import finite_real, valid_p

#: The prior strength used when it cannot be estimated (fewer than two sources use the word).
DEFAULT_STRENGTH = 10.0
#: The ceiling on an estimated strength: sources that agree within noise are pooled, near fully.
MAX_STRENGTH = 1000.0


def normalise(hedge: str) -> str:
    """Return `hedge` case-folded with whitespace collapsed, the key a `HedgeMap` compares."""
    return " ".join(hedge.casefold().split())


@dataclass(frozen=True)
class HedgeRow:
    """One `(source, hedge)` cell of a fitted map.

    Attributes:
        source: The voice.
        hedge: The normalised word or phrase.
        n: Resolved forecasts that used it.
        hits: How many happened.
        p: The shrunk probability it has earned.
        pooled: The word's pooled rate across every source (the prior mean).
    """

    source: str
    hedge: str
    n: int
    hits: int
    p: float
    pooled: float


def _tau2(cells: Mapping[str, tuple[int, int]], m: float) -> float | None:
    """Between-source variance of one word's hit rate, unclipped; None when not estimable."""
    used = [(k, n) for k, n in cells.values() if n > 0]
    if len(used) < 2 or m <= 0.0 or m >= 1.0:
        return None
    total = sum(n for _, n in used)
    # E[sum n_i (r_i - m)^2 / N] = tau^2 (1 - sum n_i^2 / N^2) + (J - 1) sigma^2 / N, with
    # sigma^2 = m(1 - m) - tau^2. Subtracting (J - 1) m(1 - m) / N leaves tau^2 times `scale`;
    # the prior's tau^2 = m(1 - m) / (s + 1) is then solved for s.
    observed = sum(n * (k / n - m) ** 2 for k, n in used) / total
    noise = (len(used) - 1) * m * (1 - m) / total
    scale = 1.0 - sum(n * n for _, n in used) / total**2 - (len(used) - 1) / total
    if scale <= 1e-9:  # every source used the word once: spread and noise are inseparable
        return None
    return (observed - noise) / scale


def _strength(cells: Mapping[str, tuple[int, int]], m: float, fallback: float) -> float:
    """Method-of-moments prior strength for one word from its per-source `(hits, n)` cells."""
    tau2 = _tau2(cells, m)
    if tau2 is None:
        return fallback
    if tau2 <= 0.0:
        return MAX_STRENGTH
    return min(max(m * (1 - m) / tau2 - 1.0, 1.0), MAX_STRENGTH)


def _count(v: object) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, bool) and int(v) >= 0


def _valid_strength(s: object) -> bool:
    return finite_real(s) and float(s) > 0  # type: ignore[arg-type]  # finite_real proved it Real


@dataclass(frozen=True)
class HedgeMap:
    """The probability each `(source, hedge)` has earned.

    Build with `HedgeMap.fit`; read with `p`.

    Attributes:
        cells: `(source, hedge) -> (hits, n)`.
        pooled: `hedge -> pooled hit rate` (Jeffreys-smoothed).
        strengths: `hedge -> prior strength` used for that word.

    Raises:
        ValueError: At construction, for a cell whose counts are not `0 <= hits <= n`, a pooled
            rate that is not in [0, 1], or a strength that is not finite and > 0.
    """

    cells: Mapping[tuple[str, str], tuple[int, int]]
    pooled: Mapping[str, float]
    strengths: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(_count(k) and _count(n) and k <= n for k, n in self.cells.values()):
            raise ValueError("HedgeMap.cells must hold integer (hits, n) with 0 <= hits <= n")
        if not all(valid_p(v) for v in self.pooled.values()):
            raise ValueError("HedgeMap.pooled rates must be probabilities in [0, 1]")
        if not all(_valid_strength(v) for v in self.strengths.values()):
            raise ValueError("HedgeMap.strengths must be finite and > 0")
        # Stored as plain int/float, so `p` and `as_dict` never mix numeric types.
        object.__setattr__(self, "cells", {key: (int(k), int(n)) for key, (k, n) in self.cells.items()})
        object.__setattr__(self, "pooled", {w: float(v) for w, v in self.pooled.items()})
        object.__setattr__(self, "strengths", {w: float(v) for w, v in self.strengths.items()})

    @classmethod
    def fit(cls, rows: Iterable[tuple[str, str, bool]], *, strength: float | None = None) -> HedgeMap:
        """Learn the map from resolved forecasts.

        Args:
            rows: `(source, hedge, outcome)`; a row whose hedge is empty or not a string is skipped.
            strength: A fixed prior strength for every word (any finite real > 0, stored as
                float); None (default) estimates it per word.

        Returns:
            The fitted map.

        Raises:
            ValueError: For a `strength` that is not finite and > 0.
        """
        if strength is not None and not _valid_strength(strength):
            raise ValueError(f"HedgeMap.fit strength must be finite and > 0, got {strength!r}")
        counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        for source, hedge, outcome in rows:
            if not isinstance(hedge, str):
                continue
            key = normalise(hedge)
            if not key:
                continue
            cell = counts[(source, key)]
            cell[0] += int(bool(outcome))
            cell[1] += 1
        by_word: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
        for (source, word), (k, n) in counts.items():
            by_word[word][source] = (k, n)
        pooled: dict[str, float] = {}
        strengths: dict[str, float] = {}
        for word, cells in by_word.items():
            k = sum(c[0] for c in cells.values())
            n = sum(c[1] for c in cells.values())
            pooled[word] = (k + 0.5) / (n + 1.0)
            strengths[word] = float(strength) if strength is not None else _strength(cells, k / n, DEFAULT_STRENGTH)
        return cls(cells={key: (c[0], c[1]) for key, c in counts.items()}, pooled=pooled, strengths=strengths)

    def p(self, source: str, hedge: str) -> float | None:
        """Return the probability `hedge` from `source` has earned.

        A source that never used the word resolved gets the word's pooled rate.

        Returns:
            The probability, or None for a word no source has used resolved (or a hedge that is
            not a string). Never raises.
        """
        if not isinstance(hedge, str):
            return None
        word = normalise(hedge)
        m = self.pooled.get(word)
        if m is None:
            return None
        k, n = self.cells.get((source, word), (0, 0))
        s = self.strengths.get(word, DEFAULT_STRENGTH)
        return (k + m * s) / (n + s)

    def rows(self) -> tuple[HedgeRow, ...]:
        """Return every fitted cell, by source then hedge, for a readout."""
        out = []
        for (source, word), (k, n) in sorted(self.cells.items()):
            p = self.p(source, word)
            assert p is not None  # every cell's word is pooled
            out.append(HedgeRow(source=source, hedge=word, n=n, hits=k, p=p, pooled=self.pooled[word]))
        return tuple(out)

    def as_dict(self) -> dict[str, Any]:
        """Return the map as a JSON-ready dict; `from_dict` reads it back."""
        return {
            "cells": [[s, w, k, n] for (s, w), (k, n) in sorted(self.cells.items())],
            "pooled": dict(sorted(self.pooled.items())),
            "strengths": dict(sorted(self.strengths.items())),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HedgeMap:
        """Build a map from `as_dict` output.

        Raises:
            ValueError: For a dict that is not `as_dict` output, or one the constructor rejects.
        """
        try:
            cells = {(str(s), str(w)): (int(k), int(n)) for s, w, k, n in data["cells"]}
            pooled = {str(w): float(v) for w, v in data["pooled"].items()}
            strengths = {str(w): float(v) for w, v in data.get("strengths", {}).items()}
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"not a HedgeMap dict: {exc}") from exc
        return cls(cells=cells, pooled=pooled, strengths=strengths)


__all__ = ["DEFAULT_STRENGTH", "MAX_STRENGTH", "HedgeMap", "HedgeRow", "normalise"]
