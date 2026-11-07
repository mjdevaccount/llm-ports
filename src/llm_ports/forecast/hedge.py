"""Verbal confidence, learned per source: what "likely" from one voice has meant.

A voice that writes "likely" does not state a number, and two voices mean different things by
the same word. `HedgeMap` learns, from resolved forecasts, the probability each `(source, hedge)`
has earned. A source with few resolved uses of a word is shrunk toward the word's pooled rate
across every source (beta-binomial shrinkage), so one lucky call does not make "could" mean 0.9.

The prior for a word is `Beta(m * s, (1 - m) * s)`: `m` is the word's pooled hit rate, `s` its
strength in pseudo-observations. `s` is estimated by the method of moments from how much the
sources disagree beyond binomial noise; with fewer than two sources, or no excess disagreement,
it falls back to `strength` (or full pooling when the sources agree).

Invariants:
    - Pure, standard library only.
    - Hedge text is compared normalised: case-folded, whitespace collapsed.
    - A word never seen resolved has no probability (None), never a guess.

Rationale: ADR-0013.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

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


def _strength(cells: Mapping[str, tuple[int, int]], m: float, fallback: float) -> float:
    """Method-of-moments prior strength for one word from its per-source `(hits, n)` cells."""
    used = [(k, n) for k, n in cells.values() if n > 0]
    if len(used) < 2 or m <= 0.0 or m >= 1.0:
        return fallback
    total = sum(n for _, n in used)
    # n-weighted spread of the source rates, less what binomial noise alone would give: the
    # between-source variance tau^2 = m(1 - m) / (s + 1), solved for s.
    observed = sum(n * (k / n - m) ** 2 for k, n in used) / total
    noise = (len(used) - 1) * m * (1 - m) / total
    excess = observed - noise
    if excess <= 0.0:
        return MAX_STRENGTH
    return min(max(m * (1 - m) / excess - 1.0, 1.0), MAX_STRENGTH)


@dataclass(frozen=True)
class HedgeMap:
    """The probability each `(source, hedge)` has earned.

    Build with `HedgeMap.fit`; read with `p`.

    Attributes:
        cells: `(source, hedge) -> (hits, n)`.
        pooled: `hedge -> pooled hit rate` (Jeffreys-smoothed).
        strengths: `hedge -> prior strength` used for that word.
    """

    cells: Mapping[tuple[str, str], tuple[int, int]]
    pooled: Mapping[str, float]
    strengths: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, rows: Iterable[tuple[str, str, bool]], *, strength: float | None = None) -> HedgeMap:
        """Learn the map from resolved forecasts.

        Args:
            rows: `(source, hedge, outcome)`; a row with an empty hedge is skipped.
            strength: A fixed prior strength for every word; None (default) estimates it per word.

        Returns:
            The fitted map. Never raises.
        """
        counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        for source, hedge, outcome in rows:
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
            strengths[word] = strength if strength is not None else _strength(cells, k / n, DEFAULT_STRENGTH)
        return cls(cells={key: (c[0], c[1]) for key, c in counts.items()}, pooled=pooled, strengths=strengths)

    def p(self, source: str, hedge: str) -> float | None:
        """Return the probability `hedge` from `source` has earned.

        A source that never used the word resolved gets the word's pooled rate.

        Returns:
            The probability, or None for a word no source has used resolved. Never raises.
        """
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
            ValueError: For a dict that is not `as_dict` output.
        """
        try:
            cells = {(str(s), str(w)): (int(k), int(n)) for s, w, k, n in data["cells"]}
            pooled = {str(w): float(v) for w, v in data["pooled"].items()}
            strengths = {str(w): float(v) for w, v in data.get("strengths", {}).items()}
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"not a HedgeMap dict: {exc}") from exc
        return cls(cells=cells, pooled=pooled, strengths=strengths)


__all__ = ["DEFAULT_STRENGTH", "MAX_STRENGTH", "HedgeMap", "HedgeRow", "normalise"]
