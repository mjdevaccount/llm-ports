"""Score a ranked retrieval against what was relevant, and compare two runs.

Example:
    rank = first_rank({"doc-7", "doc-9"}, ranked)  # 1-based; None means never retrieved
    hit_at_k(relevant, ranked, 8), reciprocal_rank(relevant, ranked)
    agree_at_k(old_ranked, new_ranked, 10)  # Jaccard of the two top-k sets
    clears(new_hit10, old_hit10, tolerance=0.05)  # a parity gate: new >= old - tolerance

Pure. `relevant` is an any-of set: one relevant id in the top k is a hit, and the reciprocal rank
is of the first relevant id. Averages are the caller's (`mean`), so a per-query row stays
inspectable.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence


def first_rank(relevant: Collection[str], ranked: Sequence[str]) -> int | None:
    """Return the 1-based rank of the first relevant id in `ranked`; None when none was retrieved."""
    return next((i + 1 for i, d in enumerate(ranked) if d in relevant), None)


def hit_at_k(relevant: Collection[str], ranked: Sequence[str], k: int) -> bool:
    """Return whether any relevant id is in the top `k`."""
    rank = first_rank(relevant, ranked)
    return rank is not None and rank <= k


def reciprocal_rank(relevant: Collection[str], ranked: Sequence[str]) -> float:
    """Return 1 / rank of the first relevant id; 0.0 when none was retrieved (MRR's per-query term)."""
    rank = first_rank(relevant, ranked)
    return 1.0 / rank if rank else 0.0


def agree_at_k(a: Sequence[str], b: Sequence[str], k: int) -> float:
    """Return the Jaccard similarity of two rankings' top-`k` id sets.

    Two empty top-`k` sets agree on nothing: 0.0.
    """
    sa, sb = set(a[:k]), set(b[:k])
    return len(sa & sb) / (len(sa | sb) or 1)


def mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean.

    Raises:
        ValueError: On no values; a metric over zero queries is undefined, not 0.
    """
    vals = list(values)
    if not vals:
        raise ValueError("mean of no values: a metric over zero queries is undefined")
    return sum(vals) / len(vals)


def clears(new: float, baseline: float, *, tolerance: float, places: int = 4) -> bool:
    """Return whether `new` clears the parity gate `new >= baseline - tolerance`.

    The floor is rounded to `places`, so a metric reported at that precision compares as written
    (0.9 - 0.05 is not 0.85 in float).
    """
    return new >= round(baseline - tolerance, places)


__all__ = ["agree_at_k", "clears", "first_rank", "hit_at_k", "mean", "reciprocal_rank"]
