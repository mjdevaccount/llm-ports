"""retrieval_metrics: hit@k / MRR over an ANY-OF relevant set, agree@k Jaccard, and the parity gate's
rounded tolerance. Moved from trade-data (scripts/rag_eval.py and usecases/rag_email_parity.py each
computed hit@k; one implementation now, v0.9.0)."""

from __future__ import annotations

import pytest

from llm_ports.retrieval_metrics import agree_at_k, clears, first_rank, hit_at_k, mean, reciprocal_rank

RANKED = ["d1", "d2", "d3", "d4", "d5"]


def test_first_rank_is_one_based_and_any_of() -> None:
    assert first_rank({"d3"}, RANKED) == 3
    assert first_rank({"d4", "d2"}, RANKED) == 2, "the FIRST relevant id, whichever alternate it is"
    assert first_rank({"zz"}, RANKED) is None and first_rank({"d1"}, []) is None


def test_hit_at_k_is_inclusive_at_k() -> None:
    """MUST FIRE: an off-by-one (`rank < k`) fails the k == rank case."""
    assert hit_at_k({"d3"}, RANKED, 3) is True
    assert hit_at_k({"d3"}, RANKED, 2) is False
    assert hit_at_k({"zz"}, RANKED, 5) is False
    assert hit_at_k({"d5", "d1"}, RANKED, 1) is True


def test_reciprocal_rank() -> None:
    assert reciprocal_rank({"d1"}, RANKED) == 1.0
    assert reciprocal_rank({"d4"}, RANKED) == 0.25
    assert reciprocal_rank({"zz"}, RANKED) == 0.0


def test_agree_at_k_is_the_jaccard_of_the_top_k_sets() -> None:
    assert agree_at_k(["a", "b", "c"], ["c", "b", "a"], 3) == 1.0  # order inside top-k is irrelevant
    assert agree_at_k(["a", "b", "x"], ["a", "b", "y"], 3) == 0.5  # 2 shared / 4 distinct
    assert agree_at_k(["a", "b", "x"], ["a", "b", "y"], 2) == 1.0  # only the top k count
    assert agree_at_k(["a"], ["b"], 1) == 0.0
    assert agree_at_k([], [], 5) == 0.0, "two empty rankings agree on nothing"
    assert agree_at_k(["a", "a", "b"], ["a", "b"], 3) == 1.0  # sets, not lists


def test_mean_and_its_refusal_of_no_queries() -> None:
    assert mean([1.0, 0.0, 0.5]) == 0.5
    assert mean(x for x in (True, False)) == 0.5
    with pytest.raises(ValueError, match="zero queries"):
        mean([])


def test_clears_rounds_the_floor_so_a_reported_metric_compares_as_written() -> None:
    """MUST FIRE: unrounded, 0.5006 - 0.05 is 0.45060000000000006 in float, so a new metric of
    exactly 0.4506 (4dp, as reported) would fail a gate it meets."""
    assert 0.4506 < 0.5006 - 0.05, "the float trap this guards"
    assert clears(0.4506, 0.5006, tolerance=0.05) is True
    assert clears(0.4505, 0.5006, tolerance=0.05) is False
    assert clears(0.85, 0.9, tolerance=0.05) is True and clears(0.8499, 0.9, tolerance=0.05) is False
    assert clears(0.95, 0.9, tolerance=0.0) is True and clears(0.89, 0.9, tolerance=0.0) is False
