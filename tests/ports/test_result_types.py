"""The result contract's value types and the failure taxonomy, with no adapter.

Result-contract acceptance test 1 (Usage arithmetic) and failures acceptance tests 2 and 9
(`OUTCOME_FOR` total, `RETRYABLE` a subset; seven outcomes, an exhaustive match type-checks).
"""

from __future__ import annotations

import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import get_args

import pytest

from llm_ports import ports
from llm_ports.ports import (
    FAILURE_KINDS,
    OUTCOME_FOR,
    RETRYABLE,
    Answered,
    Cost,
    Embedded,
    Failure,
    Finish,
    Generated,
    Outcome,
    Reranked,
    Usage,
    failure,
)
from llm_ports.ports.outcomes import REASON_MAX

# ── Usage ────────────────────────────────────────────────────────────────────────────────────────


def test_none_plus_a_count_is_the_count_and_none_plus_none_stays_none() -> None:
    got = Usage(input=None, output=3, requests=1) + Usage(input=None, output=None, requests=1)
    assert got.input is None and got.output == 3 and got.requests == 2
    zeros = Usage(input=0, output=0, requests=1) + Usage(input=0, output=0, requests=1)
    assert zeros.input == 0 and zeros.output == 0, "a reported 0 stays 0"


def test_every_bucket_and_detail_sums() -> None:
    a = Usage(input=10, output=5, cache_read=4, cache_write=1, reasoning=2, requests=1, details={"x": 1})
    b = Usage(input=20, output=7, cache_read=None, cache_write=3, reasoning=None, requests=2, details={"x": 2, "y": 5})
    got = a + b
    assert (got.input, got.output, got.cache_read, got.cache_write, got.reasoning) == (30, 12, 4, 4, 2)
    assert got.requests == 3 and got.details == {"x": 3, "y": 5}


def test_complete_is_false_when_any_request_was_unmeasured_and_true_otherwise() -> None:
    measured = Usage(input=10, output=5, requests=1)
    assert measured.complete
    assert not (measured + Usage(requests=1, unmeasured=1)).complete
    assert not Usage().complete, "no request is not a measurement"
    assert not Usage(input=10, output=None, requests=1).complete


def test_the_1x_mapping_keeps_unknown_as_none() -> None:
    assert Usage(input=10, output=None, requests=1).as_mapping() == {"input": 10, "output": None}


# ── Failure ──────────────────────────────────────────────────────────────────────────────────────


def test_outcome_for_is_total_and_retryable_is_a_subset() -> None:
    assert set(OUTCOME_FOR) == set(FAILURE_KINDS)
    assert RETRYABLE <= set(FAILURE_KINDS)
    assert set(OUTCOME_FOR.values()) <= set(get_args(Outcome)) - {"ok"}


def test_every_outcome_but_ok_has_a_kind() -> None:
    assert set(OUTCOME_FOR.values()) == set(get_args(Outcome)) - {"ok"}


def test_outcome_has_exactly_seven_members_and_refused_is_new() -> None:
    assert get_args(Outcome) == ("ok", "gated", "unavailable", "request_fault", "truncated", "invalid", "refused")


def test_a_failure_derives_its_outcome_and_retryability_from_its_kind() -> None:
    f = failure("rate_limited", retry_after_s=2.5, status=429)
    assert (f.outcome, f.retryable, f.retry_after_s, f.reason) == ("unavailable", True, 2.5, "rate_limited")
    bug = failure("internal", reason="KeyError: 'x'")
    assert (bug.outcome, bug.retryable) == ("request_fault", False), "a bug is not an outage"


@pytest.mark.parametrize("kind", ["quota", "auth", "not_found", "internal", "bad_request", "refused"])
def test_a_kind_no_wait_clears_is_not_retryable(kind: ports.FailureKind) -> None:
    assert not failure(kind).retryable


def test_an_unknown_kind_is_refused_and_a_long_reason_is_capped() -> None:
    with pytest.raises(ValueError, match="kind"):
        Failure(kind="nope")  # type: ignore[arg-type]
    assert len(failure("server", reason="x" * 1000).reason) == REASON_MAX


TYPING = Path(__file__).parents[1] / "typing" / "outcome_match.py"


def _mypy(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "--no-incremental", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_match_over_outcome_type_checks_and_a_missing_arm_does_not(tmp_path: Path) -> None:
    # The `types` job checks tests/typing on every PR; this run proves that check can fail.
    pytest.importorskip("mypy")
    assert _mypy(TYPING).returncode == 0
    missing = tmp_path / "missing_arm.py"
    missing.write_text(TYPING.read_text().replace(' | "refused"', ""))
    run = _mypy(missing)
    assert run.returncode != 0 and "assert_never" in run.stdout + run.stderr


# ── the result types ─────────────────────────────────────────────────────────────────────────────


def test_generated_derives_its_1x_fields_from_the_new_ones() -> None:
    res = Generated(
        value=None,
        outcome="truncated",
        model="m",
        usage_v2=Usage(input=10, output=None, requests=1),
        finish=Finish(unified="length", raw="length"),
        failure=failure("truncated", reason="cut"),
    )
    assert res.usage == {"input": 10, "output": None}
    assert res.finish_reason == "length" and res.reason == "cut"


def test_a_1x_field_a_caller_set_is_kept() -> None:
    res = Generated(value=None, outcome="unavailable", model="m", reason="mine", failure=failure("server"))
    assert res.reason == "mine"


def test_embedded_and_reranked_derive_reason_from_failure() -> None:
    assert Embedded(vectors=(), outcome="unavailable", failure=failure("timeout")).reason == "timeout"
    assert Reranked(scores=(), outcome="refused", failure=failure("refused", reason="no")).reason == "no"
    assert Embedded(vectors=(), outcome="ok").reason is None


def test_cost_and_answered_keep_unknown_as_none() -> None:
    assert Cost(usd=None, source="unknown", reason="unknown_model").usd is None
    assert Cost(usd=Decimal("0.0123"), source="gateway").usd == Decimal("0.0123")
    who = Answered(requested="alias")
    assert who.model is None and who.deployment is None
