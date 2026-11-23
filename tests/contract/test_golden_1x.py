"""The 1.x result fields are unchanged from 1.22.0 (result-contract acceptance test 9).

The fixture was written by `golden_1x_scenarios` run against the 1.22.0 tree; this test runs the
same scenarios against this tree and compares every 1.x field. The two 2.0 behaviour fixes are
not scenarios (they are the changes, tested in `tests/adapters/pydantic_ai/test_result_contract.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pydantic_ai")

from tests.contract.golden_1x_scenarios import run_all

GOLDEN = Path(__file__).parents[1] / "fixtures" / "golden" / "result_1x_v1_22_0.json"

#: `reason` names the exception class (ADR-0001), so it follows pydantic-ai's own classification.
#: A scenario here may read either value: the golden's (the floor pydantic-ai, `test-lowest`) or the
#: newer one. Every other field, the outcome included, must still match exactly.
#: wire_413 -- pydantic-ai 2.71 raises HTTP 413 as `ModelHTTPError.for_category(ModelRequestRejectedError)`
#: (its docstring: "an HTTP 400, 413 or 422"), where 2.68 raised a plain `ModelHTTPError`.
UPSTREAM_REASON = {"wire_413": "ModelRequestRejectedError: status 413"}


async def test_every_1x_field_matches_the_1_22_0_golden() -> None:
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    got = json.loads(json.dumps(await run_all(), sort_keys=True))
    assert sorted(got) == sorted(want)
    for name in want:
        if got[name].get("reason") == UPSTREAM_REASON.get(name):
            got[name] = {**got[name], "reason": want[name]["reason"]}
        assert got[name] == want[name], name


def test_an_upstream_reason_is_allowed_only_where_the_golden_is_another_value() -> None:
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for name, reason in UPSTREAM_REASON.items():
        assert name in want and want[name]["reason"] != reason, name


def test_the_golden_covers_every_1x_outcome() -> None:
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    seen = {v["outcome"] for v in want.values()}
    assert seen == {"ok", "gated", "unavailable", "request_fault", "truncated", "invalid"}
