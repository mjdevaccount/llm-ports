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


async def test_every_1x_field_matches_the_1_22_0_golden() -> None:
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    got = json.loads(json.dumps(await run_all(), sort_keys=True))
    assert sorted(got) == sorted(want)
    for name in want:
        assert got[name] == want[name], name


def test_the_golden_covers_every_1x_outcome() -> None:
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    seen = {v["outcome"] for v in want.values()}
    assert seen == {"ok", "gated", "unavailable", "request_fault", "truncated", "invalid"}
