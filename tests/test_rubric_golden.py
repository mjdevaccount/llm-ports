"""`judge_sampling.rubric` is pinned byte-for-byte to the v0.11.0 release.

The rubric is prompt text: any change to it moves every judge verdict. `golden/rubric_v0_11.json`
was captured by installing the released v0.11.0 wheel and calling `rubric` for every grounding mode,
web flag and role shape below. The number law now lives in `numbers.law_text`; the output must not
move.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from llm_ports import judge_sampling as js
from llm_ports import numbers
from tests.conftest import Role

GOLDEN = json.loads((Path(__file__).parent / "golden" / "rubric_v0_11.json").read_text(encoding="utf-8"))
ROLES = {
    "structured": Role("test.judged", tier="box", graders=("schema_complete", "numbers_from_inputs")),
    "prose": Role("test.prose", tier="box"),
}
MODES = ("graded", "slots", "filled", "none")
CASES = [(shape, mode, web) for shape in ROLES for mode in MODES for web in (False, True)]


def test_the_golden_file_covers_every_case() -> None:
    assert sorted(GOLDEN) == sorted(f"{s}|{m}|web={w}" for s, m, w in CASES)


@pytest.mark.parametrize(("shape", "mode", "web"), CASES)
def test_rubric_is_byte_identical_to_v0_11(shape: str, mode: str, web: bool) -> None:
    got = js.rubric(
        ROLES[shape],
        js.Seen(web=web),
        platform="ACME support platform",
        mode=mode,
        today=dt.date(2026, 10, 31),
        tz_label="UTC",
    )
    assert got == GOLDEN[f"{shape}|{mode}|web={web}"]


@pytest.mark.parametrize(("mode", "web"), [(m, w) for m in MODES for w in (False, True)])
def test_the_rubric_states_the_law_through_numbers_law_text(mode: str, web: bool) -> None:
    law = numbers.law_text(mode, web=web)
    assert law.startswith("2. ") and law.endswith("\n")
    assert law in GOLDEN[f"structured|{mode}|web={web}"]
