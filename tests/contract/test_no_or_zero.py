"""Unknown is never zero: no `x or 0` in `src/` without a stated reason.

Result-contract acceptance test 10. `count or 0` reads "the provider reported nothing" as "the call
used nothing", and an unpriced call then looks free. The check reads Python tokens, so a docstring
or a comment that names the pattern is not a hit; a line that means it carries
`# zero-ok: <why>`, and the reason is required.

Rationale: ADR-0019.
"""

from __future__ import annotations

import io
import itertools
import tokenize
from pathlib import Path

SRC = Path(__file__).parents[2] / "src" / "llm_ports"
MARKER = "# zero-ok:"
ZEROS = frozenset({"0", "0.0", "0.", "0x0"})


def or_zero_lines(source: str) -> list[int]:
    """Return the line numbers where `or 0` / `or 0.0` appears in code without a reasoned waiver."""
    tokens = [t for t in tokenize.generate_tokens(io.StringIO(source).readline) if t.type != tokenize.NL]
    lines = source.splitlines()
    hits: list[int] = []
    for a, b in itertools.pairwise(tokens):
        if a.type == tokenize.NAME and a.string == "or" and b.type == tokenize.NUMBER and b.string in ZEROS:
            line = lines[a.start[0] - 1]
            waiver = line.partition(MARKER)[2].strip() if MARKER in line else ""
            if not waiver:
                hits.append(a.start[0])
    return hits


def test_src_has_no_unwaived_or_zero() -> None:
    found = {
        str(p.relative_to(SRC)): hits
        for p in sorted(SRC.rglob("*.py"))
        if (hits := or_zero_lines(p.read_text(encoding="utf-8")))
    }
    assert found == {}, f"unknown read as 0; keep None or waive with {MARKER} <why>: {found}"


def test_the_check_fires_on_a_planted_line_and_needs_a_reason() -> None:
    assert or_zero_lines("tokens = usage.get('input') or 0\n") == [1]
    assert or_zero_lines("usd = cost or 0.0\n") == [1]
    assert or_zero_lines("n = got or 0  # zero-ok:\n") == [1], "a waiver without a reason is not a waiver"
    assert or_zero_lines("n = got or 0  # zero-ok: an absent unit is 0 of it\n") == []


def test_the_check_ignores_prose_and_other_defaults() -> None:
    assert or_zero_lines('"""Replaces `int(v or 0)`."""\n') == []
    assert or_zero_lines("x = y  # never `or 0`\n") == []
    assert or_zero_lines("n = got or 1\nm = got or None\n") == []
