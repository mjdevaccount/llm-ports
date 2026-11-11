"""Type-checked by the `types` job (mypy `files`): code that matches on `Outcome` keeps compiling.

Failures acceptance test 9: a match with a default arm, and an exhaustive match over all seven
outcomes, type-check under `mypy --strict`.
"""

from __future__ import annotations

from typing import assert_never

from llm_ports.ports import Outcome


def label(o: Outcome) -> str:
    match o:
        case "ok" | "gated":
            return "fine"
        case _:
            return "other"


def exhaustive(o: Outcome) -> str:
    match o:
        case "ok" | "gated" | "unavailable" | "request_fault" | "truncated" | "invalid" | "refused":
            return o
        case _:
            assert_never(o)
