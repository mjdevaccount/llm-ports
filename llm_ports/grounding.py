"""Shapes every grounded generator shares: a `Problem`, the retry text, and the belt.

A grounded generator audits a model's structured answer into `Problem`s. Its output validator asks
for one retry with `retry_message(problems)` (pydantic-ai's `ModelRetry` carries it; the
`Generator` port's `validate=` and `retry_message=` arguments are the seam). After that retry the
caller prunes what still fails. What counts as a problem is each generator's own audit; how a
problem is carried, said back to the model and counted lives here.

The belt is the last check on rendered text: every number in it must be one the inputs rendered,
read by the number law (`numbers`).

Invariants:
    - Pure: no I/O.
    - A part that failed its retry is pruned, never shipped.
    - When code renders every number-bearing sentence and the model cites inputs by id, the belt
      is 0 by construction; a non-zero count is an alarm, never a tolerated rate.

Rationale: ADR-0002, ADR-0003.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from llm_ports.numbers import digits
from llm_ports.numbers import loose as _loose

RETRY_PREFIX = "Fix these and answer again: "
# A code-written citation link is provenance, not a claim: `[example.com](https://example.com/a/b)`.
_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")


@dataclass(frozen=True)
class Problem:
    """One thing wrong with a model answer.

    Attributes:
        path: The part of the answer (`headline`, `claims.3`, ...).
        reason: What is wrong.
        detail: The evidence (the figures, the ids), when there is any.
    """

    path: str
    reason: str
    detail: str = ""


def describe(problems: Iterable[Problem]) -> str:
    """Return problems as `path: reason (detail); ...`, the one way they are said to a model or a log."""
    return "; ".join(f"{p.path}: {p.reason}{f' ({p.detail})' if p.detail else ''}" for p in problems)


def retry_message(problems: Iterable[Problem]) -> str:
    """Return the text of the one retry: what failed, so the model can fix exactly that."""
    return RETRY_PREFIX + describe(problems)


def allowed_numbers(texts: Iterable[str]) -> set[str]:
    """Return every digit run the inputs render, by value (`numbers.digits`)."""
    out: set[str] = set()
    for t in texts:
        out |= digits(t)
    return out


def loose(text: str, allowed: set[str], words_ok: str = "", *, strip_links: bool = False) -> list[str]:
    """Return the figures in `text` that no input rendered (`numbers.loose`).

    Each figure is named, so a verifier can say which one it refused and a retry can fix exactly
    that.

    Args:
        text: The text to check.
        allowed: Licensed digit runs by value (see `allowed_numbers`).
        words_ok: A substring haystack for spelled numbers. To license a word found in any of
            several inputs, join them on a newline.
        strip_links: Drop every markdown link whole, label and target, before checking. Use it only
            where code writes the links: a number inside a label is not checked.

    Returns:
        The unlicensed figures in order; empty when every number is licensed.
    """
    body = _LINK.sub("", text or "") if strip_links else (text or "")
    return _loose(body, allowed, words_ok)


def belt(text: str, allowed: set[str], words_ok: str = "", *, strip_links: bool = False) -> int:
    """Return how many figures in `text` no input rendered (`loose`, counted). Expected: 0."""
    return len(loose(text, allowed, words_ok, strip_links=strip_links))


__all__ = ["RETRY_PREFIX", "Problem", "allowed_numbers", "belt", "describe", "loose", "retry_message"]
