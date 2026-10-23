"""grounding -- the shapes every grounded generator shares: a Problem, the retry text, the belt (pure).

A grounded generator audits a model's structured answer into `Problem`s. The output validator
raises a retry ONCE with `retry_message(problems)` (pydantic-ai's `ModelRetry` carries it; the
`Generator` port's `validate=` / `retry_message=` arguments are the seam), and after that the
caller prunes what still fails -- a part that failed twice is never shipped. What a problem IS
stays with each generator (its audit); how a problem is carried, said back to the model, and
counted is here.

The belt is the last check on rendered text: every number in it must be one the inputs rendered.
Code renders every number-bearing sentence and the model cites inputs by id, so the belt is 0 BY
CONSTRUCTION, and a non-zero count is an alarm, never a tolerated rate. Numbers are read by the one
number law, `llm_ports.numbers`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from llm_ports.numbers import NUM, NUMBER_WORDS, digits

RETRY_PREFIX = "Fix these and answer again: "
# A code-written citation link is provenance, not a claim: `[example.com](https://example.com/2026/10/07/x)`.
_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")


@dataclass(frozen=True)
class Problem:
    """One thing wrong with a model answer. `path` names the part (`headline`, `claims.3`, ...),
    `reason` says what is wrong, `detail` names the evidence (the figures, the ids) when there is any."""

    path: str
    reason: str
    detail: str = ""


def describe(problems: Iterable[Problem]) -> str:
    """`path: reason (detail); ...` -- the one way a list of problems is said, to a model or a row."""
    return "; ".join(f"{p.path}: {p.reason}{f' ({p.detail})' if p.detail else ''}" for p in problems)


def retry_message(problems: Iterable[Problem]) -> str:
    """The text of the ONE retry: what failed, so the model can fix exactly that."""
    return RETRY_PREFIX + describe(problems)


def allowed_numbers(texts: Iterable[str]) -> set[str]:
    """Every digit run the inputs render, by the one number law (thousands separators dropped)."""
    out: set[str] = set()
    for t in texts:
        out |= digits(t)
    return out


def belt(text: str, allowed: set[str], words_ok: str = "", *, strip_links: bool = False) -> int:
    """Numbers in `text` that no input rendered: digit runs not in `allowed`, plus spelled numbers
    (and number-word tails) that do not appear in `words_ok`. Must be 0.

    `strip_links` drops every markdown link WHOLE, label and target: use it only where code writes
    the links (e.g. `[host](url)` citations), since a number in a label is not checked.

    `words_ok` is a substring haystack: to license a word found in ANY of several inputs, join them
    on a newline (a number word holds no whitespace, so a join can never glue two halves into one)."""
    body = _LINK.sub("", text or "") if strip_links else (text or "")
    hits = sum(1 for n in NUM.findall(body) if n.replace(",", "") not in allowed)
    ok = words_ok.lower()
    return hits + sum(1 for w in NUMBER_WORDS.findall(body) if w.lower() not in ok)


__all__ = ["RETRY_PREFIX", "Problem", "allowed_numbers", "belt", "describe", "retry_message"]
