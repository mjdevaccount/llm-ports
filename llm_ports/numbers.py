"""Numbers as a belt reads them: the one digit-run regex and the one spelled-number regex.

Pure. Shared by every grader and validator that asks "did the model write a number no input
rendered", so each reads a number the same way.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# A digit run with optional thousands / decimal separators: `1,234.5`, `740`, `0.62`.
NUM = re.compile(r"\d+(?:[.,]\d+)*")

# A digit-free grammar does not stop a number: on the 09-25 replay the model SPELLED them
# ("nine one point seven million"), or, with the digit token blocked, emitted the rest of the
# word it was reaching for ("ighteen-year high", "from ills to ills"). Both are numbers no fact
# rendered; the fragments are the tails of number words and never standalone English.
NUMBER_WORDS = re.compile(
    r"\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million|billion|trillion|percent|bps|"
    r"ight|ighteen|ighty|leven|welve|hirty|wenty|orty|ifty|ixty|eventy|inety|undred|housand|"
    r"illion|illions|ills)\b", re.IGNORECASE)

# A digit run as the MODEL may read it: `25bp`, `14:00`, `3.1%` and `5,000` are each one hole, so
# a masked clock or percentage does not leave half a number behind.
DIGITS = re.compile(r"\d(?:[\d%:]|[.,]\d)*")


def redact(text: str, *, hole: str, pattern: re.Pattern[str] = NUM) -> str:
    """Every number in `text` becomes `hole`. A cached research note may back a fact and never a
    number; telling the model so moved nothing (a replay: 18 of 32 claims still carried a note's
    figure), so the figure is removed before the model reads it. There is nothing to copy; the fact
    survives. The hole glyph is the caller's (e.g. `[n]`, `(n)`) and each caller validates that the
    model did not copy the hole itself."""
    return pattern.sub(hole, text or "")


def digits(text: str) -> set[str]:
    """Digit runs in `text`, thousands separators dropped: the set a belt compares against."""
    return {n.replace(",", "") for n in NUM.findall(text or "")}


def spelled(text: str) -> set[str]:
    """Spelled numbers (and number-word tails) in `text`, lower-cased."""
    return {w.lower() for w in NUMBER_WORDS.findall(text or "")}


def uncited(claim: str, spans: Iterable[tuple[str, str]]) -> list[str]:
    """THE number law, in one place: every digit run and spelled number in `claim` must appear in
    a cited span -- digits in a span's title or text, spelled words in its text, and a span that
    shows '%' licenses 'percent'. `spans` are (title, text) pairs. Returns the figures no span
    rendered, digits then words, each sorted; empty = every number is cited. No spans -> every
    number is missing. One rule, so every validator and Grounder that applies it moves together."""
    pairs = list(spans)
    have = set().union(*(digits(t) | digits(ti) for ti, t in pairs)) if pairs else set()
    words = set().union(*(spelled(t) for _, t in pairs)) if pairs else set()
    if any("%" in t for _, t in pairs):
        words.add("percent")            # "3 percent" says what a span's "3%" says
    return sorted(digits(claim) - have) + sorted(spelled(claim) - words)
