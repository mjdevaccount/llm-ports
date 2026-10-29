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


def canon(run: str) -> str:
    """One digit run as its VALUE reads: thousands separators and trailing decimal zeros dropped.
    "2.20" and "2.2" are one number, "7.0" is "7", "1,234.50" is "1234.5"; an integer is unchanged
    ("100" keeps its zeros). An exact-string compare failed honest replies that wrote "peak 2.20"
    against a rendered 2.2 (EDGE trade.autopsy, 5 of 40 failing replies on 2026-10-27)."""
    n = run.replace(",", "")
    return n.rstrip("0").rstrip(".") if "." in n else n


def digits(text: str) -> set[str]:
    """Digit runs in `text` by value (`canon`): the set a belt compares against."""
    return {canon(n) for n in NUM.findall(text or "")}


def spelled(text: str) -> set[str]:
    """Spelled numbers (and number-word tails) in `text`, lower-cased."""
    return {w.lower() for w in NUMBER_WORDS.findall(text or "")}


def loose(text: str, allowed: set[str], words_ok: str = "") -> list[str]:
    """THE number law, in one place: every figure in `text` that no input licensed, in order of
    appearance, repeats kept -- a digit run whose VALUE (`canon`) is not in `allowed`, then a
    spelled number or number-word tail not found in `words_ok`. Empty = every number is licensed.

    `words_ok` is a substring haystack, lower-cased: join several inputs on a newline (a number
    word holds no whitespace, so a join can never glue two halves into one), and an input that
    shows '%' licenses 'percent' ("3 percent" says what "3%" says). Every belt, grader and
    validator reads a number through this one function, so they can never disagree about what a
    number is or what licenses it: `grounding.belt` / `slots.belt` count it, `uncited` sorts it,
    `call_graders.numbers_from_inputs` scores it."""
    ok = (words_ok or "").lower()
    if "%" in ok:
        ok += "\npercent"
    out = [canon(n) for n in NUM.findall(text or "") if canon(n) not in allowed]
    return out + [w.lower() for w in NUMBER_WORDS.findall(text or "") if w.lower() not in ok]


def uncited(claim: str, spans: Iterable[tuple[str, str]]) -> list[str]:
    """`loose` in the cited-spans shape: every digit run and spelled number in `claim` must appear
    in a cited span -- digits in a span's title or text, spelled words in its text. `spans` are
    (title, text) pairs. Returns the figures no span rendered, digits then words, each sorted and
    unique; empty = every number is cited. No spans -> every number is missing."""
    pairs = list(spans)
    have = set().union(*(digits(t) | digits(ti) for ti, t in pairs)) if pairs else set()
    hits = loose(claim, have, "\n".join(t for _, t in pairs))
    return sorted({h for h in hits if h[0].isdigit()}) + sorted({h for h in hits if not h[0].isdigit()})
