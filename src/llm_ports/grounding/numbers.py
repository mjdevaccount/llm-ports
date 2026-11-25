"""The number law: how every grader, belt and validator reads a number, and its English statement.

A figure in model output is licensed when an input rendered the same value, as digits or spelled
out. `loose` is the check; `law_text` is the same law stated for a model judge. Every reader goes
through this module, so a rule and a reader can disagree about judgement but never about the law.

A reader of free prose may also license a figure that shows ONE input value again, coarser
(`rerenders`): rounded to fewer decimals, in thousands, millions or billions, or as a percentage.
No arithmetic over two values is ever licensed.

Invariants:
    - Pure: no I/O.
    - Digit runs compare by value (`canon`), never by string.

Rationale: ADR-0003, ADR-0027.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation

#: A digit run with optional thousands or decimal separators: `1,234.5`, `740`, `0.62`.
NUM = re.compile(r"\d+(?:[.,]\d+)*")

#: A spelled number, or the tail of a number word. A digit-free grammar does not stop a number: a
#: model spells it ("nine one point seven million"), or, with the digit token blocked, emits the
#: rest of the word it was reaching for ("ighteen-year high"). The tails are never standalone
#: English. Rationale: ADR-0004.
NUMBER_WORDS = re.compile(
    r"\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million|billion|trillion|percent|bps|"
    r"ight|ighteen|ighty|leven|welve|hirty|wenty|orty|ifty|ixty|eventy|inety|undred|housand|"
    r"illion|illions|ills)\b",
    re.IGNORECASE,
)

#: A digit run as a model may read it: `25bp`, `14:00`, `3.1%` and `5,000` are each one match, so
#: masking a clock or a percentage leaves no half a number behind.
DIGITS = re.compile(r"\d(?:[\d%:]|[.,]\d)*")


#: What may follow a digit run that shows a fraction as a percentage: `75%`, `75 percent`.
_PERCENT = re.compile(r"\s?(?:%|percent\b|pct\b)", re.IGNORECASE)
#: What may follow a digit run that shows a value in thousands: a letter directly after the run
#: (`19.8B`, `3k`) or a word (`3.2 million`). A spaced letter (`5 m`) is a unit, not a scale.
_SCALE = re.compile(r"(?:(k|mm|m|bn|b|t)\b|\s?(thousand|million|billion|trillion)\b)", re.IGNORECASE)
_SCALE_EXP = {"k": 3, "thousand": 3, "m": 6, "mm": 6, "million": 6, "b": 9, "bn": 9, "billion": 9}
_SCALE_EXP |= {"t": 12, "trillion": 12}
#: How much text after a digit run `loose` hands to `rerenders`: room for " trillion".
_AFTER = 12


def redact(text: str, *, hole: str, pattern: re.Pattern[str] = NUM) -> str:
    """Replace every number in `text` with `hole`.

    Use it on context that may back a fact but never a figure: the figure is gone before the model
    reads it, so there is nothing to copy, and the fact survives.

    Args:
        text: The text to redact; None reads as empty.
        hole: The replacement glyph, e.g. `[n]`. The caller validates that the model did not copy it.
        pattern: What counts as a number; `NUM` by default.

    Returns:
        `text` with each match replaced.

    Rationale: ADR-0004.
    """
    return pattern.sub(hole, text or "")


def shorten(text: str, width: int, *, placeholder: str = "…") -> str:
    """Return `text` cut to at most `width` characters on a word, never inside a number.

    Whitespace collapses first. A text that fits is returned whole, with no placeholder. Otherwise
    the cut falls on the last word boundary that leaves room for `placeholder`. When the first word
    alone is too long, the cut is hard but steps back to the start of any digit run (`DIGITS`) it
    would split, so "25bp" can never read as "2…". Show the full text beside the short one (a
    tooltip, a detail row): the placeholder says text is missing, not what it was.

    Args:
        text: The text to shorten; None reads as empty.
        width: The longest result, placeholder included.
        placeholder: Appended to a cut text.

    Returns:
        `text` with collapsed whitespace, or a cut prefix ending in `placeholder`. Every digit run
        in the result is a whole digit run of `text`.

    Raises:
        ValueError: `width` leaves no room for one character before `placeholder`.

    Rationale: ADR-0003.
    """
    if width <= len(placeholder):
        raise ValueError(f"width {width} leaves no room before a {len(placeholder)}-char placeholder")
    flat = " ".join((text or "").split())
    if len(flat) <= width:
        return flat
    room = width - len(placeholder)
    cut = flat.rfind(" ", 0, room + 1)
    if cut <= 0:
        cut = room
        for m in DIGITS.finditer(flat):
            if m.start() < cut < m.end():
                cut = m.start()
                break
    return flat[:cut].rstrip() + placeholder


def canon(run: str) -> str:
    """Return one digit run as its value reads.

    Thousands separators and trailing decimal zeros are dropped: "2.20" and "2.2" are one number,
    "7.0" is "7", "1,234.50" is "1234.5". An integer keeps its zeros ("100").

    Rationale: ADR-0003.
    """
    n = run.replace(",", "")
    return n.rstrip("0").rstrip(".") if "." in n else n


def digits(text: str) -> set[str]:
    """Return the digit runs in `text` by value (`canon`): the set a belt compares against."""
    return {canon(n) for n in NUM.findall(text or "")}


def spelled(text: str) -> set[str]:
    """Return the spelled numbers and number-word tails in `text`, lower-cased."""
    return {w.lower() for w in NUMBER_WORDS.findall(text or "")}


def _decimal(run: str) -> Decimal | None:
    try:
        return Decimal(run.replace(",", ""))
    except InvalidOperation:  # "1.2.3" is a digit run and not a number
        return None


def _rounds_to(value: Decimal, shown: Decimal, step: Decimal) -> bool:
    try:
        return shown in (value.quantize(step, ROUND_HALF_UP), value.quantize(step, ROUND_HALF_EVEN))
    except InvalidOperation:  # more digits than the decimal context holds
        return False


def rerenders(run: str, after: str, allowed: Iterable[str]) -> bool:
    """Return whether `run` shows one licensed value again, coarser.

    Pure. Three renderings of ONE value are recognised, and nothing computed from two:

    - rounding: the value rounded (half up or half even) to the decimals `run` is written with,
      when `run` has at least one decimal ("0.59" for 0.5918, "14.88" for 14.878);
    - a percentage: `run` followed by `%`, "percent" or "pct" shows the value times 100
      ("75%" for 0.7486), or the value itself rounded ("13.5%" for 13.49);
    - a scale: `run` followed directly by `K`, `M`, `B` or `T`, or by "thousand", "million",
      "billion" or "trillion", shows the value divided by it ("19.8B" for 19813893517).

    An integer `run` with no percent sign or scale after it is never a rounding: every small
    integer is the rounding of some fraction, so licensing it would license an invented count.
    A marked integer rounds a value of magnitude 1 or more only.

    Args:
        run: The digit run as written (`NUM`), separators and trailing zeros kept: the zeros say
            how many decimals it was rounded to.
        after: The text directly after the run; only its first characters are read.
        allowed: Licensed digit runs, canonical (see `digits`).

    Returns:
        True when some value in `allowed` renders as `run`. An equal value is not a re-rendering:
        `loose` licenses it before asking. `loose` also licenses the scale or percent word the
        re-rendering needed ("billion" in "19.8 billion"), and no other.

    Rationale: ADR-0027.
    """
    return _marker_used(run, after, allowed) is not None


def _marker_used(run: str, after: str, allowed: Iterable[str]) -> int | None:
    """Return how `run` re-renders an allowed value.

    None when it does not; else the length of the marker in `after` (the percent sign or scale)
    the licence needed, 0 for a plain rounding.
    """
    shown = _decimal(run)
    if shown is None:
        return None
    exponent = shown.as_tuple().exponent
    places = -exponent if isinstance(exponent, int) and exponent < 0 else 0
    step = Decimal(1).scaleb(-places)
    percent = _PERCENT.match(after)
    scale = None if percent else _SCALE.match(after)
    marker = percent or scale
    shift = 2 if percent else -_SCALE_EXP[(scale.group(1) or scale.group(2)).lower()] if scale else 0
    plain = None
    for known in allowed:
        base = _decimal(known)
        if base is None:
            continue
        if places and _rounds_to(base, shown, step):
            plain = 0  # the value itself, rounded to a decimal: no marker needed
        if marker is None:
            continue
        # a whole percentage the value already was, or the value moved by the marker
        whole = percent is not None and not places and abs(base) >= 1 and _rounds_to(base, shown, step)
        moved = base.scaleb(shift)
        if whole or moved == shown or ((places or abs(moved) >= 1) and _rounds_to(moved, shown, step)):
            return marker.end()
    return plain


def loose(text: str, allowed: set[str], words_ok: str = "", *, rerender: bool = False) -> list[str]:
    """Return every figure in `text` that no input licensed.

    The number law in one place: a digit run whose value (`canon`) is not in `allowed`, then a
    spelled number or number-word tail not found in `words_ok`. `grounding.belt` and `slots.belt`
    count the result, `uncited` sorts it, and `call.graders.numbers_from_inputs` scores it.

    Args:
        text: The text to check; None reads as empty.
        allowed: Licensed digit runs, already canonical (see `digits`).
        words_ok: A lower-cased substring haystack for spelled numbers. Join several inputs on a
            newline: a number word holds no whitespace, so a join never glues two halves into one.
            An input that shows `%` also licenses "percent".
        rerender: Also license a digit run that shows one allowed value again, coarser
            (`rerenders`): for free prose over raw inputs, where a model rounds what it reads.
            Off by default: a belt over values code printed compares them as printed.

    Returns:
        The unlicensed figures in order of appearance, digit runs first, repeats kept. Empty when
        every number is licensed.

    Rationale: ADR-0003, ADR-0027.
    """
    body = text or ""
    ok = (words_ok or "").lower()
    if "%" in ok:
        ok += "\npercent"
    out = []
    marked: list[tuple[int, int]] = []  # the scale or percent word a re-rendering was licensed with
    for m in NUM.finditer(body):
        if canon(m.group()) in allowed:
            continue
        used = _marker_used(m.group(), body[m.end() : m.end() + _AFTER], allowed) if rerender else None
        if used is None:
            out.append(canon(m.group()))
        elif used:
            marked.append((m.end(), m.end() + used))
    words = (w for w in NUMBER_WORDS.finditer(body) if not any(a <= w.start() < b for a, b in marked))
    return out + [w.group().lower() for w in words if w.group().lower() not in ok]


def uncited(claim: str, spans: Iterable[tuple[str, str]]) -> list[str]:
    """Return the figures in `claim` that no cited span renders: `loose` in the cited-spans shape.

    A digit run is licensed by a span's title or text; a spelled number by a span's text.

    Args:
        claim: The claim to check.
        spans: `(title, text)` pairs the claim cites.

    Returns:
        The unlicensed figures, digit runs then words, each group sorted and unique. Empty when
        every number is cited; with no spans, every number is missing.
    """
    pairs = list(spans)
    have = set().union(*(digits(t) | digits(ti) for ti, t in pairs)) if pairs else set()
    hits = loose(claim, have, "\n".join(t for _, t in pairs))
    return sorted({h for h in hits if h[0].isdigit()}) + sorted({h for h in hits if not h[0].isdigit()})


@dataclass(frozen=True)
class LawExamples:
    """The consumer's own words, named in the slots law as examples of what is not a figure.

    A judge reads a domain word more reliably when the law names one. Each item is rendered
    verbatim, so quote a literal word yourself (`"'flat'"`) and leave a description bare.

    Attributes:
        paraphrase: Words, labels and paraphrase that are not figures.
        fact_names: Words that name a fact, which are never a reason to fail, cited or not.
    """

    paraphrase: Sequence[str] = ()
    fact_names: Sequence[str] = ()


def _listed(items: Sequence[str]) -> str:
    return f" ({', '.join(items)})" if items else ""


# The law as a model judge reads it. These strings are prompt text: a change to them changes every
# judge verdict, so they are pinned byte-for-byte by tests/fixtures/golden/rubric_v0_11.json.
_GRAMMAR_VALUES = (
    "A value the OUTPUT GRAMMAR asks the model to give -- a score, conviction, confidence "
    "or urgency number, an enum label, a category -- is the model's ANSWER, not a figure "
    "taken from the Input: never fail it under this law (law 4 asks whether the Input "
    "supports it). "
)
_WEB_LAW = (
    "2. WEB FIGURES ARE NOT JUDGED ABSENT: this role searched and read web pages you cannot "
    "see, so a figure, date or name missing from the Input is expected and is NOT invented. "
    "Fail only a figure that CONTRADICTS the Input."
)
_SLOTS_LAW = (
    "2. FIGURES ARE CITED, NEVER WRITTEN: this role writes under a digit-free grammar and "
    "cites each value as `{{id}}`, naming a fact the Input lists. A number written out in the "
    "Output, as digits or as number words, is invented and fails; a cited id the Input does "
    "not list fails; a cited id the Input lists is the law kept, whatever its value. This law "
    "is about NUMBERS only: words, labels and paraphrase{paraphrase} are not figures, and a word "
    "naming a fact{fact_names} is never a reason to fail under this law, cited or not."
)
_INPUT_LAW = (
    "2. NUMBERS FROM THE INPUT ONLY: " + _GRAMMAR_VALUES + "Every other number, level, price, "
    "percentage or count in the Output appears in the Input (its prompt, its system instructions "
    "or an attached image), as digits or spelled out. A figure the Input did not carry is "
    "invented, and fails.\n"
)


def law_text(mode: str, *, web: bool = False, examples: LawExamples | None = None) -> str:
    """Return the number law as one numbered rubric line for a model judge.

    The statement matches what `loose` enforces for the same call, so a deterministic grader and a
    judge disagree only about judgement. A value the output grammar itself asks for (a score, an
    enum label) is always the model's answer, never a figure under this law.

    Args:
        mode: The role's grounding mode. `"slots"` (the model cites `{id}` under a digit-free
            grammar, so any written number is invented) changes the law; any other word reads
            figures against the Input.
        web: The call read web pages the judge cannot see. A figure missing from the Input is then
            expected; only one that contradicts the Input fails. Takes precedence over `mode`.
        examples: Domain words the `"slots"` law names as not figures; none by default.

    Returns:
        The law as rubric text, ending in a newline.

    Rationale: ADR-0003.
    """
    if web:
        return _WEB_LAW + " " + _GRAMMAR_VALUES.rstrip() + "\n"
    if mode == "slots":
        ex = examples or LawExamples()
        law = _SLOTS_LAW.format(paraphrase=_listed(ex.paraphrase), fact_names=_listed(ex.fact_names))
        return law + " " + _GRAMMAR_VALUES.rstrip() + "\n"
    return _INPUT_LAW


__all__ = [
    "DIGITS",
    "NUM",
    "NUMBER_WORDS",
    "LawExamples",
    "canon",
    "digits",
    "law_text",
    "loose",
    "redact",
    "rerenders",
    "shorten",
    "spelled",
    "uncited",
]
