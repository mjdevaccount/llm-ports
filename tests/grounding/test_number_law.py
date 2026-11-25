"""The number law for free prose, stated the slow way, and held against `numbers.loose`.

`law` is the statement of ADR-0027: one clause a line, every input value tried, no bisect and no
regex over the reply. A reply is built from figures whose parts are known (the digit run, what
follows it, and what the law reads that as: `AFTER`), so the statement never parses what it
judges. `numbers.loose(rerender=True)` must name exactly what the statement names, on the ADR's
own examples (`CLAUSES`) and on generated replies. A hole in the law is a hole in the statement:
attack `law` and `AFTER`, and the generated test keeps the fast reader equal to them.

Run it wider with `LAW_EXAMPLES=200000` (then the examples are random, not the fixed set CI runs).
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP, ROUND_UP, Decimal, InvalidOperation

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from llm_ports.grounding import numbers


@dataclass(frozen=True)
class After:
    """What follows a digit run, and how the law reads it.

    Attributes:
        text: The characters directly after the run.
        kind: `"percent"`, `"scale"`, or `""` when the law reads no marker there.
        exp: The power of ten a scale stands for.
        word: The number word `text` holds standing alone ("billion" in " billion"); a letter, a
            sign and a word glued to the run ("3million") hold none.
        glued: What a refused scale is named with, after the run: the letter as written, or a
            word the number-word scan cannot read (glued to the run, or ended by an underscore).
    """

    text: str
    kind: str = ""
    exp: int = 0
    word: str = ""
    glued: str = ""


_PCT = [After("%", "percent"), After(" %", "percent"), After("pct", "percent"), After(" pct", "percent")]
_PCT += [After("percent", "percent"), After(" percent", "percent", word="percent")]
_PCT += [After(" Percent", "percent", word="percent")]
_LETTERS = {"K": 3, "k": 3, "M": 6, "B": 9, "T": 12, "bn": 9, "BN": 9, "Bn": 9}
_WORDS = {"thousand": 3, "million": 6, "billion": 9, "Billion": 9, "trillion": 12}
_SCALES = [After(s, "scale", e, glued=s) for s, e in _LETTERS.items()]
_SCALES += [After(" " + w, "scale", e, word=w.lower()) for w, e in _WORDS.items()]
_SCALES += [After(w, "scale", e, glued=w) for w, e in _WORDS.items()]
# One joiner may sit before a word: a plain space, a no-break space, a hyphen.
_PCT += [After("-percent", "percent", word="percent"), After(chr(0xA0) + "percent", "percent", word="percent")]
_SCALES += [After("-billion", "scale", 9, word="billion"), After(chr(0xA0) + "million", "scale", 6, word="million")]
#: Read as no marker: a lower-case letter is a unit or a timeframe, a spaced letter is a unit, and
#: a marker sits on the figure's own line after at most one plain space.
_NONE = [After(t) for t in ("", " R", "m", "b", "t", "mm", "MM", "Kb", " M", " k", " B", "-minute", "  %", "\n%")]
_NONE += [After("\npercent", word="percent"), After("\nbillion", word="billion")]
_NONE += [After("  million", word="million"), After(" bps", word="bps"), After(" hundred", word="hundred")]
_NONE += [After(" -million", word="million"), After("--percent", word="percent")]
# A marker ends before anything but a letter or a digit: the underscore of markdown italics ends
# one, and then the word is no number word standing alone.
_PCT += [After("%_", "percent"), After(" percent_", "percent")]
_SCALES += [After("B_", "scale", 9, glued="B"), After(" billion_", "scale", 9, glued=" billion")]
AFTER = _PCT + _SCALES + _NONE
_BY_TEXT = {a.text: a for a in AFTER}


@dataclass(frozen=True)
class Figure:
    """One digit run and what follows it; `run` empty for a number word standing alone."""

    run: str
    after: After

    def __str__(self) -> str:
        return self.run + self.after.text


def fig(run: str, after: str = "") -> Figure:
    return Figure(run, _BY_TEXT[after])


def word(text: str) -> Figure:
    return Figure("", After(" " + text, word=text))


def render(figures: Sequence[Figure]) -> str:
    return " ; ".join(map(str, figures))


def _number(run: str) -> Decimal | None:
    try:
        return Decimal(run.replace(",", ""))
    except InvalidOperation:  # "1.2.3" is a digit run and not a number
        return None


def _rounds(x: Decimal, shown: Decimal, places: int) -> bool:
    """`x` rounds, half up or half even, to `shown` at `places` decimals. No quantize: the gap."""
    step = Decimal(1).scaleb(-places)
    gap = x - shown
    if abs(gap) * 2 < step or gap * 2 == -step:  # nearer than half a step; or a tie, rounded up to it
        return True
    return gap * 2 == step and shown.scaleb(places) % 2 == 0  # a tie, rounded down to an even digit


def _written(run: str) -> str:
    """A prompt run as its value reads. A run with two points is no number: it reads as written."""
    n = run.replace(",", "")
    return n.rstrip("0").rstrip(".") if n.count(".") == 1 else n


def judged(reply: Sequence[Figure], prompt: Sequence[Figure]) -> list[tuple[str, str, str]]:
    """Return `(name, word, clause)` for each figure of `reply`: see `_figure`."""
    printed = {_written(p.run) for p in prompt if p.run}
    values = [v for p in prompt if p.run and (v := _number(p.run)) is not None]
    at_scale = [(p.after.exp, v) for p in prompt if p.after.kind == "scale" and (v := _number(p.run)) is not None]
    return [_figure(f, printed, values, at_scale) if f.run else ("", f.after.word, "a word") for f in reply]


def law(reply: Sequence[Figure], prompt: Sequence[Figure]) -> list[str]:
    """Return what the law names in `reply`: unlicensed runs in order, then unlicensed words."""
    text = render(prompt).lower() + ("\npercent" if "%" in render(prompt) else "")
    runs: list[str] = []
    words: list[str] = []
    for run, spoken, _ in judged(reply, prompt):
        runs += [run] if run else []
        words += [spoken] if spoken and (spoken not in text or spoken.startswith("!")) else []
    return runs + [w.lstrip("!") for w in words]


def _figure(
    f: Figure, printed: set[str], values: list[Decimal], at_scale: list[tuple[int, Decimal]]
) -> tuple[str, str, str]:
    """Return `(name, word, clause)` for one figure: the run's name when unlicensed, the number
    word left for the prompt's own words to license (`!word` when the law refuses it whatever they
    say), and which clause decided it (`REACHED` counts them over the generated replies)."""
    name, after, shown = _written(f.run), f.after, _number(f.run)
    if shown is None:  # not a number: licensed only as printed, and what follows it marks nothing
        return ("", after.word, "no number, printed") if name in printed else (name, after.word, "no number, named")
    places = len(f.run.replace(",", "").partition(".")[2])  # the decimals written are the precision claimed

    def near(x: Decimal) -> bool:
        return _rounds(x, shown, places)

    def marked(x: Decimal) -> bool:  # a whole figure with a marker rounds a magnitude of 1 or more
        return near(x) and (places > 0 or x >= 1)

    # 1. As it stands: an input printed this value, or (decimals written) one input rounds to it.
    #    A whole number is never a rounding.
    rounded = places > 0 and any(near(v) for v in values)
    stands = "printed" if name in printed else "rounded" if rounded else ""
    plain = ("", after.word, "stands, " + stands) if stands else (name, after.word, "named")
    if after.kind == "percent":
        # 2. A percentage: one input rounded, or one input WITH A FRACTIONAL PART times 100.
        if any(marked(v) or (v % 1 != 0 and marked(v.scaleb(2))) for v in values):
            return "", "", "a percentage"
        return plain  # else "percent" is a plain number word
    if after.kind == "scale":
        # 3. A scale: one input IS the figure at that scale, or rounds to it once divided; or an
        #    input shown AT that scale is the figure, or rounds to it.
        if any(v == shown.scaleb(after.exp) or marked(v.scaleb(-after.exp)) for v in values):
            return "", "", "a scale, of an input"
        if any(e == after.exp and (x == shown or marked(x)) for e, x in at_scale):
            return "", "", "a scale, as an input shows it"
        if not stands:
            return plain
        # 4. A scale is a second claim: a run licensed as it stands does not carry it.
        clause = "a scale refused, " + stands
        return (name + after.glued, "", clause) if after.glued else ("", "!" + after.word, clause)
    return plain


#: Every clause the generated replies must reach, often enough to mean something.
REACHED = ["no number, printed", "no number, named", "stands, printed", "stands, rounded", "named"]
REACHED += ["a percentage", "a scale, of an input", "a scale, as an input shows it"]
REACHED += ["a scale refused, printed", "a scale refused, rounded", "a word"]


def fast(reply: Sequence[Figure], prompt: Sequence[Figure]) -> list[str]:
    """The reader under test, fed as `call.graders.numbers_from_inputs` feeds it."""
    inputs = render(prompt)
    return numbers.loose(render(reply), numbers.digits(inputs), inputs, rerender=True)


#: The ADR's sentences, one row each way a clause cuts: (clause, prompt, reply, what is named).
CLAUSES: list[tuple[str, list[Figure], list[Figure], list[str]]] = [
    ("1 an equal run", [fig("15")], [fig("15", "-minute")], []),
    ("1 a whole number is never a rounding", [fig("14.878")], [fig("15", "-minute")], ["15"]),
    ("1 rounding to the decimals written", [fig("0.5918")], [fig("0.59", " R")], []),
    ("1 trailing zeros are the precision", [fig("28.89")], [fig("28.00")], ["28"]),
    ("1 truncation is not rounding", [fig("21.9963")], [fig("21.99")], ["21.99"]),
    ("1 a tie rounds up", [fig("2.45")], [fig("2.5")], []),
    ("1 a tie rounds to even", [fig("2.45")], [fig("2.4")], []),
    ("1 a tie never rounds to odd below", [fig("2.55")], [fig("2.5")], ["2.5"]),
    ("2 a fraction as a percentage", [fig("0.7486")], [fig("75", "%")], []),
    ("2 the value itself, rounded", [fig("13.49")], [fig("13", " percent")], []),
    ("2 the value itself, below one", [fig("0.2409")], [fig("0.24", " percent")], []),
    ("2 a whole value is never moved", [fig("1")], [fig("100", "%")], ["100"]),
    ("2 a whole value is never moved, decimals", [fig("2")], [fig("200.0", "%")], ["200"]),
    ("2 a marked whole number rounds one or more", [fig("0.004")], [fig("0", "%")], ["0"]),
    ("2 a slipped point is no percentage", [fig("26")], [fig("2.6", "%")], ["2.6"]),
    ("2 the word of an equal run is a plain word", [fig("0")], [fig("0", " percent")], ["percent"]),
    ("2 and a percent sign in the inputs licenses it", [fig("0", "%")], [fig("0", " percent")], []),
    ("3 a scale letter", [fig("19813893517")], [fig("19.8", "B")], []),
    ("3 a scale word", [fig("19813893517")], [fig("19.8", " billion")], []),
    ("3 a marked whole number at a scale", [fig("19813893517")], [fig("20", "B")], []),
    ("3 exact at the scale", [fig("3000")], [fig("3", "k")], []),
    ("3 the wrong scale", [fig("19813893517")], [fig("19.8", "M")], ["19.8"]),
    ("3 a lower-case letter is a timeframe", [fig("15234567")], [fig("15", "m")], ["15"]),
    ("3 a spaced letter is a unit", [fig("19813893517")], [fig("19.8", " M")], ["19.8"]),
    ("3 a marker is on the figure's line", [fig("19813893517")], [fig("19.8", "\nbillion")], ["19.8", "billion"]),
    ("3 an equal run re-rendering another input", [fig("20"), fig("19813893517")], [fig("20", " billion")], []),
    ("4 an equal run is no magnitude", [fig("20")], [fig("20", "B")], ["20B"]),
    ("4 nor is a rounding", [fig("0.5918")], [fig("0.6", "B")], ["0.6B"]),
    ("4 the word, whatever the inputs say", [fig("20"), fig("3.2", " billion")], [fig("20", " billion")], ["billion"]),
    ("4 a glued word", [fig("20")], [fig("20", "million")], ["20million"]),
    ("4 a hyphened word", [fig("20"), word("billion")], [fig("20", "-billion")], ["billion"]),
    ("3 a hyphened word re-rendering", [fig("19813893517")], [fig("19.8", "-billion")], []),
    ("4 an input at that scale, as a letter", [fig("1.2", "B")], [fig("1.2", " billion")], []),
    ("4 an input at that scale, rounded", [fig("0.5918", "B")], [fig("0.59", "B")], []),
    ("4 an input at another scale", [fig("1.2", "M")], [fig("1.2", "B")], ["1.2B"]),
    ("3 an input at that scale, to a whole number", [fig("19.8", "B")], [fig("20", "B")], []),
    ("3 an input at that scale, in words", [fig("3.2", " billion")], [fig("3", " billion")], []),
    ("3 a marked whole number, at-scale input below one", [fig("0.6", "B")], [fig("1", "B")], ["1"]),
    ("3 any one input is a magnitude, a year too", [fig("2026")], [fig("2", "K")], []),
    ("3 a marker ends at an underscore", [fig("19813893517")], [fig("19.8", "B_")], []),
    ("3 the spellings are a closed table", [fig("20")], [fig("20", "MM")], []),
    ("4 the word after a rounding", [fig("0.5918")], [fig("0.6", " billion")], ["billion"]),
    ("4 a label is no scale", [word("net_gex_bn"), fig("19.8")], [fig("19.8", "bn")], ["19.8bn"]),
    ("4 a word the scan cannot read", [fig("20")], [fig("20", " billion_")], ["20 billion"]),
    ("5 a spelled number needs the prompt's words", [fig("3")], [word("three")], ["three"]),
    ("5 not a number: as printed only", [fig("1.2.3")], [fig("1.2.3", "B"), fig("1.2", "")], ["1.2"]),
    ("5 not a number: its zeros are no decimals", [fig("1.2.0")], [fig("120", "%"), fig("1.2", " R")], ["120", "1.2"]),
    ("5 not a number: nothing rounds to it", [fig("3.14.0")], [fig("3.1"), fig("3.14.0")], ["3.1"]),
    ("5 a comma in a run is dropped", [fig("1.2345")], [fig("1.2,3")], []),
]


@pytest.mark.parametrize(("clause", "prompt", "reply", "named"), CLAUSES, ids=[c[0] for c in CLAUSES])
def test_the_statement_and_the_reader_agree_with_the_adr(
    clause: str, prompt: list[Figure], reply: list[Figure], named: list[str]
) -> None:
    assert law(reply, prompt) == named, f"the statement: {clause}"
    assert fast(reply, prompt) == named, f"the reader: {clause}"


_PINNED = ["0", "1", "2", "20", "75", "100", "0.5", "0.5918", "0.7486", "0.004", "0.045", "13.49", "14.878"]
_PINNED += ["18.37", "28.89", "715.87", "2.45", "2.55", "3000", "999999", "15234567", "19813893517", "1.0"]
_PINNED += ["1,234.5", "1.2.3", "0.005", "1.5", "540,545,550", "1.2.0", "3.14.0", "2026.11.20", "1.2,3", "2026"]
_RUNS = st.one_of(
    st.sampled_from(_PINNED),
    st.integers(0, 10**12).map(str),
    st.builds(lambda n, p: format(Decimal(n).scaleb(-p), "f"), st.integers(0, 10**7), st.integers(0, 4)),
)
_AFTERS = st.sampled_from(AFTER)
_PROMPTS = st.lists(
    st.one_of(st.builds(Figure, _RUNS, _AFTERS), st.sampled_from(["ten", "billion", "percent"]).map(word)),
    min_size=0,
    max_size=6,
)


#: Runs with two points: a version, a date. No number, so licensed only as printed.
_DOTTED = ["1.2.3", "1.2.0", "3.14.0", "2026.11.20", "10.0.0.1"]
_HOW = ["made", "kept", "kept", "rounded", "rounded", "rounded", "scaled", "scaled", "dotted", "free", "word"]


@st.composite
def _replies(draw: st.DrawFn, prompt: Sequence[Figure]) -> list[Figure]:
    """Figures a model could have made FROM the prompt: one run kept under another marker, one
    value rounded, one value shown at a scale, one value moved and nudged, a sum."""
    sources = [p for p in prompt if p.run] or [fig("7")]
    values = [v for p in sources if (v := _number(p.run)) is not None] or [Decimal(7)]
    out = []
    for _ in range(draw(st.integers(1, 4))):
        how = draw(st.sampled_from(_HOW))
        if how == "word":
            out.append(word(draw(st.sampled_from(["ten", "three", "billion", "million", "percent", "ighteen"]))))
            continue
        if how == "free":
            out.append(Figure(draw(_RUNS), draw(_AFTERS)))
            continue
        source = draw(st.sampled_from(sources))
        after = draw(st.one_of(_AFTERS, st.sampled_from([source.after, _BY_TEXT[""], _BY_TEXT[" R"], *_SCALES[:4]])))
        if how == "dotted":
            out.append(Figure(draw(st.sampled_from(_DOTTED)), after))
            continue
        value = _number(source.run)
        if how == "kept" or value is None:
            out.append(Figure(source.run, after))
            continue
        rounding = draw(st.sampled_from([ROUND_HALF_UP, ROUND_HALF_EVEN, ROUND_HALF_EVEN, ROUND_DOWN]))
        if how == "rounded":  # one input to fewer decimals, honestly or cut short
            have = len(source.run.partition(".")[2])  # fewer decimals than the input has, when it has two
            places = draw(st.integers(1, have - 1) if have > 1 else st.integers(0, 2))
            out.append(Figure(format(value.quantize(Decimal(1).scaleb(-places), rounding), "f"), after))
            continue
        if how == "scaled":  # one input shown at a scale, under that scale's marker or another
            scale = draw(st.sampled_from(_SCALES))
            step = Decimal(1).scaleb(-draw(st.integers(0, 2)))
            run = format(value.scaleb(-scale.exp).quantize(step, rounding), "f")
            out.append(Figure(run, draw(st.sampled_from([scale, scale, scale, after]))))
            continue
        base = draw(st.sampled_from(values)) + draw(st.sampled_from([Decimal(0)] * 4 + values))
        moved = base.scaleb(draw(st.sampled_from([0, 0, 0, 2, 2, -3, -6, -9, -12, 1, -1])))
        places = draw(st.integers(0, 4))
        step = Decimal(1).scaleb(-places)
        rounding = draw(st.sampled_from([ROUND_HALF_UP, ROUND_HALF_EVEN, ROUND_DOWN, ROUND_UP]))
        shown = moved.quantize(step, rounding) + draw(st.sampled_from([0, 0, 0, 1, -1])) * step
        out.append(Figure(format(max(shown, Decimal(0)), "f"), draw(_AFTERS)))
    return out


_WIDE = int(os.environ.get("LAW_EXAMPLES", "0"))
_SETTINGS = settings(max_examples=_WIDE or 3000, deadline=None, derandomize=not _WIDE)
_PROPERTY = settings(max_examples=_WIDE or 800, deadline=None, derandomize=not _WIDE)
#: The fewest figures each clause decides over the differential's examples.
_FLOOR = 60


def test_the_reader_names_what_the_statement_names() -> None:
    """The differential, and its own witness: a clause the generated replies barely reach is a
    clause this test does not hold, so the count of each is asserted beside the equality."""
    reached: Counter[str] = Counter()

    @_SETTINGS
    @given(data=st.data(), prompt=_PROMPTS)
    def differ(data: st.DataObject, prompt: list[Figure]) -> None:
        reply = data.draw(_replies(prompt))
        reached.update(clause for _, _, clause in judged(reply, prompt))
        assert fast(reply, prompt) == law(reply, prompt), f"reply {render(reply)!r} over prompt {render(prompt)!r}"

    differ()
    thin = {clause: reached[clause] for clause in REACHED if reached[clause] < _FLOOR}
    assert not thin, f"the generated replies barely reach {thin} of {sum(reached.values())} figures: {dict(reached)}"
    assert set(reached) <= set(REACHED), f"a clause REACHED does not list: {set(reached) - set(REACHED)}"


@_PROPERTY
@given(prompt=_PROMPTS)
def test_a_reply_that_quotes_its_prompt_is_licensed(prompt: list[Figure]) -> None:
    assert fast(prompt, prompt) == []


@_PROPERTY
@given(data=st.data(), prompt=_PROMPTS, more=_PROMPTS)
def test_more_inputs_never_name_more(data: st.DataObject, prompt: list[Figure], more: list[Figure]) -> None:
    reply = data.draw(_replies(prompt))
    assert len(fast(reply, prompt + more)) <= len(fast(reply, prompt)), f"{render(reply)!r}"


@_PROPERTY
@given(data=st.data(), prompt=_PROMPTS)
def test_a_bare_whole_number_is_licensed_only_by_an_input_that_printed_it(
    data: st.DataObject, prompt: list[Figure]
) -> None:
    n = data.draw(st.integers(0, 10**6).map(str))
    printed = {_written(p.run) for p in prompt if p.run}
    assert (fast([fig(n, " R")], prompt) == []) is (n in printed)


@_PROPERTY
@given(data=st.data(), prompt=_PROMPTS)
def test_the_belt_reading_is_exact_and_reads_no_marker(data: st.DataObject, prompt: list[Figure]) -> None:
    reply, inputs = data.draw(_replies(prompt)), render(prompt)
    printed = numbers.digits(inputs)
    named = numbers.loose(render(reply), printed, inputs)
    assert [n for n in named if n[0].isdigit()] == [c for f in reply if f.run and (c := _written(f.run)) not in printed]
