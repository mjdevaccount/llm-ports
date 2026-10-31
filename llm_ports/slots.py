"""slots -- grounded prose by construction: the model cites a value by NAME, code prints the value.

Pure. The design every grounded writer shares (first built as the EDGE desk narrator, 2026-09):

* Code renders every value once, as text, under an id that holds no digits (`[a-z_:]+`):
  `{"entry_px": "2.20", "hold_min": "14"}`. The prompt shows the model `{entry_px}` and what it is.
* The output grammar forbids digits in every prose run (`prose_pattern` / `slot_pattern`), so the
  model can write `{entry_px}` and cannot write "2.20" -- or "2.4", or "22 more minutes".
* Code fills each slot with its rendered value (`fill`), then runs the belt over the FILLED text
  (`belt`): every number in it must be a rendered value. That count is 0 BY CONSTRUCTION; a
  non-zero count is an alarm, never a tolerated rate.

Telling a model "use only the numbers you were given" does not work: a 9B on a trade coach's prompt
did its own arithmetic and invented levels on ~60% of replies with the instruction in place (EDGE
trade.autopsy, 2026-10-27, 40 of 64). A grammar that has no digit token cannot.

Why a character WHITELIST and not `^[^0-9]*$` (probed on llama.cpp's json-schema-to-grammar,
2026-09-27): a pattern string is generated outside JSON string escaping, so `[^0-9]` let the model
emit raw control bytes that broke `json.loads`; the whitelist admits nothing that needs escaping.
`maxLength` is NOT enforced alongside `pattern` there, so a capped string carries its cap IN the
pattern as `{0,N}` (a slot counts as one repetition, so the grammar's cap is approximate; keep the
exact `max_length` on the pydantic field too). Groups are plain `(...)`: the converter's regex
subset has no `(?:`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from llm_ports import grounding, numbers

#: Every character a prose run may hold: no digits, nothing JSON must escape.
PROSE_CHARS = r"[A-Za-z ,.;:'()/&?!-]"
#: An unbounded words-only run. Never a capped field's pattern -- use `prose_pattern(n)`.
PROSE_PATTERN = rf"^{PROSE_CHARS}*$"
#: A slot id: no digits, so a grammar without digits can still write it.
ID = re.compile(r"^[a-z_:]+$")
#: A slot as the model writes it.
SLOT = re.compile(r"\{([a-z_:]+)\}")


def prose_pattern(n: int) -> str:
    """A words-only run of at most `n` characters, the cap inside the grammar (module doc)."""
    return rf"^{PROSE_CHARS}{{0,{n}}}$"


def slot_pattern(ids: Iterable[str], n: int) -> str:
    """Prose that may cite exactly these ids as `{id}` and nothing else, at most `n` repetitions
    per run. For one or two ids, each slot at most ONCE (a 9B wrote "Call wall {call_wall} falls
    below spot {call_wall}", and a retry returned the same line -- so the grammar holds the rule).
    Raises ValueError on an id a digit-free grammar could not write."""
    uniq = sorted(set(ids))
    bad = [i for i in uniq if not ID.match(i)]
    if bad:
        raise ValueError(f"slot ids must match {ID.pattern}: {bad}")
    if not uniq:
        return prose_pattern(n)
    c, s = f"{PROSE_CHARS}{{0,{n}}}", [re.escape("{" + i + "}") for i in uniq]
    if len(s) == 1:
        return rf"^({c}{s[0]})?{c}$"
    if len(s) == 2:
        return rf"^({c}{s[0]}({c}{s[1]})?|{c}{s[1]}({c}{s[0]})?)?{c}$"
    return rf"^({PROSE_CHARS}|{'|'.join(s)}){{0,{n}}}$"


def cited(text: str) -> list[str]:
    """The slot ids `text` cites, in order, repeats kept."""
    return SLOT.findall(text or "")


def fill(text: str, rendered: Mapping[str, str]) -> str:
    """Each `{id}` replaced by its rendered value; an unknown id is left as written (so `problems`
    and the belt still see it). The model's text is NEVER a template: one regex substitution over
    ids, nothing evaluated."""
    return SLOT.sub(lambda m: rendered.get(m.group(1), m.group(0)), text or "")


def problems(text: str, rendered: Mapping[str, str], *, once: bool = True) -> list[str]:
    """What is wrong with one prose run BEFORE it is filled, said so a retry can fix it: a slot
    that names no value, a brace that is not a slot, and (with `once`) a value cited twice -- a
    figure pasted where it means something else."""
    ids = cited(text)
    out = [f"unknown slot {{{i}}}" for i in dict.fromkeys(ids) if i not in rendered]
    if "{" in SLOT.sub("", text or "") or "}" in SLOT.sub("", text or ""):
        out.append("a stray brace; write a slot as {id} or not at all")
    twice = sorted({i for i in ids if ids.count(i) > 1})
    if once and twice:
        out.append(f"cites {{{twice[0]}}} twice; cite each value once, next to its name")
    return out


def loose(text: str, rendered: Mapping[str, str]) -> list[str]:
    """The figures in `text` that no rendered value shows, digits or spelled out, in order
    (`grounding.loose`). A spelled word is licensed by any value that renders it (values are
    joined on newlines, and a number word holds no whitespace). Over the UNFILLED text it names
    what the model wrote around the grammar (a spelled number); over the FILLED text it is the
    belt's evidence."""
    values = list(rendered.values())
    return grounding.loose(text, grounding.allowed_numbers(values), "\n".join(values))


def belt(text: str, rendered: Mapping[str, str]) -> int:
    """How many figures in FILLED `text` no rendered value shows (`loose`, counted). Must be 0."""
    return len(loose(text, rendered))


#: A character no prose run may hold: the complement of PROSE_CHARS, derived from it so the two
#: cannot drift (a test pins that every admitted character survives `to_prose`).
_NOT_PROSE = re.compile(PROSE_CHARS.replace("[", "[^", 1))


def to_prose(text: str) -> str:
    """`text` as CONTEXT a grounded writer may read: every digit run removed WHOLE
    (`numbers.DIGITS`, so `14:00`, `3.1%` and `5,000` leave no half a number behind), every
    character outside PROSE_CHARS a space, whitespace collapsed.

    Why: the grammar forbids the model a digit, but the prompt's context in WORDS (an operator's
    entry note, a tool's reason string, a JSON key) still shows it figures -- and a 9B at low
    temperature reaches for a figure it can SEE. With the digit token masked, the top admissible
    token is the tail of a number word, and with `repeat_penalty 1.0` it emits that tail to the
    length cap (EDGE trade.autopsy, 2026-10-30: ` ight` x80, 47 of 105 rows; with the context
    sanitised, 0 of 3 replays looped). A figure stays citable by its id; the digits the model must
    not copy are gone before it reads. Pure. Never for a rendered VALUE -- code fills those."""
    out = numbers.DIGITS.sub(" ", text or "")
    return " ".join(_NOT_PROSE.sub(" ", out).split())


__all__ = ["ID", "PROSE_CHARS", "PROSE_PATTERN", "SLOT", "belt", "cited", "fill", "loose", "problems",
           "prose_pattern", "slot_pattern", "to_prose"]
