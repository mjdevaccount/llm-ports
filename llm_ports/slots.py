"""Grounded prose by construction: the model cites a value by name, and code prints the value.

1. Code renders every value once, as text, under an id that holds no digits (`[a-z_:]+`):
   `{"total": "1,284.50", "count": "3"}`. The prompt shows the model `{total}` and what it means.
2. The output grammar forbids digits in every prose run (`prose_pattern`, `slot_pattern`), so the
   model can write `{total}` and cannot write "1,284.50", "1,300" or "three more".
3. Code fills each slot with its rendered value (`fill`), then runs the belt over the filled text
   (`belt`): every number in it must be a rendered value.

The grammar is a character whitelist (`PROSE_CHARS`), not a negated class such as `[^0-9]`.
Compatibility facts for llama.cpp's json-schema-to-grammar converter:

- A pattern string is compiled outside JSON string escaping, so a negated class admits raw
  control bytes that break `json.loads`. The whitelist admits nothing that needs escaping.
- `maxLength` is not enforced alongside `pattern`, so a capped run carries its cap in the pattern
  as `{0,N}`. A slot counts as one repetition, so the grammar's cap is approximate; keep the exact
  `max_length` on the pydantic field too.
- The converter's regex subset has no `(?:`, so groups are plain `(...)`.

Invariants:
    - Pure: no I/O.
    - Over filled text, `belt` is 0 by construction. A non-zero count is an alarm, never a rate.

Rationale: ADR-0002, ADR-0005.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from llm_ports import grounding, numbers

#: Every character a prose run may hold: no digits, nothing JSON must escape.
PROSE_CHARS = r"[A-Za-z ,.;:'()/&?!-]"
#: An unbounded words-only run. Never a capped field's pattern; use `prose_pattern(n)`.
PROSE_PATTERN = rf"^{PROSE_CHARS}*$"
#: A slot id: no digits, so a grammar without digits can still write it.
ID = re.compile(r"^[a-z_:]+$")
#: A slot as the model writes it.
SLOT = re.compile(r"\{([a-z_:]+)\}")


def prose_pattern(n: int) -> str:
    """Return the pattern for a words-only run of at most `n` characters, the cap inside the grammar."""
    return rf"^{PROSE_CHARS}{{0,{n}}}$"


def slot_pattern(ids: Iterable[str], n: int) -> str:
    """Return the pattern for prose that may cite exactly these ids as `{id}`, and nothing else.

    Args:
        ids: The slot ids the run may cite; duplicates are ignored.
        n: The cap, in repetitions per run (a slot counts as one).

    Returns:
        A pattern. With one or two ids, each slot may appear at most once, so the grammar itself
        stops a value being cited where it means something else. With none, `prose_pattern(n)`.

    Raises:
        ValueError: An id a digit-free grammar could not write (it does not match `ID`).
    """
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
    """Return the slot ids `text` cites, in order, repeats kept."""
    return SLOT.findall(text or "")


def fill(text: str, rendered: Mapping[str, str]) -> str:
    """Replace each `{id}` in `text` with its rendered value.

    An unknown id is left as written, so `problems` and the belt still see it.

    Invariants:
        - The model's text is never a template: one regex substitution over ids, nothing evaluated.
    """
    return SLOT.sub(lambda m: rendered.get(m.group(1), m.group(0)), text or "")


def problems(text: str, rendered: Mapping[str, str], *, once: bool = True) -> list[str]:
    """Return what is wrong with one prose run before it is filled, worded so a retry can fix it.

    Args:
        text: The unfilled run.
        rendered: The rendered values by id.
        once: Also report a value cited twice (a figure pasted where it means something else).

    Returns:
        One message per problem: a slot that names no value, a brace that is not a slot, a repeat.
        Empty when the run is clean.
    """
    ids = cited(text)
    out = [f"unknown slot {{{i}}}" for i in dict.fromkeys(ids) if i not in rendered]
    if "{" in SLOT.sub("", text or "") or "}" in SLOT.sub("", text or ""):
        out.append("a stray brace; write a slot as {id} or not at all")
    twice = sorted({i for i in ids if ids.count(i) > 1})
    if once and twice:
        out.append(f"cites {{{twice[0]}}} twice; cite each value once, next to its name")
    return out


def loose(text: str, rendered: Mapping[str, str]) -> list[str]:
    """Return the figures in `text` that no rendered value shows, digits or spelled out, in order.

    A thin wrapper over `grounding.loose`: a spelled word is licensed by any value that renders it.
    Over unfilled text it names what the model wrote around the grammar (a spelled number); over
    filled text it is the belt's evidence.
    """
    values = list(rendered.values())
    return grounding.loose(text, grounding.allowed_numbers(values), "\n".join(values))


def belt(text: str, rendered: Mapping[str, str]) -> int:
    """Return how many figures in filled `text` no rendered value shows (`loose`, counted).

    Invariants:
        - 0 for any text built by `fill` from a run the grammar admitted.
    """
    return len(loose(text, rendered))


#: A character no prose run may hold: the complement of `PROSE_CHARS`, derived from it so the two
#: cannot drift (a test pins that every admitted character survives `to_prose`).
_NOT_PROSE = re.compile(PROSE_CHARS.replace("[", "[^", 1))


def to_prose(text: str) -> str:
    """Return `text` as context a grounded writer may read.

    Every digit run is removed whole (`numbers.DIGITS`, so `14:00`, `3.1%` and `5,000` leave no
    half a number behind), every character outside `PROSE_CHARS` becomes a space, and whitespace
    collapses. A figure stays citable by its slot id; the digits the model must not copy are gone
    before it reads them. Use it on free text placed in the prompt (a user's note, a tool's reason
    string, a JSON key), never on a rendered value: code fills those.

    Rationale: ADR-0004.
    """
    out = numbers.DIGITS.sub(" ", text or "")
    return " ".join(_NOT_PROSE.sub(" ", out).split())


__all__ = [
    "ID",
    "PROSE_CHARS",
    "PROSE_PATTERN",
    "SLOT",
    "belt",
    "cited",
    "fill",
    "loose",
    "problems",
    "prose_pattern",
    "slot_pattern",
    "to_prose",
]
