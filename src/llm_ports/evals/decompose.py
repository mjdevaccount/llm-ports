"""A reply decomposed into claims and pre-matched against its Input, in code, before any judge reads it.

The pure half of the claim judge (`evals.claim_judge`, ADR-0031):

- `leaves`: each scalar of a JSON reply is one claim at its path (`items[0].name`); prose is one
  claim per sentence (`sentences`). No model decides what a claim is.
- `facts_from_text`: the Input's lines numbered as facts (`F1`, `F2`, ...), so a verdict can cite
  what it checked.
- `prematch`: what code can settle. A value that occurs in the Input as whole words (`occurs`) is
  `supported`, and so is a bare figure the number law licenses. Code rules only in that
  direction: a value it cannot find is left to the judge, with what code saw as a hint.
- `enums`: each field's allowed values from the reply's JSON schema.

Invariants:
    - Pure: no model, no clock, no I/O. Never raises on data: a reply that is not JSON is prose.
    - Code never rules a claim ungrounded.

Rationale: ADR-0031.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from llm_ports.evals.claims import ClaimLabel, Fact
from llm_ports.grounding import numbers

Kind = Literal["quote", "extract", "classify", "free"]

#: The field kinds: a `quote` copies the Input, an `extract` names something it states, a `classify`
#: value is chosen from a closed set (a score is one too), `free` is anything else (prose, a value
#: the model worked out, an identifier whose presence alone does not ground it).
KINDS: tuple[Kind, ...] = ("quote", "extract", "classify", "free")
#: Characters of one fact line `facts_from_text` keeps before it splits the line.
FACT_CHARS = 280
#: A value this short is matched case-sensitively: a short token is a symbol, and "ai" is in "said".
SHORT = 5

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")
_INDEX = re.compile(r"\[\d+\]")
_FIGURE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?")
_QUOTES = str.maketrans({0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"'})  # curly quotes made straight
_DEPTH = 12
_VERBATIM = frozenset({"quote", "extract"})


@dataclass(frozen=True)
class Leaf:
    """One claim of a reply, as code reads it.

    Attributes:
        path: Where it sits: a JSON path (`items[0].name`), or `sentence[<n>]` for prose.
        value: The claim as text: the scalar rendered, or the sentence.
        raw: The JSON scalar (None, a bool, a number or a string); for an empty list or object, that
            empty container; for a sentence, the sentence.
    """

    path: str
    value: str
    raw: Any

    @property
    def field(self) -> str:
        """The path with its list indices dropped (`items[].name`), the key a field map reads."""
        return _INDEX.sub("[]", self.path)

    @property
    def empty(self) -> bool:
        """True when the claim states that nothing applies (null, empty string, list or object)."""
        return self.raw is None or self.raw == "" or self.raw == [] or self.raw == {}


def _walk(value: Any, path: str, out: list[Leaf]) -> None:
    if isinstance(value, Mapping) and value:
        for k, v in value.items():
            _walk(v, f"{path}.{k}" if path else str(k), out)
    elif isinstance(value, list) and value:
        for i, v in enumerate(value):
            _walk(v, f"{path}[{i}]", out)
    else:
        out.append(Leaf(path or "$", value if isinstance(value, str) else json.dumps(value), value))


def sentences(text: str) -> list[str]:
    """Split prose into sentences: on line breaks, then after `.`, `!` or `?` before a capital or digit.

    A decimal point has no space after it, so `2.5` is never a boundary. Pure.
    """
    out: list[str] = []
    for line in (text or "").splitlines():
        out += [s.strip() for s in _SENTENCE.split(line) if s.strip()]
    return out


def leaves(reply: str) -> tuple[Leaf, ...]:
    """Decompose a reply into claims in code.

    A JSON object or array yields one `Leaf` per scalar (an empty list or object is one leaf
    stating that nothing applies); anything else is prose, one leaf per sentence.

    Args:
        reply: The reply as the judge reads it (`judge.reply_text`).

    Returns:
        The leaves in reply order. Never raises.
    """
    try:
        parsed = json.loads(reply)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, (Mapping, list)):
        out: list[Leaf] = []
        _walk(parsed, "", out)
        return tuple(out)
    return tuple(Leaf(f"sentence[{i}]", s, s) for i, s in enumerate(sentences(reply)))


def _wrap(line: str, cap: int) -> list[str]:
    if len(line) <= cap:
        return [line]
    parts = sentences(line)
    out: list[str] = []
    cur = ""
    for part in parts if len(parts) > 1 else line.split(" "):
        joined = f"{cur} {part}".strip()
        if cur and len(joined) > cap:
            out.append(cur)
            cur = part
        else:
            cur = joined
    return [*out, cur] if cur else out


def facts_from_text(text: str, *, prefix: str = "F", cap: int = FACT_CHARS) -> tuple[Fact, ...]:
    """Number the Input's lines as facts a verdict can cite.

    Blank lines and repeats of an earlier line are dropped; a line over `cap` characters is split at
    sentence ends, else at spaces. Ids run `F1`, `F2`, ... in Input order.

    Args:
        text: The Input as text.
        prefix: The id prefix.
        cap: The longest fact line kept whole.

    Returns:
        The facts. Pure; never raises.
    """
    seen: set[str] = set()
    out: list[Fact] = []
    for raw in (text or "").splitlines():
        line = " ".join(raw.split())
        if not line or line in seen:
            continue
        seen.add(line)
        for part in _wrap(line, cap):
            out.append(Fact(f"{prefix}{len(out) + 1}", part))
    return tuple(out)


def norm(text: str) -> str:
    """Return `text` case-folded, curly quotes made straight, whitespace collapsed: the match key."""
    return " ".join(str(text).translate(_QUOTES).casefold().split())


def occurs(value: str, text: str) -> bool:
    """Return whether `value` occurs in `text` as whole words (whitespace collapsed).

    Matched case-insensitively, except a value of at most `SHORT` characters, which is matched as
    written: a short token is a symbol. Whole words, so a value never matches inside a longer word.
    Pure.
    """
    v = " ".join(str(value).translate(_QUOTES).split())
    if not v:
        return False
    hay = " ".join(str(text).translate(_QUOTES).split())
    if len(v) > SHORT:
        v, hay = v.casefold(), hay.casefold()
    return re.search(rf"(?<!\w){re.escape(v)}(?!\w)", hay) is not None


def _bare_figure(leaf: Leaf) -> bool:
    if isinstance(leaf.raw, bool):
        return False
    if isinstance(leaf.raw, (int, float)):
        return True
    return isinstance(leaf.raw, str) and _FIGURE.fullmatch(leaf.raw.strip()) is not None


@dataclass(frozen=True)
class Prematch:
    """Code's reading of one claim before any judge sees it.

    Attributes:
        label: `supported` when code found the claim in the Input; None when it leaves it to the judge.
        reason: One line naming what code found.
        hints: What code checked and could not rule on, for the judge's prompt.
    """

    label: ClaimLabel | None
    reason: str
    hints: tuple[str, ...] = ()

    @property
    def ruled(self) -> bool:
        """True when code settled the claim."""
        return self.label is not None


def prematch(leaf: Leaf, kind: Kind, *, input_text: str, allowed: Collection[str] = ()) -> Prematch:
    """Settle what code can about one claim; leave the rest to the judge with hints.

    - A `quote` or `extract` value that occurs in the Input as whole words (`occurs`) is
      `supported`. A `free` value is not: an id or a phrase copied from the Input can still be
      used to claim what the Input does not say, so finding it is a hint.
    - A bare figure (a number, or a string that is only a number) in any field but a `classify`
      one that the number law licenses (`numbers.loose`, rerendered) is `supported`.
    - Nothing else is ruled. A `classify` value is a choice, so finding it in the Input proves only
      that it is an allowed value: that is a hint, never a ruling.
    - A consumer with a stronger test of its own (a header listing the names to extract, say)
      runs it first (`claim_judge.grade(pre=...)`).

    Args:
        leaf: The claim.
        kind: Its field kind.
        input_text: The Input the reply was written from.
        allowed: The field's allowed values (a grammar enum), if it has any.

    Returns:
        The `Prematch`. Pure; never raises.
    """
    hints: list[str] = []
    found = not leaf.empty and occurs(leaf.value, input_text)
    if kind != "classify" and _bare_figure(leaf):
        refused = numbers.loose(leaf.value, numbers.digits(input_text), input_text.lower(), rerender=True)
        if not refused:
            return Prematch("supported", "the figure is in the Input (number law)")
        hints.append("figure(s) the Input does not show as written: " + ", ".join(refused[:4]))
    elif kind in _VERBATIM and found:
        return Prematch("supported", "the value occurs in the Input verbatim")
    elif kind != "classify" and not leaf.empty:
        hints.append("the value occurs in the Input verbatim" if found else "the value is not in the Input verbatim")
    if allowed and str(leaf.raw) in {str(a) for a in allowed}:
        hints.append("the value is one of the field's allowed values")
    elif kind == "classify" and found:
        hints.append("the value's words occur in the Input")
    return Prematch(None, "left to the judge", tuple(hints))


def enums(schema: Mapping[str, Any] | None) -> dict[str, tuple[str, ...]]:
    """Return each field's allowed values from a JSON schema, keyed by `Leaf.field` (`items[].kind`).

    Reads `enum` and `const`, follows local `$ref`s into `$defs`/`definitions`, and descends
    `properties`, `items` and `anyOf`/`oneOf`/`allOf`. Pure; a schema it cannot read yields `{}`.
    """
    out: dict[str, tuple[str, ...]] = {}
    if not isinstance(schema, Mapping):
        return out
    defs: dict[str, Any] = {**(schema.get("definitions") or {}), **(schema.get("$defs") or {})}

    def visit(node: Any, path: str, depth: int) -> None:
        if not isinstance(node, Mapping) or depth > _DEPTH:
            return
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.rsplit("/", 1)[-1] in defs:
            visit(defs[ref.rsplit("/", 1)[-1]], path, depth + 1)
        vals = [*(node.get("enum") or ()), *([node["const"]] if "const" in node else [])]
        key = path or "$"
        if vals:
            out[key] = tuple(dict.fromkeys([*out.get(key, ()), *(str(v) for v in vals if v is not None)]))
        for k, sub in (node.get("properties") or {}).items():
            visit(sub, f"{path}.{k}" if path else str(k), depth + 1)
        if isinstance(node.get("items"), Mapping):
            visit(node["items"], f"{path}[]", depth + 1)
        for comb in ("anyOf", "oneOf", "allOf"):
            for sub in node.get(comb) or ():
                visit(sub, path, depth + 1)

    visit(schema, "", 0)
    return out


__all__ = [
    "FACT_CHARS",
    "KINDS",
    "SHORT",
    "Kind",
    "Leaf",
    "Prematch",
    "enums",
    "facts_from_text",
    "leaves",
    "norm",
    "occurs",
    "prematch",
    "sentences",
]
