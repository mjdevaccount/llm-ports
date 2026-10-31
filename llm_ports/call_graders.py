"""call_graders -- deterministic graders over ONE generation: the prompt in, the value out.

Pure. A `CallGrader` is `(prompt, value, role) -> CallScore`: a NUMERIC value (what a trace score
and a dashboard can sum), a pass/fail against the grader's own bar, and a reason that names the
violated property on failure. `call.generate` runs the graders a `Role.graders` names on every
`ok` reply and writes each as a score on the call's trace observation; a golden eval runs the SAME
functions over a pack's golden set (`board_graders`), so the live score and the offline verdict
can never disagree about what "complete" or "echoed" means.

Three built-in graders, for the `structured` family (dict or pydantic outputs under a grammar):

  schema_complete      share of fields that carry a value. A strict grammar admits `""` for every
                       required key, so "parsed" is not "answered". An EMPTY LIST is an answer
                       (prompts ask for `[]` when nothing applies), so only None and blank text
                       count as unanswered -- and for a PYDANTIC value, a None on a field whose
                       annotation admits None (`float | None`, `Optional[X]`) is an answer too:
                       the schema declared "unknown" legal and the prompt ordered null when the
                       source never stated the figure, so an honest reply must not fail by
                       construction. A None on a REQUIRED field still fails; a dict value (no
                       annotations to read) is graded exactly as before.
  numbers_from_inputs  figures in the reply's prose that no part of the prompt rendered: digit
                       runs by value AND spelled numbers (`numbers.loose`, the one number law;
                       a digit-free grammar made models SPELL the figure, so a digits-only
                       grader passed what the writers' belts refused). The number law, per call.
  echo                 share of the reply's sentences copied verbatim from the prompt. A model
                       that hands the evidence back is not analysing it.

A consumer adds its OWN graders by passing a table (`{**GRADERS, "mine": fn}`) to `score_all` /
`board_graders` / `call.Hooks(graders=...)`; `GRADERS` itself is never mutated by this package.

NEVER RAISES: `score_all` turns a grader's exception into a FAILED score of value -1 (below every
honest value, all of which are >= 0) whose reason names the exception -- a crashed detector
must not read as a pass.
"""

from __future__ import annotations

import re
import types
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel

from llm_ports.numbers import digits, loose
from llm_ports.roles import Role

ECHO_MAX_SHARE = 0.5          # bar for `echo`: half the sentences copied is a reply, not an analysis
ECHO_MIN_CHARS = 40           # a sentence shorter than this is a label ("unclear", a key), not prose
CRASHED = -1.0                # the value of a grader that raised; every honest value is >= 0
_SENTENCE = re.compile(r"[.!?\n]+")
_WS = re.compile(r"\s+")
_INDEX = re.compile(r"\[\d+\]")   # `tickers[3].entry_low` -> `tickers[].entry_low`: the field, not the row


@dataclass(frozen=True)
class CallScore:
    name: str
    value: float
    passed: bool
    reason: str = ""          # empty on pass; names the violated property on failure


CallGrader = Callable[[str, Any, Role], CallScore]


def leaves(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """(path, scalar) for every leaf of a dict / pydantic / list value, in order. Pure."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        value = dump()
    if isinstance(value, Mapping):
        for k, v in value.items():
            yield from leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(value, (list, tuple)):
        if not value:
            yield path, value          # an empty list IS a leaf: the field answered nothing
        for i, v in enumerate(value):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, value


def prose(value: Any) -> list[str]:
    """Every string leaf of the reply: what the number law and the echo check read."""
    return [v for _p, v in leaves(value) if isinstance(v, str)]


def _empty(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def _admits_none(ann: Any) -> bool:
    """Does the annotation accept None (`X | None`, `Optional[X]`, `Annotated[X | None, ...]`)?"""
    if ann is None or ann is type(None):
        return True
    origin = get_origin(ann)
    if origin is Annotated:
        return _admits_none(get_args(ann)[0])
    return origin in (Union, types.UnionType) and any(_admits_none(a) for a in get_args(ann))


def _models_in(ann: Any, suffix: str = "") -> Iterator[tuple[str, type[BaseModel]]]:
    """(path suffix, model) for every BaseModel an annotation reaches: `""` for the model itself
    or through a union, `[]` per list/tuple/set layer -- the spelling `leaves` emits, index-free."""
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        yield suffix, ann
        return
    origin, args = get_origin(ann), get_args(ann)
    if origin is Annotated or origin in (Union, types.UnionType):
        for a in args[:1] if origin is Annotated else args:
            yield from _models_in(a, suffix)
    elif isinstance(origin, type) and issubclass(origin, (list, tuple, set, frozenset)):
        for a in args:
            if a is not Ellipsis:
                yield from _models_in(a, suffix + "[]")


def nullable_paths(model: type[BaseModel], prefix: str = "", _seen: frozenset[type] = frozenset()) -> frozenset[str]:
    """Index-free leaf paths (`tickers[].entry_low`) of every field whose annotation admits None,
    through nested models and lists of models. Read off the class, so it never depends on which
    rows a reply happened to carry. A self-referential model stops at its own repeat. Pure."""
    out: set[str] = set()
    seen = _seen | {model}
    for name, field in model.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        if _admits_none(field.annotation):
            out.add(path)
        for suffix, sub in _models_in(field.annotation):
            if sub not in seen:
                out |= nullable_paths(sub, path + suffix, seen)
    return frozenset(out)


def schema_complete(_prompt: str, value: Any, _role: Role) -> CallScore:
    fields = list(leaves(value))
    if not fields:
        return CallScore("schema_complete", 0.0, False, "the reply carries no fields")
    nullable = nullable_paths(type(value)) if isinstance(value, BaseModel) else frozenset()
    empty = [p for p, v in fields if _empty(v) and _INDEX.sub("[]", p) not in nullable]
    share = round(1 - len(empty) / len(fields), 4)
    return CallScore("schema_complete", share, not empty, f"empty: {empty[:4]}" if empty else "")


def numbers_from_inputs(prompt: str, value: Any, _role: Role) -> CallScore:
    """Distinct figures (digit runs by value, spelled numbers) in the reply's prose that the prompt
    did not render; the prompt's own text licenses a spelled word and '%' licenses 'percent'."""
    have = digits(prompt)
    hits = sorted({n for s in prose(value) for n in loose(s, have, prompt)})
    return CallScore("numbers_from_inputs", float(len(hits)), not hits,
                     f"numbers no input rendered: {hits[:3]}" if hits else "")


def _sentences(text: str) -> list[str]:
    return [s for s in (_WS.sub(" ", p).strip().lower() for p in _SENTENCE.split(text)) if len(s) >= ECHO_MIN_CHARS]


def echo(prompt: str, value: Any, _role: Role) -> CallScore:
    hay = _WS.sub(" ", prompt).lower()
    sents = [s for p in prose(value) for s in _sentences(p)]
    if not sents:
        return CallScore("echo", 0.0, True)
    copied = [s for s in sents if s in hay]
    share = round(len(copied) / len(sents), 4)
    return CallScore("echo", share, share < ECHO_MAX_SHARE,
                     f"{len(copied)}/{len(sents)} sentences copied from the prompt: {copied[0][:60]!r}"
                     if share >= ECHO_MAX_SHARE else "")


#: By name, because a Role names its graders as strings (data, comparable, printable).
GRADERS: dict[str, CallGrader] = {
    "schema_complete": schema_complete,
    "numbers_from_inputs": numbers_from_inputs,
    "echo": echo,
}


def score_one(name: str, prompt: str, value: Any, role: Role,
              table: Mapping[str, CallGrader] | None = None) -> CallScore:
    """One grader by name (from `table`, default `GRADERS`); an unknown name or a raise is a
    FAILED score, never an exception."""
    fn = (GRADERS if table is None else table).get(name)
    if fn is None:
        return CallScore(name, CRASHED, False, f"no grader named {name!r}")
    try:
        return fn(prompt, value, role)
    except Exception as exc:     # a crashed detector must not exit 0
        return CallScore(name, CRASHED, False, f"grader raised {type(exc).__name__}: {exc}"[:200])


def score_all(role: Role, prompt: str, value: Any,
              table: Mapping[str, CallGrader] | None = None) -> tuple[CallScore, ...]:
    """Every grader the role names, in the role's order. Never raises."""
    return tuple(score_one(n, prompt, value, role, table) for n in role.graders)


def board_graders(role: Role, table: Mapping[str, CallGrader] | None = None,
                  ) -> dict[str, Callable[[dict[str, Any]], tuple[bool, str]]]:
    """The same graders in the `Evaluator` row shape (`ports.Grader`): a golden row carries
    `prompt` and `value`; a row whose model never answered (`value is None`) fails every grader
    with its `reason`, so the board names the outage rather than grading an absence."""
    def bind(name: str) -> Callable[[dict[str, Any]], tuple[bool, str]]:
        def grader(row: dict[str, Any]) -> tuple[bool, str]:
            if row.get("value") is None:
                return False, f"no reply: {row.get('reason') or row.get('outcome')}"
            s = score_one(name, str(row.get("prompt") or ""), row["value"], role, table)
            return s.passed, s.reason
        return grader
    return {n: bind(n) for n in role.graders}


__all__ = ["CRASHED", "ECHO_MAX_SHARE", "GRADERS", "CallGrader", "CallScore", "board_graders",
           "echo", "leaves", "nullable_paths", "numbers_from_inputs", "prose", "schema_complete",
           "score_all", "score_one"]
