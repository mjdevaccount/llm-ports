"""Deterministic graders over one generation: the prompt in, the value out.

Pure. A `CallGrader` is `(prompt, value, role) -> CallScore`: a numeric value (what a trace score
and a dashboard can sum), a pass or fail against the grader's own bar, and a reason that names the
violated property on failure. `call.generate` runs the graders a `Role.graders` names on every
`ok` reply and writes each as a score on the call's trace observation. A golden eval runs the same
functions over a pack's golden set (`board_graders`), so the live score and the offline verdict
agree on what "complete" or "echoed" means.

Built-in graders, for structured outputs (a dict or pydantic value under a grammar):

- `schema_complete`: the share of fields that carry a value. A strict grammar admits `""` for
  every required key, so a parsed reply is not necessarily an answered one. An empty list is an
  answer; only None and blank text count as unanswered. On a pydantic value, None on a field whose
  annotation admits None (`float | None`, `Optional[X]`) is an answer too, while None on a required
  field fails. A dict value has no annotations to read, so every None in it is unanswered.
- `numbers_from_inputs`: figures in the reply's prose that no part of the prompt rendered, digit
  runs by value and spelled numbers alike (`numbers.loose`): the number law, per call.
- `echo`: the share of the reply's sentences copied verbatim from the prompt.

A consumer adds its own graders by passing a table (`{**GRADERS, "mine": fn}`) to `score_all`,
`board_graders` or `call.Hooks(graders=...)`. This package never mutates `GRADERS`.

Invariants:
    - `score_one` and `score_all` never raise. A grader's exception becomes a failed score of value
      `CRASHED` (-1, below every honest value, all of which are >= 0) whose reason names the
      exception, so a crashed grader never reads as a pass.

Rationale: ADR-0003.
"""

from __future__ import annotations

import re
import types
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel

from llm_ports.call.roles import Role
from llm_ports.grounding.numbers import digits, loose

ECHO_MAX_SHARE = 0.5  # `echo` fails at or above this share of copied sentences
ECHO_MIN_CHARS = 40  # a sentence shorter than this is a label ("unclear", a key), not prose
CRASHED = -1.0  # the value of a grader that raised; every honest value is >= 0
_SENTENCE = re.compile(r"[.!?\n]+")
_WS = re.compile(r"\s+")
_INDEX = re.compile(r"\[\d+\]")  # `items[3].price` -> `items[].price`: the field, not the row


@dataclass(frozen=True)
class CallScore:
    """One grader's verdict on one reply.

    Attributes:
        name: The grader's name.
        value: The numeric score; `CRASHED` when the grader raised.
        passed: Whether the value meets the grader's own bar.
        reason: Empty on a pass; names the violated property on a failure.
    """

    name: str
    value: float
    passed: bool
    reason: str = ""


CallGrader = Callable[[str, Any, Role], CallScore]


def leaves(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Yield `(path, scalar)` for every leaf of a dict, pydantic or list value, in order.

    Pure. An empty list is itself a leaf.
    """
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        value = dump()
    if isinstance(value, Mapping):
        for k, v in value.items():
            yield from leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(value, (list, tuple)):
        if not value:
            yield path, value  # an empty list is a leaf: the field answered nothing
        for i, v in enumerate(value):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, value


def prose(value: Any) -> list[str]:
    """Return every string leaf of the reply: what the number law and the echo check read."""
    return [v for _p, v in leaves(value) if isinstance(v, str)]


def _empty(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def _admits_none(ann: Any) -> bool:
    """Return whether the annotation accepts None (`X | None`, `Optional[X]`, `Annotated[X | None, ...]`)."""
    if ann is None or ann is type(None):
        return True
    origin = get_origin(ann)
    if origin is Annotated:
        return _admits_none(get_args(ann)[0])
    return origin in (Union, types.UnionType) and any(_admits_none(a) for a in get_args(ann))


def _models_in(ann: Any, suffix: str = "") -> Iterator[tuple[str, type[BaseModel]]]:
    """Yield `(path suffix, model)` for every `BaseModel` an annotation reaches.

    The suffix is `""` for the model itself or through a union, plus `[]` per list, tuple or set
    layer: the spelling `leaves` emits, without indexes.
    """
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
    """Return the index-free leaf paths of every field whose annotation admits None.

    Pure. Paths (`items[].price`) run through nested models and lists of models. They are read off
    the class, so they never depend on which rows a reply carried. A self-referential model stops at
    its own repeat.
    """
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
    """Score the share of the reply's fields that carry a value.

    Passes only when no field is unanswered. None and blank text are unanswered; an empty list is an
    answer, and so is None on a pydantic field whose annotation admits None.
    """
    fields = list(leaves(value))
    if not fields:
        return CallScore("schema_complete", 0.0, False, "the reply carries no fields")
    nullable = nullable_paths(type(value)) if isinstance(value, BaseModel) else frozenset()
    empty = [p for p, v in fields if _empty(v) and _INDEX.sub("[]", p) not in nullable]
    share = round(1 - len(empty) / len(fields), 4)
    return CallScore("schema_complete", share, not empty, f"empty: {empty[:4]}" if empty else "")


def numbers_from_inputs(prompt: str, value: Any, _role: Role) -> CallScore:
    """Count the distinct figures in the reply's prose that the prompt did not render.

    Digit runs compare by value, and spelled numbers count too (`numbers.loose`). The prompt's own
    text licenses a spelled word, and '%' licenses 'percent'. Passes only at zero.

    Rationale: ADR-0003.
    """
    have = digits(prompt)
    hits = sorted({n for s in prose(value) for n in loose(s, have, prompt)})
    return CallScore(
        "numbers_from_inputs", float(len(hits)), not hits, f"numbers no input rendered: {hits[:3]}" if hits else ""
    )


def _sentences(text: str) -> list[str]:
    return [s for s in (_WS.sub(" ", p).strip().lower() for p in _SENTENCE.split(text)) if len(s) >= ECHO_MIN_CHARS]


def echo(prompt: str, value: Any, _role: Role) -> CallScore:
    """Score the share of the reply's sentences copied verbatim from the prompt.

    The compare ignores case and whitespace, and a sentence shorter than `ECHO_MIN_CHARS` is not
    counted; a reply with no counted sentence passes at 0. Fails at or above `ECHO_MAX_SHARE`.
    """
    hay = _WS.sub(" ", prompt).lower()
    sents = [s for p in prose(value) for s in _sentences(p)]
    if not sents:
        return CallScore("echo", 0.0, True)
    copied = [s for s in sents if s in hay]
    share = round(len(copied) / len(sents), 4)
    return CallScore(
        "echo",
        share,
        share < ECHO_MAX_SHARE,
        f"{len(copied)}/{len(sents)} sentences copied from the prompt: {copied[0][:60]!r}"
        if share >= ECHO_MAX_SHARE
        else "",
    )


#: The built-in graders by name; a `Role` names its graders as strings (data, comparable, printable).
GRADERS: dict[str, CallGrader] = {
    "schema_complete": schema_complete,
    "numbers_from_inputs": numbers_from_inputs,
    "echo": echo,
}


#: Graders whose verdict holds only when the prompt text is the call's whole input. A grader sees
#: `(prompt, value, role)`, so an image sent beside the prompt is invisible to it, as are the pages
#: a web-fetching tier read. `numbers_from_inputs` fails every honest reply on such a call: a chart
#: read's figures come off the image, a research reply's off the pages. `call.generate` drops these
#: for any call with `images=` or `web=True`; a text-only call keeps every grader its role names.
TEXT_ONLY_GRADERS: frozenset[str] = frozenset({"numbers_from_inputs"})


def graders_for_inputs(graders: tuple[str, ...], *, has_images: bool, has_web: bool = False) -> tuple[str, ...]:
    """Return the graders that can see this call's inputs, in their given order.

    Pure. A text-only call keeps all of them; an image beside the prompt or a web-fetching tier
    removes `TEXT_ONLY_GRADERS`.
    """
    if has_images or has_web:
        return tuple(g for g in graders if g not in TEXT_ONLY_GRADERS)
    return graders


def score_one(
    name: str, prompt: str, value: Any, role: Role, table: Mapping[str, CallGrader] | None = None
) -> CallScore:
    """Run one grader by name from `table` (default `GRADERS`).

    Never raises: an unknown name or a grader exception becomes a failed score of value `CRASHED`.
    """
    fn = (GRADERS if table is None else table).get(name)
    if fn is None:
        return CallScore(name, CRASHED, False, f"no grader named {name!r}")
    try:
        return fn(prompt, value, role)
    except Exception as exc:  # a crashed grader must not read as a pass
        return CallScore(name, CRASHED, False, f"grader raised {type(exc).__name__}: {exc}"[:200])


def score_all(
    role: Role, prompt: str, value: Any, table: Mapping[str, CallGrader] | None = None
) -> tuple[CallScore, ...]:
    """Run every grader the role names, in the role's order; never raises."""
    return tuple(score_one(n, prompt, value, role, table) for n in role.graders)


def board_graders(
    role: Role,
    table: Mapping[str, CallGrader] | None = None,
) -> dict[str, Callable[[dict[str, Any]], tuple[bool, str]]]:
    """Return the same graders in the `Evaluator` row shape (`ports.Grader`).

    A golden row carries `prompt` and `value`. A row whose model never answered (`value is None`)
    fails every grader with its `reason`, so the board names the outage rather than grading an
    absence.
    """

    def bind(name: str) -> Callable[[dict[str, Any]], tuple[bool, str]]:
        def grader(row: dict[str, Any]) -> tuple[bool, str]:
            if row.get("value") is None:
                return False, f"no reply: {row.get('reason') or row.get('outcome')}"
            s = score_one(name, str(row.get("prompt") or ""), row["value"], role, table)
            return s.passed, s.reason

        return grader

    return {n: bind(n) for n in role.graders}


__all__ = [
    "CRASHED",
    "ECHO_MAX_SHARE",
    "ECHO_MIN_CHARS",
    "GRADERS",
    "TEXT_ONLY_GRADERS",
    "CallGrader",
    "CallScore",
    "board_graders",
    "echo",
    "graders_for_inputs",
    "leaves",
    "nullable_paths",
    "numbers_from_inputs",
    "prose",
    "schema_complete",
    "score_all",
    "score_one",
]
