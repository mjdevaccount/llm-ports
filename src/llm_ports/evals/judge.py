"""A sampled model judge: what it reads, which reply it draws, how it grades, and the lane that runs it.

- `Seen` / `seen_for`: what the judged call put in front of its model beside the prompt (system
  prompt, output grammar, images, a web tier). The judge reads the same.
- `judge_input`: the judge's Input text, each part under a heading the rubric names.
- `reply_text`: the reply as the judge reads it.
- `sample_key` / `selected`: a deterministic stamp of one generation and a 1-in-N draw on it.
- `rubric`: the four laws the deterministic graders (`call.graders`) encode, stated for a model:
  schema answered honestly, no number absent from the Input (`numbers.law_text`), no echo, grounded.
- `verdict_label`: `pass`, `fail` or `could_not_run`.
- `JudgeLane`: schedules the judging of a drawn reply as a background task, off the caller's await
  path; its `start` is the body of a `call.Hooks.on_ok` hook. `Sample` is what one task judges.

The consumer passes in, and this module never assumes: whether a role is eligible for judging,
the role's grounding mode, the platform named in the rubric, today's date with its clock label,
and any domain words the number law should name (`numbers.LawExamples`). Where a verdict is stored
is the consumer's concern.

Invariants:
    - Everything but `JudgeLane` is pure: no model, no database, no clock.
    - `JudgeLane.start` never raises and never awaits; a failure inside it is an event.
    - A draw is a function of (role, prompt, reply), so whether a reply was judged is answerable
      after the fact.
"""

from __future__ import annotations

import asyncio
import contextvars
import datetime as dt
import json
import logging
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from llm_ports.call.roles import Role
from llm_ports.grounding import numbers
from llm_ports.observability import EventHook, log_events
from llm_ports.ports import Generated, JudgeVerdict
from llm_ports.ports.outcomes import REASON_MAX
from llm_ports.prompts import sha

#: Characters of the judged call's system prompt the judge reads.
SYSTEM_CAP = 12000
#: Characters of the judged call's output grammar the judge reads.
SCHEMA_CAP = 6000
#: Hex characters of a sample key that `selected` draws on (32 bits).
_DRAW_HEX = 8


@dataclass(frozen=True)
class Seen:
    """What the judged call put in front of its model beside the prompt.

    The judge reads the same context, so a rule the model was given is not an invention and a
    figure on an attached image is not absent.

    Attributes:
        system: The judged call's system prompt.
        schema: The JSON schema the reply was constrained to, or None for prose.
        images: `(bytes, media_type)` pairs sent beside the prompt.
        web: The call's tier fetched web pages the judge cannot see.
    """

    system: str = ""
    schema: Mapping[str, Any] | None = None
    images: tuple[tuple[bytes, str], ...] = ()
    web: bool = False  # the tier fetched pages the judge cannot see


def schema_of(output: Any) -> Mapping[str, Any] | None:
    """Return the JSON schema an `output=` type sends as the grammar.

    Returns:
        A `json_output` type's own schema, else a pydantic model's, else None (the reply is prose,
        and so is a type whose schema cannot be rendered).
    """
    js = getattr(output, "json_schema", None)
    if isinstance(js, Mapping) and js:
        return js
    fn = getattr(output, "model_json_schema", None)
    if callable(fn):
        try:
            got = fn()
        except Exception:  # a type that cannot render a schema is judged as prose
            return None
        return got if isinstance(got, Mapping) else None
    return None


def seen_for(
    *, system: str = "", output: Any = None, images: Sequence[tuple[bytes, str]] = (), web: bool = False
) -> Seen:
    """Build the `Seen` for a call from its system prompt, `output=` type, images and web flag."""
    return Seen(system=system or "", schema=schema_of(output), images=tuple(images), web=web)


def judge_input(prompt: str, seen: Seen) -> str:
    """Return the judge's Input text.

    The judged call's system prompt (cut to `SYSTEM_CAP`), output grammar (cut to `SCHEMA_CAP`), a
    note for any images, and the prompt, each under a heading the rubric names. A call that saw
    nothing beside its prompt yields the bare prompt.
    """
    parts: list[str] = []
    if seen.system:
        parts.append("## SYSTEM INSTRUCTIONS (given to the model; part of the Input)\n" + seen.system[:SYSTEM_CAP])
    if seen.schema:
        parts.append(
            "## OUTPUT GRAMMAR (the JSON schema the reply was constrained to; its allowed values "
            "are part of the Input)\n" + json.dumps(seen.schema, default=str)[:SCHEMA_CAP]
        )
    if seen.images:
        parts.append(
            f"## IMAGES: {len(seen.images)} image(s) were sent beside the prompt and are attached "
            "after this text; what they show is part of the Input"
        )
    if not parts:
        return prompt
    return "\n\n".join([*parts, "## PROMPT\n" + prompt])


def reply_text(value: Any) -> str:
    """Return the reply as the judge reads it: a pydantic model's JSON, a mapping's JSON, else `str`."""
    if value is None:
        return ""
    dump = getattr(value, "model_dump_json", None)
    if callable(dump):
        return str(dump())
    if isinstance(value, Mapping):
        return json.dumps(value, default=str)
    return str(value)


def sample_key(role: Role, prompt: str, output_text: str) -> str:
    """Return a deterministic 16-hex stamp of one generation: a sha over role name, prompt and reply."""
    return sha(f"{role.name}\n{prompt}\n{output_text}")


def selected(key: str, mod: int) -> bool:
    """Return whether `key` is drawn: 1-in-`mod` on its first 32 bits.

    `mod <= 0` selects nothing, which is the off switch.
    """
    if mod <= 0:
        return False
    return int(key[:_DRAW_HEX], 16) % mod == 0


def _placeholder_law(mode: str) -> str:
    cite = (
        " A `{id}` citation is this role's mandated way to state a value, never a placeholder."
        if mode == "slots"
        else ""
    )
    return (
        "1. SCHEMA ANSWERED HONESTLY: every field carries a real answer. An empty string, a placeholder "
        "('TBD', 'N/A', '...'), or a field filled to satisfy the grammar rather than the question fails. "
        "An empty list is an honest answer when the Input allows 'nothing applies'." + cite + "\n"
    )


def _today_law(today: dt.date, tz_label: str) -> str:
    """State the judge's calendar: a judge model's knowledge ends before today.

    The line is necessary but not sufficient; a small judge can still read a recent date as the
    future.
    """
    return (
        f"Today is {today.isoformat()} ({tz_label}). Your own knowledge of the world ends before today: "
        "a date, event or figure is never wrong because you do not remember it or because it looks "
        "like the future to you. Judge only against the Input.\n"
    )


def rubric(
    role: Role,
    seen: Seen | None = None,
    *,
    platform: str,
    mode: str,
    today: dt.date,
    tz_label: str,
    examples: numbers.LawExamples | None = None,
) -> str:
    """Return the judge's instruction for one reply from `role`.

    The laws are the ones `call.graders` enforce, so the rule and the reader disagree only about
    judgement. A convention the role's own instructions mandate is the law kept, not broken.

    Args:
        role: The judged role; its name and whether it is structured shape the first line.
        seen: What the judged call saw beside its prompt; None reads as nothing.
        platform: Whose reply it is, as the first line names it ("the X platform").
        mode: The role's grounding mode word. `"slots"` changes laws 1 and 2 (`numbers.law_text`).
        today: The judge's calendar date.
        tz_label: The label of the clock `today` was read on.
        examples: Domain words the number law names as not figures; none by default.

    Returns:
        The rubric text.
    """
    seen = seen or Seen()
    shape = (
        "a structured object under a strict grammar"
        if role.graders or role.pack or seen.schema
        else "prose or a structured object"
    )
    return (
        f"You are grading one reply from the {platform}'s `{role.name}` role, "
        f"which produces {shape}. The Input is everything the model was given: any SYSTEM INSTRUCTIONS, "
        "the OUTPUT GRAMMAR, the PROMPT and any attached image. The Output is its reply. Following a "
        "convention the Input mandates (a citation form, a vocabulary, a default) is the law kept. "
        "Judge the Output against these laws and nothing else:\n"
        + _today_law(today, tz_label)
        + _placeholder_law(mode)
        + numbers.law_text(mode, web=seen.web, examples=examples)
        + "3. NO ECHO: the Output analyses the Input; it does not hand the Input back. A reply that is "
        "mostly sentences copied from the prompt fails. A field the Input asks to hold a quote (a source "
        "quote, an evidence span) is not echo.\n"
        "4. GROUNDED: claims follow from the Input's evidence"
        + (" or from the web the role read" if seen.web else "")
        + "; a confident claim with no support fails. Choosing a grammar value the Input reasonably supports "
        "is grounded, even where a reader could choose another.\n"
        "Pass only when all four hold. Score 1.0 for a clean reply, 0.0 for one that invents numbers or "
        "echoes, in between for partial honesty. In the reason, name the law broken and quote the "
        "offending fragment in under 30 words."
    )


def verdict_label(v: JudgeVerdict | None) -> str:
    """Return `pass`, `fail`, or `could_not_run` when the judge never answered (None).

    A missing verdict is never guessed as a pass or a fail.
    """
    if v is None:
        return "could_not_run"
    return "pass" if v.passed else "fail"


@dataclass(frozen=True)
class Sample:
    """One drawn reply, as a `JudgeLane` task receives it.

    Attributes:
        role: The judged role.
        prompt: The judged call's prompt.
        reply: The reply as the judge reads it (`reply_text`).
        key: The sample's stamp (`sample_key`).
        seen: What the judged call saw beside its prompt.
        result: The judged call's result; its `stamp` carries any trace handle to score onto.
    """

    role: Role
    prompt: str
    reply: str
    key: str
    seen: Seen
    result: Generated[Any]


#: A lane task body: judge one `Sample` and store the verdict wherever the consumer keeps it.
JudgeRun = Callable[[Sample], Coroutine[Any, Any, Any]]
#: An eligibility gate: `(role, mod)` to whether this role's replies may be drawn at all.
Eligible = Callable[[Role, int], bool]


class JudgeLane:
    """Run a sampled judge as background tasks, off the caller's await path.

    `start` decides and schedules; the consumer's `run` judges and stores. The lane holds every task
    it created by a strong reference until the task finishes (an event loop keeps tasks only weakly)
    and starts each in a fresh `contextvars.Context`, so a task never inherits the caller's
    context-scoped state after the caller has moved on.

    Args:
        name: The prefix of the lane's event names (`<name>_sampled`, `<name>_seen_failed`,
            `<name>_sampler_failed`).
        on_event: Receives the lane's events; without one they go to the stdlib logger
            `llm_ports.evals.judge`.
    """

    def __init__(self, *, name: str = "llm_judge", on_event: EventHook | None = None) -> None:
        self._name = name
        self._emit: EventHook = on_event if on_event is not None else log_events("llm_ports.evals.judge")
        self._tasks: set[asyncio.Task[Any]] = set()

    def start(
        self,
        role: Role,
        prompt: str,
        result: Generated[Any],
        *,
        mod: int | Callable[[], int],
        run: JudgeRun,
        eligible: Eligible | None = None,
        seen: Seen | Callable[[], Seen] | None = None,
    ) -> str | None:
        """Draw `result` and, when drawn, schedule `run` on it.

        Nothing is scheduled without a running event loop, for a non-`ok` outcome, when `eligible`
        refuses, or when `selected(key, mod)` does not draw the reply. `mod` and `seen` may be
        callables, resolved only here: `mod` once per call, `seen` only for a drawn reply. A `seen`
        callable that raises is judged with an empty `Seen` and reported as `<name>_seen_failed`.

        Args:
            role: The judged role.
            prompt: The judged call's prompt.
            result: The judged call's result.
            mod: The 1-in-N draw (`selected`); `<= 0` is off.
            run: The task body, called with the `Sample`.
            eligible: Whether the role may be drawn; None admits every role.
            seen: What the judged call saw beside its prompt; None reads as nothing.

        Returns:
            The sample key when a task was scheduled, else None. Never raises: any failure is
            reported as `<name>_sampler_failed` and returns None.
        """
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:  # no loop to carry a task: a quiet no, not a failure
                return None
            n = mod() if callable(mod) else mod
            if result.outcome != "ok" or (eligible is not None and not eligible(role, n)):
                return None
            reply = reply_text(result.value)
            key = sample_key(role, prompt, reply)
            if not selected(key, n):
                return None
            task = loop.create_task(
                run(Sample(role, prompt, reply, key, self._seen(role, seen), result)),
                context=contextvars.Context(),
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            self._emit(logging.INFO, f"{self._name}_sampled", {"role": role.name, "sample_key": key, "mod": n})
            return key
        except Exception as exc:  # the lane is advisory: a sampler failure must not reach the caller
            self._emit(logging.WARNING, f"{self._name}_sampler_failed", {"role": role.name, "error": _err(exc)})
            return None

    def _seen(self, role: Role, seen: Seen | Callable[[], Seen] | None) -> Seen:
        if seen is None:
            return Seen()
        if not callable(seen):
            return seen
        try:
            return seen()
        except Exception as exc:  # judged blind, and said so
            self._emit(logging.WARNING, f"{self._name}_seen_failed", {"role": role.name, "error": _err(exc)})
            return Seen()

    def pending(self) -> int:
        """Return the number of lane tasks not yet finished."""
        return len(self._tasks)

    async def drain(self) -> None:
        """Await every lane task that belongs to the running loop.

        A task created on another loop cannot be awaited here; it is dropped from the lane. Never
        raises for a task's own exception.
        """
        loop = asyncio.get_running_loop()
        for t in list(self._tasks):
            if t.get_loop() is not loop:
                self._tasks.discard(t)
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)


def _err(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:REASON_MAX]


__all__ = [
    "SCHEMA_CAP",
    "SYSTEM_CAP",
    "Eligible",
    "JudgeLane",
    "JudgeRun",
    "Sample",
    "Seen",
    "judge_input",
    "reply_text",
    "rubric",
    "sample_key",
    "schema_of",
    "seen_for",
    "selected",
    "verdict_label",
]
