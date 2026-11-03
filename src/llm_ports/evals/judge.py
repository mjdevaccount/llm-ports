"""The pure half of a sampled model judge: what it reads, which reply it draws, how it grades.

- `Seen` / `seen_for`: what the judged call put in front of its model beside the prompt (system
  prompt, output grammar, images, a web tier). The judge reads the same.
- `judge_input`: the judge's Input text, each part under a heading the rubric names.
- `reply_text`: the reply as the judge reads it.
- `sample_key` / `selected`: a deterministic stamp of one generation and a 1-in-N draw on it.
- `rubric`: the four laws the deterministic graders (`call_graders`) encode, stated for a model:
  schema answered honestly, no number absent from the Input (`numbers.law_text`), no echo, grounded.
- `verdict_label`: `pass`, `fail` or `could_not_run`.

The consumer passes in, and this module never assumes: whether a role is eligible for judging,
the role's grounding mode, the platform named in the rubric, and today's date with its clock
label. Where a verdict is stored is the consumer's concern.

Invariants:
    - Pure: no model, no database, no clock.
    - A draw is a function of (role, prompt, reply), so whether a reply was judged is answerable
      after the fact.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from llm_ports.call.roles import Role
from llm_ports.grounding import numbers
from llm_ports.ports import JudgeVerdict
from llm_ports.prompts import sha

SYSTEM_CAP = 12000  # characters of the judged call's system prompt the judge reads
SCHEMA_CAP = 6000  # characters of its output grammar


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
    return int(key[:8], 16) % mod == 0


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


def rubric(role: Role, seen: Seen | None = None, *, platform: str, mode: str, today: dt.date, tz_label: str) -> str:
    """Return the judge's instruction for one reply from `role`.

    The laws are the ones `call_graders` enforce, so the rule and the reader disagree only about
    judgement. A convention the role's own instructions mandate is the law kept, not broken.

    Args:
        role: The judged role; its name and whether it is structured shape the first line.
        seen: What the judged call saw beside its prompt; None reads as nothing.
        platform: Whose reply it is, as the first line names it ("the X platform").
        mode: The role's grounding mode word. `"slots"` changes laws 1 and 2 (`numbers.law_text`).
        today: The judge's calendar date.
        tz_label: The label of the clock `today` was read on.

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
        + numbers.law_text(mode, web=seen.web)
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


__all__ = [
    "SCHEMA_CAP",
    "SYSTEM_CAP",
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
