"""judge_sampling -- the PURE half of a sampled LLM judge: what it reads, which reply it draws, the
rubric it grades under, and the label its verdict lands as. No model, no database, no clock.

  Seen / seen_for  what the judged call put in front of its model BESIDE the prompt (system prompt,
                   output grammar, images, a web tier). A judge handed the prompt alone fails the
                   reply for "inventing" the system prompt's vocabulary, the grammar's allowed
                   values and the chart a vision role read: measured 848 of 854 samples failed
                   that way before the judge read what the model read.
  judge_input      the judge's Input text: system prompt, grammar and prompt, each under a heading
                   the rubric names. A bare prompt stays bare.
  reply_text       the reply as the judge reads it (a pydantic model's JSON, a mapping's, else str).
  sample_key       a deterministic 16-hex stamp of one generation: sha over (role name, prompt,
                   reply). Recorded on the verdict's row, so "was this one judged" stays
                   answerable after the fact -- `random` would make it unanswerable.
  selected         1-in-`mod` on the key's first 32 bits; `mod <= 0` selects nothing (the OFF switch).
  rubric           the four laws as prose for a model -- the same laws the deterministic graders
                   (`call_graders`) encode, judged by a reader instead of a rule: schema answered
                   honestly, no number absent from the Input, no echo, grounded. Law 2 reads the
                   role's grounding `mode` (a `slots` role cites `{id}` under a digit-free grammar,
                   `llm_ports.slots`, so a figure WRITTEN is the invention) and the call's web tier
                   (pages the judge cannot see: a missing figure is not invented).
  verdict_label    pass | fail | could_not_run (the judge never answered).

What is the CONSUMER'S and is passed in, never assumed here: whether a role is eligible at all
(its tier, its trading calendar, its own off-switches), the role's grounding `mode`, the
`platform` named in the rubric's first line, and `today` with the label of the clock it was read
on. Where the verdict lands (a row, a score, a gauge) is the consumer's lane.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from llm_ports.packs import sha
from llm_ports.ports import JudgeVerdict
from llm_ports.roles import Role

SYSTEM_CAP = 12000                # chars of the judged call's system prompt the judge reads
SCHEMA_CAP = 6000                 # chars of its output grammar


@dataclass(frozen=True)
class Seen:
    """What the judged call put in front of its model BESIDE the prompt. The judge reads the
    same: a rule the model was given is not an invention, and a figure on a chart is not absent."""

    system: str = ""
    schema: Mapping[str, Any] | None = None
    images: tuple[tuple[bytes, str], ...] = ()
    web: bool = False          # the tier fetched pages the judge cannot see


def schema_of(output: Any) -> Mapping[str, Any] | None:
    """The JSON schema an `output=` type sent as the grammar: a `json_output` type's own, a
    pydantic model's, else None (prose)."""
    js = getattr(output, "json_schema", None)
    if isinstance(js, Mapping) and js:
        return js
    fn = getattr(output, "model_json_schema", None)
    if callable(fn):
        try:
            got = fn()
        except Exception:      # a type that cannot render a schema is judged as prose
            return None
        return got if isinstance(got, Mapping) else None
    return None


def seen_for(*, system: str = "", output: Any = None, images: Sequence[tuple[bytes, str]] = (),
             web: bool = False) -> Seen:
    return Seen(system=system or "", schema=schema_of(output), images=tuple(images), web=web)


def judge_input(prompt: str, seen: Seen) -> str:
    """The judge's Input text: the judged call's system prompt, its output grammar and its prompt,
    each under a heading the rubric names. A bare prompt stays bare (nothing else was seen)."""
    parts: list[str] = []
    if seen.system:
        parts.append("## SYSTEM INSTRUCTIONS (given to the model; part of the Input)\n" + seen.system[:SYSTEM_CAP])
    if seen.schema:
        parts.append("## OUTPUT GRAMMAR (the JSON schema the reply was constrained to; its allowed values "
                     "are part of the Input)\n" + json.dumps(seen.schema, default=str)[:SCHEMA_CAP])
    if seen.images:
        parts.append(f"## IMAGES: {len(seen.images)} image(s) were sent beside the prompt and are attached "
                     "after this text; what they show is part of the Input")
    if not parts:
        return prompt
    return "\n\n".join([*parts, "## PROMPT\n" + prompt])


def reply_text(value: Any) -> str:
    """The reply as the judge reads it: a pydantic model's JSON, a mapping's JSON, else str."""
    if value is None:
        return ""
    dump = getattr(value, "model_dump_json", None)
    if callable(dump):
        return str(dump())
    if isinstance(value, Mapping):
        return json.dumps(value, default=str)
    return str(value)


def sample_key(role: Role, prompt: str, output_text: str) -> str:
    """Deterministic 16-hex stamp of one generation. Pure."""
    return sha(f"{role.name}\n{prompt}\n{output_text}")


def selected(key: str, mod: int) -> bool:
    """1-in-`mod` on the key's first 32 bits; `mod <= 0` selects nothing (the OFF switch)."""
    if mod <= 0:
        return False
    return int(key[:8], 16) % mod == 0


#: Every mode: a value the grammar itself offers or requires is an ANSWER (law 1, law 4), never a
#: figure under law 2. The 848-of-854 fail run read an enum label, `conviction: low` and
#: `urgency: 3` -- the grammar's own enum and fields -- as numbers the Input "did not carry"; a
#: replay still failed `conviction: 4` with the clause appended LAST, so it leads the law.
_GRAMMAR_VALUES = ("A value the OUTPUT GRAMMAR asks the model to give -- a score, conviction, confidence "
                   "or urgency number, an enum label, a category -- is the model's ANSWER, not a figure "
                   "taken from the Input: never fail it under this law (law 4 asks whether the Input "
                   "supports it). ")


def _number_law(mode: str, *, web: bool = False) -> str:
    """Law 2 as the role's grounding mode and the call's inputs make it. A `slots` reply is judged
    RAW (its figures are `{id}` citations), so a number WRITTEN is the invention. A web call read
    pages the judge cannot see: the number grader is dropped for it
    (`call_graders.graders_for_inputs`), and the reader drops it too -- the rule and the reader
    disagree only about judgement, never about the law."""
    if web:
        law = ("2. WEB FIGURES ARE NOT JUDGED ABSENT: this role searched and read web pages you cannot "
               "see, so a figure, date or name missing from the Input is expected and is NOT invented. "
               "Fail only a figure that CONTRADICTS the Input.")
    elif mode == "slots":
        law = ("2. FIGURES ARE CITED, NEVER WRITTEN: this role writes under a digit-free grammar and "
               "cites each value as `{id}`, naming a fact the Input lists. A number written out in the "
               "Output, as digits or as number words, is invented and fails; a cited id the Input does "
               "not list fails; a cited id the Input lists is the law kept, whatever its value. This law "
               "is about NUMBERS only: words, labels and paraphrase ('flat', 'negative gamma', a "
               "headline's gist) are not figures, and a word naming a fact ('Spot', 'put wall', 'stand down') "
               "is never a reason to fail under this law, cited or not.")
    else:
        return ("2. NUMBERS FROM THE INPUT ONLY: " + _GRAMMAR_VALUES + "Every other number, level, price, "
                "percentage or count in the Output appears in the Input (its prompt, its system instructions "
                "or an attached image), as digits or spelled out. A figure the Input did not carry is "
                "invented, and fails.\n")
    return law + " " + _GRAMMAR_VALUES.rstrip() + "\n"


def _placeholder_law(mode: str) -> str:
    cite = (" A `{id}` citation is this role's mandated way to state a value, never a placeholder."
            if mode == "slots" else "")
    return ("1. SCHEMA ANSWERED HONESTLY: every field carries a real answer. An empty string, a placeholder "
            "('TBD', 'N/A', '...'), or a field filled to satisfy the grammar rather than the question fails. "
            "An empty list is an honest answer when the Input allows 'nothing applies'." + cite + "\n")


def _today_law(today: dt.date, tz_label: str) -> str:
    """The judge's calendar. A judge model's knowledge ends before today: a replay failed 6 of 6
    web research notes for dates 'in the future'. NECESSARY, NOT SUFFICIENT -- with this line a
    small judge still failed them on the same reason."""
    return (f"Today is {today.isoformat()} ({tz_label}). Your own knowledge of the world ends before today: "
            "a date, event or figure is never wrong because you do not remember it or because it looks "
            "like the future to you. Judge only against the Input.\n")


def rubric(role: Role, seen: Seen | None = None, *, platform: str, mode: str, today: dt.date,
           tz_label: str) -> str:
    """The judge's instruction, derived from the role and what its call saw. The laws are
    `call_graders`' -- the rule and the reader must disagree only about judgement, never about the
    law -- and every convention the role's own instructions MANDATE is the law kept, not broken.
    `platform` names whose reply it is ("the X platform"), `mode` the role's grounding mode word
    (`slots` changes laws 1 and 2), `today` + `tz_label` the judge's calendar."""
    seen = seen or Seen()
    shape = ("a structured object under a strict grammar" if role.graders or role.pack or seen.schema
             else "prose or a structured object")
    return (
        f"You are grading one reply from the {platform}'s `{role.name}` role, "
        f"which produces {shape}. The Input is everything the model was given: any SYSTEM INSTRUCTIONS, "
        "the OUTPUT GRAMMAR, the PROMPT and any attached image. The Output is its reply. Following a "
        "convention the Input mandates (a citation form, a vocabulary, a default) is the law kept. "
        "Judge the Output against these laws and nothing else:\n"
        + _today_law(today, tz_label)
        + _placeholder_law(mode)
        + _number_law(mode, web=seen.web) +
        "3. NO ECHO: the Output analyses the Input; it does not hand the Input back. A reply that is "
        "mostly sentences copied from the prompt fails. A field the Input asks to hold a quote (a source "
        "quote, an evidence span) is not echo.\n"
        "4. GROUNDED: claims follow from the Input's evidence"
        + (" or from the web the role read" if seen.web else "") +
        "; a confident claim with no support fails. Choosing a grammar value the Input reasonably supports "
        "is grounded, even where a reader could choose another.\n"
        "Pass only when all four hold. Score 1.0 for a clean reply, 0.0 for one that invents numbers or "
        "echoes, in between for partial honesty. In the reason, name the law broken and quote the "
        "offending fragment in under 30 words."
    )


def verdict_label(v: JudgeVerdict | None) -> str:
    """pass | fail | could_not_run (None: the judge never answered -- never a guessed verdict)."""
    if v is None:
        return "could_not_run"
    return "pass" if v.passed else "fail"


__all__ = ["SCHEMA_CAP", "SYSTEM_CAP", "Seen", "judge_input", "reply_text", "rubric", "sample_key", "schema_of",
           "seen_for", "selected", "verdict_label"]
