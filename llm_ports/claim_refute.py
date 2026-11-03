"""claim_refute -- the adversarial pass's pure half: ONE grammar builder and ONE attribution rule
for "did the adversary refute this claim".

The grammar: `verdicts[<= max_items]{idx, claim <= claim_cap, verdict uphold|refute, reason <=
reason_cap}`. llama.cpp compiles maxLength into GBNF and CLOSES a string at the cap -- valid JSON,
finish_reason=stop -- so a cut reason arrives looking whole. Measured on a production corpus: 279
of 284 and 22,732 of 27,661 reasons sat at EXACTLY the 200-char cap when the prompt never stated a
limit. A reason AT the cap is flagged `reason_truncated` on its refuted entry, never discarded: the
verdict is an enum the grammar cannot cut, so the refutation stands; only its explanation is
partial. ONE-SIDED BLIND SPOT: the grammar counts a `\\uXXXX` escape as one char, so an ESCAPED
surrogate pair is two grammar chars but one Python char -- a cut reason holding k of them reads
200-k and escapes the flag. It can miss, never fire early.

Attribution is BY `idx`, claim text only as a legacy fallback: the grammar forces `claim` to be A
string, not THE string -- a model paraphrases the sentence it echoes (measured: unexamined rate 1.0
over 99 runs when attributed by text), and a capped echo cannot carry a longer claim at all.

The PASS (`refute`) is one prompt, one grammar, any number of callers: each claim is echoed with
the 1-based `idx` its verdict must return (`refute_slots`), the consumer's refute pack renders it
(`refute_prompt` -- the PACK is the consumer's: its manifest names the consumer's role and its
store mirrors it), and the consumer's own `generate` runs it under its hooks (gate, trace, graders).
Attribute the reply with `match_verdicts(texts, value["verdicts"])`.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from llm_ports.packs import Pack
from llm_ports.ports import Generated, json_output
from llm_ports.roles import Role

REASON_MAX_CHARS = 200

#: A claim as the adversary sees it: a sentence, or a mapping whose extra keys ride along (e.g. the
#: row ids a hypothesis leaned on, so the adversary sees what it was built from).
RefuteClaim = str | Mapping[str, Any]
#: The consumer's generation call: `generate(role, prompt=..., output=...) -> Generated`, its own
#: hooks bound (e.g. `functools.partial(call.generate, generator=g, hooks=h)`).
GenerateFn = Callable[..., Awaitable[Generated[Any]]]


def verdicts_schema(*, max_items: int, claim_cap: int, reason_cap: int = REASON_MAX_CHARS) -> dict[str, Any]:
    """The JSON schema for one refute call (hand-written, through `llm_call.json_output`, like the
    rest of the structured family in PR-A). Pure; a fresh dict per call."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["verdicts"],
        "properties": {
            "verdicts": {
                "type": "array",
                "maxItems": max_items,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["idx", "claim", "verdict", "reason"],
                    "properties": {
                        "idx": {"type": "integer"},
                        # An echo, attributed by `idx`; the cap lets a faithful echo through whole.
                        "claim": {"type": "string", "maxLength": claim_cap},
                        "verdict": {"type": "string", "enum": ["uphold", "refute"]},
                        "reason": {"type": "string", "maxLength": reason_cap},
                    },
                },
            },
        },
    }


def match_verdicts(claims: list[str], raw_verdicts: Any) -> list[dict[str, Any] | None]:
    """One verdict (or None = unexamined) per claim -- BY INDEX, claim text only as a legacy
    fallback. `idx` is 1-based, matching the prompt; an out-of-range or duplicate idx is ignored
    rather than misattributed, and a verdict consumed by idx never doubles as a text match for
    another claim. Pure."""
    verdicts = [v for v in raw_verdicts or [] if isinstance(v, dict)]
    by_idx: dict[int, dict[str, Any]] = {}
    for v in verdicts:
        ix = v.get("idx")
        if (isinstance(ix, int) and not isinstance(ix, bool)
                and 1 <= ix <= len(claims) and ix not in by_idx):
            by_idx[ix] = v
    taken = {id(v) for v in by_idx.values()}
    by_claim = {str(v.get("claim") or "").strip(): v for v in verdicts if id(v) not in taken}
    return [by_idx.get(i) or by_claim.get(c) for i, c in enumerate(claims, start=1)]


def refutation(claim: str, reason: Any, cap: int = REASON_MAX_CHARS,
               **extra: Any) -> dict[str, Any]:
    """One adversary refutation as stored. A reason AT `cap` was closed by the grammar, not
    finished by the model, and says so; the key is present only when true, so every existing
    reader of the entry sees the shape it always has."""
    text = reason if isinstance(reason, str) else ""
    out: dict[str, Any] = {**extra, "claim": claim, "reason": text, "source": "adversary"}
    if len(text) >= cap:
        out["reason_truncated"] = True
    return out


def refute_slots(*, claims: Sequence[RefuteClaim], evidence: str, noun: str, subject: str,
                 evidence_label: str, reason_cap: int = REASON_MAX_CHARS) -> dict[str, Any]:
    """The refute pack's slots. Pure. Each claim gets a 1-based `idx` the verdicts must return -- a
    string becomes `{idx, claim}`, a mapping keeps its extra keys; `claims` is that list as JSON
    and `evidence` is the caller's JSON text, verbatim."""
    slim = [{"idx": i, **({"claim": c} if isinstance(c, str) else dict(c))} for i, c in enumerate(claims, start=1)]
    return {"noun": noun, "subject": subject, "evidence_label": evidence_label, "reason_cap": reason_cap,
            "claims": json.dumps(slim, default=str), "evidence": evidence}


def refute_prompt(pack: Pack, *, claims: Sequence[RefuteClaim], evidence: str, noun: str, subject: str,
                  evidence_label: str, reason_cap: int = REASON_MAX_CHARS) -> str:
    """The adversary's prompt: `pack`'s primary template rendered with `refute_slots`. Pure."""
    return pack.render(**refute_slots(claims=claims, evidence=evidence, noun=noun, subject=subject,
                                      evidence_label=evidence_label, reason_cap=reason_cap))


async def refute(generate: GenerateFn, role: Role, *, pack: Pack, claims: Sequence[RefuteClaim], evidence: str,
                 max_items: int, claim_cap: int, noun: str, subject: str, evidence_label: str, name: str,
                 reason_cap: int = REASON_MAX_CHARS) -> Generated[Any]:
    """THE ADVERSARIAL PASS: `refute_prompt` under `verdicts_schema`, through the consumer's
    `generate`. `name` is the output-schema name (`json_output`) a test stub dispatches on. The
    outcome is the caller's to read; nothing here catches."""
    prompt = refute_prompt(pack, claims=claims, evidence=evidence, noun=noun, subject=subject,
                           evidence_label=evidence_label, reason_cap=reason_cap)
    schema = verdicts_schema(max_items=max_items, claim_cap=claim_cap, reason_cap=reason_cap)
    return await generate(role, prompt=prompt, output=json_output(schema, name))


__all__ = ["REASON_MAX_CHARS", "GenerateFn", "RefuteClaim", "match_verdicts", "refutation", "refute",
           "refute_prompt", "refute_slots", "verdicts_schema"]
