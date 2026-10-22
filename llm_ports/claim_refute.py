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

The wire half -- rendering a refute pack and calling the facade -- is the consumer's.
"""

from __future__ import annotations

from typing import Any

REASON_MAX_CHARS = 200


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


__all__ = ["REASON_MAX_CHARS", "match_verdicts", "refutation", "verdicts_schema"]
