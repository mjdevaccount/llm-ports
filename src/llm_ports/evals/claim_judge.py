"""A reply judged claim by claim: decomposed and pre-matched in code, the rest asked one claim at a time.

A judge asked to grade a whole reply at once must find every claim, look each one up, and weigh
each against the rest in a single pass. A small model fails both ways under that load: it calls a
value copied from the Input "invented" and waves through one it should refuse. `grade` asks it
much less (ADR-0031):

1. `decompose.leaves` splits the reply into claims in code, and `decompose.facts_from_text`
   numbers the Input's lines as facts.
2. `decompose.prematch` settles what code can (a value in the Input verbatim, a licensed figure).
3. What remains goes to a `ClaimJudge`, one claim per call. A `classify` field (a value chosen from
   a closed set) or a stated absence is asked only whether a fact CONTRADICTS it
   (`question_for`): choosing among allowed values is a judgement the Input licenses, and the
   absence of a sentence saying "the class is X" is no evidence against it. Every other claim is
   asked the four-label question (`claims.LABELS`).
4. A verdict is held to its evidence (`read_verdict`): a quote that is not in the facts, or a
   `contradicted` with no quote, abstains instead of failing the case.
5. `aggregate` applies the ADR-0029 rule: a case fails when a material claim is `not_in_input`
   or `contradicted`.

`GenerateClaimJudge` is the model-backed `ClaimJudge`: the consumer's claim pack rendered per claim
(`claim_slots`), run through the consumer's own `generate` (its gate, trace and retry) under a
per-question grammar (`verdict_schema`).

Invariants:
    - Everything but `grade` and `GenerateClaimJudge.judge_claim` is pure.
    - Never raises: a failed call is the claim's `could_not_run`, never the case's crash.
    - Code never rules a claim ungrounded; only the judge does, and only with a quote from the facts.

Rationale: ADR-0031.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from llm_ports.call.roles import Role
from llm_ports.evals.claims import GROUNDED, LABELS, CaseLabel, ClaimLabel, Fact, fact_table
from llm_ports.evals.decompose import Kind, Leaf, Prematch, facts_from_text, leaves, norm, prematch
from llm_ports.evals.refute import GenerateFn
from llm_ports.ports import json_output
from llm_ports.prompts import Pack

Question = Literal["contradiction", "grounding"]
Outcome = Literal["settled", "judged", "abstained", "skipped", "could_not_run"]

#: The labels a contradiction-only question may answer; `consistent` reads as `derived`.
CONTRADICTION_LABELS: tuple[str, ...] = ("consistent", "contradicted")
#: Characters of the evidence quote a verdict may carry.
QUOTE_CHARS = 160
#: Characters of the reason a verdict may carry.
REASON_CHARS = 160
#: Fact ids one verdict may cite.
MAX_CITED = 4
#: Claims one case sends to the judge at most; material claims go first, the rest are `skipped`.
CLAIM_CAP = 24
_ERR = 200

#: A leaf's field kind, from the consumer's field map.
KindOf = Callable[[Leaf], Kind]
#: Whether a leaf is material, from the consumer's materiality map.
MaterialOf = Callable[[Leaf], bool]
#: A consumer's own pre-match: `(leaf, kind)` to a `Prematch`, or None when it has nothing to add.
PreMatch = Callable[[Leaf, Kind], Prematch | None]
#: The question a leaf is asked, given its kind.
QuestionOf = Callable[[Leaf, Kind], "Question"]


@dataclass(frozen=True)
class Ask:
    """One claim as the judge is asked about it.

    Attributes:
        path: The claim's path (`Leaf.path`).
        value: The claim as text.
        kind: Its field kind.
        question: `contradiction` (only "does a fact contradict it?") or `grounding` (four labels).
        material: Whether a wrong label on it changes the case.
        hints: What code checked and could not rule on.
    """

    path: str
    value: str
    kind: Kind
    question: Question
    material: bool = True
    hints: tuple[str, ...] = ()


def question_for(leaf: Leaf, kind: Kind) -> Question:
    """Return `contradiction` for a `classify` value or a stated absence, else `grounding`.

    A stated absence ("no items apply") cannot be "not in the Input": only a fact naming what
    applies can refute it.
    """
    return "contradiction" if kind == "classify" or leaf.empty else "grounding"


@dataclass(frozen=True)
class ClaimVerdict:
    """The final word on one claim.

    Attributes:
        path: The claim's path.
        label: Its label, or None when it abstained, was skipped or could not run.
        outcome: `settled` (code ruled), `judged`, `abstained` (the verdict did not hold to its
            evidence), `skipped` (not sent: immaterial, or past the cap) or `could_not_run`.
        checker: Who ruled: `code`, the judge's model name, or empty when nobody did.
        material: Whether the claim counts toward the case label.
        reason: One line.
        fact_ids: The facts the verdict cites.
        quote: The evidence it quotes.
    """

    path: str
    label: ClaimLabel | None
    outcome: Outcome
    checker: str = ""
    material: bool = True
    reason: str = ""
    fact_ids: tuple[str, ...] = ()
    quote: str = ""


class ClaimJudge(Protocol):
    """Judges one claim against the facts. An implementation never raises: a failure is `could_not_run`."""

    async def judge_claim(self, ask: Ask, facts: Sequence[Fact], *, context: str = "") -> ClaimVerdict:
        """Return the verdict on `ask` given `facts` and the consumer's `context` (the field contract)."""
        ...


def verdict_schema(question: Question, fact_ids: Sequence[str]) -> dict[str, Any]:
    """Return the JSON schema of one verdict: evidence first (`fact_ids`, `quote`), then the label, then why.

    The label enum is the question's (`CONTRADICTION_LABELS` or `claims.LABELS`), so a
    contradiction-only question cannot answer `not_in_input`. Cited ids are an enum of the table's
    ids when there are any. Pure; a fresh dict per call.
    """
    ids: dict[str, Any] = {"type": "string", **({"enum": list(fact_ids)} if fact_ids else {})}
    labels = list(CONTRADICTION_LABELS if question == "contradiction" else LABELS)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["fact_ids", "quote", "label", "reason"],
        "properties": {
            "fact_ids": {"type": "array", "items": ids, "maxItems": MAX_CITED},
            "quote": {"type": "string", "maxLength": QUOTE_CHARS},
            "label": {"type": "string", "enum": labels},
            "reason": {"type": "string", "maxLength": REASON_CHARS},
        },
    }


def read_verdict(ask: Ask, value: Any, *, evidence: str, checker: str) -> ClaimVerdict:
    """Hold one judge reply to its evidence and return the claim's verdict.

    `consistent` reads as `derived`. A quote that is not in `evidence` (ends trimmed of spaces,
    dots and quote marks), or a `contradicted` with an empty quote, abstains: a fail must point at
    the fact that refutes the claim. A reply with no known label is `could_not_run`.

    Args:
        ask: The claim asked about.
        value: The judge's reply (a mapping per `verdict_schema`).
        evidence: The facts' text the quote must come from.
        checker: Who answered.

    Returns:
        The `ClaimVerdict`. Pure; never raises.
    """
    v: Mapping[str, Any] = value if isinstance(value, Mapping) else {}
    raw = v.get("label")
    label = "derived" if raw == "consistent" else raw
    if label not in LABELS:
        return ClaimVerdict(ask.path, None, "could_not_run", checker, ask.material, f"no verdict label: {raw!r}"[:_ERR])
    quote = str(v.get("quote") or "")[:QUOTE_CHARS]
    ids = tuple(str(i) for i in v.get("fact_ids") or ())[:MAX_CITED]
    reason = str(v.get("reason") or "")[:REASON_CHARS]
    core = norm(quote.strip(" .\"'"))
    if core and core not in norm(evidence):
        why = f"abstained: the quote is not in the facts ({label})"
        return ClaimVerdict(ask.path, None, "abstained", checker, ask.material, why, ids, quote)
    if label == "contradicted" and not core:
        return ClaimVerdict(
            ask.path, None, "abstained", checker, ask.material, "abstained: contradicted, no quote", ids
        )
    return ClaimVerdict(ask.path, label, "judged", checker, ask.material, reason, ids, quote)


def claim_slots(ask: Ask, facts: Sequence[Fact], *, context: str = "") -> dict[str, Any]:
    """Return a claim pack's slots.

    `context`, `facts` (`claims.fact_table`), `field`, `value`, `kind`, `question`, `hints` (one
    `- ` line each), `labels` (the allowed answers) and `quote_chars`. Put `context` and `facts`
    first in the template and the claim last: every call of a case then shares one prefix, which a
    server's prompt cache reuses. Pure.
    """
    return {
        "context": context,
        "facts": fact_table(facts),
        "field": ask.path,
        "value": ask.value,
        "kind": ask.kind,
        "question": ask.question,
        "hints": "\n".join(f"- {h}" for h in ask.hints),
        "labels": ", ".join(CONTRADICTION_LABELS if ask.question == "contradiction" else LABELS),
        "quote_chars": QUOTE_CHARS,
    }


class GenerateClaimJudge:
    """A `ClaimJudge` that asks a model, one claim per call, through the consumer's `generate`.

    Args:
        generate: The consumer's generation call, its hooks bound (`refute.GenerateFn`).
        role: The judge's role.
        pack: The consumer's claim pack; its template reads `claim_slots`.
        template: The pack template to render.
        name: The output-schema name (`json_output`) a test stub dispatches on.
    """

    def __init__(
        self, generate: GenerateFn, role: Role, *, pack: Pack, template: str = "prompt", name: str = "claim_verdict"
    ) -> None:
        self._generate = generate
        self._role = role
        self._pack = pack
        self._template = template
        self._name = name

    async def judge_claim(self, ask: Ask, facts: Sequence[Fact], *, context: str = "") -> ClaimVerdict:
        """Render, call, and hold the reply to its evidence (`read_verdict`). Never raises."""
        try:
            prompt = self._pack.render(self._template, **claim_slots(ask, facts, context=context))
            schema = verdict_schema(ask.question, [f.id for f in facts])
            res = await self._generate(self._role, prompt=prompt, output=json_output(schema, self._name))
        except Exception as exc:  # a claim that could not be asked is its verdict, never the case's crash
            why = f"{type(exc).__name__}: {exc}"[:_ERR]
            return ClaimVerdict(ask.path, None, "could_not_run", "", ask.material, why)
        checker = str(getattr(res.answered, "model", "") or self._role.name)
        if res.outcome != "ok":
            return ClaimVerdict(
                ask.path, None, "could_not_run", checker, ask.material, f"{res.outcome}: {res.reason}"[:_ERR]
            )
        return read_verdict(ask, res.value, evidence="\n".join(f.text for f in facts), checker=checker)


@dataclass(frozen=True)
class CaseRuling:
    """A case's verdict from its claims.

    Attributes:
        label: `fail` when a material claim is ungrounded; else `unjudgeable` when a material
            claim could not run, or when material claims were asked and none reached a label;
            else `pass`.
        verdicts: Every claim's verdict, in reply order.
    """

    label: CaseLabel
    verdicts: tuple[ClaimVerdict, ...] = ()

    @property
    def failing(self) -> tuple[str, ...]:
        """The paths of the material claims that failed the case."""
        return tuple(v.path for v in self.verdicts if v.material and v.label is not None and v.label not in GROUNDED)

    def counts(self) -> dict[str, int]:
        """Return `claims`, `material`, then `outcome:<o>` and `label:<l>` counts over every verdict."""
        out: dict[str, int] = {"claims": len(self.verdicts), "material": sum(v.material for v in self.verdicts)}
        for v in self.verdicts:
            out[f"outcome:{v.outcome}"] = out.get(f"outcome:{v.outcome}", 0) + 1
            if v.label:
                out[f"label:{v.label}"] = out.get(f"label:{v.label}", 0) + 1
        return out


def aggregate(verdicts: Iterable[ClaimVerdict]) -> CaseRuling:
    """Return the case's ruling from its claims' verdicts: the ADR-0029 rule, with abstention. Pure."""
    vs = tuple(verdicts)
    material = [v for v in vs if v.material and v.outcome != "skipped"]
    if any(v.label is not None and v.label not in GROUNDED for v in material):
        return CaseRuling("fail", vs)
    if any(v.outcome == "could_not_run" for v in material) or (material and all(v.label is None for v in material)):
        return CaseRuling("unjudgeable", vs)
    return CaseRuling("pass", vs)


def _always_material(_leaf: Leaf) -> bool:
    return True


@dataclass(frozen=True)
class Policy:
    """The consumer's reading of a reply's fields: what kind each is, whether it counts, what to ask.

    Attributes:
        kind_of: A leaf's field kind (the consumer's field map).
        material_of: Whether a leaf counts toward the case label; every leaf does by default.
        question_of: The question a leaf is asked; `question_for` by default. A class that asserts
            a cause (which event drove a move, say) may be asked `grounding` instead, since such a
            class needs evidence, not just the absence of a contradiction.
        pre: The consumer's own pre-match, run before `decompose.prematch`: a `Prematch` that rules
            is code's last word; one that does not adds its hints. Like `prematch`, it should only
            ever rule `supported`.
        allowed: Allowed values per field (`Leaf.field`), e.g. from `decompose.enums(schema)`.
    """

    kind_of: KindOf
    material_of: MaterialOf = _always_material
    question_of: QuestionOf = question_for
    pre: PreMatch | None = None
    allowed: Mapping[str, Collection[str]] = field(default_factory=dict)

    def prematch(self, leaf: Leaf, kind: Kind, input_text: str) -> Prematch:
        """Return the consumer's ruling when it rules, else `decompose.prematch` with both sets of hints."""
        own = self.pre(leaf, kind) if self.pre is not None else None
        if own is not None and own.ruled:
            return own
        pm = prematch(leaf, kind, input_text=input_text, allowed=self.allowed.get(leaf.field, ()))
        return Prematch(pm.label, pm.reason, (*own.hints, *pm.hints)) if own is not None else pm


async def grade(
    reply: str,
    input_text: str,
    *,
    judge: ClaimJudge,
    policy: Policy,
    facts: Sequence[Fact] | None = None,
    context: str = "",
    judge_immaterial: bool = False,
    cap: int = CLAIM_CAP,
) -> CaseRuling:
    """Judge one reply claim by claim: `leaves`, the pre-match, the judge on what remains, `aggregate`.

    Claims are asked one at a time, material first, up to `cap`; an immaterial claim is `skipped`
    unless `judge_immaterial`, since it cannot move the case label.

    Args:
        reply: The reply as text.
        input_text: The Input it was written from.
        judge: The `ClaimJudge`.
        policy: The consumer's field map (`Policy`).
        facts: The Input's facts; `facts_from_text(input_text)` when None.
        context: The consumer's field contract, rendered before the facts.
        judge_immaterial: Also ask about immaterial claims.
        cap: The most claims one case sends to the judge.

    Returns:
        The `CaseRuling`, its verdicts in reply order. Never raises: the judge's failures are verdicts.
    """
    table = tuple(facts) if facts is not None else facts_from_text(input_text)
    ruled: dict[int, ClaimVerdict] = {}
    asks: list[tuple[int, Ask]] = []
    for i, leaf in enumerate(leaves(reply)):
        kind, material = policy.kind_of(leaf), policy.material_of(leaf)
        pm = policy.prematch(leaf, kind, input_text)
        if pm.ruled:
            ruled[i] = ClaimVerdict(leaf.path, pm.label, "settled", "code", material, pm.reason)
        elif not material and not judge_immaterial:
            ruled[i] = ClaimVerdict(leaf.path, None, "skipped", "", material, "immaterial: not asked")
        else:
            q = policy.question_of(leaf, kind)
            asks.append((i, Ask(leaf.path, leaf.value, kind, q, material, pm.hints)))
    asks.sort(key=lambda a: not a[1].material)  # stable: reply order within each group
    for n, (i, ask) in enumerate(asks):
        if n >= cap:
            ruled[i] = ClaimVerdict(ask.path, None, "skipped", "", ask.material, f"past the cap of {cap} claims")
        else:
            ruled[i] = await judge.judge_claim(ask, table, context=context)
    return aggregate(ruled[i] for i in sorted(ruled))


__all__ = [
    "CLAIM_CAP",
    "CONTRADICTION_LABELS",
    "MAX_CITED",
    "QUOTE_CHARS",
    "REASON_CHARS",
    "Ask",
    "CaseRuling",
    "ClaimJudge",
    "ClaimVerdict",
    "GenerateClaimJudge",
    "KindOf",
    "MaterialOf",
    "Outcome",
    "Policy",
    "PreMatch",
    "Question",
    "QuestionOf",
    "aggregate",
    "claim_slots",
    "grade",
    "question_for",
    "read_verdict",
    "verdict_schema",
]
