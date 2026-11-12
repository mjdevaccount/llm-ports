"""The frozen value types that cross the ports.

Each type is a frozen dataclass (or, for `JsonOutput`, a marker class) with no behaviour beyond
reading its own fields. Adapters translate their framework's objects into these types, so a
consumer never imports a framework type to read a result.

Invariants:
    - No I/O.
    - A result type carries its failure as a named `Outcome`, never as an exception.

Rationale: ADR-0001.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, ClassVar, Literal

from llm_ports.ports._failure import Failure
from llm_ports.ports._usage import Answered, Attempt, CallWarning, Cost, Finish, Usage
from llm_ports.ports.outcomes import Outcome


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of text, with its provenance.

    Attributes:
        id: The id a `Claim` cites.
        title: Display title; empty when the source has none.
        source: The corpus or feed the chunk came from (e.g. "wiki", "tickets", "web").
        url: The chunk's address; empty when it has none.
        as_of: The date the content speaks for; None when undated.
        score: Adapter-relative relevance (a cross-encoder's raw logit, a fused rank score).
        meta: Adapter extras, read by name.

    Invariants:
        - `score` is order-only: it ranks one adapter's results and is never a threshold or
          compared across adapters.
    """

    id: str
    text: str
    title: str = ""
    source: str = ""
    url: str = ""
    as_of: date | None = None
    score: float | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Claim:
    """A sentence the writer asserts, with the ids of the chunks it rests on.

    Attributes:
        evidence_ids: Ids of the cited chunks; empty means the claim is ungrounded.
        meta: Adapter extras for this claim, read by name.
    """

    text: str
    evidence_ids: tuple[str, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CitedAnswer:
    """What a `Citer` returns: the cited claims and the chunks they may cite.

    Carrying the chunks lets a `Grounder` read the cited text without a second retrieval.

    Attributes:
        question: The question answered.
        claims: The claims, in the order written.
        chunks: Chunk id to chunk; the universe the claims may cite.
        model: The model the adapter reports; empty when the adapter has none (a stub).
        extras: The adapter's own per-answer report (retries, latency, ...). A consumer reads a
            key by name; the port defines none.
    """

    question: str
    claims: tuple[Claim, ...]
    chunks: Mapping[str, Chunk]
    model: str = ""
    extras: Mapping[str, Any] = field(default_factory=dict)

    def unknown_ids(self) -> tuple[str, ...]:
        """Return the cited ids that are not in `chunks`.

        A non-empty result means the citer broke the `Citer` contract.
        """
        return tuple(i for c in self.claims for i in c.evidence_ids if i not in self.chunks)


@dataclass(frozen=True)
class GroundingVerdict:
    """What a `Grounder` returns for one claim.

    Attributes:
        supported: The gate.
        reason: One line saying why.
        score: The adapter's own scale (an entailment model's probability); None for a rule-based
            check that has none. Reported, never compared across adapters.
        checker: The adapter that produced the verdict (e.g. "number_law", "hhem", "lettucedetect").
    """

    supported: bool
    reason: str = ""
    score: float | None = None
    checker: str = ""


@dataclass(frozen=True)
class Prompt:
    """What a `PromptSource` returns: a source template and its provenance.

    Attributes:
        name: The prompt's name.
        text: The source template. The caller renders it; an adapter never compiles it.
        version: The managed copy's integer version; None for the repo seed.
        source: "repo" for the repo seed, otherwise the managed copy's origin (e.g. "langfuse:v7").
        config: The adapter's per-version extras (content sha, pack version, model role, ...),
            read by name.
        system: The system template of a two-message (chat) prompt; empty for a text prompt.

    Invariants:
        - `version is None` exactly when `source == "repo"`. The repo seed is served when the
          managed copy is absent, unreachable or disagrees with the seed.
    """

    name: str
    text: str
    version: int | None = None
    source: str = "repo"
    config: Mapping[str, Any] = field(default_factory=dict)
    system: str = ""


@dataclass(frozen=True)
class Seed:
    """What the repo says a prompt is.

    A `PromptSource` serves a seed as a `Prompt` with `source == "repo"`, and a managed store
    mirrors it.

    Attributes:
        text: The source template.
        sha: The content sha that names `text`.
        config: Carried beside the text (pack version, model role, ...).
        tags: Tags for the managed copy.
        system: The system template; non-empty makes the seed a chat prompt (system + user).
        kind: "chat" or "text" forces the managed prompt's type; empty derives it from `system`.
    """

    text: str
    sha: str
    config: Mapping[str, Any] = field(default_factory=dict)
    tags: Sequence[str] = ()
    system: str = ""
    kind: str = ""


class JsonOutput(dict[str, Any]):
    """A hand-written JSON schema used as a `Generator` output type.

    The value is a plain dict, and the schema reaches the model unchanged. Callers use
    `json_output`, which mints one subclass per schema; each `Generator` adapter translates that
    subclass into its framework's structured-output object (pydantic-ai: `StructuredDict`), so the
    facade (`call.json_output`) imports no adapter.

    Attributes:
        json_schema: The schema sent to the model.
        schema_name: The schema's title.
    """

    json_schema: ClassVar[Mapping[str, Any]] = {}
    schema_name: ClassVar[str] = ""


def json_output(schema: Mapping[str, Any], name: str) -> type[JsonOutput]:
    """Return a `JsonOutput` subclass that carries `schema` and `name`.

    Pure.

    Args:
        schema: The JSON schema, sent to the model unchanged.
        name: The schema's title, and the name a test stub dispatches on.
    """
    return type(name, (JsonOutput,), {"json_schema": dict(schema), "schema_name": name})


@dataclass(frozen=True)
class GenRequest:
    """One model call, already resolved from a role: model id, rendered prompt and wire settings.

    Attributes:
        model: The model id.
        prompt: The rendered user prompt.
        system: The rendered system prompt; empty for none.
        images: (bytes, MIME type) pairs for a vision model.
        settings: Sent to the model verbatim (max_tokens, temperature, seed, top_p, extra_body).
        self_hosted: True for a self-hosted tier. An adapter may take a lease on that model group,
            and a role sends it the thinking switch.
        timeout_s: The client timeout, in seconds.
        output_retries: Re-asks allowed when the validator finds problems.
        label: A name for the call in an adapter's logs; `call.request_for` sets the role name.
        accept_truncated: Return a length-cut reply that parsed as `ok`, with a `CallWarning`,
            instead of `truncated`. Off by default.
    """

    model: str
    prompt: str
    system: str = ""
    images: tuple[tuple[bytes, str], ...] = ()
    settings: Mapping[str, Any] = field(default_factory=dict)
    self_hosted: bool = True
    timeout_s: float = 120.0
    output_retries: int = 1
    label: str = "chat"
    accept_truncated: bool = False


@dataclass(frozen=True)
class Generated[T]:
    """What a `Generator` returns.

    Attributes:
        value: The parsed value. None when `outcome != "ok"`, except `truncated`, which keeps a
            length-cut reply that still parsed so a caller that accepts cut text can read it.
        outcome: How the call ended.
        model: The model id the request named. Deprecated: `answered` says who answered.
        reason: One line naming the failure; None on `ok`.
        retries: Re-asks used; None when the run died mid-flight and the count is unknown, never 0.
        problems: The validator's findings on the kept value once the re-asks ran out. The caller
            prunes the value or drops it.
        usage: `{"input", "output"}` token counts; None, or a None count, is unmeasured.
            Deprecated: derived from `usage_v2`.
        sources: URLs the model server fetched while answering (a research-server extension);
            None when the reply carried none.
        finish_reason: The provider's finish reason, when it sent one. Deprecated: derived from
            `finish.raw`.
        stamp: Extras the call path attaches by name (e.g. `scores`, `generation`).
        usage_v2: Every bucket, summed over every wire request; None when no request was made.
        cost: What the call cost; None only when no `Pricer` was configured and no gateway
            reported a cost.
        finish: How the last reply ended; None when the provider said nothing.
        answered: Who produced the result.
        attempts: Every wire request or fallback step that did not answer, in order.
        warnings: Settings dropped or altered on the way to the model.
        failure: The failure's kind and detail; None exactly when `outcome == "ok"`.

    Invariants:
        - When left None, `usage` is `usage_v2.as_mapping()`, `finish_reason` is `finish.raw`
          and `reason` is `failure.reason`.

    Rationale: ADR-0001, ADR-0019, ADR-0020.
    """

    value: T | None
    outcome: Outcome
    model: str = ""
    reason: str | None = None
    retries: int | None = 0
    problems: tuple[Any, ...] = ()
    usage: Mapping[str, int | None] | None = None
    sources: tuple[str, ...] | None = None
    finish_reason: str | None = None
    stamp: Mapping[str, Any] = field(default_factory=dict)
    usage_v2: Usage | None = None
    cost: Cost | None = None
    finish: Finish | None = None
    answered: Answered | None = None
    attempts: tuple[Attempt, ...] = ()
    warnings: tuple[CallWarning, ...] = ()
    failure: Failure | None = None

    def __post_init__(self) -> None:
        """Derive the 1.x fields a caller left None from the fields that carry them now."""
        if self.usage is None and self.usage_v2 is not None:
            object.__setattr__(self, "usage", self.usage_v2.as_mapping())
        if self.finish_reason is None and self.finish is not None:
            object.__setattr__(self, "finish_reason", self.finish.raw)
        if self.reason is None and self.failure is not None:
            object.__setattr__(self, "reason", self.failure.reason)


#: Which side of an asymmetric embedding model a text is on.
EmbedInput = Literal["document", "query"]


class _Outcomed:
    """Behaviour shared by the result types whose only 1.x field is `reason`.

    A plain base (not a dataclass), so it adds no fields: each subclass declares its own.
    """

    outcome: Outcome
    reason: str | None
    failure: Failure | None

    def __post_init__(self) -> None:
        """Derive the 1.x `reason` from `failure` when a caller left it None."""
        if self.reason is None and self.failure is not None:
            object.__setattr__(self, "reason", self.failure.reason)

    @property
    def ok(self) -> bool:
        """True when `outcome == "ok"`."""
        return self.outcome == "ok"


@dataclass(frozen=True)
class Embedded(_Outcomed):
    """What an `Embedder` returns: one vector per input, in input order, all in one space.

    Attributes:
        vectors: One vector per input; `()` exactly when `outcome != "ok"`.
        outcome: How the call ended.
        stamp: The vector space (model and dims). It belongs on every stored vector.
        reason: One line naming the failure; None on `ok`. Derived from `failure` when left None.
        input_tokens: Tokens the wire reported; None when it reported no usage (unknown, never 0).
        retries: Requests the adapter re-sent after a retryable failure (its `RetryPolicy`),
            counted whether the call then landed or not.
        usage_v2: Token counts summed over every request; None when no request was made.
        cost: What the call cost; None only when no `Pricer` was configured.
        answered: Who produced the vectors.
        failure: The failure's kind and detail; None exactly when `outcome == "ok"`.

    Invariants:
        - Two stamps never compare: a store holding two is a reindex in progress, not an index.
        - A batch that half-landed is a failed batch; no partial result is returned.

    Rationale: ADR-0001, ADR-0019.
    """

    vectors: tuple[tuple[float, ...], ...]
    outcome: Outcome
    stamp: str = ""
    reason: str | None = None
    input_tokens: int | None = None
    retries: int = 0
    usage_v2: Usage | None = None
    cost: Cost | None = None
    answered: Answered | None = None
    failure: Failure | None = None


@dataclass(frozen=True)
class Reranked(_Outcomed):
    """What a `Reranker` returns: one score per document, in input order.

    Attributes:
        scores: One score per document; `()` exactly when `outcome != "ok"`.
        outcome: How the call ended.
        model: The model that scored.
        reason: One line naming the failure; None on `ok`. Derived from `failure` when left None.
        answered: Who scored.
        failure: The failure's kind and detail; None exactly when `outcome == "ok"`.

    Invariants:
        - Scores are model-relative and order-only (one model emits raw logits, another 0..1).
          Rank by them; never threshold them or compare them across models.
    """

    scores: tuple[float, ...]
    outcome: Outcome
    model: str = ""
    reason: str | None = None
    answered: Answered | None = None
    failure: Failure | None = None


@dataclass(frozen=True)
class JudgeVerdict:
    """What a `Judge` returns.

    Attributes:
        passed: The gate.
        score: The judge's own reading, from 0 to 1.
        reason: The judge's one- or two-sentence justification, recorded as the score comment.
        judge: The adapter that produced the verdict (e.g. "pydantic_evals", "stub").
    """

    passed: bool
    score: float
    reason: str = ""
    judge: str = ""


__all__ = [
    "Chunk",
    "CitedAnswer",
    "Claim",
    "EmbedInput",
    "Embedded",
    "GenRequest",
    "Generated",
    "GroundingVerdict",
    "JsonOutput",
    "JudgeVerdict",
    "Outcome",
    "Prompt",
    "Reranked",
    "Seed",
    "json_output",
]
