"""Protocols and frozen value types: the seams a consumer owns.

Frameworks live behind these ports. One adapter module per framework implements a port, and the
contract tests exercise the port rather than the adapter, so any adapter that passes them can
replace any other.

Ten ports, each named for the pattern it stands for:

- `Retriever`: RAG retrieval. A query becomes ranked chunks with provenance.
- `Citer`: citation-grounded generation. A question and chunks become claims, each citing ids of
  the chunks it was given.
- `Grounder`: groundedness. One claim is checked against the spans it cites.
- `Evaluator`: the eval harness. Rows and graders become one board.
- `PromptSource`: prompt management. A name and a label resolve to a source template and its
  provenance (the repo seed or a managed version).
- `Tracer`: LLM observability. One generation observation per model call, ended with what the
  call produced and scored.
- `Generator`: typed structured generation. A prompt and an output type become a validated value
  or a named outcome. The call path around it (role, pack, gate, trace, graders) is `llm_ports.call`.
- `Embedder`: text embeddings. Texts become one vector per text, all in one named space.
- `Reranker`: cross-encoder reranking. A query and documents become one order-only score per
  document.
- `Judge`: LLM-as-a-judge. A prompt, the reply it drew and a rubric become a graded verdict.

Invariants:
    - No I/O: the module holds Protocols and frozen value types only. Adapters translate their
      framework's objects into these types.
    - `Generator`, `Embedder` and `Reranker` never raise for a model, wire or gate failure; the
      failure is the result's `outcome`, and a partial result is `invalid`.
    - `Tracer` and `Generation` never raise; a sink that is down yields a no-op `Generation`.
    - `Judge` raises when the judging model cannot answer; the caller owns the catch.

Rationale: ADR-0001.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

# ── value types ─────────────────────────────────────────────────────────────────────────────────


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


#: Every way one generation can end:
#:
#: - `ok`: the reply parsed into the output type.
#: - `gated`: the category switchboard refused the call.
#: - `unavailable`: the model server or gateway failed, and the input is not to blame.
#: - `request_fault`: the request was refused (HTTP 400, 413 or 422), so the input is to blame.
#: - `truncated`: the token ceiling cut the reply. A budget failure, never a parse error.
#: - `invalid`: the reply would not parse into the output type.
Outcome = Literal["ok", "gated", "unavailable", "request_fault", "truncated", "invalid"]


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
        box: True for the self-hosted tier. An adapter may take a lease on that model group, and a
            role sends it the thinking switch.
        timeout_s: The client timeout, in seconds.
        output_retries: Re-asks allowed when the validator finds problems.
        label: A name for the call in an adapter's logs; `call.request_for` sets the role name.
    """

    model: str
    prompt: str
    system: str = ""
    images: tuple[tuple[bytes, str], ...] = ()
    settings: Mapping[str, Any] = field(default_factory=dict)
    box: bool = True
    timeout_s: float = 120.0
    output_retries: int = 1
    label: str = "box-chat"


@dataclass(frozen=True)
class Generated[T]:
    """What a `Generator` returns.

    Attributes:
        value: The parsed value; None exactly when `outcome != "ok"`.
        outcome: How the call ended.
        model: The model that answered.
        reason: One line naming the failure; None on `ok`.
        retries: Re-asks used; None when the run died mid-flight and the count is unknown, never 0.
        problems: The validator's findings on the kept value once the re-asks ran out. The caller
            prunes the value or drops it.
        usage: Token counts as the provider reported them; None, or a None count, is unmeasured.
        sources: URLs the model server fetched while answering (a research-server extension);
            None when the reply carried none.
        finish_reason: The provider's finish reason, when it sent one.
        stamp: Extras the call path attaches by name (e.g. `scores`, `generation`).
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


#: Which side of an asymmetric embedding model a text is on.
EmbedInput = Literal["document", "query"]


@dataclass(frozen=True)
class Embedded:
    """What an `Embedder` returns: one vector per input, in input order, all in one space.

    Attributes:
        vectors: One vector per input; `()` exactly when `outcome != "ok"`.
        outcome: How the call ended.
        stamp: The vector space (model and dims). It belongs on every stored vector.
        reason: One line naming the failure; None on `ok`.
        input_tokens: Tokens the wire reported; None when it reported no usage (unknown, never 0).

    Invariants:
        - Two stamps never compare: a store holding two is a reindex in progress, not an index.
        - A batch that half-landed is a failed batch; no partial result is returned.
    """

    vectors: tuple[tuple[float, ...], ...]
    outcome: Outcome
    stamp: str = ""
    reason: str | None = None
    input_tokens: int | None = None

    @property
    def ok(self) -> bool:
        """True when `outcome == "ok"`."""
        return self.outcome == "ok"


@dataclass(frozen=True)
class Reranked:
    """What a `Reranker` returns: one score per document, in input order.

    Attributes:
        scores: One score per document; `()` exactly when `outcome != "ok"`.
        outcome: How the call ended.
        model: The model that scored.
        reason: One line naming the failure; None on `ok`.

    Invariants:
        - Scores are model-relative and order-only (one model emits raw logits, another 0..1).
          Rank by them; never threshold them or compare them across models.
    """

    scores: tuple[float, ...]
    outcome: Outcome
    model: str = ""
    reason: str | None = None

    @property
    def ok(self) -> bool:
        """True when `outcome == "ok"`."""
        return self.outcome == "ok"


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


# ── ports ───────────────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class Retriever(Protocol):
    """RAG retrieval: a query to ranked chunks with provenance.

    Hybrid search and reranking are the adapter's concern; the port promises order and provenance.
    """

    async def search(
        self, q: str, *, k: int = 8, date_from: date | None = None, date_to: date | None = None
    ) -> list[Chunk]:
        """Return up to `k` chunks for `q`, best first.

        An empty or blank `q` returns `[]`. Finding nothing is an empty list, never an exception.

        Args:
            q: The query.
            k: The most chunks to return.
            date_from: Lower bound on `Chunk.as_of`; None leaves it open.
            date_to: Upper bound on `Chunk.as_of`; None leaves it open.
        """
        ...


@runtime_checkable
class Citer(Protocol):
    """Citation-grounded generation: a question and chunks to cited claims."""

    async def answer(self, q: str, chunks: Sequence[Chunk]) -> CitedAnswer:
        """Answer `q` from `chunks`, each claim citing the ids it rests on.

        No chunks yield no claims, not an error.

        Invariants:
            - Every claim's `evidence_ids` is a subset of `{c.id for c in chunks}`, so
              `CitedAnswer.unknown_ids()` is empty. A citation is a chunk id, never a URL the
              model typed.
        """
        ...


@runtime_checkable
class Grounder(Protocol):
    """Groundedness: one claim checked against the spans it cites."""

    def verdict(self, claim: Claim, spans: Sequence[Chunk]) -> GroundingVerdict:
        """Return whether `claim` is supported by `spans`.

        Pure.

        Args:
            claim: The claim to check.
            spans: The chunks the claim cites, already resolved by the caller.

        Invariants:
            - A claim with no spans is never `supported`.
        """
        ...


#: A grader reads one row and returns `(passed, reason)`.
Grader = Callable[[dict[str, Any]], tuple[bool, str]]


@runtime_checkable
class Evaluator(Protocol):
    """The eval harness: rows and graders to one board."""

    def run(
        self,
        surface: str,
        rows: list[dict[str, Any]],
        graders: Mapping[str, Grader],
        *,
        id_field: str = "id",
        monitors: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Grade every row with every grader and return one board.

        Args:
            surface: The name of the surface under evaluation, echoed on the board.
            rows: The rows to grade.
            graders: Grader name to grader.
            id_field: The row key that names a row in `examples`.
            monitors: Computes the surface's own rates over all rows; None for none.

        Returns:
            A board with the keys `surface`, `n`, `rows_failed`, `by_grader` and `monitors`.
            `by_grader` maps each grader name to `pass`, `fail` and `examples` (at most three
            `{id, reason}` failures). `monitors` holds the callable's result verbatim, or `{}`
            when none was passed.

        Invariants:
            - Every adapter emits exactly this shape, and every key is always present.
        """
        ...


@runtime_checkable
class PromptSource(Protocol):
    """Prompt management: a name and a label to a source template and its provenance.

    Rendering is the caller's; the port never compiles a template.
    """

    def get(self, name: str, *, label: str = "production") -> Prompt:
        """Return the prompt `name` as served under `label`.

        Synchronous, because the v2 Langfuse SDK is; an async caller uses `asyncio.to_thread`.
        A managed store that is down is not an error: the result is the repo seed, with
        `source == "repo"`. An adapter with no seed for `name` has nothing to serve and raises.

        Invariants:
            - `text` is never empty.
        """
        ...


@runtime_checkable
class Generation(Protocol):
    """One open generation observation.

    Invariants:
        - Neither method raises.

    Rationale: ADR-0007.
    """

    def end(
        self,
        *,
        output: Any = None,
        usage: Mapping[str, int | None] | None = None,
        level: str = "DEFAULT",
        status_message: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Close the observation.

        Args:
            output: What the call produced.
            usage: The raw pair `{"input": int | None, "output": int | None}` as the provider
                reported it, or None when it reported neither. The caller decides what unmeasured
                means; the adapter decides the sink's key names.
            level: "DEFAULT", or "ERROR" for a failed call.
            status_message: The error line carried with `level == "ERROR"`.
            metadata: Extra fields for the sink.

        Invariants:
            - A None field is omitted from what the sink receives, never sent as null.
        """
        ...

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        """Attach a numeric score to this observation. Never raises."""
        ...


@runtime_checkable
class Tracer(Protocol):
    """LLM observability: one generation observation per model call.

    Synchronous, because the v2 Langfuse SDK is; the async wrapper around a model call is the
    facade's (`observation.observe`). Fail-open.
    """

    def start_generation(
        self,
        *,
        name: str,
        model: str,
        provider: str,
        input: Any,
        metadata: Mapping[str, Any],
        session_id: str | None,
        prompt_name: str | None,
    ) -> Generation:
        """Open one generation observation.

        Never raises: a sink that is down or refusing returns a no-op `Generation`.

        Args:
            name: The category a cost view groups by; the adapter stores it verbatim.
            model: The model id.
            provider: The provider name.
            input: What the model was sent.
            metadata: Extra fields for the sink.
            session_id: When set, groups the generation under that session.
            prompt_name: When set, links the managed prompt of that name.
        """
        ...


@runtime_checkable
class Generator(Protocol):
    """Typed structured generation: a prompt and an output type to a value or a named outcome."""

    async def generate[T](
        self,
        req: GenRequest,
        output: type[T],
        *,
        validate: Callable[[T], Sequence[Any]] | None = None,
        retry_message: Callable[[Sequence[Any]], str] | None = None,
    ) -> Generated[T]:
        """Run one call and parse the reply into `output`.

        Never raises for a model, wire or gate failure: the failure is the `outcome`, with a
        one-line `reason`.

        Args:
            req: The resolved call.
            output: A pydantic model, a `JsonOutput` subclass, or `str` for prose.
            validate: Lists the problems with a parsed value. A non-empty list earns
                `req.output_retries` re-asks, after which the value is kept with its problems.
            retry_message: Builds a re-ask's message from the problems.

        Rationale: ADR-0001.
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """Text embeddings: texts to one vector each, all in one named space."""

    @property
    def stamp(self) -> str:
        """The vector space this embedder writes: model id and dims (e.g. `example-embed-v1@1024`).

        Constant for the embedder's life.
        """
        ...

    async def embed(self, texts: Sequence[str], *, input_type: EmbedInput = "document") -> Embedded:
        """Embed `texts`, one vector per text, in input order.

        Never raises for a model, wire or gate failure: the failure is the `outcome`. A wrong
        count or a wrong dimension is `invalid`, never a partial result.

        Args:
            texts: The texts to embed.
            input_type: "query" lets an asymmetric model treat a search query differently from a
                stored document; the adapter owns any instruction prefix.

        Rationale: ADR-0001.
        """
        ...


@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder reranking: a query and documents to one order-only score per document."""

    async def rerank(self, query: str, documents: Sequence[str]) -> Reranked:
        """Score each document against `query`, in input order.

        No documents return an `ok` with no scores. Never raises for a model, wire or gate
        failure: the failure is the `outcome`, and the caller keeps the order it already had. A
        reply that does not score every document exactly once is `invalid`, never a partial
        ranking.

        Rationale: ADR-0001.
        """
        ...


@runtime_checkable
class Judge(Protocol):
    """LLM-as-a-judge: a prompt, the reply it drew and a rubric to a graded verdict."""

    async def judge(self, *, prompt: str, output: Any, rubric: str) -> JudgeVerdict:
        """Grade `output`, the reply to `prompt`, against `rubric`.

        Raises when the judging model could not answer; the caller decides what an unjudged
        sample means.

        Rationale: ADR-0001.
        """
        ...


__all__ = [
    "Chunk",
    "CitedAnswer",
    "Citer",
    "Claim",
    "EmbedInput",
    "Embedded",
    "Embedder",
    "Evaluator",
    "GenRequest",
    "Generated",
    "Generation",
    "Generator",
    "Grader",
    "Grounder",
    "GroundingVerdict",
    "JsonOutput",
    "Judge",
    "JudgeVerdict",
    "Outcome",
    "Prompt",
    "PromptSource",
    "Reranked",
    "Reranker",
    "Seed",
    "Retriever",
    "Tracer",
    "json_output",
]
