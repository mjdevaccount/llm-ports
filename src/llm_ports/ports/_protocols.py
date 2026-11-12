"""The ports: the Protocols a consumer owns and an adapter implements.

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

Two smaller seams sit beside them: `Pricer` turns a call's usage into a `Cost`, and
`GatewayReader` reads what a gateway in front of the model reported in its response headers.

Invariants:
    - No I/O: the module holds Protocols only; the value types they exchange are in `_types`.
    - `Generator`, `Embedder` and `Reranker` never raise for a model, wire or gate failure; the
      failure is the result's `outcome`, and a partial result is `invalid`.
    - `Tracer` and `Generation` never raise; a sink that is down yields a no-op `Generation`.
    - `Judge` raises when the judging model cannot answer; the caller owns the catch.
    - `Pricer.price` never raises; an unknown price is `Cost(usd=None)`, never 0.

Rationale: ADR-0001.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import date
from decimal import Decimal
from typing import Any, Protocol, TypedDict, runtime_checkable

from llm_ports.ports._failure import Failure
from llm_ports.ports._types import (
    Chunk,
    CitedAnswer,
    Claim,
    Embedded,
    EmbedInput,
    Generated,
    GenRequest,
    GroundingVerdict,
    JudgeVerdict,
    Prompt,
    Reranked,
)
from llm_ports.ports._usage import Answered, Cost, Finish, Usage


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

        Synchronous, as prompt-management SDKs are; an async caller uses `asyncio.to_thread`.
        A managed store that is down is not an error: the result is the repo seed, with
        `source == "repo"`. An adapter with no seed for `name` has nothing to serve and raises.

        Invariants:
            - `text` is never empty.
        """
        ...


class EndFields(TypedDict, total=False):
    """Every keyword of `Generation.end`, for an implementation that takes `**fields`.

    An implementation may spell the keywords out or take `**fields: Unpack[EndFields]`. An absent
    key is the port's default: `level` `"DEFAULT"`, every other key None (ADR-0025).
    """

    output: Any
    usage: Mapping[str, int | None] | None
    level: str
    status_message: str | None
    metadata: Mapping[str, Any] | None
    usage_v2: Usage | None
    cost: Cost | None
    finish: Finish | None
    answered: Answered | None
    failure: Failure | None


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
        usage_v2: Usage | None = None,
        cost: Cost | None = None,
        finish: Finish | None = None,
        answered: Answered | None = None,
        failure: Failure | None = None,
    ) -> None:
        """Close the observation.

        Args:
            output: What the call produced.
            usage: The raw pair `{"input": int | None, "output": int | None}` as the provider
                reported it, or None when it reported neither. The caller decides what unmeasured
                means; the adapter decides the sink's key names.
            level: "DEFAULT"; "ERROR" for a failed call; "CANCELLED" for a call its caller
                cancelled (or interrupted).
            status_message: The error line carried with a level other than "DEFAULT".
            metadata: Extra fields for the sink.
            usage_v2: The call's usage summed over every request (ADR-0019); None when unknown.
            cost: What the call cost, with its source; None when not priced.
            finish: How the last reply ended.
            answered: Who answered.
            failure: The failure's kind and detail; None for a call that succeeded.

        A 1.x implementation without the five keyword-only arguments added in 1.24 keeps
        working: `observability` passes them only to an `end` that accepts them.

        Invariants:
            - A None field is omitted from what the sink receives, never sent as null.
            - An unmeasured count is never written as 0.
        """
        ...

    def score(
        self, name: str, value: float, comment: str | None = None, metadata: Mapping[str, Any] | None = None
    ) -> None:
        """Attach a numeric score to this observation. Never raises.

        Args:
            name: The score's name.
            value: Its value.
            comment: A one-line reason, shown beside the score.
            metadata: Extra fields stored on the score itself (who gave it, for a label);
                omitted from what the sink receives when None.
        """
        ...


@runtime_checkable
class Tracer(Protocol):
    """LLM observability: one generation observation per model call.

    Synchronous, as tracing SDKs are; the async wrapper around a model call is
    `observability.observe`. Fail-open.
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


@runtime_checkable
class Pricer(Protocol):
    """A price table: a call's usage and who answered become what it cost."""

    def price(self, usage: Usage, answered: Answered) -> Cost:
        """Return the cost of `usage` on the model `answered` names.

        Never raises: a model the table does not know is `Cost(None, "unknown",
        reason="unknown_model")`, and usage with nothing measured is `reason="no_usage"`.

        Rationale: ADR-0019.
        """
        ...


@runtime_checkable
class GatewayReader(Protocol):
    """What a gateway in front of the model reported in a response's headers.

    A module of two functions satisfies it (`llm_ports.adapters.litellm_proxy`).
    """

    def cost_from_headers(self, headers: Mapping[str, str]) -> Decimal | None:
        """Return the cost the gateway reported for the response, or None when it reported none.

        Raises:
            ValueError: The cost header is present but not a number. A caller treats it as absent
                and reports the bad header.
        """
        ...

    def answered_from_headers(self, headers: Mapping[str, str]) -> tuple[str | None, str | None]:
        """Return `(deployment id, upstream api base)` as the gateway reported them; None for each it did not."""
        ...


@runtime_checkable
class Limiter(Protocol):
    """In-process concurrency: at most N calls at once to one group of models.

    A call takes a slot INSIDE its deadline and OUTSIDE its retry loop: waiting spends the caller's
    deadline, and a retry keeps the slot it holds. Cross-process limits stay with whatever is in
    front of the model (a gateway answers 429 and the wire policy waits).

    Rationale: ADR-0021.
    """

    def slot(self, group: str) -> AbstractAsyncContextManager[float]:
        """Hold one slot of `group` for the block; the value is the seconds waited for it.

        Raises:
            SlotWaitExceeded: The wait passed the limiter's bound; the adapter reports
                `rate_limited` / `local_limiter`.
        """
        ...


__all__ = [
    "Citer",
    "Embedder",
    "Evaluator",
    "GatewayReader",
    "Generation",
    "Generator",
    "Grader",
    "Grounder",
    "Judge",
    "Limiter",
    "Pricer",
    "PromptSource",
    "Reranker",
    "Retriever",
    "Tracer",
]
