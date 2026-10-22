"""ports -- the seams a consumer owns; frameworks live behind them.

Zero-I/O: Protocols and frozen value types only. One adapter module per framework implements a
port; contract tests exercise the SEAM, so any adapter that passes them can replace any other.

Eight ports, named by the industry pattern they stand for:

- `Retriever`    -- RAG retrieval: query -> ranked chunks with provenance (hybrid + rerank is the
                    adapter's business; the port only promises order and provenance).
- `Citer`        -- citation-grounded generation: question + chunks -> claims, each citing ids
                    from the chunks it was given, never a URL it typed.
- `Grounder`     -- groundedness / faithfulness: one claim against the spans it cites -> a verdict.
- `Evaluator`    -- the eval harness: rows + graders -> one board.
- `PromptSource` -- prompt management: name + label -> the SOURCE template and where it came
                    from (repo seed or a managed version). Rendering is the caller's (Jinja for
                    packs); the port never compiles.
- `Tracer`       -- LLM observability: open one generation observation per model call, end it
                    with what the call produced, score it. SYNC and FAIL-OPEN (a sink that is down
                    returns a no-op `Generation`, never raises).
- `Generator`    -- typed structured generation: one prompt + an output type -> a validated value
                    or a named outcome. NEVER RAISES for a model, wire or gate failure. The call
                    path around it (role, pack, gate, trace, graders) is `llm_ports.call`.
- `Judge`        -- LLM-as-a-judge: one prompt + the reply it drew + a rubric -> a graded verdict.
                    RAISES on a model/wire failure: an unjudged sample must never read as a pass,
                    so the caller owns the catch, never the port.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

# ── value types (frozen; adapters translate their framework's objects INTO these) ───────────────


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit with its provenance. `score` is adapter-relative and ORDER-ONLY
    (box_rerank: raw logits; RRF: fused rank score) -- never threshold across adapters."""

    id: str
    text: str
    title: str = ""
    source: str = ""              # corpus/feed name (e.g. 'mnco', 'tear_feed', 'web')
    url: str = ""
    as_of: date | None = None     # the date the content speaks FOR; None = undated
    score: float | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Claim:
    """A sentence the writer asserts, with the chunk ids it rests on. Empty ids = ungrounded."""

    text: str
    evidence_ids: tuple[str, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)   # adapter extras per claim (e.g. tickers)


@dataclass(frozen=True)
class CitedAnswer:
    """What a Citer returns: claims (cited) plus the chunks it was given, so a Grounder can read
    the cited text without a second retrieval. `extras` is the adapter's own per-answer report
    (horizon, retries, box_ms, ...): a consumer that needs one reads it by name, the seam does not."""

    question: str
    claims: tuple[Claim, ...]
    chunks: Mapping[str, Chunk]   # id -> chunk, the universe the claims may cite
    model: str = ""               # adapter-reported; '' when the adapter has no model (stub)
    extras: Mapping[str, Any] = field(default_factory=dict)

    def unknown_ids(self) -> tuple[str, ...]:
        """Ids cited that were not in `chunks` -- a citer that invents ids fails the contract."""
        return tuple(i for c in self.claims for i in c.evidence_ids if i not in self.chunks)


@dataclass(frozen=True)
class GroundingVerdict:
    """`supported` is the gate; `score` is the adapter's own scale (a number-law check has
    none, an entailment model has a probability) -- report it, never compare across adapters."""

    supported: bool
    reason: str = ""
    score: float | None = None
    checker: str = ""             # which adapter spoke (number_law | hhem | lettucedetect | ...)


@dataclass(frozen=True)
class Prompt:
    """What a PromptSource returns: the SOURCE template (our renderer runs it, never the
    adapter's -- the narrator's Jinja has loops a mustache compiler cannot), plus provenance.
    LAW: `version is None` exactly when `source == "repo"` (the in-image seed served because the
    managed copy was absent, unreachable or disagreed); a managed copy carries its integer
    version and a source like `langfuse:v7`. `config` is the adapter's per-version extras
    (content sha, pack version, model role ...) -- read by name, never part of the seam.
    `system` is the SOURCE system template of a two-message (chat) prompt; `""` for a text prompt
    (the narrator pack, the macro classifier) -- the classifiers that send one read it here."""

    name: str
    text: str
    version: int | None = None
    source: str = "repo"
    config: Mapping[str, Any] = field(default_factory=dict)
    system: str = ""


#: Every way one generation can end. `gated` = the category switchboard refused it; `unavailable`
#: = the box/gateway failed and no input is to blame; `request_fault` = the request was refused
#: (400/413/422), so the input is to blame; `truncated` = the token ceiling cut the reply (a BUDGET
#: failure, never a parse error); `invalid` = the reply would not parse into the output type.
Outcome = Literal["ok", "gated", "unavailable", "request_fault", "truncated", "invalid"]


class JsonOutput(dict[str, Any]):
    """A hand-written JSON schema as a `Generator` output type: the value is a plain dict, the
    schema goes to the model unchanged. Never used directly -- `json_output(schema, name)` mints one
    subclass per schema, and each `Generator` adapter translates it into its framework's own
    structured-output object (pydantic-ai: `StructuredDict`). This is what keeps the facade
    (`call.json_output`) from importing an adapter."""

    json_schema: ClassVar[Mapping[str, Any]] = {}
    schema_name: ClassVar[str] = ""


def json_output(schema: Mapping[str, Any], name: str) -> type[JsonOutput]:
    """Pure. One `JsonOutput` subclass carrying `schema` and `name` (the schema's title, and the
    name a test stub dispatches on)."""
    return type(name, (JsonOutput,), {"json_schema": dict(schema), "schema_name": name})


@dataclass(frozen=True)
class GenRequest:
    """One model call, already resolved from a role: model id, rendered prompt, wire knobs.
    `images` are (bytes, mime) pairs for vision; `settings` reach the model verbatim (max_tokens,
    temperature, seed, top_p, extra_body). `box` = a self-hosted model group (an adapter may take
    a lease for it; a role sends it the thinking switch)."""

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
    """What a Generator returns. `value` is None exactly when `outcome != "ok"`. `problems` are the
    validator's findings on the KEPT value once the re-asks ran out (the caller prunes or drops);
    `retries` is None when the run died mid-flight (how many it took is unknown, never 0).
    `sources` are URLs the box FETCHED (research-server extension); None = the reply carried none."""

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


@dataclass(frozen=True)
class JudgeVerdict:
    """What a Judge returns. `passed` is the gate; `score` is the judge's own 0..1 reading and
    `reason` its one- or two-sentence justification (the Langfuse score comment)."""

    passed: bool
    score: float
    reason: str = ""
    judge: str = ""               # which adapter spoke (pydantic_evals | stub ...)


# ── ports ───────────────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class Retriever(Protocol):
    async def search(self, q: str, *, k: int = 8, date_from: date | None = None,
                     date_to: date | None = None) -> list[Chunk]:
        """Best first. An empty/blank `q` returns []. Never raises for 'nothing found'."""
        ...


@runtime_checkable
class Citer(Protocol):
    async def answer(self, q: str, chunks: Sequence[Chunk]) -> CitedAnswer:
        """Every claim's `evidence_ids` must be a subset of `{c.id for c in chunks}`
        (`CitedAnswer.unknown_ids()` is empty). No chunks -> no claims, not a refusal error."""
        ...


@runtime_checkable
class Grounder(Protocol):
    def verdict(self, claim: Claim, spans: Sequence[Chunk]) -> GroundingVerdict:
        """Pure. `spans` are the chunks the claim cites (already resolved by the caller).
        A claim with NO spans is never `supported`."""
        ...


Grader = Callable[[dict[str, Any]], tuple[bool, str]]


@runtime_checkable
class Evaluator(Protocol):
    def run(self, surface: str, rows: list[dict[str, Any]], graders: Mapping[str, Grader],
            *, id_field: str = "id",
            monitors: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None) -> dict[str, Any]:
        """One board with the keys `surface`, `n`, `rows_failed`, `by_grader` (per grader name:
        `pass`, `fail`, `examples` -- at most three `{id, reason}` wounds) and `monitors` -- the
        gauntlet shape, which every adapter must emit so the golden-set test, the on-box job and
        an ad-hoc read never disagree. `monitors` is the surface's own rates over ALL rows
        (narrator: speak_items, mean_retries, ...), carried VERBATIM; the key is always present
        and `{}` when the caller passed none (step 7', D-1)."""
        ...


@runtime_checkable
class PromptSource(Protocol):
    def get(self, name: str, *, label: str = "production") -> Prompt:
        """SYNC (the v2 Langfuse SDK is; async callers use `asyncio.to_thread`, as for Grounder).
        Never raises for 'the managed store is down' -- that is a `Prompt` with `source == "repo"`.
        `text` is never empty: an adapter with no seed for `name` has nothing to serve and RAISES."""
        ...


@runtime_checkable
class Generation(Protocol):
    """One open observation. `usage` is the RAW pair as the provider reported it --
    `{"input": int|None, "output": int|None}` -- or None when it reported neither (the caller
    decides what "unmeasured" means; the adapter decides the sink's key names). Neither method
    ever raises: telemetry must not break the call it describes."""

    def end(self, *, output: Any = None, usage: Mapping[str, int | None] | None = None,
            level: str = "DEFAULT", status_message: str | None = None,
            metadata: Mapping[str, Any] | None = None) -> None:
        """Close the observation. `level == "ERROR"` carries `status_message`; a None field is
        OMITTED from what the sink receives, never sent as null."""
        ...

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        """A NUMERIC score on this observation. Never raises."""
        ...


@runtime_checkable
class Tracer(Protocol):
    def start_generation(self, *, name: str, model: str, provider: str, input: Any,
                         metadata: Mapping[str, Any], session_id: str | None,
                         prompt_name: str | None) -> Generation:
        """SYNC (the v2 Langfuse SDK is; the async wrapper around a model call is the facade's).
        `name` is the category the cost surface groups by -- the adapter stores it verbatim.
        `session_id` set = group under a session; `prompt_name` set = link the managed prompt.
        Never raises: a sink that is down or refusing returns a no-op `Generation`."""
        ...


@runtime_checkable
class Generator(Protocol):
    async def generate[T](self, req: GenRequest, output: type[T], *,
                       validate: Callable[[T], Sequence[Any]] | None = None,
                       retry_message: Callable[[Sequence[Any]], str] | None = None) -> Generated[T]:
        """One call into `output` (a pydantic model, or `str` for prose). `validate` lists the
        problems with a parsed value; a non-empty list earns `req.output_retries` re-asks carrying
        `retry_message(problems)`, after which the value is KEPT with its problems. Never raises
        for a model, wire or gate failure: those are the `outcome`, with a one-line `reason`."""
        ...


@runtime_checkable
class Judge(Protocol):
    async def judge(self, *, prompt: str, output: Any, rubric: str) -> JudgeVerdict:
        """Grade `output` (the reply to `prompt`) against `rubric`. Raises when the judging model
        could not answer -- the caller decides what an unjudged sample means."""
        ...


__all__ = [
    "Chunk",
    "CitedAnswer",
    "Citer",
    "Claim",
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
    "Retriever",
    "Tracer",
    "json_output",
]
