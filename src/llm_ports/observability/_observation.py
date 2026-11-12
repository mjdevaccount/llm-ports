"""One open generation observation over the `Tracer` port, and its lifecycle; `record` for a finished one.

`observe` opens one generation, yields a `PortObservation` that holds the last `update` (and, on
the package's own call path, the call's result), and ends the generation exactly once on EVERY
exit: a clean exit ends it with what was held; an exception at `level="ERROR"`; a cancellation or
an interrupt (`asyncio.CancelledError`, `KeyboardInterrupt`: a `BaseException`) at
`level="CANCELLED"`, with the latency and whatever was held so far. The exception then propagates.
`call.tracer_trace` is built on it, so a consumer that wraps its own model call shares one
lifecycle with the package's call path.

`record` is the same end for a model call the library did not make (a subprocess, a vendor's agent
loop): one call opens and ends one generation. Both build the end through one payload function, so
a sink sees the same fields from either.

Usage is the raw `{input, output}` pair as the provider reported it, each side coerced by `opt_int`.
`usage_counts()` hands the same pair to a second sink without re-reading the provider's reply.

Example:
    async with observe(tracer, name="summarize", model=m, provider="openai", input=prompt) as obs:
        reply = await client.call(...)
        obs.update(output=reply.text, usage_details={"input": reply.in_tok, "output": reply.out_tok})

Invariants:
    - Unmeasured usage is never zero: an absent or unreadable count is None, never 0.
    - A provider that reported neither count gets no usage block and `tokens_unmeasured: true` in
      the metadata. A reported zero stays a zero.
    - An observation is ended exactly once, whatever ends the block.

Rationale: ADR-0007, ADR-0025.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Protocol

from llm_ports.observability._payload import STATUS_MAX, UNMEASURED_KEY, deliver, end_payload
from llm_ports.ports import Answered, Cost, Failure, Finish, Generation, Tracer, Usage

if TYPE_CHECKING:
    from llm_ports.ports import Generated


def opt_int(v: Any) -> int | None:
    """Return `int(v)`, or None when the provider reported nothing readable.

    Replaces `int(v or 0)`, which reads "no usage block arrived" as "this call used zero tokens".
    """
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


class Observation(Protocol):
    """What a trace hook yields: one open observation of one call.

    Invariants:
        - Neither method raises.
    """

    def update(
        self,
        *,
        output: Any = None,
        usage_details: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Record what the call produced so far; a None argument leaves that field as it was."""
        ...

    def score(
        self, *, name: str, value: float, comment: str | None = None, metadata: Mapping[str, Any] | None = None
    ) -> None:
        """Attach a numeric score to the observation, with metadata stored on the score."""
        ...


class PortObservation:
    """An `Observation` over a `Generation`.

    Holds the last update (and the call's result, once `attach`ed) and ends the generation once,
    through `flush`, `fail` or `cancel`. Never raises.
    """

    def __init__(self, gen: Generation) -> None:
        self._gen = gen
        self._output: Any = None
        self._usage: tuple[int | None, int | None] = (None, None)
        self._usage_given = False
        self._meta: Mapping[str, Any] | None = None
        self._result: Generated[Any] | None = None
        self._ended = False

    def update(
        self,
        *,
        output: Any = None,
        usage_details: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Hold the call's output, usage and metadata; a None argument keeps the held value.

        Args:
            output: What the call produced.
            usage_details: The provider's `{"input": ..., "output": ...}` counts; each side is
                read through `opt_int`.
            metadata: Replaces the held metadata.
        """
        if output is not None:
            self._output = output
        if usage_details is not None:
            self._usage = (opt_int(usage_details.get("input")), opt_int(usage_details.get("output")))
            self._usage_given = True
        if metadata is not None:
            self._meta = metadata

    def attach(self, result: Generated[Any]) -> None:
        """Hold the call's result, so the end carries its usage, cost, finish, who answered and failure."""
        self._result = result

    def score(
        self, *, name: str, value: float, comment: str | None = None, metadata: Mapping[str, Any] | None = None
    ) -> None:
        """Attach a numeric score to the underlying generation."""
        if metadata is None:
            self._gen.score(name, float(value), comment)
        else:
            self._gen.score(name, float(value), comment, metadata)

    @property
    def metadata(self) -> Mapping[str, Any] | None:
        """The last update's metadata; None when no update carried any."""
        return self._meta

    def usage_counts(self) -> tuple[int | None, int | None]:
        """Return `(input, output)` exactly as reported; None where the provider said nothing."""
        return self._usage

    def _end(self, *, metadata: Mapping[str, Any] | None, level: str | None, status: str | None) -> None:
        if self._ended:
            return
        self._ended = True
        res = self._result
        pair = {"input": self._usage[0], "output": self._usage[1]} if self._usage_given else None
        deliver(
            self._gen,
            end_payload(
                output=self._output,
                usage=res.usage_v2 if res is not None else None,
                pair=pair,
                cost=res.cost if res is not None else None,
                finish=res.finish if res is not None else None,
                answered=res.answered if res is not None else None,
                failure=res.failure if res is not None else None,
                metadata=metadata,
                level=level,
                status_message=status,
            ),
        )

    def flush(self) -> None:
        """End the generation with what was held (the clean exit)."""
        self._end(metadata=self._meta, level="DEFAULT", status=None)

    def fail(self, exc: BaseException, metadata: Mapping[str, Any] | None = None) -> None:
        """End the generation at level ERROR with the error line, cut to `STATUS_MAX` characters.

        Args:
            exc: The exception that ended the call.
            metadata: The metadata to end with; None uses the last update's.
        """
        self._end(
            metadata=self._meta if metadata is None else metadata,
            level="ERROR",
            status=f"{type(exc).__name__}: {exc}"[:STATUS_MAX],
        )

    def cancel(self, exc: BaseException, metadata: Mapping[str, Any] | None = None) -> None:
        """End the generation at level CANCELLED: the caller cancelled or interrupted the call."""
        self._end(
            metadata=self._meta if metadata is None else metadata,
            level="CANCELLED",
            status=f"{type(exc).__name__}: {exc}".rstrip(": ")[:STATUS_MAX],
        )


@asynccontextmanager
async def observe(
    tracer: Tracer,
    *,
    name: str,
    model: str,
    provider: str,
    input: Any,
    metadata: Mapping[str, Any] | None = None,
    session_id: str | None = None,
    prompt_name: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncIterator[PortObservation]:
    """Open one generation around the block and end it once; see the module docstring.

    On an exception, the end carries the opening metadata, overlaid by the last update's, plus
    `latency_ms`; the exception then propagates. A cancellation or interrupt ends it the same way
    at level CANCELLED.

    Args:
        tracer: The tracer that opens the generation.
        name: The category a cost view groups by.
        model: The model id.
        provider: The provider name.
        input: What the model is sent.
        metadata: Opening metadata.
        session_id: When set, groups the generation under that session.
        prompt_name: When set, links the managed prompt of that name.
        clock: The monotonic clock the latency is measured on.

    Yields:
        The open `PortObservation`.
    """
    meta = dict(metadata or {})
    started = clock()
    obs = PortObservation(
        tracer.start_generation(
            name=name,
            model=model,
            provider=provider,
            input=input,
            metadata=meta,
            session_id=session_id,
            prompt_name=prompt_name,
        )
    )
    try:
        yield obs
    except Exception as exc:
        obs.fail(exc, {**meta, **(obs.metadata or {}), "latency_ms": int((clock() - started) * 1000)})
        raise
    except BaseException as exc:  # CancelledError, KeyboardInterrupt: never swallowed
        obs.cancel(exc, {**meta, **(obs.metadata or {}), "latency_ms": int((clock() - started) * 1000)})
        raise
    obs.flush()


def record(
    tracer: Tracer,
    *,
    name: str,
    model: str,
    provider: str,
    input: Any,
    output: Any,
    started_at: float,
    ended_at: float,
    usage: Usage | None = None,
    cost: Cost | None = None,
    finish: Finish | None = None,
    answered: Answered | None = None,
    failure: Failure | None = None,
    metadata: Mapping[str, Any] | None = None,
    session_id: str | None = None,
    prompt_name: str | None = None,
) -> None:
    """Record one finished generation in one call, for a model call the library did not make.

    A subprocess, a vendor's agent loop: whatever made the call reports it here, and every sink
    receives the same fields `generate` sends. Never raises.

    Args:
        tracer: The tracer that records it.
        name: The category a cost view groups by.
        model: The model id requested.
        provider: The provider name.
        input: What the model was sent.
        output: What it produced.
        started_at: When the call started, in seconds on any clock `ended_at` shares.
        ended_at: When it ended. Before `started_at`, the latency is omitted.
        usage: Its usage; None records it as unmeasured.
        cost: What it cost; None when not priced.
        finish: How its last reply ended.
        answered: Who answered.
        failure: Its failure; None when it succeeded (the level is then ERROR).
        metadata: Extra fields for the sink.
        session_id: When set, groups the generation under that session.
        prompt_name: When set, links the managed prompt of that name.
    """
    meta = dict(metadata or {})
    if ended_at >= started_at:
        meta["latency_ms"] = int((ended_at - started_at) * 1000)
    try:
        gen = tracer.start_generation(
            name=name,
            model=model,
            provider=provider,
            input=input,
            metadata=dict(metadata or {}),
            session_id=session_id,
            prompt_name=prompt_name,
        )
    except Exception:  # a Tracer never raises; a broken one does not break its caller either
        return
    payload = end_payload(
        output=output,
        usage=usage if usage is not None else Usage(requests=1, unmeasured=1),
        cost=cost,
        finish=finish,
        answered=answered,
        failure=failure,
        metadata=meta,
    )
    deliver(gen, payload)


__all__ = [
    "STATUS_MAX",
    "UNMEASURED_KEY",
    "Observation",
    "PortObservation",
    "observe",
    "opt_int",
    "record",
]
