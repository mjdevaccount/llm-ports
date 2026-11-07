"""One open generation observation over the `Tracer` port, and its lifecycle.

`observe` opens one generation, yields a `PortObservation` that holds the last `update`, and ends
the generation exactly once: on a clean exit with what was held; on an exception at `level="ERROR"`
with the error line and the measured latency, after which the exception propagates.
`call.tracer_trace` is built on it, so a consumer that wraps its own model call shares one
lifecycle with the package's call path.

Usage is the raw `{input, output}` pair as the provider reported it, each side coerced by `opt_int`.
`usage_counts()` hands the same pair to a second sink (an OpenTelemetry span) without re-reading
the provider's reply.

Example:
    async with observe(tracer, name="summarize", model=m, provider="openai", input=prompt) as obs:
        reply = await client.call(...)
        obs.update(output=reply.text, usage_details={"input": reply.in_tok, "output": reply.out_tok})

Invariants:
    - Unmeasured usage is never zero: an absent or unreadable count is None, never 0.
    - A provider that reported neither count gets no usage block and `tokens_unmeasured: true` in
      the metadata. A reported zero stays a zero.

Rationale: ADR-0007.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any, Protocol

from llm_ports.ports import Generation, Tracer

UNMEASURED_KEY = "tokens_unmeasured"
STATUS_MAX = 500


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

    Holds the last update and ends the generation once, through `flush` or `fail`. Never raises,
    because `Generation.end` and `Generation.score` do not.
    """

    def __init__(self, gen: Generation) -> None:
        self._gen = gen
        self._output: Any = None
        self._usage: tuple[int | None, int | None] = (None, None)
        self._usage_given = False
        self._meta: Mapping[str, Any] | None = None

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

    def flush(self) -> None:
        """End the generation with what was held (the clean exit)."""
        unmeasured = self._usage == (None, None)
        meta = self._meta
        if unmeasured and self._usage_given:
            meta = {**(meta or {}), UNMEASURED_KEY: True}
        usage = None if unmeasured else {"input": self._usage[0], "output": self._usage[1]}
        self._gen.end(output=self._output, usage=usage, metadata=meta)

    def fail(self, exc: BaseException, metadata: Mapping[str, Any] | None = None) -> None:
        """End the generation at level ERROR with the error line, cut to `STATUS_MAX` characters.

        Args:
            exc: The exception that ended the call.
            metadata: The metadata to end with; None uses the last update's.
        """
        self._gen.end(
            level="ERROR",
            status_message=f"{type(exc).__name__}: {exc}"[:STATUS_MAX],
            metadata=self._meta if metadata is None else metadata,
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

    On an exception, the `level="ERROR"` end carries the opening metadata, overlaid by the last update's,
    plus `latency_ms`; the exception then propagates.

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
    obs.flush()


__all__ = ["STATUS_MAX", "UNMEASURED_KEY", "Observation", "PortObservation", "observe", "opt_int"]
