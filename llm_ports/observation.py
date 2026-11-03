"""observation -- one open generation observation over the `Tracer` port, and its lifecycle.

    async with observe(tracer, name="summarize", model=m, provider="openai", input=prompt) as obs:
        reply = await client.call(...)
        obs.update(output=reply.text, usage_details={"input": reply.in_tok, "output": reply.out_tok})

`observe` opens ONE generation, yields a `PortObservation` that holds the last `update`, and ends
the generation exactly once: on a clean exit with what was held, on an exception at level ERROR
with the error line and the measured latency (the exception propagates). `call.tracer_trace` is
built on it, so a consumer that wraps its own model call and the package's call path share one
lifecycle.

UNMEASURED IS NOT FREE. Usage is the raw `{input, output}` pair as the provider reported it, each
side coerced by `opt_int` (absent or unreadable -> None, never 0). A provider that reported NEITHER
count gets no usage block at all and `tokens_unmeasured: true` in the metadata, so a sink never
books an unmetered call as zero tokens; a reported zero stays a zero. `usage_counts()` hands the
same pair to a second sink (an OpenTelemetry span) without re-reading the provider's reply.
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
    """int(v), or None when the provider reported nothing readable -- the replacement for
    `int(v or 0)`, which turns "no usage block arrived" into "this call used zero tokens"."""
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


class Observation(Protocol):
    """What a trace hook yields: one open observation of one call. Neither method may raise."""

    def update(
        self,
        *,
        output: Any = None,
        usage_details: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None: ...

    def score(self, *, name: str, value: float, comment: str | None = None) -> None: ...


class PortObservation:
    """An `Observation` over a `Generation`: holds the last update, ends the generation ONCE
    (`flush` or `fail`). Never raises -- `Generation.end` / `.score` do not."""

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
        if output is not None:
            self._output = output
        if usage_details is not None:
            self._usage = (opt_int(usage_details.get("input")), opt_int(usage_details.get("output")))
            self._usage_given = True
        if metadata is not None:
            self._meta = metadata

    def score(self, *, name: str, value: float, comment: str | None = None) -> None:
        self._gen.score(name, float(value), comment)

    @property
    def metadata(self) -> Mapping[str, Any] | None:
        """The last `update`'s metadata (None = never updated)."""
        return self._meta

    def usage_counts(self) -> tuple[int | None, int | None]:
        """(input, output) exactly as reported -- None where the provider said nothing."""
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
        """End the generation at level ERROR with the error line (capped). `metadata` None = the
        last update's."""
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
    """One generation around the block (module docstring). On an exception the ERROR end carries
    the opening metadata, then the last update's, then `latency_ms`."""
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
