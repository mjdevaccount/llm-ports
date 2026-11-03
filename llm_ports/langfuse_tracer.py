"""The `Tracer` port over the Langfuse Python SDK 5 (extra `[langfuse]`).

Example:
    tracer = LangfuseTracer(Langfuse(...))  # the caller builds the client
    async with observe(tracer, name="summarize", model=m, provider="openai", input=prompt) as obs:
        ...

This adapter owns the generation lifecycle, the usage block, `end`, `score` and the managed-prompt
link. The caller owns building the client (keys, host, its OpenTelemetry provider and export
filter) and what goes into the calls (names, metadata).

Langfuse SDK 5 is OpenTelemetry underneath:

- A generation is `start_observation(as_type="generation")`, opened as a root in the current OTel
  trace (`trace_context={"trace_id": ...}`), so the Langfuse trace id is the id every other span of
  that request carries. Outside a span a fresh trace id is minted.
- Trace-level fields (session, trace name, tags) ride `propagate_attributes` around the open.
  `name` is stored verbatim as the generation name, the trace name and the one tag, so a cost
  dashboard can group by it.
- Usage is `usage_details={"input", "output"}`; the server adds `total` and prices by model.
- Export is a background batch: `end` does not block on the network.

`provider` has no Langfuse field; it rides `metadata["provider"]` unless the caller set that key.

Invariants:
    - When neither token count is reported, no usage block is sent, never zero tokens; when one
      side is reported, the other is booked as 0.
    - Fails open: a client that raises on open yields a `NoopGeneration`.
    - Every failure (open, end, score) emits at most one warning event per `warn_hold_s` through
      `on_event` (default: the stdlib logger `llm_ports.langfuse_tracer`), and never raises into
      the call it describes.
    - The prompt link is best effort and silent.

Rationale: ADR-0007.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

from langfuse import propagate_attributes  # type: ignore[import-untyped]
from opentelemetry import trace as otel_trace

from llm_ports.call import EventHook
from llm_ports.ports import Generation
from llm_ports.reference import NoopGeneration

DEFAULT_PROMPT_LABEL = "production"
DEFAULT_PROMPT_CACHE_TTL_S = 300
DEFAULT_WARN_HOLD_S = 300
EVENT = "langfuse_tracer"
_ERR_CAP = 200
_log = logging.getLogger("llm_ports.langfuse_tracer")


def _log_event(level: int, event: str, fields: Mapping[str, Any]) -> None:
    _log.log(level, "%s %s", event, dict(fields))


def usage_details(prompt: int | None, completion: int | None) -> dict[str, int] | None:
    """Return the Langfuse usage block, or None when the provider reported neither count.

    When only one count is reported, the other is booked as 0. Pure.

    Rationale: ADR-0007.
    """
    if prompt is None and completion is None:
        return None
    return {"input": prompt or 0, "output": completion or 0}


def trace_id(client: Any) -> str:
    """Return the current OTel trace id as 32 hex digits, or a fresh one from the client outside a span."""
    ctx = otel_trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else client.create_trace_id()


class LangfuseGeneration:
    """`Generation` over an SDK 5 `LangfuseGeneration`.

    `end` is the one terminal call: its fields go in a single `update`, then `end()` closes the
    span. Failures go to `warn`, never raised.
    """

    def __init__(self, gen: Any, warn: Callable[[str, Exception], None]) -> None:
        self._gen = gen
        self._warn = warn

    def end(
        self,
        *,
        output: Any = None,
        usage: Mapping[str, int | None] | None = None,
        level: str = "DEFAULT",
        status_message: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Close the generation with its output, usage and metadata; never raises.

        Usage with neither count reported sends no usage block. Only `level="ERROR"` sends a level,
        together with `status_message`.
        """
        kw: dict[str, Any] = {}
        if output is not None:
            kw["output"] = output
        u = usage_details(usage.get("input"), usage.get("output")) if usage is not None else None
        if u is not None:
            kw["usage_details"] = u
        if metadata is not None:
            kw["metadata"] = metadata
        if level == "ERROR":
            kw["level"] = "ERROR"
            kw["status_message"] = status_message
        try:
            if kw:
                self._gen.update(**kw)
            self._gen.end()
        except Exception as exc:
            self._warn("end_failed", exc)

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        """Attach a numeric score to the generation; never raises."""
        # The port's value is a float, so the score's data type is `NUMERIC` (summable on a dashboard).
        try:
            self._gen.score(name=name, value=value, data_type="NUMERIC", comment=comment)
        except Exception as exc:
            self._warn("score_failed", exc)


class LangfuseTracer:
    """`Tracer` whose sink is Langfuse.

    Build one per client; the SDK client is per process.

    Args:
        client: A Langfuse SDK 5 client, built by the caller.
        prompt_label: The managed-prompt label a generation links to.
        prompt_cache_ttl_s: The SDK's cache TTL for that prompt fetch.
        warn_hold_s: The minimum seconds between two warning events.
        clock: A monotonic clock, injectable for tests.
        on_event: The warning sink; defaults to the stdlib logger `llm_ports.langfuse_tracer`.
    """

    def __init__(
        self,
        client: Any,
        *,
        prompt_label: str = DEFAULT_PROMPT_LABEL,
        prompt_cache_ttl_s: int = DEFAULT_PROMPT_CACHE_TTL_S,
        warn_hold_s: float = DEFAULT_WARN_HOLD_S,
        clock: Callable[[], float] = time.monotonic,
        on_event: EventHook | None = None,
    ) -> None:
        self._client = client
        self._label = prompt_label
        self._ttl = prompt_cache_ttl_s
        self._hold = warn_hold_s
        self._clock = clock
        self._on_event = on_event or _log_event
        self._quiet_until = 0.0

    def _warn(self, action: str, exc: Exception) -> None:
        """Emit one warning event per `warn_hold_s`."""
        now = self._clock()
        if now < self._quiet_until:
            return
        self._quiet_until = now + self._hold
        self._on_event(
            logging.WARNING,
            EVENT,
            {"action": action, "error": f"{type(exc).__name__}: {exc}"[:_ERR_CAP], "held_s": self._hold},
        )

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
        """Open a generation as a root span in the current OTel trace; never raises.

        With `prompt_name`, the generation links to the prompt's labelled version (a cached fetch),
        so the prompt's generation count follows the live version across re-syncs. Any failure to
        open returns a `NoopGeneration`.
        """
        # Best effort: a failed prompt fetch leaves the generation unlinked.
        lf_prompt = None
        if prompt_name:
            with contextlib.suppress(Exception):
                lf_prompt = self._client.get_prompt(prompt_name, label=self._label, cache_ttl_seconds=self._ttl)
        # Langfuse has no provider field: it rides the metadata (unless the caller already set one).
        meta = {"provider": provider, **metadata} if provider else dict(metadata)
        gkw: dict[str, Any] = {"name": name, "as_type": "generation", "model": model, "input": input, "metadata": meta}
        if lf_prompt is not None:
            gkw["prompt"] = lf_prompt  # increments the prompt's generation count
        try:
            gkw["trace_context"] = {"trace_id": trace_id(self._client)}
            with propagate_attributes(session_id=session_id, trace_name=name, tags=[name]):
                return LangfuseGeneration(self._client.start_observation(**gkw), self._warn)
        except Exception as exc:
            self._warn("open_failed", exc)
            return NoopGeneration()


__all__ = [
    "DEFAULT_PROMPT_CACHE_TTL_S",
    "DEFAULT_PROMPT_LABEL",
    "DEFAULT_WARN_HOLD_S",
    "EVENT",
    "LangfuseGeneration",
    "LangfuseTracer",
    "trace_id",
    "usage_details",
]
