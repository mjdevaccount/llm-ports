"""Test doubles for the Langfuse adapters (extra `[langfuse]`).

`LangfuseRig` is a real Langfuse SDK 5 client whose exporter is in memory. SDK 5 is
OpenTelemetry underneath, so the exported span is the wire, and `observed()` reads each span
back as the observation the server stores. Each rig builds under a fresh public key: the SDK's
resource manager is a singleton per key, and a reused key hands a later test an earlier test's
exporter. Scores are an ingestion-API call rather than a span, so the rig records `create_score`
calls.

`RefusingLangfuse` is a client whose server is down: every call raises `ConnectionError`, so a
test can prove an adapter over it never raises into the caller.

`FakeLangfusePrompts` fakes the two prompt-management calls `LangfusePromptSource` makes:
`get_prompt` returns what `create_prompt` last stored (versions numbered in order), else the
fallback.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NoReturn

from llm_ports._extras import extra

with extra("langfuse"):
    import httpx

if TYPE_CHECKING:  # imported for real when a rig is built, so the module imports without the SDK
    from langfuse import Langfuse
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

OBS = "langfuse.observation."
_HTTP: httpx.Client | None = None


def _http() -> httpx.Client:
    """Return the one HTTP client every rig shares.

    Building an `httpx.Client` loads the OS trust store, which is slow on some hosts.
    """
    global _HTTP
    if _HTTP is None:
        _HTTP = httpx.Client()
    return _HTTP


def observed(span: Any) -> dict[str, Any]:
    """Return one exported span as the observation the server stores.

    The `langfuse.*` attributes are unpacked: metadata is one JSON attribute per key, and usage is
    one JSON blob.

    Invariants:
        - Only keys the span carries appear; an absent field is absent, never None.
    """
    a = dict(span.attributes)
    out: dict[str, Any] = {"name": span.name}
    for key, name in (
        ("model.name", "model"),
        ("input", "input"),
        ("output", "output"),
        ("level", "level"),
        ("status_message", "status_message"),
        ("prompt.name", "prompt"),
    ):
        if OBS + key in a:
            out[name] = a[OBS + key]
    meta = {k[len(OBS + "metadata.") :]: json.loads(v) for k, v in a.items() if k.startswith(OBS + "metadata.")}
    if meta:
        out["metadata"] = meta
    if OBS + "usage_details" in a:
        out["usage"] = json.loads(a[OBS + "usage_details"])
    if "session.id" in a:
        out["session_id"] = a["session.id"]
    out["trace"] = (a.get("langfuse.trace.name"), list(a.get("langfuse.trace.tags", ())))
    return out


class LangfuseRig:
    """A real SDK 5 client exporting to memory.

    Attributes:
        exporter: The in-memory span exporter.
        client: The `Langfuse` client under test.
        scores: The keyword arguments of every `create_score` call, in call order.
    """

    def __init__(self) -> None:
        with extra("langfuse"):
            from langfuse import Langfuse
            from langfuse.span_filter import is_langfuse_span
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        self.exporter: InMemorySpanExporter = InMemorySpanExporter()
        self.client: Langfuse = Langfuse(
            public_key=f"pk-test-{uuid.uuid4().hex}",
            secret_key="sk-test",
            base_url="http://127.0.0.1:9",
            tracer_provider=TracerProvider(),
            should_export_span=is_langfuse_span,
            span_exporter=self.exporter,
            httpx_client=_http(),
        )
        self.scores: list[dict[str, Any]] = []
        self.client.create_score = lambda **kw: self.scores.append(kw)  # type: ignore[method-assign]  # recorded, not sent

    def spans(self) -> list[Any]:
        """Flush the client and return every finished span."""
        self.client.flush()
        return list(self.exporter.get_finished_spans())

    def attrs(self) -> dict[str, Any]:
        """Return the attributes of the one finished span.

        Raises:
            ValueError: The rig has not exported exactly one span.
        """
        (span,) = self.spans()
        return dict(span.attributes)

    def observed(self) -> list[dict[str, Any]]:
        """Return every finished span as an observation (see the module-level `observed`)."""
        return [observed(s) for s in self.spans()]


class RefusingLangfuse:
    """A Langfuse client whose server is down.

    Every SDK call the adapters make raises `ConnectionError` with `error` as its message.
    """

    error = "langfuse refused"

    def _refuse(self, *_a: Any, **_k: Any) -> NoReturn:
        raise ConnectionError(self.error)

    get_prompt: Callable[..., NoReturn] = _refuse
    create_prompt: Callable[..., NoReturn] = _refuse
    create_trace_id: Callable[..., NoReturn] = _refuse
    start_observation: Callable[..., NoReturn] = _refuse
    create_score: Callable[..., NoReturn] = _refuse


class FakeLangfusePrompts:
    """The prompt-management half of a Langfuse client, in memory.

    Attributes:
        stored: The prompt `get_prompt` returns, or None to return the fallback.
        created: The keyword arguments of every `create_prompt` call.
        fetches: `(name, label, type, cache_ttl_seconds)` for every `get_prompt` call.
    """

    def __init__(self, stored: SimpleNamespace | None = None) -> None:
        self.stored = stored
        self.created: list[dict[str, Any]] = []
        self.fetches: list[tuple[str, str, str, int]] = []

    def get_prompt(self, name: str, *, label: str, type: str, fallback: Any, cache_ttl_seconds: int) -> SimpleNamespace:
        """Record the fetch and return the stored prompt, or the fallback when nothing is stored."""
        self.fetches.append((name, label, type, cache_ttl_seconds))
        if self.stored is None:
            return SimpleNamespace(is_fallback=True, prompt=fallback, config={}, version=0)
        return self.stored

    def create_prompt(self, **kw: Any) -> None:
        """Record the call and store its prompt as the next version."""
        self.created.append(kw)
        self.stored = SimpleNamespace(
            is_fallback=False,
            prompt=kw["prompt"],
            config=kw["config"],
            version=(self.stored.version if self.stored else 0) + 1,
        )


__all__ = ["OBS", "FakeLangfusePrompts", "LangfuseRig", "RefusingLangfuse", "observed"]
