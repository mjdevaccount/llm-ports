"""testing_langfuse -- test doubles for the Langfuse adapters (extra `[langfuse]`).

`LangfuseRig` is a REAL SDK 5 client whose exporter is in memory: SDK 5 is OpenTelemetry
underneath, so the wire IS the exported span, and `observed()` reads each one back as the
observation the server stores. Each rig builds under a fresh public key, because the SDK's
resource manager is a singleton per key and a reused key would hand a later test an earlier one's
exporter. Scores are an ingestion-API call, not a span, so the rig records `create_score` calls.

`RefusingLangfuse` is a client whose server is down: every call raises `ConnectionError`, so a
test can prove the adapter over it never raises into the caller.

`FakeLangfusePrompts` stands in for the two prompt-management calls `LangfusePromptSource` makes:
`get_prompt` returns what `create_prompt` last stored (versions numbered in order), else the
fallback.
"""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from typing import Any

import httpx

OBS = "langfuse.observation."
_HTTP: httpx.Client | None = None


def _http() -> httpx.Client:
    """One HTTP client for every rig: building one loads the OS trust store (slow on some hosts)."""
    global _HTTP
    if _HTTP is None:
        _HTTP = httpx.Client()
    return _HTTP


def observed(span: Any) -> dict[str, Any]:
    """One exported span as the observation the server stores: the `langfuse.*` keys unpacked
    (metadata is one JSON attribute PER KEY; usage is one JSON blob). Only keys the span carries
    appear -- an absent field is absent, never None."""
    a = dict(span.attributes)
    out: dict[str, Any] = {"name": span.name}
    for key, name in (("model.name", "model"), ("input", "input"), ("output", "output"),
                      ("level", "level"), ("status_message", "status_message"),
                      ("prompt.name", "prompt")):
        if OBS + key in a:
            out[name] = a[OBS + key]
    meta = {k[len(OBS + "metadata."):]: json.loads(v) for k, v in a.items()
            if k.startswith(OBS + "metadata.")}
    if meta:
        out["metadata"] = meta
    if OBS + "usage_details" in a:
        out["usage"] = json.loads(a[OBS + "usage_details"])
    if "session.id" in a:
        out["session_id"] = a["session.id"]
    out["trace"] = (a.get("langfuse.trace.name"), list(a.get("langfuse.trace.tags", ())))
    return out


class LangfuseRig:
    """A real SDK 5 client exporting to memory, plus `scores` (every `create_score` call's kwargs)."""

    def __init__(self) -> None:
        from langfuse import Langfuse  # type: ignore[import-untyped]
        from langfuse.span_filter import is_langfuse_span  # type: ignore[import-untyped]
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        self.exporter = InMemorySpanExporter()
        self.client = Langfuse(public_key=f"pk-test-{uuid.uuid4().hex}", secret_key="sk-test",
                               base_url="http://127.0.0.1:9", tracer_provider=TracerProvider(),
                               should_export_span=is_langfuse_span, span_exporter=self.exporter,
                               httpx_client=_http())
        self.scores: list[dict[str, Any]] = []
        self.client.create_score = lambda **kw: self.scores.append(kw)

    def spans(self) -> list[Any]:
        self.client.flush()
        return list(self.exporter.get_finished_spans())

    def attrs(self) -> dict[str, Any]:
        (span,) = self.spans()
        return dict(span.attributes)

    def observed(self) -> list[dict[str, Any]]:
        return [observed(s) for s in self.spans()]


class RefusingLangfuse:
    """A Langfuse client whose server is down: every SDK call the adapters make raises."""

    error = "langfuse refused"

    def _refuse(self, *_a: Any, **_k: Any) -> Any:
        raise ConnectionError(self.error)

    get_prompt = create_prompt = create_trace_id = start_observation = create_score = _refuse


class FakeLangfusePrompts:
    """The prompt-management half of a Langfuse client, in memory. `created` keeps every
    `create_prompt` kwargs; `fetches` keeps (name, label, type, cache_ttl_seconds) per fetch."""

    def __init__(self, stored: SimpleNamespace | None = None) -> None:
        self.stored = stored
        self.created: list[dict[str, Any]] = []
        self.fetches: list[tuple[str, str, str, int]] = []

    def get_prompt(self, name: str, *, label: str, type: str, fallback: Any,
                   cache_ttl_seconds: int) -> SimpleNamespace:
        self.fetches.append((name, label, type, cache_ttl_seconds))
        if self.stored is None:
            return SimpleNamespace(is_fallback=True, prompt=fallback, config={}, version=0)
        return self.stored

    def create_prompt(self, **kw: Any) -> None:
        self.created.append(kw)
        self.stored = SimpleNamespace(is_fallback=False, prompt=kw["prompt"], config=kw["config"],
                                      version=(self.stored.version if self.stored else 0) + 1)


__all__ = ["OBS", "FakeLangfusePrompts", "LangfuseRig", "RefusingLangfuse", "observed"]
