"""A `TokenCounter` over a gateway's count endpoint (LiteLLM's `/utils/token_counter` shape).

The wire is `POST {url}` with `{model, messages, tools?}` in the OpenAI chat shape and a reply
that carries the count under one integer key (`total_tokens` by default). LiteLLM's proxy answers
it with the provider's own count API where the provider has one, else the model's tokenizer (its
`tokenizer_type`, reported as method `tokenizer`). Which models are exact is the consumer's
statement, not this module's guess: the table's `count_tokens` capability, passed as `models=`.

Shared by `adapters.http` (exported) and `adapters.pydantic_ai` (its factory builds one when a
model states `count_tokens`); adapters do not import each other, so it lives here.

Invariants:
    - `count` never raises: a transport error or a non-2xx reply is `Counted(None, "provider",
      failure=...)`, a 404/405/501 is `Counted(None, "unsupported")`, an unreadable reply is
      `Counted(None, "provider", failure=invalid_output)`.
    - A count is the endpoint's integer or None, never an estimate.
    - Tool results go in as the tool channel's text (the boundary marker is the adapter's, not
      the count's), so the count is of the history, not the wire bytes exactly.

Rationale: ADR-0024.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from typing import Any

from llm_ports._extras import extra
from llm_ports.adapters._classify import WireFacts, classify_core, kind_for_status
from llm_ports.ports import Counted, Message, Tool, failure
from llm_ports.ports.outcomes import REASON_MAX

with extra("http"):
    import httpx

#: The per-count timeout when the caller names none, in seconds.
COUNT_TIMEOUT_S = 5.0
#: Statuses that mean "this endpoint does not count": unsupported, not a failed count.
UNSUPPORTED_STATUS = frozenset({404, 405, 501})


def _message(m: Message) -> list[dict[str, Any]]:
    if m.role == "tool":
        return [
            {"role": "tool", "tool_call_id": p.call_id or "", "content": p.text or ""}
            for p in m.parts
            if p.kind == "tool_result"
        ]
    out: dict[str, Any] = {"role": m.role, "content": m.text}
    calls = [p for p in m.parts if p.kind == "tool_call"]
    if calls:
        out["tool_calls"] = [
            {
                "id": p.call_id or "",
                "type": "function",
                "function": {"name": p.name or "", "arguments": json.dumps(dict(p.args or {}))},
            }
            for p in calls
        ]
    return [out]


def count_payload(model: str, system: str, messages: Sequence[Message], tools: Sequence[Tool]) -> dict[str, Any]:
    """The count request body in the OpenAI chat shape. Pure."""
    msgs: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
    for m in messages:
        msgs.extend(_message(m))
    body: dict[str, Any] = {"model": model, "messages": msgs}
    if tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": dict(t.parameters)},
            }
            for t in tools
        ]
    return body


class HttpTokenCounter:
    """A `TokenCounter` over one count endpoint.

    Args:
        url: The full count URL, e.g. `http://gateway:4000/utils/token_counter`.
        api_key: Sent as a bearer token when set.
        key: The reply's integer key holding the count.
        models: The model ids this counter may count; others are `unsupported` with no request.
            None counts every model.
        timeout_s: The per-count timeout, in seconds.
        transport: Replaces httpx's own transport (a test's `MockTransport`).

    Raises:
        ValueError: At construction, when `url` is empty.
    """

    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        key: str = "total_tokens",
        models: Collection[str] | None = None,
        timeout_s: float = COUNT_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not url:
            raise ValueError("HttpTokenCounter: url is required")
        self.url = url
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._key = key
        self._models = frozenset(models) if models is not None else None
        self._timeout_s = timeout_s
        self._transport = transport

    async def count(
        self, model: str, *, system: str, messages: Sequence[Message], tools: Sequence[Tool] = ()
    ) -> Counted:
        """Count what `model` would read; the module invariants state the outcomes."""
        if self._models is not None and model not in self._models:
            return Counted(None, "unsupported")
        body = count_payload(model, system, messages, tools)
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s, transport=self._transport) as client:
                r = await client.post(self.url, json=body, headers=self._headers)
        except httpx.HTTPError as exc:
            got = classify_core(exc, WireFacts())
            assert got is not None  # an exception always classifies
            return Counted(None, "provider", failure=replace(got, reason=f"count transport: {got.reason}"[:REASON_MAX]))
        if r.status_code in UNSUPPORTED_STATUS:
            return Counted(None, "unsupported")
        if r.status_code >= 300:
            kind = kind_for_status(r.status_code)
            return Counted(
                None, "provider", failure=failure(kind, status=r.status_code, reason=f"count HTTP {r.status_code}")
            )
        return _read(r, self._key)


def _read(r: httpx.Response, key: str) -> Counted:
    try:
        reply = r.json()
    except ValueError:
        return Counted(None, "provider", failure=failure("invalid_output", reason="count reply is not JSON"))
    n = reply.get(key) if isinstance(reply, Mapping) else None
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        return Counted(
            None, "provider", failure=failure("invalid_output", reason=f"count reply has no integer {key!r}")
        )
    return Counted(n, "tokenizer" if isinstance(reply.get("tokenizer_type"), str) else "provider")


__all__ = ["COUNT_TIMEOUT_S", "UNSUPPORTED_STATUS", "HttpTokenCounter", "count_payload"]
