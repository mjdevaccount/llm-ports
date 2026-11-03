"""HttpGrounder -- the `Grounder` port over HTTP to a grounder SIDECAR (the model in its own process).

A grounding model (e.g. `LettuceDetectGrounder`: torch + a 575 MB checkpoint) belongs in its own
container, not in every image that wants a verdict. This is the client half: one
`POST {url}/verdict` per claim with `payload(claim, spans)`, the sidecar's JSON
(`{supported, reason, score, checker}`) mapped back onto `GroundingVerdict`. Its `checker` is
whatever the sidecar reports, so a reader of a verdict sees which model spoke, not the transport.

Fails LOUD with `GrounderUnavailable` on transport / non-200 / a malformed reply. The CALLER is
what fails soft (stops checking and records that it could not), and the consumer's watcher reads
that record. This client never guesses a verdict -- an unreachable judge is not a `supported`.

The port's `verdict` is synchronous; this client is a blocking httpx call on purpose, and an async
caller runs it in a thread. Configuration (where the sidecar is, whether there is one) is the
consumer's: it builds `HttpGrounder(url, timeout_s=...)` or skips the step.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from llm_ports.ports import Chunk, Claim, GroundingVerdict

_REASON_CAP = 300


class GrounderUnavailable(Exception):
    """The sidecar did not answer with a verdict (transport, non-200, bad payload)."""


def payload(claim: Claim, spans: Sequence[Chunk]) -> dict[str, Any]:
    """The wire shape, shared with the sidecar's request model: title and text ride separately
    so the sidecar's Grounder reads them exactly as an in-process one would."""
    return {"claim": claim.text,
            "spans": [{"id": s.id, "title": s.title, "text": s.text} for s in spans]}


class HttpGrounder:
    """`Grounder` whose model is behind `{url}/verdict`. EVERY verdict is the sidecar's, the
    no-spans law included (its own Grounder enforces it): one judge, one `checker` name on every
    verdict. A dead sidecar RAISES for a no-spans claim rather than answering for it."""

    checker = "http"      # replaced by the sidecar's own name on the first verdict

    def __init__(self, url: str, *, timeout_s: float = 20.0,
                 client: httpx.Client | None = None) -> None:
        self.url = url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_s)

    def verdict(self, claim: Claim, spans: Sequence[Chunk]) -> GroundingVerdict:
        try:
            r = self._client.post(f"{self.url}/verdict", json=payload(claim, spans))
        except httpx.HTTPError as exc:
            raise GrounderUnavailable(f"grounder transport: {exc}"[:_REASON_CAP]) from exc
        if r.status_code != 200:
            raise GrounderUnavailable(f"grounder HTTP {r.status_code}: {r.text[:120]}")
        try:
            body = r.json()
            supported = body["supported"]
            if not isinstance(supported, bool):
                raise TypeError("supported is not a bool")
            score = body.get("score")
            v = GroundingVerdict(supported, str(body.get("reason") or ""),
                                 score=None if score is None else float(score),
                                 checker=str(body.get("checker") or self.checker))
        except (ValueError, KeyError, TypeError) as exc:
            raise GrounderUnavailable(f"grounder bad payload: {exc}") from exc
        self.checker = v.checker
        return v


__all__ = ["GrounderUnavailable", "HttpGrounder", "payload"]
