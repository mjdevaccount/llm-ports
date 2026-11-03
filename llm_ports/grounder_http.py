"""The `Grounder` port over HTTP, to a grounder sidecar that runs the model in its own process.

This is the client half: one `POST {url}/verdict` per claim with `payload(claim, spans)`, and the
sidecar's JSON (`{supported, reason, score, checker}`) mapped back onto `GroundingVerdict`. A
verdict's `checker` is whatever the sidecar reports, so a reader sees which model spoke, not the
transport.

The port's `verdict` is synchronous, so this client makes a blocking httpx call; an async caller
runs it in a thread. Configuration (where the sidecar is, and whether there is one) belongs to the
consumer, which builds `HttpGrounder(url, timeout_s=...)` or skips the step.

Invariants:
    - A transport error, a non-200 or a malformed reply raises `GrounderUnavailable`. The caller
      decides how to fail soft (stop checking and record that it could not).
    - The client never guesses a verdict: an unreachable judge is never `supported`.

Rationale: ADR-0001.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from llm_ports.ports import Chunk, Claim, GroundingVerdict

_REASON_CAP = 300


class GrounderUnavailable(Exception):
    """The sidecar did not answer with a verdict: a transport error, a non-200 or a malformed reply."""


def payload(claim: Claim, spans: Sequence[Chunk]) -> dict[str, Any]:
    """Return the request body, the shape the sidecar's request model reads.

    Title and text travel separately, so the sidecar's `Grounder` reads them exactly as an
    in-process one would.
    """
    return {"claim": claim.text, "spans": [{"id": s.id, "title": s.title, "text": s.text} for s in spans]}


class HttpGrounder:
    """A `Grounder` whose model is behind `{url}/verdict`.

    Args:
        url: The sidecar's base URL; a trailing slash is dropped.
        timeout_s: The httpx timeout, in seconds, of the client this grounder builds.
        client: A ready httpx client, used instead of building one from `timeout_s`.

    Attributes:
        checker: "http" until the first verdict, then the sidecar's own checker name.

    Invariants:
        - Every verdict is the sidecar's, the no-spans rule included (its own `Grounder` enforces
          it): one judge, one `checker` name on every verdict.
        - A dead sidecar raises for a no-spans claim rather than answering for it.
    """

    checker = "http"  # replaced by the sidecar's own name on the first verdict

    def __init__(self, url: str, *, timeout_s: float = 20.0, client: httpx.Client | None = None) -> None:
        self.url = url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_s)

    def verdict(self, claim: Claim, spans: Sequence[Chunk]) -> GroundingVerdict:
        """Return the sidecar's verdict on `claim` against `spans`.

        Raises:
            GrounderUnavailable: A transport error, a non-200, or a malformed reply.
        """
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
            v = GroundingVerdict(
                supported,
                str(body.get("reason") or ""),
                score=None if score is None else float(score),
                checker=str(body.get("checker") or self.checker),
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise GrounderUnavailable(f"grounder bad payload: {exc}") from exc
        self.checker = v.checker
        return v


__all__ = ["GrounderUnavailable", "HttpGrounder", "payload"]
