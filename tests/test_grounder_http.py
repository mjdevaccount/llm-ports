"""grounder_http: the client half of a grounder sidecar over httpx.MockTransport (no sidecar, no
model) -- the wire shape, the verdict mapping, and every way the judge can fail to answer raising
GrounderUnavailable (never a guessed verdict)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from llm_ports import grounder_http as gh
from llm_ports.ports import Chunk, Claim, Grounder

SPAN = Chunk(id="a", text="SPY closed at 580.5", title="Close")


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _ok(body: dict[str, Any]) -> httpx.Client:
    return _client(lambda req: httpx.Response(200, json=body))


def test_posts_the_claim_and_spans_and_maps_the_verdict_back() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"supported": False, "reason": "581", "score": 0.1, "checker": "lettucedetect"})

    g = gh.HttpGrounder("http://grounder:8094/", client=_client(handler))
    v = g.verdict(Claim("SPY at 581", ("a",)), [SPAN])
    assert v.supported is False and v.reason == "581" and v.score == 0.1 and v.checker == "lettucedetect"
    assert g.checker == "lettucedetect", "the sidecar's name replaces the transport's"
    assert seen[0].url == "http://grounder:8094/verdict"
    assert json.loads(seen[0].content) == {
        "claim": "SPY at 581",
        "spans": [{"id": "a", "title": "Close", "text": "SPY closed at 580.5"}],
    }


def test_no_spans_is_the_sidecars_answer_not_a_client_guess() -> None:
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(
            200, json={"supported": False, "reason": "no evidence spans", "score": None, "checker": "lettucedetect"}
        )

    v = gh.HttpGrounder("http://g", client=_client(handler)).verdict(Claim("x"), [])
    assert v.supported is False and v.checker == "lettucedetect" and seen == [{"claim": "x", "spans": []}]


def test_a_null_score_stays_none() -> None:
    v = gh.HttpGrounder(
        "http://g", client=_ok({"supported": False, "reason": "r", "score": None, "checker": "lettucedetect"})
    ).verdict(Claim("x", ("a",)), [SPAN])
    assert v.score is None and v.supported is False


def _raise(_req: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused")


@pytest.mark.parametrize(
    ("handler", "match"),
    [
        (_raise, "transport"),
        (lambda req: httpx.Response(503, json={"detail": {"status": "loading"}}), "HTTP 503"),
        (lambda req: httpx.Response(200, content=b"not json"), "bad payload"),
        (lambda req: httpx.Response(200, json={"reason": "no supported key"}), "bad payload"),
        (lambda req: httpx.Response(200, json={"supported": "yes"}), "bad payload"),
    ],
)
def test_every_non_answer_raises_unavailable(handler: Callable[[httpx.Request], httpx.Response], match: str) -> None:
    with pytest.raises(gh.GrounderUnavailable, match=match):
        gh.HttpGrounder("http://g", client=_client(handler)).verdict(Claim("x", ("a",)), [SPAN])


def test_the_client_satisfies_the_port() -> None:
    assert isinstance(gh.HttpGrounder("http://g", client=_ok({})), Grounder)
