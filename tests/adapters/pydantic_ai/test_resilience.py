"""Resilience acceptance tests 1-8 (spec resilience): the one wire-retry loop, the deadline, the limiter.

Every test counts what reached the wire, through `openai_compatible` over an httpx MockTransport.
Backoff waits are recorded, not slept (the conftest `_backoff_does_not_sleep` fixture), so a test
asserts the waits a policy chose without spending them.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import replace
from typing import Any

import httpx
import pytest
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from llm_ports.adapters import _retry
from llm_ports.adapters import pydantic_ai as pag
from llm_ports.adapters.pydantic_ai import PydanticAIGenerator
from llm_ports.call.resilience import LocalLimiter, RetryPolicy
from llm_ports.ports import Deadline, GenRequest, RetryRule, WirePolicy

REQ = GenRequest(model="small-chat", prompt="narrate", output_retries=1)
FLAT = RetryRule(attempts=3, wait_initial_s=0.5, wait_max_s=8.0, multiplier=2.0, jitter=0.0)
POLICY = WirePolicy(rules={"rate_limited": FLAT, "overloaded": FLAT, "server": FLAT})


class _Note(BaseModel):
    headline: str


def _ok(headline: str = "pinned") -> tuple[int, dict[str, Any], dict[str, str]]:
    content = json.dumps({"headline": headline})
    body = {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "small-chat",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }
    return 200, body, {}


def _err(
    status: int, headers: dict[str, str] | None = None, **error: Any
) -> tuple[int, dict[str, Any], dict[str, str]]:
    return status, {"error": {"message": "nope", **error}}, headers or {}


class Wire:
    """A MockTransport that answers from a script and counts what reached it."""

    def __init__(self, *replies: tuple[int, dict[str, Any], dict[str, str]]) -> None:
        self.replies = list(replies)
        self.sent: list[dict[str, Any]] = []
        self.gate: asyncio.Event | None = None
        self.live = 0
        self.peak = 0
        self.transport = httpx.MockTransport(self._handle)

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        self.sent.append(json.loads(request.content))
        n = len(self.sent)
        self.live += 1
        self.peak = max(self.peak, self.live)
        try:
            if self.gate is not None:
                await self.gate.wait()
            await asyncio.sleep(0)
        finally:
            self.live -= 1
        code, body, headers = self.replies[min(n - 1, len(self.replies) - 1)]
        return httpx.Response(code, json=body, headers=headers)


def _gen(wire: Wire, policy: WirePolicy | None = POLICY, **kw: Any) -> PydanticAIGenerator:
    return PydanticAIGenerator(
        model_for=pag.openai_compatible("http://gateway.test/v1", "k", transport=wire.transport), policy=policy, **kw
    )


# -- 1: retries follow the rule ------------------------------------------------------------------


async def test_1_two_429s_then_200_is_ok_after_three_wire_requests(_backoff_does_not_sleep: list[float]) -> None:
    wire = Wire(_err(429), _err(429), _ok())
    res = await _gen(wire).generate(REQ, _Note)
    assert res.outcome == "ok" and len(wire.sent) == 3
    assert [a.failure.kind for a in res.attempts] == ["rate_limited", "rate_limited"]
    assert _backoff_does_not_sleep == [0.5, 1.0] == [a.waited_s for a in res.attempts]
    assert res.stamp["wire_requests"] == 3


async def test_1_a_kind_whose_tries_run_out_ends_with_its_failure() -> None:
    wire = Wire(_err(429))
    res = await _gen(wire).generate(REQ, _Note)
    assert res.failure is not None and res.failure.kind == "rate_limited" and len(wire.sent) == 3


async def test_1_the_total_cap_stops_a_call_whose_kinds_each_have_tries_left() -> None:
    capped = replace(POLICY, max_total_attempts=2)
    wire = Wire(_err(429), _err(503), _ok())
    res = await _gen(wire, capped).generate(REQ, _Note)
    assert res.failure is not None and res.failure.kind == "server" and len(wire.sent) == 2


# -- 2: the server's hints -----------------------------------------------------------------------


async def test_2_retry_after_ms_beats_retry_after(_backoff_does_not_sleep: list[float]) -> None:
    wire = Wire(_err(429, {"retry-after-ms": "1500", "retry-after": "9"}), _ok())
    res = await _gen(wire).generate(REQ, _Note)
    assert res.outcome == "ok" and _backoff_does_not_sleep == [1.5]


async def test_2_a_hint_above_the_cap_stops_and_keeps_the_hint(_backoff_does_not_sleep: list[float]) -> None:
    capped = WirePolicy(rules={"rate_limited": replace(FLAT, retry_after_max_s=60.0)})
    wire = Wire(_err(429, {"retry-after": "600"}), _ok())
    res = await _gen(wire, capped).generate(REQ, _Note)
    assert res.failure is not None and res.failure.kind == "rate_limited" and res.failure.retry_after_s == 600.0
    assert len(wire.sent) == 1 and _backoff_does_not_sleep == []


async def test_2_x_should_retry_false_stops_a_503_and_without_it_the_503_is_retried() -> None:
    told = Wire(_err(503, {"x-should-retry": "false"}), _ok())
    assert (await _gen(told).generate(REQ, _Note)).outcome == "unavailable" and len(told.sent) == 1
    plain = Wire(_err(503), _ok())
    assert (await _gen(plain).generate(REQ, _Note)).outcome == "ok" and len(plain.sent) == 2


# -- 3: a spent quota is never retried (both ways) -----------------------------------------------


async def test_3_quota_is_sent_once_and_the_same_429_without_the_marker_is_retried() -> None:
    every = WirePolicy(rules=dict.fromkeys(("rate_limited", "overloaded", "server", "transport", "timeout"), FLAT))
    spent = Wire(_err(429, code="insufficient_quota", type="insufficient_quota"), _ok())
    res = await _gen(spent, every).generate(REQ, _Note)
    assert res.failure is not None and res.failure.kind == "quota" and len(spent.sent) == 1
    busy = Wire(_err(429), _ok())
    assert (await _gen(busy, every).generate(REQ, _Note)).outcome == "ok" and len(busy.sent) == 2


def test_3_quota_cannot_be_given_a_retry_rule() -> None:
    with pytest.raises(ValueError, match="quota"):
        WirePolicy(rules={"quota": FLAT})  # type: ignore[dict-item]


# -- 4: retries never multiply -------------------------------------------------------------------


async def test_4_one_re_ask_and_one_429_on_the_second_request_is_three_wire_requests() -> None:
    not_a_note = (
        200,
        {**_ok()[1], "choices": [{**_ok()[1]["choices"][0], "message": {"role": "assistant", "content": "{}"}}]},
        {},
    )
    wire = Wire(not_a_note, _err(429), _ok("second"))
    res = await _gen(wire).generate(REQ, _Note)
    assert res.outcome == "ok" and res.value == _Note(headline="second")
    assert len(wire.sent) == 3, "re-ask 1 + its one wire retry; the request that parsed badly is not re-sent"
    assert res.retries == 1 and [a.failure.kind for a in res.attempts] == ["rate_limited"]


async def test_4_the_sdk_sends_once_without_a_policy() -> None:
    wire = Wire(_err(429), _ok())
    res = await _gen(wire, WirePolicy.none()).generate(REQ, _Note)
    assert res.failure is not None and res.failure.kind == "rate_limited" and len(wire.sent) == 1


# -- 5: the deadline -----------------------------------------------------------------------------


async def test_5_a_backoff_past_the_deadline_ends_the_call_without_sleeping(
    _backoff_does_not_sleep: list[float],
) -> None:
    slow = WirePolicy(rules={"rate_limited": RetryRule(attempts=3, wait_initial_s=5.0, wait_max_s=5.0, jitter=0.0)})
    wire = Wire(_err(429), _ok())
    events: list[str] = []
    gen = _gen(wire, slow, on_event=lambda _lvl, name, _f: events.append(name))
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    res = await gen.generate(replace(REQ, deadline=Deadline(total_s=1.0)), _Note)
    assert loop.time() - t0 < 1.0 and _backoff_does_not_sleep == []
    assert res.outcome == "unavailable" and res.failure is not None
    assert (res.failure.kind, res.failure.phase) == ("timeout", "total") and len(wire.sent) == 1
    assert "llm_deadline_expired" in events


async def test_5_a_fixed_model_honours_the_deadline() -> None:
    async def stall(_m: list[ModelMessage], _i: AgentInfo) -> ModelResponse:
        await asyncio.sleep(30)
        return ModelResponse(parts=[TextPart("late")])  # pragma: no cover

    gen = PydanticAIGenerator(FunctionModel(stall), policy=POLICY)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    res = await gen.generate(replace(REQ, deadline=Deadline(total_s=0.2)), str)
    assert loop.time() - t0 < 2.0
    assert res.failure is not None and (res.failure.kind, res.failure.phase) == ("timeout", "total")


async def test_5_an_attempt_bound_is_a_retried_timeout_inside_the_total() -> None:
    wire = Wire(_ok())
    wire.gate = asyncio.Event()
    every = WirePolicy(rules={"timeout": FLAT})

    async def release_after_two() -> None:
        while len(wire.sent) < 2:
            await asyncio.sleep(0.01)
        wire.gate.set()

    helper = asyncio.create_task(release_after_two())
    res = await _gen(wire, every).generate(replace(REQ, deadline=Deadline(total_s=10.0, attempt_s=0.2)), _Note)
    await helper
    assert res.outcome == "ok" and [a.failure.kind for a in res.attempts] == ["timeout"] and len(wire.sent) == 2


# -- 6: cancellation propagates and releases --------------------------------------------------


async def test_6_cancel_during_the_request_propagates_and_frees_the_slot() -> None:
    wire = Wire(_ok())
    wire.gate = asyncio.Event()
    limiter = LocalLimiter({"small-chat": 1})
    gen = _gen(wire, limiter=limiter)
    task = asyncio.create_task(gen.generate(REQ, _Note))
    while not wire.sent:
        await asyncio.sleep(0.01)
    assert limiter.free("small-chat") == 0
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert limiter.free("small-chat") == 1 and wire.live == 0


async def test_6_cancel_during_backoff_propagates_and_frees_the_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    parked = asyncio.Event()

    async def forever(_s: float) -> None:
        parked.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(_retry, "_sleep", forever)
    wire = Wire(_err(429), _ok())
    limiter = LocalLimiter({"small-chat": 1})
    task = asyncio.create_task(_gen(wire, limiter=limiter).generate(REQ, _Note))
    await parked.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert limiter.free("small-chat") == 1 and len(wire.sent) == 1


# -- 7: the in-process limiter -------------------------------------------------------------------


async def test_7_three_calls_on_one_slot_run_strictly_one_at_a_time() -> None:
    wire = Wire(_ok())
    gen = _gen(wire, limiter=LocalLimiter({"small-chat": 1}))
    results = await asyncio.gather(*(gen.generate(REQ, _Note) for _ in range(3)))
    assert all(r.outcome == "ok" for r in results) and wire.peak == 1 and len(wire.sent) == 3
    assert all(isinstance(r.stamp["limiter_wait_s"], float) for r in results)
    unlimited = Wire(_ok())
    unlimited.gate = asyncio.Event()
    free = _gen(unlimited, limiter=LocalLimiter({"other": 1}))
    calls = [asyncio.create_task(free.generate(REQ, _Note)) for _ in range(3)]
    while len(unlimited.sent) < 3:
        await asyncio.sleep(0.01)
    unlimited.gate.set()
    await asyncio.gather(*calls)
    assert unlimited.peak == 3, "a group the limiter does not name is not limited"


async def test_7_a_slot_wait_past_max_wait_is_rate_limited_local_limiter_and_not_retried() -> None:
    wire = Wire(_ok())
    wire.gate = asyncio.Event()
    gen = _gen(wire, limiter=LocalLimiter({"small-chat": 1}, max_wait_s=0.05))
    holder = asyncio.create_task(gen.generate(REQ, _Note))
    while not wire.sent:
        await asyncio.sleep(0.01)
    res = await gen.generate(REQ, _Note)
    wire.gate.set()
    await holder
    assert res.failure is not None and (res.failure.kind, res.failure.provider_code) == (
        "rate_limited",
        "local_limiter",
    )
    assert len(wire.sent) == 1 and res.attempts == (), "the refused call sent nothing"


async def test_7_a_retry_keeps_its_slot() -> None:
    wire = Wire(_err(429), _ok("a"), _ok("b"))
    gen = _gen(wire, limiter=LocalLimiter({"small-chat": 1}))
    first, second = await asyncio.gather(gen.generate(REQ, _Note), gen.generate(REQ, _Note))
    assert first.outcome == second.outcome == "ok" and wire.peak == 1
    assert [len(first.attempts), len(second.attempts)] == [1, 0], "the second call never got in between"


# -- 8: one client per generator per loop --------------------------------------------------------


async def test_8_ten_calls_open_one_client_and_aclose_closes_it() -> None:
    wire = Wire(_ok())
    gen = _gen(wire)
    for _ in range(10):
        assert (await gen.generate(REQ, _Note)).outcome == "ok"
    stats = gen.stats()
    assert (stats["clients_built"], stats["clients_open"], stats["requests_sent"], stats["calls_in_flight"]) == (
        1,
        1,
        10,
        0,
    )
    await gen.aclose()
    assert gen.stats()["clients_open"] == 0


def test_8_two_event_loops_get_two_clients() -> None:
    wire = Wire(_ok())
    gen = _gen(wire)
    outcomes: list[str] = []

    def run() -> None:
        outcomes.append(asyncio.run(gen.generate(REQ, _Note)).outcome)

    for _ in range(2):
        t = threading.Thread(target=run)
        t.start()
        t.join()
    assert outcomes == ["ok", "ok"] and gen.stats()["clients_built"] == 2


# -- 9 lives in test_embedder.py: its 1.22.0 tests pass unchanged through the shared loop ---------


async def test_9_the_embedder_s_retry_policy_is_the_same_wire_policy() -> None:
    old = RetryPolicy(attempts=3, wait_initial_s=30.0, wait_max_s=300.0).wire()
    assert old.rules["rate_limited"].wait_s(1, spread=0.0) == 30.0 and old.rules["server"].wait_s(2, spread=0.0) == 60.0
    assert "quota" not in old.rules and old.rules["server"].retry_after_max_s == 300.0
