"""The test doubles keep the laws of the ports they stand in for."""

from __future__ import annotations

from typing import Any

from llm_ports import ports
from llm_ports.ports import Chunk, GenRequest
from llm_ports.testing import STUB_MODEL, Reply, StubCiter, StubGenerator

SPAN = Chunk(id="a", text="Net closed at 580.5, up 3%", title="Report")


async def test_the_stub_citer_keeps_the_citer_laws() -> None:
    c = StubCiter()
    assert isinstance(c, ports.Citer)
    empty = await c.answer("q", [])
    assert empty.claims == () and empty.unknown_ids() == ()
    ans = await c.answer("q", [SPAN])
    assert ans.unknown_ids() == () and set(ans.chunks) == {"a"} and ans.model == ""


# ── StubGenerator: any failure kind, and an answering model that is not an echo ──────────────────


async def test_a_reply_function_fails_a_call_with_reply_failed() -> None:
    gen = StubGenerator(lambda _req, _out: Reply.failed("rate_limited", retry_after_s=2.0, status=429))
    res = await gen.generate(ports.GenRequest("small-chat", "p"), str)
    assert res.failure is not None
    got = (res.outcome, res.failure.kind, res.failure.retry_after_s, res.failure.status)
    assert got == ("unavailable", "rate_limited", 2.0, 429)
    assert Reply.failed("caller_limit") == ports.failure("caller_limit"), "the same Failure as ports.failure"


async def test_every_failure_kind_reaches_the_result_through_the_stub() -> None:
    for kind in ports.FAILURE_KINDS:
        res = await StubGenerator(lambda _r, _o, k=kind: Reply.failed(k)).generate(ports.GenRequest("m", "p"), str)
        assert res.failure is not None and res.failure.kind == kind and res.outcome == ports.OUTCOME_FOR[kind]


async def test_the_stub_reports_its_own_model_as_the_one_that_answered() -> None:
    res = await StubGenerator("x").generate(ports.GenRequest("small-chat", "p"), str)
    assert res.answered is not None
    assert (res.answered.requested, res.answered.model, res.model) == ("small-chat", STUB_MODEL, "small-chat")
    named = await StubGenerator("x", model="fake-8b").generate(ports.GenRequest("small-chat", "p"), str)
    assert named.answered is not None and named.answered.model == "fake-8b"
    failed = await StubGenerator(ports.failure("server")).generate(ports.GenRequest("small-chat", "p"), str)
    assert failed.answered is not None and failed.answered.model == STUB_MODEL


async def test_a_re_ask_asks_retry_message_for_its_text_once_and_keeps_it() -> None:
    """The stub runs the adapter's seam: a caller that counts re-asks in `retry_message` counts
    the stub's. No re-ask, no call."""
    asked: list[tuple[str, ...]] = []

    def message(problems: Any) -> str:
        asked.append(tuple(problems))
        return "fix: " + problems[0]

    req = GenRequest(model="m", prompt="p", output_retries=1)
    gen = StubGenerator([Reply.ok(""), Reply.ok("fine")])
    res = await gen.generate(req, str, validate=lambda v: [] if v else ["empty"], retry_message=message)
    assert (res.value, res.retries) == ("fine", 1) and asked == [("empty",)] and gen.reasks == ["fix: empty"]
    kept = StubGenerator([Reply.ok(""), Reply.ok("")])
    res = await kept.generate(req, str, validate=lambda v: [] if v else ["empty"])
    assert res.problems == ("empty",) and kept.reasks == ["empty"], "one re-ask, then kept: the default text"
    once = StubGenerator([Reply.ok("fine")])
    await once.generate(req, str, validate=lambda v: [] if v else ["empty"], retry_message=message)
    assert once.reasks == [] and len(asked) == 1

    def broken(_problems: Any) -> str:
        raise RuntimeError("message bug")

    script = [Reply.ok(""), Reply.ok("fine")]
    res = await StubGenerator(script).generate(req, str, validate=lambda _v: ["bad"], retry_message=broken)
    assert res.failure is not None and res.failure.kind == "internal"
