"""LangfuseTracer -- the Tracer over Langfuse SDK 5, on a REAL client exporting to memory
(`testing_langfuse.LangfuseRig`): the wire is the span the SDK exports, so these read exactly what
would reach the server."""

from __future__ import annotations

import json
import logging

import pytest
from langfuse.api.prompts.types.prompt import Prompt_Text  # type: ignore[import-untyped]
from langfuse.model import TextPromptClient  # type: ignore[import-untyped]
from opentelemetry.sdk.trace import TracerProvider

from llm_ports import ports
from llm_ports.adapters.langfuse import tracer as tp
from llm_ports.observability import NoopGeneration, observe
from llm_ports.testing.langfuse import OBS, LangfuseRig, RefusingLangfuse

META = {"feature": "demo.cat"}


@pytest.fixture
def rig() -> LangfuseRig:
    return LangfuseRig()


def _start(client, **over):
    kw = {
        "name": "demo.cat",
        "model": "m",
        "provider": "openai",
        "input": "hi",
        "metadata": META,
        "session_id": None,
        "prompt_name": None,
        **over,
    }
    return tp.LangfuseTracer(client).start_generation(**kw)


def test_the_tracer_and_its_generation_satisfy_the_ports(rig) -> None:
    t = tp.LangfuseTracer(rig.client)
    assert isinstance(t, ports.Tracer) and isinstance(_start(rig.client), ports.Generation)


def test_a_generation_is_one_root_generation_span_named_and_tagged_by_name(rig) -> None:
    _start(rig.client).end(output="ok")
    (span,) = rig.spans()
    a = dict(span.attributes)
    assert span.name == "demo.cat"
    assert a[OBS + "type"] == "generation" and a[OBS + "model.name"] == "m"
    assert a[OBS + "input"] == "hi" and a[OBS + "output"] == "ok"
    assert a["langfuse.trace.name"] == "demo.cat" and a["langfuse.trace.tags"] == ("demo.cat",)
    assert json.loads(a[OBS + "metadata.provider"]) == "openai"
    assert "session.id" not in a


def test_a_provider_the_caller_already_put_in_metadata_wins(rig) -> None:
    _start(rig.client, metadata={"provider": "local"}).end()
    assert json.loads(rig.attrs()[OBS + "metadata.provider"]) == "local"


def test_a_session_rides_the_span_as_session_id(rig) -> None:
    _start(rig.client, session_id="batch-01234567").end()
    assert rig.attrs()["session.id"] == "batch-01234567"


def test_inside_an_otel_span_the_langfuse_trace_id_is_that_span_s_trace_id(rig) -> None:
    ours = TracerProvider().get_tracer("ours")
    with ours.start_as_current_span("chat m") as outer:
        _start(rig.client).end()
    (span,) = rig.spans()
    assert span.context.trace_id == outer.get_span_context().trace_id
    assert dict(span.attributes)["langfuse.internal.as_root"] is True


def test_a_prompt_name_links_the_labelled_managed_prompt(rig) -> None:
    fetched: list = []

    def get_prompt(name, **kw):
        fetched.append((name, kw))
        return TextPromptClient(
            Prompt_Text(prompt="t", name=name, version=7, config={}, labels=["production"], tags=[])
        )

    rig.client.get_prompt = get_prompt
    _start(rig.client, prompt_name="classify").end()
    assert fetched == [("classify", {"label": "production", "cache_ttl_seconds": 300})]
    a = rig.attrs()
    assert a[OBS + "prompt.name"] == "classify" and a[OBS + "prompt.version"] == 7


def test_a_prompt_link_failure_is_suppressed_and_the_generation_still_opens(rig) -> None:
    def get_prompt(name, **kw):
        raise RuntimeError("404 prompt not found")

    rig.client.get_prompt = get_prompt
    g = _start(rig.client, prompt_name="gone")
    g.end()
    assert not isinstance(g, NoopGeneration)
    assert OBS + "prompt.name" not in rig.attrs()


def test_end_sends_the_raw_pair_as_usage_details_and_the_metadata(rig) -> None:
    _start(rig.client).end(output="ok", usage={"input": 11, "output": 7}, metadata={"latency_ms": 42})
    a = rig.attrs()
    assert json.loads(a[OBS + "usage_details"]) == {"input": 11, "output": 7}
    assert json.loads(a[OBS + "metadata.latency_ms"]) == 42
    assert OBS + "level" not in a


def test_an_unmeasured_pair_sends_no_usage_block(rig) -> None:
    # MUST FIRE: book `None` as 0 in `usage_details` and both spans carry a free-looking usage block.
    _start(rig.client).end(output="o", usage=None)
    _start(rig.client).end(usage={"input": None, "output": None})
    assert [OBS + "usage_details" in dict(s.attributes) for s in rig.spans()] == [False, False]


def test_a_half_reported_pair_sends_only_the_measured_side() -> None:
    # MUST FIRE: book the unreported side as 0 and a half-measured call looks cheaper than it was.
    assert tp.usage_details(120, None) == {"input": 120}
    assert tp.usage_details(None, 30) == {"output": 30}
    assert tp.usage_details(0, 0) == {"input": 0, "output": 0}
    assert tp.usage_details(None, None) is None


def test_an_error_end_sets_level_and_status_message(rig) -> None:
    _start(rig.client).end(level="ERROR", status_message="RuntimeError: upstream 500", metadata={"latency_ms": 3})
    a = rig.attrs()
    assert a[OBS + "level"] == "ERROR" and a[OBS + "status_message"] == "RuntimeError: upstream 500"


def test_score_is_numeric_on_this_generation_with_the_comment(rig) -> None:
    g = _start(rig.client)
    g.score("note.retries", 1.0, "verdict=spoke")
    g.end()
    (span,) = rig.spans()
    (s,) = rig.scores
    assert (s["name"], s["value"], s["data_type"], s["comment"]) == ("note.retries", 1.0, "NUMERIC", "verdict=spoke")
    assert s["trace_id"] == format(span.context.trace_id, "032x")
    assert s["observation_id"] == format(span.context.span_id, "016x")


def test_score_metadata_is_stored_on_the_score(rig) -> None:
    g = _start(rig.client)
    g.score("label.judge-x", 0.0, "why", {"by": "judge-model", "labeller": "judge"})
    g.score("plain", 1.0)
    g.end()
    with_meta, plain = rig.scores
    assert with_meta["metadata"] == {"by": "judge-model", "labeller": "judge"}
    assert plain.get("metadata") is None


async def test_observe_over_the_real_sdk_flags_an_unmeasured_call(rig) -> None:
    async with observe(tp.LangfuseTracer(rig.client), name="n", model="m", provider="p", input="i") as obs:
        obs.update(output="hi", usage_details={"input": None, "output": None}, metadata={"latency_ms": 12})
    (o,) = rig.observed()
    assert "usage" not in o and o["metadata"]["tokens_unmeasured"] is True and o["metadata"]["latency_ms"] == 12


def test_a_client_that_raises_yields_a_noop_generation_and_one_held_warning() -> None:
    now = [1000.0]
    events: list = []
    t = tp.LangfuseTracer(
        RefusingLangfuse(),
        warn_hold_s=300,
        clock=lambda: now[0],
        on_event=lambda lvl, ev, f: events.append((lvl, ev, dict(f))),
    )
    for _ in range(3):
        g = t.start_generation(
            name="n", model="m", provider="p", input="i", metadata={}, session_id=None, prompt_name=None
        )
        assert isinstance(g, NoopGeneration)
        g.end(output="x")
        g.score("s", 1.0)
    now[0] += 301
    t.start_generation(name="n", model="m", provider="p", input="i", metadata={}, session_id="s", prompt_name="p")
    assert [(lvl, ev, f["action"]) for lvl, ev, f in events] == [
        (logging.WARNING, tp.EVENT, "open_failed"),
        (logging.WARNING, tp.EVENT, "open_failed"),
    ]  # held, then once more
    assert "ConnectionError" in events[0][2]["error"] and events[0][2]["held_s"] == 300


def test_end_and_score_failures_warn_and_never_raise(rig) -> None:
    def boom(**kw):
        raise RuntimeError("down")

    events: list = []
    rig.client.create_score = boom
    g = tp.LangfuseTracer(rig.client, on_event=lambda lvl, ev, f: events.append(f["action"])).start_generation(
        name="n", model="m", provider="p", input="i", metadata={}, session_id=None, prompt_name=None
    )
    g.score("s", 1.0)
    g._gen.update = boom
    g.end(output="x")
    assert events == ["score_failed"]  # the end failure is inside the hold


def test_the_default_event_sink_is_the_stdlib_logger(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="llm_ports.langfuse_tracer"):
        tp.LangfuseTracer(RefusingLangfuse()).start_generation(
            name="n", model="m", provider="p", input="i", metadata={}, session_id=None, prompt_name=None
        )
    assert "open_failed" in caplog.text


# -- 1.24: the result contract reaches the trace store ---------------------------------------------


def test_cost_details_are_sent_only_when_the_cost_is_known(rig) -> None:
    # MUST FIRE both ways: a known cost is the total; an unknown one sends nothing, so the store's
    # own model pricing may apply, and `cost_source` says which happened.
    from decimal import Decimal

    _start(rig.client).end(output="o", cost=ports.Cost(Decimal("0.0125"), "price_table"))
    _start(rig.client).end(output="o", cost=ports.Cost(None, "unknown", reason="unknown_model"))
    known, unknown = (dict(s.attributes) for s in rig.spans())
    assert json.loads(known[OBS + "cost_details"]) == {"total": 0.0125}
    assert json.loads(known[OBS + "metadata.cost_source"]) == "price_table"
    assert OBS + "cost_details" not in unknown
    assert json.loads(unknown[OBS + "metadata.cost_source"]) == "unknown"


def test_usage_v2_alone_sends_its_measured_sides_and_never_a_zero_for_the_other(rig) -> None:
    _start(rig.client).end(output="o", usage_v2=ports.Usage(input=9, output=None, requests=1))
    assert json.loads(rig.attrs()[OBS + "usage_details"]) == {"input": 9}


def test_a_cancelled_end_is_a_warning_with_its_status(rig) -> None:
    _start(rig.client).end(level="CANCELLED", status_message="CancelledError")
    a = rig.attrs()
    assert a[OBS + "level"] == "WARNING" and a[OBS + "status_message"] == "CancelledError"


async def test_observe_cancelled_over_the_real_sdk_ends_once_as_a_warning(rig) -> None:
    import asyncio

    with pytest.raises(asyncio.CancelledError):
        async with observe(tp.LangfuseTracer(rig.client), name="c", model="m", provider="p", input="i"):
            raise asyncio.CancelledError
    (span,) = rig.spans()
    assert dict(span.attributes)[OBS + "level"] == "WARNING"
