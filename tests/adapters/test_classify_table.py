"""The shared classification table, from real pydantic-ai exceptions and recorded wire bodies.

Failures acceptance test 1 (the classification table), 7 (context overflow from four provider
shapes) and the order the table runs in. Each row is a real exception class, built the way
pydantic-ai's own models raise it (`ModelHTTPError.for_category`), or a recorded error body.
"""

from __future__ import annotations

import asyncio
import email.utils
import time
from typing import Any

import httpx
import pytest
from pydantic_ai.exceptions import (
    ConcurrencyLimitExceeded,
    ContentFilterError,
    FallbackExceptionGroup,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelContextWindowExceededError,
    ModelHTTPError,
    ModelNotFoundError,
    ModelOverloadedError,
    ModelQuotaExceededError,
    ModelRateLimitError,
    ModelRequestRejectedError,
    ModelRequestTimeout,
    ModelServerError,
    ModelTimeoutError,
    ModelUnavailableError,
    RunCancelled,
    RunTimedOut,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)

from llm_ports.adapters._classify import (
    SEVERITY,
    WireFacts,
    classify_core,
    kind_for_status,
    raised_in_provider,
    retry_after_s,
    worst,
)
from llm_ports.adapters.pydantic_ai import classify_failure, openai_finish
from llm_ports.ports import FAILURE_KINDS, failure

M = "small-chat"


def _http(cls: Any, status: int, **kw: Any) -> ModelHTTPError:
    return ModelHTTPError.for_category(cls, status_code=status, model_name=M, **kw)


def _chain(top: BaseException, cause: BaseException) -> BaseException:
    top.__cause__ = cause
    return top


TABLE: list[tuple[str, BaseException, str | None, str, str]] = [
    # (id, exception, finish_raw, kind, outcome)
    ("rate_limited", _http(ModelRateLimitError, 429, retry_after=3.0), None, "rate_limited", "unavailable"),
    ("quota", _http(ModelQuotaExceededError, 429), None, "quota", "unavailable"),
    ("overloaded", _http(ModelOverloadedError, 529), None, "overloaded", "unavailable"),
    ("server", _http(ModelServerError, 500), None, "server", "unavailable"),
    ("unavailable", _http(ModelUnavailableError, 503), None, "server", "unavailable"),
    ("auth", _http(ModelAuthenticationError, 401), None, "auth", "unavailable"),
    ("not_found", _http(ModelNotFoundError, 404), None, "not_found", "unavailable"),
    ("ctx_window", _http(ModelContextWindowExceededError, 400), None, "context_overflow", "request_fault"),
    ("rejected", _http(ModelRequestRejectedError, 400), None, "bad_request", "request_fault"),
    (
        "rejected_overflow_code",
        _http(ModelRequestRejectedError, 400, provider_error_code="context_length_exceeded"),
        None,
        "context_overflow",
        "request_fault",
    ),
    ("plain_400", _http(None, 400), None, "bad_request", "request_fault"),
    ("plain_413", _http(None, 413), None, "bad_request", "request_fault"),
    ("plain_502", _http(None, 502), None, "server", "unavailable"),
    ("timeout", ModelTimeoutError(M, "read timed out", phase="read"), None, "timeout", "unavailable"),
    ("req_timeout", ModelRequestTimeout(M, 5.0), None, "timeout", "unavailable"),
    ("run_timed_out", RunTimedOut("run over its deadline"), None, "timeout", "unavailable"),
    ("connection", ModelConnectionError(M, "refused", phase="connect"), None, "transport", "unavailable"),
    ("content_filter", ContentFilterError("blocked"), None, "refused", "refused"),
    ("bare_api_error", ModelAPIError(M, "stream died"), None, "server", "unavailable"),
    ("umb_length", UnexpectedModelBehavior("Exceeded maximum output retries"), "length", "truncated", "truncated"),
    ("umb_stop", UnexpectedModelBehavior("Exceeded maximum output retries"), "stop", "invalid_output", "invalid"),
    ("umb_filter", UnexpectedModelBehavior("no output"), "content_filter", "refused", "refused"),
    ("builtin_timeout", TimeoutError(), None, "timeout", "unavailable"),
    ("builtin_connection", ConnectionRefusedError(), None, "transport", "unavailable"),
    ("httpx_connect", httpx.ConnectError("refused"), None, "transport", "unavailable"),
    ("httpx_read_timeout", httpx.ReadTimeout("slow"), None, "timeout", "unavailable"),
    ("usage_limit", UsageLimitExceeded("request_limit of 3"), None, "caller_limit", "unavailable"),
    ("app_cancelled", RunCancelled("cancelled by the app"), None, "caller_limit", "unavailable"),
    ("concurrency", ConcurrencyLimitExceeded("queue full"), None, "overloaded", "unavailable"),
    ("a_bug", KeyError("headline"), None, "internal", "request_fault"),
    ("wrapped_bug", _chain(RuntimeError("validator"), ValueError("bad")), None, "internal", "request_fault"),
]


@pytest.mark.parametrize(("exc", "finish", "kind", "outcome"), [r[1:] for r in TABLE], ids=[r[0] for r in TABLE])
def test_the_classification_table(exc: BaseException, finish: str | None, kind: str, outcome: str) -> None:
    got = classify_failure(exc, WireFacts(finish_raw=finish))
    assert got is not None and (got.kind, got.outcome) == (kind, outcome)


def test_a_category_carries_its_status_retry_hint_and_phase() -> None:
    rl = classify_failure(_http(ModelRateLimitError, 429, retry_after=3.0), WireFacts())
    assert rl is not None and (rl.status, rl.retry_after_s, rl.retryable) == (429, 3.0, True)
    t = classify_failure(ModelTimeoutError(M, "slow", phase="read"), WireFacts())
    assert t is not None and t.phase == "read"
    first_event = ModelRequestTimeout(M, 5.0)
    first_event.setting = "first_event_timeout"  # type: ignore[misc]  # the `setting=` kwarg is newer than 2.62
    first = classify_failure(first_event, WireFacts())
    assert first is not None and first.phase == "first_event"
    total = classify_failure(RunTimedOut("over"), WireFacts())
    assert total is not None and total.phase == "total"
    conn = classify_failure(httpx.ConnectTimeout("slow"), WireFacts())
    assert conn is not None and (conn.kind, conn.phase) == ("timeout", "connect")


def test_a_category_anywhere_in_the_chain_wins_over_the_wrapper() -> None:
    got = classify_failure(_chain(RuntimeError("wrapped"), _http(ModelRateLimitError, 429)), WireFacts())
    assert got is not None and got.kind == "rate_limited"


def test_a_wrapped_unexpected_model_behavior_is_invalid_output() -> None:
    # Failures acceptance test 8: the wrapper is not a bug; the parse failure under it decides.
    wrapped = _chain(RuntimeError("agent failed"), UnexpectedModelBehavior("Exceeded maximum output retries"))
    got = classify_failure(wrapped, WireFacts(finish_raw="stop"))
    assert got is not None and (got.kind, got.outcome) == ("invalid_output", "invalid")


def test_gated_beats_every_category_and_extra_beats_gated() -> None:
    class Refused(Exception):
        pass

    exc = _chain(_http(ModelRateLimitError, 429), Refused("switched off"))
    got = classify_failure(exc, WireFacts(), gated=(Refused,))
    assert got is not None and got.kind == "gated"
    first = classify_failure(exc, WireFacts(), gated=(Refused,), extra=(lambda _e, _f: failure("quota"),))
    assert first is not None and first.kind == "quota"
    passes = classify_failure(exc, WireFacts(), gated=(Refused,), extra=(lambda _e, _f: None,))
    assert passes is not None and passes.kind == "gated", "an extra returning None leaves the table in charge"


def test_a_cancellation_is_never_classified() -> None:
    with pytest.raises(asyncio.CancelledError):
        classify_failure(asyncio.CancelledError(), WireFacts(), extra=(lambda _e, _f: failure("server"),))


def test_a_reason_never_carries_the_error_body() -> None:
    got = classify_failure(_http(None, 500, body={"message": "secret prompt " * 50}), WireFacts())
    assert got is not None and "secret" not in got.reason


# ── no exception: how the reply ended ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("facts", "want"),
    [
        (WireFacts(status=200, finish_raw="stop"), None),
        (WireFacts(status=200, finish_raw="length"), "truncated"),
        (WireFacts(status=200, finish_raw="content_filter"), "refused"),
        (WireFacts(status=200, finish_raw="stop", refusal=True), "refused"),
        (WireFacts(status=200), None),
    ],
    ids=["stop", "length", "content_filter", "refusal_field", "no_finish"],
)
def test_a_returned_reply_is_classified_by_how_it_ended(facts: WireFacts, want: str | None) -> None:
    got = classify_failure(None, facts)
    assert (got.kind if got else None) == want


@pytest.mark.parametrize(
    ("raw", "unified"),
    [
        ("stop", "stop"),
        ("length", "length"),
        ("content_filter", "content_filter"),
        ("tool_calls", "tool_call"),
        ("function_call", "tool_call"),
        ("tool_call", "tool_call"),
        ("error", "error"),
        ("eos_token", "other"),
        (None, None),
    ],
)
def test_the_openai_finish_mapping(raw: str | None, unified: str | None) -> None:
    # Result-contract acceptance test 7: an unknown raw string is `other`; absent is None, not `stop`.
    assert openai_finish(raw) == unified


# ── recorded wire bodies, through the status step ────────────────────────────────────────────────

# Recorded error bodies (trimmed): what each server sends for a prompt over its context window.
OVERFLOW_BODIES: list[tuple[str, WireFacts]] = [
    (
        "openai",
        WireFacts(
            status=400,
            error_code="context_length_exceeded",
            error_type="invalid_request_error",
            error_message="This model's maximum context length is 8192 tokens.",
        ),
    ),
    (
        "llama_cpp",
        WireFacts(
            status=400,
            error_type="exceed_context_size_error",
            error_message="the request exceeds the available context size",
        ),
    ),
    (
        "anthropic_via_gateway",
        WireFacts(status=400, error_message="prompt is too long: 210000 tokens > 200000 maximum"),
    ),
    (
        "litellm",
        WireFacts(
            status=400,
            error_type="None",
            error_message="litellm.ContextWindowExceededError: input token count exceeds the context window",
        ),
    ),
]


@pytest.mark.parametrize("facts", [f for _, f in OVERFLOW_BODIES], ids=[n for n, _ in OVERFLOW_BODIES])
def test_context_overflow_from_four_provider_shapes(facts: WireFacts) -> None:
    got = classify_core(None, facts)
    assert got is not None and (got.kind, got.outcome) == ("context_overflow", "request_fault")


def test_a_plain_400_is_bad_request_not_overflow() -> None:
    got = classify_core(None, WireFacts(status=400, error_message="temperature must be <= 2"))
    assert got is not None and got.kind == "bad_request"


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (400, "bad_request"),
        (401, "auth"),
        (403, "auth"),
        (404, "not_found"),
        (408, "timeout"),
        (409, "server"),
        (422, "bad_request"),
        (429, "rate_limited"),
        (500, "server"),
        (503, "overloaded"),
        (529, "overloaded"),
    ],
)
def test_status_alone(status: int, kind: str) -> None:
    assert kind_for_status(status) == kind
    got = classify_core(None, WireFacts(status=status))
    assert got is not None and got.kind == kind and got.status == status


def test_retry_after_reads_ms_seconds_and_dates() -> None:
    assert retry_after_s({"retry-after-ms": "1500"}) == 1.5
    assert retry_after_s({"retry-after": "7"}) == 7.0
    future = email.utils.formatdate(time.time() + 60, usegmt=True)
    hint = retry_after_s({"retry-after": future})
    assert hint is not None and 55 <= hint <= 61
    past = email.utils.formatdate(time.time() - 60, usegmt=True)
    assert retry_after_s({"retry-after": past}) is None
    assert retry_after_s({"retry-after": "soon"}) is None
    assert retry_after_s({"retry-after-ms": "x", "retry-after": "2"}) == 2.0
    assert retry_after_s({"retry-after": "-3"}) is None
    assert retry_after_s({}) is None


def test_a_429_takes_the_wait_from_the_header() -> None:
    got = classify_core(None, WireFacts(status=429, headers={"retry-after": "4"}))
    assert got is not None and (got.kind, got.retry_after_s) == ("rate_limited", 4.0)


def test_every_status_keeps_its_1x_outcome() -> None:
    # Additive rule: the kind is new detail, the outcome of a bare status is what 1.22.0 said.
    from llm_ports.ports.outcomes import outcome_for_status

    for status in range(400, 600):
        got = classify_core(None, WireFacts(status=status))
        assert got is not None and got.outcome == outcome_for_status(status), status


# ── step 9: the provider's parse failure is not the caller's bug ─────────────────────────────────


def _raise_from(module: str, exc: BaseException) -> BaseException:
    """Raise `exc` from a frame whose module is `module`, and return it with that traceback."""
    code = "def boom(e):\n    raise e\n"
    scope: dict[str, Any] = {"__name__": module}
    exec(compile(code, f"<{module}>", "exec"), scope)
    try:
        scope["boom"](exc)
    except BaseException as caught:
        return caught
    raise AssertionError("unreachable")


def _through_json_from(module: str) -> BaseException:
    """A `JSONDecodeError` raised inside the standard library, called from `module`."""
    code = "import json\ndef boom():\n    json.loads('{')\n"
    scope: dict[str, Any] = {"__name__": module}
    exec(compile(code, f"<{module}>", "exec"), scope)
    try:
        scope["boom"]()
    except ValueError as caught:
        return caught
    raise AssertionError("unreachable")


@pytest.mark.parametrize(
    ("module", "provider"),
    [
        ("pydantic_ai.models.openai", True),
        ("openai._models", True),
        ("httpx2._models", True),
        ("pydantic_ai._agent_graph", False),
        ("llm_ports.adapters.pydantic_ai._generator", False),
        ("consumer_app.validators", False),
    ],
)
def test_who_raised_decides_malformed_response_or_internal(module: str, provider: bool) -> None:
    exc = _raise_from(module, IndexError("list index out of range"))
    assert raised_in_provider(exc) is provider
    got = classify_failure(exc, WireFacts(status=200))
    assert got is not None
    want = ("malformed_response", "unavailable", True) if provider else ("internal", "request_fault", False)
    assert (got.kind, got.outcome, got.retryable) == want


def test_the_standard_library_is_skipped_when_finding_who_raised() -> None:
    caller = _through_json_from("consumer_app.validators")
    provider = _through_json_from("pydantic_ai.models.openai")
    assert (raised_in_provider(caller), raised_in_provider(provider)) == (False, True)


def test_an_exception_with_no_traceback_is_internal() -> None:
    got = classify_failure(IndexError("never raised"), WireFacts())
    assert got is not None and got.kind == "internal"


# ── exception groups: each member through the same table, the worst names the call ──────────────


def test_severity_ranks_every_kind_once() -> None:
    assert sorted(SEVERITY) == sorted(FAILURE_KINDS)


def test_a_fallback_group_is_its_worst_member() -> None:
    group = FallbackExceptionGroup(
        "all models failed",
        [_http(ModelRateLimitError, 429), _http(ModelAuthenticationError, 401), ModelTimeoutError(M, "slow")],
    )
    got = classify_failure(group, WireFacts())
    assert got is not None and (got.kind, got.outcome, got.status) == ("auth", "unavailable", 401)
    assert got.reason.startswith("FallbackExceptionGroup: ")


def test_a_group_of_retryable_members_stays_retryable() -> None:
    group = FallbackExceptionGroup("all failed", [_http(ModelOverloadedError, 529), _http(ModelRateLimitError, 429)])
    got = classify_failure(group, WireFacts())
    assert got is not None and got.kind == "overloaded" and got.retryable


def test_a_gated_member_and_a_cancelled_member_in_a_group() -> None:
    class Refused(Exception):
        pass

    got = classify_core(ExceptionGroup("g", [Refused("off"), ValueError("x")]), WireFacts(), gated=(Refused,))
    assert got is not None and got.kind == "internal", "a bug outranks a gate"
    got = classify_core(ExceptionGroup("g", [Refused("off")]), WireFacts(), gated=(Refused,))
    assert got is not None and got.kind == "gated"
    with pytest.raises(asyncio.CancelledError):
        classify_core(BaseExceptionGroup("g", [asyncio.CancelledError()]), WireFacts())


def test_worst_of_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="no failures"):
        worst([])
    assert worst([failure("timeout"), failure("server")]).kind == "server"


# ── step 5 needs a failure: an overflow phrase in a returned reply is not an overflow ────────────


def test_an_overflow_message_on_a_returned_reply_is_not_a_failure() -> None:
    facts = WireFacts(status=200, finish_raw="stop", error_message="maximum context length is 8192 tokens")
    assert classify_core(None, facts) is None
    coded = WireFacts(status=200, finish_raw="stop", error_code="context_length_exceeded")
    assert classify_core(None, coded) is None
    assert classify_core(None, WireFacts(status=400, error_code="context_length_exceeded")) is not None
