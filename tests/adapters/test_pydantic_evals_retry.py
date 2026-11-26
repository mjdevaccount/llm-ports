"""pydantic_evals_judge on the shared wire-retry loop (ADR-0021).

Before 1.31 the judge was the one adapter outside `WireRetry`: one 429 from a one-slot server
(`max_parallel_requests=1`) ended the sample. These pin both directions: a retryable status is
re-sent and the verdict lands; a verdict the judge could not read, a refused request, a spent
policy and a deadline that leaves no room for the wait are NOT re-sent.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from llm_ports import ports
from llm_ports.adapters.pydantic_evals import PydanticEvalsJudge
from tests.conftest import Role

JUDGE_ROLE = Role("llm.judge", tier="local", temperature=0.0)
GRADE = {"reason": "grounded", "pass": True, "score": 0.9}
RULE = ports.RetryRule(attempts=3, wait_initial_s=1.0, wait_max_s=4.0, jitter=0.0)
POLICY = ports.WirePolicy(rules={"rate_limited": RULE, "overloaded": RULE, "server": RULE})


def _flaky(statuses: list[int], sent: list[int]) -> FunctionModel:
    """Raises `ModelHTTPError(status)` for each status in turn, then answers the grade."""

    def fn(_m: list[ModelMessage], _i: AgentInfo) -> ModelResponse:
        sent.append(1)
        if len(sent) <= len(statuses):
            raise ModelHTTPError(status_code=statuses[len(sent) - 1], model_name="quick", body={"error": "busy"})
        return ModelResponse(parts=[ToolCallPart(tool_name="final_result", args=GRADE)])

    return FunctionModel(fn)


def _judge(model: FunctionModel, events: list[str] | None = None, **kw: Any) -> PydanticEvalsJudge:
    hook = (lambda _lvl, name, _f: events.append(name)) if events is not None else None
    return PydanticEvalsJudge(JUDGE_ROLE, model=model, on_event=hook, **kw)


async def test_a_429_is_re_sent_and_the_verdict_lands(_backoff_does_not_sleep: list[float]) -> None:
    sent: list[int] = []
    events: list[str] = []
    v = await _judge(_flaky([429, 503], sent), events, policy=POLICY).judge(prompt="p", output="o", rubric="r")
    assert (v.passed, v.score) == (True, 0.9)
    assert len(sent) == 3 and events == ["llm_wire_retry", "llm_wire_retry"]
    assert _backoff_does_not_sleep == [1.0, 1.0]  # each kind keeps its own tries and backoff


async def test_the_default_policy_retries_a_429_too() -> None:
    sent: list[int] = []
    v = await _judge(_flaky([429], sent)).judge(prompt="p", output="o", rubric="r")
    assert v.passed and len(sent) == 2


async def test_a_spent_policy_raises_the_last_failure() -> None:
    sent: list[int] = []
    with pytest.raises(ModelHTTPError) as got:
        await _judge(_flaky([429] * 9, sent), policy=POLICY).judge(prompt="p", output="o", rubric="r")
    assert got.value.status_code == 429 and len(sent) == RULE.attempts


async def test_a_refused_request_and_an_unreadable_verdict_are_sent_once() -> None:
    sent: list[int] = []
    with pytest.raises(ModelHTTPError):
        await _judge(_flaky([400], sent), policy=POLICY).judge(prompt="p", output="o", rubric="r")
    assert len(sent) == 1

    prose: list[int] = []

    def fn(_m: list[ModelMessage], _i: AgentInfo) -> ModelResponse:
        prose.append(1)
        return ModelResponse(parts=[TextPart("it looks fine to me")])

    with pytest.raises(UnexpectedModelBehavior):
        await _judge(FunctionModel(fn), policy=ports.WirePolicy.none()).judge(prompt="p", output="o", rubric="r")
    once = len(prose)  # pydantic-ai's own output re-ask, inside one judge request
    prose.clear()
    with pytest.raises(UnexpectedModelBehavior):
        await _judge(FunctionModel(fn), policy=POLICY).judge(prompt="p", output="o", rubric="r")
    assert len(prose) == once  # the wire loop added no re-send


async def test_policy_none_sends_once() -> None:
    sent: list[int] = []
    with pytest.raises(ModelHTTPError):
        await _judge(_flaky([429], sent), policy=ports.WirePolicy.none()).judge(prompt="p", output="o", rubric="r")
    assert len(sent) == 1


async def test_a_wait_past_the_deadline_is_not_taken(_backoff_does_not_sleep: list[float]) -> None:
    sent: list[int] = []
    events: list[str] = []
    with pytest.raises(ModelHTTPError):
        await _judge(_flaky([429], sent), events, policy=POLICY, deadline_s=0.5).judge(
            prompt="p", output="o", rubric="r"
        )
    assert len(sent) == 1 and _backoff_does_not_sleep == [] and "llm_deadline_expired" in events


async def test_model_for_yields_a_fresh_model_per_try() -> None:
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    sent: list[int] = []
    model = _flaky([429], sent)
    entered: list[int] = []

    @asynccontextmanager
    async def model_for() -> AsyncIterator[FunctionModel]:
        entered.append(1)
        yield model

    v = await PydanticEvalsJudge(JUDGE_ROLE, model_for=model_for, policy=POLICY).judge(
        prompt="p", output="o", rubric="r"
    )
    assert v.passed and entered == [1, 1] and len(sent) == 2
