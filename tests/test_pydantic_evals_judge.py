"""pydantic_evals_judge: the Judge port on pydantic-evals' LLMJudge.

A FunctionModel answers the judge agent's `final_result` tool (pydantic-evals' GradingOutput:
reason, pass, score) so the whole LLMJudge path runs with no wire; `verdict_of` / `clean_reason`
are pinned pure. Reason shapes are copied from a production judge table (400 of 854 rows leaked
the grade's own syntax into the reason under tool-mode output).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import BinaryContent, ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals.evaluators.evaluator import EvaluationReason

from llm_ports import ports
from llm_ports.pydantic_evals_judge import (
    JUDGE_PROFILE,
    PydanticEvalsJudge,
    clean_reason,
    judge_inputs,
    verdict_of,
)
from tests.conftest import Role

JUDGE_ROLE = Role("llm.judge", tier="box", temperature=0.0)


def _judge_model(grading: dict[str, Any], seen: list[str]) -> FunctionModel:
    def fn(messages: list[ModelMessage], _i: AgentInfo) -> ModelResponse:
        for m in messages:
            for part in m.parts:
                content = getattr(part, "content", None)
                if isinstance(content, str):
                    seen.append(content)
        return ModelResponse(parts=[ToolCallPart(tool_name="final_result", args=grading)])

    return FunctionModel(fn)


def _judge(model: Model) -> PydanticEvalsJudge:
    return PydanticEvalsJudge(JUDGE_ROLE, model=model)


async def test_the_judge_reads_rubric_input_and_output_and_returns_the_verdict() -> None:
    seen: list[str] = []
    j = _judge(_judge_model({"reason": "law 2: 4242 is invented", "pass": False, "score": 0.2}, seen))
    v = await j.judge(prompt="the prompt text", output='{"n": 4242}', rubric="RUBRIC-MARKER")
    assert (v.passed, v.score, v.judge) == (False, 0.2, "pydantic_evals")
    assert "4242" in v.reason
    blob = "\n".join(seen)
    assert "RUBRIC-MARKER" in blob and "the prompt text" in blob and "4242" in blob


async def test_a_passing_verdict() -> None:
    v = await _judge(_judge_model({"reason": "clean", "pass": True, "score": 1.0}, [])).judge(
        prompt="p", output="o", rubric="r"
    )
    assert v.passed is True and v.score == 1.0


async def test_a_judge_that_answers_prose_instead_of_the_tool_raises() -> None:
    """The port's contract: no default verdict -- a judge that did not answer RAISES."""

    def fn(_m: list[ModelMessage], _i: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("I refuse to grade")])

    with pytest.raises(UnexpectedModelBehavior, match="retries"):
        await _judge(FunctionModel(fn)).judge(prompt="p", output="o", rubric="r")


async def test_model_for_is_entered_per_call_and_its_model_judges() -> None:
    """The consumer's wire seam: the context manager opens around each judge call (a transport, a
    lease, a work row) and the model it yields is the one that answers."""
    entered: list[str] = []

    @asynccontextmanager
    async def model_for() -> AsyncIterator[Model]:
        entered.append("open")
        yield _judge_model({"reason": "ok", "pass": True, "score": 1.0}, [])
        entered.append("close")

    j = PydanticEvalsJudge(JUDGE_ROLE, model_for=model_for)
    assert (await j.judge(prompt="p", output="o", rubric="r")).passed is True
    assert entered == ["open", "close"]
    assert j.model_name == JUDGE_ROLE.resolved_model == "small-chat"


def test_exactly_one_model_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        PydanticEvalsJudge(JUDGE_ROLE)
    with pytest.raises(ValueError, match="exactly one"):
        PydanticEvalsJudge(
            JUDGE_ROLE,
            model=FunctionModel(lambda _m, _i: None),  # type: ignore[arg-type,return-value]
            model_for=lambda: None,
        )  # type: ignore[arg-type,return-value]


def test_the_judge_satisfies_the_port() -> None:
    assert isinstance(_judge(FunctionModel(lambda _m, _i: None)), ports.Judge)  # type: ignore[arg-type,return-value]


def test_verdict_of_unwraps_reasons_and_refuses_a_half_answer() -> None:
    v = verdict_of({"score": EvaluationReason(0.5, "half"), "pass": EvaluationReason(False, "no")})
    assert (v.passed, v.score, v.reason) == (False, 0.5, "no")
    assert verdict_of({"score": 1, "pass": True}).reason == ""
    with pytest.raises(ValueError, match="without"):
        verdict_of({"score": 1.0})


class TestCleanReason:
    """MUST-FIRE: the tool-call grade leaked its own syntax into `reason` on 400 of 854 rows."""

    def test_a_clean_reason_is_untouched(self) -> None:
        assert verdict_of({"score": 1.0, "pass": EvaluationReason(True, "clean reply")}).reason == "clean reply"

    def test_a_leaked_json_tail_that_agrees_is_cut(self) -> None:
        r = (
            'The output invents Q3 revenue. This violates Law 2 (Numbers from the Input Only).", '
            '"pass": false, "score": 0.0'
        )
        v = verdict_of({"score": 0.0, "pass": EvaluationReason(False, r)})
        assert (
            v.passed is False
            and v.reason == "The output invents Q3 revenue. This violates Law 2 (Numbers from the Input Only)."
        )

    def test_a_leaked_verdict_that_contradicts_the_parsed_one_is_refused(self) -> None:
        r = 'The output is clean.", "pass": true, "score": 1.0'
        with pytest.raises(ValueError, match="contradicts"):
            verdict_of({"score": 0.0, "pass": EvaluationReason(False, r)})
        assert clean_reason(r, True) == "The output is clean.", "the agreeing direction is cut, not refused"

    def test_tool_call_markup_in_the_reason_is_refused_either_way(self) -> None:
        """A fail row and a PASS whose reason was a second transcript: neither verdict is readable,
        so neither becomes a pass or a fail -- the caller records could-not-run."""
        fail_row = "The output invents the 'magnitude'.\",\n<parameter=pass>\nfalse"
        pass_row = (
            "The output invents the number 'geo_threat',\n  \"pass\": true\n}\n<tool_call>\n"
            "<function=final_result>\n<parameter=reason>\nThe output correctly classifies"
        )
        for passed, r in ((False, fail_row), (True, pass_row)):
            with pytest.raises(ValueError, match="markup"):
                verdict_of({"score": float(passed), "pass": EvaluationReason(passed, r)})

    def test_a_tag_or_key_the_judge_QUOTES_is_content_not_a_leak(self) -> None:
        """A rubric tells the judge to quote the offending fragment. A reply that itself carries a
        tool-call tag or the grade's keys mid-text must be read as the verdict, not refused nor
        truncated at the quote."""
        quoted_tag = "The reply leaks raw <tool_call> and <function_call> tags into the note."
        assert verdict_of({"score": 0.0, "pass": EvaluationReason(False, quoted_tag)}).reason == quoted_tag
        quoted_key = 'The reply emits the literal text ", "pass": true inside its prose, which is wrong.'
        v = verdict_of({"score": 0.0, "pass": EvaluationReason(False, quoted_key)})
        assert v.reason == quoted_key, "a mid-reason quote was cut as if it were the leaked tail"


async def test_the_grade_rides_the_native_grammar_not_a_tool_call() -> None:
    """JUDGE_PROFILE on the model: pydantic-ai sends `GradingOutput` as a json_schema
    response_format and reads a JSON TEXT reply. A model answering text (no tool) is graded --
    the tool-mode test above shows the same reply was a failure there."""
    modes: list[Any] = []

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        modes.append(info.model_request_parameters.output_mode)
        return ModelResponse(parts=[TextPart('{"reason": "clean", "pass": true, "score": 1.0}')])

    v = await _judge(FunctionModel(fn, profile=JUDGE_PROFILE)).judge(prompt="p", output="o", rubric="r")
    assert (v.passed, v.score, v.reason) == (True, 1.0, "clean") and modes == ["native"]


async def test_images_reach_the_judge_model_beside_the_text() -> None:
    """A vision role judged without its image failed 39 of 39 for figures 'not present in the input'.
    Now the image rides as multimodal Input."""
    got: list[Any] = []

    def fn(messages: list[ModelMessage], _i: AgentInfo) -> ModelResponse:
        for m in messages:
            for part in m.parts:
                c = getattr(part, "content", None)
                if isinstance(c, list):
                    got.extend(c)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result", args={"reason": "levels match the chart", "pass": True, "score": 1.0}
                )
            ]
        )

    v = await _judge(FunctionModel(fn)).judge(
        prompt="read the chart", output='{"support": 580.0}', rubric="r", images=[(b"\x89PNG-bytes", "image/png")]
    )
    assert v.passed is True
    imgs = [c for c in got if isinstance(c, BinaryContent)]
    assert len(imgs) == 1 and imgs[0].data == b"\x89PNG-bytes" and imgs[0].media_type == "image/png"
    assert any(isinstance(c, str) and "read the chart" in c for c in got)
    assert judge_inputs("p") == "p", "a text call stays text"
