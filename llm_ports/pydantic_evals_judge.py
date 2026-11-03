"""PydanticEvalsJudge -- the `Judge` port on pydantic-evals' `LLMJudge` (OUR seam, THEIR impl).

pydantic-evals' `LLMJudge` renders the Input/Output/Rubric prompt and binds the verdict to its
`GradingOutput` grammar (reason, pass, score); this adapter only picks the model and translates the
verdict into a `JudgeVerdict`. Install with the `evals` extra (`llm-ports[evals]`).

WHERE THE MODEL COMES FROM is the consumer's, as for the Generator and the Embedder:
  * `model=`      a fixed pydantic-ai `Model` (tests pass a `FunctionModel`);
  * `model_for=`  `() -> async context manager yielding a Model`, per call -- how a consumer puts
                  the judge's own call on its transport (admission, leases, a work row), so a judge
                  is never the one uncapped call. Build it with `profile=JUDGE_PROFILE`.

THE GRADE RIDES THE NATIVE GRAMMAR. pydantic-evals' agent asks for `GradingOutput` as a TOOL call
by default; a self-hosted model answered it in a mix of its XML tool-call syntax and the JSON the
agent's examples show, so 400 of 854 judged rows carried a reason ending `", "pass": false,
"score": 0.0` or a `<parameter=pass>` tag, and one "pass" reason was a whole second transcript.
`JUDGE_PROFILE` puts the judge on `NativeOutput` -- a strict json_schema response_format, the mode
a grammar-compiling server enforces -- and `verdict_of` refuses a reason still carrying tool-call
markup, or a leaked verdict that contradicts the parsed one (the caller records "could not run"
naming why, never a guessed verdict).

IMAGES: a call that sent images is judged WITH them (`images=`, beside the port's text-only
signature), as pydantic-evals multimodal Input -- a vision reply judged against a prompt that said
only "read this chart" is graded blind.

RAISES when the judging model could not answer (the port's contract): the caller decides what an
unjudged sample means, never this adapter. There is no default verdict anywhere in this module.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any

from pydantic_ai.messages import BinaryContent
from pydantic_ai.models import Model
from pydantic_ai.profiles import ModelProfile
from pydantic_evals.evaluators import EvaluatorContext, LLMJudge
from pydantic_evals.evaluators.context import SpanTreeRecordingError
from pydantic_evals.evaluators.evaluator import EvaluationReason

from llm_ports.ports import JudgeVerdict
from llm_ports.roles import Role

JudgeModelFor = Callable[[], AbstractAsyncContextManager[Model]]

JUDGE_NAME = "pydantic_evals"
SCORE_KEY = "score"
PASS_KEY = "pass"
#: The judge's grade as a strict json_schema response_format, never a tool call (see module doc).
JUDGE_PROFILE: ModelProfile = {"supports_json_schema_output": True, "default_structured_output_mode": "native"}
#: A tool call the MODEL emitted inside the reason: a tag opening a line, or the Qwen XML form
#: (`<parameter=pass>`, `<function=final_result>`), which no prose spells. Anchored so a reason
#: that QUOTES a tag mid-sentence from the judged reply (a rubric tells the judge to quote) is
#: read, not refused.
_MARKUP = re.compile(r"(?m)^\s*</?(?:tool_call|parameter|function)\b|<(?:parameter|function)=", re.I)
#: The grade's own JSON keys leaking onto the TAIL of the reason (`... law 2.", "pass": false,
#: "score": 0.0`). Anchored to the end: the same keys quoted mid-reason are content, not a leak.
_VAL = r"(?:true|false|-?\d+(?:\.\d+)?)"
_LEAK = re.compile(
    rf'"\s*,\s*"(?:pass|score)"\s*:\s*{_VAL}(?:\s*,\s*"(?:pass|score)"\s*:\s*{_VAL})*'
    r"\s*\}?\s*$",
    re.I,
)
_STATED_PASS = re.compile(r'"pass"\s*:\s*(true|false)', re.I)


def _scalar(v: Any) -> tuple[Any, str | None]:
    """An LLMJudge output entry is a scalar or an `EvaluationReason(value, reason)`."""
    if isinstance(v, EvaluationReason):
        return v.value, v.reason
    return v, None


def clean_reason(reason: str, passed: bool) -> str:
    """Pure. The reason up to where the grade's own JSON leaked into it. RAISES when the reason
    carries tool-call markup (its verdict is not readable) or states a verdict that contradicts
    the parsed one."""
    if _MARKUP.search(reason):
        raise ValueError(f"judge reason carries tool-call markup: {reason[:120]!r}")
    m = _LEAK.search(reason)
    if m is None:
        return reason
    stated = _STATED_PASS.search(reason[m.start() :])
    if stated is not None and (stated.group(1).lower() == "true") != passed:
        raise ValueError(f"judge reason states a verdict that contradicts pass={passed}")
    return reason[: m.start()].rstrip(' ",')


def verdict_of(out: dict[str, Any]) -> JudgeVerdict:
    """Pure: the `LLMJudge.evaluate` dict (`score` + `pass`, each with the model's reason) into
    the port's value. A verdict missing either key is a judge that did not answer the question."""
    if SCORE_KEY not in out or PASS_KEY not in out:
        raise ValueError(f"LLMJudge returned {sorted(out)} without {SCORE_KEY!r}/{PASS_KEY!r}")
    score, s_reason = _scalar(out[SCORE_KEY])
    passed, p_reason = _scalar(out[PASS_KEY])
    reason = clean_reason(p_reason or s_reason or "", bool(passed))
    return JudgeVerdict(passed=bool(passed), score=float(score), reason=reason, judge=JUDGE_NAME)


def judge_inputs(prompt: str, images: Sequence[tuple[bytes, str]] = ()) -> str | list[Any]:
    """The LLMJudge Input: the text alone, or the text then each image as `BinaryContent`
    (pydantic-evals renders a sequence Input part by part, an image as multimodal content)."""
    if not images:
        return prompt
    return [prompt, *(BinaryContent(data=b, media_type=mime) for b, mime in images)]


class PydanticEvalsJudge:
    """The `Judge` port. `role` carries the judge's model name and sampling (`model_settings()`);
    pass exactly one of `model=` / `model_for=`."""

    def __init__(self, role: Role, *, model: Model | None = None, model_for: JudgeModelFor | None = None) -> None:
        if (model is None) == (model_for is None):
            raise ValueError("PydanticEvalsJudge: pass exactly one of model= or model_for=")
        self._role = role
        self._model = model
        self._model_for = model_for

    @property
    def model_name(self) -> str:
        return self._role.resolved_model

    async def judge(
        self, *, prompt: str, output: Any, rubric: str, images: Sequence[tuple[bytes, str]] = ()
    ) -> JudgeVerdict:
        inputs = judge_inputs(prompt, images)
        if self._model is not None:
            return await self._run(self._model, inputs, output, rubric)
        assert self._model_for is not None
        async with self._model_for() as model:
            return await self._run(model, inputs, output, rubric)

    async def _run(self, model: Model, inputs: str | list[Any], output: Any, rubric: str) -> JudgeVerdict:
        judge = LLMJudge(
            rubric=rubric,
            model=model,
            include_input=True,
            model_settings=self._role.model_settings(),  # type: ignore[arg-type]
            score={"evaluation_name": SCORE_KEY, "include_reason": True},
            assertion={"evaluation_name": PASS_KEY, "include_reason": True},
        )
        ctx: EvaluatorContext[Any, Any, Any] = EvaluatorContext(
            name=self._role.name,
            inputs=inputs,
            metadata=None,
            expected_output=None,
            output=output,
            duration=0.0,
            _span_tree=SpanTreeRecordingError("no span tree: one-off judge, not a dataset run"),
            attributes={},
            metrics={},
        )
        out = await judge.evaluate(ctx)
        return verdict_of(dict(out) if isinstance(out, dict) else {})


__all__ = [
    "JUDGE_NAME",
    "JUDGE_PROFILE",
    "PASS_KEY",
    "SCORE_KEY",
    "JudgeModelFor",
    "PydanticEvalsJudge",
    "clean_reason",
    "judge_inputs",
    "verdict_of",
]
