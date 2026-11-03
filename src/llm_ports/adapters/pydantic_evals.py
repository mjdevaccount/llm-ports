"""The `Judge` port implemented on pydantic-evals' `LLMJudge`.

pydantic-evals' `LLMJudge` renders the Input/Output/Rubric prompt and binds the verdict to its
`GradingOutput` grammar (reason, pass, score); this adapter picks the model and translates the
verdict into a `JudgeVerdict`. Install with the `evals` extra (`llm-ports[evals]`).

The model source is the caller's, as for the generator and the embedder:

- `model=`: a fixed pydantic-ai `Model` (tests pass a `FunctionModel`).
- `model_for=`: `() -> async context manager yielding a Model`, called per request, so the
  judge's own call goes through the caller's transport (admission control, leases, a work queue)
  like every other call. Build it with `profile=JUDGE_PROFILE`.

The grade uses native structured output. pydantic-evals' agent asks for `GradingOutput` as a tool
call by default; `JUDGE_PROFILE` puts the judge on `NativeOutput`, a strict json_schema
`response_format`, which is the mode a grammar-compiling server enforces. `verdict_of` refuses a
reason that still carries tool-call markup, or a leaked verdict that contradicts the parsed one,
so the caller records that the judge could not run, and why.

A call that sent images is judged with them (`images=`, beside the port's text-only signature),
as pydantic-evals multimodal Input.

Invariants:
    - `PydanticEvalsJudge.judge` raises when the judging model could not answer, as the port
      requires; the caller decides what an unjudged sample means.
    - No default verdict exists anywhere in this module.

Rationale: ADR-0001.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, cast

from llm_ports._extras import extra
from llm_ports.call.roles import Role
from llm_ports.ports import JudgeVerdict

with extra("evals"):
    from pydantic_ai.messages import BinaryContent
    from pydantic_ai.models import Model
    from pydantic_ai.profiles import ModelProfile
    from pydantic_ai.settings import ModelSettings
    from pydantic_evals.evaluators import EvaluatorContext, LLMJudge
    from pydantic_evals.evaluators.evaluator import EvaluationReason
    from pydantic_evals.otel import SpanTreeRecordingError

JudgeModelFor = Callable[[], AbstractAsyncContextManager[Model]]

JUDGE_NAME = "pydantic_evals"
SCORE_KEY = "score"
PASS_KEY = "pass"
_EXCERPT = 120  # the refused reason quoted in the `ValueError`
#: The judge's grade as a strict json_schema `response_format`, never a tool call.
JUDGE_PROFILE: ModelProfile = {"supports_json_schema_output": True, "default_structured_output_mode": "native"}
#: Tool-call markup the model emitted inside the reason: a tag opening a line, or the XML tool-call
#: form (`<parameter=pass>`, `<function=final_result>`), which prose does not use. Anchored to a
#: line start, so a reason that quotes a tag mid-sentence from the judged reply is read, not refused.
_MARKUP = re.compile(r"(?m)^\s*</?(?:tool_call|parameter|function)\b|<(?:parameter|function)=", re.I)
#: The grade's own JSON keys leaked onto the tail of the reason (`...", "pass": false, "score": 0.0`).
#: Anchored to the end: the same keys quoted mid-reason are content, not a leak.
_VAL = r"(?:true|false|-?\d+(?:\.\d+)?)"
_LEAK = re.compile(
    rf'"\s*,\s*"(?:pass|score)"\s*:\s*{_VAL}(?:\s*,\s*"(?:pass|score)"\s*:\s*{_VAL})*'
    r"\s*\}?\s*$",
    re.I,
)
_STATED_PASS = re.compile(r'"pass"\s*:\s*(true|false)', re.I)


def _scalar(v: Any) -> tuple[Any, str | None]:
    """Split an `LLMJudge` output entry (a scalar or an `EvaluationReason`) into value and reason."""
    if isinstance(v, EvaluationReason):
        return v.value, v.reason
    return v, None


def clean_reason(reason: str, passed: bool) -> str:
    """Return the reason cut where the grade's own JSON leaked into it.

    Pure.

    Args:
        reason: The judge's reason text.
        passed: The parsed verdict, checked against any verdict the leaked tail states.

    Raises:
        ValueError: When the reason carries tool-call markup (its verdict is unreadable), or its
            leaked tail states a verdict that contradicts `passed`.
    """
    if _MARKUP.search(reason):
        raise ValueError(f"judge reason carries tool-call markup: {reason[:_EXCERPT]!r}")
    m = _LEAK.search(reason)
    if m is None:
        return reason
    stated = _STATED_PASS.search(reason[m.start() :])
    if stated is not None and (stated.group(1).lower() == "true") != passed:
        raise ValueError(f"judge reason states a verdict that contradicts pass={passed}")
    return reason[: m.start()].rstrip(' ",')


def verdict_of(out: dict[str, Any]) -> JudgeVerdict:
    """Translate an `LLMJudge.evaluate` result into the port's `JudgeVerdict`.

    `out` holds `score` and `pass`, each with the model's reason. Pure.

    Raises:
        ValueError: When either key is missing (the judge did not answer the question), or when
            `clean_reason` refuses the reason.
    """
    if SCORE_KEY not in out or PASS_KEY not in out:
        raise ValueError(f"LLMJudge returned {sorted(out)} without {SCORE_KEY!r}/{PASS_KEY!r}")
    score, s_reason = _scalar(out[SCORE_KEY])
    passed, p_reason = _scalar(out[PASS_KEY])
    reason = clean_reason(p_reason or s_reason or "", bool(passed))
    return JudgeVerdict(passed=bool(passed), score=float(score), reason=reason, judge=JUDGE_NAME)


def judge_inputs(prompt: str, images: Sequence[tuple[bytes, str]] = ()) -> str | list[Any]:
    """Build the `LLMJudge` Input: the text alone, or the text then each image as `BinaryContent`.

    pydantic-evals renders a sequence Input part by part, an image as multimodal content.
    """
    if not images:
        return prompt
    return [prompt, *(BinaryContent(data=b, media_type=mime) for b, mime in images)]


class PydanticEvalsJudge:
    """The `Judge` port on pydantic-evals; the module docstring states the contract.

    `role` carries the judge's model name and sampling (`Role.model_settings()`). Exactly one of
    `model` and `model_for` is required.

    Raises:
        ValueError: When neither or both of `model` and `model_for` are given.
    """

    def __init__(self, role: Role, *, model: Model | None = None, model_for: JudgeModelFor | None = None) -> None:
        if (model is None) == (model_for is None):
            raise ValueError("PydanticEvalsJudge: pass exactly one of model= or model_for=")
        self._role = role
        self._model = model
        self._model_for = model_for

    @property
    def model_name(self) -> str:
        """The judge's resolved model name, from its role."""
        return self._role.resolved_model

    async def judge(
        self, *, prompt: str, output: Any, rubric: str, images: Sequence[tuple[bytes, str]] = ()
    ) -> JudgeVerdict:
        """Grade `output` against `rubric`, with `prompt` and any `images` as the judged input.

        A failure of the judging model's call propagates: the caller decides what an unjudged
        sample means.

        Raises:
            ValueError: When the reply lacks a score or a verdict, or its reason is unreadable
                (`verdict_of`).

        Rationale: ADR-0001.
        """
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
            model_settings=cast(ModelSettings, self._role.model_settings()),
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
