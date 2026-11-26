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

The judge's model request rides the same wire-retry loop as every other adapter
(`adapters._retry.WireRetry`, ADR-0021): a rate-limited, overloaded or server-failed request is
re-sent on `policy=` (default `WirePolicy.default()`), classified by the shared table
(`adapters._classify.classify_core`, which reads the status off pydantic-ai's error by duck type,
so this adapter imports no other). Before 1.31 the judge was the one adapter outside it: a single
429 from a one-slot server ended the sample. An unreadable verdict (`verdict_of`'s ValueError) is
`internal` and never retried.

Invariants:
    - `PydanticEvalsJudge.judge` raises when the judging model could not answer, as the port
      requires; the caller decides what an unjudged sample means.
    - No default verdict exists anywhere in this module.
    - One retry owner: the model source's SDK retries stay 0; this loop is the only re-send.

Rationale: ADR-0001.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, cast

from llm_ports._extras import extra
from llm_ports.adapters._classify import Classifier, WireFacts, classify_core
from llm_ports.adapters._retry import Judged, WireRetry
from llm_ports.call.models import translate_settings
from llm_ports.call.roles import Role
from llm_ports.observability import EventHook, log_events
from llm_ports.ports import JudgeVerdict, WirePolicy
from llm_ports.ports.outcomes import exception_chain

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

    Args:
        role: The judge's model name and sampling.
        model: A fixed pydantic-ai `Model`.
        model_for: A per-request model factory (the caller's transport).
        policy: How a failed judge request is re-sent; None is `WirePolicy.default()`.
        deadline_s: The seconds one `judge` call may spend, retry waits included; a wait that
            would end past it is not taken. None for no bound (the caller's own timeout rules).
        extra: Classifiers consulted before the shared table, in order.
        on_event: Receives `llm_wire_retry` / `llm_deadline_expired`; without one they go to the
            stdlib logger `llm_ports.adapters.pydantic_evals`.

    Raises:
        ValueError: When neither or both of `model` and `model_for` are given.
    """

    def __init__(
        self,
        role: Role,
        *,
        model: Model | None = None,
        model_for: JudgeModelFor | None = None,
        policy: WirePolicy | None = None,
        deadline_s: float | None = None,
        extra: Sequence[Classifier] = (),
        on_event: EventHook | None = None,
    ) -> None:
        if (model is None) == (model_for is None):
            raise ValueError("PydanticEvalsJudge: pass exactly one of model= or model_for=")
        self._role = role
        self._model = model
        self._model_for = model_for
        self._policy = policy if policy is not None else WirePolicy.default()
        self._deadline_s = deadline_s
        self._extra = tuple(extra)
        self._emit: EventHook = on_event or log_events("llm_ports.adapters.pydantic_evals")

    @property
    def model_name(self) -> str:
        """The judge's resolved model name, from its role."""
        return self._role.resolved_model

    async def judge(
        self, *, prompt: str, output: Any, rubric: str, images: Sequence[tuple[bytes, str]] = ()
    ) -> JudgeVerdict:
        """Grade `output` against `rubric`, with `prompt` and any `images` as the judged input.

        A failure of the judging model's call propagates once the wire-retry policy gives up:
        the caller decides what an unjudged sample means.

        Raises:
            ValueError: When the reply lacks a score or a verdict, or its reason is unreadable
                (`verdict_of`).

        Rationale: ADR-0001.
        """
        inputs = judge_inputs(prompt, images)
        loop = asyncio.get_running_loop()
        deadline_at = loop.time() + self._deadline_s if self._deadline_s is not None else None
        wire = WireRetry(
            self._policy, model=self.model_name, deadline_at=deadline_at, emit=self._emit, label=self._role.name
        )
        return await wire.run(lambda: self._once(inputs, output, rubric), self._classify)

    async def _once(self, inputs: str | list[Any], output: Any, rubric: str) -> JudgeVerdict:
        """One judge request: a fresh model from the source per try, so each try is its own request."""
        if self._model is not None:
            return await self._run(self._model, inputs, output, rubric)
        assert self._model_for is not None
        async with self._model_for() as model:
            return await self._run(model, inputs, output, rubric)

    def _classify(self, outcome: Any) -> Judged:
        """The shared table's verdict on one try: its `Failure` and the response headers it carried."""
        if not isinstance(outcome, BaseException):
            return None, {}
        headers: Mapping[str, str] = {}
        for e in exception_chain(outcome):
            got = getattr(e, "headers", None)
            if isinstance(got, Mapping) and got:
                headers = {str(k).lower(): str(v) for k, v in got.items()}
                break
        return classify_core(outcome, WireFacts(headers=headers), extra=self._extra), headers

    def _settings(self) -> dict[str, Any]:
        """The role's settings, with `thinking` spelled the way the judge's model takes it.

        A schema 2 role emits provider-neutral `thinking: bool`; a `chat_template` model reads it
        only as `extra_body.chat_template_kwargs.enable_thinking`. Untranslated, the box's template
        default (thinking ON) spent the whole token budget reasoning and every verdict was
        "token limit exceeded" (1.26.1). A role whose model override is not its tier's model has
        no `ModelInfo` here, so its settings pass through unchanged.
        """
        info = self._role.resolved_tier.info
        if info is not None and self._role.model not in (None, info.id):
            info = None
        return translate_settings(self._role.model_settings(), info)[0]

    async def _run(self, model: Model, inputs: str | list[Any], output: Any, rubric: str) -> JudgeVerdict:
        judge = LLMJudge(
            rubric=rubric,
            model=model,
            include_input=True,
            model_settings=cast(ModelSettings, self._settings()),
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
