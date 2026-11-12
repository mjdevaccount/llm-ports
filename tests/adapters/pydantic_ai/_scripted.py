"""A FunctionModel that answers from a script, shared by the pydantic-ai adapter tests."""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


def scripted(*texts: str) -> tuple[FunctionModel, list[int]]:
    """Return a model answering request N with `texts[N]` (the last text repeats), and its call log."""
    calls: list[int] = []

    def fn(_m: list[ModelMessage], _i: AgentInfo) -> ModelResponse:
        calls.append(1)
        return ModelResponse(parts=[TextPart(texts[min(len(calls) - 1, len(texts) - 1)])])

    return FunctionModel(fn), calls


def scripted_model(*texts: str) -> FunctionModel:
    """`scripted` without the call log."""
    return scripted(*texts)[0]
