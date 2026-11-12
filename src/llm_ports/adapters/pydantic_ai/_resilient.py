"""The wire-retry model wrapper: every model request the agent makes goes through the shared loop.

`ResilientModel` wraps the call's pydantic-ai `Model` and sends each model request through
`adapters._retry.WireRetry`. So the three loops a call can run stay three loops: pydantic-ai's
output re-ask makes a NEW model request (which gets its own wire retries), a wire retry re-sends
only the request that failed (never a re-ask that already succeeded), and fallback runs in the
facade only after this loop gave up. Its per-request bound is `Deadline.attempt_s`; the call's
own bound is enforced around it by the generator.

A streamed request is retried only while it is being opened: once the stream is open, a failure
is `in_stream` and never retried.

Invariants:
    - The provider SDK's own retries stay 0 and pydantic-ai's retrying transport is not installed.
    - `asyncio.CancelledError` passes through untouched.

Rationale: ADR-0021.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from llm_ports._extras import extra
from llm_ports.adapters._classify import Classifier, WireFacts
from llm_ports.adapters._retry import Judged, WireRetry
from llm_ports.adapters.pydantic_ai._capture import CaptureLike
from llm_ports.adapters.pydantic_ai._classify import classify_failure
from llm_ports.ports.outcomes import exception_chain

with extra("pydantic-ai"):
    from pydantic_ai import RunContext
    from pydantic_ai.exceptions import ModelHTTPError, ModelRequestTimeout
    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
    from pydantic_ai.models.wrapper import WrapperModel
    from pydantic_ai.settings import ModelSettings


def judge_for(
    cap: CaptureLike, *, extra: Sequence[Classifier], gated: tuple[type[BaseException], ...]
) -> Callable[[Any], Judged]:
    """The shared table's verdict on one model request: its `Failure` and the headers it saw.

    A returned `ModelResponse` is a success; a raised exception is classified with the capture's
    facts of the latest response.
    """

    def judge(outcome: Any) -> Judged:
        if not isinstance(outcome, BaseException):
            return None, {}
        last: WireFacts | None = getattr(cap, "last", None)
        facts = last or WireFacts()
        got = classify_failure(outcome, facts, extra=extra, gated=gated)
        headers: Mapping[str, str] = facts.headers
        for e in exception_chain(outcome):
            if isinstance(e, ModelHTTPError) and e.headers:
                headers = {**facts.headers, **{k.lower(): v for k, v in e.headers.items()}}
                break
        return got, headers

    return judge


class ResilientModel(WrapperModel):
    """A pydantic-ai `Model` whose every request is sent through one call's `WireRetry`.

    Args:
        wrapped: The model the call runs on.
        wire: The call's retry state.
        judge: Classifies one request's outcome (`judge_for`).
        bound: The seconds one request may take, read when it starts; None for no per-request bound.
        reset: Called before every try, so the capture's latest facts are this try's.
    """

    def __init__(
        self,
        wrapped: Model,
        wire: WireRetry,
        judge: Callable[[Any], Judged],
        *,
        bound: Callable[[], float | None],
        reset: Callable[[], None],
    ) -> None:
        super().__init__(wrapped)
        self._wire = wire
        self._judge = judge
        self._bound = bound
        self._reset = reset

    async def _once[T](self, send: Callable[[], Awaitable[T]]) -> T:
        self._reset()
        bound = self._bound()
        if bound is None:
            return await send()
        try:
            async with asyncio.timeout(bound):
                return await send()
        except TimeoutError:
            raise ModelRequestTimeout(self.model_name, bound) from None

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Send one model request, re-sent while the call's wire policy allows."""
        return await self._wire.run(
            lambda: self._once(lambda: self.wrapped.request(messages, model_settings, model_request_parameters)),
            self._judge,
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Open one streamed request, re-opened while the wire policy allows; never retried once open."""

        async def open_() -> tuple[AbstractAsyncContextManager[StreamedResponse], StreamedResponse]:
            cm = self.wrapped.request_stream(messages, model_settings, model_request_parameters, run_context)
            return cm, await cm.__aenter__()

        cm, stream = await self._wire.run(lambda: self._once(open_), self._judge)
        try:
            yield stream
        except BaseException as exc:
            if not await cm.__aexit__(type(exc), exc, exc.__traceback__):
                raise
        else:
            await cm.__aexit__(None, None, None)


__all__: list[str] = []
