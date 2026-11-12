"""The `openai_compatible` provider kind: a schema 2 provider built on pydantic-ai.

`OpenAICompatibleFactory` is the built-in `call.providers.ProviderFactory` for any OpenAI-wire
endpoint (a LiteLLM gateway, vLLM, llama.cpp's server, OpenAI). For one provider it builds:

- a `Generator` (`PydanticAIGenerator` over `openai_compatible`) that spells each request's
  provider-neutral settings the way the request's model needs (`translate_settings`);
- a `Chat` (`PydanticAIChat` on the same client) with the same settings translation;
- a `TokenCounter` per model id that states `count_tokens` (`HttpTokenCounter` on the gateway's
  `/utils/token_counter`, LiteLLM's count route, at the base URL's root);
- an `Embedder` per model id, when one of its models states `embed`;
- a `Probe` (`GET <base_url>/models`, 5 s).

Register it through the provider registry:

    providers = load_providers(builtins=[builtin("openai_compatible", OpenAICompatibleFactory())], pin=pin)

Invariants:
    - Provider detail lives here, not in a `Role`: a schema 2 role sends `thinking: bool`, and
      `translate_settings` turns it into what the model's `ModelInfo.thinking` names.
    - A role that asks to think on a model with no thinking switch is sent without it, and
      `llm_call_warning` (kind `unsupported`) says so once per model.

Rationale: ADR-0022.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import replace
from typing import Any

from llm_ports._extras import extra
from llm_ports._headers import HEADER_VALUE_RULE, header_value_ok
from llm_ports.call.models import ModelInfo, ProviderSpec, TableInvalid
from llm_ports.call.providers import BuildDeps, Built
from llm_ports.observability import EventHook, log_events
from llm_ports.ports import (
    Chat,
    ChatRequest,
    Embedder,
    Generated,
    Generator,
    GenRequest,
    RetryMessage,
    StreamEvent,
    TokenCounter,
    Validator,
)

with extra("pydantic-ai"):
    import httpx

    from llm_ports.adapters._probe import HttpProbe
    from llm_ports.adapters._token_count import HttpTokenCounter
    from llm_ports.adapters.pydantic_ai._chat import PydanticAIChat
    from llm_ports.adapters.pydantic_ai._embedder import PydanticAIEmbedder, openai_compatible_embeddings
    from llm_ports.adapters.pydantic_ai._generator import PydanticAIGenerator, openai_compatible

_log_event = log_events("llm_ports.call")


def translate_settings(settings: Mapping[str, Any], info: ModelInfo | None) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Spell a request's provider-neutral `thinking` switch the way its model needs. Pure.

    Args:
        settings: The request's settings; a schema 2 role puts `thinking: bool` in them.
        info: The request's model, or None when the table does not know it (the settings pass
            through unchanged, so pydantic-ai's own unified `thinking` applies).

    Returns:
        The settings to send, and the names of settings the model cannot take (dropped):
        `unified` keeps `thinking` (pydantic-ai's `ModelSettings["thinking"]`); `chat_template`
        moves it to `extra_body.chat_template_kwargs.enable_thinking`; `none` drops it, and names
        it when the role asked to think.
    """
    out = dict(settings)
    if info is None or "thinking" not in out or info.thinking == "unified":
        return out, ()
    want = bool(out.pop("thinking"))
    if info.thinking == "chat_template":
        extra_body = dict(out.get("extra_body") or {})
        extra_body["chat_template_kwargs"] = {**(extra_body.get("chat_template_kwargs") or {}), "enable_thinking": want}
        out["extra_body"] = extra_body
        return out, ()
    return out, ("thinking",) if want else ()


class _Translating:
    """Translates a request's settings for its model, warning once per dropped setting and model.

    `aclose` closes the wrapped port's client, so `Backends.aclose` reaches it through the wrapper.
    """

    def __init__(self, models: Sequence[ModelInfo], *, provider: str, on_event: EventHook | None) -> None:
        self._models = {m.id: m for m in models}
        self._provider = provider
        self._emit = on_event or _log_event
        self._warned: set[tuple[str, str]] = set()

    async def aclose(self) -> None:
        """Close the wrapped port's client. Idempotent."""
        close = getattr(getattr(self, "_inner", None), "aclose", None)
        if callable(close):
            await close()

    def _settings(self, model: str, settings: Mapping[str, Any]) -> dict[str, Any]:
        out, dropped = translate_settings(settings, self._models.get(model))
        for name in dropped:
            if (model, name) not in self._warned:
                self._warned.add((model, name))
                self._emit(
                    logging.WARNING,
                    "llm_call_warning",
                    {
                        "kind": "unsupported",
                        "setting": name,
                        "detail": f"model {model!r} on provider {self._provider!r} has no {name} switch; not sent",
                    },
                )
        return out


class _TableGenerator(_Translating):
    """A `Generator` that translates each request's settings for its model, then delegates."""

    def __init__(
        self, inner: Generator, models: Sequence[ModelInfo], *, provider: str, on_event: EventHook | None
    ) -> None:
        super().__init__(models, provider=provider, on_event=on_event)
        self._inner = inner

    async def generate[T](
        self,
        req: GenRequest,
        output: type[T],
        *,
        validate: Validator[T] | None = None,
        retry_message: RetryMessage | None = None,
    ) -> Generated[T]:
        req = replace(req, settings=self._settings(req.model, req.settings))
        return await self._inner.generate(req, output, validate=validate, retry_message=retry_message)


class _TableChat(_Translating):
    """A `Chat` that translates each request's settings for its model, then delegates."""

    def __init__(self, inner: Chat, models: Sequence[ModelInfo], *, provider: str, on_event: EventHook | None) -> None:
        super().__init__(models, provider=provider, on_event=on_event)
        self._inner = inner

    async def chat[T](
        self,
        req: ChatRequest,
        output: type[T] = str,  # type: ignore[assignment]  # omitted, T is str; no TypeVar default spells it
        *,
        validate: Validator[T] | None = None,
        retry_message: RetryMessage | None = None,
    ) -> Generated[T]:
        req = replace(req, settings=self._settings(req.model, req.settings))
        return await self._inner.chat(req, output, validate=validate, retry_message=retry_message)

    async def stream[T](
        self,
        req: ChatRequest,
        output: type[T] = str,  # type: ignore[assignment]  # omitted, T is str; no TypeVar default spells it
        *,
        validate: Validator[T] | None = None,
        retry_message: RetryMessage | None = None,
    ) -> AsyncGenerator[StreamEvent]:
        req = replace(req, settings=self._settings(req.model, req.settings))
        events = self._inner.stream(req, output, validate=validate, retry_message=retry_message)
        try:
            async for ev in events:
                yield ev
        finally:
            close = getattr(events, "aclose", None)
            if close is not None:
                await close()


def count_url(base_url: str) -> str:
    """The gateway's count route for an OpenAI-wire base URL: `/utils/token_counter` at its root. Pure."""
    root = base_url.rstrip("/")
    root = root.removesuffix("/v1")
    return f"{root}/utils/token_counter"


class OpenAICompatibleFactory:
    """The `openai_compatible` provider kind.

    Args:
        gated: Exception types the generator classifies as `gated` (a refusal raised by the
            caller's transport).
        probe_deadline_s: The probe's deadline, in seconds.

    Attributes:
        kind: `openai_compatible`.
    """

    kind: str = "openai_compatible"

    def __init__(self, *, gated: tuple[type[BaseException], ...] = (), probe_deadline_s: float = 5.0) -> None:
        self._gated = gated
        self._probe_deadline_s = probe_deadline_s

    def build(
        self, spec: ProviderSpec, *, url: str | None, secret: str | None, models: Sequence[ModelInfo], deps: BuildDeps
    ) -> Built:
        """Build the provider's `Generator`, `Embedder` source and `Probe`.

        Raises:
            TableInvalid: The provider names no `base_url`, or sets an option this kind does not
                take.
        """
        path = f"providers.{spec.name}"
        if spec.options:
            raise TableInvalid(f"{path}.{sorted(spec.options)[0]}", "openai_compatible takes no option")
        if url is None:
            raise TableInvalid(f"{path}.base_url", "openai_compatible needs a base_url key")
        if secret is not None and not header_value_ok(secret):
            raise TableInvalid(f"{path}.secret", f"secret {spec.secret!r} {HEADER_VALUE_RULE}")
        transport = deps.transport
        model_for = openai_compatible(url, secret, transport=transport)
        generator = _TableGenerator(
            PydanticAIGenerator(
                model_for=model_for,
                gated=self._gated,
                pricer=deps.pricer,
                on_event=deps.on_event,
                policy=deps.policy,
                limiter=deps.limiter,
            ),
            models,
            provider=spec.name,
            on_event=deps.on_event,
        )

        def embedder_for(model_id: str) -> Embedder:
            model_for = openai_compatible_embeddings(url, model_id, secret, transport=transport)
            return PydanticAIEmbedder(
                model_for=model_for,
                stamp=model_id,
                gated=self._gated,
                pricer=deps.pricer,
                on_event=deps.on_event,
                policy=deps.policy,
            )

        probe = HttpProbe(
            f"{url.rstrip('/')}/models",
            secret=secret,
            deadline_s=self._probe_deadline_s,
            transport=transport if isinstance(transport, httpx.AsyncBaseTransport) else None,
        )
        chat = _TableChat(
            PydanticAIChat(
                model_for=model_for,
                gated=self._gated,
                pricer=deps.pricer,
                on_event=deps.on_event,
                policy=deps.policy,
                limiter=deps.limiter,
            ),
            models,
            provider=spec.name,
            on_event=deps.on_event,
        )
        countable = frozenset(m.id for m in models if "count_tokens" in m.supports)
        counter = HttpTokenCounter(
            count_url(url),
            api_key=secret,
            models=countable,
            transport=transport if isinstance(transport, httpx.AsyncBaseTransport) else None,
        )

        def counter_for(model_id: str) -> TokenCounter | None:
            return counter if model_id in countable else None

        embeds = any("embed" in m.supports for m in models)
        return Built(
            generator=generator,
            chat=chat,
            counter_for=counter_for if countable else None,
            embedder_for=embedder_for if embeds else None,
            probe=probe,
        )


__all__ = ["OpenAICompatibleFactory", "count_url", "translate_settings"]
