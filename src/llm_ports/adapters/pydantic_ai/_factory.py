"""The `openai_compatible` provider kind: a schema 2 provider built on pydantic-ai.

`OpenAICompatibleFactory` is the built-in `call.providers.ProviderFactory` for any OpenAI-wire
endpoint (a LiteLLM gateway, vLLM, llama.cpp's server, OpenAI). For one provider it builds:

- a `Generator` (`PydanticAIGenerator` over `openai_compatible`) that spells each request's
  provider-neutral settings the way the request's model needs (`translate_settings`);
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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from llm_ports._extras import extra
from llm_ports.call.models import ModelInfo, ProviderSpec, TableInvalid
from llm_ports.call.providers import BuildDeps, Built
from llm_ports.observability import EventHook, log_events
from llm_ports.ports import Embedder, Generated, Generator, GenRequest

with extra("pydantic-ai"):
    import httpx

    from llm_ports.adapters._probe import HttpProbe
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


class _TableGenerator:
    """A `Generator` that translates each request's settings for its model, then delegates."""

    __slots__ = ("_emit", "_inner", "_models", "_provider", "_warned")

    def __init__(
        self, inner: Generator, models: Sequence[ModelInfo], *, provider: str, on_event: EventHook | None
    ) -> None:
        self._inner = inner
        self._models = {m.id: m for m in models}
        self._provider = provider
        self._emit = on_event or _log_event
        self._warned: set[tuple[str, str]] = set()

    async def generate[T](
        self,
        req: GenRequest,
        output: type[T],
        *,
        validate: Callable[[T], Sequence[Any]] | None = None,
        retry_message: Callable[[Sequence[Any]], str] | None = None,
    ) -> Generated[T]:
        settings, dropped = translate_settings(req.settings, self._models.get(req.model))
        for name in dropped:
            if (req.model, name) not in self._warned:
                self._warned.add((req.model, name))
                self._emit(
                    logging.WARNING,
                    "llm_call_warning",
                    {
                        "kind": "unsupported",
                        "setting": name,
                        "detail": f"model {req.model!r} on provider {self._provider!r} has no {name} switch; not sent",
                    },
                )
        return await self._inner.generate(
            replace(req, settings=settings), output, validate=validate, retry_message=retry_message
        )


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
        transport = deps.transport
        generator = _TableGenerator(
            PydanticAIGenerator(model_for=openai_compatible(url, secret, transport=transport), gated=self._gated),
            models,
            provider=spec.name,
            on_event=deps.on_event,
        )

        def embedder_for(model_id: str) -> Embedder:
            model_for = openai_compatible_embeddings(url, model_id, secret, transport=transport)
            return PydanticAIEmbedder(model_for=model_for, stamp=model_id, gated=self._gated)

        probe = HttpProbe(
            f"{url.rstrip('/')}/models",
            secret=secret,
            deadline_s=self._probe_deadline_s,
            transport=transport if isinstance(transport, httpx.AsyncBaseTransport) else None,
        )
        embeds = any("embed" in m.supports for m in models)
        return Built(generator=generator, embedder_for=embedder_for if embeds else None, probe=probe)


__all__ = ["OpenAICompatibleFactory", "translate_settings"]
