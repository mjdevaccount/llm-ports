"""The `openai_compatible` provider kind: thinking translated per `ThinkingStyle` (spec test 4), and `build`."""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar

import httpx
import pytest

from llm_ports import call
from llm_ports.adapters.pluggy import builtin, load_providers
from llm_ports.adapters.pydantic_ai import OpenAICompatibleFactory, translate_settings
from llm_ports.call import roles
from llm_ports.call.models import ModelInfo, ProviderSpec, TableInvalid
from llm_ports.call.providers import BuildDeps, build
from llm_ports.ports import Embedder, GenRequest, Probe
from tests.call.test_table_v2 import V2
from tests.market._events import Events

URLS = {"gateway": "http://gw.test/v1", "local": "http://box.test/v1"}
SECRETS = {"gateway_key": "sk-test-value"}


def _info(thinking: str) -> ModelInfo:
    return ModelInfo(key="m", id="m-1", provider="p", supports=frozenset(), thinking=thinking)  # type: ignore[arg-type]


# -- spec test 4: one translation per ThinkingStyle ---------------------------------------------
def test_4_unified_keeps_the_frameworks_own_thinking_setting() -> None:
    assert translate_settings({"thinking": True, "seed": 1}, _info("unified")) == ({"thinking": True, "seed": 1}, ())


def test_4_chat_template_moves_it_into_the_chat_template_kwargs() -> None:
    settings = {"thinking": False, "extra_body": {"top_k": 20, "chat_template_kwargs": {"other": 1}}}
    out, dropped = translate_settings(settings, _info("chat_template"))
    assert out == {"extra_body": {"top_k": 20, "chat_template_kwargs": {"other": 1, "enable_thinking": False}}}
    assert dropped == () and settings["thinking"] is False, "the request's own settings are not mutated"


@pytest.mark.parametrize(("want", "dropped"), [(True, ("thinking",)), (False, ())])
def test_4_none_drops_it_and_names_it_only_when_the_role_asked_to_think(want: bool, dropped: tuple[str, ...]) -> None:
    assert translate_settings({"thinking": want}, _info("none")) == ({}, dropped)


def test_unknown_models_and_settings_without_thinking_pass_through() -> None:
    assert translate_settings({"thinking": True}, None) == ({"thinking": True}, ())
    assert translate_settings({"seed": 3}, _info("chat_template")) == ({"seed": 3}, ())


# -- the built provider, over a mock wire -------------------------------------------------------
def _completion(content: str) -> dict[str, Any]:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "small-chat",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


class Wire:
    """A mock OpenAI server: records each body; `/models` lists one model."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.paths: list[str] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.paths.append(req.url.path)
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "small-chat"}]})
        self.bodies.append(json.loads(req.content))
        return httpx.Response(200, json=_completion("fine"))


class V2Role(roles.Role):
    TIERS: ClassVar = V2.tiers


def _backends(wire: Wire, events: Events) -> Any:
    providers = load_providers(builtins=[builtin("core", OpenAICompatibleFactory())], entry_points=(), on_event=events)
    deps = BuildDeps(transport=httpx.MockTransport(wire), on_event=events)
    return build(V2, urls=URLS, secrets=SECRETS.get, deps=deps, factories=providers)


async def test_the_thinking_switch_reaches_the_wire_in_each_models_spelling() -> None:
    wire, events = Wire(), Events()
    backends = _backends(wire, events)
    chat = V2Role("t.chat", tier="local", thinking=True)
    res = await call.generate(chat, prompt="p", generator=backends.generator_for("local"))
    assert res.outcome == "ok", res.reason
    assert wire.bodies[-1]["chat_template_kwargs"] == {"enable_thinking": True}
    assert "thinking" not in json.dumps(wire.bodies[-1]).replace("enable_thinking", "")
    assert backends.loaded.origin == {"core": "builtin"}


async def test_4_asking_a_model_with_no_switch_to_think_warns_once_and_sends_without_it() -> None:
    wire, events = Wire(), Events()
    backends = _backends(wire, events)
    web = V2Role("t.web", tier="local-web", thinking=True)
    for _ in range(2):
        res = await call.generate(web, prompt="p", generator=backends.generator_for("local-web"))
        assert res.outcome == "ok"
    warned = [f for lvl, e, f in events.seen if e == "llm_call_warning" and lvl == logging.WARNING]
    assert len(warned) == 1 and (warned[0]["kind"], warned[0]["setting"]) == ("unsupported", "thinking")
    assert "small-web" in warned[0]["detail"] and "'local'" in warned[0]["detail"]
    assert len(wire.bodies) == 2 and all("chat_template_kwargs" not in b for b in wire.bodies)


async def test_the_factory_builds_a_probe_on_the_models_endpoint_and_an_embedder_when_stated() -> None:
    wire, events = Wire(), Events()
    backends = _backends(wire, events)
    probe = backends.probe_for("local")
    assert isinstance(probe, Probe)
    health = await probe.probe()
    assert health.state == "up" and health.models == ("small-chat",)
    assert wire.paths[-1] == "/v1/models"
    assert isinstance(backends.embedder_for("embed"), Embedder)
    assert backends.providers["gateway"].embedder_for is None, "no model of the gateway states embed"


@pytest.mark.parametrize(
    ("spec", "url", "path"),
    [
        (
            ProviderSpec("p", "openai_compatible", "k", None, options={"verify": False}),
            "http://x",
            "providers.p.verify",
        ),
        (ProviderSpec("p", "openai_compatible", None, None), None, "providers.p.base_url"),
    ],
)
def test_the_factory_refuses_an_option_or_a_missing_url(spec: ProviderSpec, url: str | None, path: str) -> None:
    with pytest.raises(TableInvalid) as caught:
        OpenAICompatibleFactory().build(spec, url=url, secret=None, models=(), deps=BuildDeps())
    assert caught.value.path == path


def test_the_factory_called_directly_refuses_a_secret_no_header_can_carry() -> None:
    spec = ProviderSpec("p", "openai_compatible", "k", "vault_key")
    with pytest.raises(TableInvalid) as caught:
        OpenAICompatibleFactory().build(spec, url="http://x", secret="sk-SECRETVALUE\n", models=(), deps=BuildDeps())
    assert caught.value.path == "providers.p.secret" and "'vault_key'" in str(caught.value)
    assert "sk-SECRETVALUE" not in str(caught.value)


async def test_a_request_for_a_model_the_table_does_not_know_passes_through() -> None:
    wire = Wire()
    spec = ProviderSpec("p", "openai_compatible", "k", None)
    built = OpenAICompatibleFactory().build(
        spec, url="http://gw.test/v1", secret=None, models=(), deps=BuildDeps(transport=httpx.MockTransport(wire))
    )
    assert built.generator is not None and built.embedder_for is None
    res = await built.generator.generate(GenRequest(model="other", prompt="p", settings={"seed": 1}), str)
    assert res.outcome == "ok" and wire.bodies[-1]["seed"] == 1
    assert OpenAICompatibleFactory.kind == "openai_compatible"
