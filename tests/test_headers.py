"""`llm_ports._headers`: one rule for a secret sent as a header, enforced at every entry that sends one.

The class of defect: a secret read from a file keeps its trailing newline, the HTTP stack refuses
the header when the request is sent, and its refusal (exception message and httpcore's DEBUG
trace) quotes the header, value and all. Each entry that puts a caller's secret in a header checks
it with the same validator, before any request, and never shows the value.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import pytest

from llm_ports._headers import header_value_ok
from llm_ports.adapters.http import HttpProbe
from llm_ports.adapters.pydantic_ai import OpenAICompatibleFactory, openai_compatible, openai_compatible_embeddings
from llm_ports.call.models import ProviderSpec
from llm_ports.call.providers import BuildDeps, build
from tests.adapters.provider_plugins.kinds import Kind
from tests.call.test_providers import URLS
from tests.call.test_table_v2 import V2

SECRET = "sk-SECRETVALUE123"
BAD_TAILS = ["\n", "\r\n", " ", "\t", "\x00", "\x7f", "é"]


@pytest.mark.parametrize("tail", BAD_TAILS)
def test_a_value_with_whitespace_a_control_or_non_ascii_is_refused(tail: str) -> None:
    assert not header_value_ok(SECRET + tail)


def test_visible_ascii_is_accepted() -> None:
    assert header_value_ok(SECRET) and header_value_ok("") and header_value_ok("a!~Z0-._")


def _build(secret: str) -> object:
    kinds = {"openai_compatible": Kind("openai_compatible")}
    return build(V2, urls=URLS, secrets={"gateway_key": secret}.get, deps=BuildDeps(), factories=kinds)


def _factory(secret: str) -> object:
    spec = ProviderSpec("p", "openai_compatible", "k", "vault_key")
    return OpenAICompatibleFactory().build(spec, url="http://x", secret=secret, models=(), deps=BuildDeps())


ENTRIES: dict[str, Callable[[str], object]] = {
    "build": _build,
    "factory": _factory,
    "openai_compatible": lambda s: openai_compatible("http://127.0.0.1:9/v1", s),
    "openai_compatible_embeddings": lambda s: openai_compatible_embeddings("http://127.0.0.1:9/v1", "m", s),
}


@pytest.mark.parametrize("tail", ["\n", "\r\n", "é"])
@pytest.mark.parametrize("entry", sorted(ENTRIES))
def test_every_entry_that_sends_a_secret_refuses_it_before_any_request_and_never_shows_it(
    entry: str, tail: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG), pytest.raises(ValueError) as caught:
        ENTRIES[entry](SECRET + tail)
    assert SECRET not in str(caught.value) and SECRET not in repr(caught.value.args)
    assert SECRET not in caplog.text
    assert "whitespace, a control or a non-ASCII character" in str(caught.value), "the one shared rule"


@pytest.mark.parametrize("entry", sorted(ENTRIES))
def test_every_entry_accepts_a_clean_secret(entry: str) -> None:
    ENTRIES[entry](SECRET)  # must fire both ways: the refusal is the value, not the entry


async def test_the_probe_reports_it_unknown_without_a_request() -> None:
    health = await HttpProbe("http://127.0.0.1:9/v1/models", secret=SECRET + "\n").probe()
    assert health.state == "unknown" and SECRET not in repr(health)
