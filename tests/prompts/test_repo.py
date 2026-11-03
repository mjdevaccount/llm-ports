"""RepoPromptSource and the stub source keep the PromptSource port's laws; the law test FIRES on a liar."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from llm_ports import ports
from llm_ports.ports import Prompt, Seed
from llm_ports.prompts import RepoPromptSource
from llm_ports.testing import StubPromptSource


def _prompt_laws(s: ports.PromptSource, name: str) -> Prompt:
    assert isinstance(s, ports.PromptSource)
    p = s.get(name)
    assert isinstance(p, Prompt) and p.name == name and p.text
    assert (p.version is None) == (p.source == "repo"), "version None <=> served from the repo seed"
    assert isinstance(p.config, Mapping)
    return p


def test_the_repo_and_stub_sources_keep_the_prompt_source_laws() -> None:
    _prompt_laws(StubPromptSource({"demo": "You are {{ who }}."}), "demo")
    p = _prompt_laws(
        RepoPromptSource(lambda n: Seed("You are {{ who }}.", "0123456789abcdef", {"model": "m"}, system="sys")), "demo"
    )
    assert p.system == "sys" and p.config == {"model": "m"}


def test_the_prompt_source_laws_fire_on_a_versioned_repo_prompt() -> None:
    class Lying:
        def get(self, name: str, *, label: str = "production") -> Prompt:
            return Prompt(name, "t", 3, "repo")

    with pytest.raises(AssertionError, match="version None"):
        _prompt_laws(Lying(), "demo")


def test_a_name_without_a_seed_raises_never_invents_text() -> None:
    with pytest.raises(KeyError):
        StubPromptSource({}).get("missing")

    def no_seed(name: str) -> Seed:
        raise KeyError(name)

    with pytest.raises(KeyError):
        RepoPromptSource(no_seed).get("missing")


def test_the_stub_source_records_the_label_asked_for() -> None:
    assert StubPromptSource({"d": "t"}).get("d", label="staging").config == {"label": "staging"}
