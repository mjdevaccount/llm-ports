"""`Pack.seed()` through `LangfusePromptSource` (acceptance test 8): the repo pack leads, the mirror is
pushed on a content-sha OR identity mismatch, the repo seed serves on an outage, and the label comes
from the manifest when the source opts in (`label_from_seed=True`); the 1.x default is unchanged."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_ports import prompts
from llm_ports.adapters.langfuse import prompt_source as lp
from llm_ports.ports import Seed
from llm_ports.testing.langfuse import FakeLangfusePrompts, RefusingLangfuse

MANIFEST = "id: demo.note\nversion: 1.0.0\nrole: small\nlabel: staging\ntemplates:\n  prompt: prompt.j2\n"


def _pack(root: Path, manifest: str = MANIFEST) -> prompts.Pack:
    d = root / "demo.note"
    d.mkdir(parents=True, exist_ok=True)
    (d / "pack.yaml").write_bytes(manifest.encode())
    (d / "prompt.j2").write_bytes(b"You are {{ who }}.")
    return prompts.load("demo.note", root)


@pytest.fixture
def pack(tmp_path: Path) -> prompts.Pack:
    return _pack(tmp_path)


def test_the_seed_carries_both_names_and_the_manifest_label(pack: prompts.Pack) -> None:
    s = pack.seed()
    assert (s.text, s.sha, s.system) == (pack.source, pack.content_sha, "")
    assert s.config == {
        "pack": "demo.note",
        "version": "1.0.0",
        "role": "small",
        "template": "prompt",
        "content_sha": pack.content_sha,
        "identity": pack.identity,
        "label": "staging",
    }
    with pytest.raises(KeyError):
        pack.seed("absent")


def test_an_absent_mirror_is_pushed_under_the_manifest_label_when_opted_in(pack: prompts.Pack) -> None:
    client = FakeLangfusePrompts()
    p = lp.LangfusePromptSource(client, lambda _n: pack.seed(), label_from_seed=True).get("demo.note")
    assert (p.source, p.version, p.text) == ("langfuse:v1", 1, pack.source)
    assert client.created[0]["labels"] == ["staging"] and {f[1] for f in client.fetches} == {"staging"}
    assert client.created[0]["config"]["identity"] == pack.identity
    assert client.created[0]["config"]["content_sha"] == pack.content_sha


def test_the_seed_label_is_opt_in_and_the_default_is_unchanged(pack: prompts.Pack) -> None:
    """1.x keeps `get`'s default: without the opt-in, the manifest label is not read."""
    client = FakeLangfusePrompts()
    lp.LangfusePromptSource(client, lambda _n: pack.seed()).get("demo.note")
    assert client.created[0]["labels"] == ["production"]
    canary = FakeLangfusePrompts()
    lp.LangfusePromptSource(canary, lambda _n: pack.seed()).get("demo.note", label="canary")
    assert canary.created[0]["labels"] == ["canary"]
    bare = FakeLangfusePrompts()
    lp.LangfusePromptSource(bare, lambda _n: Seed("x", "0" * 16), label_from_seed=True).get("demo.note", label="qa")
    assert bare.created[0]["labels"] == ["qa"], "a seed naming no label keeps the caller's"


def test_a_stale_identity_under_the_same_content_sha_is_pushed(tmp_path: Path) -> None:
    """A version bump leaves the text, so `content_sha`, alone; the mirror still follows the repo."""
    old = _pack(tmp_path / "old")
    new = _pack(tmp_path / "new", MANIFEST.replace("1.0.0", "1.1.0"))
    assert old.content_sha == new.content_sha and old.identity != new.identity
    client = FakeLangfusePrompts()
    lp.LangfusePromptSource(client, lambda _n: old.seed()).get("demo.note")
    lp.LangfusePromptSource(client, lambda _n: old.seed()).get("demo.note")
    assert len(client.created) == 1, "a current mirror is left alone"
    p = lp.LangfusePromptSource(client, lambda _n: new.seed()).get("demo.note")
    assert len(client.created) == 2 and p.config["identity"] == new.identity and p.source == "langfuse:v2"


def test_the_repo_seed_serves_when_the_store_is_down(pack: prompts.Pack) -> None:
    p = lp.LangfusePromptSource(RefusingLangfuse(), lambda _n: pack.seed(), label_from_seed=True).get("demo.note")
    assert (p.source, p.version, p.text) == ("repo", None, pack.source)
    assert p.config["identity"] == pack.identity and p.config["label"] == "staging"
