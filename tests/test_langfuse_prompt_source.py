"""LangfusePromptSource -- the repo leads, the label mirrors: the seed is pushed when the label is
missing or stale, left alone when current, and SERVED when the store is down, refuses the push, or
holds text that is not the seed's. Against `testing_langfuse.FakeLangfusePrompts`."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from llm_ports import langfuse_prompt_source as lp
from llm_ports import ports
from llm_ports.ports import Prompt, Seed
from llm_ports.testing_langfuse import FakeLangfusePrompts as FakeClient
from llm_ports.testing_langfuse import RefusingLangfuse

TEXT = "You are {{ who }}.\n{% if what %}Say {{ what }}.{% endif %}\n"
SHA = "a1b2c3d4e5f60718"
SEED = Seed(TEXT, SHA, {"pack_version": "1.0.0", "model": "cheap"}, tags=["demo"])
# The repo seed SERVED (store down, refusing, or disagreeing): its text, config and system ride along
# (a caller reads its knobs from `config`), version None, source "repo".
REPO = Prompt("demo.note", TEXT, None, "repo", config=dict(SEED.config))

# A CHAT seed: a system template beside the user text.
SYSTEM = "You classify {{ what }}.\n"
CHAT_SHA = "0f1e2d3c4b5a6978"
CHAT_SEED = Seed(TEXT, CHAT_SHA, {"model": "small", "version": 2}, tags=["demo"], system=SYSTEM)
CHAT_WIRE = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TEXT}]
CHAT_REPO = Prompt("demo.chat", TEXT, None, "repo", config=dict(CHAT_SEED.config), system=SYSTEM)


def seed_for(name: str) -> Seed:
    assert name == "demo.note"
    return SEED


def chat_seed_for(name: str) -> Seed:
    assert name == "demo.chat"
    return CHAT_SEED


def _stored(text: str, sha: str, version: int = 1) -> SimpleNamespace:
    return SimpleNamespace(is_fallback=False, prompt=text, config={"content_sha": sha}, version=version)


def test_an_absent_label_is_seeded_then_served_as_langfuse_v1() -> None:
    c = FakeClient()
    p = lp.LangfusePromptSource(c, seed_for).get("demo.note")
    assert p == Prompt("demo.note", TEXT, 1, "langfuse:v1",
                       config={"pack_version": "1.0.0", "model": "cheap", "content_sha": SHA})
    kw = c.created[0]
    assert kw["name"] == "demo.note" and kw["type"] == "text" and kw["labels"] == ["production"]
    assert kw["tags"] == ["demo"] and kw["config"]["content_sha"] == SHA and kw["prompt"] == TEXT
    # cached first fetch under the SDK ttl, then an UNCACHED refetch after the push
    assert [f[3] for f in c.fetches] == [lp.DEFAULT_CACHE_TTL_S, 0]
    assert all(f[1:3] == ("production", "text") for f in c.fetches)


def test_a_stale_mirror_is_overwritten_the_repo_wins() -> None:
    c = FakeClient(_stored("an edit made in the UI", "deadbeef", version=4))
    p = lp.LangfusePromptSource(c, seed_for).get("demo.note")
    assert p.text == TEXT and p.version == 5 and p.source == "langfuse:v5" and len(c.created) == 1


def test_a_current_mirror_is_left_alone() -> None:
    c = FakeClient(_stored(TEXT, SHA, version=7))
    p = lp.LangfusePromptSource(c, seed_for, cache_ttl_s=60).get("demo.note")
    assert p == Prompt("demo.note", TEXT, 7, "langfuse:v7", config={"content_sha": SHA})
    assert not c.created and c.fetches == [("demo.note", "production", "text", 60)]


def test_the_label_asked_for_is_the_one_fetched_and_pushed() -> None:
    c = FakeClient()
    lp.LangfusePromptSource(c, seed_for).get("demo.note", label="staging")
    assert c.created[0]["labels"] == ["staging"] and c.fetches[0][1] == "staging"


def test_a_refused_push_serves_the_repo_seed_not_a_raise() -> None:
    class Refusing(FakeClient):
        def create_prompt(self, **kw):
            raise RuntimeError("403 project is read-only")

    p = lp.LangfusePromptSource(Refusing(), seed_for).get("demo.note")
    assert p == REPO


def test_langfuse_down_serves_the_repo_seed_not_a_raise() -> None:
    p = lp.LangfusePromptSource(RefusingLangfuse(), seed_for).get("demo.note")
    assert p == REPO


def test_a_down_verdict_is_held_for_one_ttl_then_retried() -> None:
    """The SDK never caches a `fallback` reply, so the adapter holds the down-verdict itself: inside
    the ttl no network call, past it one retry -- and a success clears the hold."""
    class Flaky:
        def __init__(self) -> None:
            self.calls = 0
            self.up = False

        def get_prompt(self, name, *, label, type, fallback, cache_ttl_seconds):
            self.calls += 1
            if not self.up:
                raise ConnectionError("down")
            return _stored(TEXT, SEED.sha)

        def create_prompt(self, **kw):
            raise AssertionError("never pushed in this test")

    now = [1000.0]
    client = Flaky()
    src = lp.LangfusePromptSource(client, seed_for, cache_ttl_s=300, clock=lambda: now[0])
    assert src.get("demo.note").source == "repo" and client.calls == 1
    now[0] += 299
    assert src.get("demo.note").source == "repo" and client.calls == 1   # held: no round-trip
    now[0] += 2
    client.up = True
    assert src.get("demo.note").source == "langfuse:v1" and client.calls == 2
    assert src.get("demo.note").source == "langfuse:v1" and client.calls == 3  # hold cleared, SDK caches now


def test_a_mirror_whose_text_disagrees_after_the_push_is_not_trusted() -> None:
    """The sha said current but the text is not the seed's (a hand edit that kept the config):
    the repo text serves, as `repo`, so the trace link is honest about what the model saw."""
    c = FakeClient(_stored("not the seed", SHA, version=2))
    p = lp.LangfusePromptSource(c, seed_for).get("demo.note")
    assert p == REPO and not c.created


# ── chat seeds: system + user ride ONE Langfuse chat prompt under the seed's name ─────────────────


def test_a_chat_seed_is_pushed_and_fetched_as_a_two_message_chat_prompt() -> None:
    c = FakeClient()
    p = lp.LangfusePromptSource(c, chat_seed_for).get("demo.chat")
    assert p == Prompt("demo.chat", TEXT, 1, "langfuse:v1",
                       config={"model": "small", "version": 2, "content_sha": CHAT_SHA}, system=SYSTEM)
    kw = c.created[0]
    assert kw["type"] == "chat" and kw["prompt"] == CHAT_WIRE and kw["tags"] == ["demo"]
    assert all(f[2] == "chat" for f in c.fetches), c.fetches


def test_a_system_less_seed_forced_to_chat_is_a_one_message_chat_prompt() -> None:
    """Langfuse refuses a type change on an existing name, so a name once pushed as chat stays chat
    even for a seed with no system message (`kind="chat"`)."""
    c = FakeClient()
    seed = Seed(TEXT, "sha-nosys", {"model": "m"}, tags=["demo"], kind="chat")
    p = lp.LangfusePromptSource(c, lambda n: seed).get("demo.nosys")
    kw = c.created[0]
    assert kw["type"] == "chat" and kw["prompt"] == [{"role": "user", "content": TEXT}]
    assert all(f[2] == "chat" for f in c.fetches), c.fetches
    assert p.source == "langfuse:v1" and p.system == "" and p.text == TEXT


def test_a_chat_copy_pushed_without_the_system_message_is_re_pushed() -> None:
    """A one-message chat with no `content_sha` (pushed by some other tool, without the system
    message): the sha mismatch re-pushes the seed; no migration step."""
    old = SimpleNamespace(is_fallback=False, prompt=[{"role": "user", "content": TEXT}],
                          config={"model": "small"}, version=3)
    c = FakeClient(old)
    p = lp.LangfusePromptSource(c, chat_seed_for).get("demo.chat")
    assert p.source == "langfuse:v4" and p.system == SYSTEM and c.created[0]["prompt"] == CHAT_WIRE


@pytest.mark.parametrize("messages", [
    [{"role": "system", "content": "edited in the UI\n"}, {"role": "user", "content": TEXT}],
    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "edited in the UI\n"}],
    [{"role": "system", "content": SYSTEM}],
], ids=["system_differs", "user_differs", "user_missing"])
def test_a_chat_copy_whose_sha_is_current_but_text_differs_in_either_message_serves_repo(messages) -> None:
    c = FakeClient(SimpleNamespace(is_fallback=False, prompt=messages, config={"content_sha": CHAT_SHA}, version=2))
    p = lp.LangfusePromptSource(c, chat_seed_for).get("demo.chat")
    assert p == CHAT_REPO and not c.created


def test_a_current_chat_copy_is_left_alone_and_compared_on_role_and_content_only() -> None:
    extra = [{**m, "type": "chatmessage"} for m in CHAT_WIRE]       # SDK-side extra keys are not drift
    c = FakeClient(SimpleNamespace(is_fallback=False, prompt=extra, config={"content_sha": CHAT_SHA}, version=5))
    p = lp.LangfusePromptSource(c, chat_seed_for).get("demo.chat")
    assert p.source == "langfuse:v5" and p.text == TEXT and p.system == SYSTEM and not c.created


def test_a_malformed_chat_reply_serves_repo_not_a_raise() -> None:
    c = FakeClient(SimpleNamespace(is_fallback=False, prompt=["not a message"], config={"content_sha": CHAT_SHA}, version=2))
    assert lp.LangfusePromptSource(c, chat_seed_for).get("demo.chat") == CHAT_REPO


def test_the_source_satisfies_the_port() -> None:
    assert isinstance(lp.LangfusePromptSource(FakeClient(), seed_for), ports.PromptSource)


def test_a_failed_mirror_is_one_warning_event_naming_the_prompt() -> None:
    events: list = []
    lp.LangfusePromptSource(RefusingLangfuse(), seed_for, on_event=lambda *e: events.append(e)).get("demo.note")
    ((lvl, ev, f),) = events
    assert (lvl, ev, f["action"], f["name"], f["label"]) == (
        logging.WARNING, lp.EVENT, "mirror_failed", "demo.note", "production")
    assert f["error"].startswith("ConnectionError")
