"""The kill list: unset means the default, set (even empty) overrides, parsed once, typos named."""

from __future__ import annotations

import logging

from llm_ports.market import ActionRegistry, KillSwitch, parse_names
from tests.market._events import CACHE, Events


class Setting:
    def __init__(self, value: str | None) -> None:
        self.value = value
        self.reads = 0

    def __call__(self) -> str | None:
        self.reads += 1
        return self.value


DEFAULT = {"a.slow", "b.prose"}


def test_unset_falls_back_to_the_default_and_is_read_and_emitted_once() -> None:
    setting, events = Setting(None), Events()
    switch = KillSwitch(setting, default=DEFAULT, on_event=events)
    assert switch.names() == frozenset(DEFAULT) == switch.default
    assert switch.is_disabled("a", "slow") and not switch.is_disabled("c", "kept")
    assert setting.reads == 1
    assert events.seen == [
        (logging.INFO, "market_actions_disabled", {"actions": ["a.slow", "b.prose"], "source": "default"})
    ]


def test_set_overrides_the_default_even_when_empty() -> None:
    events = Events()
    assert KillSwitch(Setting(""), default=DEFAULT, on_event=events).names() == frozenset()
    assert events.seen == [], "an empty paused set says nothing"
    switch = KillSwitch(Setting(" c.one , ,d.two"), default=DEFAULT, on_event=events)
    assert switch.names() == {"c.one", "d.two"}
    assert events.seen[0][2]["source"] == "configured"


def test_reset_re_reads_the_setting() -> None:
    setting = Setting("a.slow")
    switch = KillSwitch(setting, on_event=Events())
    assert switch.is_disabled("a", "slow")
    setting.value = None
    assert switch.is_disabled("a", "slow"), "cached until reset"
    switch.reset()
    assert not switch.is_disabled("a", "slow")
    assert setting.reads == 2


def test_unknown_names_a_paused_name_no_action_carries() -> None:
    async def h(_session: object, key: str) -> None: ...

    reg = ActionRegistry("main")
    reg.action(feature="a", action="slow", desired_ttl="6h", **CACHE)(h)
    events = Events()
    switch = KillSwitch(Setting("a.slow,a.solw"), on_event=events)
    assert switch.unknown(reg) == {"a.solw"}
    assert events.seen[-1] == (logging.WARNING, "market_kill_unknown", {"registry": "main", "names": ["a.solw"]})
    events.seen.clear()
    assert KillSwitch(Setting("a.slow"), on_event=events).unknown(reg) == frozenset()
    assert [e[1] for e in events.seen] == ["market_actions_disabled"]


def test_parse_names_drops_blanks() -> None:
    assert parse_names("x.y,, z.w ,") == {"x.y", "z.w"}
    assert parse_names("") == frozenset()
