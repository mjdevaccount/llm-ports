"""`call.context`: budgets, the fit verdict, trimming, the cache plan, the volatile-system lint (context spec)."""

from __future__ import annotations

from pathlib import Path

from llm_ports import prompts
from llm_ports.call.context import Budget, TrimPolicy, cache_plan, cut_middle, fit, trim, volatile_system
from llm_ports.call.models import ModelInfo
from llm_ports.ports import Counted, Message, Part


def _info(*supports: str, self_hosted: bool = False) -> ModelInfo:
    return ModelInfo(key="m", id="m-1", provider="p", supports=frozenset(supports), self_hosted=self_hosted)  # type: ignore[arg-type]


def test_budget_and_fit_read_unknown_as_unknown() -> None:
    b = Budget(1000, reserve_output=200, margin=0.1)
    assert b.input_limit == 720
    assert Budget(None, reserve_output=0).input_limit is None
    assert fit(Counted(700, "provider"), b).verdict == "fits"
    over = fit(Counted(800, "provider"), b)
    assert over.verdict == "over" and over.over_by == 80
    assert fit(Counted(None, "unsupported"), b).verdict == "unknown"
    assert fit(Counted(5, "provider"), Budget(None, reserve_output=0)).verdict == "unknown"


def test_cut_middle_keeps_head_and_tail() -> None:
    assert cut_middle("short", 10) == ("short", 0)
    text, removed = cut_middle("H" * 50 + "T" * 50, 40)
    assert len(text) <= 40 and text.startswith("H") and text.endswith("T") and "omitted" in text
    assert removed == 100 - (len(text) - len(text[text.index("[") : text.index("]") + 1]))


def _u(t: str) -> Message:
    return Message("user", (Part("text", text=t),))


def _a(t: str) -> Message:
    return Message("assistant", (Part("text", text=t),))


def test_trim_cuts_tool_outputs_first_and_stops_when_that_is_enough() -> None:
    tool = Message("tool", (Part("tool_result", text="r" * 500, call_id="c", name="t"),))
    msgs = [_u("task"), _a("call"), tool, _u("last")]
    got = trim(msgs, TrimPolicy(tool_output_chars=100, keep_last=1), over_by_tokens=50)
    assert got.cut_tool_outputs == 1 and got.dropped_messages == 0 and got.changed
    assert len(got.messages[2].parts[0].text or "") <= 100
    assert got.as_stamp()["chars_removed"] == got.chars_removed


def test_trim_drops_whole_turns_oldest_first_inside_the_keep_bounds() -> None:
    call = Message("assistant", (Part("tool_call", call_id="c", name="t", args={}),))
    res = Message("tool", (Part("tool_result", text="ok", call_id="c", name="t"),))
    msgs = [_u("task"), _u("q1"), call, res, _u("q2"), _a("a2"), _u("last")]
    got = trim(msgs, TrimPolicy(keep_first=1, keep_last=1), over_by_tokens=1)
    assert [m.text for m in got.messages][:2] == ["task", "q2"], "the q1 turn and its tool result went together"
    assert got.dropped_messages == 3
    everything = trim(msgs, TrimPolicy(keep_first=1, keep_last=1), over_by_tokens=None)
    assert [m.text for m in everything.messages] == ["task", "last"]
    untouched = trim([_a("lead"), _u("only")], TrimPolicy(keep_first=0, keep_last=1), over_by_tokens=None)
    assert untouched.dropped_messages == 1 and not trim([_u("x")], TrimPolicy(), over_by_tokens=5).changed


def test_cache_plan_follows_the_stated_capability() -> None:
    assert cache_plan(None, wanted=True, settings={}) == (True, {}, ())
    assert cache_plan(_info("cache"), wanted=False, settings={"a": 1}) == (False, {"a": 1}, ())
    hint, _, warns = cache_plan(_info(), wanted=True, settings={})
    assert hint is False and warns[0].kind == "unsupported" and warns[0].setting == "cache"
    hint, settings, warns = cache_plan(_info("cache", self_hosted=True), wanted=True, settings={"extra_body": {"x": 1}})
    assert hint is False and settings["extra_body"] == {"x": 1, "cache_prompt": True} and not warns
    assert cache_plan(_info("cache"), wanted=True, settings={})[0] is True


def _pack(root: Path, system: str) -> prompts.Pack:
    d = root / "demo.chat"
    d.mkdir(parents=True, exist_ok=True)
    manifest = "id: demo.chat\nversion: 1.0.0\nrole: small\ntemplates:\n  prompt: prompt.j2\n  system: system.j2\n"
    (d / "pack.yaml").write_bytes(manifest.encode())
    (d / "prompt.j2").write_bytes(b"Q: {{ q }}")
    (d / "system.j2").write_bytes(system.encode())
    return prompts.load("demo.chat", root)


def test_volatile_system_flags_a_per_call_value(tmp_path: Path) -> None:
    ctx = [{"q": "a", "now": "1"}, {"q": "b", "now": "2"}]
    assert volatile_system(_pack(tmp_path / "v", "Now is {{ now }}."), ctx)
    assert not volatile_system(_pack(tmp_path / "s", "You are terse."), ctx)
    assert not volatile_system(_pack(tmp_path / "n", "x"), ctx, template="missing")
