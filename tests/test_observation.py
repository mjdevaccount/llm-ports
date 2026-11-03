"""observation -- one generation, ended once; unmeasured usage is never zero tokens."""

from __future__ import annotations

from typing import Any

import pytest

from llm_ports.observation import PortObservation, observe, opt_int
from llm_ports.testing import RecordingTracer


class _Sink:
    def __init__(self) -> None:
        self.ends: list[dict[str, Any]] = []
        self.scores: list[tuple] = []

    def end(self, **kw: Any) -> None:
        self.ends.append(kw)

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        self.scores.append((name, value, comment))


@pytest.mark.parametrize(("v", "want"), [(None, None), (0, 0), ("7", 7), (3.9, 3), ("x", None), ({}, None)])
def test_opt_int_keeps_zero_and_turns_nothing_into_none(v: Any, want: int | None) -> None:
    assert opt_int(v) == want


def test_measured_counts_flush_once_as_input_output() -> None:
    s = _Sink()
    o = PortObservation(s)
    o.update(output="hi", usage_details={"input": 12, "output": "3"}, metadata={"k": "v"})
    o.flush()
    assert s.ends == [{"output": "hi", "usage": {"input": 12, "output": 3}, "metadata": {"k": "v"}}]
    assert o.usage_counts() == (12, 3)


def test_a_measured_zero_is_a_zero_not_unmeasured() -> None:
    s = _Sink()
    o = PortObservation(s)
    o.update(usage_details={"input": 0, "output": 0})
    o.flush()
    assert s.ends[0]["usage"] == {"input": 0, "output": 0} and s.ends[0]["metadata"] is None


def test_neither_count_reported_omits_usage_and_flags_tokens_unmeasured() -> None:
    # MUST FIRE: drop the flag (or book the pair as zeros) and an unmetered call reads as free.
    s = _Sink()
    o = PortObservation(s)
    o.update(output="o", usage_details={"input": None, "output": None}, metadata={"latency_ms": 12})
    o.flush()
    assert s.ends == [{"output": "o", "usage": None,
                       "metadata": {"latency_ms": 12, "tokens_unmeasured": True}}]
    assert o.usage_counts() == (None, None)


def test_no_update_at_all_is_an_end_with_nothing_claimed() -> None:
    s = _Sink()
    PortObservation(s).flush()
    assert s.ends == [{"output": None, "usage": None, "metadata": None}]


def test_fail_ends_at_error_level_with_the_capped_error_string() -> None:
    s = _Sink()
    PortObservation(s).fail(ValueError("x" * 600), {"feature": "f"})
    (end,) = s.ends
    assert end["level"] == "ERROR" and end["metadata"] == {"feature": "f"}
    assert end["status_message"].startswith("ValueError: x") and len(end["status_message"]) == 500


def test_score_is_a_float_on_the_generation() -> None:
    s = _Sink()
    PortObservation(s).score(name="claims_kept", value=3, comment="c")
    assert s.scores == [("claims_kept", 3.0, "c")]


async def test_observe_opens_one_generation_and_ends_it_once() -> None:
    t = RecordingTracer()
    async with observe(t, name="cat", model="m", provider="p", input="hi", metadata={"a": 1},
                       session_id="s", prompt_name="pk") as obs:
        obs.update(output="ok", usage_details={"input": 1, "output": 2})
    (g,) = t.generations
    assert g.opened == {"name": "cat", "model": "m", "provider": "p", "input": "hi", "metadata": {"a": 1},
                        "session_id": "s", "prompt_name": "pk"}
    assert g.ended == {"output": "ok", "usage": {"input": 1, "output": 2}, "level": "DEFAULT",
                       "status_message": None, "metadata": None}


async def test_observe_ends_an_exception_at_error_with_the_latency_and_lets_it_propagate() -> None:
    t = RecordingTracer()
    now = iter([10.0, 10.25])
    with pytest.raises(RuntimeError, match="upstream 500"):
        async with observe(t, name="c", model="m", provider="p", input="", metadata={"a": 1},
                           clock=lambda: next(now)) as obs:
            obs.update(metadata={"outcome": "unavailable"})
            raise RuntimeError("upstream 500")
    (g,) = t.generations
    assert g.ended["level"] == "ERROR" and g.ended["status_message"] == "RuntimeError: upstream 500"
    assert g.ended["metadata"] == {"a": 1, "outcome": "unavailable", "latency_ms": 250}
