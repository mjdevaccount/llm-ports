"""lettucedetect_grounder: the verdict SHAPE over `testing.StubDetector` (no model, no torch), and
the missing-dependency path. The real model is the consumer's lane (it installs the extra)."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from llm_ports.adapters import lettucedetect as lg
from llm_ports.grounding import NumberLawGrounder
from llm_ports.ports import Chunk, Claim, Grounder
from llm_ports.testing import StubDetector

SPAN = Chunk(id="a", text="SPY closed at 580.5", title="Close")


def test_supported_when_no_span_is_flagged() -> None:
    fake = StubDetector()
    v = lg.LettuceDetectGrounder(fake).verdict(Claim("SPY at 580.5", ("a",)), [SPAN])
    assert v.supported is True and v.score == 1.0 and v.checker == "lettucedetect"
    assert fake.calls[0] == {
        "context": ["Close: SPY closed at 580.5"],
        "answer": "SPY at 580.5",
        "output_format": "spans",
    }


def test_title_is_part_of_the_premise_like_the_number_law() -> None:
    """`numbers.uncited` counts a digit in a span's TITLE as cited, so the model must read the
    title too or two Grounders disagree on the same Chunk."""
    chunk = Chunk(id="t", title="SPY 580 close", text="SPY closed flat")
    claim = Claim("SPY at 580", ("t",))
    assert NumberLawGrounder().verdict(claim, [chunk]).supported is True
    assert lg.LettuceDetectGrounder(StubDetector()).verdict(claim, [chunk]).supported is True
    fake = StubDetector()
    lg.LettuceDetectGrounder(fake).verdict(Claim("x", ("u",)), [Chunk(id="u", text="no title here")])
    assert fake.calls[0]["context"] == ["no title here"], "an empty title adds no prefix"


def test_a_flagged_span_without_confidence_has_no_score_not_a_perfect_one() -> None:
    """Unknown is never zero: worst=0.0 would score 1.0, the 'fully supported' end, on an
    UNSUPPORTED verdict."""

    class NoConfidence:
        def predict(self, *, context: list[str], answer: str, output_format: str) -> list[dict[str, Any]]:
            return [{"text": "581", "start": 7, "end": 10}]

    v = lg.LettuceDetectGrounder(NoConfidence()).verdict(Claim("SPY at 581", ("a",)), [SPAN])
    assert v.supported is False and v.score is None and "581" in v.reason


def test_unsupported_names_the_span_and_scores_one_minus_worst() -> None:
    v = lg.LettuceDetectGrounder(StubDetector()).verdict(Claim("SPY at 581", ("a",)), [SPAN])
    assert v.supported is False and "581" in v.reason and v.score == pytest.approx(0.1)


def test_no_spans_is_never_supported_and_never_calls_the_model() -> None:
    fake = StubDetector()
    v = lg.LettuceDetectGrounder(fake).verdict(Claim("SPY at 580.5"), [])
    assert v.supported is False and fake.calls == []


def test_context_over_window_is_named_in_the_reason() -> None:
    """MUST-FIRE: a premise past the model's window is still judged, and the reason SAYS so."""
    at_cap = Chunk(id="c", text="x" * lg.MAX_CONTEXT_CHARS)
    over = Chunk(id="b", text="x" * (lg.MAX_CONTEXT_CHARS + 1))
    assert (
        not lg.LettuceDetectGrounder(StubDetector())
        .verdict(Claim("fine", ("c",)), [at_cap])
        .reason.startswith(lg.WINDOW_REASON)
    )
    v = lg.LettuceDetectGrounder(StubDetector()).verdict(Claim("fine", ("b",)), [over])
    assert (
        v.supported is True
        and v.reason == f"context_over_window: {lg.MAX_CONTEXT_CHARS + 1} chars; no unsupported span"
    )


def test_the_window_counts_every_span_with_its_title() -> None:
    half = lg.MAX_CONTEXT_CHARS // 2
    spans = [Chunk(id="a", text="x" * half, title="T"), Chunk(id="b", text="y" * half)]
    v = lg.LettuceDetectGrounder(StubDetector()).verdict(Claim("fine", ("a", "b")), spans)
    assert v.reason.startswith("context_over_window: "), "3 title chars tip two half-window spans over"


def test_missing_dependency_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    for mod in ("lettucedetect", "lettucedetect.models", "lettucedetect.models.inference"):
        monkeypatch.setitem(sys.modules, mod, None)
    with pytest.raises(RuntimeError, match=r"llm-ports\[lettucedetect\]"):
        lg.LettuceDetectGrounder()


def test_the_grounder_satisfies_the_port() -> None:
    assert isinstance(lg.LettuceDetectGrounder(StubDetector()), Grounder)
