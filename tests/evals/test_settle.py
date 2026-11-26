"""Code's ruling on a labelled claim: only the recheck rules, and only in the direction it carries."""

from __future__ import annotations

from llm_ports.evals.claims import Claim, ClaimLabel
from llm_ports.evals.settle import settle
from llm_ports.grounding import numbers

INPUT = "reading = 590.4\nbaseline = 584.2\nsize = 4820000000.0"
ALLOWED = numbers.digits(INPUT)
FACTS = {"reading": 590.4, "baseline": 584.2, "size": 4820000000.0}
GAP = "abs(reading - baseline) / reading * 100 = abs(590.4 - 584.2) / 590.4 * 100 = 1.050136"


def _claim(label: ClaimLabel, quote: str, op: str | None = None, shown: str | None = None) -> Claim:
    extra = {"tol": {"abs": 0.05}} | ({"shown": shown} if shown else {})
    return Claim(id="C1", quote=quote, text=quote, label=label, op=op, extra=extra)


def _settle(claim: Claim) -> tuple[bool | None, str | None]:
    s = settle(claim, allowed=ALLOWED, words_ok=INPUT, facts=FACTS)
    return s.grounded, s.label


def test_a_breaking_figure_is_contradicted_whatever_the_judge_said() -> None:
    claim = _claim("derived", "The reading is 3.15% above the baseline.", GAP, "3.15%")
    assert _settle(claim) == (False, "contradicted")


def test_a_holding_figure_confirms_a_grounded_label() -> None:
    assert _settle(_claim("derived", "The reading is 1.05% above the baseline.", GAP, "1.05%")) == (True, "derived")
    bare = _claim("supported", "Size is $4.82B.", "size = 4820000000.0 = 4820000000.0", "$4.82B")
    assert _settle(bare) == (True, "supported")


def test_a_holding_figure_never_overturns_an_ungrounded_label() -> None:
    # the figure is right and the direction word is flipped: only the judge sees words
    claim = _claim("contradicted", "The reading is 1.05% below the baseline.", GAP, "1.05%")
    s = settle(claim, allowed=ALLOWED, facts=FACTS)
    assert (s.grounded, s.ruled, s.verdict) == (None, False, "holds")


def test_abstain_and_unchecked_leave_the_claim_to_the_judge() -> None:
    near = _claim("derived", "The reading is 1.13% above the baseline.", GAP, "1.13%")
    prose = _claim("derived", "The gap is small.", "a regime = pin")
    assert settle(near, allowed=ALLOWED, facts=FACTS).verdict == "abstain"
    assert not settle(near, allowed=ALLOWED, facts=FACTS).ruled
    assert settle(prose, allowed=ALLOWED).verdict == "unchecked"
    assert not settle(prose, allowed=ALLOWED).ruled


def test_a_refused_figure_is_a_hint_never_a_ruling() -> None:
    claim = _claim("supported", "The reading closed at 591.7.", None)
    s = settle(claim, allowed=ALLOWED, words_ok=INPUT)
    assert (s.claim_id, s.ruled, s.refused) == ("C1", False, ("591.7",))


def test_a_claim_with_licensed_figures_is_not_ruled_grounded() -> None:
    claim = _claim("not_in_input", "The reading of 590.4 rose because of the weather.", None)
    s = settle(claim, allowed=ALLOWED, words_ok=INPUT)
    assert (s.ruled, s.refused) == (False, ())
