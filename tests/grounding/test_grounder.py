"""NumberLawGrounder keeps the Grounder port's laws and names its blind spot."""

from __future__ import annotations

from llm_ports import ports
from llm_ports.grounding import NumberLawGrounder
from llm_ports.ports import Chunk, Claim

SPAN = Chunk(id="a", text="SPY closed at 580.5, up 3%", title="Close")


def test_the_number_law_grounder_keeps_the_grounder_laws() -> None:
    g = NumberLawGrounder()
    assert isinstance(g, ports.Grounder)
    v = g.verdict(Claim("SPY at 580.5"), [])
    assert v.supported is False and v.checker == "number_law" and v.reason
    assert g.verdict(Claim("SPY at 580.5, up 3 percent", ("a",)), [SPAN]).supported is True
    bad = g.verdict(Claim("SPY at 581", ("a",)), [SPAN])
    assert bad.supported is False and "581" in bad.reason


def test_the_number_law_grounder_names_its_blind_spot() -> None:
    g = NumberLawGrounder()
    # a spelled number is a number no span rendered, even when '3%' is cited
    assert g.verdict(Claim("up three percent", ("a",)), [SPAN]).supported is False
    none = g.verdict(Claim("SPY will rally hard", ("a",)), [SPAN])
    assert none.supported is True and "no_numbers" in none.reason
