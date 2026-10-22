"""claim_refute: the grammar, attribution by idx (text only as fallback), and the cut-reason flag."""

from __future__ import annotations

from llm_ports import claim_refute as cr


def test_the_grammar_caps_items_claims_and_reasons() -> None:
    s = cr.verdicts_schema(max_items=3, claim_cap=50, reason_cap=80)
    v = s["properties"]["verdicts"]
    assert v["maxItems"] == 3 and v["items"]["properties"]["claim"]["maxLength"] == 50
    assert v["items"]["properties"]["reason"]["maxLength"] == 80
    assert v["items"]["properties"]["verdict"]["enum"] == ["uphold", "refute"]
    assert cr.verdicts_schema(max_items=1, claim_cap=1) is not cr.verdicts_schema(max_items=1, claim_cap=1)


def test_attribution_is_by_idx_and_text_is_the_fallback() -> None:
    claims = ["alpha holds", "beta holds", "gamma holds"]
    raw = [{"idx": 2, "claim": "a paraphrase", "verdict": "refute"},
           {"idx": 2, "claim": "dup idx", "verdict": "uphold"},         # duplicate idx: ignored
           {"idx": 9, "claim": "gamma holds", "verdict": "uphold"},     # out of range -> text fallback
           {"idx": True, "claim": "x"}, "junk"]
    got = cr.match_verdicts(claims, raw)
    assert got[0] is None
    assert got[1] is not None and got[1]["verdict"] == "refute"
    assert got[2] is not None and got[2]["idx"] == 9
    assert cr.match_verdicts(claims, None) == [None, None, None]


def test_a_reason_at_the_cap_is_flagged_truncated_and_kept() -> None:
    assert "reason_truncated" not in cr.refutation("c", "short")
    cut = cr.refutation("c", "x" * cr.REASON_MAX_CHARS, idx=1)
    assert cut["reason_truncated"] is True and cut["idx"] == 1 and cut["source"] == "adversary"
    assert cr.refutation("c", None)["reason"] == ""
