"""claim_refute: the grammar, attribution by idx (text only as fallback), the cut-reason flag, and
the pass itself -- the idx-numbered slots, the consumer's pack rendered, the consumer's generate."""

from __future__ import annotations

import json
from functools import partial
from typing import Any

from llm_ports import call
from llm_ports import prompts as packs
from llm_ports.evals import refute as cr
from llm_ports.ports import GenRequest
from llm_ports.testing import StubGenerator
from tests.conftest import PACKS, Role

REFUTE = Role("test.refute", tier="local")
PACK = packs.load("demo.refute", PACKS)
NOUNS = {"noun": "claims", "subject": "one decision", "evidence_label": "EVIDENCE"}


class TestThePass:
    """MUST-FIRE: every claim carries the 1-based idx its verdict must return -- the one thing
    `match_verdicts` attributes by."""

    def test_each_claim_is_numbered_from_one_and_a_mapping_keeps_its_keys(self) -> None:
        s = cr.refute_slots(claims=["a holds", {"claim": "b holds", "work_ids": [1, 2]}], evidence='{"e": 1}', **NOUNS)
        assert json.loads(s["claims"]) == [
            {"idx": 1, "claim": "a holds"},
            {"idx": 2, "claim": "b holds", "work_ids": [1, 2]},
        ]
        assert s["evidence"] == '{"e": 1}' and s["reason_cap"] == cr.REASON_MAX_CHARS
        assert (s["noun"], s["subject"], s["evidence_label"]) == ("claims", "one decision", "EVIDENCE")

    def test_the_prompt_is_the_consumers_pack_rendered(self) -> None:
        p = cr.refute_prompt(PACK, claims=["a holds"], evidence="[]", reason_cap=150, **NOUNS)
        assert p.startswith("Refute these claims about one decision; reasons under 150 chars.")
        assert 'CLAIMS:\n[{"idx": 1, "claim": "a holds"}]' in p and p.endswith("EVIDENCE:\n[]")

    async def test_the_pass_runs_through_the_consumers_generate(self) -> None:
        seen: list[GenRequest] = []
        outs: list[type[Any]] = []

        def reply(req: GenRequest, out: type[Any]) -> Any:
            seen.append(req)
            outs.append(out)
            return {"verdicts": [{"idx": 1, "claim": "a", "verdict": "refute", "reason": "no"}]}

        gen = partial(call.generate, generator=StubGenerator(reply))
        res = await cr.refute(
            gen,
            REFUTE,
            pack=PACK,
            claims=["a holds"],
            evidence="[]",
            max_items=3,
            claim_cap=40,
            name="refute_verdicts",
            reason_cap=90,
            **NOUNS,
        )
        assert res.outcome == "ok" and cr.match_verdicts(["a holds"], res.value["verdicts"])[0]["verdict"] == "refute"
        assert '"idx": 1' in seen[0].prompt
        schema = outs[0].json_schema  # type: ignore[attr-defined]
        assert outs[0].__name__ == "refute_verdicts"
        assert schema == cr.verdicts_schema(max_items=3, claim_cap=40, reason_cap=90)


def test_the_grammar_caps_items_claims_and_reasons() -> None:
    s = cr.verdicts_schema(max_items=3, claim_cap=50, reason_cap=80)
    v = s["properties"]["verdicts"]
    assert v["maxItems"] == 3 and v["items"]["properties"]["claim"]["maxLength"] == 50
    assert v["items"]["properties"]["reason"]["maxLength"] == 80
    assert v["items"]["properties"]["verdict"]["enum"] == ["uphold", "refute"]
    assert cr.verdicts_schema(max_items=1, claim_cap=1) is not cr.verdicts_schema(max_items=1, claim_cap=1)


def test_attribution_is_by_idx_and_text_is_the_fallback() -> None:
    claims = ["alpha holds", "beta holds", "gamma holds"]
    raw = [
        {"idx": 2, "claim": "a paraphrase", "verdict": "refute"},
        {"idx": 2, "claim": "dup idx", "verdict": "uphold"},  # duplicate idx: ignored
        {"idx": 9, "claim": "gamma holds", "verdict": "uphold"},  # out of range -> text fallback
        {"idx": True, "claim": "x"},
        "junk",
    ]
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
