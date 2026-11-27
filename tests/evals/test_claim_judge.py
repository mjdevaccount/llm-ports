"""claim_judge: the question per claim, the per-question grammar, a verdict held to its quote, the case
rule with abstention, and `grade` end to end -- an always-fail judge goes red, a judge that answers
like the gold goes green, and code-settled claims never reach the judge."""

from __future__ import annotations

import json
from collections.abc import Sequence
from functools import partial
from typing import Any

from llm_ports import call
from llm_ports import prompts as packs
from llm_ports.evals import claim_judge as cj
from llm_ports.evals import decompose as d
from llm_ports.evals.claims import Fact
from llm_ports.ports import GenRequest
from llm_ports.testing import Reply, StubGenerator
from tests.conftest import PACKS, Role

CLAIM = Role("test.claim", tier="local")
PACK = packs.load("demo.claim", PACKS)
INVOICE = "Vendor: Acme Supplies\nSubtotal: 90.00\nTax: 10.00\nTotal due: 100.00\nStatus: overdue since March"
FACTS = d.facts_from_text(INVOICE)
REPLY = json.dumps({"vendor": "Acme Supplies", "total": 100.0, "status": "overdue", "risk": "high", "notes": []})


def _ask(path: str = "risk", value: str = "high", *, question: cj.Question = "grounding", **kw: Any) -> cj.Ask:
    return cj.Ask(path, value, kw.pop("kind", "free"), question, **kw)


KINDS: dict[str, d.Kind] = {"vendor": "extract", "total": "extract", "status": "classify"}


def _kind(leaf: d.Leaf) -> d.Kind:
    return KINDS.get(leaf.field, "free")


POLICY = cj.Policy(kind_of=_kind, allowed={"status": ("current", "overdue")})


class ScriptedJudge:
    """A `ClaimJudge` that answers from a function of the ask, and records what it was asked."""

    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.asked: list[cj.Ask] = []

    async def judge_claim(self, ask: cj.Ask, facts: Sequence[Fact], *, context: str = "") -> cj.ClaimVerdict:
        self.asked.append(ask)
        return cj.read_verdict(ask, self.answer(ask), evidence="\n".join(f.text for f in facts), checker="stub")


class TestQuestion:
    def test_a_classify_value_and_a_stated_absence_are_asked_only_for_a_contradiction(self) -> None:
        assert cj.question_for(d.Leaf("status", "overdue", "overdue"), "classify") == "contradiction"
        assert cj.question_for(d.Leaf("notes", "[]", []), "free") == "contradiction"
        assert cj.question_for(d.Leaf("risk", "high", "high"), "free") == "grounding"

    def test_the_grammar_is_the_questions_labels_and_the_tables_ids(self) -> None:
        c = cj.verdict_schema("contradiction", ["F1", "F2"])
        assert c["properties"]["label"]["enum"] == ["consistent", "contradicted"]
        assert c["properties"]["fact_ids"]["items"]["enum"] == ["F1", "F2"]
        g = cj.verdict_schema("grounding", [])
        assert g["properties"]["label"]["enum"] == ["supported", "derived", "not_in_input", "contradicted"]
        assert "enum" not in g["properties"]["fact_ids"]["items"]
        assert c["required"][:2] == ["fact_ids", "quote"]  # evidence before the label
        assert cj.verdict_schema("grounding", []) is not g


class TestReadVerdict:
    def test_consistent_reads_as_derived_and_a_real_quote_is_judged(self) -> None:
        v = cj.read_verdict(
            _ask(), {"label": "consistent", "quote": "", "fact_ids": ["F1"]}, evidence=INVOICE, checker="m"
        )
        assert (v.label, v.outcome, v.checker, v.fact_ids) == ("derived", "judged", "m", ("F1",))
        q = cj.read_verdict(
            _ask(), {"label": "supported", "quote": '"total due: 100.00."'}, evidence=INVOICE, checker="m"
        )
        assert q.outcome == "judged" and q.label == "supported"

    def test_a_quote_not_in_the_facts_abstains(self) -> None:
        v = cj.read_verdict(_ask(), {"label": "contradicted", "quote": "paid in full"}, evidence=INVOICE, checker="m")
        assert v.label is None and v.outcome == "abstained" and "not in the facts" in v.reason

    def test_a_contradiction_without_a_quote_abstains(self) -> None:
        v = cj.read_verdict(_ask(), {"label": "contradicted", "quote": ""}, evidence=INVOICE, checker="m")
        assert v.label is None and v.outcome == "abstained"

    def test_no_known_label_could_not_run(self) -> None:
        for raw in ({"label": "maybe"}, None, "text"):
            assert cj.read_verdict(_ask(), raw, evidence=INVOICE, checker="m").outcome == "could_not_run"


def _v(label: str | None, outcome: cj.Outcome = "judged", *, material: bool = True) -> cj.ClaimVerdict:
    return cj.ClaimVerdict("p", label, outcome, material=material)  # type: ignore[arg-type]


class TestAggregate:
    def test_a_material_ungrounded_claim_fails_and_an_immaterial_one_does_not(self) -> None:
        assert cj.aggregate([_v("supported"), _v("not_in_input")]).label == "fail"
        assert cj.aggregate([_v("supported"), _v("contradicted", material=False)]).label == "pass"

    def test_a_material_claim_that_could_not_run_or_all_abstentions_are_unjudgeable(self) -> None:
        assert cj.aggregate([_v("supported"), _v(None, "could_not_run")]).label == "unjudgeable"
        assert cj.aggregate([_v(None, "abstained"), _v(None, "abstained")]).label == "unjudgeable"
        assert cj.aggregate([_v("supported"), _v(None, "abstained")]).label == "pass"

    def test_a_fail_outranks_could_not_run_and_skipped_claims_do_not_count(self) -> None:
        assert cj.aggregate([_v(None, "could_not_run"), _v("contradicted")]).label == "fail"
        assert cj.aggregate([_v(None, "skipped"), _v("derived")]).label == "pass"
        assert cj.aggregate([]).label == "pass"

    def test_counts_and_failing_name_what_happened(self) -> None:
        r = cj.aggregate([_v("supported", "settled"), _v("not_in_input"), _v(None, "skipped", material=False)])
        assert r.failing == ("p",)
        assert r.counts() == {
            "claims": 3,
            "material": 2,
            "outcome:settled": 1,
            "label:supported": 1,
            "outcome:judged": 1,
            "label:not_in_input": 1,
            "outcome:skipped": 1,
        }


class TestGrade:
    async def test_code_settled_claims_never_reach_the_judge(self) -> None:
        judge = ScriptedJudge(lambda a: {"label": "consistent" if a.question == "contradiction" else "derived"})
        r = await cj.grade(REPLY, INVOICE, judge=judge, policy=POLICY)
        by = {v.path: v for v in r.verdicts}
        assert by["vendor"].outcome == by["total"].outcome == "settled" and by["vendor"].checker == "code"
        assert [a.path for a in judge.asked] == ["status", "risk", "notes"]
        assert {a.path: a.question for a in judge.asked} == {
            "status": "contradiction",
            "risk": "grounding",
            "notes": "contradiction",
        }
        assert "the value is one of the field's allowed values" in judge.asked[0].hints
        assert r.label == "pass" and [v.path for v in r.verdicts] == ["vendor", "total", "status", "risk", "notes"]

    async def test_an_always_fail_judge_goes_red_and_a_gold_echo_goes_green(self) -> None:
        red = await cj.grade(REPLY, INVOICE, judge=ScriptedJudge(lambda a: {"label": "not_in_input"}), policy=POLICY)
        assert red.label == "fail" and "risk" in red.failing
        gold = {"status": {"label": "consistent"}, "risk": {"label": "derived", "quote": "overdue since March"}}
        green = await cj.grade(
            REPLY, INVOICE, judge=ScriptedJudge(lambda a: gold.get(a.path, {"label": "consistent"})), policy=POLICY
        )
        assert green.label == "pass" and green.failing == ()

    async def test_immaterial_claims_are_skipped_material_go_first_and_the_cap_holds(self) -> None:
        policy = cj.Policy(kind_of=lambda _l: "free", material_of=lambda lf: lf.path != "a")
        judge = ScriptedJudge(lambda a: {"label": "derived"})
        reply = json.dumps({"a": "x1", "b": "x2", "c": "x3"})
        r = await cj.grade(reply, "nothing", judge=judge, policy=policy, cap=1)
        assert [a.path for a in judge.asked] == ["b"]
        assert {v.path: v.outcome for v in r.verdicts} == {"a": "skipped", "b": "judged", "c": "skipped"}
        judge2 = ScriptedJudge(lambda a: {"label": "derived"})
        await cj.grade(reply, "nothing", judge=judge2, policy=policy, judge_immaterial=True)
        assert [a.path for a in judge2.asked] == ["b", "c", "a"]

    async def test_the_consumers_pre_rules_first_or_adds_its_hints(self) -> None:
        def pre(leaf: d.Leaf, _kind: d.Kind) -> d.Prematch | None:
            if leaf.path == "risk":
                return d.Prematch("supported", "the header lists it")
            return d.Prematch(None, "", ("consumer hint",)) if leaf.path == "status" else None

        judge = ScriptedJudge(lambda a: {"label": "consistent"})
        policy = cj.Policy(kind_of=_kind, pre=pre, allowed=POLICY.allowed)
        r = await cj.grade(REPLY, INVOICE, judge=judge, policy=policy)
        assert {v.path: v.reason for v in r.verdicts}["risk"] == "the header lists it"
        assert judge.asked[0].hints[0] == "consumer hint" and len(judge.asked[0].hints) == 2

    async def test_given_facts_are_the_table_the_judge_sees(self) -> None:
        seen: list[Sequence[Fact]] = []

        class Spy(ScriptedJudge):
            async def judge_claim(self, ask: cj.Ask, facts: Sequence[Fact], *, context: str = "") -> cj.ClaimVerdict:
                seen.append(facts)
                return await super().judge_claim(ask, facts, context=context)

        own = (Fact("R1", "risk: high"),)
        await cj.grade('{"risk": "high"}', "", judge=Spy(lambda a: {"label": "supported"}), policy=POLICY, facts=own)
        assert seen == [own]


class TestGenerateClaimJudge:
    async def test_the_pack_is_rendered_and_the_grammar_sent_through_the_consumers_generate(self) -> None:
        seen: list[GenRequest] = []
        outs: list[type[Any]] = []

        def reply(req: GenRequest, out: type[Any]) -> Any:
            seen.append(req)
            outs.append(out)
            return {"fact_ids": ["F5"], "quote": "overdue since March", "label": "supported", "reason": "F5 says so"}

        judge = cj.GenerateClaimJudge(partial(call.generate, generator=StubGenerator(reply)), CLAIM, pack=PACK)
        v = await judge.judge_claim(_ask("status", "overdue", kind="classify"), FACTS, context="CONTRACT")
        assert (v.label, v.outcome, v.fact_ids) == ("supported", "judged", ("F5",)) and v.checker
        assert seen[0].prompt.startswith("CONTRACT\nFACTS:\nF1 Vendor: Acme Supplies")
        assert "CLAIM status = overdue (classify, grounding)" in seen[0].prompt
        assert outs[0].__name__ == "claim_verdict"
        assert outs[0].json_schema == cj.verdict_schema("grounding", [f.id for f in FACTS])

    async def test_a_failed_call_or_a_broken_pack_is_the_claims_could_not_run(self) -> None:
        down = StubGenerator(Reply.failed("server", reason="503"))
        v = await cj.GenerateClaimJudge(partial(call.generate, generator=down), CLAIM, pack=PACK).judge_claim(
            _ask(), FACTS
        )
        assert v.outcome == "could_not_run" and v.label is None
        broken = cj.GenerateClaimJudge(
            partial(call.generate, generator=StubGenerator({})), CLAIM, pack=PACK, template="missing"
        )
        b = await broken.judge_claim(_ask(), FACTS)
        assert b.outcome == "could_not_run" and b.checker == ""

    def test_the_slots_put_the_shared_prefix_first(self) -> None:
        s = cj.claim_slots(_ask(hints=("h1", "h2")), FACTS, context="C")
        assert list(s)[:2] == ["context", "facts"] and s["hints"] == "- h1\n- h2"
        assert s["labels"] == "supported, derived, not_in_input, contradicted"
        assert cj.claim_slots(_ask(question="contradiction"), FACTS)["labels"] == "consistent, contradicted"
