"""Claim-level grounding: the case rule, the loader's two fact shapes and its refusals, per-label agreement,
the stable split and the paraphrase flip count -- each both ways."""

from __future__ import annotations

import json
from typing import Any

import pytest

from llm_ports.evals import claims as cl


def _claim(cid: str, label: str, **kw: Any) -> dict[str, Any]:
    return {"id": cid, "quote": f"q{cid}", "text": f"t{cid}", "label": label, **kw}


def _row(case_id: str = "inv-1", **kw: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case_id,
        "input": "F1 subtotal = 90.00\nF2 tax = 10.00",
        "output": "The invoice totals 100.00.",
        "claims": [_claim("C1", "derived", facts=["F1", "F2"], op="90.00 + 10.00 = 100.00")],
        "case_label": "pass",
    }
    row.update(kw)
    return row


def _lines(*rows: dict[str, Any]) -> list[str]:
    return [json.dumps(r) for r in rows]


class TestCaseLabel:
    def test_a_material_ungrounded_claim_fails_the_case(self) -> None:
        bad = cl.Claim("C1", "q", "t", "not_in_input")
        assert cl.case_label([bad]) == "fail"
        assert cl.case_label([cl.Claim("C1", "q", "t", "contradicted")]) == "fail"

    def test_an_immaterial_slip_and_grounded_claims_pass(self) -> None:
        slip = cl.Claim("C1", "q", "t", "not_in_input", material=False)
        ok = [cl.Claim("C2", "q", "t", "supported"), cl.Claim("C3", "q", "t", "derived")]
        assert cl.case_label([slip, *ok]) == "pass"
        assert cl.case_label([]) == "pass"


class TestReadCases:
    def test_both_fact_shapes_load_and_extra_keys_ride_along(self) -> None:
        listed = _row("a", facts=[{"id": "F1", "span": "Subtotal: 90.00"}], provenance="production")
        mapped = _row("b", facts={"subtotal": 90.0, "tax": 10.0}, synthetic=True)
        got = cl.read_cases(_lines(listed, mapped))
        assert got.problems == ()
        a, b = got.cases
        assert a.facts == (cl.Fact("F1", "Subtotal: 90.00"),) and a.extra == {"provenance": "production"}
        assert b.facts == (cl.Fact("subtotal", "subtotal = 90.0"), cl.Fact("tax", "tax = 10.0"))
        assert b.extra == {"synthetic": True}
        assert a.claims[0].facts == ("F1", "F2") and a.claims[0].op == "90.00 + 10.00 = 100.00"

    def test_material_absent_reads_true_and_false_reads_false(self) -> None:
        row = _row(
            claims=[
                _claim("C1", "supported"),
                _claim("C2", "supported", material=False),
                _claim("C3", "supported", material=None),
            ]
        )
        (case,) = cl.read_cases(_lines(row)).cases
        assert [c.material for c in case.claims] == [True, False, True]

    def test_rule_aliases_rename_without_editing_the_data(self) -> None:
        row = _row(claims=[_claim("C1", "derived", rule="OLD-1", tag="x"), _claim("C2", "supported")])
        (case,) = cl.read_cases(_lines(row), rule_aliases={"OLD-1": "NEW-1"}).cases
        assert case.claims[0].rule == "NEW-1" and case.claims[1].rule is None
        assert case.claims[0].extra == {"tag": "x"}

    @pytest.mark.parametrize(
        ("row", "says"),
        [
            ({"case_id": "x", "input": "", "output": "", "claims": []}, "lacks ['case_label']"),
            (_row(case_label="maybe"), "case_label 'maybe'"),
            (_row(claims=[_claim("C1", "plausible")]), "label 'plausible'"),
            (_row(claims=[{"id": "C1", "quote": "q", "label": "supported"}]), "lacks ['text']"),
            (_row(claims=[_claim("C1", "supported"), _claim("C1", "derived")]), "duplicate claim ids"),
        ],
        ids=["no-case-label", "bad-case-label", "fifth-label", "claim-key", "dup-claim"],
    )
    def test_a_bad_row_is_a_named_problem_never_a_raise(self, row: dict[str, Any], says: str) -> None:
        got = cl.read_cases(["", json.dumps(_row("good")), json.dumps(row)])
        assert [c.case_id for c in got.cases] == ["good"]
        assert len(got.problems) == 1 and got.problems[0].startswith("line 3: ") and says in got.problems[0]

    def test_unparseable_json_and_a_duplicate_case_id_are_problems(self) -> None:
        got = cl.read_cases([json.dumps(_row("a")), "{not json", json.dumps(_row("a"))])
        assert len(got.cases) == 1
        assert got.problems[0].startswith("line 2: ") and "duplicate case_id 'a'" in got.problems[1]


class TestCheck:
    def _case(self, **kw: Any) -> cl.Case:
        (case,) = cl.read_cases(_lines(_row(facts={"F1": "90.00", "F2": "10.00"}, **kw))).cases
        return case

    def test_a_sound_case_has_no_defect(self) -> None:
        assert cl.check(self._case(claims=[_claim("C1", "derived", quote="totals 100.00", facts=["F1", "F2"])])) == []

    def test_each_defect_is_named(self) -> None:
        case = self._case(
            claims=[
                _claim("C1", "supported", quote="not in the reply"),
                _claim("C2", "supported", quote="totals", facts=["F9"]),
                _claim("C3", "not_in_input", quote="100.00"),
            ]
        )
        assert cl.check(case) == [
            "C1: quote not in the output",
            "C2: cites unknown fact 'F9'",
            "case_label 'pass' but its claims imply 'fail'",
        ]
        assert cl.check(case, known=("F9",))[1:] == ["case_label 'pass' but its claims imply 'fail'"]

    def test_unjudgeable_is_never_contradicted_by_its_claims(self) -> None:
        case = self._case(case_label="unjudgeable", claims=[_claim("C1", "not_in_input", quote="100.00")])
        assert cl.check(case) == []


class TestAgreement:
    def test_perfect_labels_agree_on_every_class(self) -> None:
        pairs = [(lab, lab) for lab in cl.LABELS for _ in range(6)]
        got = cl.agreement(pairs)
        assert set(got) == {*cl.LABELS, "grounded"}
        assert all(a.kappa == pytest.approx(1.0) and a.n == 24 for a in got.values())

    def test_a_judge_that_calls_every_derivation_invented_fails_on_derived_and_grounded(self) -> None:
        """The failure the scheme exists to see: inference refused as invention."""
        pairs = (
            [("supported", "supported")] * 8
            + [("not_in_input", "derived")] * 8
            + [("not_in_input", "not_in_input")] * 4
        )
        got = cl.agreement(pairs)
        assert got["supported"].kappa == pytest.approx(1.0)
        derived = got["derived"]
        assert derived.kappa is not None and abs(derived.kappa) < 1e-9  # never called derived: chance
        assert got["grounded"].kappa is not None and got["grounded"].kappa < 0.6

    def test_an_abstention_is_left_out_not_counted_wrong(self) -> None:
        got = cl.agreement([("supported", "supported"), ("abstain", "derived")])
        assert got["supported"].n == 1


class TestSplitAndFlips:
    def test_split_is_stable_and_near_its_shares(self) -> None:
        ids = [f"case-{i}" for i in range(2000)]
        first = [cl.split(i) for i in ids]
        assert first == [cl.split(i) for i in ids]
        share = {s: first.count(s) / len(ids) for s in ("train", "dev", "test")}
        assert share["test"] == pytest.approx(0.2, abs=0.03) and share["dev"] == pytest.approx(0.2, abs=0.03)
        assert all(cl.split(i, dev=0.0, test=0.0) == "train" for i in ids[:50])

    def test_flips_count_only_cases_both_runs_answered(self) -> None:
        assert cl.flips({"a": "pass", "b": "fail", "c": "pass"}, {"a": "pass", "b": "pass", "d": "fail"}) == (1, 2)
        assert cl.flips({}, {"a": "pass"}) == (0, 0)

    def test_fact_table_numbers_each_fact_by_its_id(self) -> None:
        assert cl.fact_table([cl.Fact("F1", "subtotal = 90.00"), cl.Fact("F2", "tax = 10.00")]) == (
            "F1 subtotal = 90.00\nF2 tax = 10.00"
        )
