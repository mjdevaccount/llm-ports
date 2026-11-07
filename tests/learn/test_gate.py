"""gate: clauses in, one verdict out; missing evidence never passes."""

from __future__ import annotations

import json
import math

import pytest

from llm_ports.learn import Clause, Verdict
from llm_ports.learn import gate as g


class TestClauses:
    def test_at_least_passes_at_the_bar(self) -> None:
        assert g.at_least("n", 200, 200).passed is True
        assert g.at_least("n", 199, 200).passed is False

    def test_at_most_passes_at_the_bar(self) -> None:
        assert g.at_most("ece", 0.05, 0.05).passed is True
        assert g.at_most("ece", 0.06, 0.05).passed is False

    @pytest.mark.parametrize("missing", [None, math.nan, math.inf, "x", True])
    def test_a_value_that_is_not_a_finite_number_is_not_evaluated(self, missing: object) -> None:
        clause = g.at_least("n", missing, 1)  # type: ignore[arg-type]  # deliberately not a number
        assert clause.passed is None and clause.value is None

    def test_a_failed_label_check_names_both_labels(self) -> None:
        v = g.decide("sign", [g.equals("sign", "-", "+")])
        assert (v.decision, v.reason) == ("fail", "sign: - not == +")

    def test_equals_and_check(self) -> None:
        assert g.equals("sign", "+", "+").passed is True
        assert g.equals("sign", None, "+").passed is None
        assert g.check("seeded_bad_caught", False, 3).passed is False


class TestDecide:
    def test_every_clause_passing_is_pass(self) -> None:
        v = g.decide("cal", [g.at_least("n", 300, 200, role="floor"), g.at_most("ece", 0.01, 0.05)], subject="SPY")
        assert (v.decision, v.blocker, v.subject) == ("pass", None, "SPY")
        assert v.passed

    def test_a_failed_floor_is_insufficient_not_fail(self) -> None:
        v = g.decide("cal", [g.at_most("ece", 0.9, 0.05), g.at_least("n", 10, 200, role="floor")])
        assert (v.decision, v.blocker) == ("insufficient", "n")

    def test_an_unevaluated_floor_is_insufficient(self) -> None:
        assert g.decide("cal", [g.at_least("n", None, 200, role="floor")]).decision == "insufficient"

    def test_the_first_blocking_clause_is_the_blocker(self) -> None:
        v = g.decide("cal", [g.at_least("a", 1, 0), g.at_least("b", -1, 0), g.at_least("c", -1, 0)])
        assert (v.decision, v.blocker) == ("fail", "b")
        assert "b: -1 not >= 0" in v.reason

    def test_an_unevaluated_clause_fails_closed(self) -> None:
        v = g.decide("cal", [g.at_least("n", 300, 200, role="floor"), g.at_most("ece", math.nan, 0.05)])
        assert (v.decision, v.blocker) == ("fail", "ece")
        assert "not evaluated" in v.reason

    def test_a_veto_fails_even_under_a_failed_floor(self) -> None:
        v = g.decide("cal", [g.at_least("n", 5, 200, role="floor"), g.check("artifact_free", False, role="veto")])
        assert (v.decision, v.blocker) == ("fail", "artifact_free")
        assert [c.name for c in v.blocking] == ["n", "artifact_free"], "every non-passing clause is kept"

    def test_an_unevaluated_veto_does_not_disqualify(self) -> None:
        v = g.decide("cal", [g.check("artifact_free", None, role="veto")], unevaluated="insufficient")
        assert (v.decision, v.blocker) == ("insufficient", "artifact_free")
        assert g.decide("cal", [g.check("artifact_free", None, role="veto")]).decision == "fail"

    def test_unevaluated_insufficient_still_fails_a_measured_bar(self) -> None:
        clauses = [g.check("unknown", None), g.at_least("lift", -0.1, 0.0)]
        v = g.decide("cal", clauses, unevaluated="insufficient")
        assert (v.decision, v.blocker) == ("fail", "lift"), "we looked and it failed beats could not look"
        assert g.decide("cal", [g.check("unknown", None)], unevaluated="insufficient").decision == "insufficient"

    @pytest.mark.parametrize(
        ("n", "lift_role", "lift_ok", "displacement", "deflated", "want"),
        [
            (865, "floor", True, True, True, "pass"),  # promote
            (865, "floor", True, None, True, "insufficient"),  # displacement unknown: hold
            (865, "floor", True, False, True, "fail"),  # artifact: reject
            (865, "veto", False, True, True, "fail"),  # no lift at all: reject
            (865, "floor", False, True, True, "insufficient"),  # lift not robust: hold
            (50, "floor", True, False, True, "fail"),  # reject beats hold (underpowered + artifact)
            (50, "floor", True, True, True, "insufficient"),  # underpowered: hold
            (865, "floor", True, True, None, "insufficient"),  # no deflation bar supplied: hold
            (865, "floor", True, True, False, "insufficient"),  # under the deflated bar: hold
        ],
    )
    def test_promote_hold_reject_maps_exactly(
        self, n: int, lift_role: str, lift_ok: bool, displacement: bool | None, deflated: bool | None, want: str
    ) -> None:
        clauses = [
            g.at_least("min_sample", n, 100, role="floor"),
            g.check("reaction_lift", lift_ok, role=lift_role),  # type: ignore[arg-type]  # parametrized Role
            g.check("displacement", displacement, role="veto"),
            g.check("deflated_bar", deflated, role="floor"),
        ]
        assert g.decide("level-source", clauses, unevaluated="insufficient").decision == want

    def test_no_clauses_never_passes(self) -> None:
        v = g.decide("cal", [])
        assert v.decision == "could_not_run" and not v.passed

    def test_could_not_run_carries_its_reason(self) -> None:
        v = g.could_not_run("cal", "store down", subject="SPY")
        assert (v.decision, v.reason, v.clauses, v.blocker) == ("could_not_run", "store down", (), None)


class TestRoundTrip:
    def test_a_verdict_survives_json(self) -> None:
        clauses = [g.at_least("n", 12, 200, role="floor"), g.check("ok", None), g.check("v", True, role="veto")]
        v = g.decide("cal", clauses, subject="QQQ")
        assert Verdict.from_dict(json.loads(json.dumps(v.as_dict()))) == v

    def test_an_unknown_role_reads_as_bar(self) -> None:
        assert Clause.from_dict({"name": "x", "passed": True, "role": "boss"}).role == "bar"

    def test_a_non_finite_value_is_stored_as_null(self) -> None:
        raw = Clause("x", None, math.nan, math.inf, ">=").as_dict()
        assert raw["value"] is None and raw["threshold"] is None
        json.dumps(raw, allow_nan=False)

    @pytest.mark.parametrize(
        "raw",
        [
            {},
            {"decision": "maybe"},
            {"decision": "pass", "clauses": [{"passed": True}]},
            {"decision": "pass", "clauses": 3},
        ],
    )
    def test_something_that_is_not_a_verdict_reads_as_none(self, raw: dict[str, object]) -> None:
        assert Verdict.from_dict(raw) is None
