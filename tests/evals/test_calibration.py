"""calibration: judge agreement (kappa, AC1), seeded-bad replies, and the both-direction verdict."""

from __future__ import annotations

import json

from llm_ports.evals import calibration as cal


class TestAgreement:
    def test_perfect_agreement_is_one(self) -> None:
        assert cal.cohen_kappa([("yes", "yes"), ("no", "no"), ("yes", "yes"), ("no", "no")]) == 1.0

    def test_chance_level_agreement_is_zero(self) -> None:
        k = cal.cohen_kappa([("yes", "yes"), ("yes", "no"), ("no", "yes"), ("no", "no")])
        assert k is not None and abs(k) < 1e-9

    def test_a_lopsided_base_rate_deflates_raw_agreement(self) -> None:
        k = cal.cohen_kappa([("yes", "yes")] * 9 + [("yes", "no")])
        assert k is not None and k < 0.5, "raw agreement is 0.9; kappa corrects for the one-sided marginals"

    def test_an_undefined_kappa_is_none_never_a_number(self) -> None:
        assert cal.cohen_kappa([]) is None
        assert cal.cohen_kappa([("yes", "yes")]) is None
        assert cal.cohen_kappa([("yes", "yes"), ("yes", "yes")]) is None, "chance agreement 1.0"

    def test_ac1_stays_defined_where_kappa_degenerates(self) -> None:
        assert (cal.gwet_ac1([("yes", "yes")] * 9 + [("yes", "no")]) or 0) > 0.8
        assert cal.gwet_ac1([("yes", "yes")] * 5) == 1.0
        assert cal.cohen_kappa([("yes", "yes")] * 5) is None
        assert cal.gwet_ac1([("yes", "no")]) is None, "under two pairs"


class TestSeeds:
    def test_the_placeholder_seed_blanks_every_string_and_a_prose_reply(self) -> None:
        reply = json.dumps({"kind": "chart", "levels": ["410 support"], "n": 3, "ok": True})
        got = cal.seed_placeholder(reply)
        assert got is not None and json.loads(got) == {"kind": "TBD", "levels": ["TBD"], "n": 3, "ok": True}
        assert cal.seed_placeholder("A prose note.") == "TBD"
        assert cal.seed_placeholder("A prose note.", "N/A") == "N/A"
        assert cal.seed_placeholder(json.dumps({"n": 1})) is None, "nothing to blank is not a seeded case"

    def test_the_figure_seed_lands_on_the_longest_free_text_field(self) -> None:
        reply = json.dumps({"kind": "chart", "summary": "Price holds the range.", "note": "short one"})
        got = cal.seed_figure(reply)
        assert got is not None
        parsed = json.loads(got)
        assert parsed["summary"] == "Price holds the range." + cal.FIGURE and parsed["kind"] == "chart"
        assert cal.seed_figure("Prose.") == "Prose." + cal.FIGURE
        assert cal.seed_figure(json.dumps({"kind": "chart"})) is None, "an enum is not free text"
        assert cal.seed_figure(json.dumps([1, 2])) is None


def _r(role: str, kind: str, after: str) -> dict[str, str]:
    return {"role": role, "kind": kind, "after": after}


class TestTallyAndVerdict:
    def test_tally_counts_both_directions_and_a_could_not_run_is_no_catch(self) -> None:
        rows = [
            {"role": "r", "kind": "good", "before": "fail", "after": "pass", "held_out": True},
            {"role": "r", "kind": "good", "before": "fail", "after": "could_not_run", "held_out": True},
            {"role": "r", "kind": "tbd", "after": "fail"},
            {"role": "r", "kind": "figure", "after": "could_not_run"},
        ]
        t = cal.tally(rows)["r"]
        assert (t["n"], t["before_pass"], t["after_pass"], t["after_could_not_run"]) == (2, 0, 1, 1)
        assert (t["tbd_n"], t["tbd_fail"], t["figure_n"], t["figure_fail"]) == (1, 1, 1, 0)

    def test_an_all_could_not_run_replay_is_blind_not_a_measurement(self) -> None:
        assert cal.verdict([])[0] == cal.BLIND
        code, why = cal.verdict([_r("a", "good", "could_not_run"), _r("a", "tbd", "could_not_run")])
        assert code == cal.BLIND and "BLIND: 2/2" in why
        one_dark = [_r("a", "good", "pass"), _r("a", "tbd", "fail"), _r("b", "good", "could_not_run")]
        code, why = cal.verdict(one_dark)
        assert code == cal.BLIND and why.endswith("b (good)")

    def test_an_unmeasured_seeded_bad_direction_is_blind_not_calibrated(self) -> None:
        cnr_bad = [_r("a", "good", "pass"), _r("a", "tbd", "could_not_run"), _r("a", "figure", "could_not_run")]
        code, why = cal.verdict(cnr_bad)
        assert code == cal.BLIND and why.endswith("a (seeded-bad)")
        assert cal.verdict([_r("a", "good", "pass")])[0] == cal.BLIND, "nothing seeded proves one direction"

    def test_both_directions_decide_the_exit(self) -> None:
        good = [
            _r("a", "good", "pass"),
            _r("a", "good", "could_not_run"),
            _r("a", "tbd", "fail"),
            _r("a", "figure", "fail"),
        ]
        code, why = cal.verdict(good)
        assert code == cal.CALIBRATED and "1/4 could_not_run" in why
        leaky = [*good[:2], _r("a", "tbd", "pass")]
        assert cal.verdict(leaky)[0] == cal.FINDING, "a seeded-bad reply that passed: the judge passes everything"
        harsh = [_r("a", "good", "fail"), _r("a", "good", "pass"), _r("a", "tbd", "fail")]
        assert cal.verdict(harsh, floor=0.5)[0] == cal.CALIBRATED
        code, why = cal.verdict(harsh, floor=0.75)
        assert code == cal.FINDING and "a 0.50" in why
