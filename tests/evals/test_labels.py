"""labels -- topics name model replies, draw closes each role's gap and stratifies by the judge's
verdict, label_of never mints a human, and the core's coverage/agreement read the examples."""

from __future__ import annotations

import pytest

from llm_ports.evals import labels as lb
from llm_ports.learn import Example
from llm_ports.learn.examples import agreement, coverage
from llm_ports.ports import JudgeVerdict


def E(role: str, subject: str, value: str, source: str) -> Example:
    return Example(lb.topic(role), subject, value, source)  # type: ignore[arg-type]  # test literals


def test_topic_and_role_of_round_trip_and_reject_other_topics() -> None:
    assert lb.topic("summarize") == "llm:summarize"
    assert lb.role_of("llm:summarize") == "summarize"
    assert lb.role_of("trend:1d") is None
    assert lb.role_of("llm:") is None


def _cands(role: str, n_fail: int, n_pass: int) -> list[lb.Candidate]:
    return [lb.Candidate(role, f"{role}-f{i}", judged="fail") for i in range(n_fail)] + [
        lb.Candidate(role, f"{role}-p{i}", judged="pass") for i in range(n_pass)
    ]


def test_draw_closes_only_the_gap_and_takes_the_rare_class_first() -> None:
    cov = coverage([], [lb.topic("a")], target=4)
    got = lb.draw(_cands("a", 20, 1), cov, per_run=3)
    assert len(got) == 3
    assert sum(c.judged == "pass" for c in got) == 1  # the one pass made it in
    assert lb.draw(_cands("a", 20, 1), cov, per_run=3) == got  # deterministic


def test_draw_skips_seen_covered_unknown_and_non_llm_topics() -> None:
    cov = coverage([E("a", "x", "good", "human")], [lb.topic("a"), lb.topic("b"), "trend:1d"], target=1)
    assert cov[lb.topic("a")].covered
    got = lb.draw([*_cands("a", 3, 3), *_cands("b", 2, 0), *_cands("c", 5, 5)], cov, per_run=5, seen=["b-f0"])
    assert [c.subject for c in got] == ["b-f1"]  # a covered, c not asked about, b-f0 seen


def test_draw_does_not_queue_one_subject_twice() -> None:
    cov = coverage([], [lb.topic("a")], target=5)
    dup = [lb.Candidate("a", "s1", judged="fail"), lb.Candidate("a", "s1", judged="fail")]
    assert [c.subject for c in lb.draw(dup, cov, per_run=5)] == ["s1"]


def test_the_core_measures_judge_against_human_on_shared_replies() -> None:
    labels = [
        *(E("a", s, v, "human") for s, v in (("1", "good"), ("2", "bad"), ("3", "good"), ("4", "bad"))),
        *(E("a", s, v, "judge") for s, v in (("1", "good"), ("2", "bad"), ("3", "bad"), ("4", "bad"))),
        E("a", "9", "good", "judge"),  # no human label: not counted
    ]
    j = agreement(labels)[lb.topic("a")]["judge"]
    assert (j.n, j.agree) == (4, 0.75)
    assert j.kappa == pytest.approx(0.5)


def test_label_of_maps_a_verdict_and_refuses_to_mint_a_human() -> None:
    v = JudgeVerdict(passed=False, score=0.2, reason="invented a figure", judge="pydantic_evals")
    ex = lb.label_of(v, role="a", subject="o1", trace="t1")
    assert (ex.topic, ex.value, ex.source, ex.by, ex.reason) == (
        "llm:a",
        "bad",
        "model",
        "pydantic_evals",
        "invented a figure",
    )
    assert ex.ref == {"trace": "t1"}
    judged = lb.label_of(
        JudgeVerdict(passed=True, score=1.0, reason="", judge="j"), role="a", subject="o1", by="gemini", source="judge"
    )
    assert (judged.value, judged.source, judged.by, judged.ref) == ("good", "judge", "gemini", {})
    with pytest.raises(ValueError, match="never a 'human' label"):
        lb.label_of(v, role="a", subject="o1", source="human")  # type: ignore[arg-type]  # the refusal under test
