"""labels -- coverage names every role's gap, draw stratifies by the judge's verdict, agreement
measures judge and model against the human on shared replies only, label_of never mints a human."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from llm_ports.evals import labels as lb
from llm_ports.ports import JudgeVerdict, Label


def L(role: str, subject: str, value: str, who: str, at: int | None = None) -> Label:
    when = datetime(2026, 11, 1, at, tzinfo=UTC) if at is not None else None
    return Label(role, subject, value, who, at=when)  # type: ignore[arg-type]  # test literals


def test_coverage_reports_every_role_even_unlabelled_and_counts_distinct_replies() -> None:
    labels = [
        L("a", "1", "good", "human"),
        L("a", "1", "bad", "human"),  # same reply relabelled: counts once
        L("a", "2", "bad", "human"),
        L("a", "1", "good", "judge"),
        L("x", "9", "good", "model"),  # a labelled role nobody asked about is still reported
    ]
    cov = lb.coverage(labels, ["a", "b"], target=3)
    assert list(cov) == ["a", "b", "x"]
    a = cov["a"]
    assert (a.human, a.model, a.judge, a.human_good, a.gap, a.covered) == (2, 0, 1, 0, 1, False)
    assert (cov["b"].human, cov["b"].gap) == (0, 3)
    assert cov["x"].model == 1 and cov["x"].target == 3


def test_coverage_target_mapping_defaults_a_missing_role_to_zero() -> None:
    cov = lb.coverage([], ["a", "b"], target={"a": 5})
    assert (cov["a"].gap, cov["b"].gap, cov["b"].covered) == (5, 0, True)


def test_the_latest_relabel_wins() -> None:
    got = lb.latest([L("a", "1", "bad", "human", at=3), L("a", "1", "good", "human", at=1)])
    assert got[("a", "1", "human")].value == "bad"
    got = lb.latest([L("a", "1", "bad", "human", at=1), L("a", "1", "good", "human", at=3)])
    assert got[("a", "1", "human")].value == "good"


def _cands(role: str, n_fail: int, n_pass: int) -> list[lb.Candidate]:
    return [lb.Candidate(role, f"{role}-f{i}", judged="fail") for i in range(n_fail)] + [
        lb.Candidate(role, f"{role}-p{i}", judged="pass") for i in range(n_pass)
    ]


def test_draw_closes_only_the_gap_and_takes_the_rare_class_first() -> None:
    cov = lb.coverage([], ["a"], target=4)
    got = lb.draw(_cands("a", 20, 1), cov, per_run=3)
    assert len(got) == 3
    assert sum(c.judged == "pass" for c in got) == 1  # the one pass made it in
    assert lb.draw(_cands("a", 20, 1), cov, per_run=3) == got  # deterministic


def test_draw_skips_seen_covered_and_unknown_roles() -> None:
    cov = lb.coverage([L("a", "x", "good", "human")], ["a", "b"], target=1)
    assert cov["a"].covered
    got = lb.draw([*_cands("a", 3, 3), *_cands("b", 2, 0), *_cands("c", 5, 5)], cov, per_run=5, seen=["b-f0"])
    assert [c.subject for c in got] == ["b-f1"]  # a covered, c not asked about, b-f0 seen


def test_draw_does_not_queue_one_subject_twice() -> None:
    cov = lb.coverage([], ["a"], target=5)
    dup = [lb.Candidate("a", "s1", judged="fail"), lb.Candidate("a", "s1", judged="fail")]
    assert [c.subject for c in lb.draw(dup, cov, per_run=5)] == ["s1"]


def test_agreement_uses_shared_replies_and_is_none_where_undefined() -> None:
    labels = [
        L("a", "1", "good", "human"),
        L("a", "2", "bad", "human"),
        L("a", "3", "good", "human"),
        L("a", "4", "bad", "human"),
        L("a", "1", "good", "judge"),
        L("a", "2", "bad", "judge"),
        L("a", "3", "bad", "judge"),
        L("a", "4", "bad", "judge"),
        L("a", "9", "good", "judge"),  # no human label: not counted
        L("b", "1", "good", "judge"),  # role with no human: absent
    ]
    got = lb.agreement(labels)
    assert set(got) == {"a"}
    j = got["a"]["judge"]
    assert (j.n, j.agree) == (4, 0.75)
    assert j.kappa == pytest.approx(0.5) and j.ac1 is not None
    m = got["a"]["model"]
    assert (m.n, m.agree, m.kappa, m.ac1) == (0, None, None, None)


def test_label_of_maps_a_verdict_and_refuses_to_mint_a_human() -> None:
    v = JudgeVerdict(passed=False, score=0.2, reason="invented a figure", judge="pydantic_evals")
    lab = lb.label_of(v, role="a", subject="o1", trace="t1")
    assert (lab.value, lab.labeller, lab.by, lab.reason, lab.trace) == (
        "bad",
        "model",
        "pydantic_evals",
        "invented a figure",
        "t1",
    )
    assert lb.label_of(v, role="a", subject="o1", by="gemini", labeller="judge").by == "gemini"
    with pytest.raises(ValueError, match="never a human"):
        lb.label_of(v, role="a", subject="o1", labeller="human")
