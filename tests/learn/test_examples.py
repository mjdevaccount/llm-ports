"""examples: coverage per topic, agreement with the human, a balanced few-shot pick."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from llm_ports.learn import Example, Source
from llm_ports.learn import examples as ex

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def lab(topic: str, subject: str, value: str | float, source: Source = "human", at: datetime | None = None) -> Example:
    return Example(topic, subject, value, source, at=at)


class TestCoverage:
    def test_every_topic_asked_for_appears_zeros_included(self) -> None:
        cov = ex.coverage([lab("a", "1", "good")], ["a", "b"], target=5)
        assert list(cov) == ["a", "b"]
        assert (cov["b"].have, cov["b"].gap, cov["b"].covered) == (0, 5, False)
        assert cov["b"].counts == {"human": 0, "model": 0, "judge": 0, "outcome": 0}

    def test_a_labelled_topic_not_asked_for_is_still_reported(self) -> None:
        assert list(ex.coverage([lab("z", "1", "good")], ["a"], target=1)) == ["a", "z"]

    def test_a_relabel_counts_once_and_the_latest_wins(self) -> None:
        rows = [lab("a", "1", "good", at=T0), lab("a", "1", "bad", at=T0 + timedelta(hours=1))]
        cov = ex.coverage(rows, ["a"], target={"a": 1})
        assert (cov["a"].have, cov["a"].positive, cov["a"].covered) == (1, 0, True)

    def test_sources_are_counted_apart(self) -> None:
        rows = [lab("a", "1", "good"), lab("a", "1", "bad", "judge"), lab("a", "2", 0.4, "outcome")]
        assert ex.coverage(rows, ["a"], target=3)["a"].counts == {"human": 1, "model": 0, "judge": 1, "outcome": 1}


class TestAgreement:
    def test_only_shared_subjects_count(self) -> None:
        rows = [
            lab("a", "1", "good"),
            lab("a", "2", "bad"),
            lab("a", "3", "good"),
            lab("a", "1", "good", "judge"),
            lab("a", "2", "good", "judge"),
        ]
        got = ex.agreement(rows)["a"]
        assert (got["judge"].n, got["judge"].agree) == (2, 0.5)
        assert (got["model"].n, got["model"].agree, got["model"].kappa) == (0, None, None)

    def test_a_topic_without_a_reference_label_is_absent(self) -> None:
        assert ex.agreement([lab("a", "1", "good", "judge")]) == {}


class TestFewShot:
    def test_balanced_across_classes_rarest_first(self) -> None:
        rows = [lab("a", str(i), "good") for i in range(10)] + [lab("a", "b1", "bad"), lab("a", "b2", "bad")]
        pick = ex.few_shot(rows, 4, topic="a")
        assert sorted(str(p.value) for p in pick) == ["bad", "bad", "good", "good"]
        assert pick[0].value == "bad"

    def test_deterministic_and_scoped_to_topic_and_source(self) -> None:
        rows = [lab("a", str(i), "good") for i in range(6)] + [lab("b", "x", "good"), lab("a", "j", "bad", "judge")]
        assert ex.few_shot(rows, 3, topic="a") == ex.few_shot(list(reversed(rows)), 3, topic="a")
        assert {p.topic for p in ex.few_shot(rows, 10, topic="a")} == {"a"}
        assert all(p.source == "human" for p in ex.few_shot(rows, 10, topic="a"))

    def test_k_larger_than_the_pool_returns_the_pool(self) -> None:
        assert len(ex.few_shot([lab("a", "1", "good")], 5, topic="a")) == 1
