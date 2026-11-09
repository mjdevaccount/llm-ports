"""adapters.gepa: GEPA evolves a candidate from rollout feedback, and every failure is named."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

pytest.importorskip("gepa")

from llm_ports.adapters import gepa as g
from llm_ports.forecast.calibrate import Insufficient

ITEMS = [f"q{i}" for i in range(12)]
SEED = {"instruction": "Answer the question."}


def evaluate(candidate: Mapping[str, str], items: Sequence[str]) -> list[g.Rollout[str]]:
    """Score 1.0 when the instruction asks for a citation, else 0.0 with feedback saying so."""
    ok = "cite" in candidate["instruction"].lower()
    return [
        g.Rollout(output=f"{q}: answer", score=1.0 if ok else 0.0, feedback="" if ok else "no citation given")
        for q in items
    ]


def reflect(prompt: str) -> str:
    """A reflection model that reads the feedback and adds the citation rule."""
    assert "no citation given" in prompt
    return "```\nAnswer the question and cite your source.\n```"


def test_optimize_finds_a_better_candidate() -> None:
    seed = dict(SEED)
    got = g.optimize(seed, ITEMS[:8], ITEMS[8:], evaluate=evaluate, reflect=reflect, max_metric_calls=40)
    assert isinstance(got, g.Optimized)
    assert got.improved
    assert got.seed_score == 0.0
    assert got.best_score == 1.0
    assert "cite" in got.candidate["instruction"]
    assert got.candidates >= 2
    assert 0 < got.metric_calls
    assert (got.n_train, got.n_val) == (8, 4)
    assert seed == SEED  # the seed is never mutated


def test_no_improvement_is_reported_not_promoted() -> None:
    got = g.optimize(
        SEED, ITEMS[:8], ITEMS[8:], evaluate=evaluate, reflect=lambda p: "```\nJust answer.\n```", max_metric_calls=30
    )
    assert isinstance(got, g.Optimized)
    assert not got.improved
    assert got.candidate == SEED


@pytest.mark.parametrize(
    ("seed", "train", "val", "need"),
    [
        ({}, ITEMS[:8], ITEMS[8:], 1),
        (SEED, ITEMS[:3], ITEMS[8:], g.MIN_TRAIN),
        (SEED, ITEMS[:8], ITEMS[8:9], g.MIN_VAL),
    ],
)
def test_too_little_is_insufficient(seed: dict[str, str], train: list[str], val: list[str], need: int) -> None:
    got = g.optimize(seed, train, val, evaluate=evaluate, reflect=reflect)
    assert isinstance(got, Insufficient)
    assert got.need == need


def test_an_evaluator_that_raises_is_unavailable() -> None:
    def boom(candidate: Mapping[str, str], items: Sequence[str]) -> list[g.Rollout[str]]:
        raise RuntimeError("model down")

    got = g.optimize(SEED, ITEMS[:8], ITEMS[8:], evaluate=boom, reflect=reflect)
    assert isinstance(got, g.Unavailable)
    assert "model down" in got.reason


def test_a_rollout_count_mismatch_is_unavailable() -> None:
    got = g.optimize(SEED, ITEMS[:8], ITEMS[8:], evaluate=lambda c, items: [], reflect=reflect)
    assert isinstance(got, g.Unavailable)
    assert "0 rollouts" in got.reason


def test_a_reflection_that_raises_is_unavailable() -> None:
    def down(prompt: str) -> str:
        raise ConnectionError("gateway refused")

    got = g.optimize(SEED, ITEMS[:8], ITEMS[8:], evaluate=evaluate, reflect=down, max_metric_calls=30)
    assert isinstance(got, g.Unavailable)
    assert "gateway refused" in got.reason


def test_scores_are_cleaned_and_the_feedback_says_why() -> None:
    def odd(candidate: Mapping[str, str], items: Sequence[str]) -> list[g.Rollout[str]]:
        return [g.Rollout("a", float("nan"), "bad"), g.Rollout("b", 1.7), g.Rollout("c", -0.2, "")]

    adapter = g.RolloutAdapter(odd, describe=lambda q: f"<{q}>")
    batch = adapter.evaluate(["x", "y", "z"], dict(SEED), capture_traces=True)
    assert batch.scores == [0.0, 1.0, 0.0]
    data = adapter.make_reflective_dataset(dict(SEED), batch, ["instruction"])
    records = data["instruction"]
    assert records[0]["Inputs"] == "<x>"
    assert records[0]["Feedback"].startswith("bad (score nan is not finite")
    assert "clamped" in records[1]["Feedback"]
    assert "clamped" in records[2]["Feedback"]
    assert adapter.evaluate(["x", "y", "z"], dict(SEED)).trajectories is None


def test_feedback_falls_back_to_the_score() -> None:
    adapter = g.RolloutAdapter(lambda c, items: [g.Rollout("out", 0.25)])
    batch = adapter.evaluate(["x"], dict(SEED), capture_traces=True)
    assert adapter.make_reflective_dataset(dict(SEED), batch, ["instruction"])["instruction"][0]["Feedback"] == (
        "score 0.250"
    )


def test_some_failed_reflections_are_counted() -> None:
    calls = {"n": 0}

    def flaky(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("slow")
        return reflect(prompt)

    got = g.optimize(SEED, ITEMS[:8], ITEMS[8:], evaluate=evaluate, reflect=flaky, max_metric_calls=40)
    assert isinstance(got, g.Optimized)
    assert got.reflection_failures == 1
    assert got.reflections >= 2
    assert got.improved
    assert 0 < len(got.log_tail) <= g.LOG_TAIL


def test_a_message_list_prompt_is_flattened() -> None:
    assert g._as_text([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]) == "a\n\nb"
    assert g._as_text("plain") == "plain"


def test_a_dead_reflector_stops_the_search_without_the_budget() -> None:
    """GEPA swallows each failed reflection; an unspendable budget must not keep it calling."""
    calls = {"n": 0}

    def out_of_quota(prompt: str) -> str:
        calls["n"] += 1
        raise RuntimeError("429 free tier requests, limit: 20")

    got = g.optimize(
        SEED,
        ITEMS[:8],
        ITEMS[8:],
        evaluate=evaluate,
        reflect=out_of_quota,
        max_metric_calls=10**9,
        max_reflection_failures=3,
    )
    assert isinstance(got, g.Unavailable)
    # GEPA retries a failed batched reflection once per task, so an iteration can make 2 calls.
    assert 3 <= calls["n"] <= 4
    assert got.reason.startswith(f"{calls['n']} reflections failed in a row")
    assert "429 free tier" in got.reason


def test_a_success_resets_the_failure_run() -> None:
    calls = {"n": 0}

    def every_other(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] % 2:
            raise TimeoutError("slow")
        return "```\nJust answer.\n```"

    got = g.optimize(
        SEED,
        ITEMS[:8],
        ITEMS[8:],
        evaluate=evaluate,
        reflect=every_other,
        max_metric_calls=60,
        max_reflection_failures=2,
    )
    assert isinstance(got, g.Optimized)
    assert got.reflection_failures >= 2
    assert got.stopped == "budget"


def test_a_wall_clock_bound_stops_the_search_and_says_so() -> None:
    got = g.optimize(
        SEED, ITEMS[:8], ITEMS[8:], evaluate=evaluate, reflect=reflect, max_metric_calls=10**9, timeout_s=0.0
    )
    assert isinstance(got, g.Optimized)
    assert got.stopped == "timeout"
    assert got.metric_calls < 10**9
