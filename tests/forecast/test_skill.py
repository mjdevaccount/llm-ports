from __future__ import annotations

import random

import pytest

from llm_ports.forecast import Skill, brier_skill
from llm_ports.forecast.skill import MIN_CLUSTERS


def _world(n: int, seed: int = 7) -> tuple[list[bool], list[float]]:
    rng = random.Random(seed)
    truth = [rng.uniform(0.2, 0.8) for _ in range(n)]
    return [rng.random() < p for p in truth], truth


def test_a_voice_equal_to_its_reference_has_zero_skill() -> None:
    obs, truth = _world(300)
    s = brier_skill(obs, truth, truth)
    assert s.n == 300 and s.clusters == 300
    assert s.skill == pytest.approx(0.0)
    assert s.brier_score == pytest.approx(s.ref_brier_score)
    assert s.lo == pytest.approx(0.0) and s.hi == pytest.approx(0.0)


def test_the_true_probability_beats_a_flat_reference() -> None:
    obs, truth = _world(2000)
    s = brier_skill(obs, truth, [0.5] * len(obs))
    assert s.skill is not None and s.skill > 0
    assert s.lo is not None and s.lo > 0  # the interval clears zero at this n


def test_a_flat_voice_loses_to_a_sharp_reference() -> None:
    obs, truth = _world(2000)
    s = brier_skill(obs, [0.5] * len(obs), truth)
    assert s.skill is not None and s.skill < 0 and s.hi is not None and s.hi < 0


def test_the_point_is_the_ratio_of_mean_briers() -> None:
    s = brier_skill([True, False], [0.8, 0.4], [0.6, 0.5])
    bf = ((0.8 - 1) ** 2 + 0.4**2) / 2
    br = ((0.6 - 1) ** 2 + 0.5**2) / 2
    assert s.brier_score == pytest.approx(bf) and s.ref_brier_score == pytest.approx(br)
    assert s.skill == pytest.approx(1 - bf / br)
    assert s.lo is None and s.hi is None  # 2 clusters < MIN_CLUSTERS: no interval


def test_clusters_widen_the_interval_over_iid() -> None:
    # 40 days x 25 events per day, the day's tape shared: the honest interval is wider.
    rng = random.Random(3)
    obs, fct, ref, day = [], [], [], []
    for d in range(40):
        shock = rng.choice([0.25, 0.75])
        for _ in range(25):
            obs.append(rng.random() < shock)
            fct.append(0.6)
            ref.append(0.5)
            day.append(d)
    iid, clu = brier_skill(obs, fct, ref), brier_skill(obs, fct, ref, day)
    assert clu.clusters == 40 and iid.clusters == 1000
    assert iid.skill == pytest.approx(clu.skill)
    assert (clu.hi - clu.lo) > (iid.hi - iid.lo)  # type: ignore[operator]  # both intervals exist at this n


def test_invalid_pairs_are_skipped_and_misaligned_inputs_score_nothing() -> None:
    s = brier_skill([True, False, True], [0.7, float("nan"), 0.6], [0.5, 0.5, 2.0])
    assert s.n == 1 and s.clusters == 1
    assert brier_skill([True], [0.5, 0.5], [0.5]) == Skill(0, 0, None, None, None, None, None)
    assert brier_skill([True], [0.5], [0.5], clusters=["a", "b"]).n == 0


def test_a_perfect_reference_leaves_skill_undefined() -> None:
    s = brier_skill([True] * MIN_CLUSTERS, [0.9] * MIN_CLUSTERS, [1.0] * MIN_CLUSTERS)
    assert s.n == MIN_CLUSTERS and s.ref_brier_score == 0.0 and s.skill is None and s.lo is None


def test_seeded_and_bad_arguments_refused() -> None:
    obs, truth = _world(200)
    flat = [0.5] * len(obs)
    assert brier_skill(obs, truth, flat, seed=1) == brier_skill(obs, truth, flat, seed=1)
    with pytest.raises(ValueError):
        brier_skill(obs, truth, flat, level=1.0)
    with pytest.raises(ValueError):
        brier_skill(obs, truth, flat, n_resamples=0)
