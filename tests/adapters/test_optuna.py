"""adapters.optuna: tuning reads only the train split and reports the holdout out of sample."""

from __future__ import annotations

import random
from collections.abc import Mapping
from datetime import datetime, timedelta

import pytest

pytest.importorskip("optuna")
pytest.importorskip("river")

from llm_ports.adapters import optuna as tuner
from llm_ports.adapters.river import RiverLogitPool
from llm_ports.forecast import Combiner, Event, PriorPool
from llm_ports.forecast.calibrate import Insufficient

T0 = datetime(2026, 1, 1)
SPACE = {
    "lr": tuner.Range(1e-3, 0.3, log=True),
    "alpha": tuner.Range(0.7, 2.0),
    "shrink": tuner.Range(0.0, 1.0),
}


def events(n: int, seed: int = 7) -> list[Event]:
    rng = random.Random(seed)
    out = []
    for h in range(n):
        y = rng.random() < 0.55
        right = rng.random() < 0.8
        x = {"sharp": 0.75 if y == right else 0.25, "noise": rng.random(), "prior": 0.55}
        t = T0 + timedelta(hours=h)
        out.append(Event(x=x, forecast_time=t, resolve_time=t + timedelta(hours=2), resolution=y))
    return out


def build(p: dict[str, float]) -> Combiner:
    return PriorPool(inner=RiverLogitPool(lr=p["lr"], warmup=20), alpha=p["alpha"], shrink=p["shrink"])


class Spy:
    """Abstains, and records the forecast time (carried in x) of every outcome it hears."""

    def __init__(self, heard: list[float]) -> None:
        self.heard = heard

    def predict_proba_one(self, x: Mapping[str, float]) -> dict[bool, float]:
        del x
        return {}

    def learn_one(self, x: Mapping[str, float], y: bool) -> None:
        del y
        self.heard.append(x["t"])


def test_tune_finds_parameters_and_scores_the_holdout() -> None:
    got = tuner.tune(build, events(600), SPACE, study_name="t", n_trials=12, seed=1)
    assert isinstance(got, tuner.Tuned)
    assert set(got.params) == {"lr", "alpha", "shrink"}
    assert got.trials == 12 and got.failed_trials == 0
    assert got.n_train == 480 and got.holdout.n == 120
    assert got.holdout.brier_score is not None and got.holdout.brier_score < 0.25  # beats a coin
    assert got.train_brier < 0.25


def test_the_objective_never_reads_a_holdout_event() -> None:
    evs = [
        Event(
            x={"t": float(h)},
            forecast_time=T0 + timedelta(hours=h),
            resolve_time=T0 + timedelta(hours=h),
            resolution=True,
        )
        for h in range(200)
    ]
    runs: list[list[float]] = []

    def spy(_: dict[str, float]) -> Combiner:
        runs.append([])
        return Spy(runs[-1])

    got = tuner.tune(spy, evs, {"a": tuner.Range(0.0, 1.0)}, study_name="spy", n_trials=1)
    assert isinstance(got, tuner.Tuned) and len(runs) == 2  # one trial, then the full replay
    assert got.train_brier == pytest.approx(0.25)  # every abstention scored as a coin
    cut = (got.holdout_start - T0).total_seconds() / 3600
    assert runs[0] and max(runs[0]) < cut  # the trial heard train outcomes only
    assert max(runs[1]) >= cut  # the full replay learns through the holdout, as it would live
    assert got.holdout.n == 0  # an abstaining combiner scores nothing out of sample


def test_too_few_events_is_insufficient() -> None:
    got = tuner.tune(build, events(50), SPACE, study_name="small")
    assert isinstance(got, Insufficient) and got.need == tuner.MIN_TRAIN


def test_a_failing_build_fails_its_trials_not_the_call() -> None:
    def boom(_: dict[str, float]) -> Combiner:
        raise RuntimeError("no")

    got = tuner.tune(boom, events(300), SPACE, study_name="boom", n_trials=3)
    assert isinstance(got, tuner.Unavailable) and "3 failed" in got.reason


def test_an_unreachable_storage_is_unavailable() -> None:
    url = "sqlite:///Z:/no/such/dir/at/all/optuna.db"
    got = tuner.tune(build, events(300), SPACE, study_name="x", storage=url, n_trials=1)
    assert isinstance(got, tuner.Unavailable)


def test_a_reused_study_name_adds_trials() -> None:
    import optuna

    storage = optuna.storages.InMemoryStorage()
    evs = events(300)
    tuner.tune(build, evs, SPACE, study_name="again", storage=storage, n_trials=2)
    got = tuner.tune(build, evs, SPACE, study_name="again", storage=storage, n_trials=2)
    assert isinstance(got, tuner.Tuned) and got.trials == 2
    assert len(optuna.load_study(study_name="again", storage=storage).trials) == 4


def test_objective_brier_scores_an_abstention_as_a_coin() -> None:
    assert tuner.objective_brier([None] * 4, events(4)) == pytest.approx(0.25)
    assert tuner.objective_brier([], []) is None
