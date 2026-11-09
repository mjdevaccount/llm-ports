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


# ---- SqlJournal: the journal backend on one SQL table -------------------------------------------

sa = pytest.importorskip("sqlalchemy")


def engine(tmp_path) -> sa.engine.Engine:
    return sa.create_engine(f"sqlite:///{tmp_path / 'j.db'}")


def test_journal_appends_in_order_and_reads_from_a_log_number(tmp_path) -> None:
    j = tuner.SqlJournal(engine(tmp_path), "journal", create=True)
    j.append_logs([{"op": 0}, {"op": 1}])
    j.append_logs([])
    j.append_logs([{"op": 2}])
    assert [r["op"] for r in j.read_logs(0)] == [0, 1, 2]
    assert [r["op"] for r in j.read_logs(2)] == [2]
    assert list(j.read_logs(3)) == []


def test_two_journals_on_one_table_share_one_gapless_log(tmp_path) -> None:
    eng = engine(tmp_path)
    a, b = tuner.SqlJournal(eng, "journal", create=True), tuner.SqlJournal(eng, "journal")
    a.append_logs([{"w": "a"}])
    b.append_logs([{"w": "b"}, {"w": "b"}])
    a.append_logs([{"w": "a"}])
    with eng.connect() as c:
        assert [r[0] for r in c.execute(sa.text("SELECT n FROM journal ORDER BY n"))] == [0, 1, 2, 3]
    assert [r["w"] for r in b.read_logs(0)] == ["a", "b", "b", "a"]


def test_an_append_that_lost_the_race_retries_from_the_new_end(tmp_path) -> None:
    eng = engine(tmp_path)
    j = tuner.SqlJournal(eng, "journal", create=True)
    raced = []

    def race(*args: object) -> None:  # another writer lands between this append's read and insert
        if str(args[2]).startswith("INSERT INTO journal") and not raced:
            raced.append(1)
            with eng.begin() as other:
                other.execute(sa.text('INSERT INTO journal (n, log) VALUES (0, \'{"w": "other"}\')'))

    sa.event.listen(eng, "before_cursor_execute", race)
    j.append_logs([{"w": "me"}])
    assert raced == [1]
    assert [r["w"] for r in j.read_logs(0)] == ["other", "me"]


def test_a_table_name_that_is_not_an_identifier_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="not a table name"):
        tuner.SqlJournal(engine(tmp_path), "journal; DROP TABLE x")


def test_tune_on_a_journal_resumes_the_study(tmp_path) -> None:
    storage = tuner.journal_storage(engine(tmp_path), "journal", create=True)
    first = tuner.tune(build, events(400), SPACE, study_name="j", storage=storage, n_trials=3, seed=1)
    again = tuner.tune(
        build,
        events(400),
        SPACE,
        study_name="j",
        storage=tuner.journal_storage(engine(tmp_path), "journal"),
        n_trials=2,
        seed=1,
    )
    assert isinstance(first, tuner.Tuned) and isinstance(again, tuner.Tuned)
    import optuna

    assert (
        len(optuna.load_study(study_name="j", storage=tuner.journal_storage(engine(tmp_path), "journal")).trials) == 5
    )


def _get(app, path: str) -> tuple[str, bytes]:
    """Call a WSGI app with one GET and return its status and body."""
    from wsgiref.util import setup_testing_defaults

    environ: dict = {"PATH_INFO": path, "REQUEST_METHOD": "GET"}
    setup_testing_defaults(environ)
    seen: list[str] = []
    body = b"".join(app(environ, lambda status, headers, *_: seen.append(status)))
    return seen[0], body


def test_the_dashboard_reads_a_journal_table_live(tmp_path) -> None:
    pytest.importorskip("optuna_dashboard")
    storage = tuner.journal_storage(engine(tmp_path), "journal", create=True)
    tuner.tune(build, events(400), SPACE, study_name="dash", storage=storage, n_trials=2, seed=1)
    status, body = _get(tuner.dashboard_app(tuner.journal_storage(engine(tmp_path), "journal")), "/api/studies")
    assert status.startswith("200")
    assert b'"dash"' in body


def test_serve_dashboard_binds_loopback_by_default(monkeypatch) -> None:
    optuna_dashboard = pytest.importorskip("optuna_dashboard")
    seen: dict = {}
    monkeypatch.setattr(optuna_dashboard, "run_server", lambda storage, host, port: seen.update(host=host, port=port))
    tuner.serve_dashboard("sqlite://")
    assert seen == {"host": "127.0.0.1", "port": 8080}
