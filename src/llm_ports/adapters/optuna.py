"""Tune a combiner's hyperparameters with Optuna, walk-forward (extra `[optuna]`).

`tune(build, events, space)` splits the resolved events in time (`forecast.walk.holdout_start`):
the tuner only ever sees the TRAIN events, those forecast before the holdout starts. Each trial
builds a fresh combiner from its parameters (`build(params)`) and replays it over the train events
with `walk_forward`; its objective is the mean Brier score over the resolved train events, with
an abstention scored as 0.5 (Brier 0.25), so a configuration cannot win by answering less. The
best parameters are then replayed over EVERY event, learning through the holdout as they would
live, and scored on the holdout only: `Tuned.holdout` is out of sample.

The study goes to Optuna's `storage` (an RDB URL, an Optuna storage object, or None for memory)
under `study_name`; a reused name adds trials to that study. The TPE sampler is seeded.

`SqlJournal(engine, table)` is an Optuna journal backend on one SQL table the caller owns
(`journal_storage(...)` wraps it in Optuna's `JournalStorage`). Optuna's own RDB storage creates
its tables wherever the connection's search path points and checks its schema through Alembic,
which reads the default schema; behind a transaction-mode pooler (PgBouncer) a session-level
search path does not hold, so the RDB storage cannot be pinned to one schema there. A journal is
one append-only table: the caller creates it (a migration; `create=True` for tests and SQLite),
names it schema-qualified, and every statement carries the name.

Invariants:
    - Never raises: too few events is `Insufficient`; a storage or Optuna failure is
      `Unavailable`; a trial whose build or replay raises is pruned as failed, not fatal.
    - The objective never reads a holdout event.

Rationale: ADR-0015.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from llm_ports._extras import extra
from llm_ports.forecast.calibrate import Insufficient
from llm_ports.forecast.combine import Combiner
from llm_ports.forecast.score import Score, brier_score, score
from llm_ports.forecast.walk import Event, holdout_start, walk_forward

with extra("optuna"):
    import optuna
    import sqlalchemy as sa
    from optuna.storages import JournalStorage
    from optuna.storages.journal import BaseJournalBackend

#: Resolved train events below which `tune` refuses to search.
MIN_TRAIN = 100
#: The probability an abstention is scored at in the objective.
ABSTAIN_P = 0.5


@dataclass(frozen=True)
class Range:
    """One hyperparameter's search range.

    Attributes:
        low: The smallest value tried.
        high: The largest value tried.
        log: Sample on a log scale (a learning rate); `low` must then be > 0.
    """

    low: float
    high: float
    log: bool = False


@dataclass(frozen=True)
class Tuned:
    """The best parameters a study found, and how they scored out of sample.

    Attributes:
        params: `name -> value` of the best trial.
        study_name: The Optuna study the trials went to.
        trials: Trials completed in this call.
        failed_trials: Trials whose build or replay raised in this call.
        holdout_start: The first holdout event's forecast time.
        n_train: Resolved train events the objective scored.
        train_brier: The best trial's objective (mean Brier over the train events).
        holdout: The best parameters' `Score` on the resolved holdout events they answered.
    """

    params: dict[str, float]
    study_name: str
    trials: int
    failed_trials: int
    holdout_start: datetime
    n_train: int
    train_brier: float
    holdout: Score


@dataclass(frozen=True)
class Unavailable:
    """The study could not run (storage unreachable, every trial failed, an Optuna error).

    Attributes:
        reason: The exception's type and message, or why no trial completed.
    """

    reason: str


def objective_brier(preds: Sequence[float | None], events: Sequence[Event]) -> float | None:
    """Return the mean Brier score over the resolved events, an abstention scored as `ABSTAIN_P`; None when none resolved."""
    got = [
        brier_score(bool(e.resolution), ABSTAIN_P if p is None else p)
        for p, e in zip(preds, events, strict=True)
        if e.resolved
    ]
    return sum(got) / len(got) if got else None


def tune(
    build: Callable[[dict[str, float]], Combiner],
    events: Sequence[Event],
    space: Mapping[str, Range],
    *,
    study_name: str,
    storage: Any = None,
    n_trials: int = 40,
    holdout: float = 0.2,
    seed: int = 0,
    min_train: int = MIN_TRAIN,
) -> Tuned | Insufficient | Unavailable:
    """Search `space` for the combiner parameters with the lowest walk-forward Brier score.

    Args:
        build: Makes a fresh combiner from one trial's `name -> value`.
        events: Every event, resolved and open, in any order.
        space: `name -> Range` to search.
        study_name: The Optuna study name; trials join an existing study of that name.
        storage: An Optuna storage (URL or object); None keeps the study in memory.
        n_trials: Trials to run in this call.
        holdout: The share of the resolved events held out, in (0, 1).
        seed: The TPE sampler's seed.
        min_train: Resolved train events below which nothing is searched.

    Returns:
        `Tuned`; `Insufficient` when the train split has fewer than `min_train` resolved events;
        `Unavailable` when the study could not run. Never raises.
    """
    start = holdout_start(events, holdout)
    train = [e for e in events if start is not None and e.forecast_time < start]
    n_train = sum(1 for e in train if e.resolved)
    if start is None or n_train < min_train:
        return Insufficient(n=n_train, need=min_train, reason=f"{n_train} resolved train events < {min_train}")
    failed = 0

    def objective(trial: optuna.Trial) -> float:
        nonlocal failed
        params = {k: float(trial.suggest_float(k, r.low, r.high, log=r.log)) for k, r in space.items()}
        try:
            got = objective_brier(walk_forward(build(params), train), train)
        except Exception:  # a vendor combiner that raises fails its trial, never the study
            failed += 1
            raise optuna.TrialPruned() from None
        if got is None or not math.isfinite(got):
            failed += 1
            raise optuna.TrialPruned()
        return got

    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            load_if_exists=True,
        )
        before = len(study.trials)
        study.optimize(objective, n_trials=n_trials)
        done = [t for t in study.trials[before:] if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        if not done:
            return Unavailable(reason=f"no trial completed ({failed} failed)")
        best = min(done, key=lambda t: float(t.value or math.inf))
        params = {k: float(v) for k, v in best.params.items()}
        preds = walk_forward(build(params), events)
    except Exception as exc:  # storage, network, an Optuna error: a named outcome, never a raise
        return Unavailable(reason=f"{type(exc).__name__}: {exc}")
    keep = [i for i, e in enumerate(events) if e.resolved and e.forecast_time >= start and preds[i] is not None]
    return Tuned(
        params=params,
        study_name=study_name,
        trials=len(done),
        failed_trials=failed,
        holdout_start=start,
        n_train=n_train,
        train_brier=float(best.value or math.nan),
        holdout=score([bool(events[i].resolution) for i in keep], [p for i in keep if (p := preds[i]) is not None]),
    )


#: A journal table name: `table` or `schema.table`, identifiers only.
_TABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")
#: Appends retried when another process took the same log numbers first.
APPEND_RETRIES = 8


class SqlJournal(BaseJournalBackend):
    """An Optuna journal backend on one SQL table `(n bigint primary key, log text)`.

    Log `n` is the journal's 0-based log number. An append reads the next number and inserts the
    whole batch in one transaction; the primary key makes a concurrent append that read the same
    number fail, and it retries (`APPEND_RETRIES`) from the new end, so the journal stays gapless
    and in order across processes. Any SQLAlchemy dialect (PostgreSQL behind a pooler, SQLite).

    Args:
        engine: A synchronous SQLAlchemy engine.
        table: The table, `name` or `schema.name`.
        create: Create the table if it is missing (tests, SQLite); a production table comes from
            the caller's migration.

    Raises:
        ValueError: `table` is not one or two plain identifiers.
    """

    def __init__(self, engine: sa.engine.Engine, table: str, *, create: bool = False) -> None:
        if not _TABLE.match(table):
            raise ValueError(f"not a table name: {table!r}")
        self._engine = engine
        self._table = table
        if create:
            with engine.begin() as c:
                c.execute(sa.text(f"CREATE TABLE IF NOT EXISTS {table} (n bigint PRIMARY KEY, log text NOT NULL)"))

    def read_logs(self, log_number_from: int) -> Iterable[dict[str, Any]]:
        """Every log numbered `log_number_from` or later, in order."""
        with self._engine.connect() as c:
            rows = c.execute(
                sa.text(f"SELECT log FROM {self._table} WHERE n >= :n ORDER BY n"), {"n": int(log_number_from)}
            ).all()
        return [json.loads(r[0]) for r in rows]

    def append_logs(self, logs: list[dict[str, Any]]) -> None:
        """Append `logs` after the current end, atomically."""
        if not logs:
            return
        body = [json.dumps(log, separators=(",", ":")) for log in logs]
        for attempt in range(APPEND_RETRIES):
            try:
                with self._engine.begin() as c:
                    start = int(c.execute(sa.text(f"SELECT coalesce(max(n) + 1, 0) FROM {self._table}")).scalar_one())
                    c.execute(
                        sa.text(f"INSERT INTO {self._table} (n, log) VALUES (:n, :log)"),
                        [{"n": start + i, "log": b} for i, b in enumerate(body)],
                    )
                return
            except sa.exc.IntegrityError:
                if attempt == APPEND_RETRIES - 1:
                    raise


def journal_storage(engine: sa.engine.Engine, table: str, *, create: bool = False) -> Any:
    """Optuna's `JournalStorage` over a `SqlJournal`; pass it to `tune(storage=...)`."""
    return JournalStorage(SqlJournal(engine, table, create=create))


__all__ = [
    "ABSTAIN_P",
    "APPEND_RETRIES",
    "MIN_TRAIN",
    "Range",
    "SqlJournal",
    "Tuned",
    "Unavailable",
    "journal_storage",
    "objective_brier",
    "tune",
]
