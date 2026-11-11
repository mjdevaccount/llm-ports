"""Tune a prompt's text components with GEPA, reflective prompt evolution (extra `[gepa]`).

`optimize(seed, train, val, evaluate=..., reflect=...)` runs the `gepa` core engine over a
candidate: a mapping of named text components (an instruction block, a system prompt) to their
text. The caller owns everything domain-specific through two callables:

- `evaluate(candidate, items)` runs the program with the candidate's texts on each item and
  returns one `Rollout` per item: the output, a score in [0, 1] (higher is better) and the
  feedback a reflection model should read (grader reasons, validation problems).
- `reflect(prompt)` is the reflection model: prompt text in, completion text out.

GEPA evaluates candidates on minibatches of `train`, asks `reflect` to rewrite one component from
the rollouts' feedback, keeps a rewrite only when it beats its parent on the same minibatch, and
tracks every kept candidate's mean score on `val`. `Optimized` reports the best candidate by that
mean next to the seed's own, so a caller promotes only when `improved` is true and `val` was held
out from anything the caller tuned by hand.

Invariants:
    - Never raises: too few items is `Insufficient`; an `evaluate` that raises, a rollout count
      that does not match its batch, a GEPA engine error, a `reflect` that failed on every
      call, or one that failed `max_reflection_failures` times in a row is `Unavailable`.
    - A reflection failure never passes silently: GEPA itself logs a failed reflection and moves
      on, so `optimize` counts every `reflect` call and failure and reports both on `Optimized`.
    - GEPA's own log lines are kept (`Optimized.log_tail`, `Unavailable.log_tail`), never printed.
    - A non-finite score is scored 0.0 and its feedback says so; a score outside [0, 1] is clamped.
    - The seed candidate is never mutated; `Optimized.candidate` is a fresh dict.
    - `max_metric_calls` bounds the item evaluations GEPA spends (its own budget accounting).
    - The search always ends without the budget: GEPA swallows a failed reflection and an
      evaluator that caches spends no budget, so a dead reflection model would otherwise spin.
      `max_reflection_failures` consecutive failures stop it as `Unavailable`; `timeout_s`, when
      set, stops it on wall-clock time and `Optimized.stopped` says so.

Rationale: ADR-0017.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from llm_ports._extras import extra
from llm_ports.forecast.calibrate import Insufficient

with extra("gepa"):
    from gepa.api import optimize as gepa_optimize
    from gepa.core.adapter import EvaluationBatch, ProposalFn
    from gepa.core.result import GEPAResult
    from gepa.utils.stop_condition import StopperProtocol, TimeoutStopCondition

#: Train items below which `optimize` refuses to search.
MIN_TRAIN = 8
#: Validation items below which `optimize` refuses to search.
MIN_VAL = 4
#: Item evaluations GEPA may spend when the caller sets no budget.
DEFAULT_METRIC_CALLS = 200
#: Consecutive failed reflections after which `optimize` stops the search.
MAX_REFLECTION_FAILURES = 5


@dataclass(frozen=True)
class Rollout[O]:
    """One item run with one candidate, and how it scored.

    Attributes:
        output: What the program produced (opaque to GEPA; shown to the reflection model).
        score: In [0, 1], higher is better. GEPA sums it on minibatches and averages it on `val`.
        feedback: Why the item scored as it did, in words the reflection model can act on.
    """

    output: O
    score: float
    feedback: str = ""


@dataclass(frozen=True)
class Optimized:
    """The best candidate GEPA found, scored on the validation items.

    Attributes:
        candidate: `component -> text` of the best candidate by mean validation score.
        seed_score: The seed candidate's mean validation score.
        best_score: The best candidate's mean validation score.
        improved: `best_score > seed_score`; a caller promotes only when this is true.
        candidates: Candidates GEPA kept (the seed included).
        metric_calls: Item evaluations spent.
        n_train: Train items given.
        n_val: Validation items given.
        reflections: `reflect` calls made.
        reflection_failures: `reflect` calls that raised (GEPA skipped those steps).
        log_tail: GEPA's last log lines.
        stopped: Why the search ended: `"budget"` (`max_metric_calls` spent) or `"timeout"`.
    """

    candidate: dict[str, str]
    seed_score: float
    best_score: float
    improved: bool
    candidates: int
    metric_calls: int
    n_train: int
    n_val: int
    reflections: int
    reflection_failures: int
    log_tail: tuple[str, ...] = ()
    stopped: str = "budget"


@dataclass(frozen=True)
class Unavailable:
    """The search could not run or finish (an evaluator, reflection or engine failure).

    Attributes:
        reason: The exception's type and message, or why no reflection succeeded.
        log_tail: GEPA's last log lines.
    """

    reason: str
    log_tail: tuple[str, ...] = ()


#: GEPA log lines an outcome keeps.
LOG_TAIL = 20


class _Log:
    """A GEPA logger that keeps its last lines instead of printing them."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, message: str) -> None:
        self.lines.append(str(message))
        del self.lines[:-LOG_TAIL]


class _Reflect:
    """Counts the caller's reflection calls and failures, which GEPA itself only logs."""

    def __init__(self, reflect: Callable[[str], str]) -> None:
        self._reflect = reflect
        self.calls = 0
        self.failures = 0
        self.consecutive = 0
        self.last_error = ""

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        self.calls += 1
        try:
            text = self._reflect(_as_text(prompt))
        except Exception as exc:
            self.failures += 1
            self.consecutive += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise
        self.consecutive = 0
        return text


class _DeadReflector:
    """A GEPA stopper: True once `limit` reflections in a row have failed.

    GEPA logs a failed reflection and starts the next iteration, so without this a reflection
    model that is down (a spent daily quota) is called until the metric budget runs out, which
    never happens when the caller's evaluator caches.
    """

    def __init__(self, reflector: _Reflect, limit: int) -> None:
        self._reflector = reflector
        self._limit = limit
        self.fired = False

    def __call__(self, gepa_state: Any) -> bool:
        if self._reflector.consecutive >= self._limit:
            self.fired = True
        return self.fired


class _Timeout:
    """GEPA's wall-clock stopper, remembering that it fired."""

    def __init__(self, seconds: float) -> None:
        self._inner = TimeoutStopCondition(seconds)
        self.fired = False

    def __call__(self, gepa_state: Any) -> bool:
        if self._inner(gepa_state):
            self.fired = True
        return self.fired


def _clean(score: float) -> tuple[float, str]:
    """Return a GEPA-safe score and a note when it had to change."""
    if not math.isfinite(score):
        return 0.0, f"score {score!r} is not finite, scored 0.0"
    if score < 0.0 or score > 1.0:
        return min(1.0, max(0.0, score)), f"score {score!r} clamped to [0, 1]"
    return score, ""


class RolloutAdapter[I, O]:
    """A GEPA adapter over a caller's batch evaluator.

    Implements GEPA's `GEPAAdapter` protocol: `evaluate` runs the caller's evaluator and returns an
    `EvaluationBatch`; `make_reflective_dataset` turns each rollout into the record GEPA's
    default proposer shows the reflection model (`Inputs`, `Generated Outputs`, `Feedback`).

    Args:
        evaluate: `(candidate, items) -> one Rollout per item`.
        describe: Renders an item for the reflection model; `str` by default.

    Raises:
        ValueError: From `evaluate`, when the evaluator returns a rollout count that does not
            match its batch (GEPA treats it as a systemic failure; `optimize` names it).
    """

    #: No custom proposer: GEPA's default reflective proposer writes the new texts.
    propose_new_texts: ProposalFn | None = None

    def __init__(
        self,
        evaluate: Callable[[Mapping[str, str], Sequence[I]], Sequence[Rollout[O]]],
        describe: Callable[[I], str] = str,
    ) -> None:
        self._evaluate = evaluate
        self._describe = describe

    def evaluate(
        self, batch: list[I], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvaluationBatch[dict[str, Any], O]:
        """Run `candidate` on `batch` through the caller's evaluator."""
        rollouts = list(self._evaluate(dict(candidate), list(batch)))
        if len(rollouts) != len(batch):
            raise ValueError(f"evaluator returned {len(rollouts)} rollouts for {len(batch)} items")
        scores: list[float] = []
        traces: list[dict[str, Any]] = []
        for item, r in zip(batch, rollouts, strict=True):
            s, note = _clean(float(r.score))
            scores.append(s)
            feedback = f"{r.feedback} ({note})" if note and r.feedback else note or r.feedback
            traces.append({"item": item, "output": r.output, "score": s, "feedback": feedback})
        return EvaluationBatch(
            outputs=[r.output for r in rollouts],
            scores=scores,
            trajectories=traces if capture_traces else None,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[dict[str, Any], O],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Return the per-component records the reflection model reads."""
        records = [
            {
                "Inputs": self._describe(t["item"]),
                "Generated Outputs": str(t["output"]),
                "Feedback": t["feedback"] or f"score {t['score']:.3f}",
            }
            for t in eval_batch.trajectories or []
        ]
        return {name: list(records) for name in components_to_update}


def _as_text(prompt: str | list[dict[str, Any]]) -> str:
    """Flatten a chat-message list to one prompt string (GEPA passes either)."""
    if isinstance(prompt, str):
        return prompt
    return "\n\n".join(str(m.get("content", "")) for m in prompt)


def optimize[I, O](
    seed: Mapping[str, str],
    train: Sequence[I],
    val: Sequence[I],
    *,
    evaluate: Callable[[Mapping[str, str], Sequence[I]], Sequence[Rollout[O]]],
    reflect: Callable[[str], str],
    describe: Callable[[I], str] = str,
    max_metric_calls: int = DEFAULT_METRIC_CALLS,
    minibatch_size: int = 3,
    seed_value: int = 0,
    run_dir: str | None = None,
    min_train: int = MIN_TRAIN,
    min_val: int = MIN_VAL,
    max_reflection_failures: int = MAX_REFLECTION_FAILURES,
    timeout_s: float | None = None,
) -> Optimized | Insufficient | Unavailable:
    """Evolve `seed`'s text components with GEPA and return the best by validation score.

    Args:
        seed: `component -> text` to start from; every component may be rewritten.
        train: Items GEPA's minibatches and reflections draw from.
        val: Held-out items every kept candidate is scored on; disjoint from `train`.
        evaluate: `(candidate, items) -> one Rollout per item`; score per-item failures 0.0 with
            feedback rather than raising (a raise ends the search as `Unavailable`).
        reflect: The reflection model, prompt text in, completion text out.
        describe: Renders an item for the reflection model.
        max_metric_calls: The item-evaluation budget.
        minibatch_size: Train items per reflection step.
        seed_value: GEPA's random seed (minibatch sampling, component order).
        run_dir: A directory GEPA checkpoints to, so a stopped run resumes; None keeps no state.
        min_train: Train items below which nothing is searched.
        min_val: Validation items below which nothing is searched.
        max_reflection_failures: Consecutive failed `reflect` calls that end the search as
            `Unavailable` (a reflection model that is down or out of quota).
        timeout_s: Wall-clock seconds after which the search stops and reports what it has
            (`Optimized.stopped == "timeout"`); None sets no time bound.

    Returns:
        `Optimized`; `Insufficient` when `train` or `val` is under its floor or `seed` has no
        component; `Unavailable` when the search failed or the reflection model kept failing.
        Never raises.
    """
    if not seed:
        return Insufficient(n=0, need=1, reason="seed has no component to tune")
    if len(train) < min_train:
        return Insufficient(n=len(train), need=min_train, reason=f"{len(train)} train items < {min_train}")
    if len(val) < min_val:
        return Insufficient(n=len(val), need=min_val, reason=f"{len(val)} validation items < {min_val}")
    log, reflector = _Log(), _Reflect(reflect)
    dead = _DeadReflector(reflector, max(1, max_reflection_failures))
    timeout = _Timeout(timeout_s) if timeout_s is not None else None
    stoppers: list[StopperProtocol] = [dead] if timeout is None else [dead, timeout]
    try:
        result: GEPAResult[O, Any] = gepa_optimize(
            seed_candidate=dict(seed),
            trainset=list(train),
            valset=list(val),
            adapter=RolloutAdapter(evaluate, describe),
            reflection_lm=reflector,
            reflection_minibatch_size=minibatch_size,
            max_metric_calls=max_metric_calls,
            seed=seed_value,
            run_dir=run_dir,
            stop_callbacks=stoppers,
            raise_on_exception=True,
            display_progress_bar=False,
            logger=log,
        )
        scores = [float(s) for s in result.val_aggregate_scores]
        best = max(range(len(scores)), key=scores.__getitem__)
    except Exception as exc:  # evaluator, reflection or engine failure: a named outcome
        return Unavailable(reason=f"{type(exc).__name__}: {exc}", log_tail=tuple(log.lines))
    if dead.fired:
        return Unavailable(
            reason=f"{reflector.consecutive} reflections failed in a row: {reflector.last_error}",
            log_tail=tuple(log.lines),
        )
    if reflector.calls and reflector.failures == reflector.calls:
        return Unavailable(
            reason=f"every reflection failed ({reflector.failures}): {reflector.last_error}",
            log_tail=tuple(log.lines),
        )
    return Optimized(
        candidate=dict(result.candidates[best]),
        seed_score=scores[0],
        best_score=scores[best],
        improved=scores[best] > scores[0],
        candidates=len(scores),
        metric_calls=int(result.total_metric_calls or 0),  # zero-ok: a search-budget count, not usage or cost
        n_train=len(train),
        n_val=len(val),
        reflections=reflector.calls,
        reflection_failures=reflector.failures,
        log_tail=tuple(log.lines),
        stopped="timeout" if timeout is not None and timeout.fired else "budget",
    )


__all__ = [
    "DEFAULT_METRIC_CALLS",
    "LOG_TAIL",
    "MAX_REFLECTION_FAILURES",
    "MIN_TRAIN",
    "MIN_VAL",
    "Optimized",
    "Rollout",
    "RolloutAdapter",
    "Unavailable",
    "optimize",
]
