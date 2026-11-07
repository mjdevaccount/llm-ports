"""Labelled model replies as Langfuse scores, with a Langfuse annotation queue per role (extra `[langfuse]`).

`LangfuseExampleStore` is a `learn.WritableExampleStore` and an `evals.labels.LabelQueue` for
topics `llm:<role>` (`evals.labels.topic`). An example is a score on the labelled generation,
named by source and role:

- `label:<role>`: a HUMAN label, given in Langfuse's annotation-queue UI against a categorical
  score config of the same name (`good` / `bad`). This adapter creates the config and the queue
  and never writes this name.
- `label.model:<role>`: a stronger model's label, written by `put`.
- `label.judge:<role>`: the production judge's verdict as a label, written by `put`.

Every role gets one score config and one annotation queue, both named `label:<role>`, created on
first `enqueue`. `put` writes each score under a deterministic id (source, role, subject), so a
relabel replaces the earlier score instead of adding a second one. The SDK is synchronous; each
call runs in a worker thread.

Example:
    store = LangfuseExampleStore(Langfuse(...))  # the caller builds the client
    await store.enqueue(topic("summarize"), [(observation_id, trace_id)])
    read = await store.examples([topic("summarize")])

Invariants:
    - Never raises for a server or SDK failure: the failure is the outcome.
    - A score this adapter cannot read as an example (another name, an unknown value, no
      observation subject) is skipped, never guessed.
    - A topic that is not `llm:<role>` holds no examples here and is refused by `put`/`enqueue`.

Rationale: ADR-0010.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Iterator, Sequence
from datetime import datetime
from typing import Any, Literal

from llm_ports._extras import extra
from llm_ports.evals.labels import role_of, topic
from llm_ports.learn import Example, ExamplesRead, Written
from llm_ports.learn._types import Source
from llm_ports.ports.outcomes import failure_reason

with extra("langfuse"):
    import langfuse  # noqa: F401  # the extra is what makes a Langfuse client buildable

#: The sources a Langfuse label score can carry.
Labeller = Literal["human", "model", "judge"]
#: The score name prefix of each source.
PREFIX: dict[Labeller, str] = {"human": "label", "model": "label.model", "judge": "label.judge"}
_BY_PREFIX: dict[str, Labeller] = {v: k for k, v in PREFIX.items()}
#: The categorical values a human picks, and the numeric value Langfuse books each under.
CATEGORIES: tuple[dict[str, str | float], ...] = ({"label": "good", "value": 1.0}, {"label": "bad", "value": 0.0})
#: Scores read per page.
PAGE = 100
_FIELDS = "details,subject,annotation"


def score_name(source: Labeller, role: str) -> str:
    """Return the score name a `source`'s label on a `role` reply is stored under."""
    return f"{PREFIX[source]}:{role}"


def parse_name(name: str) -> tuple[Labeller, str] | None:
    """Return `(source, role)` for a label score name, or None for any other score."""
    prefix, sep, role = name.partition(":")
    who = _BY_PREFIX.get(prefix)
    return (who, role) if sep and role and who else None


def _value(d: Any) -> str | None:
    v = getattr(d, "value", None)
    if isinstance(v, str):
        return v if v in ("good", "bad") else None
    if isinstance(v, bool | int | float):
        return "good" if float(v) >= 0.5 else "bad"
    return None


def example_of_score(d: Any) -> Example | None:
    """Return a v3 score object as an `Example`, or None when it is not a label score this adapter reads."""
    parsed = parse_name(getattr(d, "name", "") or "")
    subject = getattr(d, "subject", None)
    value = _value(d)
    if parsed is None or value is None or subject is None or getattr(subject, "kind", None) != "observation":
        return None
    who, role = parsed
    meta = getattr(d, "metadata", None) or {}
    trace = str(getattr(subject, "trace_id", "") or "")
    return Example(
        topic=topic(role),
        subject=str(subject.id),
        value=value,
        source=who,
        by=str(meta.get("by") or getattr(d, "author_user_id", "") or ""),
        at=getattr(d, "timestamp", None),
        reason=str(getattr(d, "comment", "") or ""),
        ref={"trace": trace} if trace else {},
    )


def _refusal(examples: Sequence[Example]) -> str | None:
    """Return why `put` refuses the batch, or None when every example can be written."""
    for ex in examples:
        if ex.source not in ("model", "judge"):
            return f"a {ex.source} label is not written by code (a human labels in the annotation UI)"
        if role_of(ex.topic) is None:
            return f"topic {ex.topic!r} is not a model reply's (llm:<role>)"
        if ex.value not in ("good", "bad"):
            return f"label {ex.value!r} is not good or bad"
        if not ex.ref.get("trace"):
            return "an example needs ref['trace']: Langfuse scopes observations by trace"
    return None


class LangfuseExampleStore:
    """Labelled model replies stored as Langfuse scores, queued for humans in annotation queues.

    Args:
        client: A Langfuse SDK 5 client, built by the caller.
        description: The description given to each created score config and queue.
    """

    def __init__(self, client: Any, *, description: str = "Is this reply good? (labelled data)") -> None:
        self._c = client
        self._description = description
        self._configs: dict[str, str] = {}
        self._queues: dict[str, str] = {}

    def _config_id(self, role: str) -> str:
        name = score_name("human", role)
        page = 1
        while name not in self._configs:
            got = self._c.api.score_configs.get(page=page, limit=PAGE)
            for cfg in got.data:
                if cfg.name == name and not getattr(cfg, "is_archived", False):
                    self._configs[name] = cfg.id
            if len(got.data) < PAGE:
                break
            page += 1
        if name not in self._configs:
            cfg = self._c.api.score_configs.create(
                name=name, data_type="CATEGORICAL", categories=list(CATEGORIES), description=self._description
            )
            self._configs[name] = cfg.id
        return self._configs[name]

    def _queue_id(self, role: str) -> str:
        name = score_name("human", role)
        page = 1
        while name not in self._queues:
            got = self._c.api.annotation_queues.list_queues(page=page, limit=PAGE)
            for q in got.data:
                if q.name == name:
                    self._queues[name] = q.id
            if len(got.data) < PAGE:
                break
            page += 1
        if name not in self._queues:
            q = self._c.api.annotation_queues.create_queue(
                name=name, score_config_ids=[self._config_id(role)], description=self._description
            )
            self._queues[name] = q.id
        return self._queues[name]

    def _enqueue(self, role: str, subjects: Sequence[tuple[str, str]], done: list[int]) -> None:
        queue = self._queue_id(role)
        for subject, _trace in subjects:
            self._c.api.annotation_queues.create_queue_item(queue, object_id=subject, object_type="OBSERVATION")
            done[0] += 1

    async def enqueue(self, topic: str, subjects: Sequence[tuple[str, str]]) -> Written:
        """Add each `(observation id, trace id)` to the role's annotation queue. Never raises."""
        role = role_of(topic)
        if role is None:
            return Written(0, "request_fault", f"topic {topic!r} is not a model reply's (llm:<role>)")
        done = [0]
        try:
            await asyncio.to_thread(self._enqueue, role, subjects, done)
        except Exception as exc:  # the store's failure is the outcome
            return Written(done[0], "unavailable", failure_reason(exc))
        return Written(done[0], "ok")

    def _put(self, examples: Sequence[Example], done: list[int]) -> None:
        for ex in examples:
            role = role_of(ex.topic) or ""
            name = score_name("judge" if ex.source == "judge" else "model", role)
            self._c.create_score(
                name=name,
                value=str(ex.value),
                data_type="CATEGORICAL",
                trace_id=ex.ref["trace"],
                observation_id=ex.subject,
                score_id=hashlib.sha256(f"{name}\n{ex.subject}".encode()).hexdigest(),
                comment=ex.reason or None,
                metadata={"by": ex.by, "labeller": ex.source},
                timestamp=ex.at,
            )
            done[0] += 1
        self._c.flush()

    async def put(self, examples: Sequence[Example]) -> Written:
        """Write `model` and `judge` labels as categorical scores. Never raises.

        Returns:
            How many landed; `request_fault` (none written) when any example is a `human` or
            `outcome` label, not on an `llm:<role>` topic, not `good`/`bad`, or has no
            `ref["trace"]`.
        """
        why = _refusal(examples)
        if why is not None:
            return Written(0, "request_fault", why)
        done = [0]
        try:
            await asyncio.to_thread(self._put, examples, done)
        except Exception as exc:  # the store's failure is the outcome
            return Written(done[0], "unavailable", failure_reason(exc))
        return Written(done[0], "ok")

    def _pages(self, names: str, since: datetime | None) -> Iterator[Any]:
        cursor: str | None = None
        while True:
            got = self._c.api.scores.get_many(
                name=names, from_timestamp=since, fields=_FIELDS, limit=PAGE, cursor=cursor
            )
            yield from got.data
            cursor = getattr(got.meta, "cursor", None)
            if not cursor or not got.data:
                return

    def _read(self, names: str, since: datetime | None, source: Source | None) -> tuple[Example, ...]:
        got = (ex for d in self._pages(names, since) if (ex := example_of_score(d)) is not None)
        return tuple(ex for ex in got if source is None or ex.source == source)

    async def examples(
        self, topics: Sequence[str], *, since: datetime | None = None, source: Source | None = None
    ) -> ExamplesRead:
        """Read every source's labels on the replies of `topics`. Never raises.

        A topic that is not `llm:<role>` contributes nothing; an `outcome` source reads empty.
        """
        roles = [r for t in topics if (r := role_of(t)) is not None]
        names = ",".join(score_name(who, r) for r in roles for who in PREFIX if source is None or who == source)
        if not names:
            return ExamplesRead((), "ok")
        try:
            got = await asyncio.to_thread(self._read, names, since, source)
        except Exception as exc:  # an unreadable store is never an empty one
            return ExamplesRead((), "unavailable", failure_reason(exc))
        return ExamplesRead(got, "ok")


__all__ = ["CATEGORIES", "PREFIX", "Labeller", "LangfuseExampleStore", "example_of_score", "parse_name", "score_name"]
