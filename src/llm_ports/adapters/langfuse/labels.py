"""Labelled model replies as Langfuse scores, with a Langfuse annotation queue per role (extra `[langfuse]`).

`LangfuseExampleStore` is a `learn.WritableExampleStore` and an `evals.labels.LabelQueue` for
topics `llm:<role>` (`evals.labels.topic`). An example is a score on the labelled generation,
named by source and role:

- `label-<role>`: a HUMAN label against a categorical score config of the same name (`good` /
  `bad`), given in Langfuse's annotation-queue UI or relayed by `put` from another surface that
  names the person (`Example.by`). A relayed label completes that reply's queue item, as the
  annotation UI does.
- `label.model-<role>`: a stronger model's label, written by `put`.
- `label.judge-<role>`: the production judge's verdict as a label, written by `put`.

The separator is `-` because Langfuse refuses a score-config name outside `CONFIG_NAME` (no `:`);
reads also accept the earlier `label:<role>` spelling, so scores written under it stay readable.
Every role gets one score config and one annotation queue, both named `label-<role>`, created on
first `enqueue`. `enqueue` skips a reply the queue already holds (pending or done), and `queued`
reads what it holds, so a nightly draw neither duplicates items nor grows the backlog unbounded.
`put` writes each score under a deterministic id (source, role, subject, and the person for a
human label), so a relabel replaces the earlier score instead of adding a second one. The SDK is
synchronous; each call runs in a worker thread.

Example:
    store = LangfuseExampleStore(Langfuse(...))  # the caller builds the client
    await store.enqueue(topic("summarize"), [(observation_id, trace_id)])
    read = await store.examples([topic("summarize")])

Invariants:
    - Never raises for a server or SDK failure: the failure is the outcome.
    - A score this adapter cannot read as an example (another name, an unknown value, no
      observation subject) is skipped, never guessed.
    - A topic that is not `llm:<role>` holds no examples here and is refused by `put`/`enqueue`.
    - A role whose label name Langfuse would refuse (`CONFIG_NAME`) is refused here as a
      `request_fault` before any write, never sent to fail remotely.
    - A human label is written only with the person who gave it (`by`); an `outcome` label never.
    - A failure reason carries the server's own validation message when it sends one.

Rationale: ADR-0010, ADR-0011, ADR-0014.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Iterator, Sequence
from datetime import datetime
from typing import Any, Literal, cast

from llm_ports._extras import extra
from llm_ports.evals.labels import Queued, role_of, topic
from llm_ports.learn import Example, ExamplesRead, Written
from llm_ports.learn._types import Source
from llm_ports.ports.outcomes import REASON_MAX, failure_reason

with extra("langfuse"):
    import langfuse  # noqa: F401  # the extra is what makes a Langfuse client buildable

#: The sources a Langfuse label score can carry.
Labeller = Literal["human", "model", "judge"]
#: The score name prefix of each source.
PREFIX: dict[Labeller, str] = {"human": "label", "model": "label.model", "judge": "label.judge"}
_BY_PREFIX: dict[str, Labeller] = {v: k for k, v in PREFIX.items()}
#: Between a label's source prefix and its role.
SEP = "-"
_LEGACY_SEP = ":"
#: The characters Langfuse accepts in a score-config name: letters, digits, `_`, space, `.`, `(`, `)`, `-`.
CONFIG_NAME = re.compile(r"[\w .()-]+")
#: The longest score-config name Langfuse accepts, counted as the server counts (UTF-16 code units).
CONFIG_NAME_MAX = 80
#: Score names per read request: the names travel in the query string, so a read is split to bound it.
NAMES_PER_READ = 40
#: The categorical values a human picks, and the numeric value Langfuse books each under.
CATEGORIES: tuple[dict[str, str | float], ...] = ({"label": "good", "value": 1.0}, {"label": "bad", "value": 0.0})
#: Scores read per page.
PAGE = 100
#: The status Langfuse gives a queue item once its reply is labelled.
COMPLETED = "COMPLETED"
_FIELDS = "details,subject,annotation"


def score_name(source: Labeller, role: str) -> str:
    """Return the score name a `source`'s label on a `role` reply is stored under."""
    return f"{PREFIX[source]}{SEP}{role}"


def _names(source: Labeller, role: str) -> tuple[str, str]:
    """Return the current and the earlier (`:`) spelling of a label score name."""
    return score_name(source, role), f"{PREFIX[source]}{_LEGACY_SEP}{role}"


def parse_name(name: str) -> tuple[Labeller, str] | None:
    """Return `(source, role)` for a label score name in either spelling, or None for any other score."""
    for prefix in sorted(_BY_PREFIX, key=len, reverse=True):
        for sep in (SEP, _LEGACY_SEP):
            head = prefix + sep
            if name.startswith(head) and len(name) > len(head):
                return _BY_PREFIX[prefix], name[len(head) :]
    return None


def _bad_name(role: str) -> str | None:
    """Return why Langfuse would refuse `role`'s label names, or None when every source's name is valid."""
    for who in PREFIX:
        name = score_name(who, role)
        if CONFIG_NAME.fullmatch(name) is None or len(name.encode("utf-16-le")) // 2 > CONFIG_NAME_MAX:
            return f"label name {name!r} is not a Langfuse score-config name (letters, digits, _ .()- , 1-80)"
    return None


def _reason(exc: BaseException) -> str:
    """Return `failure_reason(exc)` plus the first line of the server's validation message, cut to `REASON_MAX`."""
    base = failure_reason(exc)
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return base
    errors = body.get("error")
    first = errors[0] if isinstance(errors, list) and errors and isinstance(errors[0], dict) else {}
    msg = first.get("message") or body.get("message")
    if not msg:
        return base
    line = str(msg).strip().splitlines()[0] if str(msg).strip() else ""
    return f"{base}: {line}"[:REASON_MAX] if line else base


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
        if ex.source not in PREFIX:
            return f"a {ex.source} label is not written by code (it is read from where it happened)"
        if ex.source == "human" and not ex.by.strip():
            return "a human label needs `by`, the person who gave it: code only relays it"
        role = role_of(ex.topic)
        if role is None:
            return f"topic {ex.topic!r} is not a model reply's (llm:<role>)"
        if (bad := _bad_name(role)) is not None:
            return bad
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

    def _queue_id(self, role: str, *, create: bool = True) -> str | None:
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
            if not create:
                return None
            q = self._c.api.annotation_queues.create_queue(
                name=name, score_config_ids=[self._config_id(role)], description=self._description
            )
            self._queues[name] = q.id
        return self._queues[name]

    def _items(self, queue: str) -> Iterator[Any]:
        page = 1
        while True:
            got = self._c.api.annotation_queues.list_queue_items(queue, page=page, limit=PAGE)
            yield from got.data
            if len(got.data) < PAGE:
                return
            page += 1

    def _queued(self, role: str) -> Queued:
        queue = self._queue_id(role, create=False)
        if queue is None:
            return Queued()
        pending: set[str] = set()
        done: set[str] = set()
        for item in self._items(queue):
            status = str(getattr(item, "status", "") or "").upper()
            (done if status == "COMPLETED" else pending).add(str(item.object_id))
        return Queued(frozenset(pending), frozenset(done))

    def _enqueue(self, role: str, subjects: Sequence[tuple[str, str]], done: list[int]) -> None:
        queue = self._queue_id(role)
        if queue is None:  # unreachable: create=True finds or makes the queue
            return
        held = {str(item.object_id) for item in self._items(queue)}
        for subject, _trace in subjects:
            if subject in held:
                continue
            self._c.api.annotation_queues.create_queue_item(queue, object_id=subject, object_type="OBSERVATION")
            held.add(subject)
            done[0] += 1

    async def enqueue(self, topic: str, subjects: Sequence[tuple[str, str]]) -> Written:
        """Add each `(observation id, trace id)` the role's annotation queue does not hold yet.

        Returns:
            How many were newly queued, or a failure outcome. Never raises.
        """
        role = role_of(topic)
        if role is None:
            return Written(0, "request_fault", f"topic {topic!r} is not a model reply's (llm:<role>)")
        if (bad := _bad_name(role)) is not None:
            return Written(0, "request_fault", bad)
        done = [0]
        try:
            await asyncio.to_thread(self._enqueue, role, subjects, done)
        except Exception as exc:  # the store's failure is the outcome
            return Written(done[0], "unavailable", _reason(exc))
        return Written(done[0], "ok")

    async def queued(self, topic: str) -> Queued:
        """Read the observation ids the role's annotation queue holds, pending and done. Never raises.

        A role with no queue yet reads `ok` and empty; nothing is created by reading.
        """
        role = role_of(topic)
        if role is None:
            return Queued(outcome="request_fault", reason=f"topic {topic!r} is not a model reply's (llm:<role>)")
        try:
            return await asyncio.to_thread(self._queued, role)
        except Exception as exc:  # an unreadable queue is never an empty one
            return Queued(outcome="unavailable", reason=_reason(exc))

    def _complete(self, role: str, subjects: set[str]) -> None:
        """Mark the role's queue items for `subjects` done; a role with no queue has none to mark."""
        queue = self._queue_id(role, create=False)
        if queue is None:
            return
        for item in list(self._items(queue)):
            status = str(getattr(item, "status", "") or "").upper()
            if str(item.object_id) in subjects and status != COMPLETED:
                self._c.api.annotation_queues.update_queue_item(queue, item.id, status=COMPLETED)

    def _put(self, examples: Sequence[Example], done: list[int]) -> None:
        relayed: dict[str, set[str]] = {}
        for ex in examples:
            role = role_of(ex.topic) or ""
            who = cast("Labeller", ex.source)
            name = score_name(who, role)
            key = f"{name}\n{ex.subject}" + (f"\n{ex.by}" if who == "human" else "")
            self._c.create_score(
                name=name,
                value=str(ex.value),
                data_type="CATEGORICAL",
                trace_id=ex.ref["trace"],
                observation_id=ex.subject,
                score_id=hashlib.sha256(key.encode()).hexdigest(),
                comment=ex.reason or None,
                config_id=self._config_id(role) if who == "human" else None,
                metadata={"by": ex.by, "labeller": ex.source},
                timestamp=ex.at,
            )
            done[0] += 1
            if who == "human":
                relayed.setdefault(role, set()).add(ex.subject)
        self._c.flush()
        for role, subjects in relayed.items():
            self._complete(role, subjects)

    async def put(self, examples: Sequence[Example]) -> Written:
        """Write labels as categorical scores. Never raises.

        A `human` label is one a person gave on another surface and code relays: it is written
        under `label-<role>` against the role's score config (created when missing), keyed by the
        person as well, and the role's queue item for that reply is then marked completed.

        Returns:
            How many scores landed; `request_fault` (none written) when any example is an
            `outcome` label, a `human` label with no `by`, not on an `llm:<role>` topic, not
            `good`/`bad`, or has no `ref["trace"]`. `unavailable` when the server failed; a
            failure while completing a queue item comes after every score landed, so `written`
            then counts them all.
        """
        why = _refusal(examples)
        if why is not None:
            return Written(0, "request_fault", why)
        done = [0]
        try:
            await asyncio.to_thread(self._put, examples, done)
        except Exception as exc:  # the store's failure is the outcome
            return Written(done[0], "unavailable", _reason(exc))
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

    def _read(self, names: Sequence[str], since: datetime | None, source: Source | None) -> tuple[Example, ...]:
        chunks = (",".join(names[i : i + NAMES_PER_READ]) for i in range(0, len(names), NAMES_PER_READ))
        got = (ex for c in chunks for d in self._pages(c, since) if (ex := example_of_score(d)) is not None)
        return tuple(ex for ex in got if source is None or ex.source == source)

    async def examples(
        self, topics: Sequence[str], *, since: datetime | None = None, source: Source | None = None
    ) -> ExamplesRead:
        """Read every source's labels on the replies of `topics`. Never raises.

        A topic that is not `llm:<role>` contributes nothing; an `outcome` source reads empty.
        """
        roles = [r for t in topics if (r := role_of(t)) is not None]
        names = [n for r in roles for who in PREFIX if source is None or who == source for n in _names(who, r)]
        if not names:
            return ExamplesRead((), "ok")
        try:
            got = await asyncio.to_thread(self._read, names, since, source)
        except Exception as exc:  # an unreadable store is never an empty one
            return ExamplesRead((), "unavailable", _reason(exc))
        return ExamplesRead(got, "ok")


__all__ = [
    "CATEGORIES",
    "COMPLETED",
    "CONFIG_NAME",
    "CONFIG_NAME_MAX",
    "NAMES_PER_READ",
    "PREFIX",
    "SEP",
    "Labeller",
    "LangfuseExampleStore",
    "example_of_score",
    "parse_name",
    "score_name",
]
