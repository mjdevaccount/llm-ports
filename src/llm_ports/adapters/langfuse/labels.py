"""The `LabelStore` port over Langfuse scores and annotation queues (extra `[langfuse]`).

A label is a score on the labelled generation, named by labeller and role:

- `label:<role>`: a HUMAN label, given in Langfuse's annotation-queue UI against a categorical
  score config of the same name (`good` / `bad`). This adapter creates the config and the queue
  and never writes this name.
- `label.model:<role>`: a stronger model's label, written by `put`.
- `label.judge:<role>`: the production judge's verdict as a label, written by `put`.

Every role gets one score config and one annotation queue, both named `label:<role>`, created on
first `enqueue`. `put` writes each score under a deterministic id (labeller, role, subject), so a
relabel replaces the earlier score instead of adding a second one.

Example:
    store = LangfuseLabelStore(Langfuse(...))  # the caller builds the client
    store.enqueue("summarize", [(observation_id, trace_id)])
    read = store.labels(["summarize"])

Invariants:
    - Never raises for a server or SDK failure: the failure is the outcome.
    - A score this adapter cannot read as a label (another name, an unknown value, no subject) is
      skipped, never guessed.

Rationale: ADR-0010.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import datetime
from typing import Any

from llm_ports._extras import extra
from llm_ports.ports import Label, Labeller, LabelsRead, LabelsWritten
from llm_ports.ports.outcomes import failure_reason
from llm_ports.prompts import sha

with extra("langfuse"):
    import langfuse  # noqa: F401  # the extra is what makes a Langfuse client buildable

#: The score name prefix of each labeller.
PREFIX: dict[Labeller, str] = {"human": "label", "model": "label.model", "judge": "label.judge"}
_BY_PREFIX = {v: k for k, v in PREFIX.items()}
#: The categorical values a human picks, and the numeric value Langfuse books each under.
CATEGORIES = ({"label": "good", "value": 1.0}, {"label": "bad", "value": 0.0})
#: Scores read per page.
PAGE = 100
_FIELDS = "details,subject,annotation"


def score_name(labeller: Labeller, role: str) -> str:
    """Return the score name a `labeller`'s label on a `role` reply is stored under."""
    return f"{PREFIX[labeller]}:{role}"


def parse_name(name: str) -> tuple[Labeller, str] | None:
    """Return `(labeller, role)` for a label score name, or None for any other score."""
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


def label_of_score(d: Any) -> Label | None:
    """Return a v3 score object as a `Label`, or None when it is not one this adapter wrote or reads."""
    parsed = parse_name(getattr(d, "name", "") or "")
    subject = getattr(d, "subject", None)
    value = _value(d)
    if parsed is None or value is None or subject is None or getattr(subject, "kind", None) != "observation":
        return None
    who, role = parsed
    meta = getattr(d, "metadata", None) or {}
    return Label(
        role=role,
        subject=str(subject.id),
        value=value,  # type: ignore[arg-type]  # narrowed to good/bad by _value
        labeller=who,
        trace=str(getattr(subject, "trace_id", "") or ""),
        by=str(meta.get("by") or getattr(d, "author_user_id", "") or ""),
        reason=str(getattr(d, "comment", "") or ""),
        at=getattr(d, "timestamp", None),
    )


class LangfuseLabelStore:
    """`LabelStore` whose labels are Langfuse scores and whose queue is a Langfuse annotation queue.

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
        if name not in self._configs:
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
        if name not in self._queues:
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

    def enqueue(self, role: str, subjects: Sequence[tuple[str, str]]) -> LabelsWritten:
        """Add each `(observation id, trace id)` to the role's annotation queue. Never raises."""
        written = 0
        try:
            queue = self._queue_id(role)
            for subject, _trace in subjects:
                self._c.api.annotation_queues.create_queue_item(queue, object_id=subject, object_type="OBSERVATION")
                written += 1
        except Exception as exc:  # the store's failure is the outcome
            return LabelsWritten(written, "unavailable", failure_reason(exc))
        return LabelsWritten(written, "ok")

    def put(self, labels: Sequence[Label]) -> LabelsWritten:
        """Write `model` and `judge` labels as categorical scores; a `human` label refuses the batch."""
        if any(lab.labeller == "human" for lab in labels):
            return LabelsWritten(0, "request_fault", "a human label is given in the annotation UI, never written by code")
        if any(not lab.trace for lab in labels):
            return LabelsWritten(0, "request_fault", "a label needs its trace id: Langfuse scopes observations by trace")
        written = 0
        try:
            for lab in labels:
                name = score_name(lab.labeller, lab.role)
                self._c.create_score(
                    name=name,
                    value=lab.value,
                    data_type="CATEGORICAL",
                    trace_id=lab.trace,
                    observation_id=lab.subject,
                    score_id=sha(f"{name}\n{lab.subject}"),
                    comment=lab.reason or None,
                    metadata={"by": lab.by, "labeller": lab.labeller},
                    timestamp=lab.at,
                )
                written += 1
            self._c.flush()
        except Exception as exc:  # the store's failure is the outcome
            return LabelsWritten(written, "unavailable", failure_reason(exc))
        return LabelsWritten(written, "ok")

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

    def labels(self, roles: Sequence[str], *, since: datetime | None = None) -> LabelsRead:
        """Read every labeller's labels on `roles`' replies. Never raises."""
        if not roles:
            return LabelsRead((), "ok")
        names = ",".join(score_name(who, r) for r in roles for who in PREFIX)
        try:
            got = tuple(lab for d in self._pages(names, since) if (lab := label_of_score(d)) is not None)
        except Exception as exc:  # an unreadable store is never an empty one
            return LabelsRead((), "unavailable", failure_reason(exc))
        return LabelsRead(got, "ok")


__all__ = ["CATEGORIES", "PREFIX", "LangfuseLabelStore", "label_of_score", "parse_name", "score_name"]
