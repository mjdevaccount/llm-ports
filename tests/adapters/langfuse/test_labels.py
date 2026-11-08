"""LangfuseExampleStore -- labels are scores named by source and role, a human label is written
only when it names its person, queues and configs are found before they are created, and every
failure is an outcome."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from llm_ports.adapters.langfuse import labels as lf
from llm_ports.evals.labels import LabelQueue, topic
from llm_ports.learn import Example, ExampleStore, WritableExampleStore
from llm_ports.testing.langfuse import RefusingLangfuse

pytest.importorskip("langfuse")


def _score(name: str, value: Any, obs: str = "o1", trace: str = "t1", kind: str = "observation") -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        value=value,
        subject=SimpleNamespace(kind=kind, id=obs, trace_id=trace),
        metadata={"by": "ann"},
        comment="clear",
        timestamp=datetime(2026, 11, 1, tzinfo=UTC),
    )


#: The server's own rule for a score-config name (Langfuse `packages/shared/src/domain/score-configs.ts`),
#: copied here so the double refuses what the server refuses, independent of the adapter's check.
_SERVER_CONFIG_NAME = re.compile(r"[\w .()-]{1,80}")


class ServerRefused(Exception):
    """Shaped like the SDK's API error: a status code and the server's JSON body."""

    def __init__(self, body: dict[str, Any]) -> None:
        super().__init__(f"status_code: 400, body: {body}")
        self.status_code = 400
        self.body = body


class FakeApi:
    def __init__(self, scores: list[Any], configs: list[Any] | None = None, queues: list[Any] | None = None) -> None:
        self.created: list[str] = []
        self.items: list[tuple[str, str]] = []
        self.reads: list[dict[str, Any]] = []
        pages = [scores[:2], scores[2:]] if len(scores) > 2 else [scores]
        cfgs, qs = configs or [], queues or []

        def get_many(**kw: Any) -> SimpleNamespace:
            self.reads.append(kw)
            i = int(kw["cursor"] or 0)
            nxt = str(i + 1) if i + 1 < len(pages) else None
            return SimpleNamespace(data=pages[i], meta=SimpleNamespace(cursor=nxt))

        def create_config(**kw: Any) -> SimpleNamespace:
            if _SERVER_CONFIG_NAME.fullmatch(kw["name"]) is None:
                raise ServerRefused(
                    {"message": "Invalid request data", "error": [{"message": "Name contains invalid characters"}]}
                )
            self.created.append("config:" + kw["name"])
            return SimpleNamespace(id="cfg-new")

        def create_queue(**kw: Any) -> SimpleNamespace:
            self.created.append("queue:" + kw["name"] + ":" + ",".join(kw["score_config_ids"]))
            return SimpleNamespace(id="q-new")

        def list_items(q: str, **kw: Any) -> SimpleNamespace:
            self.item_reads.append(kw["page"])
            held = [
                SimpleNamespace(id=f"item-{o}", object_id=o, status=self.done.get(o, "PENDING"))
                for qq, o in self.items
                if qq == q
            ]
            return SimpleNamespace(data=held[(kw["page"] - 1) * kw["limit"] : kw["page"] * kw["limit"]])

        def update_item(q: str, item: str, **kw: Any) -> None:
            self.updated.append((q, item, kw["status"]))
            self.done[item.removeprefix("item-")] = kw["status"]

        self.updated: list[tuple[str, str, str]] = []
        self.item_reads: list[int] = []
        self.done: dict[str, str] = {}
        self.scores = SimpleNamespace(get_many=get_many)
        self.score_configs = SimpleNamespace(get=lambda **kw: SimpleNamespace(data=cfgs), create=create_config)
        self.annotation_queues = SimpleNamespace(
            list_queues=lambda **kw: SimpleNamespace(data=qs),
            create_queue=create_queue,
            create_queue_item=lambda q, **kw: self.items.append((q, kw["object_id"])),
            list_queue_items=list_items,
            update_queue_item=update_item,
        )


class FakeClient:
    def __init__(self, api: FakeApi) -> None:
        self.api = api
        self.written: list[dict[str, Any]] = []
        self.flushed = 0

    def create_score(self, **kw: Any) -> None:
        self.written.append(kw)

    def flush(self) -> None:
        self.flushed += 1


def test_it_is_the_core_store_and_the_queue() -> None:
    store = lf.LangfuseExampleStore(FakeClient(FakeApi([])))
    readable: ExampleStore = store  # the type checkers prove the core's Protocols
    writable: WritableExampleStore = store
    assert readable is writable
    assert isinstance(store, LabelQueue)


def test_score_names_parse_both_ways_and_other_scores_are_not_labels() -> None:
    assert lf.score_name("model", "summarize") == "label.model-summarize"
    assert lf.parse_name("label-summarize") == ("human", "summarize")
    assert lf.parse_name("label.judge-a") == ("judge", "a")
    assert lf.parse_name("label.model-x.y_z") == ("model", "x.y_z")
    assert lf.parse_name("label-model-x") == ("human", "model-x")
    assert lf.parse_name("faithfulness") is None and lf.parse_name("label-") is None
    # the earlier `:` spelling still reads, so scores written under it are not lost
    assert lf.parse_name("label:summarize") == ("human", "summarize")
    assert lf.parse_name("label.judge:a") == ("judge", "a") and lf.parse_name("label:") is None


@pytest.mark.parametrize("role", ["summarize", "deep_extract.mentions", "curation.deep_research", "day_brief.map"])
def test_every_label_name_is_one_the_server_accepts(role: str) -> None:
    for who in lf.PREFIX:
        assert _SERVER_CONFIG_NAME.fullmatch(lf.score_name(who, role)), (who, role)


async def test_a_role_the_server_would_refuse_is_refused_before_any_write() -> None:
    api = FakeApi([])
    client = FakeClient(api)
    store = lf.LangfuseExampleStore(client)
    got = await store.enqueue(topic("a:b"), [("o1", "t1")])
    assert (got.written, got.outcome) == (0, "request_fault") and "score-config name" in (got.reason or "")
    wrote = await store.put([_ex(tp="llm:" + "x" * 80)])
    assert (wrote.written, wrote.outcome) == (0, "request_fault")
    assert api.created == [] and api.items == [] and client.written == []


async def test_a_server_refusal_names_the_servers_reason() -> None:
    api = FakeApi([])
    api.score_configs.create = lambda **kw: (_ for _ in ()).throw(
        ServerRefused({"message": "Invalid request data", "error": [{"message": "Name contains invalid characters"}]})
    )
    got = await lf.LangfuseExampleStore(FakeClient(api)).enqueue(topic("a"), [("o1", "t1")])
    assert got.outcome == "unavailable" and "Name contains invalid characters" in (got.reason or "")
    api.score_configs.create = lambda **kw: (_ for _ in ()).throw(ServerRefused({"message": "line one\n" + "x" * 500}))
    long = await lf.LangfuseExampleStore(FakeClient(api)).enqueue(topic("b"), [("o1", "t1")])
    assert "line one" in (long.reason or "") and "\n" not in (long.reason or "") and len(long.reason or "") <= 200


def test_the_length_rule_counts_as_the_server_does() -> None:
    fits = "a" * (lf.CONFIG_NAME_MAX - len("label.model-"))
    assert lf._bad_name(fits) is None and lf._bad_name(fits + "a") is not None
    astral = "\U0001d400" * 35  # with a prefix: 47 code points, 82 UTF-16 units; the server counts units
    assert lf._bad_name(astral) is not None


async def test_a_wide_read_is_split_so_no_request_carries_every_name() -> None:
    api = FakeApi([])
    roles = [f"r{i}" for i in range(30)]
    got = await lf.LangfuseExampleStore(FakeClient(api)).examples([topic(r) for r in roles])
    assert got.ok
    sent = [kw["name"].split(",") for kw in api.reads]
    assert all(len(names) <= lf.NAMES_PER_READ for names in sent)
    assert sorted(n for names in sent for n in names) == sorted(
        n for r in roles for w in lf.PREFIX for n in lf._names(w, r)
    )


def test_example_of_score_reads_value_kinds_and_skips_what_it_cannot_read() -> None:
    ex = lf.example_of_score(_score("label:a", "good"))
    assert ex is not None
    assert (ex.topic, ex.subject, ex.value, ex.source, ex.by, ex.reason, ex.ref) == (
        "llm:a",
        "o1",
        "good",
        "human",
        "ann",
        "clear",
        {"trace": "t1"},
    )
    assert lf.example_of_score(_score("label:a", 0.0)).value == "bad"  # type: ignore[union-attr]
    assert lf.example_of_score(_score("label:a", "meh")) is None
    assert lf.example_of_score(_score("other", "good")) is None
    assert lf.example_of_score(_score("label:a", "good", kind="trace")) is None


async def test_examples_pages_filters_by_source_and_ignores_other_topics() -> None:
    api = FakeApi(
        [
            _score("label:a", "good", "1"),
            _score("label.judge:a", "bad", "1"),
            _score("x", "good"),
            _score("label:a", 1, "2"),
        ]
    )
    store = lf.LangfuseExampleStore(FakeClient(api))
    got = await store.examples([topic("a"), "trend:1d"])
    assert got.ok and [(e.subject, e.source, e.value) for e in got.examples] == [
        ("1", "human", "good"),
        ("1", "judge", "bad"),
        ("2", "human", "good"),
    ]
    assert len(api.reads) == 2  # two pages
    human = await store.examples([topic("a")], source="human")
    assert {e.source for e in human.examples} == {"human"}
    assert api.reads[-1]["name"] == "label-a,label:a"  # both spellings
    assert (await store.examples(["trend:1d"])).examples == ()
    assert (await store.examples([topic("a")], source="outcome")).examples == ()


def _ex(
    source: str = "model", trace: str = "t1", value: str = "bad", tp: str = "llm:a", by: str = "gemini", obs: str = "o1"
) -> Example:
    return Example(tp, obs, value, source, by=by, reason="r", ref={"trace": trace} if trace else {})  # type: ignore[arg-type]


async def test_put_writes_model_and_judge_scores_under_a_stable_id() -> None:
    client = FakeClient(FakeApi([]))
    store = lf.LangfuseExampleStore(client)
    got = await store.put([_ex("model"), _ex("judge", value="good")])
    assert (got.written, got.outcome) == (2, "ok") and client.flushed == 1
    assert [w["name"] for w in client.written] == ["label.model-a", "label.judge-a"]
    first = client.written[0]
    assert (first["trace_id"], first["observation_id"], first["value"]) == ("t1", "o1", "bad")
    again = FakeClient(FakeApi([]))
    await lf.LangfuseExampleStore(again).put([_ex("model", value="good")])
    assert again.written[0]["score_id"] == first["score_id"]  # a relabel replaces, never adds


@pytest.mark.parametrize(
    ("ex", "why"),
    [
        (_ex("human", by=" "), "human label needs `by`"),
        (_ex("outcome"), "outcome label is not written by code"),
        (_ex(trace=""), "ref['trace']"),
        (_ex(value="meh"), "not good or bad"),
        (_ex(tp="trend:1d"), "not a model reply"),
    ],
)
async def test_put_refuses_a_batch_it_cannot_write_and_writes_none(ex: Example, why: str) -> None:
    client = FakeClient(FakeApi([]))
    got = await lf.LangfuseExampleStore(client).put([_ex(), ex])
    assert (got.written, got.outcome) == (0, "request_fault") and why in (got.reason or "")
    assert client.written == []


async def test_a_relayed_human_label_is_scored_on_the_config_and_completes_its_queue_item() -> None:
    api = FakeApi(
        [], configs=[SimpleNamespace(name="label-a", id="cfg-a")], queues=[SimpleNamespace(name="label-a", id="q-a")]
    )
    api.items = [("q-a", "o1"), ("q-a", "o2")]
    client = FakeClient(api)
    store = lf.LangfuseExampleStore(client)
    got = await store.put([_ex("human", value="good", by="ann")])
    assert (got.written, got.outcome) == (1, "ok")
    first = client.written[0]
    assert (first["name"], first["value"], first["config_id"]) == ("label-a", "good", "cfg-a")
    assert first["metadata"] == {"by": "ann", "labeller": "human"}
    assert api.updated == [("q-a", "item-o1", "COMPLETED")]  # the other reply stays pending
    held = await store.queued(topic("a"))
    assert held.done == {"o1"} and held.pending == {"o2"}
    await store.put([_ex("human", value="bad", by="ann")])
    assert client.written[1]["score_id"] == first["score_id"]  # the same person relabelling replaces
    assert api.updated == [("q-a", "item-o1", "COMPLETED")]  # a done item is not re-marked
    await store.put([_ex("human", by="bob")])
    assert client.written[2]["score_id"] != first["score_id"]  # another person's label is its own
    model = await store.put([_ex("model")])
    assert model.ok and client.written[3]["config_id"] is None


async def test_a_relayed_human_label_on_a_role_with_no_queue_creates_its_config_only() -> None:
    api = FakeApi([])
    client = FakeClient(api)
    got = await lf.LangfuseExampleStore(client).put([_ex("human", by="ann")])
    assert got.ok and api.created == ["config:label-a"] and api.updated == []
    assert client.written[0]["config_id"] == "cfg-new"


async def test_a_queue_failure_after_the_scores_landed_still_counts_them() -> None:
    api = FakeApi([], queues=[SimpleNamespace(name="label-a", id="q-a")])
    api.items = [("q-a", "o1")]
    api.annotation_queues.update_queue_item = lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("down"))
    client = FakeClient(api)
    got = await lf.LangfuseExampleStore(client).put([_ex("human", by="ann"), _ex("human", by="ann", obs="o2")])
    assert (got.written, got.outcome) == (2, "unavailable") and len(client.written) == 2


async def test_enqueue_finds_or_creates_the_config_and_queue_once() -> None:
    api = FakeApi([], configs=[SimpleNamespace(name="label-a", id="cfg-a")])
    store = lf.LangfuseExampleStore(FakeClient(api))
    got = await store.enqueue(topic("a"), [("o1", "t1"), ("o2", "t1")])
    assert (got.written, got.outcome) == (2, "ok")
    assert api.created == ["queue:label-a:cfg-a"] and api.items == [("q-new", "o1"), ("q-new", "o2")]
    await store.enqueue(topic("a"), [("o3", "t1")])
    assert api.created == ["queue:label-a:cfg-a"]  # cached

    api2 = FakeApi([], queues=[SimpleNamespace(name="label-b", id="q-b")])
    store2 = lf.LangfuseExampleStore(FakeClient(api2))
    assert (await store2.enqueue(topic("b"), [("o1", "t1")])).ok
    assert api2.created == [] and api2.items == [("q-b", "o1")]

    api3 = FakeApi([])
    assert (await lf.LangfuseExampleStore(FakeClient(api3)).enqueue(topic("c"), [])).ok
    assert api3.created == ["config:label-c", "queue:label-c:cfg-new"]


async def test_enqueue_refuses_a_non_llm_topic() -> None:
    got = await lf.LangfuseExampleStore(FakeClient(FakeApi([]))).enqueue("trend:1d", [("o1", "t1")])
    assert got.outcome == "request_fault"


async def test_every_server_failure_is_an_outcome_never_a_raise() -> None:
    store = lf.LangfuseExampleStore(RefusingLangfuse())
    read = await store.examples([topic("a")])
    assert (read.outcome, read.examples) == ("unavailable", ())
    wrote = await store.put([_ex()])
    assert (wrote.outcome, wrote.written) == ("unavailable", 0)
    queued = await store.enqueue(topic("a"), [("o1", "t1")])
    assert queued.outcome == "unavailable"
    held = await store.queued(topic("a"))
    assert (held.outcome, held.seen) == ("unavailable", frozenset())


async def test_enqueue_skips_what_the_queue_already_holds() -> None:
    api = FakeApi([], queues=[SimpleNamespace(name="label-a", id="q-a")])
    store = lf.LangfuseExampleStore(FakeClient(api))
    assert (await store.enqueue(topic("a"), [("o1", "t1"), ("o1", "t1")])).written == 1
    got = await store.enqueue(topic("a"), [("o1", "t1"), ("o2", "t1")])
    assert (got.written, got.outcome) == (1, "ok")
    assert api.items == [("q-a", "o1"), ("q-a", "o2")]


async def test_queued_splits_pending_from_done_and_pages() -> None:
    api = FakeApi([], queues=[SimpleNamespace(name="label-a", id="q-a")])
    api.items = [("q-a", f"o{i}") for i in range(lf.PAGE + 1)] + [("q-other", "x")]
    api.done = {"o0": "COMPLETED"}
    got = await lf.LangfuseExampleStore(FakeClient(api)).queued(topic("a"))
    assert got.ok and got.done == {"o0"} and len(got.pending) == lf.PAGE
    assert got.seen == {f"o{i}" for i in range(lf.PAGE + 1)}
    assert api.item_reads == [1, 2]


async def test_queued_creates_nothing_and_refuses_a_non_llm_topic() -> None:
    api = FakeApi([])
    store = lf.LangfuseExampleStore(FakeClient(api))
    none = await store.queued(topic("a"))
    assert none.ok and none.seen == frozenset() and api.created == []
    bad = await store.queued("trend:1d")
    assert bad.outcome == "request_fault" and not bad.ok
