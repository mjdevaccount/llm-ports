"""fire_pack: a resolved pack's golden set fires through its own source and output type, grades on its
own graders, and stamps the run and every row with `identity`; a run with nothing to fire is
`could_not_run`, never a pass over zero items (acceptance test 7). `resolve_pack` falls back to the
built-in call graders."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from llm_ports import prompts
from llm_ports.call.graders import GRADERS, CallScore, echo
from llm_ports.evals import golden
from llm_ports.testing import StubGenerator
from tests.conftest import Role

ROLE = Role("test.golden", tier="local", pack="demo.note")
MANIFEST = (
    "id: demo.note\nversion: 1.0.0\nrole: test.golden\ntemplates:\n  prompt: prompt.j2\n"
    "graders: [echo, schema_complete]\ngolden: golden.json\n"
)
ITEMS = [{"id": "g-1", "slots": {"who": "a clerk"}, "stratum": "a"}, {"id": "g-2", "slots": {"who": "a judge"}}]
REPLY = {"summary": "The clerk greets the room in a sentence long enough to count as prose."}

Events = list[tuple[int, str, Mapping[str, Any]]]


def _pack(tmp_path: Path, golden_file: str | None, manifest: str = MANIFEST) -> prompts.Pack:
    d = tmp_path / "demo.note"
    d.mkdir(parents=True, exist_ok=True)
    (d / "pack.yaml").write_bytes(manifest.encode())
    (d / "prompt.j2").write_bytes(b"You are {{ who }}.")
    if golden_file is not None:
        (d / "golden.json").write_bytes(golden_file.encode())
    return prompts.load("demo.note", tmp_path)


async def _run(resolved: prompts.ResolvedPack, events: Events, gen: StubGenerator | None = None) -> golden.GoldenRun:
    stub = gen or StubGenerator(REPLY)
    return await golden.fire_pack(
        resolved,
        role=ROLE,
        generator_for=lambda: (stub, []),
        on_event=lambda lv, ev, f: events.append((lv, ev, f)),
        table_fingerprint="tbl0123456789abc",
    )


class TestCouldNotRun:
    @pytest.mark.parametrize(
        ("golden_file", "reason"),
        [
            (None, "golden file missing: 'golden.json' resolved to "),
            ("[]", "golden set is empty"),
            ('{"items": []}', "golden set is empty"),
            ("{not json", "golden file unreadable: JSONDecodeError"),
            ('{"meta": 1}', "golden file unreadable: ValueError: not a list of items"),
            ('[{"slots": {}}]', "golden file unreadable: ValueError: item 0 is not a mapping with an `id`"),
            (
                '[{"id": "g-1", "slots": {"who": "x"}}, {"id": "g-2", "slots": {}}]',
                "golden item 'g-2' does not render: Undef",
            ),
            ('[{"id": "g-1", "slots": ["who"]}]', "golden item 'g-1' does not render: TypeError"),
        ],
        ids=[
            "missing",
            "empty-list",
            "empty-items",
            "not-json",
            "no-items",
            "no-id",
            "slot-missing",
            "slots-not-a-map",
        ],
    )
    async def test_nothing_to_fire_is_could_not_run_with_the_reason(
        self, tmp_path: Path, golden_file: str | None, reason: str
    ) -> None:
        """Acceptance 7, the refusing half: no generator is ever called, the WARNING names the reason."""
        events: Events = []
        gen = StubGenerator(REPLY)
        run = await _run(golden.resolve_pack(_pack(tmp_path, golden_file)), events, gen)
        assert run.status == "could_not_run" and run.reason.startswith(reason)
        assert run.board is None and run.rows == [] and gen.requests == []
        assert [(lv, ev, f["status"]) for lv, ev, f in events] == [(logging.WARNING, "golden_run", "could_not_run")]
        assert events[0][2]["identity"] == run.stamp["identity"]

    async def test_a_pack_with_no_golden_set_could_not_run(self, tmp_path: Path) -> None:
        manifest = MANIFEST.replace("graders: [echo, schema_complete]\ngolden: golden.json\n", "")
        run = await _run(golden.resolve_pack(_pack(tmp_path, None, manifest)), [])
        assert (run.status, run.reason) == ("could_not_run", "the pack declares no golden set")

    async def test_a_golden_set_with_no_graders_is_never_green_over_nothing(self, tmp_path: Path) -> None:
        manifest = MANIFEST.replace("graders: [echo, schema_complete]\n", "")
        gen = StubGenerator(REPLY)
        run = await _run(golden.resolve_pack(_pack(tmp_path, json.dumps(ITEMS), manifest)), [], gen)
        assert (run.status, run.board, gen.requests) == ("could_not_run", None, [])
        assert run.reason.startswith("the pack declares no graders")

    async def test_the_missing_reason_names_the_manifest_value_and_the_resolved_path(self, tmp_path: Path) -> None:
        resolved = golden.resolve_pack(_pack(tmp_path, None))
        run = await _run(resolved, [])
        assert "'golden.json'" in run.reason and str(resolved.golden) in run.reason


class TestRan:
    async def test_rows_and_the_run_are_stamped_with_identity_and_graded_on_the_packs_graders(
        self, tmp_path: Path
    ) -> None:
        """Acceptance 7, the running half."""
        pack = _pack(tmp_path, json.dumps({"version": "v1", "items": ITEMS}))
        events: Events = []
        gen = StubGenerator(REPLY)
        run = await _run(golden.resolve_pack(pack), events, gen)
        assert run.status == "ran" and run.reason == ""
        assert run.stamp == {
            **pack.stamp(),
            "tier": "local",
            "model": ROLE.resolved_model,
            "table_fingerprint": "tbl0123456789abc",
            "output_schema_sha": None,
        }
        assert [r["identity"] for r in run.rows] == [pack.identity] * 2
        assert [r["id"] for r in run.rows] == ["g-1", "g-2"] and run.rows[0]["stratum"] == "a"
        assert sorted(r.prompt for r in gen.requests) == ["You are a clerk.", "You are a judge."]
        assert run.board is not None and set(run.board["by_grader"]) == {"echo", "schema_complete"}
        assert run.board["n"] == 2 and run.board["surface"] == "demo.note" and set(run.outs) == {"g-1", "g-2"}
        items = [f for _, ev, f in events if ev == "golden_item"]
        assert [f["identity"] for f in items] == [pack.identity] * 2
        assert events[-1][:2] == (logging.INFO, "golden_run") and events[-1][2]["status"] == "ran"

    async def test_json_lines_and_a_bare_list_are_read_alike(self, tmp_path: Path) -> None:
        d = tmp_path / "jl"
        _pack(d, json.dumps(ITEMS))
        (d / "demo.note" / "golden.jsonl").write_bytes(("\n".join(json.dumps(i) for i in ITEMS) + "\n\n").encode())
        assert golden.read_golden(d / "demo.note" / "golden.jsonl") == golden.read_golden(
            d / "demo.note" / "golden.json"
        )

    async def test_the_pack_output_type_is_what_fires(self, tmp_path: Path) -> None:
        manifest = MANIFEST.replace("graders:", "output_schema: tests.prompts.outputs_fixture:Note\ngraders:")
        pack = _pack(tmp_path, json.dumps(ITEMS), manifest)
        from tests.prompts.outputs_fixture import Note

        seen: list[Any] = []

        def reply(_req: Any, out: type[Any]) -> Any:
            seen.append(out)
            return Note(text="a note")

        resolved = golden.resolve_pack(pack, import_roots=("tests.prompts",))
        run = await _run(resolved, [], StubGenerator(reply))
        assert run.status == "ran" and seen == [Note, Note]
        assert run.stamp["output_schema_sha"] == resolved.output_schema_sha == prompts.schema_sha(Note)

    async def test_a_prose_pack_fires_as_text(self, tmp_path: Path) -> None:
        seen: list[Any] = []

        def reply(_req: Any, out: type[Any]) -> Any:
            seen.append(out)
            return "plain prose"

        run = await _run(golden.resolve_pack(_pack(tmp_path, json.dumps(ITEMS[:1]))), [], StubGenerator(reply))
        assert run.status == "ran" and seen == [str]


class TestResolvePack:
    def test_the_callers_registry_leads_and_the_built_in_graders_follow(self, tmp_path: Path) -> None:
        def mine(_p: str, _v: Any, _r: Any) -> CallScore:
            return CallScore("echo", 0.0, True, "")

        pack = _pack(tmp_path, None)
        assert golden.resolve_pack(pack).graders == {"echo": echo, "schema_complete": GRADERS["schema_complete"]}
        assert golden.resolve_pack(pack, registry={"echo": mine}).graders["echo"] is mine

        class Proto:
            def get(self, name: str) -> Any:
                return {"echo": mine}[name]

        assert golden.resolve_pack(pack, registry=Proto()).graders["schema_complete"] is GRADERS["schema_complete"]

    def test_a_name_neither_knows_is_invalid(self, tmp_path: Path) -> None:
        pack = _pack(tmp_path, None, MANIFEST.replace("[echo, schema_complete]", "[typo]"))
        with pytest.raises(prompts.PromptPackInvalid, match="graders: 'typo'"):
            golden.resolve_pack(pack)
