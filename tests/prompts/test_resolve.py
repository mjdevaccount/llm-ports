"""resolve: a manifest's `output_schema`, `graders` and `golden` resolve strictly, by the library, and
each bad pointer names its field (acceptance test 6). `scan` reports resolution failures beside load
failures when given the same arguments."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from llm_ports import prompts
from llm_ports.call.graders import GRADERS, echo
from tests.prompts import outputs_fixture

ROOTS = ("tests.prompts",)
BASE = "id: demo.note\nversion: 1.0.0\nrole: m\ntemplates:\n  prompt: prompt.j2\n"


def _grader(_prompt: str, _value: Any, _role: Any) -> Any:
    return None


class Registry:
    """A `GraderRegistry` that is not a mapping."""

    def __init__(self, table: dict[str, Callable[..., Any]]) -> None:
        self.table = table

    def get(self, name: str) -> Callable[..., Any]:
        return self.table[name]


def _load(tmp_path: Path, extra: str, pid: str = "demo.note") -> prompts.Pack:
    d = tmp_path / pid
    d.mkdir(parents=True, exist_ok=True)
    (d / "pack.yaml").write_bytes((BASE.replace("demo.note", pid) + extra).encode())
    (d / "prompt.j2").write_bytes(b"Say {{ x }}.")
    return prompts.load(pid, tmp_path)


def test_a_good_pack_resolves_its_type_graders_and_golden(tmp_path: Path) -> None:
    """Acceptance 6, the good half."""
    pack = _load(
        tmp_path, "output_schema: tests.prompts.outputs_fixture:Note\ngraders: [echo, mine]\ngolden: golden.json\n"
    )
    r = prompts.resolve(pack, registry={**GRADERS, "mine": _grader}, import_roots=ROOTS)
    assert r.pack is pack and r.output is outputs_fixture.Note
    assert dict(r.graders) == {"echo": echo, "mine": _grader} and list(r.graders) == ["echo", "mine"]
    assert r.golden == (tmp_path / "demo.note" / "golden.json").resolve()
    assert r.output_schema_sha == prompts.schema_sha(outputs_fixture.Note)
    assert len(r.output_schema_sha) == 16
    assert pack.resolved(registry=Registry({"echo": echo, "mine": _grader}), import_roots=ROOTS) == r


def test_a_json_output_resolves_and_a_prose_pack_resolves_to_nothing(tmp_path: Path) -> None:
    pack = _load(tmp_path, "output_schema: tests.prompts.outputs_fixture.NOTE_JSON\n")
    r = prompts.resolve(pack, import_roots=ROOTS)
    assert r.output is outputs_fixture.NOTE_JSON
    assert r.output_schema_sha == prompts.schema_sha(outputs_fixture.NOTE_JSON)
    prose = prompts.resolve(_load(tmp_path, "", "demo.prose"))
    assert (prose.output, dict(prose.graders), prose.golden, prose.output_schema_sha) == (None, {}, None, None)


def test_the_schema_sha_moves_with_the_schema() -> None:
    assert prompts.schema_sha(outputs_fixture.Note) != prompts.schema_sha(outputs_fixture.NOTE_JSON)
    assert not isinstance(outputs_fixture.NOT_A_TYPE, type), "the `not-a-type` case below points here"


@pytest.mark.parametrize(
    ("extra", "registry", "roots", "msg"),
    [
        (
            "output_schema: tests.prompts.outputs_fixture:Note\n",
            None,
            (),
            r"output_schema: .* none of the import roots",
        ),
        ("output_schema: os.path:join\n", None, ROOTS, r"output_schema: module 'os.path' is under none"),
        ("output_schema: tests.prompts.absent:Note\n", None, ROOTS, r"output_schema: .* does not import"),
        ("output_schema: tests.prompts.outputs_fixture:Absent\n", None, ROOTS, r"output_schema: .* does not import"),
        ("output_schema: tests.prompts.outputs_fixture:NOT_A_TYPE\n", None, ROOTS, r"output_schema: .* not a pydantic"),
        ("graders: [echo]\ngolden: g.json\n", None, (), r"graders: 'echo' cannot resolve: no grader registry"),
        ("graders: [typo]\ngolden: g.json\n", GRADERS, (), r"graders: 'typo' is not in the grader registry"),
        ("graders: [typo]\ngolden: g.json\n", Registry({}), (), r"graders: 'typo' is not in the grader registry"),
        ("graders: [echo]\ngolden: ../outside.json\n", GRADERS, (), r"golden: '../outside.json' is outside the pack"),
    ],
    ids=[
        "no-roots",
        "outside-root",
        "no-module",
        "no-attr",
        "not-a-type",
        "no-registry",
        "typo",
        "typo-proto",
        "escape",
    ],
)
def test_each_bad_pointer_names_its_field(
    tmp_path: Path, extra: str, registry: Any, roots: tuple[str, ...], msg: str
) -> None:
    """Acceptance 6, the bad half: each pointer that misses names its field."""
    pack = _load(tmp_path, extra)
    with pytest.raises(prompts.PromptPackInvalid, match=msg):
        prompts.resolve(pack, registry=registry, import_roots=roots)


def test_a_registry_that_answers_none_is_a_miss(tmp_path: Path) -> None:
    class Nones:
        def get(self, _name: str) -> Any:
            return None

    with pytest.raises(prompts.PromptPackInvalid, match="not in the grader registry"):
        prompts.resolve(_load(tmp_path, "graders: [echo]\ngolden: g.json\n"), registry=Nones())


def test_scan_reports_resolution_failures_beside_load_failures_when_given_the_arguments(tmp_path: Path) -> None:
    _load(tmp_path, "graders: [echo]\ngolden: g.json\n", "demo.good")
    _load(tmp_path, "graders: [typo]\ngolden: g.json\n", "demo.typo")
    (tmp_path / "demo.broken").mkdir()
    (tmp_path / "demo.broken" / "pack.yaml").write_bytes(b"- not a mapping\n")
    loaded, invalid = prompts.scan(tmp_path)
    assert [p.id for p in loaded] == ["demo.good", "demo.typo"] and list(invalid) == ["demo.broken"]
    loaded, invalid = prompts.scan(tmp_path, registry=GRADERS)
    assert [p.id for p in loaded] == ["demo.good"] and sorted(invalid) == ["demo.broken", "demo.typo"]
    assert "graders: 'typo'" in invalid["demo.typo"]
    _, invalid = prompts.scan(tmp_path, import_roots=ROOTS)
    assert "no grader registry" in invalid["demo.good"]


def test_registry_fails_at_build_on_a_mistyped_grader_when_given_the_arguments(tmp_path: Path) -> None:
    _load(tmp_path, "graders: [echo]\ngolden: g.json\n", "demo.good")
    _load(tmp_path, "graders: [typo]\ngolden: g.json\n", "demo.typo")
    assert [p.id for p in prompts.registry(tmp_path)] == ["demo.good", "demo.typo"], "unchanged without arguments"
    with pytest.raises(prompts.PromptPackInvalid, match=r"demo\.typo: graders: 'typo'"):
        prompts.registry(tmp_path, registry=GRADERS)
    (tmp_path / "demo.typo" / "pack.yaml").unlink()
    assert [p.id for p in prompts.registry(tmp_path, registry=GRADERS, import_roots=ROOTS)] == ["demo.good"]
