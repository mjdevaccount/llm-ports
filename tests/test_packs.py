"""packs: manifest contract, loader, the FROZEN content sha, scan/registry, render."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from llm_ports import packs
from llm_ports.manifest import PackManifest
from tests.conftest import PACKS

MANIFEST = "id: demo.note\nversion: 1.0.0\nrole: m\ntemplates:\n  prompt: prompt.j2\n"


def _pack(root: Path, pid: str, manifest: str, files: dict[str, str]) -> Path:
    d = root / pid
    d.mkdir(parents=True)
    (d / "pack.yaml").write_text(manifest, encoding="utf-8")
    for name, text in files.items():
        (d / name).write_text(text, encoding="utf-8")
    return d


def test_the_sha_algorithm_is_frozen_on_fixed_input() -> None:
    """PINNED: a persisted stamp may never change algorithm. These literals are sha256[:16]; a
    change here is a breaking release for every consumer that stored a stamp."""
    assert packs.sha("") == "e3b0c44298fc1c14"
    assert packs.sha("hello") == "2cf24dba5fb0a30e"
    assert packs.load("demo.one", PACKS).content_sha == "1b952fc947eb43f2"
    assert packs.load("demo.multi", PACKS).content_sha == "6d1f6ee4195c8fb8"


def test_a_single_template_pack_hashes_exactly_its_source() -> None:
    one = packs.load("demo.one", PACKS)
    assert one.content_sha == hashlib.sha256(one.source.encode()).hexdigest()[:16]
    assert packs.content_sha(["a"]) == packs.sha("a")
    assert packs.content_sha(["a", "b"]) == packs.sha("a" + chr(10) + "b")


def test_load_renders_under_strict_undefined() -> None:
    one = packs.load("demo.one", PACKS)
    assert one.render(who="a desk", what="hi") == "You are a desk.\nSay hi."
    assert one.render(who="a desk", what="") == "You are a desk.\n"
    with pytest.raises(Exception, match="who"):
        one.render(what="x")
    assert packs.load("demo.multi", PACKS).render("second", b="2") == "second 2"
    assert one.render_source("cand {{ who }}", who="x") == "cand x"


def test_stamp_names_pack_version_and_sha() -> None:
    one = packs.load("demo.one", PACKS)
    assert one.stamp() == {"pack": "demo.one", "version": "1.2.3", "sha": "1b952fc947eb43f2"}
    assert one.stamp_str() == "demo.one@1.2.3+1b952fc947eb43f2"


@pytest.mark.parametrize(("manifest", "msg"), [
    (MANIFEST + "invented: 1\n", "Extra inputs"),
    (MANIFEST.replace("demo.note", "Demo"), "feature.action"),
    (MANIFEST.replace("1.0.0", "1.0.x"), "MAJOR.MINOR.PATCH"),
    (MANIFEST.replace("prompt: prompt.j2", "other: prompt.j2"), "primary template"),
    (MANIFEST.replace("prompt.j2", "../x.j2"), "INSIDE the pack"),
    (MANIFEST + "graders: [pkg.mod:G]\n", "golden"),
], ids=["extra", "id", "semver", "primary", "path", "golden"])
def test_the_manifest_is_strict(manifest: str, msg: str) -> None:
    with pytest.raises(ValidationError, match=msg):
        PackManifest.model_validate(yaml.safe_load(manifest))


def test_an_evaluable_pack_names_graders_and_golden() -> None:
    m = PackManifest.model_validate(yaml.safe_load(MANIFEST + "graders: [pkg.mod:G]\ngolden: g.json\n"))
    assert m.evaluable and m.feature == "demo"


@pytest.mark.parametrize(("manifest", "files", "msg"), [
    (MANIFEST, {}, "not in the pack"),
    (MANIFEST, {"prompt.j2": "{% if %}"}, "line 1"),
    (MANIFEST.replace("demo.note", "demo.other"), {"prompt.j2": "x"}, "must equal the directory"),
    ("- a list\n", {}, "not a mapping"),
    ("id: [unclosed\n", {}, "unreadable"),
], ids=["missing-file", "syntax", "id-mismatch", "not-mapping", "bad-yaml"])
def test_an_invalid_pack_fails_closed(tmp_path: Path, manifest: str, files: dict[str, str], msg: str) -> None:
    _pack(tmp_path, "demo.note", manifest, files)
    with pytest.raises(packs.PromptPackInvalid, match=msg):
        packs.load("demo.note", tmp_path)


def test_a_missing_pack_is_invalid_not_a_fallback(tmp_path: Path) -> None:
    with pytest.raises(packs.PromptPackInvalid, match=r"no pack.yaml"):
        packs.load("demo.none", tmp_path)


def test_scan_names_every_invalid_pack_and_registry_refuses(tmp_path: Path) -> None:
    _pack(tmp_path, "demo.note", MANIFEST, {"prompt.j2": "ok"})
    _pack(tmp_path, "demo.bad", MANIFEST.replace("demo.note", "demo.bad"), {})
    (tmp_path / "README_dir").mkdir()
    loaded, invalid = packs.scan(tmp_path)
    assert [p.id for p in loaded] == ["demo.note"] and list(invalid) == ["demo.bad"]
    with pytest.raises(packs.PromptPackInvalid, match=r"demo.bad"):
        packs.registry(tmp_path)
    assert packs.served_shas(tmp_path) == {"demo.note": packs.sha("ok")}
    assert packs.scan(tmp_path / "absent") == ((), {})


def test_unevaluated_is_served_minus_graded() -> None:
    assert packs.unevaluated({"a": "1", "b": "2", "c": "3"}, {"a": "1", "b": "old"}) == ["b", "c"]
