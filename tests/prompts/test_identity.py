"""Pack identity (ADR-0026): the FROZEN `byte_sha` vectors, the untouched `content_sha`, what moves
`identity`, the closed template set (must fire both ways), CRLF refusal, and one environment per pack.

Acceptance tests 1-5 and 9 of the packs-and-evals spec are named in each test's docstring.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml
from jinja2 import TemplateNotFound
from pydantic import ValidationError

from llm_ports import prompts
from llm_ports.prompts import _packs
from tests.conftest import PACKS

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "prompt_pack" / "packs"
MANIFEST = "id: demo.note\nversion: 1.0.0\nrole: m\ntemplates:\n  prompt: prompt.j2\n"


def _pack(root: Path, files: dict[str, str | bytes], pid: str = "demo.note") -> Path:
    """Write a pack as BYTES: a text-mode write is CRLF on Windows, which the loader refuses."""
    d = root / pid
    d.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (d / name).write_bytes(data if isinstance(data, bytes) else data.encode())
    return d


class TestByteSha:
    def test_pinned_vectors(self) -> None:
        """Acceptance 1. PINNED: `identity` is persisted, so these literals never change."""
        assert prompts.byte_sha([]) == "e3b0c44298fc1c14"
        assert prompts.byte_sha([("a", b"")]) == "beae57e48f2dcae4"
        two = [("pack.yaml", b"id: demo.x\n"), ("prompt.j2", b"Hello {{ who }}.")]
        assert prompts.byte_sha(two) == "275b4d91fc035065"
        assert prompts.byte_sha(two[::-1]) == "ed45fa29eec79507", "order is part of the input"
        assert prompts.byte_sha([("a", b"")], n=64) == hashlib.sha256(b"a\x000\x00").hexdigest()

    def test_the_re_split_pair_that_collides_under_content_sha_does_not_collide_here(self) -> None:
        """Acceptance 1: `content_sha` joins by a bare newline, so a re-split collides; `byte_sha` does not."""
        assert prompts.content_sha(["a\nb", "c"]) == prompts.content_sha(["a", "b\nc"])
        left = prompts.byte_sha([("prompt", b"a\nb"), ("second", b"c")])
        right = prompts.byte_sha([("prompt", b"a"), ("second", b"b\nc")])
        assert (left, right) == ("1e124057097852d6", "5c501c2dd40eb9e4")

    def test_a_nul_in_a_part_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="NUL"):
            prompts.byte_sha([("a\0", b"")])


class TestContentShaIsUntouched:
    @pytest.mark.parametrize(
        ("root", "pid", "content_sha"),
        [
            (PACKS, "demo.one", "1b952fc947eb43f2"),
            (PACKS, "demo.multi", "6d1f6ee4195c8fb8"),
            (PACKS, "demo.refute", "34c32a0cf171d09f"),
            (EXAMPLES, "orders.refute", "c5b8b56bcb53ee95"),
            (EXAMPLES, "orders.summarize", "5f9b8b56cf9d4d56"),
        ],
    )
    def test_every_fixture_pack_keeps_its_1_22_0_content_sha(self, root: Path, pid: str, content_sha: str) -> None:
        """Acceptance 2: each literal is the value 1.22.0 computed for that pack."""
        assert prompts.load(pid, root).content_sha == content_sha


class TestIdentity:
    @pytest.mark.parametrize(
        ("pid", "identity"),
        [("demo.one", "f0fb2e91025130fb"), ("demo.multi", "867d9ec5f1d88441"), ("demo.refute", "1bb5cb516bc01458")],
    )
    def test_fixture_identities_are_pinned(self, pid: str, identity: str) -> None:
        """PINNED on the day it ships: manifest bytes, then each closed-set file by sorted path."""
        pack = prompts.load(pid, PACKS)
        assert pack.identity == identity
        manual = [("pack.yaml", (PACKS / pid / "pack.yaml").read_bytes())]
        manual += [(f, (PACKS / pid / f).read_bytes()) for f in pack.files()]
        assert prompts.byte_sha(manual) == identity

    def test_identity_moves_on_every_behaviour_change_and_not_on_a_reload(self, tmp_path: Path) -> None:
        """Acceptance 3: version bump, grader edit, template key rename, included partial edit."""
        base = (
            "id: demo.note\nversion: 1.0.0\nrole: m\n"
            "templates:\n  prompt: prompt.j2\n  second: second.j2\n"
            "includes: [part.j2]\ngraders: [echo]\ngolden: golden.json\n"
        )
        files = {"prompt.j2": "A {% include 'part.j2' %}", "second.j2": "B", "part.j2": "partial"}

        def build(name: str, manifest: str, **edit: str) -> prompts.Pack:
            _pack(tmp_path, {"pack.yaml": manifest.replace("demo.note", name), **files, **edit}, name)
            return prompts.load(name, tmp_path)

        ref = build("demo.ref", base)
        assert prompts.load("demo.ref", tmp_path).identity == ref.identity, "a no-op reload keeps identity"
        variants = {
            "version": build("demo.ver", base.replace("1.0.0", "1.0.1")),
            "graders": build("demo.grd", base.replace("[echo]", "[echo, no_refusal]")),
            "key-rename": build("demo.key", base.replace("second: second.j2", "other: second.j2")),
            "partial": build("demo.inc", base, **{"part.j2": "partial, edited"}),
        }
        for what, pack in variants.items():
            assert pack.identity != ref.identity, what
        # The holes content_sha leaves: it covers neither the manifest nor an included file.
        renamed = variants["key-rename"]
        assert renamed.content_sha == ref.content_sha and renamed.identity != ref.identity
        assert variants["partial"].content_sha == ref.content_sha
        assert ref.stamp()["identity"] == ref.identity and ref.files() == ("part.j2", "prompt.j2", "second.j2")

    def test_a_hand_built_pack_has_an_identity_over_its_templates(self) -> None:
        loaded = prompts.load("demo.one", PACKS)
        hand = prompts.Pack(loaded.manifest, loaded.dir, dict(loaded.sources), loaded.content_sha)
        assert hand.files() == ("prompt.j2",) and hand.identity == prompts.byte_sha(
            [("prompt.j2", loaded.source.encode())]
        )
        assert hand.render(who="x", what="") == loaded.render(who="x", what="")


class TestClosedTemplateSet:
    def test_an_unlisted_include_fails_at_load_and_listing_it_loads_renders_and_moves_identity(
        self, tmp_path: Path
    ) -> None:
        """Acceptance 4, must fire both ways."""
        prompt = "Head. {% include 'part.j2' %}"
        _pack(tmp_path, {"pack.yaml": MANIFEST, "prompt.j2": prompt, "part.j2": "Tail."})
        with pytest.raises(prompts.PromptPackInvalid, match=r"references 'part.j2', which is not in `includes`"):
            prompts.load("demo.note", tmp_path)
        _pack(tmp_path, {"pack.yaml": MANIFEST + "includes: [part.j2]\n"})
        pack = prompts.load("demo.note", tmp_path)
        assert pack.render() == "Head. Tail." and pack.files() == ("part.j2", "prompt.j2")
        without = prompts.byte_sha([("pack.yaml", MANIFEST.encode()), ("prompt.j2", prompt.encode())])
        assert pack.identity != without

    @pytest.mark.parametrize(
        "prompt",
        [
            "{% include name %}",
            "{% include 'a.j2' if x else 'b.j2' %}",
        ],
        ids=["variable", "conditional"],
    )
    def test_a_dynamic_include_is_refused_at_load(self, tmp_path: Path, prompt: str) -> None:
        _pack(tmp_path, {"pack.yaml": MANIFEST, "prompt.j2": prompt})
        with pytest.raises(prompts.PromptPackInvalid, match="dynamic include"):
            prompts.load("demo.note", tmp_path)

    @pytest.mark.parametrize(
        "prompt",
        ["{% extends 'base.j2' %}", "{% import 'base.j2' as b %}x", "{% from 'base.j2' import m %}x"],
        ids=["extends", "import", "from"],
    )
    def test_extends_and_imports_are_checked_like_includes(self, tmp_path: Path, prompt: str) -> None:
        _pack(tmp_path, {"pack.yaml": MANIFEST, "prompt.j2": prompt, "base.j2": "{% macro m() %}{% endmacro %}"})
        with pytest.raises(prompts.PromptPackInvalid, match=r"'base.j2'"):
            prompts.load("demo.note", tmp_path)
        _pack(tmp_path, {"pack.yaml": MANIFEST + "includes: [base.j2]\n"})
        assert prompts.load("demo.note", tmp_path).files() == ("base.j2", "prompt.j2")

    def test_a_missing_or_broken_include_fails_closed(self, tmp_path: Path) -> None:
        _pack(tmp_path, {"pack.yaml": MANIFEST + "includes: [part.j2]\n", "prompt.j2": "x"})
        with pytest.raises(prompts.PromptPackInvalid, match=r"includes names 'part.j2', which is not in the pack"):
            prompts.load("demo.note", tmp_path)
        _pack(tmp_path, {"part.j2": "{% if %}"})
        with pytest.raises(prompts.PromptPackInvalid, match=r"'part.j2' line 1"):
            prompts.load("demo.note", tmp_path)

    def test_a_candidate_source_renders_only_inside_the_closed_set(self, tmp_path: Path) -> None:
        _pack(tmp_path, {"pack.yaml": MANIFEST + "includes: [part.j2]\n", "prompt.j2": "x", "part.j2": "P"})
        (tmp_path / "demo.note" / "stray.j2").write_bytes(b"never rendered")
        pack = prompts.load("demo.note", tmp_path)
        assert pack.render_source("{% include 'part.j2' %}!") == "P!"
        with pytest.raises(TemplateNotFound):
            pack.render_source("{% include 'stray.j2' %}")

    @pytest.mark.parametrize(
        ("line", "msg"),
        [("includes: [../x.j2]\n", "INSIDE the pack"), ("includes: [x.txt]\n", "INSIDE the pack")],
    )
    def test_includes_name_j2_files_inside_the_pack(self, line: str, msg: str) -> None:
        with pytest.raises(ValidationError, match=msg):
            prompts.PackManifest.model_validate(yaml.safe_load(MANIFEST + line))


class TestCarriageReturns:
    @pytest.mark.parametrize("target", ["pack.yaml", "prompt.j2", "part.j2"])
    def test_crlf_is_refused_and_the_lf_twin_loads(self, tmp_path: Path, target: str) -> None:
        """Acceptance 5."""
        lf: dict[str, str | bytes] = {
            "pack.yaml": MANIFEST + "includes: [part.j2]\n",
            "prompt.j2": "line one\n{% include 'part.j2' %}\n",
            "part.j2": "part\n",
        }
        _pack(tmp_path, lf, "demo.lf")
        _pack(tmp_path, {"pack.yaml": str(lf["pack.yaml"]).replace("demo.note", "demo.lf")}, "demo.lf")
        assert prompts.load("demo.lf", tmp_path).render() == "line one\npart"
        crlf = {**lf, target: str(lf[target]).replace("\n", "\r\n").encode()}
        _pack(tmp_path, crlf)
        with pytest.raises(prompts.PromptPackInvalid, match=rf"{target} holds a carriage return"):
            prompts.load("demo.note", tmp_path)

    def test_a_lone_carriage_return_is_refused_too(self, tmp_path: Path) -> None:
        _pack(tmp_path, {"pack.yaml": MANIFEST, "prompt.j2": b"a\rb"})
        with pytest.raises(prompts.PromptPackInvalid, match="carriage return"):
            prompts.load("demo.note", tmp_path)

    def test_a_file_that_is_not_utf8_is_refused(self, tmp_path: Path) -> None:
        _pack(tmp_path, {"pack.yaml": MANIFEST, "prompt.j2": b"\xff\xfe"})
        with pytest.raises(prompts.PromptPackInvalid, match=r"prompt.j2 is not UTF-8"):
            prompts.load("demo.note", tmp_path)

    def test_an_unreadable_manifest_fails_closed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _pack(tmp_path, {"pack.yaml": MANIFEST, "prompt.j2": "x"})

        def refuse(_self: Path) -> bytes:
            raise PermissionError("denied")

        monkeypatch.setattr(Path, "read_bytes", refuse)
        with pytest.raises(prompts.PromptPackInvalid, match="unreadable: PermissionError"):
            prompts.load("demo.note", tmp_path)


class TestOneEnvironment:
    def test_render_builds_no_new_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Acceptance 9: the environment is built once, at load, and every render reuses it."""
        pack = prompts.load("demo.multi", PACKS)
        env = pack._env

        def no_new_env(*_a: object, **_k: object) -> None:
            raise AssertionError("render built an Environment")

        monkeypatch.setattr(_packs, "Environment", no_new_env)
        assert pack.render(a="1") == "first 1" and pack.render("second", b="2") == "second 2"
        assert pack.render_source("c {{ x }}", x="3") == "c 3"
        assert pack._env is env and env.get_template("prompt.j2") is env.get_template("prompt.j2")


class TestManifestAdditions:
    def test_label_is_the_neutral_alias_of_langfuse_label(self) -> None:
        m = prompts.PackManifest.model_validate(yaml.safe_load(MANIFEST + "label: staging\n"))
        assert m.label == m.langfuse_label == "staging"
        legacy = prompts.PackManifest.model_validate(yaml.safe_load(MANIFEST + "langfuse_label: canary\n"))
        assert legacy.label == "canary"
        with pytest.raises(ValidationError, match="Extra inputs"):
            prompts.PackManifest.model_validate(yaml.safe_load(MANIFEST + "label: a\nlangfuse_label: b\n"))

    def test_schema_version_defaults_to_one_and_a_newer_one_fails_closed(self) -> None:
        assert prompts.PackManifest.model_validate(yaml.safe_load(MANIFEST)).schema_version == prompts.SCHEMA_VERSION
        with pytest.raises(ValidationError, match="schema_version 2"):
            prompts.PackManifest.model_validate(yaml.safe_load(MANIFEST + "schema_version: 2\n"))
        assert prompts.PackManifest.model_validate(yaml.safe_load(MANIFEST + "schema_version: 1\n")).schema_version == 1

    @pytest.mark.parametrize(
        ("line", "msg"),
        [("label: Not A Word\n", "lowercase word"), ("schema_version: 3\n", "schema_version 3")],
    )
    def test_a_bad_manifest_fails_at_load_naming_the_defect(self, tmp_path: Path, line: str, msg: str) -> None:
        _pack(tmp_path, {"pack.yaml": MANIFEST + line, "prompt.j2": "x"})
        with pytest.raises(prompts.PromptPackInvalid, match=f"manifest invalid: .*{msg}"):
            prompts.load("demo.note", tmp_path)
        with pytest.raises(prompts.PromptPackInvalid, match=r"demo.note"):
            prompts.registry(tmp_path)
        _pack(tmp_path, {"pack.yaml": MANIFEST})
        assert [p.id for p in prompts.registry(tmp_path)] == ["demo.note"]

    def test_a_template_key_is_a_lowercase_identifier(self) -> None:
        with pytest.raises(ValidationError, match="lowercase identifier"):
            prompts.PackManifest.model_validate(yaml.safe_load(MANIFEST + "  Bad-Key: x.j2\n"))

    @pytest.mark.parametrize(
        ("graders", "ok"),
        [("[echo, no_refusal]", True), ("[pkg.mod:TABLE]", True), ("[echo, echo]", False), ("['not a name']", False)],
    )
    def test_graders_are_names_or_dotted_paths_named_once(self, graders: str, ok: bool) -> None:
        doc = yaml.safe_load(MANIFEST + f"graders: {graders}\ngolden: g.json\n")
        if ok:
            assert prompts.PackManifest.model_validate(doc).evaluable
        else:
            with pytest.raises(ValidationError):
                prompts.PackManifest.model_validate(doc)

    def test_an_output_schema_must_be_a_dotted_path(self) -> None:
        with pytest.raises(ValidationError, match="dotted path"):
            prompts.PackManifest.model_validate(yaml.safe_load(MANIFEST + "output_schema: nope\n"))


def test_served_and_unevaluated_read_identities_as_well_as_shas() -> None:
    by_sha = prompts.served_shas(PACKS)
    by_identity = prompts.served_shas(PACKS, key="identity")
    assert set(by_sha) == set(by_identity) and by_identity["demo.one"] == prompts.load("demo.one", PACKS).identity
    assert prompts.unevaluated(by_identity, {**by_identity, "demo.one": by_sha["demo.one"]}) == ["demo.one"]
