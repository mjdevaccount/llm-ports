"""Load and fingerprint prompt packs: `<root>/<id>/pack.yaml` and the `.j2` templates it names.

The repo file is the source of truth, a content sha names the text, and a managed store, if any,
holds a labelled mirror. A pack ships with the code it serves; the consumer names the root
directory. Mirroring to a managed store is a `PromptSource` adapter's job, and the mirror never
decides what the text is.

`Pack.content_sha` is sha256, cut to 16 hex characters, over the template source. A
single-template pack hashes exactly that one source; a multi-template pack hashes the sources
joined by one newline, in manifest order. The hash is over the source, never a render, because a
render depends on its context.

Templates render in a Jinja2 environment with `StrictUndefined`, `trim_blocks` and
`lstrip_blocks`, no autoescape and no trailing newline.

Invariants:
    - No database and no network: the module reads files and hashes text.
    - A prompt that moves into a single-template pack keeps its content sha.
    - The sha algorithm never changes; tests pin it on fixed inputs.
    - A missing template variable raises at render time.
    - A pack fails closed: a malformed manifest, a template that does not parse, or a file the
      manifest names but the directory lacks raises `PromptPackInvalid`, never falls back to an
      inline string. `scan` reports each invalid pack by id instead of skipping it.

Rationale: ADR-0008.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateSyntaxError
from pydantic import ValidationError

from llm_ports.manifest import PRIMARY_TEMPLATE, PackManifest

MANIFEST = "pack.yaml"


class PromptPackInvalid(ValueError):
    """A pack that cannot be served. The message names the pack and the defect."""


def _env(pack_dir: Path) -> Environment:
    # These settings are fixed: a template moved into a pack renders byte-equal to its old output.
    return Environment(
        loader=FileSystemLoader(str(pack_dir)),
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
    )


@dataclass(frozen=True)
class Pack:
    """One loaded prompt pack.

    Attributes:
        manifest: The validated manifest.
        dir: The pack's directory.
        sources: Template name to source text, in manifest order.
        content_sha: The `content_sha` of the sources.
    """

    manifest: PackManifest
    dir: Path
    sources: dict[str, str]
    content_sha: str

    @property
    def id(self) -> str:
        """The pack id, `feature.action`."""
        return self.manifest.id

    @property
    def source(self) -> str:
        """The primary template's source; a single-template pack's sha is over this text."""
        return self.sources[PRIMARY_TEMPLATE]

    def stamp(self) -> dict[str, str]:
        """Return the stamp a writer stores beside an output row: `{pack, version, sha}`.

        The version says what kind of change the text had; the sha says whether it changed.
        """
        return {"pack": self.id, "version": self.manifest.version, "sha": self.content_sha}

    def stamp_str(self) -> str:
        """Return the stamp as one string, `<id>@<version>+<sha>`."""
        return f"{self.id}@{self.manifest.version}+{self.content_sha}"

    def render(self, name: str = PRIMARY_TEMPLATE, /, **ctx: Any) -> str:
        """Render the template `name` of this pack with `ctx`.

        Raises:
            jinja2.UndefinedError: The template reads a variable `ctx` does not supply.
        """
        return _env(self.dir).get_template(self.manifest.templates[name]).render(**ctx)

    def render_source(self, source: str, **ctx: Any) -> str:
        """Render candidate template text under this pack's environment.

        Lets an eval grade a template revision that is not yet the pack's installed source.
        """
        return _env(self.dir).from_string(source).render(**ctx)


def sha(text: str, *, n: int = 16) -> str:
    """Return the sha256 hex digest of `text`, cut to `n` characters.

    Invariants:
        - The algorithm never changes: stamps built on it are persisted.
    """
    return hashlib.sha256(text.encode()).hexdigest()[:n]


def content_sha(sources: list[str]) -> str:
    """Return `sha` over the sources joined by one newline; a single source hashes as itself."""
    return sha(sources[0] + "".join("\n" + s for s in sources[1:]))


def load(pack_id: str, root: Path) -> Pack:
    """Read, validate and fingerprint the pack `pack_id` under `root`.

    Raises:
        PromptPackInvalid: The manifest is missing, unreadable or invalid, its id is not the
            directory name, or a template it names is missing or does not parse. The message
            names the pack and the defect.
    """
    pack_dir = root / pack_id
    manifest_path = pack_dir / MANIFEST
    if not manifest_path.is_file():
        raise PromptPackInvalid(f"{pack_id}: no {MANIFEST} under {pack_dir}")
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError) as exc:
        raise PromptPackInvalid(f"{pack_id}: {MANIFEST} unreadable: {type(exc).__name__}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PromptPackInvalid(f"{pack_id}: {MANIFEST} is not a mapping")
    try:
        manifest = PackManifest.model_validate(raw)
    except ValidationError as exc:
        raise PromptPackInvalid(f"{pack_id}: manifest invalid: {exc.errors()[0]['msg']}") from exc
    if manifest.id != pack_id:
        raise PromptPackInvalid(f"{pack_id}: manifest id {manifest.id!r} must equal the directory name")
    env = _env(pack_dir)
    sources: dict[str, str] = {}
    for name, file in manifest.templates.items():
        path = pack_dir / file
        if not path.is_file():
            raise PromptPackInvalid(f"{pack_id}: template {name!r} names {file!r}, which is not in the pack")
        src = path.read_text(encoding="utf-8")
        try:
            env.parse(src)
        except TemplateSyntaxError as exc:
            raise PromptPackInvalid(f"{pack_id}: template {file!r} line {exc.lineno}: {exc.message}") from exc
        sources[name] = src
    return Pack(manifest=manifest, dir=pack_dir, sources=sources, content_sha=content_sha(list(sources.values())))


def scan(root: Path) -> tuple[tuple[Pack, ...], dict[str, str]]:
    """Load every pack directory under `root`, keeping the ones that fail by name.

    A directory with no manifest is not a pack and is not reported. A missing `root` returns
    `((), {})`.

    Returns:
        The loaded packs, and `{id: reason}` for each invalid pack.
    """
    packs: list[Pack] = []
    invalid: dict[str, str] = {}
    if not root.is_dir():
        return (), {}
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / MANIFEST).is_file()):
        try:
            packs.append(load(d.name, root))
        except PromptPackInvalid as exc:
            invalid[d.name] = str(exc)
    return tuple(packs), invalid


def registry(root: Path) -> tuple[Pack, ...]:
    """Load every pack under `root`, strictly.

    Suits a test that every pack loads. A runtime reader that must keep serving the valid packs
    uses `scan` instead.

    Raises:
        PromptPackInvalid: Any pack is invalid; the message joins every pack's reason.
    """
    packs, invalid = scan(root)
    if invalid:
        raise PromptPackInvalid("; ".join(invalid.values()))
    return packs


def served_shas(root: Path) -> dict[str, str]:
    """Return `{pack id: content sha}` for every loadable pack: the served side of `unevaluated`."""
    packs, _ = scan(root)
    return {p.id: p.content_sha for p in packs}


def unevaluated(served: dict[str, str], evaluated: dict[str, str]) -> list[str]:
    """Return the pack ids whose served sha is not the sha of their newest pass or fail grade.

    A changed pack reads as due until it has been graded. This is the one definition of
    "unevaluated"; every reader uses it.

    Args:
        served: `{pack id: content sha}` as served (see `served_shas`).
        evaluated: `{pack id: sha}` of each pack's newest pass or fail grade.
    """
    return sorted(pid for pid, sha in served.items() if evaluated.get(pid) != sha)
