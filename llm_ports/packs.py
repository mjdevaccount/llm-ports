"""packs -- load and fingerprint prompt packs: `<root>/<id>/pack.yaml` + the `.j2` it names.

The repo file is the truth, a content SHA names the text, a managed store (if any) holds a
labelled mirror. A pack ships with the code it serves; the consumer names the root directory.

CONTINUITY BY CONSTRUCTION. `Pack.content_sha` is sha256[:16] over the template SOURCE, and a
single-template pack hashes exactly that one source, so a prompt that MOVES into a pack keeps the
sha its recording was graded under. Multi-template packs hash `first + NEWLINE + second ...` in
manifest order. Over source, never over a render: a render depends on the context. A persisted
stamp may never change algorithm (a changed stamp for unchanged content reads as a config change
that never happened), so `sha` is frozen and tests pin it on fixed inputs.

The Jinja2 environment: StrictUndefined, trim_blocks, lstrip_blocks, no autoescape, no trailing
newline. A missing slot RAISES.

FAIL CLOSED ON THE PACK. A malformed manifest, a template that will not parse, a file the
manifest names but the directory lacks: `PromptPackInvalid`, never a fallback to an inline string.
`scan()` keeps every invalid pack as a NAMED row (`id -> reason`) so a gauge or a health block can
say which one, instead of a quiet skip. A managed-store mirror is a `PromptSource` adapter's job;
it never decides what the text IS -- the repo does, at every boot.

Zero DB, zero network: this module reads files and hashes text.
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
    # FROZEN knobs: a template moved into a pack must render byte-equal to where it came from.
    return Environment(loader=FileSystemLoader(str(pack_dir)), autoescape=False,
                       undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True,
                       keep_trailing_newline=False)


@dataclass(frozen=True)
class Pack:
    manifest: PackManifest
    dir: Path
    sources: dict[str, str]        # template name -> source text, manifest order
    content_sha: str

    @property
    def id(self) -> str:
        return self.manifest.id

    @property
    def source(self) -> str:
        """The primary template's source -- what a single-template pack's sha is over."""
        return self.sources[PRIMARY_TEMPLATE]

    def stamp(self) -> dict[str, str]:
        """What a writer puts beside an output row: `{pack, version, sha}` (the marker says WHAT
        changed -- version says the kind, sha says the text)."""
        return {"pack": self.id, "version": self.manifest.version, "sha": self.content_sha}

    def stamp_str(self) -> str:
        return f"{self.id}@{self.manifest.version}+{self.content_sha}"

    def render(self, name: str = PRIMARY_TEMPLATE, /, **ctx: Any) -> str:
        """Render one template of this pack. StrictUndefined: a missing slot raises."""
        return _env(self.dir).get_template(self.manifest.templates[name]).render(**ctx)

    def render_source(self, source: str, **ctx: Any) -> str:
        """Render CANDIDATE text under this pack's environment (the eval judges text the image
        does not carry yet -- the box runs the old image while a PR is open)."""
        return _env(self.dir).from_string(source).render(**ctx)


def sha(text: str, *, n: int = 16) -> str:
    """sha256 hex over a string, truncated to n chars. FROZEN (continuity): stamps are persisted."""
    return hashlib.sha256(text.encode()).hexdigest()[:n]


def content_sha(sources: list[str]) -> str:
    """`sha` over the sources joined by one newline; ONE source hashes as itself (continuity)."""
    return sha(sources[0] + "".join("\n" + s for s in sources[1:]))


def load(pack_id: str, root: Path) -> Pack:
    """Read, validate and fingerprint one pack. Raises PromptPackInvalid naming the defect."""
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
    """Every pack directory under `root`: (loaded packs, {id: why} for the ones that failed). A
    directory with no manifest is not a pack (a README beside the packs is not a finding). An
    invalid pack is a NAMED row here, never a quiet skip."""
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
    """Every pack, STRICT: one invalid pack raises (a consumer's `every pack loads` test is this call).
    Runtime readers that must keep serving the valid ones use `scan()`."""
    packs, invalid = scan(root)
    if invalid:
        raise PromptPackInvalid("; ".join(invalid.values()))
    return packs


def served_shas(root: Path) -> dict[str, str]:
    """{pack id: content sha} of every LOADABLE pack -- the 'served' side of the unevaluated gauge."""
    packs, _ = scan(root)
    return {p.id: p.content_sha for p in packs}


def unevaluated(served: dict[str, str], evaluated: dict[str, str]) -> list[str]:
    """Pack ids whose SERVED sha is not the sha of their newest pass|fail grade: a pack change
    reads as due until it has been graded. ONE definition, for every reader of "unevaluated"."""
    return sorted(pid for pid, sha in served.items() if evaluated.get(pid) != sha)
