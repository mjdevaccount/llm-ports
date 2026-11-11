"""Load and fingerprint prompt packs: `<root>/<id>/pack.yaml` and the `.j2` templates it names.

The repo file is the source of truth, a content sha names the text, and a managed store, if any,
holds a labelled mirror. A pack ships with the code it serves; the consumer names the root
directory. Mirroring to a managed store is a `PromptSource` adapter's job, and the mirror never
decides what the text is.

A pack has two names:

- `content_sha` is sha256, cut to 16 hex characters, over the template source. A single-template
  pack hashes exactly that one source; a multi-template pack hashes the sources joined by one
  newline, in manifest order. It never covers the manifest, and it is frozen as it is.
- `identity` is `byte_sha` over the manifest's bytes, then every file of the closed set by
  sorted relative path. A version bump, a re-pointed grader, a renamed template key or an edited
  include moves it. Two grades are comparable only when their identities match.

The closed set is `templates` plus `includes`. Templates render from exactly those files: a
reference to any other name, or a dynamic include whose target cannot be read statically, fails
at load. One Jinja2 environment is built per pack at load, with `StrictUndefined`, `trim_blocks`
and `lstrip_blocks`, no autoescape and no trailing newline; `render` reuses it.

A pack file holding a carriage return is refused, so `identity` cannot differ by the platform a
repo was checked out on (`* text=auto eol=lf` in `.gitattributes` keeps a checkout LF).

Invariants:
    - No database and no network: the module reads files and hashes bytes.
    - A prompt that moves into a single-template pack keeps its content sha.
    - Neither algorithm ever changes; tests pin both on fixed inputs.
    - A missing template variable raises at render time.
    - A pack fails closed: a malformed manifest, a template that does not parse or reaches outside
      the closed set, a carriage return, or a file the manifest names but the directory lacks
      raises `PromptPackInvalid`, never falls back to an inline string. `scan` reports each
      invalid pack by id instead of skipping it.

Rationale: ADR-0008, ADR-0026.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from jinja2 import DictLoader, Environment, StrictUndefined, TemplateSyntaxError, meta
from pydantic import ValidationError

from llm_ports.ports import Seed
from llm_ports.prompts._identity import byte_sha
from llm_ports.prompts._manifest import PRIMARY_TEMPLATE, PackManifest

if TYPE_CHECKING:
    from llm_ports.prompts._resolve import GraderRegistry, Registry, ResolvedPack

MANIFEST = "pack.yaml"


class PromptPackInvalid(ValueError):
    """A pack that cannot be served. The message names the pack and the defect."""


def _env(files: Mapping[str, str]) -> Environment:
    # These settings are fixed: a template moved into a pack renders byte-equal to its old output.
    return Environment(
        loader=DictLoader(dict(files)),
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
    )


@dataclass(frozen=True)
class Pack:
    """One loaded prompt pack. `load` builds it.

    Attributes:
        manifest: The validated manifest.
        dir: The pack's directory.
        sources: Template name to source text, in manifest order.
        content_sha: The `content_sha` of the sources.
        blobs: Relative path to the bytes read at load: `pack.yaml` and every file of the closed
            set. A pack built by hand without them has an identity over its templates alone.
    """

    manifest: PackManifest
    dir: Path
    sources: dict[str, str]
    content_sha: str
    blobs: Mapping[str, bytes] = field(default_factory=dict, repr=False, compare=False)
    _env: Environment = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.blobs:
            blobs = {self.manifest.templates[k]: v.encode() for k, v in self.sources.items()}
            object.__setattr__(self, "blobs", blobs)
        texts = {f: b.decode() for f, b in self.blobs.items() if f != MANIFEST}
        object.__setattr__(self, "_env", _env(texts))

    @property
    def id(self) -> str:
        """The pack id, `feature.action`."""
        return self.manifest.id

    @property
    def source(self) -> str:
        """The primary template's source; a single-template pack's sha is over this text."""
        return self.sources[PRIMARY_TEMPLATE]

    @cached_property
    def identity(self) -> str:
        """`byte_sha` over the manifest's bytes, then each file of the closed set by sorted path.

        Rationale: ADR-0026.
        """
        head = [(MANIFEST, self.blobs[MANIFEST])] if MANIFEST in self.blobs else []
        return byte_sha(head + [(f, self.blobs[f]) for f in self.files()])

    def files(self) -> tuple[str, ...]:
        """Return the closed template set: every file a render may read, sorted, relative to `dir`."""
        return tuple(sorted(f for f in self.blobs if f != MANIFEST))

    def stamp(self) -> dict[str, str]:
        """Return the stamp a writer stores beside an output row: `{pack, version, sha, identity}`.

        The version says what kind of change the pack had; `sha` says whether its template text
        changed; `identity` says whether anything that changes its behaviour did.
        """
        return {"pack": self.id, "version": self.manifest.version, "sha": self.content_sha, "identity": self.identity}

    def stamp_str(self) -> str:
        """Return the stamp as one string, `<id>@<version>+<sha>`."""
        return f"{self.id}@{self.manifest.version}+{self.content_sha}"

    def render(self, name: str = PRIMARY_TEMPLATE, /, **ctx: Any) -> str:
        """Render the template `name` of this pack with `ctx`, in the environment built at load.

        Raises:
            jinja2.UndefinedError: The template reads a variable `ctx` does not supply.
        """
        return self._env.get_template(self.manifest.templates[name]).render(**ctx)

    def render_source(self, source: str, **ctx: Any) -> str:
        """Render candidate template text under this pack's environment and closed set.

        Lets an eval grade a template revision that is not yet the pack's installed source.

        Raises:
            jinja2.TemplateNotFound: The candidate includes a file outside the closed set.
        """
        return self._env.from_string(source).render(**ctx)

    def seed(self, name: str = PRIMARY_TEMPLATE) -> Seed:
        """Return the repo `Seed` of template `name`, the text a managed mirror copies.

        `sha` is the pack's `content_sha`. `config` carries `pack`, `version`, `role`, `template`,
        `content_sha`, `identity` and `label` (the manifest's mirror label), so a mirror holding a
        stale identity is re-pushed and serves under the manifest's label.

        Raises:
            KeyError: The pack has no template `name`.
        """
        m = self.manifest
        config = {
            "pack": self.id,
            "version": m.version,
            "role": m.role,
            "template": name,
            "content_sha": self.content_sha,
            "identity": self.identity,
            "label": m.label,
        }
        return Seed(self.sources[name], self.content_sha, config)

    def resolved(
        self, *, registry: GraderRegistry | Registry | None = None, import_roots: Sequence[str] = ()
    ) -> ResolvedPack:
        """Return `prompts.resolve(self, registry=registry, import_roots=import_roots)`.

        Raises:
            PromptPackInvalid: A manifest pointer does not resolve; the message names the field.
        """
        from llm_ports.prompts._resolve import resolve  # _resolve imports this module

        return resolve(self, registry=registry, import_roots=import_roots)


def sha(text: str, *, n: int = 16) -> str:
    """Return the sha256 hex digest of `text`, cut to `n` characters.

    Invariants:
        - The algorithm never changes: stamps built on it are persisted.
    """
    return hashlib.sha256(text.encode()).hexdigest()[:n]


def content_sha(sources: list[str]) -> str:
    """Return `sha` over the sources joined by one newline; a single source hashes as itself."""
    return sha(sources[0] + "".join("\n" + s for s in sources[1:]))


def _read(pack_id: str, path: Path, what: str) -> bytes:
    data = path.read_bytes()
    if b"\r" in data:
        raise PromptPackInvalid(f"{pack_id}: {what} holds a carriage return (CRLF); pack files are LF only")
    return data


def _text(pack_id: str, data: bytes, what: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PromptPackInvalid(f"{pack_id}: {what} is not UTF-8: {exc.reason}") from exc


def _manifest(pack_id: str, data: bytes) -> PackManifest:
    try:
        raw = yaml.safe_load(_text(pack_id, data, MANIFEST))
    except yaml.YAMLError as exc:
        raise PromptPackInvalid(f"{pack_id}: {MANIFEST} unreadable: {type(exc).__name__}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PromptPackInvalid(f"{pack_id}: {MANIFEST} is not a mapping")
    try:
        manifest = PackManifest.model_validate(raw)
    except ValidationError as exc:
        raise PromptPackInvalid(f"{pack_id}: manifest invalid: {exc.errors()[0]['msg']}") from exc
    if manifest.id != pack_id:
        raise PromptPackInvalid(f"{pack_id}: manifest id {manifest.id!r} must equal the directory name")
    return manifest


def _closed(pack_id: str, texts: Mapping[str, str]) -> None:
    """Refuse a file that does not parse, or that references a name outside `texts`."""
    env = _env(texts)
    for file, src in texts.items():
        try:
            ast = env.parse(src)
        except TemplateSyntaxError as exc:
            raise PromptPackInvalid(f"{pack_id}: template {file!r} line {exc.lineno}: {exc.message}") from exc
        for ref in meta.find_referenced_templates(ast):
            if ref is None:
                raise PromptPackInvalid(f"{pack_id}: template {file!r} has a dynamic include, which cannot be checked")
            if ref not in texts:
                raise PromptPackInvalid(f"{pack_id}: template {file!r} references {ref!r}, which is not in `includes`")


def load(pack_id: str, root: Path) -> Pack:
    """Read, validate and fingerprint the pack `pack_id` under `root`.

    Raises:
        PromptPackInvalid: The manifest is missing, unreadable or invalid, or its id is not the
            directory name; a file of the closed set is missing, not UTF-8 or does not parse; a
            template references a file outside the closed set or includes dynamically; or a pack
            file holds a carriage return. The message names the pack and the defect.
    """
    pack_dir = root / pack_id
    manifest_path = pack_dir / MANIFEST
    if not manifest_path.is_file():
        raise PromptPackInvalid(f"{pack_id}: no {MANIFEST} under {pack_dir}")
    try:
        blobs = {MANIFEST: _read(pack_id, manifest_path, MANIFEST)}
    except OSError as exc:
        raise PromptPackInvalid(f"{pack_id}: {MANIFEST} unreadable: {type(exc).__name__}: {exc}") from exc
    manifest = _manifest(pack_id, blobs[MANIFEST])
    named = {file: f"template {name!r}" for name, file in manifest.templates.items()}
    for file in manifest.includes:
        named.setdefault(file, "includes")
    texts: dict[str, str] = {}
    inside = pack_dir.resolve()
    for file, what in named.items():
        if not (pack_dir / file).is_file():
            raise PromptPackInvalid(f"{pack_id}: {what} names {file!r}, which is not in the pack")
        if not (pack_dir / file).resolve().is_relative_to(inside):
            raise PromptPackInvalid(f"{pack_id}: {what} names {file!r}, which resolves outside the pack")
        blobs[file] = _read(pack_id, pack_dir / file, file)
        texts[file] = _text(pack_id, blobs[file], file)
    _closed(pack_id, texts)
    sources = {name: texts[file] for name, file in manifest.templates.items()}
    return Pack(manifest, pack_dir, sources, content_sha(list(sources.values())), blobs)


def scan(
    root: Path, *, registry: GraderRegistry | Registry | None = None, import_roots: Sequence[str] | None = None
) -> tuple[tuple[Pack, ...], dict[str, str]]:
    """Load every pack directory under `root`, keeping the ones that fail by name.

    A directory with no manifest is not a pack and is not reported. A missing `root` returns
    `((), {})`. Given `registry` or `import_roots`, each loaded pack is also resolved with them
    (`prompts.resolve`), and a pack that does not resolve is reported beside the ones that do not
    load.

    Returns:
        The loaded packs, and `{id: reason}` for each invalid pack.
    """
    from llm_ports.prompts._resolve import resolve  # _resolve imports this module

    packs: list[Pack] = []
    invalid: dict[str, str] = {}
    if not root.is_dir():
        return (), {}
    check = registry is not None or import_roots is not None
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / MANIFEST).is_file()):
        try:
            pack = load(d.name, root)
            if check:
                resolve(pack, registry=registry, import_roots=import_roots or ())
            packs.append(pack)
        except PromptPackInvalid as exc:
            invalid[d.name] = str(exc)
    return tuple(packs), invalid


def registry(
    root: Path, *, registry: GraderRegistry | Registry | None = None, import_roots: Sequence[str] | None = None
) -> tuple[Pack, ...]:
    """Load every pack under `root`, strictly.

    Suits a boot path or a test that every pack loads. Given `registry` or `import_roots`, every
    pack is also resolved with them (`scan`), so a mistyped grader or output schema fails here. A
    runtime reader that must keep serving the valid packs uses `scan` instead.

    Raises:
        PromptPackInvalid: Any pack is invalid; the message joins every pack's reason.
    """
    packs, invalid = scan(root, registry=registry, import_roots=import_roots)
    if invalid:
        raise PromptPackInvalid("; ".join(invalid.values()))
    return packs


def served_shas(root: Path, *, key: Literal["content_sha", "identity"] = "content_sha") -> dict[str, str]:
    """Return `{pack id: content sha}` (or identity) for every loadable pack: the served side of `unevaluated`."""
    packs, _ = scan(root)
    return {p.id: p.identity if key == "identity" else p.content_sha for p in packs}


def unevaluated(served: dict[str, str], evaluated: dict[str, str]) -> list[str]:
    """Return the pack ids whose served name is not the name of their newest pass or fail grade.

    A changed pack reads as due until it has been graded. This is the one definition of
    "unevaluated"; every reader uses it. Both sides name packs the same way: content shas, or
    identities (`served_shas(root, key="identity")`), never one of each.

    Args:
        served: `{pack id: content sha or identity}` as served (see `served_shas`).
        evaluated: `{pack id: the same}` of each pack's newest pass or fail grade.
    """
    return sorted(pid for pid, sha in served.items() if evaluated.get(pid) != sha)


__all__ = [
    "MANIFEST",
    "Pack",
    "PromptPackInvalid",
    "content_sha",
    "load",
    "registry",
    "scan",
    "served_shas",
    "sha",
    "unevaluated",
]
