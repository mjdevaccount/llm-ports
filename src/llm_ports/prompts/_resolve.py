"""Resolve a pack manifest's pointers: its output type, its graders and its golden set.

`resolve(pack, registry=..., import_roots=...)` turns the manifest's strings into the things they
name, strictly, so a typo is found when the pack is resolved, not when an eval first reaches it:

- `output_schema` is a `module:attr` (or `module.attr`) path, imported only from a module under one
  of `import_roots`. The library imports nothing by default: with no roots, a pack that names an
  output schema does not resolve. The attribute must be a pydantic model class or a `JsonOutput`
  subclass; `output_schema_sha` is `byte_sha` of its canonical JSON schema.
- `graders` are names, each looked up in `registry`. This layer sits below the call graders, so it
  has no default table: `evals.golden.resolve_pack` falls back to `call.graders.GRADERS`.
- `golden` is a path relative to the pack directory and must stay inside it.

Any miss raises `PromptPackInvalid` naming the pack and the field.

Invariants:
    - Imports only what a pack names under a root its caller listed.
    - Never resolves a grader by importing its name: the registry is the only source.

Rationale: ADR-0026.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

from llm_ports.ports import JsonOutput
from llm_ports.prompts._identity import byte_sha
from llm_ports.prompts._packs import PromptPackInvalid

if TYPE_CHECKING:
    from llm_ports.prompts._packs import Pack

#: A grader as this layer sees it: a callable it hands on without calling.
AnyGrader = Callable[..., Any]
#: A plain grader table (`{**call.graders.GRADERS, "mine": fn}`), accepted wherever a registry is.
Registry = Mapping[str, AnyGrader]


@runtime_checkable
class GraderRegistry(Protocol):
    """Grader names to graders, owned by the consumer's composition root."""

    def get(self, name: str) -> AnyGrader:
        """Return the grader named `name`.

        Raises:
            KeyError: No grader has that name.
        """
        ...


@dataclass(frozen=True, slots=True)
class ResolvedPack:
    """A pack with its manifest's pointers resolved.

    Attributes:
        pack: The pack.
        output: The imported output type, or None for a prose pack.
        graders: The manifest's grader names, in manifest order, to the graders they resolved to.
        golden: The golden set's path inside the pack directory, or None when the pack has none.
            Not checked for existence: `evals.golden.fire_pack` reports a missing file.
        output_schema_sha: `byte_sha` of the canonical JSON schema of `output`; None with no output.
    """

    pack: Pack
    output: type[BaseModel] | type[JsonOutput] | None
    graders: Mapping[str, AnyGrader]
    golden: Path | None
    output_schema_sha: str | None


def _invalid(pack: Pack, field: str, why: str) -> PromptPackInvalid:
    return PromptPackInvalid(f"{pack.id}: {field}: {why}")


def _import(pack: Pack, path: str, roots: Sequence[str]) -> type[BaseModel] | type[JsonOutput]:
    module, _, attr = path.partition(":") if ":" in path else path.rpartition(".")
    if not any(module == r or module.startswith(r + ".") for r in roots):
        raise _invalid(pack, "output_schema", f"module {module!r} is under none of the import roots {list(roots)}")
    try:
        obj = getattr(importlib.import_module(module), attr)
    except Exception as exc:  # any import-time failure is the pointer's defect, named
        raise _invalid(pack, "output_schema", f"{path!r} does not import: {type(exc).__name__}: {exc}") from exc
    if isinstance(obj, type) and issubclass(obj, BaseModel | JsonOutput):
        return obj
    raise _invalid(pack, "output_schema", f"{path!r} is not a pydantic model or a JsonOutput")


def schema_sha(output: type[BaseModel] | type[JsonOutput]) -> str:
    """Return `byte_sha` of the output type's JSON schema, serialized canonically (sorted keys, compact).

    Pure. A `JsonOutput` hashes the schema it carries; a pydantic model hashes `model_json_schema()`.
    """
    schema = dict(output.json_schema) if issubclass(output, JsonOutput) else output.model_json_schema()
    canon = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return byte_sha([("output_schema", canon.encode())])


def _grader(pack: Pack, name: str, registry: GraderRegistry | Registry | None) -> AnyGrader:
    if registry is None:
        raise _invalid(pack, "graders", f"{name!r} cannot resolve: no grader registry was passed")
    try:
        fn = registry[name] if isinstance(registry, Mapping) else registry.get(name)
    except KeyError:
        fn = None
    if fn is None:
        raise _invalid(pack, "graders", f"{name!r} is not in the grader registry")
    return fn


def _golden(pack: Pack, golden: str) -> Path:
    base = pack.dir.resolve()
    path = (base / golden).resolve()
    if not path.is_relative_to(base):
        raise _invalid(pack, "golden", f"{golden!r} is outside the pack directory")
    return path


def resolve(
    pack: Pack, *, registry: GraderRegistry | Registry | None = None, import_roots: Sequence[str] = ()
) -> ResolvedPack:
    """Resolve the pack's `output_schema`, `graders` and `golden`, strictly.

    Args:
        pack: A loaded pack.
        registry: Where grader names resolve: a `GraderRegistry` or a plain mapping. None resolves
            no grader, so a pack that names one does not resolve.
        import_roots: Module prefixes an `output_schema` may be imported from; empty imports
            nothing.

    Returns:
        The resolved pack.

    Raises:
        PromptPackInvalid: A pointer does not resolve; the message names the pack and the field.
    """
    m = pack.manifest
    output = _import(pack, m.output_schema, import_roots) if m.output_schema else None
    graders = {name: _grader(pack, name, registry) for name in m.graders}
    golden = _golden(pack, m.golden) if m.golden else None
    return ResolvedPack(pack, output, graders, golden, schema_sha(output) if output else None)


__all__ = ["AnyGrader", "GraderRegistry", "Registry", "ResolvedPack", "resolve", "schema_sha"]
