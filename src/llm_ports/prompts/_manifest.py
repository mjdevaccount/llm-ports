"""The manifest a prompt pack's `<packs root>/<id>/pack.yaml` must satisfy.

A prompt pack is a directory holding this manifest and the Jinja2 templates it names.

Invariants:
    - Pure data: no I/O.
    - Strict (`extra="forbid"`): a field the model does not declare fails validation.
"""

from __future__ import annotations

import re
from typing import ClassVar

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

_ID = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
_DOTTED = re.compile(r"^[A-Za-z_][\w]*(\.[A-Za-z_]\w*)*[:.][A-Za-z_]\w*$")
_TEMPLATE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_GRADER_NAME = re.compile(r"^[A-Za-z_][\w]*$")
PRIMARY_TEMPLATE = "prompt"
#: The manifest schema this library reads. A manifest declaring a newer one fails closed.
SCHEMA_VERSION = 1


def _is_pack_file(file: str) -> bool:
    return not ("/" in file or "\\" in file or file.startswith(".") or not file.endswith(".j2"))


def _include_defect(file: str) -> str:
    """Return why `file` is not a relative POSIX path inside the pack, or "" when it is."""
    if not file or "\\" in file or ":" in file or file.startswith("/"):
        return "a relative POSIX path (`/` separators, no leading `/`, no drive)"
    if any(part in ("", ".", "..") or part.startswith(".") for part in file.split("/")):
        return "no `..`, no empty or hidden segment"
    if file == "pack.yaml":
        return "not the manifest"
    return ""


class PackManifest(BaseModel):
    """A validated, frozen `pack.yaml`.

    Attributes:
        id: `feature.action`. Equals the pack's directory name and the managed prompt's name, so
            the stamp on an output row, the prompt-store link and the repo path agree by name.
        version: Semver. MAJOR: the output schema changed. MINOR: an instruction or claim-rule
            changed. PATCH: wording. The content sha, computed by the loader and never written
            here, says whether the text changed; the version says what kind of change it was.
        role: The model alias or `Role` name the pack is written for.
        templates: Template name to a `.j2` file inside the pack directory. `prompt` is the
            primary template and is required. The content sha covers every template in manifest
            order, and a single-template pack hashes that template alone.
        output_schema: `module:attr` (or `module.attr`) path of the pydantic class or `JsonOutput`
            the output parses as; None for prose. `prompts.resolve` imports it from the roots
            its caller names.
        graders: Grader names, looked up by `prompts.resolve` in the caller's grader registry. A
            dotted path is accepted as a name. A pack with graders is evaluable and requires
            `golden`.
        golden: Path of the golden set the graders replay, relative to the pack directory.
        langfuse_label: The label a managed mirror serves under (YAML key `langfuse_label` or
            its neutral alias `label`). The managed copy mirrors the repo; it never leads.
        includes: Files a template reaches by `{% include %}`, `{% import %}` or `{% extends %}`
            that are not in `templates`: relative POSIX paths inside the pack directory, any
            extension (`partials/header.j2`, `rubric.md`). With `templates`, the closed set: no
            other file is ever rendered.
        schema_version: The manifest schema the pack is written to; only `SCHEMA_VERSION` loads.

    Raises:
        pydantic.ValidationError: A field is unknown, malformed, or `graders` lacks `golden`.

    Rationale: ADR-0008.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    id: str
    version: str
    role: str = Field(min_length=1)
    templates: dict[str, str] = Field(min_length=1)
    output_schema: str | None = None
    graders: list[str] = Field(default_factory=list)
    golden: str | None = None
    langfuse_label: str = Field("production", validation_alias=AliasChoices("langfuse_label", "label"))
    includes: list[str] = Field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    @field_validator("id")
    @classmethod
    def _id_is_feature_dot_action(cls, v: str) -> str:
        if not _ID.match(v):
            raise ValueError(f"pack id {v!r} must be `feature.action` (lowercase, one dot)")
        return v

    @field_validator("version")
    @classmethod
    def _version_is_semver(cls, v: str) -> str:
        if not _SEMVER.match(v):
            raise ValueError(f"version {v!r} is not MAJOR.MINOR.PATCH")
        return v

    @field_validator("templates")
    @classmethod
    def _templates_are_files_in_the_pack(cls, v: dict[str, str]) -> dict[str, str]:
        if PRIMARY_TEMPLATE not in v:
            raise ValueError(f"templates must name {PRIMARY_TEMPLATE!r} (the primary template)")
        for name, file in v.items():
            if not _TEMPLATE_NAME.match(name):
                raise ValueError(f"template name {name!r}: lowercase identifier")
            if not _is_pack_file(file):
                raise ValueError(f"template {name!r} -> {file!r}: a `.j2` file INSIDE the pack dir, no paths")
        return v

    @field_validator("includes")
    @classmethod
    def _includes_are_files_in_the_pack(cls, v: list[str]) -> list[str]:
        for file in v:
            if why := _include_defect(file):
                raise ValueError(f"includes {file!r}: a file INSIDE the pack dir: {why}")
        return v

    @field_validator("output_schema")
    @classmethod
    def _dotted_path(cls, v: str | None) -> str | None:
        if v is not None and not _DOTTED.match(v):
            raise ValueError(f"{v!r} is not a dotted path (`pkg.module:NAME` or `pkg.module.NAME`)")
        return v

    @field_validator("graders")
    @classmethod
    def _grader_names(cls, v: list[str]) -> list[str]:
        for item in v:
            if not (_GRADER_NAME.match(item) or _DOTTED.match(item)):
                raise ValueError(f"grader {item!r} is neither a name nor a dotted path")
        if len(set(v)) != len(v):
            raise ValueError(f"graders {v!r} name one grader twice")
        return v

    @field_validator("schema_version")
    @classmethod
    def _schema_version_is_known(cls, v: int) -> int:
        if v != SCHEMA_VERSION:
            raise ValueError(f"schema_version {v} is not {SCHEMA_VERSION}, the one this library reads")
        return v

    @field_validator("langfuse_label")
    @classmethod
    def _label_is_a_word(cls, v: str) -> str:
        if not re.match(r"^[a-z][a-z0-9_-]*$", v):
            raise ValueError(f"langfuse_label {v!r}: lowercase word")
        return v

    @model_validator(mode="after")
    def _graders_need_a_golden_set(self) -> PackManifest:
        if self.graders and not self.golden:
            raise ValueError("a pack with graders needs `golden` -- graders without a golden set grade nothing")
        return self

    @property
    def feature(self) -> str:
        """The `feature` half of `id`."""
        return self.id.split(".", 1)[0]

    @property
    def label(self) -> str:
        """The managed mirror's label; the neutral name of `langfuse_label`."""
        return self.langfuse_label

    @property
    def evaluable(self) -> bool:
        """True when the pack declares graders and a golden set, so an eval can grade it."""
        return bool(self.graders) and self.golden is not None


__all__ = ["PRIMARY_TEMPLATE", "SCHEMA_VERSION", "PackManifest"]
