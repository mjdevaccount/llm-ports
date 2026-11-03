"""The manifest a prompt pack's `<packs root>/<id>/pack.yaml` must satisfy.

A prompt pack is a directory holding this manifest and the Jinja2 templates it names.

Invariants:
    - Pure data: no I/O.
    - Strict (`extra="forbid"`): a field the model does not declare fails validation.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_ID = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
_DOTTED = re.compile(r"^[A-Za-z_][\w]*(\.[A-Za-z_]\w*)*[:.][A-Za-z_]\w*$")
_TEMPLATE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
PRIMARY_TEMPLATE = "prompt"


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
        output_schema: Dotted path to the pydantic class the output parses as; None for prose.
        graders: Dotted paths to grader tables (`module:NAME`). A pack with graders is evaluable
            and requires `golden`.
        golden: Repo-relative path of the golden set the graders replay.
        langfuse_label: The label a managed mirror serves under. The managed copy mirrors the
            repo; it never leads.

    Raises:
        pydantic.ValidationError: A field is unknown, malformed, or `graders` lacks `golden`.

    Rationale: ADR-0008.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    version: str
    role: str = Field(min_length=1)
    templates: dict[str, str] = Field(min_length=1)
    output_schema: str | None = None
    graders: list[str] = Field(default_factory=list)
    golden: str | None = None
    langfuse_label: str = "production"

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
            if "/" in file or "\\" in file or file.startswith(".") or not file.endswith(".j2"):
                raise ValueError(f"template {name!r} -> {file!r}: a `.j2` file INSIDE the pack dir, no paths")
        return v

    @field_validator("output_schema", "graders")
    @classmethod
    def _dotted_paths(cls, v: str | list[str] | None) -> str | list[str] | None:
        for item in [v] if isinstance(v, str) else (v or []):
            if not _DOTTED.match(item):
                raise ValueError(f"{item!r} is not a dotted path (`pkg.module:NAME` or `pkg.module.NAME`)")
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
    def evaluable(self) -> bool:
        """True when the pack declares graders and a golden set, so an eval can grade it."""
        return bool(self.graders) and self.golden is not None
