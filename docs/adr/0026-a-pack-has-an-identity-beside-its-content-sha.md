---
status: accepted
date: 2026-11-11
---

# A pack has an identity beside its content sha

Decided in the release after 1.22.0. Extends ADR-0008; changes none of it.

## Context and problem statement

ADR-0008 names a pack by `content_sha`, a hash of its template sources. Three holes make that name
weaker than the thing it is used for, which is tying a grade to exactly what was graded:

1. **The manifest is excluded.** A version bump, a re-pointed grader, a different golden set or a
   renamed template key changes what the pack does under evaluation, and none of them changes
   `content_sha`.
2. **Re-split templates collide.** Sources are joined by a bare newline, so `["a\nb", "c"]` and
   `["a", "b\nc"]` hash alike.
3. **Included files are missed.** The loader was a filesystem loader over the pack directory, so a
   template could `{% include %}` any file there; the sha covered none of them, and the manifest
   did not name them.

The manifest's `output_schema`, `graders` and `golden` were validated for shape and read by
nothing: each consumer resolved the dotted paths itself, and a typo surfaced when an eval first
reached it.

`content_sha` cannot be repaired in place. Stamps beside stored output rows, mirrored prompts in a
managed store, and tests in consumers all hold today's values; a changed algorithm would read as a
configuration change that never happened.

## Decision drivers

- Every existing `content_sha` stays valid, byte for byte.
- A name that moves whenever behaviour can move, and does not move otherwise.
- No file a render can read is outside what the name covers.
- The name is the same on every checkout platform.
- A broken pointer in a manifest fails when the pack is resolved, not when an eval reaches it.

## Considered options

1. Change `content_sha` to cover the manifest and includes.
2. Hash `PackManifest.model_dump()` beside the sources.
3. A new name, `identity`, over raw bytes, with a closed template set.

## Decision outcome

Chosen option 3.

- **`byte_sha(parts)`**: sha256 over each `(name, data)` part as `name NUL len(data) NUL data`, in
  order, cut to 16 hex characters. The length prefix makes a re-split, a rename or a reorder a
  different input. It is a public primitive so a consumer can hash a wider scope (a pack plus its
  generated config and dataset) on the same rule, taking `Pack.identity` as its first part.
- **`Pack.identity`** is `byte_sha` over the manifest file's raw bytes, then every file of the
  closed set by sorted relative path. `Pack.stamp()` gains the key `identity`; `stamp_str()` and
  the `sha` key are unchanged.
- **The template set is closed.** `PackManifest.includes` names the files a template reaches by
  `{% include %}`, `{% import %}` or `{% extends %}`: relative POSIX paths inside the pack, any
  extension, subdirectories allowed (`partials/h.j2`, `rubric.md`), never `..` or a path that
  resolves outside it. The loader is a `DictLoader` over exactly
  `templates` and `includes`; at load, `jinja2.meta.find_referenced_templates` checks every file,
  and a reference outside the set, or a dynamic one that cannot be read statically, fails with
  `PromptPackInvalid`. One environment is built per pack at load and reused by every render.
- **A carriage return is refused** in any pack file. `content_sha` read text through universal
  newlines, so a CRLF checkout hashed like an LF one; raw bytes would not, and the identity of one
  commit would differ by platform. Refusing is the only rule that keeps both names stable.
  Repositories keep packs LF with `* text=auto eol=lf` in `.gitattributes`.
- **The manifest is resolved by the library** (`prompts.resolve`): `output_schema` imported only
  from module roots the caller names, graders looked up in the caller's registry, `golden` kept
  inside the pack directory. Any miss is `PromptPackInvalid` naming the field. `prompts` sits
  below `call` in the layer contract, so the fallback to the built-in call graders is
  `evals.golden.resolve_pack`, one layer up.
- **The vectors are frozen the day `identity` ships**: three `byte_sha` part lists (including the
  re-split pair that collides under `content_sha`) and the identity of each fixture pack.
- `call.graders.GRADERS` becomes a read-only mapping, so one consumer's grader cannot leak into
  another's table; `{**GRADERS, ...}` still copies, and every reader takes a table argument.

### Consequences

- Good: a grade, a mirror and a stamp can name everything that could change the pack's behaviour;
  `unevaluated` works on identities as well as shas.
- Good: no file is rendered that the name does not cover.
- Good: `content_sha` and every value derived from it are untouched.
- Bad: a pack with an unlisted include, a dynamic include, or a CRLF file now fails at load. Each
  is fixed in the pack (list the include, make it static, convert to LF), and each failure names
  the file.
- Bad: two names per pack. `content_sha` answers "did the text change", `identity` answers "can
  this pack behave differently"; a run is comparable to another only when `identity` matches.
- Neutral: a model setting does not belong in a pack. Model, temperature and retries are the
  role's and its tier's; one prompt carrying them would be two experiments.

### Option 1, rejected

Every stored stamp and every mirrored prompt would silently stop matching. ADR-0008's frozen
algorithm is the reason the name is worth storing.

### Option 2, rejected

`model_dump()` changes when the library adds a field with a default (this change adds two), so the
name would move on a library upgrade with no edit to the pack.
