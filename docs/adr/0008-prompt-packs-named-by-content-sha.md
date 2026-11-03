---
status: accepted
date: 2026-11-03
---

# Prompt packs live in the repo and are named by a content sha

Decided in 0.1.0 (`packs`, `manifest`); the managed-store mirror in 0.11.0
(`langfuse_prompt_source`). Recorded in 0.12.0.

## Context and problem statement

A prompt changes the behaviour of every call that uses it, so a recorded result is only
comparable to another made under the same prompt text. Prompts edited in a managed store drift from
the code that renders them; prompts inlined in code have no name a result can be stamped with; and
a version number assigned by hand changes when nothing did, or fails to change when something did.

## Decision drivers

- A result row can name the exact prompt text it was produced under.
- Moving a prompt into a pack must not change its name.
- A broken pack fails loudly, never by falling back to some other text.

## Considered options

1. Prompts in a managed store, referenced by label.
2. Prompts in code, versioned by hand.
3. Prompts in the repo as packs, named by a hash of their source; a managed store mirrors them.

## Decision outcome

Chosen option 3, in `packs` and `manifest`:

- `Pack.content_sha` is sha256, cut to 16 hex characters, over the template source, never a
  render (a render depends on its context). A single-template pack hashes exactly that one source,
  so a prompt that moves into a pack keeps the sha its recordings were graded under.
- The sha algorithm never changes; tests pin it on fixed inputs. A changed stamp for unchanged
  content would read as a configuration change that never happened.
- The manifest is strict (`extra="forbid"`): a field invented in YAML fails in a unit test, not as
  a silent extra the loader never reads.
- A pack fails closed: a malformed manifest, an unparseable template or a missing file raises
  `PromptPackInvalid`. `scan` reports each invalid pack by id instead of skipping it.
- The renderer runs the source template itself, so a managed store's template language never
  limits it (a mustache compiler cannot run Jinja loops). A managed store is a `PromptSource`
  adapter's mirror: the repo leads and the `production` label follows.

### Consequences

- Good: every recorded result is attributable to exact prompt text; a moved prompt keeps its
  history.
- Good: a prompt change is a code review.
- Bad: editing a prompt in the managed store's UI does not change what runs; the mirror is
  read-only by design.
