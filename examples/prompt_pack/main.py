"""Prompt packs: prompts as versioned files, named by the sha of their text.

A pack is a directory, `<root>/<feature.action>/`, holding `pack.yaml` and its jinja templates.
`prompts.load` validates the manifest and fingerprints the sources; the stamp stored beside each
output row (`{pack, version, sha}`) says which text produced it, so an edited prompt is never
mistaken for the one that was evaluated.
"""

from pathlib import Path

from llm_ports import prompts

ROOT = Path(__file__).with_name("packs")

pack = prompts.load("orders.summarize", ROOT)
print(pack.stamp_str())  # orders.summarize@1.0.0+<sha of the template>
print(pack.render(slots={"count": "how many items", "total": "the order total"}))

# The registry loads every pack strictly; `unevaluated` names the packs whose served text has no
# grade yet. Here nothing was graded, so every pack is due.
served = prompts.served_shas(ROOT)
print(sorted(served))
print(prompts.unevaluated(served, evaluated={}))

# The stamp follows the text, not the version field: an edited template is a new sha.
candidate = pack.source.replace("one sentence", "two sentences")
assert prompts.content_sha([candidate]) != pack.content_sha
