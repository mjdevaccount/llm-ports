# Prompt packs

A pack is a directory under a root: `pack.yaml` plus jinja templates. Loading validates the manifest and names the text by its sha, so the stamp beside an output row says exactly which prompt produced it. The packs this example loads live in `examples/prompt_pack/packs/`.

```{.python title="examples/prompt_pack/main.py"}
--8<-- "examples/prompt_pack/main.py"
```

A pack has two names. `content_sha` covers the template text and never changes algorithm; `identity` covers the manifest bytes and every file the templates can render (`templates` plus `includes`), so a version bump, a re-pointed grader or an edited partial moves it ([ADR-0026](../adr/0026-a-pack-has-an-identity-beside-its-content-sha.md)). A template that includes a file the manifest does not list fails at load (list partials in `includes` as relative paths, e.g. `partials/h.j2`), and so does a pack file with a carriage return: keep packs LF with `* text=auto eol=lf` in `.gitattributes`.

Source: [`examples/prompt_pack/main.py`](https://github.com/mjdevaccount/llm-ports/blob/main/examples/prompt_pack/main.py)
