# Prompt packs

A pack is a directory under a root: `pack.yaml` plus jinja templates. Loading validates the manifest and names the text by its sha, so the stamp beside an output row says exactly which prompt produced it. The packs this example loads live in `examples/prompt_pack/packs/`.

```{.python title="examples/prompt_pack/main.py"}
--8<-- "examples/prompt_pack/main.py"
```

Source: [`examples/prompt_pack/main.py`](https://github.com/mjdevaccount/llm-ports/blob/main/examples/prompt_pack/main.py)
