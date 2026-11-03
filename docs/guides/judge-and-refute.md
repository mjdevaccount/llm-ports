# A judge and an adversary

The judge's pure half picks which replies to grade (a deterministic draw) and writes the instruction from what the judged call saw. The adversarial pass asks a model to refute each claim against the evidence, through the consumer's own `generate`, and attributes each verdict by index.

```{.python title="examples/judge_and_refute.py"}
--8<-- "examples/judge_and_refute.py"
```

Source: [`examples/judge_and_refute.py`](https://github.com/mjdevaccount/llm-ports/blob/main/examples/judge_and_refute.py)
