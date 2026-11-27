# Grounded prose with slots

The model cites a value by its id and code prints the value. The grammar has no digit token, so a figure the inputs never rendered cannot be written; `problems` words a bad citation as a retry message, and `belt` over the filled text is zero by construction.

```{.python title="examples/slots_grounded_prose.py"}
--8<-- "examples/slots_grounded_prose.py"
```

Source: [`examples/slots_grounded_prose.py`](https://github.com/mjdevaccount/llm-ports/blob/main/examples/slots_grounded_prose.py)

## When the input is an image

The rule holds when the model reads an image too: code prints what it already knows, and the model
is asked only what the image alone shows. Measured on one vision model reading values off rendered
plots:

- **Printed text is read exactly; a position is not.** The model transcribed a printed label to the
  last digit, but a value estimated between two gridlines was off by 8-9% of the axis range.
  Render every value you hold as text (a title, a legend outside the plotted area) instead of
  asking for it to be read off an axis.
- **What is drawn over the data changes the reading, confidently.** A full-width reference line
  and a label badge drawn over the last data point each produced a wrong answer at high stated
  confidence. Keep annotations outside the plotted area.
- **Stated confidence does not separate wrong reads from right ones** (0.86 against 0.94 mean).
  Grade against ground truth, never against the model's own confidence.
- **Reasoning tokens do not repair perception.** Where the scratchpad sits made no difference
  ([ADR-0032](../adr/0032-the-research-scratchpad-stays-first.md)).

### Check that the image arrived

A server that drops an image, or a tier that ignores it, still answers: from the text alone, and
in the same shape. Print a code into the image you generate, ask for it back in a field, and make
it a validator. A reply that does not carry it is re-asked, then returned `invalid` with no value,
never an answer read from the prompt alone:

```{.python test="skip"}
from pydantic import BaseModel


class Read(BaseModel):
    image_code: str  # "the short code printed in the image's top-left corner"
    total: float


def saw_the_image(value: Read) -> list[str]:
    return [] if value.image_code.strip() == CODE else ["image_code is not the code printed in the image"]


res = await generate(
    role,
    prompt="Read the total. Also copy the code printed in the top-left corner.",
    output=Read,
    generator=gen,
    images=[(png_bytes, "image/png")],
    validate=saw_the_image,
)
```

`evals.golden` fires an item's `images` the same way, so a golden set for a vision role is graded
on what the live call sees.
