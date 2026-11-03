"""Grounded prose by construction: the model names a value, and code prints it.

Code renders each figure once under a digit-free id. The output grammar (`slot_pattern`) has no
digit token, so the model can write `{total}` but not "1,284.50", "1,300" or "about twelve
hundred". `problems` words what is wrong before the fill, as a retry message; `fill` substitutes
the values; `belt` counts the figures in the filled text that no rendered value shows.
"""

import re

from llm_ports.grounding import slots

rendered = {"count": "3", "total": "$1,284.50", "eta": "Friday"}
pattern = slots.slot_pattern(rendered, 240)

# What a model can write under the grammar, and what it cannot.
good = "Your {count} items, {total} in all, arrive {eta}."
assert re.fullmatch(pattern, good)
assert not re.fullmatch(pattern, "Your 3 items came to $1,284.50.")  # digits are not in the grammar

# A run the grammar admits can still cite a slot that does not exist: a retry fixes that.
assert slots.problems("Your {count} items ship {when}.", rendered) == ["unknown slot {when}"]

prose = slots.fill(good, rendered)
print(prose)
assert slots.belt(prose, rendered) == 0  # every figure in the text is a rendered value

# The same belt over text a model wrote freely names the invented figure.
print(slots.loose("Your 3 items came to $1,300.", rendered))  # ['1300']
