"""Grounded prose: the number law (`numbers`), the slot grammar (`slots`), and the belt.

The package exports the shapes every grounded generator shares (`Problem`, the retry text, the
belt) and `NumberLawGrounder`. `numbers` and `slots` are read module-qualified: each has its own
`loose` and `belt`.
"""

from llm_ports.grounding import numbers, slots
from llm_ports.grounding._belt import RETRY_PREFIX, Problem, allowed_numbers, belt, describe, loose, retry_message
from llm_ports.grounding._grounder import NumberLawGrounder

__all__ = [
    "RETRY_PREFIX",
    "NumberLawGrounder",
    "Problem",
    "allowed_numbers",
    "belt",
    "describe",
    "loose",
    "numbers",
    "retry_message",
    "slots",
]
