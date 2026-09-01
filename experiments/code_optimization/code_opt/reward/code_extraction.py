"""Pull the candidate C++ program out of a model completion.

This is the first thing the reward does to every rollout, so it decides what gets
compiled and graded. It is deliberately forgiving about surrounding prose and
deliberately strict about *which* code block wins: the last complete fenced block
after the prompt's anchor, matching the LiveCodeBench convention.
"""

from __future__ import annotations

import re

# Present when the sampler decodes with skip_special_tokens=False.
_SPECIAL_TOK_RE = re.compile(r"<\|[^>]+\|>")

#: The prompt ends with this line, so everything the model writes follows it. With
#: the reasoning-style prompt the model is asked to emit it rather than having it
#: prefilled; either way it marks the start of the answer.
ANSWER_ANCHOR = "### Optimized Version:"


def extract_code(solution_str: str) -> str:
    """Return the C++ source in ``solution_str``, or ``""`` if there is none.

    1. Strip ``<|...|>`` special tokens.
    2. For reasoning models, drop everything up to and including ``</think>``.
    3. If the answer anchor is present, keep only what follows it -- otherwise a
       model that echoes the *slow* program in its preamble can have that echo
       graded instead of its answer.
    4. Take the last complete fenced block. Models routinely emit several fences
       (the original, an intermediate attempt, the final answer); the last complete
       one is the answer.
    5. With no fence at all, treat the whole post-anchor body as code -- the
       anchor-prefilled prompt makes bare code the expected shape, not an error.
    """
    s = solution_str.strip()
    s = _SPECIAL_TOK_RE.sub("", s).strip()
    if "</think>" in s:
        s = s.split("</think>", 1)[1].strip()
    if ANSWER_ANCHOR in s:
        s = s.split(ANSWER_ANCHOR, 1)[1].strip()

    lines = s.split("\n")
    fences = [i for i, line in enumerate(lines) if "```" in line]
    if len(fences) >= 2:
        return "\n".join(lines[fences[-2] + 1:fences[-1]]).strip()
    return s
