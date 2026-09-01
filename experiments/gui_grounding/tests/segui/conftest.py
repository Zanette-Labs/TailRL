"""Shared test setup for the SE-GUI reward suite.

Mirrors tests/screenspot/conftest.py: puts the repo's `examples/reward_function/` (and `scripts/`)
dirs on sys.path so tests can `import segui_point` / `import gui_grounding` directly, the same way
verl loads a reward file at train time.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _sub in ("scripts", "examples/reward_function"):
    _p = os.path.join(REPO, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
