"""Shared test setup for the GUI-grounding (screenspot) suite.

Puts the repo's `scripts/` and `examples/reward_function/` dirs on sys.path so tests can
`import gui_grounding`, `import gui_sparse`, `import eval_screenspot_pro`,
`import convert_click100k_to_easyr1`, `import convert_screenspot_to_easyr1` directly.
(`verl` is pip-installed in the overlay, so `from verl.trainer.core_algos import ...` works
without help.) Also registers the `gpu` marker used by the integration smoke test.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _sub in ("scripts", "examples/reward_function"):
    _p = os.path.join(REPO, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: test requires a GPU (skipped in CPU-only runs)")
