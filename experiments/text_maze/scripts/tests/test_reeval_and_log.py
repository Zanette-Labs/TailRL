#!/usr/bin/env python3
"""Regression tests for reeval_and_log.py (no wandb / no network).

Guards the two bugs that silently corrupted ~30 runs of the v3 maze campaign:

  1. verl pretty-prints the val_metrics dict WRAPPED across lines, with a Ray
     `(TaskRunner pid=N)` prefix injected on each line, so a key and its value land
     on different lines. A line-anchored regex misses those keys -- including
     is_shortest, the headline metric.
  2. The parser then reported "OK logged 153 metrics (is_shortest=None)" and exited
     0, so every caller believed it had succeeded. A partial parse must fail loudly.

Run:  python scripts/tests/test_reeval_and_log.py
"""
import importlib.util
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.normpath(os.path.join(HERE, "..", "reeval_and_log.py"))

spec = importlib.util.spec_from_file_location("reeval_and_log", TARGET)
ral = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ral)

# verl's real output shape: ANSI codes, a Ray pid prefix, and a dict wrapped so that
# is_shortest's value sits on the NEXT line from its key.
WRAPPED = (
    '\x1b[36m(TaskRunner pid=98177)\x1b[0m  "\'val-aux/maze_17_continuous/goal_reached/mean@64\': 0.96790625, "\n'
    '\x1b[36m(TaskRunner pid=98177)\x1b[0m  "\'val-aux/maze_17_continuous/is_shortest/mean@64\': "\n'
    '\x1b[36m(TaskRunner pid=98177)\x1b[0m  "0.853109375, "\n'
    '\x1b[36m(TaskRunner pid=98177)\x1b[0m  "\'val-aux/maze_17_continuous/is_shortest/best@4/mean\': 0.8862, "\n'
    '\x1b[36m(TaskRunner pid=98177)\x1b[0m  "\'mean_accuracies/maze_17_continuous/reward/mean@64\': 0.9, "\n'
)
OLD_LINE_ANCHORED = re.compile(r"'(val-(?:aux|core)/[^']+)':\s*([0-9.eE+-]+)")


def main():
    passed = failed = 0

    def check(name, cond):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  PASS  {name}")
        else:
            failed += 1
            print(f"  FAIL  {name}")

    with tempfile.TemporaryDirectory() as tmp:
        wrapped = os.path.join(tmp, "wrapped.log")
        open(wrapped, "w").write(WRAPPED)

        m = ral.parse_metrics(wrapped)
        check("wrapped key recovered (value on next line)",
              m.get(ral.REQUIRED) == 0.853109375)
        check("all 4 metrics parsed despite wrapping", len(m) == 4)
        check("non-val-aux prefixes captured too (mean_accuracies/...)",
              m.get("mean_accuracies/maze_17_continuous/reward/mean@64") == 0.9)

        # the fixture must actually reproduce the original bug, else it guards nothing
        old = dict(OLD_LINE_ANCHORED.findall(WRAPPED))
        check("fixture reproduces the bug (old regex misses is_shortest)",
              ral.REQUIRED not in old)

        # a partial parse must be a hard failure, never a silent success
        partial = os.path.join(tmp, "partial.log")
        open(partial, "w").write("'val-aux/maze_17_continuous/goal_reached/mean@64': 0.5,\n")
        check("partial parse exits 2 (no silent success)",
              ral.main(partial, "dummy", 5000) == 2)

        empty = os.path.join(tmp, "empty.log")
        open(empty, "w").write("nothing to see here\n")
        check("empty parse exits 2", ral.main(empty, "dummy", 5000) == 2)

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
