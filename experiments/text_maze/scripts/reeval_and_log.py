#!/usr/bin/env python3
"""Parse verl val_only console metrics and log them into an EXISTING wandb run.

Used by train.sh --val-only to push a clean end-of-training evaluation
into a run whose own step-5k validation never made it to wandb.

TWO BUGS THIS FILE EXISTS TO NOT REPEAT
---------------------------------------
1. LINE WRAPPING. verl pretty-prints the val_metrics dict wrapped across lines, and
   Ray injects a `(TaskRunner pid=N)` prefix on every line, so a key and its value
   routinely land on different lines:

       (TaskRunner pid=98177)  "'val-aux/.../is_shortest/mean@64': "
       (TaskRunner pid=98177)  "0.853109375, "

   A line-anchored regex matches only ~153 of ~182 metrics and silently drops
   is_shortest -- the one metric that matters. The text is de-wrapped (ANSI + pid
   prefixes stripped, newlines collapsed) BEFORE matching.

2. SILENT PARTIAL SUCCESS. The old version printed "OK logged 153 metrics
   (is_shortest=None)" and exited 0, so every caller believed it had worked. Now a
   missing required metric is a hard failure (exit 2) -- a partial parse must never
   look like success.

Usage:  python reeval_and_log.py <metrics_stdout_file> <wandb_run_id> <step>
Env:    WANDB_PROJECT, WANDB_ENTITY. Credentials come from ~/.netrc via
        `wandb login` -- never from an API key in the repo or the environment.
"""
import os
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PID = re.compile(r"\((?:TaskRunner|WorkerDict) pid=\d+\)")
# keys may be val-aux/..., val-core/..., mean_accuracies/..., pass@N_accuracies/...
KV = re.compile(r"'([A-Za-z@0-9_\-][^']*maze_17_continuous/[^']+)':\s*([0-9.eE+-]+)")
REQUIRED = "val-aux/maze_17_continuous/is_shortest/mean@64"


def parse_metrics(path):
    raw = open(path, errors="ignore").read()
    text = ANSI.sub("", raw)
    text = PID.sub("", text)
    text = text.replace('"', "")
    text = re.sub(r"\s*\n\s*", " ", text)          # de-wrap: key and value rejoin
    out = {}
    for k, v in KV.findall(text):
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def main(metrics_file, run_id, step):
    metrics = parse_metrics(metrics_file)
    if not metrics:
        print(f"REEVAL_LOG: FAIL no val metrics parsed from {metrics_file}", file=sys.stderr)
        return 2
    if metrics.get(REQUIRED) is None:
        # a partial parse must not look like success -- this is exactly how the first
        # attempt silently corrupted ~30 runs.
        print(f"REEVAL_LOG: FAIL parsed {len(metrics)} metrics but {REQUIRED} is missing "
              f"-- refusing to log a partial result", file=sys.stderr)
        return 2

    import wandb

    run = wandb.init(
        project=os.environ["WANDB_PROJECT"],
        entity=os.environ.get("WANDB_ENTITY") or None,   # None -> your default entity
        id=run_id,
        resume="must",
    )
    wandb.log(metrics, step=step)
    for k, v in metrics.items():
        run.summary[k] = v
    run.summary["reeval_step5k_done"] = 1
    run.summary["reeval_step5k_is_shortest_mean64"] = metrics[REQUIRED]
    wandb.finish()
    print(f"REEVAL_LOG: OK logged {len(metrics)} metrics to run {run_id} at step {step} "
          f"(is_shortest/mean@64={metrics[REQUIRED]})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], int(sys.argv[3])))
