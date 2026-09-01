#!/bin/bash
# THE HEADLINE EXPERIMENT — "Text-Maze navigation from low-success initial policies".
#
#   4 estimators x 7 SFT initializations x 3 seeds = 84 RL runs
#
# Every arm is identical except the advantage estimator and the checkpoint it
# starts from, so the comparison isolates exactly one thing: how each estimator
# behaves as the initial policy's shortest-path success rate falls from 0.83%
# to 0.012%.
#
# Produces the data behind the paper's Pass@1 initialization-sweep figure, and
# (after scripts/reproduce/eval_ladder.sh) the Pass@k / Best-of-k ladders.
#
# Usage:
#   bash scripts/reproduce/checkpoint_sweep.sh [--dry-run]
#
# Each arm is one sequential single-GPU run of ~5000 steps. Re-running the
# script resumes: finished arms are skipped and interrupted ones continue from
# their last checkpoint. On one GPU the full matrix is weeks of wall-clock —
# see scripts/slurm/ to fan it out, or narrow it:
#
#   METHODS=tailrl CKPTS=3000 SEEDS=0 bash scripts/reproduce/checkpoint_sweep.sh
set -uo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/_common.sh"

n=0
for seed in $SEEDS; do          # seed-major: one full seed finishes before the next
  for ckpt in $CKPTS; do
    for method in $METHODS; do
      n=$((n + 1))
      run_arm "$method" "$ckpt" "$seed"
    done
  done
done
echo "[done] checkpoint sweep: ${n} arms (methods='${METHODS}' ckpts='${CKPTS}' seeds='${SEEDS}')"
