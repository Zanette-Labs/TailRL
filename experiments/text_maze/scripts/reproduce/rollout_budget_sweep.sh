#!/bin/bash
# TRAINING-ROLLOUT-BUDGET SWEEP — how each estimator converts more rollouts per
# prompt into learning, from a fixed weak initialization (ckpt-3000, 0.024%
# shortest-path success before RL).
#
#   4 estimators x N in {4, 16, 64, 256} x 3 seeds = 48 runs
#
# N=16 is the same cell as the checkpoint sweep's ckpt-3000 column, so if you
# already ran checkpoint_sweep.sh those 12 arms are skipped automatically.
#
# PKPO note: k_opt is held at 8 except at N=4, where k_opt=8 is impossible
# (max@k needs k <= N) and the launcher's guard would refuse the run — so N=4
# uses k_opt=4. The guard is deliberate: at k=n the estimator is degenerate
# (only the argmax rollout gets a nonzero advantage), so PKPO_ALLOW_DEGENERATE=1
# is required to opt in, and that is what this script does for the N=4 cell.
#
# Usage:
#   bash scripts/reproduce/rollout_budget_sweep.sh [--dry-run]
set -uo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/_common.sh"

BUDGET_CKPT="${BUDGET_CKPT:-3000}"
BUDGETS="${BUDGETS:-4 16 64 256}"

n=0
for seed in $SEEDS; do
  for N in $BUDGETS; do
    for method in $METHODS; do
      k="$PASS_K"
      if [ "$method" = "pkpo" ] && [ "$k" -ge "$N" ]; then
        k=$(( N / 2 )); [ "$k" -lt 1 ] && k=1
        # at N=4 the paper uses k_opt=4, i.e. the degenerate k=n cell, on purpose
        [ "$N" = "4" ] && { k=4; export PKPO_ALLOW_DEGENERATE=1; }
      fi
      n=$((n + 1))
      run_arm "$method" "$BUDGET_CKPT" "$seed" "$N" "$k"
      unset PKPO_ALLOW_DEGENERATE
    done
  done
done
echo "[done] rollout-budget sweep: ${n} arms at ckpt${BUDGET_CKPT} (N='${BUDGETS}' methods='${METHODS}' seeds='${SEEDS}')"
