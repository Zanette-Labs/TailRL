#!/bin/bash
# Reproduce: TailRL vs. the expected-reward RL baselines.
#
# Trains every RL method across the rollout budgets N in {16, 64, 256, 1024}, so
# each estimator is compared against the others at matched budgets. This is the
# comparison behind the RL curves and the rollout-budget scaling figure.
#
#   bash scripts/reproduce/rl_methods.sh
#
# 16 arms x 3 seeds = 48 runs, sequential, one GPU. Cost grows with N: a step
# scores N rollouts per image, so N=1024 is far slower than N=16. Use
# TRAIN_SUBSAMPLE / EPOCHS for a fast pass first, and see _common.sh for the
# knobs:
#
#   TRAIN_SUBSAMPLE=20000 EPOCHS=2 SEEDS=42 bash scripts/reproduce/rl_methods.sh
#
# Already-finished arms are skipped, so re-running resumes.

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Every method at every budget, so the comparison is matched throughout.
METHODS=(tailrl grpo rloo reinforce)
N_VALUES=(16 64 256 1024)

echo "=============================================================="
echo "  TailRL vs RL baselines"
echo "  methods : ${METHODS[*]}"
echo "  N       : ${N_VALUES[*]}"
echo "  seeds   : ${SEEDS}"
echo "  epochs  : ${EPOCHS}   K=${K}   subsample=${TRAIN_SUBSAMPLE:-full}"
echo "=============================================================="

for method in "${METHODS[@]}"; do
    echo
    echo "-- ${method} across rollout budgets --"
    for n in "${N_VALUES[@]}"; do
        for seed in ${SEEDS}; do
            run_one "${method}" "${n}" "${seed}"
        done
    done
done

finish
