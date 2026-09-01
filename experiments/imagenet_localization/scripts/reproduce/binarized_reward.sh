#!/bin/bash
# Reproduce: the cost of binarizing a continuous reward.
#
# Trains MaxRL on IoU binarized at 0.5 and at 0.75, against TailRL on the raw
# continuous IoU, across the rollout budgets. This is the comparison behind the
# binarization figure: the binarized arms specialize to their own threshold and
# fall off at stricter CorLoc levels, while TailRL improves across all of them.
#
#   bash scripts/reproduce/binarized_reward.sh
#
# 12 arms x 3 seeds = 36 runs, sequential, one GPU. The TailRL arms here are the
# same runs rl_methods.sh produces, so if you ran that first they are skipped.
#
#   TRAIN_SUBSAMPLE=20000 EPOCHS=2 SEEDS=42 bash scripts/reproduce/binarized_reward.sh
#
# NOTE: binary_maxrl binarizes at exact equality r == 1.0, which continuous IoU
# never reaches — so it MUST be paired with a binarizing reward transform, which
# is what this script does. Running it with the default transform trains on
# identically zero advantages.

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

N_VALUES=(16 64 256 1024)
THRESHOLDS=(binary_0.5 binary_0.75)

echo "=============================================================="
echo "  Binarized reward: MaxRL@{0.5,0.75} vs TailRL on continuous IoU"
echo "  N       : ${N_VALUES[*]}"
echo "  seeds   : ${SEEDS}"
echo "  epochs  : ${EPOCHS}   K=${K}   subsample=${TRAIN_SUBSAMPLE:-full}"
echo "=============================================================="

echo
echo "-- MaxRL on binarized IoU --"
for xform in "${THRESHOLDS[@]}"; do
    for n in "${N_VALUES[@]}"; do
        for seed in ${SEEDS}; do
            run_one binary_maxrl "${n}" "${seed}" "${xform}"
        done
    done
done

echo
echo "-- TailRL on the raw continuous reward (reference) --"
for n in "${N_VALUES[@]}"; do
    for seed in ${SEEDS}; do
        run_one tailrl "${n}" "${seed}"
    done
done

finish
