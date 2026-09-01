#!/bin/bash
# Reproduce: TailRL vs. direct supervision.
#
# Trains the population-level TailRL objective against supervised anchors that
# see the ground-truth box directly. This is the comparison behind the
# "population-level TailRL versus direct supervision" figure.
#
#   bash scripts/reproduce/supervised_baselines.sh
#
# 7 arms x 3 seeds = 21 runs, sequential, one GPU. At the paper's 30 epochs over
# the full train split that is days of GPU time — set TRAIN_SUBSAMPLE for a fast
# pass first, and see _common.sh for every knob:
#
#   TRAIN_SUBSAMPLE=20000 EPOCHS=2 SEEDS=42 bash scripts/reproduce/supervised_baselines.sh
#
# Already-finished arms are skipped, so re-running resumes.

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# N is irrelevant for these arms — none of them sample rollouts — but it still
# lands in the directory name, so keep it fixed at the paper's primary budget.
N_FIXED=64

# The four supervised anchors named in the paper, plus the two categorical
# baselines that share the policy architecture, plus the population objective.
METHODS=(
    tailrl_population   # the exact N -> infinity TailRL objective
    mse                 # MSE on the ground-truth coordinates
    l1_iou_match        # L1
    giou                # GIoU
    l1_giou             # L1 + GIoU (DETR weights: l1_weight=5, giou_weight=2)
    ordinal_ce          # ordinal cross-entropy over the K bins
    cross_entropy       # plain cross-entropy over the K bins
)

echo "=============================================================="
echo "  TailRL vs supervised baselines"
echo "  methods : ${METHODS[*]}"
echo "  seeds   : ${SEEDS}"
echo "  epochs  : ${EPOCHS}   K=${K}   subsample=${TRAIN_SUBSAMPLE:-full}"
echo "=============================================================="

for method in "${METHODS[@]}"; do
    for seed in ${SEEDS}; do
        run_one "${method}" "${N_FIXED}" "${seed}"
    done
done

finish
