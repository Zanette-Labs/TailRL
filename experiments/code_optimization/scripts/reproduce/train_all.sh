#!/usr/bin/env bash
# Train all three arms. This is the headline comparison.
#
#   bash scripts/reproduce/train_all.sh              # sequentially, on this node
#   bash scripts/reproduce/train_all.sh --dry-run    # print the commands only
#   METHODS=tailrl bash scripts/reproduce/train_all.sh
#
# Every arm is identical except `algorithm.adv_estimator`. Each is ~2570 steps at
# roughly 400-440 s/step on one 4-GPU node, so about 8-13 days per arm run
# sequentially. Run them concurrently on three nodes instead:
#
#   bash scripts/slurm/submit_all.sh
#
# Interrupted runs resume: re-run this and each arm continues from its last
# checkpoint. Finished arms exit immediately.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh" "$@"

for seed in ${SEEDS}; do
  for m in ${METHODS}; do
    echo "=== ${m} seed ${seed} ==="
    run bash "${EXP_ROOT}/scripts/train.sh" --method "${m}" --seed "${seed}"
  done
done
