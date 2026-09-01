#!/usr/bin/env bash
# Submit the three training arms, one job each, so they run concurrently.
#
#   export TAILRL_SLURM_PARTITION=gpu TAILRL_SLURM_GRES=gpu:4 TAILRL_SLURM_TIME=24:00:00
#   bash scripts/slurm/submit_all.sh --dry-run     # inspect first
#   bash scripts/slurm/submit_all.sh
#
# Idempotent: an arm that already has a job in the queue is skipped, and an arm whose
# run directory has reached the step cap is skipped too. Re-run it on a timer and the
# campaign converges without any bookkeeping on your part.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/reproduce/_common.sh" "$@"

STEPS="${PIE_TOTAL_STEPS:-2570}"
queued() { squeue -u "${USER}" -h -o '%j' 2>/dev/null | grep -Fxq "$1"; }

for seed in ${SEEDS}; do
  for m in ${METHODS}; do
    name="codeopt_${m}_s${seed}"
    tag="$(run_tag "${m}" "${seed}")"
    done_steps="$(cat "${CODEOPT_RUN_DIR}/${tag}/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)"

    if [ "${done_steps}" -ge "${STEPS}" ]; then
      echo "skip ${name}: finished (${done_steps}/${STEPS})"
      continue
    fi
    if queued "${name}"; then
      echo "skip ${name}: already in the queue"
      continue
    fi

    run sbatch \
      --job-name="${name}" \
      --partition="${TAILRL_SLURM_PARTITION}" \
      ${TAILRL_SLURM_ACCOUNT:+--account="${TAILRL_SLURM_ACCOUNT}"} \
      ${TAILRL_SLURM_QOS:+--qos="${TAILRL_SLURM_QOS}"} \
      --gres="${TAILRL_SLURM_GRES}" \
      --cpus-per-task="${TAILRL_SLURM_CPUS}" \
      --mem="${TAILRL_SLURM_MEM}" \
      --time="${TAILRL_SLURM_TIME}" \
      --signal=B:USR1@600 \
      --output="${CODEOPT_LOG_DIR}/${name}-%j.out" \
      --error="${CODEOPT_LOG_DIR}/${name}-%j.err" \
      "${EXP_ROOT}/scripts/slurm/sbatch_train.sh" --method "${m}" --seed "${seed}"
  done
done
