#!/usr/bin/env bash
# Fan the n=4096 evaluation out as one SLURM array per arm.
#
#   export TAILRL_SLURM_PARTITION=gpu TAILRL_SLURM_GRES=gpu:1
#   bash scripts/slurm/submit_eval.sh --dry-run
#   bash scripts/slurm/submit_eval.sh
#
# 176 shards x 4 arms = 704 single-GPU tasks, roughly 20-80 minutes each depending on
# how long that arm's completions are (a model that writes long programs spends much
# longer in generation than one that has collapsed to copying its input). Budget a
# few hundred GPU-hours for the whole thing.
#
# Idempotent at the shard level, not just the job level: each task claims its shard
# and skips one that is already finished, so re-running this after a partial failure
# only redoes what is missing. `%N` caps how many array tasks run at once.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/reproduce/_common.sh" "$@"

SHARDS="${SHARDS:-176}"
CONCURRENCY="${CONCURRENCY:-32}"

[ -f "${CODEOPT_EVAL_DIR}/dump_prompts.json" ] || \
  run python3 "${EXP_ROOT}/scripts/make_dump_prompts.py"

submit_arm() {   # submit_arm <arm-name> <checkpoint>
  local arm="$1" ckpt="$2"
  local remaining=0
  for sh in $(seq 0 $((SHARDS - 1))); do
    [ -f "${CODEOPT_EVAL_DIR}/${arm}/shard_${sh}of${SHARDS}.done.json" ] || remaining=$((remaining + 1))
  done
  if [ "${remaining}" -eq 0 ]; then
    echo "skip ${arm}: all ${SHARDS} shards done"
    return
  fi
  echo "submit ${arm}: ${remaining}/${SHARDS} shards remaining"
  run sbatch \
    --job-name="codeopt_eval_${arm}" \
    --array="0-$((SHARDS - 1))%${CONCURRENCY}" \
    --partition="${TAILRL_SLURM_PARTITION}" \
    ${TAILRL_SLURM_ACCOUNT:+--account="${TAILRL_SLURM_ACCOUNT}"} \
    ${TAILRL_SLURM_QOS:+--qos="${TAILRL_SLURM_QOS}"} \
    --gres="${TAILRL_SLURM_EVAL_GRES:-gpu:1}" \
    --cpus-per-task="${TAILRL_SLURM_CPUS}" \
    --mem="${TAILRL_SLURM_MEM}" \
    --time="${TAILRL_SLURM_EVAL_TIME:-4:00:00}" \
    --output="${CODEOPT_LOG_DIR}/eval_${arm}-%A_%a.out" \
    --error="${CODEOPT_LOG_DIR}/eval_${arm}-%A_%a.err" \
    --wrap "bash '${EXP_ROOT}/scripts/eval_shard.sh' --ckpt '${ckpt}' --arm '${arm}' --shard \$SLURM_ARRAY_TASK_ID --shards ${SHARDS}"
}

for m in ${METHODS}; do
  step="$(eval_step_for "${m}")"
  submit_arm "${m}_step${step}" \
    "${CODEOPT_RUN_DIR}/$(run_tag "${m}" 0)/actor_ckpts/step_${step}/huggingface"
done
[ "${CODEOPT_EVAL_BASE:-1}" = "1" ] && submit_arm "base_step0" "${BASE_MODEL}"

echo
echo "When every array has drained, summarise with:"
echo "    bash scripts/reproduce/best_at_k_eval.sh --aggregate"
