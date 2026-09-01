#!/usr/bin/env bash
# The headline evaluation: unbiased pass@k and best-of-k speedup on the 878 held-out
# programs, at n = 4096 rollouts per program, for every arm plus the untrained model.
#
#   bash scripts/reproduce/best_at_k_eval.sh              # walk every shard here
#   bash scripts/reproduce/best_at_k_eval.sh --dry-run
#   bash scripts/reproduce/best_at_k_eval.sh --aggregate  # skip to the summary
#
# n = 4096 is not gratuitous. The unbiased best@k estimator needs n >> k to have low
# variance, and the headline number is k = 1024. That is 878 x 4096 = 3.6M rollouts
# per arm, each of which is compiled, run against every usable test case for
# correctness, and then simulated under gem5 -- so this is a cluster job, not a
# laptop job. Use scripts/slurm/submit_eval.sh to fan the shards out; this script
# walks them sequentially and is mostly useful for a small --shards value or to
# finish a tail.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh" "$@"

SHARDS="${SHARDS:-176}"
AGGREGATE_ONLY=0
for arg in "$@"; do [ "${arg}" = "--aggregate" ] && AGGREGATE_ONLY=1; done

# Resolve each arm to a checkpoint directory. The base model is evaluated as step 0.
declare -a ARMS=() CKPTS=()
for m in ${METHODS}; do
  step="$(eval_step_for "${m}")"
  ckpt="${CODEOPT_RUN_DIR}/$(run_tag "${m}" 0)/actor_ckpts/step_${step}/huggingface"
  ARMS+=("${m}_step${step}"); CKPTS+=("${ckpt}")
done
if [ "${CODEOPT_EVAL_BASE:-1}" = "1" ]; then
  ARMS+=("base_step0"); CKPTS+=("${BASE_MODEL}")
fi

if [ "${AGGREGATE_ONLY}" -eq 0 ]; then
  # The dumped-prompt set must exist before the first shard, and must be identical
  # across arms, so make it once up front.
  [ -f "${CODEOPT_EVAL_DIR}/dump_prompts.json" ] || \
    run python3 "${EXP_ROOT}/scripts/make_dump_prompts.py"

  for i in "${!ARMS[@]}"; do
    arm="${ARMS[$i]}"; ckpt="${CKPTS[$i]}"
    # A bare Hub id (no slash-prefixed path) is always fine; a local path must exist.
    if [[ "${ckpt}" == /* ]] && [ ! -f "${ckpt}/config.json" ]; then
      echo "SKIP ${arm}: no checkpoint at ${ckpt}" >&2
      continue
    fi
    echo "=== ${arm} (${SHARDS} shards) ==="
    for sh in $(seq 0 $((SHARDS - 1))); do
      run bash "${EXP_ROOT}/scripts/eval_shard.sh" \
        --ckpt "${ckpt}" --arm "${arm}" --shard "${sh}" --shards "${SHARDS}"
    done
  done
fi

echo "=== aggregating ==="
run python3 -m code_opt.eval.besteval_aggregate \
  --eval_root "${CODEOPT_EVAL_DIR}" \
  --runs "${ARMS[@]}" \
  --k_values 1 4 16 64 256 1024 \
  --expect_prompts 878 \
  --out "${CODEOPT_EVAL_DIR}/metrics"

run python3 -m code_opt.eval.besteval_gens_merge \
  --eval_root "${CODEOPT_EVAL_DIR}" \
  --runs "${ARMS[@]}" \
  --out "${CODEOPT_EVAL_DIR}/generations"

echo
echo "metrics      ${CODEOPT_EVAL_DIR}/metrics/summary.json"
echo "curves       ${CODEOPT_EVAL_DIR}/metrics/besteval_test.png"
echo "generations  ${CODEOPT_EVAL_DIR}/generations/"
echo "reference    paper_results/best_at_k_n4096.json"
