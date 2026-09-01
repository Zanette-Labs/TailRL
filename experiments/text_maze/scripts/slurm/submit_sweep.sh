#!/bin/bash
# Fan a reproduce matrix out over SLURM: one job per arm, submitted through
# sbatch_train.sh. Idempotent — an arm that has already finished, or that
# already has a live job, is not resubmitted, so you can re-run this on a timer
# until the campaign converges.
#
#   bash scripts/slurm/submit_sweep.sh checkpoint_sweep [--dry-run]   # 84 arms
#   bash scripts/slurm/submit_sweep.sh rollout_budget   [--dry-run]   # 48 arms
#
# Narrow it the same way as the reproduce scripts:
#   METHODS=tailrl SEEDS=0 bash scripts/slurm/submit_sweep.sh checkpoint_sweep
#
# Configure the queue through TAILRL_SLURM_* (scripts/env.sh):
#   TAILRL_SLURM_PARTITION, _QOS, _ACCOUNT, _GRES, _CPUS, _MEM, _TIME
#
# NOTE ON WALLTIME: the arms are long (5000 steps). sbatch_train.sh requeues
# itself on USR1 and verl resumes from the last checkpoint, so a short walltime
# costs restarts but never progress — as long as save_freq fits inside one
# window. At N=256 a step is slow enough that the default save_freq=250 may not
# checkpoint before the first requeue; lower it (SAVE_FREQ=25) for large N.
set -uo pipefail
EXP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SWEEP="${1:?usage: submit_sweep.sh <checkpoint_sweep|rollout_budget> [--dry-run]}"
shift || true
# shellcheck disable=SC1091
. "${EXP_ROOT}/scripts/reproduce/_common.sh"

SBATCH_FLAGS=(--partition="$TAILRL_SLURM_PARTITION" --gres="$TAILRL_SLURM_GRES"
              --cpus-per-task="$TAILRL_SLURM_CPUS" --mem="$TAILRL_SLURM_MEM"
              --time="$TAILRL_SLURM_TIME"
              --output="${TAILRL_LOG_DIR}/%x-%j.out" --error="${TAILRL_LOG_DIR}/%x-%j.err")
[ -n "$TAILRL_SLURM_QOS" ]     && SBATCH_FLAGS+=(--qos="$TAILRL_SLURM_QOS")
[ -n "$TAILRL_SLURM_ACCOUNT" ] && SBATCH_FLAGS+=(--account="$TAILRL_SLURM_ACCOUNT")

QUEUE="$(squeue -h -u "$USER" -o '%j' 2>/dev/null || true)"

submit_arm() {  # submit_arm <method> <ckpt> <seed> <n_rollouts> <pass_k>
  local method="$1" ckpt="$2" seed="$3" N="$4" k="$5"
  local name exp
  name="tm_${method}_ck${ckpt}_G${N}_s${seed}"
  exp="$(exp_name "$(method_tag "$method")" "$ckpt" "$N" "$seed")"
  if is_done "$exp";                    then echo "[skip] ${name}: finished";  return; fi
  if grep -qx "$name" <<< "$QUEUE";     then echo "[skip] ${name}: in queue";  return; fi
  local args=(--suite "$SUITE" --method "$method" --ckpt-step "$ckpt" --seed "$seed"
              --n-rollouts "$N" --pass-k "$k" --reward "$REWARD"
              --reward-transform "$TRANSFORM" --batch-size "$BATCH_SIZE"
              --total-steps "$TOTAL_STEPS" --max-response-length "$MAX_RESPONSE_LENGTH"
              --extra-eos-token-ids "$EXTRA_EOS_TOKEN_IDS" --val-n "$VAL_N"
              --test-freq "$TEST_FREQ" --save-freq "$SAVE_FREQ")
  if [ "$DRY_RUN" = "1" ]; then
    echo "sbatch -J ${name} ${SBATCH_FLAGS[*]} scripts/slurm/sbatch_train.sh ${args[*]}"
    return
  fi
  # PPO_MICRO_BATCH must reach the job; sbatch --export=ALL forwards the environment.
  PPO_MICRO_BATCH="$(ppo_micro_for "$N")" \
    sbatch --parsable -J "$name" "${SBATCH_FLAGS[@]}" \
      "${EXP_ROOT}/scripts/slurm/sbatch_train.sh" "${args[@]}" \
    | xargs -I{} echo "[sub ] ${name} -> job {}"
}

n=0
case "$SWEEP" in
  checkpoint_sweep)
    for seed in $SEEDS; do for ckpt in $CKPTS; do for method in $METHODS; do
      n=$((n + 1)); submit_arm "$method" "$ckpt" "$seed" "$N_ROLLOUTS" "$PASS_K"
    done; done; done ;;
  rollout_budget)
    for seed in $SEEDS; do for N in ${BUDGETS:-4 16 64 256}; do for method in $METHODS; do
      k="$PASS_K"; [ "$k" -ge "$N" ] && k="$N"
      n=$((n + 1)); submit_arm "$method" "${BUDGET_CKPT:-3000}" "$seed" "$N" "$k"
    done; done; done ;;
  *) echo "unknown sweep '${SWEEP}' (checkpoint_sweep|rollout_budget)" >&2; exit 1 ;;
esac
echo "[done] ${SWEEP}: ${n} arms considered"
