#!/bin/bash
# INFERENCE-TIME SCALING LADDER — Pass@k and Best-of-k reward for every finished
# RL run, evaluated on the 1000 held-out mazes with a large sampling budget.
#
# For each arm it re-enters the trained policy with `--val-only`, draws VAL_N
# rollouts per prompt in ONE validation pass, and prints every metric verl
# computes — mean@VAL_N plus best@{2,4,8,...,VAL_N} for three families:
#   is_shortest    solved via a shortest path  (the headline; = Pass@k)
#   goal_reached   reached the goal at all
#   reward         the continuous composite_v2 score  (= Best-of-k reward)
# The console logs are then compiled into one JSON with per-seed mean and std.
#
# Usage:
#   bash scripts/reproduce/eval_ladder.sh [--dry-run]
#   VAL_N=1024 bash scripts/reproduce/eval_ladder.sh          # cheaper ladder
#   OUT_JSON=paper_results/my_eval.json bash scripts/reproduce/eval_ladder.sh
#
# Cost scales linearly in VAL_N: 4096 rollouts x 1000 prompts is ~4.1M
# generations per arm.
set -uo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/_common.sh"

EVAL_VAL_N="${EVAL_VAL_N:-4096}"
EVAL_BUDGETS="${EVAL_BUDGETS:-$N_ROLLOUTS}"     # which trained-N arms to evaluate
EVAL_CKPTS="${EVAL_CKPTS:-$CKPTS}"
OUT_JSON="${OUT_JSON:-${EXP_ROOT}/paper_results/eval_ladder_k${EVAL_VAL_N}.json}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-${TAILRL_LOG_DIR}/eval}"
mkdir -p "$EVAL_LOG_DIR"

evaluated=0; skipped=0; failed=0
for seed in $SEEDS; do
  for ckpt in $EVAL_CKPTS; do
    for N in $EVAL_BUDGETS; do
      for method in $METHODS; do
        name="$(exp_name "$(method_tag "$method")" "$ckpt" "$N" "$seed")"
        if ! is_done "$name"; then
          echo "[skip] ${method} ckpt${ckpt} G${N} seed${seed}: training not finished"
          skipped=$((skipped + 1)); continue
        fi
        log="${EVAL_LOG_DIR}/eval_${method}_ck${ckpt}_G${N}_s${seed}.log"
        if grep -qaE 'Loaded model from .*global_step_[0-9]+' "$log" 2>/dev/null; then
          echo "[have] $(basename "$log")"; evaluated=$((evaluated + 1)); continue
        fi
        k="$PASS_K"; [ "$k" -ge "$N" ] && { k="$N"; export PKPO_ALLOW_DEGENERATE=1; }
        args=(
          --suite "$SUITE" --method "$method" --ckpt-step "$ckpt" --seed "$seed"
          --n-rollouts "$N" --pass-k "$k" --reward "$REWARD" --reward-transform "$TRANSFORM"
          --batch-size "$BATCH_SIZE" --max-response-length "$MAX_RESPONSE_LENGTH"
          --extra-eos-token-ids "$EXTRA_EOS_TOKEN_IDS"
          --val-n "$EVAL_VAL_N" --val-only
          --val-parquet-dir "${EXP_ROOT}/data/eval1000"
          --gpu-ids "$GPU_IDS"
        )
        if [ "$DRY_RUN" = "1" ]; then
          echo "bash scripts/train.sh ${args[*]}"; unset PKPO_ALLOW_DEGENERATE; continue
        fi
        echo "[eval] ${method} ckpt${ckpt} G${N} seed${seed} -> best@{2..${EVAL_VAL_N}}"
        PPO_MICRO_BATCH="$(ppo_micro_for "$N")" \
          bash scripts/train.sh "${args[@]}" 2>&1 | tee "$log"
        unset PKPO_ALLOW_DEGENERATE

        # GUARD. `--val-only` restores the trained weights via resume_mode=auto from
        # $TAILRL_MAZE_RUN_DIR/checkpoints/$WANDB_PROJECT/<experiment>/. If WANDB_PROJECT
        # differs from the one the run TRAINED under, that directory does not exist,
        # verl silently falls back to the BASE SFT checkpoint, and the job exits 0
        # having measured the wrong model. This has happened: an entire evaluation
        # batch reported base-checkpoint numbers. A real resume always logs
        # "Loaded model from .../global_step_N"; its absence is a hard failure.
        if ! grep -qaE 'Loaded model from .*global_step_[0-9]+' "$log"; then
          echo "[FATAL] ${name}: no 'Loaded model from .../global_step_N' in the log."
          echo "        This eval ran on the BASE SFT checkpoint, not the trained model."
          echo "        WANDB_PROJECT=${WANDB_PROJECT} — is that where this run trained?"
          rm -f "$log"; failed=$((failed + 1)); continue
        fi
        evaluated=$((evaluated + 1))
      done
    done
  done
done

echo "[done] evaluated=${evaluated} skipped=${skipped} failed=${failed}"
if [ "$DRY_RUN" = "0" ] && [ "$evaluated" -gt 0 ]; then
  mkdir -p "$(dirname "$OUT_JSON")"
  python3 scripts/eval_logs_to_json.py "$OUT_JSON" "$EVAL_LOG_DIR"
fi
