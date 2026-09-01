#!/bin/bash
# Shared configuration for the reproduce scripts. Sourced, not run.
#
# Every number below is the value used for the paper's Text-Maze results. The
# reproduce scripts pass them explicitly rather than relying on launcher
# defaults, so a change to scripts/train.sh cannot silently move the numbers.

# shellcheck disable=SC1091
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
cd "$EXP_ROOT" || exit 1

# --- the four estimators compared in the paper --------------------------------
#   tailrl  TailRL: tail-likelihood / gap-over-survivors  (this paper)
#   grpo    group-relative, z-scored baseline
#   rloo    leave-one-out baseline
#   pkpo    Pass@K policy optimization, continuous form (k_opt = PASS_K)
METHODS="${METHODS:-tailrl grpo rloo pkpo}"

# --- the seven SFT initializations (weakest -> strongest) ----------------------
# Shortest-path success rate of each before any RL, in percent:
#   2450 0.0122 | 3000 0.0244 | 3250 0.0427 | 3350 0.1556
#   3400 0.2197 | 3450 0.4822 | 3550 0.8270
CKPTS="${CKPTS:-2450 3000 3250 3350 3400 3450 3550}"
SEEDS="${SEEDS:-0 1 2}"

# --- RL configuration (identical across every arm, so only the estimator varies)
SUITE="${SUITE:-main}"
REWARD="${REWARD:-composite_v2}"       # r = .5*min(1,(L*-d)/L*) + .5*min(1,L*/L)*1[goal]
TRANSFORM="${TRANSFORM:-raw}"          # no reward transform
BATCH_SIZE="${BATCH_SIZE:-256}"        # prompts per optimization step
N_ROLLOUTS="${N_ROLLOUTS:-16}"         # N, rollouts per prompt
PASS_K="${PASS_K:-8}"                  # k_opt for pkpo; must be < N_ROLLOUTS
TOTAL_STEPS="${TOTAL_STEPS:-5001}"     # 5000 steps + the step-0 validation
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-180}"
EXTRA_EOS_TOKEN_IDS="${EXTRA_EOS_TOKEN_IDS:-7}"   # maze DONE token terminates generation
VAL_N="${VAL_N:-64}"                   # validation samples/prompt -> best@{2..64}
TEST_FREQ="${TEST_FREQ:-1000}"
SAVE_FREQ="${SAVE_FREQ:-250}"
GPU_IDS="${GPU_IDS:-0}"

# --- derived, and easy to get wrong -------------------------------------------
# ppo_micro_batch_size_per_gpu must DIVIDE batch_size * N. 4096 is the tuned
# value but asserts at N=4 (256*4 = 1024 < 4096), hence the min().
ppo_micro_for() { local n="$1"; local v=$(( BATCH_SIZE * n )); [ "$v" -gt 4096 ] && v=4096; echo "$v"; }
# The generation chunk must divide every generation batch (hf_rollout splits with
# DataProto.chunk, which asserts an equal split). 8000 works for N in {4,16,64,256}
# at batch 256; drop to 4000 on cards with <= 48 GB.
export ROLLOUT_MICRO_BATCH_SIZE="${ROLLOUT_MICRO_BATCH_SIZE:-8000}"
export REWARD_NUM_PROCESSES="${REWARD_NUM_PROCESSES:-16}"

DRY_RUN=0
for a in "$@"; do [ "$a" = "--dry-run" ] && DRY_RUN=1; done

# experiment_name as scripts/train.sh derives it — used for the skip-if-done check
exp_name() {  # exp_name <method_tag> <ckpt> <n_rollouts> <seed>
  echo "textmaze_${SUITE}_${1}_${REWARD}_ckpt${2}_bs${BATCH_SIZE}_G${3}_1gpu_${GPU_IDS//,/_}_seed${4}"
}
method_tag() {  # pkpo runs register under its concrete estimator name
  case "$1" in pkpo) echo "pkpo-pkpo_continuous" ;; *) echo "$1" ;; esac
}

is_done() {  # is_done <experiment_name>
  local f="${TAILRL_MAZE_RUN_DIR}/checkpoints/${WANDB_PROJECT}/${1}/latest_checkpointed_iteration.txt"
  [ -f "$f" ] || return 1
  local it; it=$(tr -d '[:space:]' < "$f" 2>/dev/null)
  [ -n "$it" ] && [ "$it" -ge "$TOTAL_STEPS" ] 2>/dev/null
}

# Run one arm. Re-running is safe: verl resumes from the latest checkpoint
# (resume_mode=auto), and a finished arm is skipped outright.
run_arm() {  # run_arm <method> <ckpt> <seed> [n_rollouts] [pass_k]
  local method="$1" ckpt="$2" seed="$3" n="${4:-$N_ROLLOUTS}" k="${5:-$PASS_K}"
  local name; name="$(exp_name "$(method_tag "$method")" "$ckpt" "$n" "$seed")"
  if is_done "$name"; then
    echo "[skip] $name already at ${TOTAL_STEPS} steps"
    return 0
  fi
  local args=(
    --suite "$SUITE" --method "$method" --ckpt-step "$ckpt" --seed "$seed"
    --n-rollouts "$n" --pass-k "$k" --reward "$REWARD" --reward-transform "$TRANSFORM"
    --batch-size "$BATCH_SIZE" --total-steps "$TOTAL_STEPS"
    --max-response-length "$MAX_RESPONSE_LENGTH"
    --extra-eos-token-ids "$EXTRA_EOS_TOKEN_IDS"
    --val-n "$VAL_N" --test-freq "$TEST_FREQ" --save-freq "$SAVE_FREQ"
    --train-parquet-dir "${TAILRL_MAZE_DATA_DIR}/main_parquet"
    --val-parquet-dir "${EXP_ROOT}/data/eval1000"
    --gpu-ids "$GPU_IDS"
  )
  if [ "$DRY_RUN" = "1" ]; then
    echo "bash scripts/train.sh ${args[*]}"
    return 0
  fi
  mkdir -p "$TAILRL_LOG_DIR"
  echo "[run ] $name"
  PPO_MICRO_BATCH="$(ppo_micro_for "$n")" \
    bash scripts/train.sh "${args[@]}" 2>&1 | tee "${TAILRL_LOG_DIR}/${name}.log"
}
