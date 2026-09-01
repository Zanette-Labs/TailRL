#!/usr/bin/env bash
# Train one arm of the comparison. The ONLY difference between the three arms is
# `--method`, which sets `algorithm.adv_estimator`; every other knob below is shared.
#
#   bash scripts/train.sh --method tailrl
#   bash scripts/train.sh --method grpo --seed 1
#   bash scripts/train.sh --method rloo --dry-run
#
# The setup is a deliberately plain single-update on-policy policy gradient: no KL
# penalty, no entropy bonus, no reference model, one PPO epoch with
# `train_batch_size == ppo_mini_batch_size`, so the importance ratio is identically 1
# and the clip never engages. That is what makes the comparison clean -- whatever
# separates the arms is the advantage estimator and nothing else.
#
# Idempotent and resumable: verl's `resume_mode=auto` picks up the rolling checkpoint,
# and a run already at the step cap exits immediately without touching a GPU. Re-run
# the same command to continue an interrupted run.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"

METHOD=""
SEED=0
DRY=0
STEPS="${PIE_TOTAL_STEPS:-2570}"

usage() {
  cat <<'EOF'
usage: train.sh --method {tailrl|grpo|rloo} [options]

  --method M        advantage estimator (required)
  --seed N          data order + case-sampling seed (default 0)
  --steps N         total training steps (default 2570, ~10 epochs at B=64)
  --dry-run         resolve and print the full configuration, start nothing

Shape and capacity, if you need to fit a different GPU (all env vars):
  PIE_B=64                batch size in prompts; also the PPO mini-batch, which is
                          what keeps the update exactly on-policy. Changing it
                          changes steps-per-epoch.
  PIE_G=16                rollouts per prompt. The estimator sees this many rewards
                          per group; below ~8 the tail is too sparse to be informative.
  PIE_MICRO_BS=2          per-GPU micro-batch for the actor forward/backward.
  PIE_GPU_MEM_UTIL=0.5    vLLM KV-cache fraction. The other half is the FSDP actor.
  PIE_MAX_RESP=4096       max response tokens.
  PIE_SAVE_FREQ=20        rolling full-state checkpoint cadence (for resume).
  PIE_CARVEOUT_EVERY=100  permanent weights-only snapshot cadence (for evaluation).
                          Must be a multiple of PIE_SAVE_FREQ.
  BASE_MODEL              default Qwen/Qwen3-1.7B.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --method)  METHOD="$2"; shift 2 ;;
    --seed)    SEED="$2";   shift 2 ;;
    --steps)   STEPS="$2";  shift 2 ;;
    --dry-run) DRY=1;       shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
case "${METHOD}" in
  tailrl|grpo|rloo) ;;
  "") echo "FATAL: --method is required" >&2; usage; exit 2 ;;
  *)  echo "FATAL: --method must be tailrl, grpo or rloo (got '${METHOD}')" >&2; exit 2 ;;
esac

# --- shape ---------------------------------------------------------------------
B="${PIE_B:-64}"
G="${PIE_G:-16}"
MICRO_BS="${PIE_MICRO_BS:-2}"
GPU_MEM_UTIL="${PIE_GPU_MEM_UTIL:-0.5}"
MAX_RESP="${PIE_MAX_RESP:-4096}"
MAX_PROMPT=2560
MAX_MODEL_LEN="${PIE_MAX_MODEL_LEN:-6656}"
TOP_P="${PIE_TOP_P:-1.0}"
SAVE_FREQ="${PIE_SAVE_FREQ:-20}"
CARVEOUT_EVERY="${PIE_CARVEOUT_EVERY:-100}"
KEEP_CKPT="${PIE_KEEP_CKPT:-1}"
MODEL_TAG="${PIE_MODEL_TAG:-$(basename "${BASE_MODEL}" | tr '[:upper:]' '[:lower:]')}"

# vLLM allocates a context window; a prompt plus its response must fit inside it.
if [ "${MAX_MODEL_LEN}" -lt $(( MAX_PROMPT + MAX_RESP )) ]; then
  echo "FATAL: PIE_MAX_MODEL_LEN (${MAX_MODEL_LEN}) < max_prompt(${MAX_PROMPT}) + max_resp(${MAX_RESP})" >&2
  exit 2
fi
# A permanent snapshot is taken during a checkpoint save, so it has to land on one.
if [ $(( CARVEOUT_EVERY % SAVE_FREQ )) -ne 0 ]; then
  echo "FATAL: PIE_CARVEOUT_EVERY (${CARVEOUT_EVERY}) must be a multiple of PIE_SAVE_FREQ (${SAVE_FREQ})" >&2
  exit 2
fi

# --- paths ---------------------------------------------------------------------
RUN_TAG="${MODEL_TAG}_${METHOD}_g${G}_bs${B}_s${SEED}"
OUTPUT_DIR="${CODEOPT_RUN_DIR}/${RUN_TAG}"
ACTOR_CKPT_DIR="${OUTPUT_DIR}/actor_ckpts"
TRAIN_PARQUET="${PIE_PARQUET_ROOT}/pie_gem5_train.parquet"
VAL_PARQUET="${PIE_PARQUET_ROOT}/pie_gem5_val_200.parquet"
# Under --dry-run these are a warning, so the resolved configuration can be inspected
# before the data has been built.
for f in "${TRAIN_PARQUET}" "${VAL_PARQUET}"; do
  [ -f "${f}" ] && continue
  if [ "${DRY}" -eq 1 ]; then
    echo "warning: missing ${f} (run scripts/prepare_dataset.sh)" >&2
  else
    echo "FATAL: missing ${f}; run scripts/prepare_dataset.sh" >&2; exit 2
  fi
done
mkdir -p "${OUTPUT_DIR}" "${ACTOR_CKPT_DIR}"

# --- reward --------------------------------------------------------------------
# The reward engine is configured entirely through the environment: verl's batch
# reward path hands the function no keyword arguments, so there is nowhere else to
# put this. Every value here is part of the experiment's definition.
export PIE_REWARD_IMPL=gem5
export PIE_GEM5_REWARD_KIND="${PIE_GEM5_REWARD_KIND:-ratio}"   # r = src_ticks / rollout_ticks
export PIE_GEM5_REWARD_K=1              # gem5-time ONE case per prompt per step: simulation
                                        # is the cost bottleneck, and per-case times within a
                                        # program are near-uniform, so one case is a
                                        # near-lossless estimate of the whole set.
export PIE_GEM5_CASE_SELECT="${PIE_GEM5_CASE_SELECT:-sample}"  # which case: random per step
export PIE_GEM5_REWARD_SAMPLE=group     # the SAME case for all G rollouts of a prompt, so the
                                        # group's rewards are comparable and the advantage is
                                        # not contaminated by case-to-case variation
export PIE_GEM5_REWARD_SEED="${SEED}"
export PIE_GEM5_BUDGET_MULT=3.0         # a rollout more than 3x slower than the source is
                                        # stopped early and scored 0; without this a pathological
                                        # program can occupy a simulator slot for the whole step
export PIE_GEM5_REWARD_WORKERS="${PIE_GEM5_REWARD_WORKERS:-$(( $(nproc) - 8 ))}"
export PIE_NAT_TIMEOUT="${PIE_NAT_TIMEOUT:-10}"                 # per-case wall cap, correctness gate
export PIE_GEM5_REWARD_NATIVE_BUDGET="${PIE_GEM5_REWARD_NATIVE_BUDGET:-120}"  # per-rollout cumulative
export PIE_GEM5_REWARD_DEADLINE="${PIE_GEM5_REWARD_DEADLINE:-3000}"  # backstop: a step can never stall

# Ablation: replace each group's rewards by their within-group empirical CDF before
# the estimator sees them, which keeps only rank and discards magnitude. Off here.
export PIE_REWARD_CDF_TRANSFORM="${PIE_REWARD_CDF_TRANSFORM:-0}"

# --- checkpoints ---------------------------------------------------------------
# verl keeps only KEEP_CKPT full-state checkpoints, which is enough to resume but
# loses the trajectory. The carve-out hard-links a weights-only snapshot out of every
# CARVEOUT_EVERY-th save, and those snapshots are what the evaluation reads.
export PIE_ACTOR_CARVEOUT_DIR="${ACTOR_CKPT_DIR}"
export PIE_CARVEOUT_EVERY="${CARVEOUT_EVERY}"

# --- runtime -------------------------------------------------------------------
export TMPDIR="${TMPDIR:-/tmp}/codeopt_${USER}_${SLURM_JOB_ID:-$$}"
export PIE_COMPILE_SCRATCH="${TMPDIR}/scratch"
export RAY_TMPDIR="${TMPDIR}/ray"
mkdir -p "${PIE_COMPILE_SCRATCH}" "${RAY_TMPDIR}"
RAY_CPUS="${CODEOPT_RAY_CPUS:-$(( $(nproc) - 8 ))}"
N_GPUS="${CODEOPT_N_GPUS:-${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}}"
[ "${N_GPUS:-0}" -gt 0 ] || { echo "FATAL: no GPUs visible" >&2; exit 2; }

export WANDB_RUN_GROUP="${MODEL_TAG}_code-opt_${METHOD}_G${G}_BS${B}"
EXPERIMENT_NAME="${WANDB_RUN_GROUP}_s${SEED}"
STEPS_DONE="$(cat "${OUTPUT_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)"

cat <<EOF
=== code optimization | ${MODEL_TAG} | ${METHOD} | G=${G} B=${B} seed=${SEED} ===
    reward     ${PIE_GEM5_REWARD_KIND}, K=1 case (${PIE_GEM5_CASE_SELECT}), budget ${PIE_GEM5_BUDGET_MULT}x, ${PIE_GEM5_REWARD_WORKERS} workers
    model      ${BASE_MODEL}  max_resp=${MAX_RESP} micro_bs=${MICRO_BS} gpu_util=${GPU_MEM_UTIL}
    data       ${TRAIN_PARQUET}
    output     ${OUTPUT_DIR}
    progress   ${STEPS_DONE}/${STEPS} steps   save_freq=${SAVE_FREQ} carveout_every=${CARVEOUT_EVERY}
    gpus       ${N_GPUS}
EOF

if [ "${STEPS_DONE}" -ge "${STEPS}" ]; then
  echo "[done] already at ${STEPS_DONE} >= ${STEPS} steps; nothing to do"
  exit 0
fi
if [ "${DRY}" -eq 1 ]; then
  echo "[dry-run] stopping here"
  exit 0
fi

set +e
python3 -m code_opt.train \
  +data.seed="${SEED}" \
  algorithm.adv_estimator="${METHOD}" \
  algorithm.filter_overlong_responses=False \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.type=fixed \
  algorithm.kl_ctrl.kl_coef=0.0 \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  data.train_batch_size="${B}" \
  data.max_prompt_length="${MAX_PROMPT}" \
  data.max_response_length="${MAX_RESP}" \
  data.shuffle=True \
  data.filter_overlong_prompts=True \
  data.enable_thinking=False \
  data.reward_fn_key=data_source \
  actor_rollout_ref.model.path="${BASE_MODEL}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding="${PIE_REMOVE_PADDING:-False}" \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${B}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BS}" \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.clip_ratio=100 \
  actor_rollout_ref.actor.clip_ratio_high=100 \
  actor_rollout_ref.actor.grad_clip=2.0 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO_BS}" \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="${G}" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p="${TOP_P}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
  actor_rollout_ref.rollout.max_num_seqs=256 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_BS}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  reward_model.enable=False \
  reward_model.launch_reward_fn_async=True \
  custom_reward_function.path="${EXP_ROOT}/code_opt/reward/gem5_reward.py" \
  custom_reward_function.name=compute_score \
  trainer.total_training_steps="${STEPS}" \
  trainer.total_epochs=11 \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  "trainer.logger=${CODEOPT_LOGGER:-['console','wandb']}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  "+actor_rollout_ref.actor.checkpoint.contents=['model','optimizer','extra','hf_model']" \
  trainer.test_freq=-1 \
  trainer.resume_mode=auto \
  trainer.max_actor_ckpt_to_keep="${KEEP_CKPT}" \
  trainer.val_before_train=False \
  "insitu_eval.eval_steps=[]" \
  ray_init.num_cpus="${RAY_CPUS}"
TRAIN_EXIT=$?
set -e

echo "=== ${METHOD} seed=${SEED} exit=${TRAIN_EXIT} steps=$(cat "${OUTPUT_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null || echo '?')/${STEPS} ==="
exit "${TRAIN_EXIT}"
