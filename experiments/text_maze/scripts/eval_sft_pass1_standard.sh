#!/bin/bash
# Standard generative eval for one maze SFT checkpoint.
# Uses verl.trainer.main_ppo with trainer.val_only=True and the canonical
# maze_17 binary reward, so pass@1 is the goal_reached sample mean.
# Usage:
#   bash scripts/eval_sft_pass1_standard.sh <ckpt_step> <gpu_ids> [val_n] [parquet_dir]
set -euo pipefail

CKPT_STEP=${1:?"Usage: $0 <ckpt_step> <gpu_ids> [val_n] [parquet_dir]"}
GPU_IDS=${2:?}
VAL_N=${3:-1000}
PARQUET_DIR=${4:-${TAILRL_MAZE_DATA_DIR:-data}/main_parquet_eval1000}
ROLLOUT_MICRO_BATCH_SIZE="${ROLLOUT_MICRO_BATCH_SIZE:-5000}"

# shellcheck disable=SC1091
. "$(dirname "$0")/env.sh"
cd "$EXP_ROOT"

HF_REPO="${HF_REPO:-${TAILRL_MAZE_CKPT_REPO}}"

if [ -n "${MODEL_BASE_DIR:-}" ]; then
  MODEL_PATH="${MODEL_BASE_DIR%/}/ckpt-${CKPT_STEP}"
else
  MODEL_PATH="${HF_CACHE_DIR%/}/ckpt-${CKPT_STEP}"
  if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "[init] downloading ckpt-${CKPT_STEP} from ${HF_REPO} -> ${HF_CACHE_DIR}"
    hf download "${HF_REPO}" --include "ckpt-${CKPT_STEP}/*" --local-dir "${HF_CACHE_DIR}"
  fi
fi

TRAIN_DATA="${PARQUET_DIR}/test.parquet"
VAL_DATA="${PARQUET_DIR}/test.parquet"
if [ ! -f "$VAL_DATA" ]; then
  echo "Missing eval parquet: ${VAL_DATA}" >&2
  echo "Build it with: TEST_PROMPTS=1000 bash scripts/prepare_dataset.sh data/main_1.3M.jsonl data/sft_data_eval1000 data/main_parquet_eval1000" >&2
  exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
  echo "Missing model checkpoint: ${MODEL_PATH}" >&2
  exit 1
fi

# MANDATORY preflight: repair + verify the checkpoint for the container's
# transformers 4.x (rope_theta / tokenizer_class). Idempotent; aborts on a
# checkpoint that would load broken. See scripts/checkpoint_doctor.py.
python3 scripts/checkpoint_doctor.py "$MODEL_PATH" \
  || { echo "[FATAL] ${MODEL_PATH} failed checkpoint repair/verify" >&2; exit 1; }

GPU_TAG="${GPU_IDS//,/_}"
FIRST_GPU="${GPU_IDS%%,*}"
NUM_GPUS=$(awk -F',' '{print NF}' <<< "$GPU_IDS")
PROJECT_NAME="tailrl-text-maze"
EXPERIMENT_NAME="eval1000x${VAL_N}_standard_ckpt${CKPT_STEP}_${NUM_GPUS}gpu_${GPU_TAG}"
LOG_DIR="${LOG_DIR:-${TAILRL_LOG_DIR:-runs/logs}/sft_pass1}"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONHASHSEED="${SEED:-0}"
export NCCL_TIMEOUT=18000
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# SKIP_CONDA_ACTIVATE=1: containerized runs (apptainer/docker) already have the
# full python env on PATH and no conda.
if [ "${SKIP_CONDA_ACTIVATE:-0}" != "1" ] && [ -n "${CONDA_ENV:-}" ]; then
  if ! command -v conda >/dev/null 2>&1; then
    for candidate in \
      "${CONDA_PREFIX:-}/etc/profile.d/conda.sh" \
      "$HOME/miniconda3/etc/profile.d/conda.sh" \
      "$HOME/anaconda3/etc/profile.d/conda.sh" \
      "/opt/conda/etc/profile.d/conda.sh" \
      "/data/miniconda/etc/profile.d/conda.sh"; do
      if [ -n "$candidate" ] && [ -f "$candidate" ]; then
        # shellcheck disable=SC1090
        source "$candidate"
        break
      fi
    done
  fi
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

# Port-band index defaults to the first visible GPU id; sbatch wrappers export
# RAY_PORT_INDEX (first *physical* GPU id) since cgroups renumber GPUs from 0.
RAY_PORT_INDEX="${RAY_PORT_INDEX:-$FIRST_GPU}"
# 43x00 family: outside co-tenant Ray default worker range (10002-19999, which
# swallowed the old 19x00 band) and outside the OpenRLHF 20000-40000 head band.
RAY_PORT=$((43000 + RAY_PORT_INDEX))
RAY_CLIENT_PORT=$((43100 + RAY_PORT_INDEX))
RAY_NODE_MANAGER_PORT=$((43200 + RAY_PORT_INDEX))
RAY_OBJECT_MANAGER_PORT=$((43300 + RAY_PORT_INDEX))
RAY_RUNTIME_ENV_AGENT_PORT=$((43400 + RAY_PORT_INDEX))
RAY_METRICS_PORT=$((43500 + RAY_PORT_INDEX))
RAY_DASHBOARD_PORT=$((43600 + RAY_PORT_INDEX))
RAY_DASHBOARD_AGENT_HTTP_PORT=$((43700 + RAY_PORT_INDEX))
RAY_DASHBOARD_AGENT_GRPC_PORT=$((43800 + RAY_PORT_INDEX))
RAY_TEMP_DIR="/tmp/ray_maze_eval1000_p${RAY_PORT_INDEX}_gpu${GPU_TAG}"
case "$RAY_PORT_INDEX" in
  0) RAY_MIN_WORKER_PORT=50000; RAY_MAX_WORKER_PORT=50399 ;;
  1) RAY_MIN_WORKER_PORT=50400; RAY_MAX_WORKER_PORT=50799 ;;
  2) RAY_MIN_WORKER_PORT=50800; RAY_MAX_WORKER_PORT=51199 ;;
  3) RAY_MIN_WORKER_PORT=51200; RAY_MAX_WORKER_PORT=51599 ;;
  4) RAY_MIN_WORKER_PORT=51600; RAY_MAX_WORKER_PORT=51999 ;;
  5) RAY_MIN_WORKER_PORT=54400; RAY_MAX_WORKER_PORT=54799 ;;
  6) RAY_MIN_WORKER_PORT=54800; RAY_MAX_WORKER_PORT=55199 ;;
  7) RAY_MIN_WORKER_PORT=55200; RAY_MAX_WORKER_PORT=55599 ;;
  *) echo "RAY_PORT_INDEX (default: first GPU id) must be 0..7, got ${RAY_PORT_INDEX} (gpu_ids=${GPU_IDS})" >&2; exit 1 ;;
esac

kill_by_pattern() {
  local pattern="$1"
  local pid=""
  while read -r pid; do
    [ -n "$pid" ] || continue
    [ "$pid" = "$$" ] && continue
    kill -9 "$pid" 2>/dev/null || true
  done < <(pgrep -f -- "$pattern" 2>/dev/null || true)
}

echo "[GPU ${GPU_IDS}] cleanup Ray temp=${RAY_TEMP_DIR}, port=${RAY_PORT}"
for _ in 1 2; do
  kill_by_pattern "${RAY_TEMP_DIR}/"
  kill_by_pattern "gcs-address=[^ ]*:${RAY_PORT}"
  kill_by_pattern "gcs_server_port=${RAY_PORT}"
  kill_by_pattern "--port=${RAY_PORT}"
  sleep 1
done
rm -rf "$RAY_TEMP_DIR"
mkdir -p "$RAY_TEMP_DIR"

ray start --head \
  --num-gpus="${NUM_GPUS}" \
  --num-cpus=8 \
  --port="${RAY_PORT}" \
  --ray-client-server-port="${RAY_CLIENT_PORT}" \
  --node-manager-port="${RAY_NODE_MANAGER_PORT}" \
  --object-manager-port="${RAY_OBJECT_MANAGER_PORT}" \
  --include-dashboard=False \
  --dashboard-port="${RAY_DASHBOARD_PORT}" \
  --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_HTTP_PORT}" \
  --dashboard-agent-grpc-port="${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
  --runtime-env-agent-port="${RAY_RUNTIME_ENV_AGENT_PORT}" \
  --metrics-export-port="${RAY_METRICS_PORT}" \
  --temp-dir="${RAY_TEMP_DIR}" \
  --min-worker-port="${RAY_MIN_WORKER_PORT}" \
  --max-worker-port="${RAY_MAX_WORKER_PORT}" \
  --disable-usage-stats

export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"
LOG_FILE="${LOG_DIR}/ckpt-${CKPT_STEP}_G${VAL_N}_gpu${GPU_TAG}.log"

echo "[GPU ${GPU_IDS}] standard eval ckpt-${CKPT_STEP}, val_n=${VAL_N}, log=${LOG_FILE}"
set +e
timeout --signal=SIGTERM "${TIMEOUT_SECONDS:-1209600}" \
python3 -m verl.trainer.main_ppo \
  ray_init.ray_dir="${RAY_TEMP_DIR}" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  +data.seed="${SEED:-0}" \
  data.train_files="${TRAIN_DATA}" \
  data.val_files="${VAL_DATA}" \
  data.train_batch_size=256 \
  data.max_prompt_length=320 \
  data.max_response_length=180 \
  data.apply_chat_template=False \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=1e-4 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.dtype=${DTYPE:-float16} \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=256 \
  actor_rollout_ref.rollout.name=hf \
  +actor_rollout_ref.rollout.micro_batch_size="${ROLLOUT_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.rollout.dtype=${DTYPE:-float16} \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8192 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8192 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}" \
  actor_rollout_ref.rollout.val_kwargs.gen_batch_size="${VAL_GEN_BATCH_SIZE:-100}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  reward_model.reward_manager=prime \
  +reward_model.reward_kwargs.num_processes=4 \
  +reward_model.reward_kwargs.chunksize=64 \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.logger=['console'] \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.n_gpus_per_node="${NUM_GPUS}" \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_training_steps=1 \
  trainer.default_local_dir="./checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
  trainer.total_epochs=1 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e

for _ in 1 2; do
  kill_by_pattern "${RAY_TEMP_DIR}/"
  kill_by_pattern "gcs-address=[^ ]*:${RAY_PORT}"
  kill_by_pattern "gcs_server_port=${RAY_PORT}"
  kill_by_pattern "--port=${RAY_PORT}"
  sleep 1
done
rm -rf "$RAY_TEMP_DIR" 2>/dev/null || true

echo "[GPU ${GPU_IDS}] finished ckpt-${CKPT_STEP} exit=${EXIT_CODE}"
exit "${EXIT_CODE}"
