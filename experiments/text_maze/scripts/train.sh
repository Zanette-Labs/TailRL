#!/bin/bash
# Single-run launcher: one RL fine-tune of one SFT checkpoint on the 17x17 text maze.
#
# Resolves an experiment name plus every Hydra override and calls
# `python -m verl.trainer.main_ppo` against the vendored verl fork. One run = one
# (method, checkpoint, seed, G) cell of the sweeps in scripts/reproduce/.
#
# Usage example:
#   bash scripts/train.sh --method tailrl --ckpt-step 3000 --seed 0 --gpu-ids 0
set -euo pipefail

# shellcheck disable=SC1091
. "$(dirname "$0")/env.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train.sh [options]

Required:
  --method METHOD              tailrl|grpo|rloo|pkpo, or any registered adv_estimator
  --gpu-ids IDS                comma-separated visible GPU ids, e.g. 0 or 0,1

Choose one init:
  --pass-rate-index IDX        0..7 for the exp_plan_0528 pass-rate ladder
  --pass-rate-id ID            one of pr0p01,pr0p02,pr0p04,pr0p08,pr0p16,pr0p32,pr0p64,pr1p28
  --ckpt-step STEP             explicit SFT checkpoint step

Common options:
  --suite NAME                 campaign label baked into W&B/run names (default: main)
  --seed SEED                  random seed (default: 0)
  --n-rollouts N               rollout samples per prompt, G in old scripts (default: 16)
  --reward TYPE                continuous_ub|continuous_ratio|continuous_2stage|binary_shortest|composite_v1|composite_v2|binary (default: composite_v2)
  --reward-param VALUE         UB for continuous_ub or partial reward for continuous_2stage
  --reward-transform T         raw|cdf reward transform before the advantage estimator (default: raw)
  --pass-k K                   PKPO/PKPO-continuous max@K value (default: 16)
  --batch-size N               train prompt batch size (default: 256)
  --total-steps N              trainer.total_training_steps (default: 10001)
  --train-parquet-dir DIR      train parquet dir (default: data/main_parquet)
  --val-parquet-dir DIR        validation parquet dir (default: data/main_parquet_eval1000)
  --parquet-dir DIR            backward-compatible shortcut setting both train/val dirs
  --val-prompts N              expected val prompt count; 0 disables check (default: 1000)
  --val-n N                    validation samples per prompt (default: 512)
  --model-base-dir DIR         local dir containing ckpt-<step>; overrides HF download
  --hf-repo REPO               HF SFT checkpoint repo (default: $TAILRL_MAZE_CKPT_REPO)
  --hf-cache-dir DIR           HF checkpoint cache dir (default: $HF_CACHE_DIR)
  --conda-env NAME             conda env to activate (default: none)
  --dry-run                    resolve config and print it without starting Ray/training

Env overrides:
  PASS_RATE_CKPTS              comma list of 8 ckpt steps for the ladder
  PKPO_ESTIMATOR               pkpo_continuous|pkpo (default: pkpo_continuous for non-binary rewards)
  WANDB_ENTITY, WANDB_MODE     W&B settings
  WANDB_PROJECT                W&B project name (default: tailrl-text-maze)
  SKIP_CONDA_ACTIVATE=1        skip `conda activate` (containers / already-active envs)
  ROLLOUT_MICRO_BATCH_SIZE     generation chunk size; must divide every generation
                               batch (default 8000; use 4000 on <=48 GB cards)
EOF
}

SUITE="main"
METHOD=""
GPU_IDS=""
SEED=0
N_ROLLOUTS=16
REWARD_TYPE="composite_v2"
REWARD_PARAM=""
REWARD_TRANSFORM="raw"
PASS_K=16
BATCH_SIZE=256
TOTAL_STEPS=10001
MAX_RESPONSE_LENGTH=180
# Comma-separated extra token ids that also terminate generation (e.g. the maze DONE
# token, id 7). Empty -> vanilla EOS behaviour. Passed to rollout.extra_eos_token_ids.
EXTRA_EOS_TOKEN_IDS="${EXTRA_EOS_TOKEN_IDS:-}"
TRAIN_PARQUET_DIR="${TAILRL_MAZE_DATA_DIR}/main_parquet"
# The validation split is FIXED: the 1000 held-out mazes committed at
# data/eval1000/, which every published number was measured on. It lives with
# the source, not in $TAILRL_MAZE_DATA_DIR, so pointing that at scratch for the
# 5 GB training corpus does not move the eval set out from under you.
VAL_PARQUET_DIR="${EXP_ROOT}/data/eval1000"
VAL_PROMPTS=1000
VAL_N=512
CKPT_STEP=""
PASS_RATE_INDEX=""
PASS_RATE_ID=""
MODEL_BASE_DIR_ARG=""
HF_REPO="${HF_REPO:-${TAILRL_MAZE_CKPT_REPO}}"
CONDA_ENV_NAME="${CONDA_ENV:-}"
DRY_RUN=0
SAVE_FREQ=250
TEST_FREQ=250
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1209600}"
ROLLOUT_MICRO_BATCH_SIZE="${ROLLOUT_MICRO_BATCH_SIZE:-8000}"
VAL_GEN_BATCH_SIZE="${VAL_GEN_BATCH_SIZE:-128}"
# Re-eval (val-only) mode: load the final checkpoint, run ONE validation, log it to
# the existing wandb run at a fixed step, then exit. No training, no new checkpoints.
VAL_ONLY=0
WANDB_RESUME_ID=""          # existing wandb run id to append the step-5k eval into
RESUME_GLOBAL_STEPS=""      # force the logged step (e.g. 5000) regardless of ckpt step

while [ "$#" -gt 0 ]; do
  case "$1" in
    --suite) SUITE="${2:?}"; shift 2 ;;
    --method) METHOD="${2:?}"; shift 2 ;;
    --gpu-ids) GPU_IDS="${2:?}"; shift 2 ;;
    --seed) SEED="${2:?}"; shift 2 ;;
    --n-rollouts) N_ROLLOUTS="${2:?}"; shift 2 ;;
    --reward) REWARD_TYPE="${2:?}"; shift 2 ;;
    --reward-param) REWARD_PARAM="${2:?}"; shift 2 ;;
    --reward-transform) REWARD_TRANSFORM="${2:?}"; shift 2 ;;
    --pass-k) PASS_K="${2:?}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:?}"; shift 2 ;;
    --total-steps) TOTAL_STEPS="${2:?}"; shift 2 ;;
    --max-response-length) MAX_RESPONSE_LENGTH="${2:?}"; shift 2 ;;
    --extra-eos-token-ids) EXTRA_EOS_TOKEN_IDS="${2:?}"; shift 2 ;;
    --train-parquet-dir) TRAIN_PARQUET_DIR="${2:?}"; shift 2 ;;
    --val-parquet-dir) VAL_PARQUET_DIR="${2:?}"; shift 2 ;;
    --parquet-dir) TRAIN_PARQUET_DIR="${2:?}"; VAL_PARQUET_DIR="${2:?}"; shift 2 ;;
    --val-prompts) VAL_PROMPTS="${2:?}"; shift 2 ;;
    --val-n) VAL_N="${2:?}"; shift 2 ;;
    --ckpt-step) CKPT_STEP="${2:?}"; shift 2 ;;
    --pass-rate-index) PASS_RATE_INDEX="${2:?}"; shift 2 ;;
    --pass-rate-id) PASS_RATE_ID="${2:?}"; shift 2 ;;
    --model-base-dir) MODEL_BASE_DIR_ARG="${2:?}"; shift 2 ;;
    --hf-repo) HF_REPO="${2:?}"; shift 2 ;;
    --hf-cache-dir) HF_CACHE_DIR="${2:?}"; shift 2 ;;
    --conda-env) CONDA_ENV_NAME="${2:?}"; shift 2 ;;
    --save-freq) SAVE_FREQ="${2:?}"; shift 2 ;;
    --test-freq) TEST_FREQ="${2:?}"; shift 2 ;;
    --val-only) VAL_ONLY=1; shift ;;
    --wandb-resume-id) WANDB_RESUME_ID="${2:?}"; shift 2 ;;
    --resume-global-steps) RESUME_GLOBAL_STEPS="${2:?}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [ -z "$METHOD" ] || [ -z "$GPU_IDS" ]; then
  usage >&2
  exit 1
fi

cd "$EXP_ROOT"

IFS=',' read -r -a PASS_RATE_IDS <<< "${PASS_RATE_IDS:-pr0p01,pr0p02,pr0p04,pr0p08,pr0p16,pr0p32,pr0p64,pr1p28}"
IFS=',' read -r -a PASS_RATE_PCTS <<< "${PASS_RATE_PCTS:-0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28}"
IFS=',' read -r -a PASS_RATE_CKPT_STEPS <<< "${PASS_RATE_CKPTS:-2850,2950,3150,3200,3400,3600,3450,3550}"

if [ "${#PASS_RATE_IDS[@]}" -ne 8 ] || [ "${#PASS_RATE_PCTS[@]}" -ne 8 ] || [ "${#PASS_RATE_CKPT_STEPS[@]}" -ne 8 ]; then
  echo "PASS_RATE_IDS, PASS_RATE_PCTS, and PASS_RATE_CKPTS must each contain 8 comma-separated entries." >&2
  exit 1
fi

PASS_RATE_TAG=""
PASS_RATE_PCT=""
if [ -n "$PASS_RATE_ID" ] && [ -z "$PASS_RATE_INDEX" ]; then
  for i in "${!PASS_RATE_IDS[@]}"; do
    if [ "${PASS_RATE_IDS[$i]}" = "$PASS_RATE_ID" ]; then
      PASS_RATE_INDEX="$i"
      break
    fi
  done
  if [ -z "$PASS_RATE_INDEX" ]; then
    echo "Unknown --pass-rate-id ${PASS_RATE_ID}" >&2
    exit 1
  fi
fi

if [ -n "$PASS_RATE_INDEX" ]; then
  if ! [[ "$PASS_RATE_INDEX" =~ ^[0-7]$ ]]; then
    echo "--pass-rate-index must be 0..7, got ${PASS_RATE_INDEX}" >&2
    exit 1
  fi
  PASS_RATE_TAG="${PASS_RATE_IDS[$PASS_RATE_INDEX]}"
  PASS_RATE_PCT="${PASS_RATE_PCTS[$PASS_RATE_INDEX]}"
  if [ -z "$CKPT_STEP" ]; then
    CKPT_STEP="${PASS_RATE_CKPT_STEPS[$PASS_RATE_INDEX]}"
  fi
fi

if [ -z "$CKPT_STEP" ]; then
  echo "Provide --ckpt-step or --pass-rate-index/--pass-rate-id." >&2
  exit 1
fi

if [ -z "$PASS_RATE_TAG" ]; then
  PASS_RATE_TAG="ckpt${CKPT_STEP}"
  PASS_RATE_PCT="custom"
fi

case "$METHOD" in
  pkpo)
    case "$REWARD_TYPE" in
      binary|binary_shortest) ADVANTAGE_ESTIMATOR="${PKPO_ESTIMATOR:-pkpo}" ;;
      *) ADVANTAGE_ESTIMATOR="${PKPO_ESTIMATOR:-pkpo_continuous}" ;;
    esac
    METHOD_TAG="pkpo-${ADVANTAGE_ESTIMATOR}"
    ;;
  *)
    ADVANTAGE_ESTIMATOR="$METHOD"
    METHOD_TAG="$METHOD"
    ;;
esac

# PKPO guard: pass_k must be STRICTLY LESS than n_rollouts.
#   k > n  -- invalid: C(n-1,k-1)=0, so max@k is undefined with fewer than k samples per
#             prompt. pkpo_continuous asserts; the binary pkpo used to emit silent all-NaN
#             advantages (and under actor.dtype=float16 the grad scaler then skips every
#             step forever). Fail here rather than after the container and Ray have started.
#   k == n -- mathematically valid but DEGENERATE: max(g_1..g_n) is invariant to every
#             non-maximal sample, so the advantage is 0 for all but the argmax and
#             (n-1)/n of the batch contributes no gradient at all (it collapses to
#             grpo_passk without its /std). This is the SHIPPED DEFAULT, because PASS_K
#             and N_ROLLOUTS both default to 16 and are set independently above -- so
#             `--method pkpo` with no other flags silently lands on it. Set
#             PKPO_ALLOW_DEGENERATE=1 to run it deliberately.
case "$ADVANTAGE_ESTIMATOR" in
  pkpo|pkpo_continuous)
    if ! [[ "$PASS_K" =~ ^[0-9]+$ ]] || ! [[ "$N_ROLLOUTS" =~ ^[0-9]+$ ]]; then
      echo "[FATAL] --pass-k ('${PASS_K}') and --n-rollouts ('${N_ROLLOUTS}') must be integers." >&2
      exit 1
    fi
    if [ "$PASS_K" -lt 1 ]; then
      echo "[FATAL] --pass-k must be >= 1, got ${PASS_K}." >&2
      exit 1
    fi
    if [ "$PASS_K" -gt "$N_ROLLOUTS" ]; then
      echo "[FATAL] pass_k=${PASS_K} > n_rollouts=${N_ROLLOUTS}: max@k needs at least k samples per prompt." >&2
      echo "        Fix: --pass-k <${N_ROLLOUTS}, or raise --n-rollouts above ${PASS_K}." >&2
      exit 1
    fi
    if [ "$PASS_K" -eq "$N_ROLLOUTS" ] && [ "${PKPO_ALLOW_DEGENERATE:-0}" != "1" ]; then
      echo "[FATAL] pass_k=${PASS_K} == n_rollouts=${N_ROLLOUTS}: max@k is DEGENERATE at k=n --" >&2
      echo "        only the single best rollout per prompt gets a nonzero advantage, so" >&2
      echo "        $((N_ROLLOUTS - 1))/${N_ROLLOUTS} of every batch contributes no gradient." >&2
      echo "        Both flags default to 16, so this is what you get if you set neither." >&2
      echo "        Fix: --pass-k < ${N_ROLLOUTS} (e.g. --pass-k $((N_ROLLOUTS / 4)))," >&2
      echo "        or set PKPO_ALLOW_DEGENERATE=1 to run k=n on purpose." >&2
      exit 1
    fi
    if [ "$PASS_K" -eq "$N_ROLLOUTS" ]; then
      echo "[config] pkpo guard: pass_k=${PASS_K} == n_rollouts=${N_ROLLOUTS} -- DEGENERATE k=n run, allowed only because PKPO_ALLOW_DEGENERATE=1 (estimator=${ADVANTAGE_ESTIMATOR})"
    else
      echo "[config] pkpo guard OK: pass_k=${PASS_K} < n_rollouts=${N_ROLLOUTS} (estimator=${ADVANTAGE_ESTIMATOR})"
    fi
    ;;
esac

CONTINUOUS_UB=${REWARD_PARAM:-${CONTINUOUS_UB:-60}}
CONTINUOUS_2STAGE_PARTIAL=${REWARD_PARAM:-${CONTINUOUS_2STAGE_PARTIAL:-0.5}}

REWARD_FN_CONTINUOUS_UB="src/maze_continuous_ub_reward.py"
REWARD_FN_CONTINUOUS_RATIO="src/maze_continuous_ratio_reward.py"
REWARD_FN_CONTINUOUS_2STAGE="src/maze_continuous_2stage_reward.py"
REWARD_FN_BINARY_SHORTEST="src/maze_binary_shortest_reward.py"
REWARD_FN_COMPOSITE_V1="src/maze_composite_v1_reward.py"
REWARD_FN_COMPOSITE_V2="src/maze_composite_v2_reward.py"

case "$REWARD_TYPE" in
  continuous_ub|continuous_ub60)
    TRAIN_DATA="${TRAIN_PARQUET_DIR}/train_maze_17_continuous.parquet"
    VAL_DATA="${VAL_PARQUET_DIR}/test_maze_17_continuous.parquet"
    REWARD_LABEL="continuous_ub"
    REWARD_HPARAM_TAG="_ub${CONTINUOUS_UB}"
    REWARD_ARGS=(
      custom_reward_function.path="${REWARD_FN_CONTINUOUS_UB}"
      custom_reward_function.name=compute_score
      +custom_reward_function.reward_kwargs.UB="${CONTINUOUS_UB}"
    )
    ;;
  continuous_ratio|continuous_lstar_over_l)
    TRAIN_DATA="${TRAIN_PARQUET_DIR}/train_maze_17_continuous.parquet"
    VAL_DATA="${VAL_PARQUET_DIR}/test_maze_17_continuous.parquet"
    REWARD_LABEL="continuous_ratio"
    REWARD_HPARAM_TAG=""
    REWARD_ARGS=(
      custom_reward_function.path="${REWARD_FN_CONTINUOUS_RATIO}"
      custom_reward_function.name=compute_score
    )
    ;;
  continuous_2stage)
    TRAIN_DATA="${TRAIN_PARQUET_DIR}/train_maze_17_continuous.parquet"
    VAL_DATA="${VAL_PARQUET_DIR}/test_maze_17_continuous.parquet"
    REWARD_LABEL="continuous_2stage"
    REWARD_HPARAM_TAG="_partial${CONTINUOUS_2STAGE_PARTIAL}"
    REWARD_ARGS=(
      custom_reward_function.path="${REWARD_FN_CONTINUOUS_2STAGE}"
      custom_reward_function.name=compute_score
      +custom_reward_function.reward_kwargs.partial_reward="${CONTINUOUS_2STAGE_PARTIAL}"
    )
    ;;
  binary_shortest)
    TRAIN_DATA="${TRAIN_PARQUET_DIR}/train_maze_17_continuous.parquet"
    VAL_DATA="${VAL_PARQUET_DIR}/test_maze_17_continuous.parquet"
    REWARD_LABEL="binary_shortest"
    REWARD_HPARAM_TAG=""
    REWARD_ARGS=(
      custom_reward_function.path="${REWARD_FN_BINARY_SHORTEST}"
      custom_reward_function.name=compute_score
    )
    ;;
  composite_v1|composite_lstar_dist)
    TRAIN_DATA="${TRAIN_PARQUET_DIR}/train_maze_17_continuous.parquet"
    VAL_DATA="${VAL_PARQUET_DIR}/test_maze_17_continuous.parquet"
    REWARD_LABEL="composite_v1"
    REWARD_HPARAM_TAG=""
    REWARD_ARGS=(
      custom_reward_function.path="${REWARD_FN_COMPOSITE_V1}"
      custom_reward_function.name=compute_score
    )
    ;;
  composite_v2)
    TRAIN_DATA="${TRAIN_PARQUET_DIR}/train_maze_17_continuous.parquet"
    VAL_DATA="${VAL_PARQUET_DIR}/test_maze_17_continuous.parquet"
    REWARD_LABEL="composite_v2"
    REWARD_HPARAM_TAG=""
    REWARD_ARGS=(
      custom_reward_function.path="${REWARD_FN_COMPOSITE_V2}"
      custom_reward_function.name=compute_score
    )
    ;;
  binary)
    TRAIN_DATA="${TRAIN_PARQUET_DIR}/train.parquet"
    VAL_DATA="${VAL_PARQUET_DIR}/test.parquet"
    REWARD_LABEL="binary"
    REWARD_HPARAM_TAG=""
    REWARD_ARGS=()
    ;;
  *)
    echo "Unknown --reward ${REWARD_TYPE}" >&2
    exit 1
    ;;
esac

case "$REWARD_TRANSFORM" in
  raw) TRANSFORM_TAG="" ;;
  cdf) TRANSFORM_TAG="_cdf" ;;
  *) echo "--reward-transform must be raw or cdf, got ${REWARD_TRANSFORM}" >&2; exit 1 ;;
esac

if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$VAL_DATA" ]; then
  echo "Missing parquet files." >&2
  echo "  train: ${TRAIN_DATA}" >&2
  echo "  val:   ${VAL_DATA}" >&2
  echo "Build them first:  bash scripts/download_dataset.sh && bash scripts/prepare_dataset.sh" >&2
  exit 1
fi

if [ "$VAL_PROMPTS" != "0" ]; then
  python3 - "$VAL_DATA" "$VAL_PROMPTS" <<'PY'
import sys
import pyarrow.parquet as pq

path, expected = sys.argv[1], int(sys.argv[2])
rows = pq.ParquetFile(path).metadata.num_rows
if rows != expected:
    raise SystemExit(f"{path} has {rows} rows, expected {expected}. Use --val-prompts 0 to bypass.")
PY
fi

if [ -n "$MODEL_BASE_DIR_ARG" ]; then
  MODEL_PATH="${MODEL_BASE_DIR_ARG%/}/ckpt-${CKPT_STEP}"
elif [ -n "${MODEL_BASE_DIR:-}" ]; then
  MODEL_PATH="${MODEL_BASE_DIR%/}/ckpt-${CKPT_STEP}"
else
  MODEL_PATH="${HF_CACHE_DIR%/}/ckpt-${CKPT_STEP}"
  if [ ! -f "${MODEL_PATH}/config.json" ]; then
    if [ "$DRY_RUN" = "1" ]; then
      echo "[dry-run] would download ckpt-${CKPT_STEP} from ${HF_REPO} to ${HF_CACHE_DIR}"
    else
      echo "[init] downloading ckpt-${CKPT_STEP} from ${HF_REPO} -> ${HF_CACHE_DIR}"
      hf download "${HF_REPO}" --include "ckpt-${CKPT_STEP}/*" --local-dir "${HF_CACHE_DIR}"
    fi
  fi
fi

if [ "$DRY_RUN" != "1" ] && [ ! -d "$MODEL_PATH" ]; then
  echo "Missing model checkpoint: $MODEL_PATH" >&2
  exit 1
fi

# MANDATORY preflight. Both gates exist because the failure they catch is
# SILENT: the model emits well-formed maze paths, the curves move, and it
# reaches the goal 0% of the time.
if [ "$DRY_RUN" != "1" ]; then
  python3 scripts/check_transformers.py || exit 1
  python3 scripts/checkpoint_doctor.py "$MODEL_PATH" \
    || { echo "[FATAL] ${MODEL_PATH} failed checkpoint repair/verify; refusing to train on a broken model" >&2; exit 1; }
fi

PROJECT_NAME="${WANDB_PROJECT:-tailrl-text-maze}"
CHECKPOINT_DIR="${TAILRL_MAZE_RUN_DIR}/checkpoints"
GPU_TAG="${GPU_IDS//,/_}"
FIRST_GPU="${GPU_IDS%%,*}"
NUM_GPUS=$(awk -F',' '{print NF}' <<< "$GPU_IDS")

if [ "$PASS_RATE_PCT" = "custom" ]; then
  INIT_TAG="ckpt${CKPT_STEP}"
else
  INIT_TAG="${PASS_RATE_TAG}_ckpt${CKPT_STEP}"
fi
EXPERIMENT_NAME="textmaze_${SUITE}_${METHOD_TAG}_${REWARD_LABEL}${REWARD_HPARAM_TAG}${TRANSFORM_TAG}_${INIT_TAG}_bs${BATCH_SIZE}_G${N_ROLLOUTS}_${NUM_GPUS}gpu_${GPU_TAG}_seed${SEED}"

echo "[config] suite=${SUITE} method=${METHOD} adv_estimator=${ADVANTAGE_ESTIMATOR}"
echo "[config] pass_rate=${PASS_RATE_TAG} target_pct=${PASS_RATE_PCT} ckpt=${CKPT_STEP}"
echo "[config] reward=${REWARD_LABEL}${REWARD_HPARAM_TAG} transform=${REWARD_TRANSFORM} seed=${SEED} G=${N_ROLLOUTS} pass_k=${PASS_K}"
echo "[config] gpus=${GPU_IDS} train_parquet_dir=${TRAIN_PARQUET_DIR} val_parquet_dir=${VAL_PARQUET_DIR} val_n=${VAL_N}"
echo "[config] experiment=${EXPERIMENT_NAME}"

if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export PYTHONHASHSEED="${SEED}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
# Hydra and W&B default to writing under the CWD, which here is the source
# tree. Send both to the run directory instead.
export WANDB_DIR="${TAILRL_MAZE_RUN_DIR}"
# Group = full config minus the seed, so seed-replicas share one W&B group.
export WANDB_RUN_GROUP="textmaze_${SUITE}_${METHOD_TAG}_${REWARD_LABEL}${REWARD_HPARAM_TAG}${TRANSFORM_TAG}_${PASS_RATE_TAG}_bs${BATCH_SIZE}_G${N_ROLLOUTS}"
export NCCL_TIMEOUT=18000
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# SKIP_CONDA_ACTIVATE=1: containerized runs (apptainer/docker) already have the
# full python env on PATH and no conda.
if [ "${SKIP_CONDA_ACTIVATE:-0}" != "1" ] && [ -n "$CONDA_ENV_NAME" ]; then
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
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found; use --conda-env only after conda is available in this shell." >&2
    exit 1
  fi
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

# Port-band index defaults to the first visible GPU id (interactive whole-node
# usage). Under SLURM cgroups every job sees CUDA id 0, so sbatch wrappers must
# export RAY_PORT_INDEX (first *physical* GPU id from SLURM_JOB_GPUS) to keep
# concurrent jobs on the same node in disjoint port bands.
RAY_PORT_INDEX="${RAY_PORT_INDEX:-$FIRST_GPU}"
# Fixed service ports live at 7000/7x00 (2026-07-15). They MUST be below the OS
# ephemeral range (gpu2: /proc/sys/net/ipv4/ip_local_port_range = 32768-60999):
# the old 41000/42x00 ports were INSIDE that range, so a co-tenant Ray job could
# grab another band's fixed port (e.g. its runtime-env-agent port 42500+idx) as
# an ephemeral SOURCE port for a worker connection -> the band's next job can't
# bind its agent -> "runtime env agent connection refused" -> raylet dies ~57s in
# (the GPU-4/port-42504 crashloop). Ports <32768 are never auto-assigned as
# ephemeral, so they can't be squatted. 7000-7907 verified free on gpu2 and
# clear of co-tenants (10002-19999 = default Ray workers, 20000-40000 = OpenRLHF).
RAY_PORT=$((7000 + RAY_PORT_INDEX))
RAY_CLIENT_PORT=$((7100 + RAY_PORT_INDEX))
RAY_NODE_MANAGER_PORT=$((7200 + RAY_PORT_INDEX))
RAY_OBJECT_MANAGER_PORT=$((7300 + RAY_PORT_INDEX))
RAY_RUNTIME_ENV_AGENT_PORT=$((7500 + RAY_PORT_INDEX))
RAY_METRICS_PORT=$((7600 + RAY_PORT_INDEX))
RAY_DASHBOARD_PORT=$((7700 + RAY_PORT_INDEX))
RAY_DASHBOARD_AGENT_HTTP_PORT=$((7800 + RAY_PORT_INDEX))
RAY_DASHBOARD_AGENT_GRPC_PORT=$((7900 + RAY_PORT_INDEX))
RAY_TEMP_DIR="/tmp/ray_textmaze_p${RAY_PORT_INDEX}_gpu${GPU_TAG}"
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

echo "[GPU ${GPU_IDS}] Cleanup stale Ray for temp=${RAY_TEMP_DIR}, port=${RAY_PORT}"
for _ in 1 2; do
  kill_by_pattern "${RAY_TEMP_DIR}/"
  kill_by_pattern "gcs-address=[^ ]*:${RAY_PORT}"
  kill_by_pattern "gcs_server_port=${RAY_PORT}"
  kill_by_pattern "--port=${RAY_PORT}"
  sleep 1
done
rm -rf "$RAY_TEMP_DIR"
mkdir -p "$RAY_TEMP_DIR"

for port in \
  "$RAY_PORT" \
  "$RAY_CLIENT_PORT" \
  "$RAY_NODE_MANAGER_PORT" \
  "$RAY_OBJECT_MANAGER_PORT" \
  "$RAY_RUNTIME_ENV_AGENT_PORT" \
  "$RAY_METRICS_PORT" \
  "$RAY_DASHBOARD_PORT" \
  "$RAY_DASHBOARD_AGENT_HTTP_PORT" \
  "$RAY_DASHBOARD_AGENT_GRPC_PORT"; do
  for _ in $(seq 1 60); do
    if ! ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":${port}$"; then
      break
    fi
    sleep 1
  done
  if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":${port}$"; then
    echo "[GPU ${GPU_IDS}] Port ${port} still busy after 60s; refusing to start Ray (another job holds this band?)" >&2
    exit 1
  fi
done

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
mkdir -p "${CHECKPOINT_DIR}/logs"

echo "[GPU ${GPU_IDS}] Starting ${EXPERIMENT_NAME}"

# Extra generation stop tokens (e.g. maze DONE=7). Opt-in; empty -> no override.
EOS_ARGS=()
if [ -n "${EXTRA_EOS_TOKEN_IDS}" ]; then
  EOS_ARGS+=("+actor_rollout_ref.rollout.extra_eos_token_ids=[${EXTRA_EOS_TOKEN_IDS}]")
fi

# Re-eval (val-only) overrides. val_only=True -> load latest ckpt (resume_mode=auto),
# run one _validate(), log at global_steps, exit. wandb_id_to_resume appends the eval
# into the existing run; resume_global_steps pins the logged step (e.g. 5000).
VAL_ONLY_ARGS=()
VAL_OUT=""
if [ "${VAL_ONLY}" = "1" ]; then
  # console-only: verl's own wandb resume exits uncleanly and never syncs, so we let
  # reeval_and_log.py push the parsed metrics into the existing run id afterwards.
  VAL_ONLY_ARGS+=("trainer.val_only=True" "trainer.logger=[console]")
  VAL_OUT="$(mktemp /tmp/valout_${SLURM_JOB_ID:-$$}_XXXX.log)"
fi

set +e
{ timeout --signal=SIGTERM "${TIMEOUT_SECONDS}" \
python3 -m verl.trainer.main_ppo \
  hydra.run.dir="${TAILRL_MAZE_RUN_DIR}/hydra/${EXPERIMENT_NAME}" \
  ray_init.ray_dir="${RAY_TEMP_DIR}" \
  algorithm.adv_estimator="${ADVANTAGE_ESTIMATOR}" \
  algorithm.use_kl_in_reward=False \
  algorithm.pass_k="${PASS_K}" \
  algorithm.reward_transform="${REWARD_TRANSFORM}" \
  +data.seed="${SEED}" \
  data.train_files="${TRAIN_DATA}" \
  data.val_files="${VAL_DATA}" \
  data.train_batch_size="${BATCH_SIZE}" \
  data.max_prompt_length=320 \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.apply_chat_template=False \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=1e-4 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.dtype=float16 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH:-${BATCH_SIZE}}" \
  actor_rollout_ref.rollout.name=hf \
  +actor_rollout_ref.rollout.micro_batch_size="${ROLLOUT_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.rollout.dtype=float16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8192 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8192 \
  actor_rollout_ref.rollout.n="${N_ROLLOUTS}" \
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}" \
  "${EOS_ARGS[@]}" \
  actor_rollout_ref.rollout.val_kwargs.gen_batch_size="${VAL_GEN_BATCH_SIZE}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  reward_model.reward_manager=prime \
  +reward_model.reward_kwargs.num_processes="${REWARD_NUM_PROCESSES:-4}" \
  +reward_model.reward_kwargs.chunksize=64 \
  "${REWARD_ARGS[@]}" \
  algorithm.kl_ctrl.kl_coef=0.0 \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.logger=['console','wandb'] \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node="${NUM_GPUS}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.max_actor_ckpt_to_keep=3 \
  trainer.default_local_dir="${CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
  "${VAL_ONLY_ARGS[@]}" \
  trainer.total_epochs=100 ; } 2>&1 | { [ -n "${VAL_OUT}" ] && tee "${VAL_OUT}" || cat ; }
EXIT_CODE=${PIPESTATUS[0]}
set -e

# Re-eval: parse verl's val_metrics and push them into the existing wandb run (clean finish).
if [ "${VAL_ONLY}" = "1" ] && [ -n "${WANDB_RESUME_ID}" ] && [ -f "${VAL_OUT}" ]; then
  echo "[reeval] logging step-${RESUME_GLOBAL_STEPS:-5000} metrics to wandb run ${WANDB_RESUME_ID}"
  if WANDB_PROJECT="${PROJECT_NAME}" WANDB_ENTITY="${WANDB_ENTITY}" \
    python3 "${EXP_ROOT}/scripts/reeval_and_log.py" \
    "${VAL_OUT}" "${WANDB_RESUME_ID}" "${RESUME_GLOBAL_STEPS:-5000}"; then
    mkdir -p "${TAILRL_LOG_DIR}/reeval_state"
    touch "${TAILRL_LOG_DIR}/reeval_state/${EXPERIMENT_NAME}.done"
  else
    echo "[reeval] log step FAILED"
  fi
  rm -f "${VAL_OUT}"
fi

for _ in 1 2; do
  kill_by_pattern "${RAY_TEMP_DIR}/"
  kill_by_pattern "gcs-address=[^ ]*:${RAY_PORT}"
  kill_by_pattern "gcs_server_port=${RAY_PORT}"
  kill_by_pattern "--port=${RAY_PORT}"
  sleep 1
done
rm -rf "$RAY_TEMP_DIR" 2>/dev/null || true

echo "[GPU ${GPU_IDS}] Finished ${EXPERIMENT_NAME} exit=${EXIT_CODE}"
exit "${EXIT_CODE}"
