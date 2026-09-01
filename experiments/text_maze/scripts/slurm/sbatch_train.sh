#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@600
#
# Generic single-GPU SLURM wrapper around scripts/train.sh. Everything
# site-specific comes from TAILRL_SLURM_* (see scripts/env.sh), so submit with:
#
#   TAILRL_SLURM_PARTITION=gpu TAILRL_SLURM_GRES=gpu:a6000:1 \
#     sbatch experiments/text_maze/scripts/slurm/sbatch_train.sh \
#       --method tailrl --ckpt-step 3000 --seed 0 --n-rollouts 16
#
# Arguments are forwarded verbatim to scripts/train.sh; --gpu-ids is added here.
#
# What this adds over running train.sh directly:
#   * USR1 ~10 min before the time limit -> `scontrol requeue`. Combined with
#     verl's resume_mode=auto (model + optimizer + RNG + LR schedule restored
#     from the newest global_step_N), preemption and walltime are lossless.
#   * A duplicate-run lock. Two jobs with identical arguments derive the same
#     experiment name and would interleave into one checkpoint directory and one
#     W&B run. The second aborts instead.
#   * A per-job Ray port band derived from the first PHYSICAL GPU id, so
#     concurrent jobs sharing a node never collide on Ray's fixed ports.
set -uo pipefail

EXP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
. "${EXP_ROOT}/scripts/env.sh"
cd "$EXP_ROOT" || exit 1

# Ray port band: inside a cgroup every job sees CUDA id 0, so the visible ids are
# useless for disambiguation. The first *physical* id is unique per job. Refusing
# to guess matters: a wrong guess would make the launcher's stale-Ray cleanup
# SIGKILL a co-tenant's Ray.
GPUS_RAW="${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-}}"
if [ -z "$GPUS_RAW" ] && [ -z "${RAY_PORT_INDEX:-}" ]; then
  echo "[sbatch] SLURM_JOB_GPUS/SLURM_STEP_GPUS unset; export RAY_PORT_INDEX to run outside an allocation" >&2
  exit 1
fi
export RAY_PORT_INDEX="${RAY_PORT_INDEX:-${GPUS_RAW%%,*}}"

NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"
GPU_IDS="$(seq -s, 0 $((NUM_GPUS - 1)))"
export SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_${SLURM_JOB_ID:-$$}}"
mkdir -p "$TRITON_CACHE_DIR"

# Duplicate-run guard, keyed on the exact argument set. A requeue keeps its
# SLURM_JOB_ID and so passes; a chained dependent takes over once the head job
# has terminated; a *different live* job with this config aborts us.
LOCK_DIR="${TAILRL_MAZE_RUN_DIR}/.job_locks"; mkdir -p "$LOCK_DIR"
LOCK_FILE="${LOCK_DIR}/train_$(printf '%s|' "$@" | md5sum | awk '{print $1}').jobid"
if [ -f "$LOCK_FILE" ]; then
  OTHER="$(cat "$LOCK_FILE" 2>/dev/null || true)"
  if [ -n "$OTHER" ] && [ "$OTHER" != "${SLURM_JOB_ID:-}" ] \
     && squeue -h -j "$OTHER" -t PENDING,RUNNING,SUSPENDED 2>/dev/null | grep -q .; then
    echo "[sbatch] job ${OTHER} already runs this exact config (${LOCK_FILE}); aborting duplicate" >&2
    exit 1
  fi
fi
echo "${SLURM_JOB_ID:-manual_$$}" > "$LOCK_FILE"

on_usr1() {
  echo "[sbatch] USR1: near the time limit; requeueing ${SLURM_JOB_ID} (resume is automatic)"
  scontrol requeue "${SLURM_JOB_ID}" \
    || echo "[sbatch] requeue failed; resubmit manually — the run resumes from its last checkpoint"
}
trap on_usr1 USR1

echo "[sbatch] job=${SLURM_JOB_ID:-?} host=$(hostname) gpus=${GPUS_RAW:-?} port_index=${RAY_PORT_INDEX} project=${WANDB_PROJECT} args: $*"

# Training must run in the BACKGROUND with an explicit `wait`: exec'ing it would
# block the shell and the USR1 trap would never fire, turning every walltime hit
# into a lost run instead of a requeue.
if [ -n "${TAILRL_MAZE_SIF}" ]; then
  BIND_ARGS=(); [ -n "${TAILRL_APPTAINER_BIND}" ] && BIND_ARGS=(--bind "${TAILRL_APPTAINER_BIND}")
  "$TAILRL_APPTAINER_BIN" exec --nv "${BIND_ARGS[@]}" "$TAILRL_MAZE_SIF" \
    bash scripts/train.sh "$@" --gpu-ids "$GPU_IDS" &
else
  bash scripts/train.sh "$@" --gpu-ids "$GPU_IDS" &
fi
CHILD=$!
wait "$CHILD"; EXIT_CODE=$?
# A trapped signal interrupts `wait` with 128+signo while the child is still
# alive; keep waiting so EXIT_CODE ends up as the child's real status.
while [ "$EXIT_CODE" -gt 128 ] && kill -0 "$CHILD" 2>/dev/null; do
  wait "$CHILD"; EXIT_CODE=$?
done
echo "[sbatch] finished exit=${EXIT_CODE}"
exit "$EXIT_CODE"
