#!/bin/bash
# Pre-flight GATE. Prove the environment and the checkpoints are sound BEFORE
# committing to a multi-day sweep.
#
#   1. transformers version gate + checkpoint-repair tests  (fast, CPU only)
#   2. generation smoke test          (needs a GPU): loads a real SFT checkpoint
#                                     and asserts it actually reaches goals.
#
# Step 2 is the one that matters. A checkpoint broken by the transformers
# forward-incompatibility (see checkpoint_doctor.py) still emits well-formed
# maze paths and still moves the training curves — it just reaches the goal 0%
# of the time. Nothing else in the pipeline warns you.
#
# Usage (on a node with a GPU):
#   bash scripts/verify_setup.sh [ckpt_step]      # default 3550
#
# Exit 0 = safe to train.
set -uo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/env.sh"
cd "$EXP_ROOT" || exit 1

CKPT_STEP="${1:-3550}"

echo "=============================================================="
echo " 1/2  environment + checkpoint-repair tests (fast, no GPU)"
echo "=============================================================="
python3 scripts/check_transformers.py || exit 1
python3 scripts/tests/test_transformers_version.py \
  || { echo "[verify] transformers version tests FAILED"; exit 1; }
python3 scripts/tests/test_checkpoint_doctor.py \
  || { echo "[verify] doctor tests FAILED"; exit 1; }

echo "=============================================================="
echo " 2/2  generation smoke test (GPU) on ckpt-${CKPT_STEP}"
echo "=============================================================="
SMOKE="python3 scripts/tests/smoke_goal_reaching.py ${HF_CACHE_DIR}/ckpt-${CKPT_STEP} 50 32"

if [ -n "${TAILRL_MAZE_SIF}" ]; then
  [ -f "$TAILRL_MAZE_SIF" ] || { echo "[verify] container image missing: $TAILRL_MAZE_SIF"; exit 1; }
  BIND_ARGS=()
  [ -n "${TAILRL_APPTAINER_BIND}" ] && BIND_ARGS=(--bind "${TAILRL_APPTAINER_BIND}")
  "$TAILRL_APPTAINER_BIN" exec --nv "${BIND_ARGS[@]}" "$TAILRL_MAZE_SIF" bash -c \
    "export PYTHONPATH='${PYTHONPATH}' PYTHONNOUSERSITE=1 HF_CACHE_DIR='${HF_CACHE_DIR}'; cd '${EXP_ROOT}'; ${SMOKE}"
else
  $SMOKE
fi
# shellcheck disable=SC2181
[ $? -eq 0 ] || {
  echo "[verify] SMOKE FAILED — do NOT start the sweep."
  echo "[verify] The model is not reaching goals. Every downstream number would be"
  echo "[verify] plausible and wrong. Re-run scripts/setup_checkpoints.sh and check"
  echo "[verify] your transformers version."
  exit 1
}

echo "=============================================================="
echo " VERIFY OK — environment + checkpoint sound. Safe to run the sweep."
echo "=============================================================="
