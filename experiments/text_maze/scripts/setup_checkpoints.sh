#!/bin/bash
# One command to get the maze SFT checkpoints ready for the verl container:
# download from Hugging Face, then repair + VERIFY every checkpoint
# (rope_theta + tokenizer_class fixes -- see checkpoint_doctor.py). Idempotent.
#
# Usage:
#   bash scripts/setup_checkpoints.sh              # all checkpoints
#   bash scripts/setup_checkpoints.sh 2450,3000    # only these steps
#
# Env: HF_REPO, HF_CACHE_DIR, HF_BIN (default: hf) -- the first two default
#      via scripts/env.sh, which is the single definition of both. Do not
#      set a different default here: training reads $HF_CACHE_DIR too, and a
#      mismatch downloads the checkpoints somewhere training will not look.
set -euo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/env.sh"
cd "$EXP_ROOT" || exit 1

HF_REPO="${HF_REPO:-${TAILRL_MAZE_CKPT_REPO}}"
HF_BIN="${HF_BIN:-hf}"
STEPS="${1:-}"

if ! command -v "$HF_BIN" >/dev/null 2>&1; then
  echo "[FATAL] '$HF_BIN' CLI not found. Install huggingface_hub or set HF_BIN=/path/to/hf" >&2
  exit 1
fi
mkdir -p "$HF_CACHE_DIR"

echo "[setup] downloading ${HF_REPO} -> ${HF_CACHE_DIR}"
if [ -n "$STEPS" ]; then
  for s in ${STEPS//,/ }; do
    "$HF_BIN" download "$HF_REPO" --include "ckpt-${s}/*" --local-dir "$HF_CACHE_DIR"
  done
else
  "$HF_BIN" download "$HF_REPO" --local-dir "$HF_CACHE_DIR"
fi

echo "[setup] repairing + verifying checkpoints"
python3 scripts/checkpoint_doctor.py --all "$HF_CACHE_DIR"
echo "[setup] done — checkpoints ready and verified in ${HF_CACHE_DIR}"
