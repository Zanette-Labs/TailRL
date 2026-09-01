#!/bin/bash
# MEASURE THE INITIALIZATIONS — pass@1 of each SFT checkpoint before any RL.
#
# This is what puts the x-axis on the initialization-sweep figure: each of the
# seven checkpoints is labelled by the fraction of single rollouts that solve a
# held-out maze along a shortest path, and those rates span 0.012% to 0.83%.
#
# Rates that small need a lot of samples. The paper's figures use 1000 rollouts
# on each of the 1000 held-out mazes (10^6 draws per checkpoint), which resolves
# a rate of 1e-4 to roughly +/- 10% relative. Fewer samples will not separate
# ckpt-2450 from zero.
#
# Usage:
#   bash scripts/reproduce/sft_pass1_ladder.sh [--dry-run]
#   VAL_N=100 bash scripts/reproduce/sft_pass1_ladder.sh     # quick, low resolution
set -uo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/_common.sh"

LADDER_VAL_N="${LADDER_VAL_N:-1000}"
OUT_CSV="${OUT_CSV:-${TAILRL_MAZE_RUN_DIR}/sft_pass1_ladder.csv}"
LADDER_LOG_DIR="${LADDER_LOG_DIR:-${TAILRL_LOG_DIR}/sft_pass1}"
mkdir -p "$LADDER_LOG_DIR" "$(dirname "$OUT_CSV")"

for ckpt in $CKPTS; do
  log="${LADDER_LOG_DIR}/sft_pass1_ckpt${ckpt}.log"
  if [ -s "$log" ]; then echo "[have] ckpt-${ckpt}"; continue; fi
  if [ "$DRY_RUN" = "1" ]; then
    echo "bash scripts/eval_sft_pass1_standard.sh ${ckpt} ${GPU_IDS} ${LADDER_VAL_N}"
    continue
  fi
  echo "[eval] SFT ckpt-${ckpt} at ${LADDER_VAL_N} rollouts x 1000 mazes"
  bash scripts/eval_sft_pass1_standard.sh "$ckpt" "$GPU_IDS" "$LADDER_VAL_N" \
    "${TAILRL_MAZE_DATA_DIR}/eval1000" 2>&1 | tee "$log"
done

if [ "$DRY_RUN" = "0" ]; then
  python3 scripts/parse_sft_pass1_eval.py "$LADDER_LOG_DIR" > "$OUT_CSV" 2>/dev/null \
    && echo "[done] wrote ${OUT_CSV}" \
    || echo "[done] logs in ${LADDER_LOG_DIR}; parse them with scripts/parse_sft_pass1_eval.py"
fi
