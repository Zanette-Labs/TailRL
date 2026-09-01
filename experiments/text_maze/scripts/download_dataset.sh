#!/bin/bash
# Download the maze corpus into $TAILRL_MAZE_DATA_DIR (override with $1).
set -euo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/env.sh"

REPO_ID="${REPO_ID:-${TAILRL_MAZE_DATASET_REPO}}"
OUT_DIR="${1:-${TAILRL_MAZE_DATA_DIR}}"

echo "[data] ${REPO_ID} -> ${OUT_DIR}  (~3.7 GB)"

mkdir -p "$OUT_DIR/sft_data"

hf download "$REPO_ID" main_1.3M.jsonl --repo-type dataset --local-dir "$OUT_DIR"
hf download "$REPO_ID" main_1.3M.jsonl.meta.json --repo-type dataset --local-dir "$OUT_DIR"
hf download "$REPO_ID" train_random.json --repo-type dataset --local-dir "$OUT_DIR/sft_data"
hf download "$REPO_ID" train_shortest.json --repo-type dataset --local-dir "$OUT_DIR/sft_data"
hf download "$REPO_ID" test.json --repo-type dataset --local-dir "$OUT_DIR/sft_data"
