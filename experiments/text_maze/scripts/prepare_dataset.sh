#!/bin/bash
set -euo pipefail

# shellcheck disable=SC1091
. "$(dirname "$0")/env.sh"
cd "$EXP_ROOT"

JSONL="${1:-${TAILRL_MAZE_DATA_DIR}/main_1.3M.jsonl}"
SFT_DIR="${2:-${TAILRL_MAZE_DATA_DIR}/sft_data}"
PARQUET_DIR="${3:-${TAILRL_MAZE_DATA_DIR}/main_parquet}"
TEST_PROMPTS="${TEST_PROMPTS:-256}"
SEED="${SEED:-0}"

mkdir -p "$SFT_DIR" "$PARQUET_DIR"

python src/to_sft_json.py \
  --in_path "$JSONL" \
  --out_train "$SFT_DIR/train_random.json" \
  --out_test "$SFT_DIR/test.json" \
  --test_prompts "$TEST_PROMPTS" \
  --mode random \
  --seed "$SEED"

python src/to_sft_json.py \
  --in_path "$JSONL" \
  --out_train "$SFT_DIR/train_shortest.json" \
  --out_test "$SFT_DIR/test.json" \
  --test_prompts "$TEST_PROMPTS" \
  --mode shortest \
  --seed "$SEED"

python src/to_rl_parquet.py \
  --in_path "$JSONL" \
  --out_dir "$PARQUET_DIR" \
  --test_prompts "$TEST_PROMPTS" \
  --seed "$SEED" \
  --variants maze_17_continuous

python src/to_rl_parquet.py \
  --in_path "$JSONL" \
  --out_dir "$PARQUET_DIR" \
  --test_prompts "$TEST_PROMPTS" \
  --seed "$SEED" \
  --variants maze_17 \
  --canonical
