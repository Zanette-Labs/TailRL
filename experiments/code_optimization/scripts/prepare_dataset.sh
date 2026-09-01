#!/usr/bin/env bash
# Convert the gem5-timed PIE dataset on the Hugging Face Hub into the parquets verl
# reads. A few minutes, no GPU, ~21 MB out.
#
# The Hub files carry `src_code` plus 51 provenance columns; verl's RLHFDataset wants
# a rendered chat prompt, a `data_source`, and a `reward_model` struct whose
# ground_truth carries the per-case reference ticks. This is that conversion, and it
# is mandatory -- the Hub parquets cannot be handed to the trainer directly.
#
# Writes into $PIE_PARQUET_ROOT:
#   pie_gem5_train.parquet     16,452 source programs   <- training
#   pie_gem5_val_full.parquet     588
#   pie_gem5_val_200.parquet      200 (seeded subsample)
#   pie_gem5_test.parquet         878 held-out programs <- evaluation
#   MANIFEST_GEM5.json            what was built, and with which tokenizer/prompt
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"

PROMPT="${PIE_PROMPT_VARIANT:-minimal}"   # minimal (used by every run in the paper) | p3

if [ -f "${PIE_PARQUET_ROOT}/pie_gem5_train.parquet" ] && [ -f "${PIE_PARQUET_ROOT}/pie_gem5_test.parquet" ]; then
  echo "[data] parquets already built in ${PIE_PARQUET_ROOT}"
  exit 0
fi

mkdir -p "${PIE_PARQUET_ROOT}"
echo "[data] building verl parquets -> ${PIE_PARQUET_ROOT} (prompt=${PROMPT})"
python3 -m code_opt.build_pie_gem5_parquet \
  --output_root "${PIE_PARQUET_ROOT}" \
  --tokenizer "${BASE_MODEL}" \
  --prompt "${PROMPT}" \
  "$@"

echo
echo "  export PIE_PARQUET_ROOT=${PIE_PARQUET_ROOT}"
