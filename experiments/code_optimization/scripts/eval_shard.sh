#!/usr/bin/env bash
# Evaluate ONE contiguous slice of the held-out test set for one checkpoint.
#
#   bash scripts/eval_shard.sh --ckpt <hf_dir> --arm tailrl_step300 --shard 7 --shards 176
#
# The full evaluation is 878 programs x 4096 rollouts x 4 arms, which is far too much
# for one process; this is the unit of work. Shards are CONTIGUOUS in prompt order, so
# every prompt's complete set of 4096 completions lands in exactly one shard. That is
# what makes the per-prompt best@k exact rather than an average of partial maxima.
#
# Idempotent: a finished shard writes shard_<i>of<T>.done.json and is skipped. A
# claim directory keeps two workers off the same shard, which matters when this is
# driven by a SLURM array whose tasks can be requeued.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"

CKPT=""; ARM=""; SHARD=""; SHARDS=176
NCOMP="${NCOMP:-4096}"
# Sampling matches the reference implementation this work builds on, and must be
# identical across arms or the comparison is meaningless.
TEMP="${TEMP:-0.6}"; TOPP="${TOPP:-0.95}"; MAXTOK="${MAXTOK:-4096}"
K_VALUES="${K_VALUES:-1 4 16 64 256 1024}"

usage() { cat <<'EOF'
usage: eval_shard.sh --ckpt DIR --arm NAME --shard I --shards T

  --ckpt DIR    a HuggingFace-format model directory. For a trained arm this is
                <run>/actor_ckpts/step_<N>/huggingface; for the untrained baseline
                just pass the base model id.
  --arm NAME    output subdirectory, by convention <method>_step<N> (e.g. tailrl_step300).
  --shard I     which slice, 0 <= I < T.
  --shards T    how many slices (default 176, about 5 programs each).

  NCOMP=4096  TEMP=0.6  TOPP=0.95  MAXTOK=4096  K_VALUES="1 4 16 64 256 1024"
EOF
}
while [ $# -gt 0 ]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2 ;;
    --arm) ARM="$2"; shift 2 ;;
    --shard) SHARD="$2"; shift 2 ;;
    --shards) SHARDS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
[ -n "${CKPT}" ] && [ -n "${ARM}" ] && [ -n "${SHARD}" ] || { usage; exit 2; }

TEST_PARQUET="${PIE_PARQUET_ROOT}/pie_gem5_test.parquet"
[ -f "${TEST_PARQUET}" ] || { echo "FATAL: missing ${TEST_PARQUET}; run scripts/prepare_dataset.sh" >&2; exit 2; }
OUT_DIR="${CODEOPT_EVAL_DIR}/${ARM}"
mkdir -p "${OUT_DIR}/claims"

# The reward must be configured exactly as it was in training, except that the timed
# case is the BIGGEST usable one rather than a random one: evaluation should be
# deterministic and should exercise the most compute-heavy case, where an algorithmic
# improvement actually shows up.
export PIE_REWARD_IMPL=gem5
export PIE_GEM5_REWARD_KIND=ratio
export PIE_GEM5_REWARD_K="${PIE_GEM5_REWARD_K:-1}"
export PIE_GEM5_CASE_SELECT="${PIE_GEM5_CASE_SELECT:-biggest}"
export PIE_GEM5_REWARD_SAMPLE=group
export PIE_GEM5_BUDGET_MULT=3.0
export PIE_GEM5_REWARD_WORKERS="${PIE_GEM5_REWARD_WORKERS:-$(( $(nproc) - 8 ))}"
export PIE_NAT_TIMEOUT="${PIE_NAT_TIMEOUT:-10}"
export PIE_GEM5_REWARD_NATIVE_BUDGET="${PIE_GEM5_REWARD_NATIVE_BUDGET:-120}"
export PIE_GEM5_REWARD_DEADLINE="${PIE_GEM5_REWARD_DEADLINE:-6000}"

# Optional: dump raw generations for a fixed set of prompts, the same set for every
# arm, so completions can be compared side by side. Created by make_dump_prompts.py.
export BESTEVAL_DUMP_PROMPTS_JSON="${BESTEVAL_DUMP_PROMPTS_JSON:-${CODEOPT_EVAL_DIR}/dump_prompts.json}"
export BESTEVAL_DUMP_N="${BESTEVAL_DUMP_N:-64}"

export TMPDIR="${TMPDIR:-/tmp}/codeopt_eval_${USER}_${SLURM_JOB_ID:-$$}"
export PIE_COMPILE_SCRATCH="${TMPDIR}/scratch"
mkdir -p "${PIE_COMPILE_SCRATCH}"

DONE_F="${OUT_DIR}/shard_${SHARD}of${SHARDS}.done.json"
[ -f "${DONE_F}" ] && { echo "[${ARM} ${SHARD}/${SHARDS}] already done"; exit 0; }
CLAIM="${OUT_DIR}/claims/${SHARD}of${SHARDS}"
mkdir "${CLAIM}" 2>/dev/null || { echo "[${ARM} ${SHARD}/${SHARDS}] claimed by another worker"; exit 0; }
echo "${SLURM_JOB_ID:-$$}" > "${CLAIM}/owner"
# Release the claim unless we actually finished, so a killed worker does not
# permanently block its shard.
trap '[ -f "${DONE_F}" ] || rm -rf "${CLAIM}"' EXIT

echo "=== ${ARM} shard ${SHARD}/${SHARDS} | n=${NCOMP} temp=${TEMP} top_p=${TOPP} | ${CKPT} ==="
python3 -m code_opt.eval.besteval_shard \
  --hf_dir "${CKPT}" --val_parquet "${TEST_PARQUET}" \
  --shard "${SHARD}" --n_shards "${SHARDS}" \
  --n_completions "${NCOMP}" --temperature "${TEMP}" --top_p "${TOPP}" \
  --max_tokens "${MAXTOK}" --tensor_parallel_size 1 \
  --out_dir "${OUT_DIR}" --k_values ${K_VALUES}
