#!/usr/bin/env bash
# Evaluate ONE checkpoint on ScreenSpot-Pro: unbiased pass@k and best_reward@k.
#
#   bash scripts/eval.sh --ckpt $CKPT_ROOT/<run>/actor_only/global_step_26448/actor --arm tailrl
#   bash scripts/eval.sh --ckpt Qwen/Qwen2.5-VL-3B-Instruct --arm base       # untrained control
#   bash scripts/eval.sh --ckpt <dir> --arm tailrl --n 512 --shards 4        # 4 GPUs, one per shard
#
# --n is the number of samples drawn per item; k must be <= n. n=512 supports k up to 128; the
# headline k=1024 numbers in paper_results/ used n=4096. Cost scales with ITEMS, not with n: the
# prompt is a screenshot (~3,100 tokens) and the response is a coordinate (~13), so prefill
# dominates and extra samples on an already-prefilled image are comparatively cheap.
#
# Sampling protocol is T=0.6 / top_p=0.95 / top_k=-1 and is RECORDED into every results JSON.
# Every arm in a comparison must use identical values.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
cd "$TAILRL_GUI_ROOT"

CKPT=""; ARM=""; N=512; SHARDS=1; KS="1 2 4 8 16 32 64 128"; TOK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2 ;;
    --arm)  ARM="$2";  shift 2 ;;
    --n)    N="$2";    shift 2 ;;
    --shards) SHARDS="$2"; shift 2 ;;
    --ks)   KS="$2";   shift 2 ;;
    --tokenizer) TOK="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done
[ -n "$CKPT" ] || { echo "--ckpt is required" >&2; exit 1; }
[ -n "$ARM" ]  || { echo "--arm is required (tailrl|grpo|rloo|base) -- it names the output subdir" >&2; exit 1; }
# The processor must match the model being evaluated. Pairing a 3B checkpoint with a 7B processor
# mis-sizes every image and the resulting numbers still look entirely sane.
TOK="${TOK:-$CKPT}"

OUT="$EVAL_OUT/$ARM"
mkdir -p "$OUT"

# One process per shard, each pinned to its own GPU. The pin must be exported INSIDE the subshell
# and BEFORE the process starts, so each shard is launched in its own ().
for s in $(seq 0 $((SHARDS-1))); do
  (
    [ "$SHARDS" -gt 1 ] && export CUDA_VISIBLE_DEVICES="$s"
    "${RUN[@]}" python3 scripts/eval_screenspot_pro.py \
        --checkpoint_dir "$CKPT" \
        --tokenizer      "$TOK" \
        --data_dir       "$GUI_DATA_DIR" \
        --image_dir      "$GUI_IMAGE_DIR" \
        --output_dir     "$OUT" \
        --split          screenspot_pro \
        --num_shards "$SHARDS" --shard_id "$s" \
        --bestk_subset 0 --bestk_n "$N" --bestk_ks $KS \
        --dump_samples \
        --gpu_memory_utilization 0.85
  ) &
done
wait

if [ "$SHARDS" -gt 1 ]; then
  "${RUN[@]}" python3 scripts/eval_screenspot_pro.py --merge \
      --checkpoint_dir "$CKPT" --tokenizer "$TOK" \
      --data_dir "$GUI_DATA_DIR" --image_dir "$GUI_IMAGE_DIR" \
      --output_dir "$OUT" --num_shards "$SHARDS" --format_weight 0.5
fi

echo
echo "per-sample records: $OUT/samples.shard*.parquet"
echo "aggregate with:  python3 analysis/dump_bestk_json.py --set 3B:26448:$EVAL_OUT --out results.json"
