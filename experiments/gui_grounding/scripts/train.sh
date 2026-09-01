#!/usr/bin/env bash
# Train ONE arm of the estimator comparison.
#
#   bash scripts/train.sh --method tailrl            # or grpo, rloo
#   bash scripts/train.sh --method grpo --model 7b --seed 1 --gpus 4
#   bash scripts/train.sh --method tailrl --dry-run  # print the command, run nothing
#
# The three arms differ in EXACTLY ONE flag, algorithm.adv_estimator. Everything else -- data order
# (data.seed), reward, optimizer, schedule -- is identical, which is what makes the comparison a
# comparison. Do not vary the seed or the learning rate between arms.
#
# Resumable: the checkpoint path is a pure function of (method, model, seed), and the trainer
# resumes from the last full-state checkpoint, so re-running the same command continues an
# interrupted run rather than restarting it.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
cd "$TAILRL_GUI_ROOT"

METHOD=tailrl; MODEL=3b; SEED=1; GPUS=4; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --method) METHOD="$2"; shift 2 ;;
    --model)  MODEL="$2";  shift 2 ;;
    --seed)   SEED="$2";   shift 2 ;;
    --gpus)   GPUS="$2";   shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done
case "$METHOD" in tailrl|grpo|rloo) ;; *) echo "--method must be tailrl|grpo|rloo" >&2; exit 1 ;; esac
case "$MODEL"  in 3b|7b) ;;           *) echo "--model must be 3b|7b" >&2; exit 1 ;; esac

CONFIG="examples/gui_grounding/qwen2_5_vl_${MODEL}.yaml"
# The config's model_path default tracks the 3B model; keep BASE_MODEL and --model consistent
# unless the caller deliberately set BASE_MODEL to a local directory.
if [ "$MODEL" = 7b ] && [ "${BASE_MODEL}" = "Qwen/Qwen2.5-VL-3B-Instruct" ]; then
  export BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
fi
EXP="qwen25vl${MODEL}_gta1_segui_${METHOD}_s${SEED}"

# Checked only for a real run, so --dry-run stays useful before the ~54 GB download.
if [ "$DRY" = 0 ]; then
  for f in "$GUI_DATA_DIR/gta1_train.parquet" "$GUI_DATA_DIR/screenspot_pro.parquet"; do
    [ -f "$f" ] || { echo "missing $f -- run: bash scripts/prepare_data.sh" >&2; exit 1; }
  done
fi

CMD=("${RUN[@]}" python3 -m verl.trainer.main
     "config=$CONFIG"
     "algorithm.adv_estimator=$METHOD"
     "trainer.experiment_name=$EXP"
     "trainer.save_checkpoint_path=$CKPT_ROOT/$EXP"
     "trainer.n_gpus_per_node=$GPUS"
     "data.seed=$SEED")

echo "arm=$METHOD model=$MODEL seed=$SEED gpus=$GPUS"
echo "ckpt: $CKPT_ROOT/$EXP"
printf '  %q' "${CMD[@]}"; echo
[ "$DRY" = 1 ] && exit 0
exec "${CMD[@]}"
