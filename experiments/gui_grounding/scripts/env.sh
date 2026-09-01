# Paths and defaults for the GUI-grounding experiment. Sourced by every script here.
#
# Nothing below is a hardcoded absolute path: each value is an override-able default rooted at the
# experiment directory, so a clean clone runs with zero configuration. Export any of them (or put
# them in ../../.env) to point at scratch storage instead -- which you will want to, since the
# images alone are ~16 GB and the raw downloads another ~38 GB.
#
# shellcheck shell=bash

# Experiment root = the parent of scripts/. Resolved from this file, not from $PWD, so the scripts
# work when invoked from anywhere.
TAILRL_GUI_ROOT="${TAILRL_GUI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TAILRL_GUI_ROOT

# --- data ------------------------------------------------------------------------------------
# GUI_DATA_DIR   holds the two parquets the trainer reads (gta1_train, screenspot_pro).
# GUI_IMAGE_DIR  becomes data.image_dir; holds gta1_images/ and screenspot_pro/.
# GUI_RAW_DIR    is only needed while building; it can be deleted afterwards EXCEPT for the
#                ScreenSpot-Pro images, which are symlinked rather than copied.
export GUI_DATA_DIR="${GUI_DATA_DIR:-$TAILRL_GUI_ROOT/data/gui}"
export GUI_IMAGE_DIR="${GUI_IMAGE_DIR:-$TAILRL_GUI_ROOT/gui_images}"
export GUI_RAW_DIR="${GUI_RAW_DIR:-$TAILRL_GUI_ROOT/raw}"

# --- model -----------------------------------------------------------------------------------
# A HuggingFace id works directly. Point it at a local directory to avoid re-downloading, or to
# evaluate a trained checkpoint (an actor_only export).
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"

# --- outputs ---------------------------------------------------------------------------------
export CKPT_ROOT="${CKPT_ROOT:-$TAILRL_GUI_ROOT/checkpoints}"
export EVAL_OUT="${EVAL_OUT:-$TAILRL_GUI_ROOT/eval_out}"

# --- runtime ---------------------------------------------------------------------------------
# `import verl` must resolve to the vendored fork in this directory, not to a pip-installed verl.
export PYTHONPATH="$TAILRL_GUI_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Batch shapes vary widely here (GTA1 image tokens span 646..9,000), so the caching allocator sees a
# constantly changing block-size distribution over tens of thousands of steps. Without expandable
# segments, reserved memory drifts upward as unusable free blocks accumulate.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# W&B is optional: the configs log to a local file as well. Credentials come from `wandb login`
# (~/.netrc). No API key is read from, or belongs in, this repository.
export WANDB_PROJECT="${WANDB_PROJECT:-tailrl-gui-grounding}"
# Uncomment to train with no W&B account at all:
# export WANDB_MODE=offline

# Optional container. If TAILRL_GUI_SIF points at an Apptainer image, RUN wraps every command in it;
# otherwise commands run directly in your current environment.
if [ -n "${TAILRL_GUI_SIF:-}" ]; then
  RUN=(apptainer exec --nv --cleanenv --pwd "$TAILRL_GUI_ROOT"
       ${TAILRL_GUI_BIND:+--bind "$TAILRL_GUI_BIND"}
       --env PYTHONPATH="$TAILRL_GUI_ROOT"
       --env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
       --env TOKENIZERS_PARALLELISM=false
       "$TAILRL_GUI_SIF")
else
  RUN=()
fi
export TAILRL_GUI_ROOT
