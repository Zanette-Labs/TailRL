#!/bin/bash
# Shared prelude for every shell script in this experiment. Source it, do not run it.
#
#   EXP_ROOT   experiments/text_maze — every relative path in this experiment is
#              resolved against it, and it is also what must be on PYTHONPATH so
#              that `import verl` finds the vendored fork.
#
# Everything below has a working default; override any of them in your shell, in
# your job script, or in a `.env` at the TailRL repo root (sourced here if present).

# --- locate the experiment root regardless of where the caller cd'd to ---------
EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EXP_ROOT
REPO_ROOT="$(cd "${EXP_ROOT}/../.." && pwd)"
export REPO_ROOT

# `.env` at the TailRL repo root, if the user made one (gitignored).
# shellcheck disable=SC1091
[ -f "${REPO_ROOT}/.env" ] && . "${REPO_ROOT}/.env"

# --- data, checkpoints, outputs -----------------------------------------------
# Built datasets (parquet + SFT json). ~4 GB for the full 1.3M-maze corpus.
export TAILRL_MAZE_DATA_DIR="${TAILRL_MAZE_DATA_DIR:-${EXP_ROOT}/data}"
# Downloaded SFT initialization checkpoints, one ckpt-<step>/ per pretraining step.
export HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/tailrl_maze_sft_ckpts}"
# Hugging Face repos holding the released dataset and SFT checkpoints.
export TAILRL_MAZE_DATASET_REPO="${TAILRL_MAZE_DATASET_REPO:-max-rl/maze_17x17_diverse_1.3m}"
export TAILRL_MAZE_CKPT_REPO="${TAILRL_MAZE_CKPT_REPO:-max-rl/maze_v2_sft_ckpts_guanning}"
# RL checkpoints + logs. Kept out of the source tree by default.
export TAILRL_MAZE_RUN_DIR="${TAILRL_MAZE_RUN_DIR:-${EXP_ROOT}/runs}"

# --- experiment tracking -------------------------------------------------------
# Credentials come from ~/.netrc via `wandb login`. Never put an API key in the repo.
export WANDB_PROJECT="${WANDB_PROJECT:-tailrl-text-maze}"
# WANDB_ENTITY intentionally unset by default -> your account's default entity.
export WANDB_MODE="${WANDB_MODE:-online}"

# --- python --------------------------------------------------------------------
# The vendored verl fork is imported as top-level `verl`, so EXP_ROOT goes on the path.
export PYTHONPATH="${EXP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Never let ~/.local/lib site-packages shadow the environment's pinned versions.
export PYTHONNOUSERSITE=1

# --- SLURM (only read by scripts/slurm/*) ---------------------------------------
export TAILRL_SLURM_PARTITION="${TAILRL_SLURM_PARTITION:-gpu}"
export TAILRL_SLURM_QOS="${TAILRL_SLURM_QOS:-}"
export TAILRL_SLURM_ACCOUNT="${TAILRL_SLURM_ACCOUNT:-}"
export TAILRL_SLURM_GRES="${TAILRL_SLURM_GRES:-gpu:1}"
export TAILRL_SLURM_CPUS="${TAILRL_SLURM_CPUS:-16}"
export TAILRL_SLURM_MEM="${TAILRL_SLURM_MEM:-64G}"
export TAILRL_SLURM_TIME="${TAILRL_SLURM_TIME:-24:00:00}"
export TAILRL_LOG_DIR="${TAILRL_LOG_DIR:-${TAILRL_MAZE_RUN_DIR}/logs}"

# --- optional container ---------------------------------------------------------
# If TAILRL_MAZE_SIF points at an apptainer image, the SLURM wrappers run training
# inside it; otherwise they use whatever python is already on PATH.
export TAILRL_MAZE_SIF="${TAILRL_MAZE_SIF:-}"
export TAILRL_APPTAINER_BIN="${TAILRL_APPTAINER_BIN:-apptainer}"
# Comma-separated host paths to bind into the container (scratch filesystems etc.).
export TAILRL_APPTAINER_BIND="${TAILRL_APPTAINER_BIND:-}"

mkdir -p "${TAILRL_MAZE_RUN_DIR}" "${TAILRL_LOG_DIR}" 2>/dev/null || true
