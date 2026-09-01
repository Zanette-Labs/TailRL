#!/bin/bash
# Common environment prelude for every shell script in this experiment.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
#
# It resolves the repository root, applies repo-relative defaults for every path,
# and fails loudly (rather than silently training on nothing) if the one value
# that cannot be defaulted — the ImageNet location — is missing.
#
# Override any of these by exporting them before you call a script, or by
# putting them in a .env file at the repo root (see .env.example).

# --- repo root -------------------------------------------------------------
# Resolved from this file's own location so a clone works from anywhere.
if [[ -z "${TAILRL_ROOT:-}" ]]; then
    TAILRL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
export TAILRL_ROOT

# --- optional .env at the repo root ----------------------------------------
if [[ -f "${TAILRL_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${TAILRL_ROOT}/.env"
    set +a
fi

export TAILRL_EXPERIMENT_DIR="${TAILRL_ROOT}/experiments/imagenet_localization"

# --- paths -----------------------------------------------------------------
: "${TAILRL_RESULTS_DIR:=${TAILRL_ROOT}/results/imagenet_localization}"
: "${TAILRL_FIGURES_DIR:=${TAILRL_ROOT}/figures/imagenet_localization}"
: "${TAILRL_SWEEP_DIR:=${TAILRL_EXPERIMENT_DIR}/sweep_scripts}"
: "${TAILRL_LOG_DIR:=${TAILRL_SWEEP_DIR}/logs}"
export TAILRL_RESULTS_DIR TAILRL_FIGURES_DIR TAILRL_SWEEP_DIR TAILRL_LOG_DIR

# --- cluster ---------------------------------------------------------------
# Defaults are deliberately generic; set these for your own scheduler.
: "${TAILRL_SLURM_PARTITION:=}"          # empty -> let SLURM pick the default partition
: "${TAILRL_SLURM_GRES:=gpu:1}"          # e.g. gpu:a6000:1
: "${TAILRL_SLURM_CPUS:=8}"
: "${TAILRL_SLURM_MEM:=32G}"
: "${TAILRL_SLURM_TIME:=48:00:00}"
export TAILRL_SLURM_PARTITION TAILRL_SLURM_GRES TAILRL_SLURM_CPUS \
       TAILRL_SLURM_MEM TAILRL_SLURM_TIME

# --- python environment ----------------------------------------------------
# TAILRL_PYTHON  — absolute path to the interpreter to use. Falls back to the
#                  active conda env, then to whatever `python` is on PATH.
# TAILRL_CONDA_ENV — if set, `conda activate` it first. Leave empty to skip
#                  conda entirely (e.g. when you use venv or a container).
: "${TAILRL_CONDA_ENV:=}"
if [[ -n "${TAILRL_CONDA_ENV}" ]]; then
    # shellcheck disable=SC1090
    [[ -f "${HOME}/.bashrc" ]] && source "${HOME}/.bashrc"
    conda activate "${TAILRL_CONDA_ENV}"
fi
: "${TAILRL_PYTHON:=${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}}"
: "${TAILRL_PYTHON:=python}"
export TAILRL_CONDA_ENV TAILRL_PYTHON

# --- weights & biases ------------------------------------------------------
# wandb reads credentials from ~/.netrc (written by `wandb login`) — no API key
# is ever read from, or written into, this repository.
: "${WANDB_PROJECT:=tailrl-imagenet-localization}"
: "${WANDB_DIR:=${TAILRL_ROOT}/wandb}"
export WANDB_PROJECT WANDB_DIR
# WANDB_ENTITY is intentionally left unset so runs land in your default entity.
[[ -n "${WANDB_ENTITY:-}" ]] && export WANDB_ENTITY

# --- imports ---------------------------------------------------------------
export PYTHONPATH="${TAILRL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# --- the one thing that cannot be defaulted --------------------------------
# Scripts that do not touch the dataset can set TAILRL_SKIP_DATA_CHECK=1.
if [[ -z "${TAILRL_SKIP_DATA_CHECK:-}" ]]; then
    if [[ -z "${IMAGENET_DIR:-}" ]]; then
        echo "ERROR: IMAGENET_DIR is not set." >&2
        echo "  export IMAGENET_DIR=/path/to/imagenet   # must contain LOC_val_solution.csv and ILSVRC/" >&2
        echo "  (or copy .env.example to .env at the repo root and fill it in)" >&2
        return 1 2>/dev/null || exit 1
    fi
    if [[ ! -d "${IMAGENET_DIR}" ]]; then
        echo "ERROR: IMAGENET_DIR=${IMAGENET_DIR} is not a directory." >&2
        return 1 2>/dev/null || exit 1
    fi
    export IMAGENET_DIR
fi

mkdir -p "${TAILRL_RESULTS_DIR}" "${TAILRL_LOG_DIR}"
