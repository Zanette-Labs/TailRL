#!/bin/bash
# Shared prelude for every script in this experiment. Source it, do not run it.
#
# Nothing here is a hardcoded absolute path: every value is an overridable default
# rooted at the experiment directory, so a fresh clone runs with no configuration.
# You will still want to point the big ones at scratch storage -- the test-case
# corpus is ~3.4 GB and the gem5 build tree is ~7 GB.
#
# Three prefixes, and they mean different things:
#   PIE_*      the task: dataset, test cases, gem5 toolchain, reward configuration.
#              These are read directly by the Python in code_opt/, not just by shell.
#   CODEOPT_*  where this experiment writes: runs, checkpoints, logs, evaluations.
#   TAILRL_*   repo-wide conventions shared with the other experiments (SLURM, W&B).
#
# shellcheck shell=bash

# --- locate the experiment root regardless of where the caller cd'd to ---------
EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EXP_ROOT
REPO_ROOT="$(cd "${EXP_ROOT}/../.." && pwd)"
export REPO_ROOT

# A `.env` at the TailRL repo root, if you made one (it is gitignored).
# shellcheck disable=SC1091
[ -f "${REPO_ROOT}/.env" ] && . "${REPO_ROOT}/.env"

# --- data ----------------------------------------------------------------------
# The verl parquets built by scripts/prepare_dataset.sh (~21 MB).
export PIE_PARQUET_ROOT="${PIE_PARQUET_ROOT:-${EXP_ROOT}/data/parquet}"
# The merged PIE test-case corpus: one directory per problem holding
# input.<i>.txt / output.<i>.txt. ~3.4 GB, 3,907 problems. Fetched by
# scripts/download_data.sh. The reward reads this variable directly and hard-errors
# if it is unset, because a missing corpus would otherwise look like a model that
# never writes correct code.
export PIE_TEST_CASE_DIR="${PIE_TEST_CASE_DIR:-${EXP_ROOT}/data/merged_test_cases}"

# --- gem5 timing stack ---------------------------------------------------------
# One root holding everything scripts/setup_gem5.sh builds. ~9 GB when finished.
# The individual paths below are derived from it in code_opt/measurement/gem5_backend.py
# and can each be overridden separately if you already have a gem5 build.
export PIE_GEM5_HOME="${PIE_GEM5_HOME:-${EXP_ROOT}/gem5/build}"

# --- model ---------------------------------------------------------------------
# A Hub id works directly. Point it at a local directory to avoid re-downloading.
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"

# --- outputs -------------------------------------------------------------------
# Checkpoints, logs, and evaluation artifacts. Kept out of the source tree.
export CODEOPT_RUN_DIR="${CODEOPT_RUN_DIR:-${EXP_ROOT}/runs}"
export CODEOPT_LOG_DIR="${CODEOPT_LOG_DIR:-${CODEOPT_RUN_DIR}/logs}"
export CODEOPT_EVAL_DIR="${CODEOPT_EVAL_DIR:-${CODEOPT_RUN_DIR}/eval}"

# --- caches --------------------------------------------------------------------
# Default to the usual per-user locations rather than writing into the checkout.
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${HOME}/.cache/vllm}"

# --- python --------------------------------------------------------------------
# The vendored verl fork is imported as top-level `verl`, and this experiment's own
# package as `code_opt`; both live directly under EXP_ROOT.
export PYTHONPATH="${EXP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Never let ~/.local/lib site-packages shadow the environment's pinned versions.
export PYTHONNOUSERSITE=1
# vLLM v1 changes the rollout worker's log-prob plumbing; the fork targets v0.
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

# --- experiment tracking -------------------------------------------------------
# Credentials come from ~/.netrc via `wandb login`. No API key is read from, or
# belongs in, this repository. Training also logs to the console alone.
export WANDB_PROJECT="${WANDB_PROJECT:-tailrl-code-optimization}"
# WANDB_ENTITY intentionally unset by default -> your account's default entity.

# --- SLURM (only read by scripts/slurm/*) --------------------------------------
export TAILRL_SLURM_PARTITION="${TAILRL_SLURM_PARTITION:-gpu}"
export TAILRL_SLURM_ACCOUNT="${TAILRL_SLURM_ACCOUNT:-}"
export TAILRL_SLURM_QOS="${TAILRL_SLURM_QOS:-}"
export TAILRL_SLURM_GRES="${TAILRL_SLURM_GRES:-gpu:4}"
export TAILRL_SLURM_CPUS="${TAILRL_SLURM_CPUS:-64}"
export TAILRL_SLURM_MEM="${TAILRL_SLURM_MEM:-0}"
export TAILRL_SLURM_TIME="${TAILRL_SLURM_TIME:-24:00:00}"

mkdir -p "${CODEOPT_RUN_DIR}" "${CODEOPT_LOG_DIR}" 2>/dev/null || true
