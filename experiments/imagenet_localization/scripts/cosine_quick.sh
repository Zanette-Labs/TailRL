#!/bin/bash
# One-shot cosine check (TailRL vs tailrl_population, fresh init) on one GPU.
#
# Usage:
#   bash scripts/cosine_quick.sh
#
# SLURM reads #SBATCH directives from a job file as literal text before the
# script runs, so they cannot see TAILRL_SLURM_* / TAILRL_LOG_DIR. Launch this
# with bash: outside an allocation it re-submits itself with those settings on
# the sbatch command line, and inside one (a job, or an salloc/srun shell) it
# runs the work directly. It takes a few minutes, so a short TAILRL_SLURM_TIME
# (e.g. 00:30:00) and a debug/interactive partition suit it well.

set -eo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    sbatch_opts=(
        --job-name=cos_quick
        --gres="${TAILRL_SLURM_GRES}"
        --cpus-per-task="${TAILRL_SLURM_CPUS}"
        --mem="${TAILRL_SLURM_MEM}"
        --time="${TAILRL_SLURM_TIME}"
        --output="${TAILRL_LOG_DIR}/cos_quick_%j.out"
    )
    if [[ -n "${TAILRL_SLURM_PARTITION}" ]]; then
        sbatch_opts+=(--partition="${TAILRL_SLURM_PARTITION}")
    fi
    exec sbatch "${sbatch_opts[@]}" "${BASH_SOURCE[0]}" "$@"
fi

cd "${TAILRL_ROOT}"

"${TAILRL_PYTHON}" -u "${TAILRL_EXPERIMENT_DIR}/scripts/cosine_quick.py" \
    --Gs 1024,16384,262144 \
    --n_images 8 \
    --seed 42
