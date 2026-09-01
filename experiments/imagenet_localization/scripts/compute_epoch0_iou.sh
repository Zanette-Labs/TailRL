#!/bin/bash
# Epoch-0 (pre-training) val IoU for both architectures, on one GPU.
#
# Usage:
#   bash scripts/compute_epoch0_iou.sh
#
# SLURM reads #SBATCH directives from a job file as literal text before the
# script runs, so they cannot see TAILRL_SLURM_* / TAILRL_LOG_DIR. Launch this
# with bash: outside an allocation it re-submits itself with those settings on
# the sbatch command line, and inside one (a job, or an salloc/srun shell) it
# runs the work directly. It is a short job, so a modest TAILRL_SLURM_TIME
# (e.g. 01:00:00) is enough.

set -eo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    sbatch_opts=(
        --job-name=epoch0_iou
        --gres="${TAILRL_SLURM_GRES}"
        --cpus-per-task="${TAILRL_SLURM_CPUS}"
        --mem="${TAILRL_SLURM_MEM}"
        --time="${TAILRL_SLURM_TIME}"
        --output="${TAILRL_LOG_DIR}/epoch0_iou_%j.out"
    )
    if [[ -n "${TAILRL_SLURM_PARTITION}" ]]; then
        sbatch_opts+=(--partition="${TAILRL_SLURM_PARTITION}")
    fi
    exec sbatch "${sbatch_opts[@]}" "${BASH_SOURCE[0]}" "$@"
fi

cd "${TAILRL_ROOT}"

"${TAILRL_PYTHON}" -u "${TAILRL_EXPERIMENT_DIR}/scripts/compute_epoch0_iou.py"
