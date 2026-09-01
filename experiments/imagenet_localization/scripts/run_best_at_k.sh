#!/bin/bash
# Compute unbiased best@k IoU on the val set for a set of runs, then plot.
#
# Usage:
#   bash scripts/run_best_at_k.sh OUT_PREFIX RUN_DIR1 [RUN_DIR2 ...]
# e.g.
#   bash scripts/run_best_at_k.sh "${TAILRL_RESULTS_DIR}/bestk/vanilla_N256" \
#       "${TAILRL_RESULTS_DIR}/tailrl_K50_N256_seed43" \
#       "${TAILRL_RESULTS_DIR}/grpo_K50_N256_seed43" \
#       "${TAILRL_RESULTS_DIR}/rloo_K50_N256_seed43"
#
# OUT_PREFIX is an absolute path prefix; writes <prefix>.json/.csv/.png.
# Single GPU; M=4096 rollouts/image; k grid = 1,4,16,64,256,1024.
#
# Resources come from TAILRL_SLURM_* (see scripts/env.sh); the job needs well
# under the default walltime, so e.g. TAILRL_SLURM_TIME=02:00:00 is plenty.

set -eo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if [[ $# -lt 2 ]]; then
    echo "usage: bash ${BASH_SOURCE[0]} OUT_PREFIX RUN_DIR1 [RUN_DIR2 ...]" >&2
    exit 1
fi

# SLURM reads #SBATCH directives from a job file as literal text before the
# script runs, so they cannot see TAILRL_SLURM_* / TAILRL_LOG_DIR. Outside an
# allocation this file therefore re-submits itself with those settings on the
# sbatch command line; inside one (a job, or an salloc/srun shell) it runs the
# work directly.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    sbatch_opts=(
        --job-name=bestk_eval
        --gres="${TAILRL_SLURM_GRES}"
        --cpus-per-task="${TAILRL_SLURM_CPUS}"
        --mem="${TAILRL_SLURM_MEM}"
        --time="${TAILRL_SLURM_TIME}"
        --output="${TAILRL_LOG_DIR}/bestk_eval_%j.out"
    )
    if [[ -n "${TAILRL_SLURM_PARTITION}" ]]; then
        sbatch_opts+=(--partition="${TAILRL_SLURM_PARTITION}")
    fi
    exec sbatch "${sbatch_opts[@]}" "${BASH_SOURCE[0]}" "$@"
fi

OUT_PREFIX="$1"; shift
RUN_DIRS=("$@")

cd "${TAILRL_ROOT}"

mkdir -p "$(dirname "${OUT_PREFIX}")"

# Optional smoke cap: MAX_IMAGES=1024 bash scripts/run_best_at_k.sh ... -> only
# evaluate that many val images.
MAX_IMAGES_ARG=()
if [[ -n "${MAX_IMAGES:-}" ]]; then
    MAX_IMAGES_ARG=(--max_images "${MAX_IMAGES}")
fi

"${TAILRL_PYTHON}" -m experiments.imagenet_localization.evaluation.eval_best_at_k \
    --run_dirs "${RUN_DIRS[@]}" \
    --data_dir "${IMAGENET_DIR}" \
    --K 50 --M 4096 --ks 1,4,16,64,256,1024 \
    --batch_size 128 --num_workers 8 --seed 0 "${MAX_IMAGES_ARG[@]}" \
    --out_json "${OUT_PREFIX}.json" --out_csv "${OUT_PREFIX}.csv"

"${TAILRL_PYTHON}" -m experiments.imagenet_localization.plotting.plot_best_at_k \
    --in_json "${OUT_PREFIX}.json" --out_png "${OUT_PREFIX}.png"

echo "DONE: ${OUT_PREFIX}.{json,csv,png}"
