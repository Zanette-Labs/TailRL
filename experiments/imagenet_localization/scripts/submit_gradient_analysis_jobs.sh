#!/bin/bash
# Submit gradient + cosine analysis as parallel SLURM jobs.
#
# QOS limit: 8 GPUs/user. We batch coarsely:
#   - one per_image job per seed: loops through every (method, N) combo with
#     checkpoints for that seed and runs all milestone epochs sequentially
#   - one cosine job per (ref_method, seed): loops through milestone epochs
#
# That keeps the total job count comfortably under the QOS cap.
#
# Usage:
#   bash scripts/submit_gradient_analysis_jobs.sh           # default seeds 43 44 45
#   SEEDS="43" bash scripts/submit_gradient_analysis_jobs.sh
#   DRY_RUN=1 bash scripts/submit_gradient_analysis_jobs.sh

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

JOBS_DIR="${TAILRL_SWEEP_DIR}/grad_jobs"

SEEDS="${SEEDS:-43 44 45}"
K="${K:-50}"
DRY_RUN="${DRY_RUN:-0}"

# (method, N) — N is the variant whose checkpoints are populated.
PER_IMAGE_METHOD_N=(
    "tailrl:64"
    "grpo:256"
    "reinforce:256"
    "rloo:256"
    "binary_maxrl:64"
    "tailrl_population:64"
    "cross_entropy:64"
)

# Cosine reference runs whose checkpoints are populated. (ordinal_ce / mse
# refs are blocked because ordinal_ce checkpoints don't exist.)
COSINE_REF_RUN_BASE=(
    "cross_entropy:cross_entropy_K${K}_N64"
)
COSINE_METHODS="tailrl,binary_maxrl,grpo,rloo,reinforce"

mkdir -p "${TAILRL_LOG_DIR}" "${JOBS_DIR}"

# An empty --partition is a submission error, so emit the directive only when
# TAILRL_SLURM_PARTITION is actually set; otherwise SLURM picks the default.
SBATCH_OPTIONAL=""
if [[ -n "${TAILRL_SLURM_PARTITION}" ]]; then
    SBATCH_OPTIONAL="#SBATCH --partition=${TAILRL_SLURM_PARTITION}"
fi

# Header writer — the paths are baked in at generation time because SLURM reads
# the #SBATCH directives as literal text, before any variable exists. The
# TAILRL_* exports then rebuild the generating environment on the compute node
# so env.sh can supply PYTHONPATH, the conda env (if any) and TAILRL_PYTHON.
write_sbatch_header() {
    local job_name="$1"
    local out_path="$2"
    cat <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --gres=${TAILRL_SLURM_GRES}
#SBATCH --cpus-per-task=${TAILRL_SLURM_CPUS}
#SBATCH --mem=${TAILRL_SLURM_MEM}
#SBATCH --time=${TAILRL_SLURM_TIME}
#SBATCH --output=${out_path}
${SBATCH_OPTIONAL}

set -eo pipefail

export TAILRL_ROOT="${TAILRL_ROOT}"
export TAILRL_RESULTS_DIR="${TAILRL_RESULTS_DIR}"
export TAILRL_LOG_DIR="${TAILRL_LOG_DIR}"
export TAILRL_CONDA_ENV="${TAILRL_CONDA_ENV}"
export IMAGENET_DIR="${IMAGENET_DIR}"
source "${TAILRL_EXPERIMENT_DIR}/scripts/env.sh"

cd "\${TAILRL_ROOT}"
EOF
}

submit() {
    local script="$1"
    chmod +x "${script}"
    if [[ "${DRY_RUN}" = "1" ]]; then
        echo "DRY: sbatch ${script}"
    else
        sbatch "${script}"
    fi
}

# ---------------------------------------------------------------------------
# 1. per_image — one job per seed.
# ---------------------------------------------------------------------------
echo "=== per_image jobs (1 per seed) ==="
n_per_image=0
for SEED in ${SEEDS}; do
    job_name="gpi_seed${SEED}"
    script="${JOBS_DIR}/${job_name}.sh"
    log_path="${TAILRL_LOG_DIR}/${job_name}_%j.out"

    {
        write_sbatch_header "${job_name}" "${log_path}"
        echo
        echo "echo \"### per_image  seed=${SEED}  $(date) ###\""
        for entry in "${PER_IMAGE_METHOD_N[@]}"; do
            method="${entry%:*}"
            N="${entry#*:}"
            run_name="${method}_K${K}_N${N}_seed${SEED}"
            run_dir="${TAILRL_RESULTS_DIR}/${run_name}"
            ckpt_dir="${run_dir}/checkpoints"
            out_dir="${run_dir}/gradient_analysis"

            [[ ! -d "${ckpt_dir}" ]] && continue
            epochs=()
            for ep in 1 10 25 50; do
                [[ -f "${ckpt_dir}/epoch_${ep}.pt" ]] && epochs+=("${ep}")
            done
            [[ ${#epochs[@]} -eq 0 ]] && continue

            echo
            echo "mkdir -p \"${out_dir}\""
            for ep in "${epochs[@]}"; do
                cat <<EOF
echo "=== per_image  ${run_name}  epoch=${ep} ==="
"\${TAILRL_PYTHON}" -m experiments.imagenet_localization.analysis.gradient_analysis per_image \\
    --checkpoint "${ckpt_dir}/epoch_${ep}.pt" \\
    --data_dir "\${IMAGENET_DIR}" \\
    --method "${method}" \\
    --K ${K} \\
    --seed ${SEED} \\
    --n_images 500 \\
    --n_threshold_images 50 \\
    --output_dir "${out_dir}"
EOF
            done
        done
    } > "${script}"
    submit "${script}"
    n_per_image=$((n_per_image+1))
done

# ---------------------------------------------------------------------------
# 2. cosine — one job per (ref, seed).
# ---------------------------------------------------------------------------
echo
echo "=== cosine jobs (1 per ref × seed) ==="
n_cosine=0
for SEED in ${SEEDS}; do
    for entry in "${COSINE_REF_RUN_BASE[@]}"; do
        ref="${entry%:*}"
        run_base="${entry#*:}"
        run_name="${run_base}_seed${SEED}"
        run_dir="${TAILRL_RESULTS_DIR}/${run_name}"
        ckpt_dir="${run_dir}/checkpoints"
        out_dir="${run_dir}/gradient_analysis"

        [[ ! -d "${ckpt_dir}" ]] && continue
        epochs=()
        for ep in 1 10 25 50; do
            [[ -f "${ckpt_dir}/epoch_${ep}.pt" ]] && epochs+=("${ep}")
        done
        [[ ${#epochs[@]} -eq 0 ]] && continue

        job_name="gcos_${ref}_s${SEED}"
        script="${JOBS_DIR}/${job_name}.sh"
        log_path="${TAILRL_LOG_DIR}/${job_name}_%j.out"

        {
            write_sbatch_header "${job_name}" "${log_path}"
            echo
            echo "mkdir -p \"${out_dir}\""
            for ep in "${epochs[@]}"; do
                cat <<EOF
echo "=== cosine  ref=${ref}  ${run_name}  epoch=${ep} ==="
"\${TAILRL_PYTHON}" -m experiments.imagenet_localization.analysis.gradient_analysis cosine \\
    --checkpoint "${ckpt_dir}/epoch_${ep}.pt" \\
    --data_dir "\${IMAGENET_DIR}" \\
    --K ${K} \\
    --seed ${SEED} \\
    --ref_method "${ref}" \\
    --methods "${COSINE_METHODS}" \\
    --Gs "4,8,16,32,64,128,256,512,1024,4096" \\
    --n_images 200 \\
    --n_trials 10 \\
    --output_dir "${out_dir}"
EOF
            done
        } > "${script}"
        submit "${script}"
        n_cosine=$((n_cosine+1))
    done
done

echo
echo "Submitted: per_image=${n_per_image}  cosine=${n_cosine}  total=$((n_per_image+n_cosine))"
echo "Job scripts in: ${JOBS_DIR}"
echo "Logs in:        ${TAILRL_LOG_DIR}"
