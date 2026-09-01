#!/bin/bash
# Generate + submit the bmaxrl_adv_est sweep:
#   method = bmaxrl_adv_est  (continuous (r-mu)/(mu+eps), no binarization)
#   N      ∈ {16, 64, 256, 1024}
#   seeds  ∈ {43, 44, 45}
#   K = 50, epochs = 30, Adam, lr = 5e-4, batch_size = 128, grad_clip = 4.0
#
# Mirrors gen_rl_sweep.sh's layout. grad_clip is 4.0 for all N (per user
# request) — same value tailrl uses at N=1024.
#
# Usage:
#   bash scripts/gen_bmaxrl_adv_est_sweep.sh           # writes scripts + submits
#   DRY_RUN=1 bash scripts/gen_bmaxrl_adv_est_sweep.sh # writes, prints submissions

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

DRY_RUN="${DRY_RUN:-0}"
SEEDS=(43 44 45)
NS=(16 64 256 1024)

mkdir -p "${TAILRL_SWEEP_DIR}"

# An empty --partition is a submission error, so emit the directive only when
# TAILRL_SLURM_PARTITION is actually set; otherwise SLURM picks the default.
SBATCH_OPTIONAL=""
if [[ -n "${TAILRL_SLURM_PARTITION}" ]]; then
    SBATCH_OPTIONAL="#SBATCH --partition=${TAILRL_SLURM_PARTITION}"
fi

# Pin the W&B entity only when the generating shell has one — otherwise runs
# land in your default entity. Credentials always come from ~/.netrc
# (`wandb login`); no key is ever written into a generated script.
WANDB_ENTITY_EXPORT=""
if [[ -n "${WANDB_ENTITY:-}" ]]; then
    WANDB_ENTITY_EXPORT="export WANDB_ENTITY=\"${WANDB_ENTITY}\""
fi

write_script() {
    local N="$1" seed="$2"
    local job="bmaxrl_adv_est_K50_N${N}_seed${seed}"
    local f="${TAILRL_SWEEP_DIR}/${job}.sh"
    cat > "${f}" <<EOF
#!/bin/bash
#SBATCH --job-name=${job}
#SBATCH --gres=${TAILRL_SLURM_GRES}
#SBATCH --cpus-per-task=${TAILRL_SLURM_CPUS}
#SBATCH --mem=${TAILRL_SLURM_MEM}
#SBATCH --time=${TAILRL_SLURM_TIME}
#SBATCH --output=${TAILRL_LOG_DIR}/${job}_%j.out
${SBATCH_OPTIONAL}

set -eo pipefail

# Rebuild the generating environment on the compute node; env.sh then supplies
# PYTHONPATH, the conda env (if any) and the remaining W&B defaults.
export TAILRL_ROOT="${TAILRL_ROOT}"
export TAILRL_RESULTS_DIR="${TAILRL_RESULTS_DIR}"
export TAILRL_LOG_DIR="${TAILRL_LOG_DIR}"
export TAILRL_CONDA_ENV="${TAILRL_CONDA_ENV}"
export IMAGENET_DIR="${IMAGENET_DIR}"
export WANDB_DIR="${WANDB_DIR}"
export WANDB_PROJECT="${WANDB_PROJECT}"
${WANDB_ENTITY_EXPORT}
export WANDB_RUN_GROUP="bmaxrl_adv_est_K50_N${N}"
source "${TAILRL_EXPERIMENT_DIR}/scripts/env.sh"

export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_\${SLURM_JOB_ID}

cd "\${TAILRL_ROOT}"

"\${TAILRL_PYTHON}" -m experiments.imagenet_localization.run \\
    --method bmaxrl_adv_est \\
    --K 50 \\
    --N ${N} \\
    --seed ${seed} \\
    --epochs 30 \\
    --batch_size 128 \\
    --lr 5e-4 \\
    --grad_clip 4.0 \\
    --data_dir "\${IMAGENET_DIR}" \\
    --output_dir "\${TAILRL_RESULTS_DIR}/${job}" \\
    --num_workers 8 \\
    --wandb
EOF
    chmod +x "${f}"
    echo "${f}"
}

submit() {
    local f="$1"
    if [[ "${DRY_RUN}" = "1" ]]; then
        echo "DRY: sbatch ${f}"
    else
        sbatch "${f}"
    fi
}

n_jobs=0
for N in "${NS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        f=$(write_script "${N}" "${seed}")
        submit "${f}"
        n_jobs=$((n_jobs+1))
    done
done

echo
echo "Total jobs: ${n_jobs}"
