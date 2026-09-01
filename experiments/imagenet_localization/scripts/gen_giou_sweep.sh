#!/bin/bash
# Generate + submit the GIoU supervised arm (Rezatofighi et al., CVPR 2019)
# for seeds 43/44/45 — one SLURM job per seed.
#
# `giou` is a REGRESSION arm: it uses LocalizationRegressor and a single
# predicted box, so --N is not a rollout count here. It is kept at 64 purely
# so the job/results naming lines up with the sibling regression arms
# (l1_iou_match_K50_N64_seed*, mse_iou_match_K50_N64_seed*).
#
# Usage:
#   bash scripts/gen_giou_sweep.sh           # writes scripts + submits
#   DRY_RUN=1 bash scripts/gen_giou_sweep.sh # writes, prints submissions

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

DRY_RUN="${DRY_RUN:-0}"
SEEDS=(43 44 45)

METHOD="giou"
N=64
BATCH_SIZE=128
MEM="${TAILRL_SLURM_MEM}"
TIME="${TAILRL_SLURM_TIME}"

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
    local seed="$1"
    local job="${METHOD}_K50_N${N}_seed${seed}"
    local f="${TAILRL_SWEEP_DIR}/${job}.sh"
    cat > "${f}" <<EOF
#!/bin/bash
#SBATCH --job-name=${job}
#SBATCH --gres=${TAILRL_SLURM_GRES}
#SBATCH --cpus-per-task=${TAILRL_SLURM_CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
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
export WANDB_RUN_GROUP="${METHOD}_K50_N${N}"
source "${TAILRL_EXPERIMENT_DIR}/scripts/env.sh"

cd "\${TAILRL_ROOT}"

"\${TAILRL_PYTHON}" -m experiments.imagenet_localization.run \\
    --method ${METHOD} \\
    --K 50 \\
    --N ${N} \\
    --seed ${seed} \\
    --epochs 30 \\
    --batch_size ${BATCH_SIZE} \\
    --lr 5e-4 \\
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
for seed in "${SEEDS[@]}"; do
    f=$(write_script "${seed}")
    submit "${f}"
    n_jobs=$((n_jobs+1))
done

echo
echo "Total jobs: ${n_jobs}"
