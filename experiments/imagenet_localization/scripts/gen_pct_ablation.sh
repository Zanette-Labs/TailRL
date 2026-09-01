#!/bin/bash
# Generate + submit the PERCENTILE-REWARD ablation: tailrl/grpo/rloo at N=256,
# seed 43, with --reward_transform percentile. Single-GPU jobs (any GPU type).
#
# The percentile transform maps each rollout's IoU reward to its
# average-percentile rank in [0, 1] within the group of N rollouts
# (highest -> 1.0, lowest -> 0.0, ties -> median percentile) BEFORE the
# advantage estimator runs. This strips IoU magnitude and keeps only the
# ordering. Compares against the existing raw-IoU N=256 baselines already
# in ${TAILRL_RESULTS_DIR} (tailrl/grpo/rloo _K50_N256_seed43).
#
# Mirrors scripts/gen_rl_sweep.sh's N=256 config exactly (adam, lr=5e-4,
# epochs=30, batch=128, K=50, num_workers=8) so the only difference vs the
# raw baselines is --reward_transform percentile.
#
# Usage:
#   bash scripts/gen_pct_ablation.sh            # writes scripts + submits
#   DRY_RUN=1 bash scripts/gen_pct_ablation.sh  # writes scripts + prints submissions

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

DRY_RUN="${DRY_RUN:-0}"

SEED=43
N="${N:-256}"          # override group size, e.g. N=16 bash scripts/gen_pct_ablation.sh
K=50
TRANSFORM="percentile"

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
    local method="$1"
    local job="${method}_${TRANSFORM}_K${K}_N${N}_seed${SEED}"
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
export WANDB_RUN_GROUP="${method}_${TRANSFORM}_K${K}_N${N}"
source "${TAILRL_EXPERIMENT_DIR}/scripts/env.sh"

cd "\${TAILRL_ROOT}"

"\${TAILRL_PYTHON}" -m experiments.imagenet_localization.run \\
    --method ${method} \\
    --reward_transform ${TRANSFORM} \\
    --K ${K} \\
    --N ${N} \\
    --seed ${SEED} \\
    --epochs 30 \\
    --batch_size 128 \\
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
for method in tailrl grpo rloo; do
    f=$(write_script "${method}")
    submit "${f}"
    n_jobs=$((n_jobs+1))
done

echo
echo "Total jobs: ${n_jobs}  (group suffix: _${TRANSFORM}_K${K}_N${N})"
