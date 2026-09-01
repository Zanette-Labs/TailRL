#!/bin/bash
# Generate + submit a clean RL sweep: tailrl at all N's, plus grpo/rloo/
# reinforce/binary_maxrl at N=256, for seeds 43/44/45.
#
# Reflects the fix (no * N inside tailrl_advantage, .sum() over rollouts in
# the TailRL branch of train.py) — both pieces are already in the codebase.
#
# Usage:
#   bash scripts/gen_rl_sweep.sh           # writes scripts + submits
#   DRY_RUN=1 bash scripts/gen_rl_sweep.sh # writes, prints submissions

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

DRY_RUN="${DRY_RUN:-0}"
SEEDS=(43 44 45)

# Tag this sweep so the W&B groups stay distinct from prior sweeps. The
# run displayName still matches the local output_dir so files line up,
# but W&B group is suffixed to keep histories separate.
GROUP_SUFFIX="${GROUP_SUFFIX:-fixedN}"

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
    local method="$1" N="$2" seed="$3" batch_size="$4" mem="$5" time="$6"
    local job="${method}_K50_N${N}_seed${seed}_${GROUP_SUFFIX}"
    local f="${TAILRL_SWEEP_DIR}/${job}.sh"
    cat > "${f}" <<EOF
#!/bin/bash
#SBATCH --job-name=${job}
#SBATCH --gres=${TAILRL_SLURM_GRES}
#SBATCH --cpus-per-task=${TAILRL_SLURM_CPUS}
#SBATCH --mem=${mem}
#SBATCH --time=${time}
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
export WANDB_RUN_GROUP="${method}_K50_N${N}_${GROUP_SUFFIX}"
source "${TAILRL_EXPERIMENT_DIR}/scripts/env.sh"

cd "\${TAILRL_ROOT}"

"\${TAILRL_PYTHON}" -m experiments.imagenet_localization.run \\
    --method ${method} \\
    --K 50 \\
    --N ${N} \\
    --seed ${seed} \\
    --epochs 30 \\
    --batch_size ${batch_size} \\
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

# (method, N, batch_size, mem, time) — params chosen for stability:
#  - small N: the cluster defaults (bs=128, mem=$TAILRL_SLURM_MEM)
#  - large N: smaller bs + an explicit bigger mem (rollout tensors get large)
declare -a CONFIGS=(
    "tailrl          16   128  ${TAILRL_SLURM_MEM}  ${TAILRL_SLURM_TIME}"
    "tailrl          64   128  ${TAILRL_SLURM_MEM}  ${TAILRL_SLURM_TIME}"
    "tailrl         256   128  ${TAILRL_SLURM_MEM}  ${TAILRL_SLURM_TIME}"
    "tailrl        1024   128  ${TAILRL_SLURM_MEM}  ${TAILRL_SLURM_TIME}"
    "tailrl        4096   128  ${TAILRL_SLURM_MEM}  ${TAILRL_SLURM_TIME}"
    "tailrl       16384    64  64G                  ${TAILRL_SLURM_TIME}"
    "tailrl       65536    16 128G                  ${TAILRL_SLURM_TIME}"
    "grpo        256   128  ${TAILRL_SLURM_MEM}  ${TAILRL_SLURM_TIME}"
    "rloo        256   128  ${TAILRL_SLURM_MEM}  ${TAILRL_SLURM_TIME}"
    "reinforce   256   128  ${TAILRL_SLURM_MEM}  ${TAILRL_SLURM_TIME}"
    "binary_maxrl 256  128  ${TAILRL_SLURM_MEM}  ${TAILRL_SLURM_TIME}"
)

n_jobs=0
for cfg in "${CONFIGS[@]}"; do
    read -r method N batch_size mem time <<< "${cfg}"
    for seed in "${SEEDS[@]}"; do
        f=$(write_script "${method}" "${N}" "${seed}" "${batch_size}" "${mem}" "${time}")
        submit "${f}"
        n_jobs=$((n_jobs+1))
    done
done

echo
echo "Total jobs: ${n_jobs}"
