#!/bin/bash
# Submit large-N TailRL runs as SINGLE-GPU jobs (bs=128, eff_batch=128).
#
# Why single-GPU: 53 BatchNorm2d modules in the ResNet backbone — DDP at
# bs=32/GPU computes BN running stats from per-rank mini-batches of 32,
# while the tailrl_population baseline used bs=128 on a single GPU (BN over
# 128 images). That difference subtly shifts early training even when
# eff_batch is otherwise matched. Single-GPU bs=128 = exactly the baseline.
#
# Usage:
#   bash scripts/gen_singlegpu_largeN.sh           # writes scripts + submits
#   DRY_RUN=1 bash scripts/gen_singlegpu_largeN.sh # writes, prints submissions

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

DRY_RUN="${DRY_RUN:-0}"
SEEDS=(43 44 45)
GROUP_SUFFIX="${GROUP_SUFFIX:-singleGPU}"

mkdir -p "${TAILRL_SWEEP_DIR}"

# An empty --partition is a submission error, so emit the directive only when
# TAILRL_SLURM_PARTITION is set. TAILRL_SLURM_EXCLUDE is likewise optional:
# these configs want ≥24GB of VRAM, so set it to a node list (or give
# TAILRL_SLURM_GRES an explicit GPU type) to keep them off smaller cards.
SBATCH_OPTIONAL=""
if [[ -n "${TAILRL_SLURM_PARTITION}" ]]; then
    SBATCH_OPTIONAL="#SBATCH --partition=${TAILRL_SLURM_PARTITION}"
fi
if [[ -n "${TAILRL_SLURM_EXCLUDE:-}" ]]; then
    SBATCH_OPTIONAL="${SBATCH_OPTIONAL:+${SBATCH_OPTIONAL}
}#SBATCH --exclude=${TAILRL_SLURM_EXCLUDE}"
fi

# Pin the W&B entity only when the generating shell has one — otherwise runs
# land in your default entity. Credentials always come from ~/.netrc
# (`wandb login`); no key is ever written into a generated script.
WANDB_ENTITY_EXPORT=""
if [[ -n "${WANDB_ENTITY:-}" ]]; then
    WANDB_ENTITY_EXPORT="export WANDB_ENTITY=\"${WANDB_ENTITY}\""
fi

# (N, mem) — partition and walltime come from TAILRL_SLURM_PARTITION and
# TAILRL_SLURM_TIME; mem is per-job because rollout tensors grow with N.
# - N ≤ 16384 fits comfortably in the default 48h walltime
# - N ≥ 65536 may need a longer-walltime (preemptible) partition — runs are
#   checkpointed every 10 epochs so a preempted job can resume.
declare -a CONFIGS=(
    "4096    96G"
    "16384   96G"
    "65536   128G"
)

write_script() {
    local N="$1" mem="$2" seed="$3"
    local job="tailrl_K50_N${N}_seed${seed}_${GROUP_SUFFIX}"
    local f="${TAILRL_SWEEP_DIR}/${job}.sh"
    cat > "${f}" <<EOF
#!/bin/bash
#SBATCH --job-name=${job}
#SBATCH --gres=${TAILRL_SLURM_GRES}
#SBATCH --cpus-per-task=${TAILRL_SLURM_CPUS}
#SBATCH --mem=${mem}
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
export WANDB_RUN_GROUP="tailrl_K50_N${N}_${GROUP_SUFFIX}"
source "${TAILRL_EXPERIMENT_DIR}/scripts/env.sh"

cd "\${TAILRL_ROOT}"

# Speedup stack: bf16 autocast (~1.4×, gradient cos verified ≈ 1.0 vs fp32)
# + torch.compile mode='default' (~1.3×) + channels_last layout (~1.5×,
# baked into model.to() inside run.py)
export BF16=1
# Compile is enabled by default; set DISABLE_COMPILE=1 to turn off.

"\${TAILRL_PYTHON}" -m experiments.imagenet_localization.run \\
    --method tailrl \\
    --K 50 \\
    --N ${N} \\
    --seed ${seed} \\
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

n_jobs=0
for cfg in "${CONFIGS[@]}"; do
    read -r N mem <<< "${cfg}"
    for seed in "${SEEDS[@]}"; do
        f=$(write_script "${N}" "${mem}" "${seed}")
        if [[ "${DRY_RUN}" = "1" ]]; then
            echo "DRY: sbatch ${f}  (N=${N}, partition=${TAILRL_SLURM_PARTITION:-<scheduler default>})"
        else
            sbatch "${f}"
        fi
        n_jobs=$((n_jobs+1))
    done
done

echo
echo "Total single-GPU jobs: ${n_jobs}"
