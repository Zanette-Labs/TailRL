#!/bin/bash
# Submit large-N TailRL runs as multi-GPU DDP jobs.
#
# Sized for an 8-GPU-per-user cap. Per-N GPU allocation:
#   N=4096:  4 GPUs/job → 2 jobs concurrent
#   N=16384: 4 GPUs/job → 2 jobs concurrent
#   N=65536: 8 GPUs/job → 1 job concurrent (uses the entire allowance)
#
# Per-GPU batch_size kept conservative (memory-safe). Effective batch is
# nproc_per_node * batch_size; gradient direction unchanged in expectation.
#
# Usage:
#   bash scripts/gen_ddp_largeN.sh           # writes scripts + submits
#   DRY_RUN=1 bash scripts/gen_ddp_largeN.sh # writes, prints submissions

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

DRY_RUN="${DRY_RUN:-0}"
SEEDS=(43 44 45)
# Tag: original TailRL (with *N) plus a survival-prob clamp on the empirical
# CDF (advantages.SURVIVAL_FLOOR_EPS=1e-4) — exact analog of the eps=1e-4
# clamp in localization_tailrl_population_loss on P(IoU > τ).
GROUP_SUFFIX="${GROUP_SUFFIX:-survFloor}"

mkdir -p "${TAILRL_SWEEP_DIR}"

# An empty --partition is a submission error, so emit the directive only when
# TAILRL_SLURM_PARTITION is set. TAILRL_SLURM_EXCLUDE is likewise optional:
# these configs want >=24GB of VRAM per GPU, so set it to a node list (or give
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

# (N, n_gpus, per_gpu_batch_size, mem)
# Per-GPU batch is set so EFFECTIVE batch = n_gpus * per_gpu_batch = 128,
# matching the single-GPU tailrl_population baseline (bs=128, eff=128).
# This keeps the lr schedule (steps/epoch, warmup, cosine) identical
# across all configurations — apples-to-apples comparison.
# mem is per-job rather than $TAILRL_SLURM_MEM: rollout tensors grow with N.
declare -a CONFIGS=(
    "4096    4   32    96G"
    "16384   4   32   128G"
    "65536   8   16   256G"
    "131072  8   16   384G"
    "262144  8   16   384G"
)

write_script() {
    local N="$1" ngpu="$2" batch_size="$3" mem="$4" seed="$5"
    local job="tailrl_K50_N${N}_seed${seed}_${GROUP_SUFFIX}"
    local f="${TAILRL_SWEEP_DIR}/${job}.sh"
    local cpus=$((ngpu * TAILRL_SLURM_CPUS))
    # Keep any GPU type from TAILRL_SLURM_GRES, swap its count for this job's.
    local gres="${TAILRL_SLURM_GRES%:*}:${ngpu}"
    cat > "${f}" <<EOF
#!/bin/bash
#SBATCH --job-name=${job}
#SBATCH --gres=${gres}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${cpus}
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

export MASTER_ADDR=\$(hostname)
export MASTER_PORT=\$((20000 + RANDOM % 20000))

cd "\${TAILRL_ROOT}"

"\${TAILRL_PYTHON}" -m torch.distributed.run \\
    --nproc_per_node=${ngpu} \\
    --master_addr=\${MASTER_ADDR} \\
    --master_port=\${MASTER_PORT} \\
    -m experiments.imagenet_localization.run \\
        --method tailrl \\
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

n_jobs=0
for cfg in "${CONFIGS[@]}"; do
    read -r N ngpu batch_size mem <<< "${cfg}"
    for seed in "${SEEDS[@]}"; do
        f=$(write_script "${N}" "${ngpu}" "${batch_size}" "${mem}" "${seed}")
        if [[ "${DRY_RUN}" = "1" ]]; then
            echo "DRY: sbatch ${f}  (N=${N}, gpus=${ngpu})"
        else
            sbatch "${f}"
        fi
        n_jobs=$((n_jobs+1))
    done
done

echo
echo "Total DDP jobs: ${n_jobs}"
