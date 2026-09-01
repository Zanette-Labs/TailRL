#!/bin/bash
# Gradient + cosine analysis over this experiment checkpoints.
#
# Produces, for each eligible run and each milestone epoch in {1,10,25,50}:
#   <run_dir>/gradient_analysis/gradient_per_image_epoch{E}.json
#   <run_dir>/gradient_analysis/gradient_per_threshold_epoch{E}.json   (only
#     when the method is in gradient_analysis.PER_THRESHOLD_METHODS)
#
# And, for each reference loss and each epoch, one file per run that "owns"
# that reference model:
#   <ordinal_ce_seed{S}>/gradient_analysis/cosine_vs_G_epoch{E}.json
#   <cross_entropy_seed{S}>/gradient_analysis/cosine_vs_G_epoch{E}_ref_cross_entropy.json
#   <ordinal_ce_seed{S}>/gradient_analysis/cosine_vs_G_epoch{E}_ref_mse.json
#
# The default seed is 43 — matching experiment5_ordinal which produces the
# equivalent figures (2, 5, 6, 7) from a single seed. Figures 1, 3 and the
# val_iou_comparison_* figures aggregate over the training-dynamics
# metrics.json across seeds {43, 44, 45} (no gradient data needed) and don't
# require running this script multiple times.
#
# To also compute gradient analysis for seeds 44 / 45 (enabling per-seed
# error bands on figures 2 / 5 / 6 / 7), override the SEEDS env var:
#
#   SEEDS="43 44 45" bash scripts/run_gradient_analysis.sh
#
# That roughly triples the runtime — the cosine-vs-G sweep is the
# dominant cost.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# ---------------------------------------------------------------------------
# Config — override via env if needed. Paths (results, ImageNet, interpreter)
# come from env.sh.
# ---------------------------------------------------------------------------
SEEDS="${SEEDS:-43}"
K="${K:-50}"
MILESTONE_EPOCHS=(1 10 25 50)

# Per-image + per-threshold analyses are needed for figures 2 / 5 /
# gradient_geometry. Include both supervised and RL methods here.
PER_IMAGE_METHODS=(
    "ordinal_ce"
    "cross_entropy"
    "tailrl"
    "grpo"
    "reinforce"
    "rloo"
    "binary_maxrl"
    "tailrl_population"
)

# RL methods to compare against each reference in the cosine sweep.
COSINE_METHODS="tailrl,binary_maxrl,grpo,rloo,reinforce"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
cd "${TAILRL_ROOT}"

# ---------------------------------------------------------------------------
# Main loop over seeds.
# ---------------------------------------------------------------------------
for SEED in ${SEEDS}; do
    echo
    echo "###############################################################"
    echo "###        seed = ${SEED}                                      "
    echo "###############################################################"

    # Cosine analysis mapping depends on the current seed.
    declare -A REF_TO_RUN=(
        [ordinal_ce]="ordinal_ce_K${K}_N64_seed${SEED}"
        [cross_entropy]="cross_entropy_K${K}_N64_seed${SEED}"
        [mse]="ordinal_ce_K${K}_N64_seed${SEED}"
    )

    # ---------------------------------------------------------------------
    # 1. Per-image + per-threshold on each eligible run.
    # ---------------------------------------------------------------------
    for method in "${PER_IMAGE_METHODS[@]}"; do
        N=64
        run_name="${method}_K${K}_N${N}_seed${SEED}"
        run_dir="${TAILRL_RESULTS_DIR}/${run_name}"
        ckpt_dir="${run_dir}/checkpoints"
        out_dir="${run_dir}/gradient_analysis"

        if [[ ! -d "${ckpt_dir}" ]]; then
            echo "[skip] no checkpoint dir at ${ckpt_dir}"
            continue
        fi

        mkdir -p "${out_dir}"

        for epoch in "${MILESTONE_EPOCHS[@]}"; do
            ckpt="${ckpt_dir}/epoch_${epoch}.pt"
            if [[ ! -f "${ckpt}" ]]; then
                echo "[skip] ${ckpt} missing (epoch=${epoch})"
                continue
            fi
            echo "=== per_image  seed=${SEED}  ${run_name}  epoch=${epoch} ==="
            "${TAILRL_PYTHON}" -m experiments.imagenet_localization.analysis.gradient_analysis \
                per_image \
                --checkpoint "${ckpt}" \
                --data_dir "${IMAGENET_DIR}" \
                --method "${method}" \
                --K "${K}" \
                --seed "${SEED}" \
                --n_images 500 \
                --n_threshold_images 50 \
                --output_dir "${out_dir}"
        done
    done

    # ---------------------------------------------------------------------
    # 2. Cosine vs G for each reference loss.
    # ---------------------------------------------------------------------
    for ref in "${!REF_TO_RUN[@]}"; do
        run_name="${REF_TO_RUN[$ref]}"
        run_dir="${TAILRL_RESULTS_DIR}/${run_name}"
        ckpt_dir="${run_dir}/checkpoints"
        out_dir="${run_dir}/gradient_analysis"

        if [[ ! -d "${ckpt_dir}" ]]; then
            echo "[skip cosine ref=${ref}] no checkpoint dir at ${ckpt_dir}"
            continue
        fi
        mkdir -p "${out_dir}"

        for epoch in "${MILESTONE_EPOCHS[@]}"; do
            ckpt="${ckpt_dir}/epoch_${epoch}.pt"
            if [[ ! -f "${ckpt}" ]]; then
                echo "[skip] ${ckpt} missing (epoch=${epoch})"
                continue
            fi
            echo "=== cosine  seed=${SEED}  ref=${ref}  ${run_name}  epoch=${epoch} ==="
            "${TAILRL_PYTHON}" -m experiments.imagenet_localization.analysis.gradient_analysis \
                cosine \
                --checkpoint "${ckpt}" \
                --data_dir "${IMAGENET_DIR}" \
                --K "${K}" \
                --seed "${SEED}" \
                --ref_method "${ref}" \
                --methods "${COSINE_METHODS}" \
                --Gs "4,8,16,32,64,128,256,512,1024,4096" \
                --n_images 200 \
                --n_trials 10 \
                --output_dir "${out_dir}"
        done
    done

    unset REF_TO_RUN
done

echo
echo "Done. Gradient-analysis JSONs written under:"
echo "  ${TAILRL_RESULTS_DIR}/<run>/gradient_analysis/"
