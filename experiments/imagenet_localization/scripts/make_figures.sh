#!/bin/bash
# Generate every figure for this experiment after training + gradient analysis finish.
#
# Usage:
#   bash scripts/make_figures.sh                   # default: K=50, seeds 43 44 45
#   FIG_K=50 FIG_SEEDS="42 43" bash scripts/make_figures.sh
#
# Emits PDFs + PNGs under $TAILRL_FIGURES_DIR:
#   figure1_training_dynamics.*
#   figure2_gradient_geometry.*
#   figure3_convergence.*
#   figure5_per_threshold.*
#   figure6_cosine_vs_G.*
#   figure7_grid_{ordinal_ce,cross_entropy,mse}.*
#   val_iou_comparison_v{1_neurips,2_presentation,3_minimalist,4_grouped_zoom}.*
#   cosine_grid_{ordinal_ce,cross_entropy,mse}.*
#   gradient_geometry_iou.*
#
# Prerequisites:
#   - metrics.json populated under $TAILRL_RESULTS_DIR/<run>/ for each seed
#     (written by run.py when training completes at least one epoch).
#   - gradient_analysis/*.json populated by scripts/run_gradient_analysis.sh
#     (default SEED=43; checkpoint-based figures 2, 5, 6, 7, cosine_grid_*,
#     gradient_geometry_iou). Override with SEEDS="43 44 45" for multi-seed
#     gradient analysis.
# If gradient-analysis data is missing, the figures that depend on it are skipped.

set -euo pipefail

# Plotting only reads the JSON written by earlier stages, so the dataset itself
# is not needed here.
TAILRL_SKIP_DATA_CHECK=1
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

FIG_K="${FIG_K:-50}"
FIG_SEEDS="${FIG_SEEDS:-43 44 45}"
SEED_ANALYSIS="${SEED_ANALYSIS:-43}"

cd "${TAILRL_ROOT}"
mkdir -p "${TAILRL_FIGURES_DIR}"

echo "=== plot.py — figures 1, 2, 3, 5, 6, 7 ==="
"${TAILRL_PYTHON}" -m experiments.imagenet_localization.plotting.plot \
    --K "${FIG_K}" \
    --seeds ${FIG_SEEDS} \
    --seed_analysis "${SEED_ANALYSIS}" \
    --results_dir "${TAILRL_RESULTS_DIR}" \
    --output_dir "${TAILRL_FIGURES_DIR}"

echo
echo "=== plot_val_iou_comparison.py — 4 stylistic variants ==="
"${TAILRL_PYTHON}" -m experiments.imagenet_localization.plotting.plot_val_iou_comparison

echo
echo "=== plot_cosine_grid.py — heatmap grids ==="
"${TAILRL_PYTHON}" -m experiments.imagenet_localization.plotting.plot_cosine_grid

echo
echo "=== plot_gradient_geometry.py — IoU-error scatter ==="
"${TAILRL_PYTHON}" -m experiments.imagenet_localization.plotting.plot_gradient_geometry

echo
echo "Figures written under: ${TAILRL_FIGURES_DIR}"
