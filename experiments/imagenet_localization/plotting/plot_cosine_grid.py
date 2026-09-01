"""Cosine similarity & magnitude ratio grids vs Ordinal CE gradient.

Port of experiment5_ordinal/plot_cosine_grid.py — restyled heatmap grids
(G × epoch) for each RL method, one set vs ordinal CE, one vs plain CE.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from experiments.imagenet_localization import paths


CMAP_COSINE = mcolors.LinearSegmentedColormap.from_list(
    "green_yellow_orange", ["#b85200", "#faf4b0", "#1a6e1a"], N=256,
)
CMAP_MAG = mcolors.LinearSegmentedColormap.from_list(
    "yellow_orange", ["#f7f7a0", "#b85200"], N=256,
)

# $TAILRL_RESULTS_DIR / $TAILRL_FIGURES_DIR override these (see paths.py).
RESULTS_DIR = Path(paths.results_dir())
OUTPUT_DIR = Path(paths.figures_dir())

BG_COLOR = "#FFFDF2"
EXP_IOU_LABEL = r"$J_{\mathrm{ord}}$"

EPOCHS = [1, 10, 25, 50]
G_VALUES = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
SEED = 43
K = 50


def _cosine_path(model_method: str, epoch: int, suffix: str = "") -> Path:
    return (RESULTS_DIR / f"{model_method}_K{K}_N64_seed{SEED}"
            / "gradient_analysis" / f"cosine_vs_G_epoch{epoch}{suffix}.json")


def generate_grid(
    rl_methods: list[str], rl_labels: list[str],
    output_name: str,
    ref_key: str = "ordinal_ce",
    ref_label: str = r"Ordinal CE ($\mathcal{J}_{\mathrm{OrdCE}}$)",
    model_method: str | None = None,
):
    """Generate a 2 × len(rl_methods) heatmap grid for one reference loss."""
    if model_method is None:
        # Defaults match experiment5's convention: ref=ordinal_ce/cross_entropy
        # uses its own model; ref=mse reads grads from the ordinal_ce model.
        model_method = "ordinal_ce" if ref_key == "mse" else ref_key
    suffix = f"_ref_{ref_key}" if ref_key != "ordinal_ce" else ""

    all_data = {}
    for ep in EPOCHS:
        f = _cosine_path(model_method, ep, suffix)
        if f.exists():
            with open(f) as fh:
                all_data[ep] = json.load(fh)
    if not all_data:
        print(f"[{output_name}] no cosine data for ref={ref_key}; skipping")
        return
    avail = sorted(all_data.keys())

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 11,
    })
    n_methods = len(rl_methods)
    fig, axes = plt.subplots(2, n_methods, figsize=(4.2 * n_methods, 9))
    if n_methods == 1:
        axes = axes.reshape(2, 1)

    metric_keys = [
        ("mean",           "Cosine Similarity", CMAP_COSINE, -0.2, 1.0),
        ("mag_ratio_mean", "Magnitude Ratio",    CMAP_MAG,    0.0,  None),
    ]

    all_mag = []
    for ep_data in all_data.values():
        results = ep_data.get("results", {})
        for m in rl_methods:
            if m not in results:
                continue
            for G in G_VALUES:
                v = results[m].get(str(G), {}).get("mag_ratio_mean", np.nan)
                if not np.isnan(v):
                    all_mag.append(v)
    shared_mag_vmax = max(3.0, max(all_mag) if all_mag else 3.0)

    im_row0 = im_row1 = None
    for col, (method, mlabel) in enumerate(zip(rl_methods, rl_labels)):
        for row, (mkey, mtitle, cmap, vmin, vmax_fixed) in enumerate(metric_keys):
            ax = axes[row, col]
            ax.set_facecolor(BG_COLOR)

            grid = np.full((len(G_VALUES), len(avail)), np.nan)
            for ei, ep in enumerate(avail):
                results = all_data[ep].get("results", {})
                if method not in results:
                    continue
                for gi, G in enumerate(G_VALUES):
                    grid[gi, ei] = results[method].get(str(G), {}).get(
                        mkey, np.nan,
                    )
            vmax = vmax_fixed if vmax_fixed is not None else shared_mag_vmax
            im = ax.imshow(grid, aspect="auto", origin="lower",
                           vmin=vmin, vmax=vmax, cmap=cmap)
            if row == 0:
                im_row0 = im
            else:
                im_row1 = im
            for gi in range(len(G_VALUES)):
                for ei in range(len(avail)):
                    v = grid[gi, ei]
                    if not np.isnan(v):
                        ax.text(ei, gi, f"{v:.2f}", ha="center", va="center",
                                fontsize=10.8, fontweight="medium", color="black")
            ax.set_xticks(range(len(avail)))
            ax.set_xticklabels([str(e) for e in avail], fontsize=14)
            ax.set_yticks(range(len(G_VALUES)))
            ax.set_yticklabels([str(g) for g in G_VALUES], fontsize=12)
            if row == 0:
                ax.set_title(mlabel, fontsize=18, pad=10)
            if col == 0:
                ax.set_ylabel(f"{mtitle}\n$G$", fontsize=18, labelpad=8)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08, top=0.88, hspace=0.25, right=0.92)

    cbar_ax0 = fig.add_axes([0.935, 0.50, 0.012, 0.35])
    fig.colorbar(im_row0, cax=cbar_ax0)
    cbar_ax0.tick_params(labelsize=12)
    cbar_ax1 = fig.add_axes([0.935, 0.08, 0.012, 0.35])
    fig.colorbar(im_row1, cax=cbar_ax1)
    cbar_ax1.tick_params(labelsize=12)

    fig.suptitle(f"vs. {ref_label} gradient", fontsize=24, y=0.97)
    fig.text(0.46, 0.01, "Epoch", ha="center", fontsize=22)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUTPUT_DIR / f"{output_name}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved {output_name}.{{png,pdf}}")


def main():
    # Full version vs J_ord (6 RL methods, tailrl_population model + ref).
    generate_grid(
        ["tailrl", "binary_maxrl", "grpo", "rloo", "reinforce"],
        ["TailRL", "Binary MaxRL", "GRPO", "RLOO", "REINFORCE", "DG"],
        output_name="cosine_grid_J_ord",
        ref_key="tailrl_population",
        ref_label=EXP_IOU_LABEL,
        model_method="tailrl_population",
    )
    # Vs plain CE (legacy — kept for cross-checks).
    generate_grid(
        ["tailrl", "binary_maxrl", "grpo", "rloo", "reinforce"],
        ["TailRL", "Binary MaxRL", "GRPO", "RLOO", "REINFORCE", "DG"],
        output_name="cosine_grid_cross_entropy",
        ref_key="cross_entropy",
        ref_label=r"Plain CE ($\mathcal{J}_{\mathrm{CE}}$)",
    )


if __name__ == "__main__":
    main()
