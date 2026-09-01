"""Gradient geometry scatter — port of experiment5_ordinal/plot_gradient_geometry.py.

One panel per method: scatter of per-image gradient L2 norm (log y-axis)
against "IoU error" := 1 − E[IoU]. Replaces experiment5's (L-1) × (1 - E[r])
= MAE with the IoU-domain analog.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.imagenet_localization import paths


# $TAILRL_RESULTS_DIR / $TAILRL_FIGURES_DIR override these (see paths.py).
RESULTS_DIR = Path(paths.results_dir())
OUTPUT_DIR = Path(paths.figures_dir())
K = 50
EPOCH = 25
SEED = 43

BG_COLOR = "#FFFDF2"

EXP_IOU_LABEL = r"$J_{\mathrm{ord}}$"

# (run_dir, panel_title, scatter_color)
METHODS = [
    (f"tailrl_population_K{K}_N64_seed{SEED}", EXP_IOU_LABEL, "black"),
    (f"tailrl_K{K}_N64_seed{SEED}",            "TailRL",         "#E54040"),
    (f"grpo_K{K}_N256_seed{SEED}",          "GRPO",        "#00897B"),
    (f"reinforce_K{K}_N256_seed{SEED}",     "REINFORCE",   "#2979FF"),
]


def load_gradient_data(run_name: str):
    path = (RESULTS_DIR / run_name / "gradient_analysis"
            / f"gradient_per_image_epoch{EPOCH}.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.labelsize": 33.6,
        "axes.titlesize": 33.6,
        "xtick.labelsize": 21.8,
        "ytick.labelsize": 21.8,
    })

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5), sharey=True)

    all_data = []
    for run_name, _, _ in METHODS:
        data = load_gradient_data(run_name)
        all_data.append(data)

    for idx, (ax, (run_name, label, color), data) in enumerate(zip(axes, METHODS, all_data)):
        ax.set_facecolor(BG_COLOR)
        if data is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
        else:
            x = np.array([d["expected_iou"] for d in data])
            y = np.array([d["gradient_l2_norm"] for d in data])
            ax.scatter(x, y, c=color, s=27, alpha=0.5,
                       edgecolors="none", zorder=3)
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_title(label, fontweight="bold")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        if idx == 0:
            ax.set_ylabel(r"$\|\nabla_\theta \mathcal{L}\|_2$", labelpad=8)

    fig.supxlabel(r"$\mathbb{E}[\mathrm{IoU}]$", fontsize=33.6, y=-0.01)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.18)

    for ext in ("png", "pdf"):
        out = OUTPUT_DIR / f"gradient_geometry_iou.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved gradient_geometry_iou.{{png,pdf}}")


if __name__ == "__main__":
    main()
