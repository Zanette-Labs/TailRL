"""Plot generation for the ImageNet localization RL experiment.

Ports experiment5_ordinal/plot.py 1:1 with metric-key renames:

  val/mae                -> 1 - val/iou_greedy   (displayed as "IoU error ↓")
  val/best_of_64_reward  -> val/iou_best_of_N
  val/ordinal_ce_objective -> -val/ordinal_ce_objective   (this experiment logs
      the positive LOSS; we flip sign to show J_ML ↑ like experiment5)
  val/acc_exact / val/acc_within_K -> val/iou_at_{50,75,90}
  cont_maxrl             -> tailrl
  cont_maxrl_no_bl       -> (no analog in this experiment; dropped)

Produces (under $TAILRL_FIGURES_DIR):
  figure1_training_dynamics.{pdf,png}
  figure2_gradient_geometry.{pdf,png}
  figure3_convergence.{pdf,png}
  figure5_per_threshold.{pdf,png}
  figure6_cosine_vs_G.{pdf,png}
  figure7_grid_{ordinal_ce,cross_entropy,mse}.{pdf,png}

Reads metrics.json + gradient_analysis/ JSONs written by train.py and
experiments.imagenet_localization.analysis.gradient_analysis.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.imagenet_localization import paths


# NeurIPS-style defaults — match plot_val_iou_comparison v1.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 18,
    "legend.fontsize": 11,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# Method display styling — the analog of METHOD_STYLES in experiment5.
# Shared visual conventions across every figure in this module.
BG_COLOR = "#FFFDF2"

# All-red TailRL palette, darker as G grows.
TAILRL_COLOR_BY_N = {
    16:   "#FF8B8B",
    64:   "#E54040",
    256:  "#A8001E",
    1024: "#5A0010",
}
EXP_IOU_LABEL = r"$J_{\mathrm{ord}}$"

METHOD_STYLES = {
    "tailrl_population":      dict(label=EXP_IOU_LABEL,             color="black",   ls="--", lw=3.6, zorder=12, dashes=(6, 6)),
    "l1_centroid_match":   dict(label="L1",                       color="#a04000", ls="--", lw=2.4),
    "mse_centroid_match":  dict(label="MSE",                      color="#8c564b", ls="--", lw=2.4),
    "tailrl_N16":             dict(label="TailRL (N=16)",               color=TAILRL_COLOR_BY_N[16],   ls="-", lw=2.7, alpha=1.0, zorder=6),
    "tailrl_N64":             dict(label="TailRL (N=64)",               color=TAILRL_COLOR_BY_N[64],   ls="-", lw=3.4, alpha=1.0, zorder=8),
    "tailrl_N256":            dict(label="TailRL (N=256)",              color=TAILRL_COLOR_BY_N[256],  ls="-", lw=4.0, alpha=1.0, zorder=10),
    "tailrl_N1024":           dict(label="TailRL (N=1024)",             color=TAILRL_COLOR_BY_N[1024], ls="-", lw=4.4, alpha=1.0, zorder=11),
    "grpo_N256":           dict(label="GRPO (N=256)",             color="#00897B", ls="-",  lw=2.8),
    "rloo_N256":           dict(label="RLOO (N=256)",             color="#9467bd", ls="-",  lw=2.4),
    "reinforce_N256":      dict(label="REINFORCE (N=256)",        color="#2979FF", ls="-",  lw=2.6),
}


def _smooth(y: np.ndarray, window: int = 3) -> np.ndarray:
    """Light moving-average smoothing that preserves NaNs."""
    if window < 2 or len(y) < window:
        return y
    pad = window // 2
    padded = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    out = np.full_like(y, np.nan, dtype=float)
    for i in range(len(y)):
        win = padded[i:i + window]
        m = ~np.isnan(win)
        if m.any():
            out[i] = win[m].mean()
    return out


# =============================================================================
# metrics.json loader
# =============================================================================

def _run_name(method: str, K: int, N: int, seed: int) -> str:
    return f"{method}_K{K}_N{N}_seed{seed}"


def load_metrics(
    results_dir: str, method: str, K: int, N: int, seeds: list[int],
) -> dict:
    """Load metrics.json across seeds and align to the union of epochs.

    Seeds that ran shorter contribute NaN past their end so longer seeds
    still inform the curve (np.nanmean / np.nanstd handle this gracefully).
    Returns {metric_key: np.ndarray(n_seeds, n_epochs), "epoch": np.ndarray}.
    """
    seed_rows: list[list[dict]] = []
    for seed in seeds:
        path = Path(results_dir) / _run_name(method, K, N, seed) / "metrics.json"
        if not path.exists():
            continue
        with open(path) as f:
            seed_rows.append(json.load(f))
    if not seed_rows:
        return {}

    epochs_union = sorted({r["epoch"] for rows in seed_rows for r in rows})
    epoch_to_idx = {e: i for i, e in enumerate(epochs_union)}
    n_seeds = len(seed_rows)
    n_epochs = len(epochs_union)

    keys: set[str] = set()
    for rows in seed_rows:
        for r in rows:
            keys.update(r.keys())
    keys.discard("epoch")

    result: dict = {"epoch": np.array(epochs_union)}
    for key in keys:
        arr = np.full((n_seeds, n_epochs), np.nan, dtype=float)
        for si, rows in enumerate(seed_rows):
            for r in rows:
                ei = epoch_to_idx.get(r["epoch"])
                if ei is None:
                    continue
                v = r.get(key, np.nan)
                try:
                    arr[si, ei] = float(v)
                except (TypeError, ValueError):
                    pass
        result[key] = arr
    return result


# =============================================================================
# Figure 1 — Training dynamics (1x3)
# =============================================================================

def plot_figure1(
    results_dir: str, output_dir: str,
    K: int = 50, seeds: list[int] | None = None,
):
    """Fig 1: 1×3 panels over epochs — IoU@50 ↑, IoU@75 ↑, IoU@90 ↑."""
    if seeds is None:
        seeds = [42, 43, 44, 45]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # (method, N, style_key) — N is the variant the run was trained at.
    methods_to_plot = [
        ("tailrl",                16,   "tailrl_N16"),
        ("tailrl",                64,   "tailrl_N64"),
        ("tailrl",               256,   "tailrl_N256"),
        ("tailrl",              1024,   "tailrl_N1024"),
        ("grpo",              256,   "grpo_N256"),
        ("rloo",              256,   "rloo_N256"),
        ("reinforce",         256,   "reinforce_N256"),
        ("l1_centroid_match",  64,   "l1_centroid_match"),
        ("mse_centroid_match", 64,   "mse_centroid_match"),
        ("tailrl_population",     64,   "tailrl_population"),
    ]

    # (metric_key, ylabel) — IoU@τ thresholds, all higher-is-better.
    metric_panels = [
        ("val/iou_at_50", "IoU@50 ↑"),
        ("val/iou_at_75", "IoU@75 ↑"),
        ("val/iou_at_90", "IoU@90 ↑"),
    ]

    for ax, (metric_key, ylabel) in zip(axes, metric_panels):
        ax.set_facecolor(BG_COLOR)
        for method, N, style_key in methods_to_plot:
            data = load_metrics(results_dir, method, K, N, seeds)
            if not data or metric_key not in data:
                continue
            vals = data[metric_key]                    # (n_seeds, n_epochs), NaN-padded
            epochs = data["epoch"]
            mean = _smooth(np.nanmean(vals, axis=0))
            std = _smooth(np.nanstd(vals, axis=0))
            style = METHOD_STYLES.get(style_key, {})
            zorder = style.get("zorder", 2)
            plot_kw = dict(
                label=style.get("label", style_key),
                color=style.get("color", "black"),
                linestyle=style.get("ls", "-"),
                linewidth=style.get("lw", 1.5),
                alpha=style.get("alpha", 1.0),
                zorder=zorder,
            )
            if "dashes" in style:
                plot_kw["dashes"] = style["dashes"]
            ax.plot(epochs, mean, **plot_kw)
            ax.fill_between(epochs, mean - std, mean + std,
                            alpha=style.get("alpha", 1.0) * 0.15,
                            color=style.get("color", "black"),
                            zorder=zorder - 1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

    fig.tight_layout()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figure1_training_dynamics.pdf")
    fig.savefig(out / "figure1_training_dynamics.png")
    plt.close(fig)
    print(f"Figure 1 saved to {out}")


# =============================================================================
# Figure 2 — Gradient geometry scatter (1x4)
# =============================================================================

def _load_per_image(
    results_dir: str, method: str, K: int, N: int, seed: int, epoch: int,
) -> list[dict] | None:
    path = (Path(results_dir) / _run_name(method, K, N, seed)
            / "gradient_analysis" / f"gradient_per_image_epoch{epoch}.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def plot_figure2(
    results_dir: str, output_dir: str,
    epoch: int = 25, seed: int = 43, K: int = 50,
):
    """Fig 2: per-image ‖∇L‖ scatter vs expected-IoU error.

    Uses (1 - expected_iou) as x-axis — analog of MAE for localization.
    Panels: ordinal_ce, tailrl, grpo, reinforce.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)

    methods = [
        ("tailrl_population", 64,  EXP_IOU_LABEL, "black"),
        ("tailrl",            64,  "TailRL",         TAILRL_COLOR_BY_N[64]),
        ("grpo",          256,  "GRPO",        "#00897B"),
        ("reinforce",     256,  "REINFORCE",   "#2979FF"),
    ]

    for ax, (method, N, title, color) in zip(axes, methods):
        ax.set_facecolor(BG_COLOR)
        data = _load_per_image(results_dir, method, K, N, seed, epoch)
        if data is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
        else:
            x = np.array([d["expected_iou"] for d in data])
            y = np.array([d["gradient_l2_norm"] for d in data])
            ax.scatter(x, y, alpha=0.45, s=14, c=color, edgecolors="none")

        ax.set_xlabel(r"$\mathbb{E}[\mathrm{IoU}]$")
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(r"$\|\nabla_\theta \mathcal{L}\|_2$")
    fig.tight_layout()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figure2_gradient_geometry.pdf")
    fig.savefig(out / "figure2_gradient_geometry.png")
    plt.close(fig)
    print(f"Figure 2 saved to {out}")


# =============================================================================
# Figure 3 — Convergence as N → ∞ (1x2)
# =============================================================================

def plot_figure3(
    results_dir: str, output_dir: str,
    K: int = 50, seeds: list[int] | None = None,
):
    if seeds is None:
        seeds = [42, 43, 44, 45]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    N_values = [16, 64, 256, 1024]

    # Reference line: prefer J_ord (tailrl_population); fall back to ordinal_ce.
    ref = None
    ref_label = None
    for cand, label in [("tailrl_population", EXP_IOU_LABEL),
                        ("ordinal_ce", "Ordinal CE")]:
        d = load_metrics(results_dir, cand, K, 64, seeds)
        if d:
            ref, ref_label = d, label
            break

    panels = [
        ("val/iou_at_50", r"Final IoU@50 ↑", lambda v: v),
        ("val/iou_greedy", r"Final IoU$_{\mathrm{greedy}}$ ↑", lambda v: v),
    ]

    for ax, (metric_key, ylabel, transform) in zip(axes, panels):
        ax.set_facecolor(BG_COLOR)
        if ref and metric_key in ref:
            final_vals = transform(ref[metric_key][:, -1])
            ref_mean = float(np.nanmean(final_vals))
            ax.axhline(ref_mean, color="black", ls="--", lw=2,
                       dashes=(6, 6),
                       label=f"{ref_label}: {ref_mean:.3f}")

        means, stds, ns = [], [], []
        for N in N_values:
            data = load_metrics(results_dir, "tailrl", K, N, seeds)
            if not data or metric_key not in data:
                continue
            finals = transform(data[metric_key][:, -1])
            means.append(float(np.nanmean(finals)))
            stds.append(float(np.nanstd(finals)))
            ns.append(N)
            for v in finals:
                ax.scatter(N, v, c=TAILRL_COLOR_BY_N.get(N, "#E54040"),
                           alpha=0.65, s=40, zorder=3, edgecolors="none")

        if ns:
            colors = [TAILRL_COLOR_BY_N.get(n, "#E54040") for n in ns]
            for i in range(len(ns) - 1):
                ax.plot(ns[i:i+2], means[i:i+2], "-", color=colors[i+1],
                        lw=2.6, zorder=4)
            ax.scatter(ns, means, c=colors, s=80, zorder=5,
                       edgecolors="black", linewidths=0.6, label="TailRL")

        ax.set_xscale("log")
        ax.set_xlabel("N (rollouts per image)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(N_values)
        ax.set_xticklabels([str(n) for n in N_values])
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25, linewidth=0.5)

    fig.tight_layout()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figure3_convergence.pdf")
    fig.savefig(out / "figure3_convergence.png")
    plt.close(fig)
    print(f"Figure 3 saved to {out}")


# =============================================================================
# Figure 5 — Per-threshold survival + gradient (2x3)
# =============================================================================

def _load_per_threshold(
    results_dir: str, method: str, K: int, N: int, seed: int, epoch: int,
) -> list[dict] | None:
    path = (Path(results_dir) / _run_name(method, K, N, seed)
            / "gradient_analysis"
            / f"gradient_per_threshold_epoch{epoch}.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def plot_figure5(
    results_dir: str, output_dir: str,
    epoch: int = 50, seed: int = 43, K: int = 50,
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    f5_title, f5_label, f5_tick, f5_legend = 16, 15, 14, 12

    # Anchor for picking representative images: prefer J_ord
    # (tailrl_population), fall back to tailrl.
    anchor = None
    anchor_method = None
    for cand in ("tailrl_population", "tailrl"):
        anchor = _load_per_threshold(results_dir, cand, K, 64, seed, epoch)
        if anchor is not None:
            anchor_method = cand
            break
    if anchor is None:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "figure5_per_threshold.pdf")
        fig.savefig(out / "figure5_per_threshold.png")
        plt.close(fig)
        print(f"Figure 5 saved to {out} (no data)")
        return
    print(f"Figure 5: anchored on {anchor_method}")

    by_image = defaultdict(list)
    for d in anchor:
        by_image[d["image_idx"]].append(d)

    # Rank images by survival at the smallest threshold (higher = "easy").
    # Threshold floats ~ 0.05 .. 0.95; pick the entry with min threshold.
    surv_at_lowest: dict[int, float] = {}
    for idx, recs in by_image.items():
        r = min(recs, key=lambda r: r["threshold_d"])
        surv_at_lowest[idx] = r["survival_prob"]
    ranked = sorted(surv_at_lowest.items(), key=lambda kv: kv[1])
    n = len(ranked)
    if n == 0:
        print("Figure 5: empty anchor data")
        return
    hard_idx = ranked[0][0]
    med_idx = ranked[n // 2][0]
    easy_idx = ranked[-1][0]
    col_labels = [("Easy", easy_idx), ("Medium", med_idx), ("Hard", hard_idx)]

    methods = [
        ("tailrl",            64, TAILRL_COLOR_BY_N[64], "TailRL (N=64)"),
        ("tailrl_population", 64, "black",            EXP_IOU_LABEL),
    ]

    for col, (tag, img_idx) in enumerate(col_labels):
        ax_surv, ax_grad = axes[0, col], axes[1, col]
        ax_surv.set_facecolor(BG_COLOR)
        ax_grad.set_facecolor(BG_COLOR)
        for method, N, color, disp in methods:
            mdata = _load_per_threshold(results_dir, method, K, N, seed, epoch)
            if not mdata:
                continue
            rows = [r for r in mdata if r["image_idx"] == img_idx]
            if not rows:
                continue
            rows.sort(key=lambda r: r["threshold_d"])
            taus = [r["threshold_d"] for r in rows]
            surv = [r["survival_prob"] for r in rows]
            gnorm = [r["gradient_norm_at_threshold"] for r in rows]
            extra = dict(dashes=(6, 6), ls="--") if method == "tailrl_population" else {}
            ax_surv.plot(taus, surv, color=color, lw=2.5, label=disp, **extra)
            ax_grad.plot(taus, gnorm, color=color, lw=2.5, label=disp, **extra)

        ax_surv.set_title(f"{tag} image (idx={img_idx})", fontsize=f5_title)
        ax_surv.set_xlabel(r"IoU threshold $\tau$", fontsize=f5_label)
        ax_surv.set_ylabel(r"$P(\mathrm{IoU} > \tau)$", fontsize=f5_label)
        ax_surv.legend(fontsize=f5_legend)
        ax_surv.tick_params(labelsize=f5_tick)
        ax_surv.grid(True, alpha=0.3)

        ax_grad.set_xlabel(r"IoU threshold $\tau$", fontsize=f5_label)
        ax_grad.set_ylabel(r"$\|\nabla [-\log P(\mathrm{IoU} > \tau)]\|_2$",
                           fontsize=f5_label)
        ax_grad.legend(fontsize=f5_legend)
        ax_grad.tick_params(labelsize=f5_tick)
        ax_grad.grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figure5_per_threshold.pdf")
    fig.savefig(out / "figure5_per_threshold.png")
    plt.close(fig)
    print(f"Figure 5 saved to {out}")


# =============================================================================
# Figure 6 — Cosine vs G (1x2)
# =============================================================================

def _plot_cosine_panel(ax, cosine_file: Path, methods_styles: dict, ylabel: str):
    ax.set_facecolor(BG_COLOR)
    with open(cosine_file) as f:
        data = json.load(f)
    results = data["results"]
    for method, style in methods_styles.items():
        if method not in results:
            continue
        G_dict = results[method]
        Gs = sorted([int(g) for g in G_dict.keys()])
        means = np.array([G_dict[str(g)]["mean"] for g in Gs])
        stds = np.array([G_dict[str(g)]["std"] for g in Gs])
        ax.plot(Gs, means, label=style["label"], color=style["color"],
                linestyle=style["ls"], marker=style["marker"],
                linewidth=2.4, markersize=6)
        ax.fill_between(Gs, means - stds, np.minimum(means + stds, 1.0),
                        alpha=0.15, color=style["color"])
    ax.set_xscale("log")
    ax.set_xlabel("$G$ (rollouts per image)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.5)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25, linewidth=0.5)


def plot_figure6(
    results_dir: str, output_dir: str,
    epoch: int = 25, seed: int = 43, K: int = 50,
):
    """Cosine similarity vs G for each available reference. With only
    cross_entropy ref data populated (ordinal_ce checkpoints are missing for
    all seeds) we render a single panel using that ref."""
    refs = []
    population = (Path(results_dir) / _run_name("tailrl_population", K, 64, seed)
            / "gradient_analysis"
            / f"cosine_vs_G_epoch{epoch}_ref_tailrl_population.json")
    if population.exists():
        refs.append((population,
                     r"Cosine sim. with $\nabla " + EXP_IOU_LABEL.strip("$") + r"$",
                     f"vs. {EXP_IOU_LABEL} gradient"))
    pc = (Path(results_dir) / _run_name("cross_entropy", K, 64, seed)
          / "gradient_analysis"
          / f"cosine_vs_G_epoch{epoch}_ref_cross_entropy.json")
    if pc.exists():
        refs.append((pc, r"Cosine sim. with $\nabla \mathcal{J}_{\mathrm{CE}}$",
                     "vs. Plain CE gradient"))
    if not refs:
        print(f"Figure 6: no cosine files for epoch={epoch}, seed={seed}")
        return

    rl_styles = {
        "tailrl":          dict(label="TailRL",          color=TAILRL_COLOR_BY_N[64], ls="-", marker="o"),
        "binary_maxrl": dict(label="Binary MaxRL", color="#8c564b",          ls="-", marker="^"),
        "grpo":         dict(label="GRPO",         color="#00897B",          ls="-", marker="D"),
        "rloo":         dict(label="RLOO",         color="#9467bd",          ls="-", marker="s"),
        "reinforce":    dict(label="REINFORCE",    color="#2979FF",          ls="-", marker="v"),
    }

    fig, axes = plt.subplots(1, len(refs), figsize=(8 * len(refs), 5.5),
                             squeeze=False)
    for ax, (file, ylabel, title) in zip(axes[0], refs):
        _plot_cosine_panel(ax, file, rl_styles, ylabel)
        ax.set_title(title)

    fig.tight_layout()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figure6_cosine_vs_G.pdf")
    fig.savefig(out / "figure6_cosine_vs_G.png")
    plt.close(fig)
    print(f"Figure 6 saved to {out}")


# =============================================================================
# Figure 7 — Cosine grid (G × epoch) per reference loss
# =============================================================================

def plot_figure7(
    results_dir: str, output_dir: str,
    seed: int = 43, K: int = 50,
    epochs: list[int] | None = None,
    G_values: list[int] | None = None,
):
    epochs = epochs or [1, 10, 25]
    G_values = G_values or [4, 8, 16, 32, 64, 128, 256, 512, 1024]

    refs = [
        ("tailrl_population", "tailrl_population", EXP_IOU_LABEL),
        ("cross_entropy",       "cross_entropy",       r"Plain CE ($\mathcal{J}_{\mathrm{CE}}$)"),
    ]
    rl_methods = ["tailrl", "binary_maxrl", "grpo", "rloo", "reinforce"]
    rl_labels  = ["TailRL", "Binary MaxRL", "GRPO", "RLOO", "REINFORCE", "DG"]

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

    for ref_key, model_dir, ref_label in refs:
        all_data = {}
        for ep in epochs:
            suffix = f"_ref_{ref_key}"
            f = (Path(results_dir) / _run_name(model_dir, K, 64, seed)
                 / "gradient_analysis" / f"cosine_vs_G_epoch{ep}{suffix}.json")
            if f.exists():
                with open(f) as fh:
                    all_data[ep] = json.load(fh)
        if not all_data:
            print(f"Figure 7 ({ref_key}): no data, skipping")
            continue

        avail = sorted(all_data.keys())
        fig, axes = plt.subplots(2, len(rl_methods),
                                 figsize=(4 * len(rl_methods), 8))
        metric_keys = [("mean", "Cosine similarity"),
                       ("mag_ratio_mean", "Magnitude ratio")]

        for col, (method, mlabel) in enumerate(zip(rl_methods, rl_labels)):
            for row, (mkey, mtitle) in enumerate(metric_keys):
                ax = axes[row, col]
                grid = np.full((len(G_values), len(avail)), np.nan)
                for ei, ep in enumerate(avail):
                    res = all_data[ep].get("results", {})
                    if method not in res:
                        continue
                    for gi, G in enumerate(G_values):
                        v = res[method].get(str(G), {}).get(mkey, np.nan)
                        grid[gi, ei] = v
                if mkey == "mean":
                    vmin, vmax, cmap = -0.2, 1.0, "RdYlGn"
                else:
                    finite = grid[np.isfinite(grid)]
                    vmax = max(3.0, float(finite.max()) if finite.size else 3.0)
                    vmin, cmap = 0.0, "YlOrRd"
                im = ax.imshow(grid, aspect="auto", origin="lower",
                               vmin=vmin, vmax=vmax, cmap=cmap)
                for gi in range(len(G_values)):
                    for ei in range(len(avail)):
                        v = grid[gi, ei]
                        if not np.isnan(v):
                            color = "white" if ((mkey == "mean" and v < 0.3)
                                                or (mkey != "mean"
                                                    and v > vmax * 0.7)) else "black"
                            ax.text(ei, gi, f"{v:.2f}", ha="center", va="center",
                                    fontsize=7, color=color)
                ax.set_xticks(range(len(avail)))
                ax.set_xticklabels([str(e) for e in avail])
                ax.set_yticks(range(len(G_values)))
                ax.set_yticklabels([str(g) for g in G_values])
                if row == 0:
                    ax.set_title(mlabel, fontsize=11)
                if col == 0:
                    ax.set_ylabel(f"{mtitle}\nG", fontsize=10)
                if row == 1:
                    ax.set_xlabel("Epoch", fontsize=10)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"vs. {ref_label} gradient", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(out / f"figure7_grid_{ref_key}.pdf")
        fig.savefig(out / f"figure7_grid_{ref_key}.png")
        plt.close(fig)
        print(f"Figure 7 ({ref_key}) saved to {out}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default=paths.results_dir(),
                        help="Directory holding the per-run output directories "
                             "(default: $TAILRL_RESULTS_DIR).")
    parser.add_argument("--output_dir", default=paths.figures_dir(),
                        help="Where the figures are written "
                             "(default: $TAILRL_FIGURES_DIR).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45],
                        help="Seeds aggregated for multi-seed training-dynamics "
                             "figures (1, 3, val_iou_comparison).")
    parser.add_argument("--seed_analysis", type=int, default=43,
                        help="Single seed used for gradient/cosine analysis "
                             "figures (2, 5, 6, 7). Matches experiment5 which "
                             "uses a single seed for these figures.")
    parser.add_argument("--K", type=int, default=50)
    parser.add_argument("--figures", type=int, nargs="+",
                        default=[1, 2, 3, 5, 6, 7])
    args = parser.parse_args()

    if 1 in args.figures:
        plot_figure1(args.results_dir, args.output_dir,
                     K=args.K, seeds=args.seeds)
    if 2 in args.figures:
        plot_figure2(args.results_dir, args.output_dir,
                     epoch=25, seed=args.seed_analysis, K=args.K)
    if 3 in args.figures:
        plot_figure3(args.results_dir, args.output_dir,
                     K=args.K, seeds=args.seeds)
    if 5 in args.figures:
        plot_figure5(args.results_dir, args.output_dir,
                     epoch=25, seed=args.seed_analysis, K=args.K)
    if 6 in args.figures:
        plot_figure6(args.results_dir, args.output_dir,
                     epoch=25, seed=args.seed_analysis, K=args.K)
    if 7 in args.figures:
        plot_figure7(args.results_dir, args.output_dir,
                     seed=args.seed_analysis, K=args.K)


if __name__ == "__main__":
    main()
