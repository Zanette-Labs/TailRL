"""Val IoU@50 comparison (4 stylistic variants) — port of
experiment5_ordinal/plot_val_mae_comparison.py.

Y-axis is val/iou_at_50 (higher is better). All aesthetic variants mirror
NeurIPS / presentation / minimalist / grouped-zoom from experiment5.

Methods compared (with the N each was trained at):
  - TailRL:        N = 16, 64, 256, 1024
  - GRPO:       N = 256
  - RLOO:       N = 256
  - REINFORCE:  N = 256
  - l1_centroid_match (supervised)
  - mse_centroid_match (supervised)
  - tailrl_population (supervised, "ord-ce" variant)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments.imagenet_localization import paths


# $TAILRL_RESULTS_DIR / $TAILRL_FIGURES_DIR override these (see paths.py).
RESULTS_DIR = Path(paths.results_dir())
OUTPUT_DIR = Path(paths.figures_dir())
SEEDS = [42, 43, 44, 45]
K = 50

# Pale warm background (lighter than the original cream)
BG_COLOR = "#FFFDF2"

METRIC_KEY = "val/iou_at_50"
YLABEL = r"IoU@50 ↑"


# ── data loading (NaN-padded; does NOT truncate to shortest seed) ───────────

def load_metrics(method: str, N: int) -> dict:
    """Load every available seed for (method, N) and align to the union of
    epochs. Seeds that ran shorter contribute NaN past their end so longer
    seeds still inform the curve.
    """
    seed_rows = []
    for seed in SEEDS:
        path = RESULTS_DIR / f"{method}_K{K}_N{N}_seed{seed}" / "metrics.json"
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
    keys = set()
    for rows in seed_rows:
        for r in rows:
            keys.update(r.keys())
    keys.discard("epoch")
    out: dict = {"epoch": np.array(epochs_union)}
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
        out[key] = arr
    return out


EXP_IOU_LABEL = r"$J_{\mathrm{ord}}$"

# (method, N, display_name) — order matters for legend layout.
METHODS = [
    ("tailrl",                16,  "TailRL (N=16)"),
    ("tailrl",                64,  "TailRL (N=64)"),
    ("tailrl",               256,  "TailRL (N=256)"),
    ("tailrl",              1024,  "TailRL (N=1024)"),
    ("grpo",              256,  "GRPO (N=256)"),
    ("rloo",              256,  "RLOO (N=256)"),
    ("reinforce",         256,  "REINFORCE (N=256)"),
    ("l1_centroid_match",  64,  "L1"),
    ("mse_centroid_match", 64,  "MSE"),
    ("giou",               64,  "GIoU"),
    ("l1_giou",            64,  "L1+GIoU"),
    ("tailrl_population",     64,  EXP_IOU_LABEL),
]


def _smooth(y: np.ndarray, window: int = 3) -> np.ndarray:
    """Light moving-average smoothing that preserves NaNs (so partial seeds
    don't make the curve dive at the boundary)."""
    if window < 2 or len(y) < window:
        return y
    pad = window // 2
    padded = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        win = padded[i:i + window]
        m = ~np.isnan(win)
        if m.any():
            out[i] = win[m].mean()
    return out


def _load_all():
    series = []
    for method, N, name in METHODS:
        data = load_metrics(method, N)
        if not data or METRIC_KEY not in data:
            print(f"  [skip] {name}: no {METRIC_KEY} data")
            continue
        vals = data[METRIC_KEY]                              # (n_seeds, n_epochs)
        epochs = data["epoch"]
        mean = _smooth(np.nanmean(vals, axis=0))
        std = _smooth(np.nanstd(vals, axis=0))
        series.append((name, epochs, mean, std))
    return series


# ── shared style ───────────────────────────────────────────────────────────

ORD_DASHES = (6, 6)

# All-red TailRL palette, darker as G grows.
TAILRL_COLOR_BY_N = {
    16:   "#FF8B8B",
    64:   "#E54040",
    256:  "#A8001E",
    1024: "#5A0010",
}

_SHARED = {
    "TailRL (N=16)":         dict(color=TAILRL_COLOR_BY_N[16],   ls="-",  lw=2.7, alpha=1.0, zorder=6),
    "TailRL (N=64)":         dict(color=TAILRL_COLOR_BY_N[64],   ls="-",  lw=3.4, alpha=1.0, zorder=8),
    "TailRL (N=256)":        dict(color=TAILRL_COLOR_BY_N[256],  ls="-",  lw=4.0, alpha=1.0, zorder=10),
    "TailRL (N=1024)":       dict(color=TAILRL_COLOR_BY_N[1024], ls="-",  lw=4.4, alpha=1.0, zorder=11),
    "GRPO (N=256)":       dict(color="#00897B", ls="-",  lw=3.0, alpha=1.0, zorder=4),
    "RLOO (N=256)":       dict(color="#9467bd", ls="-",  lw=2.6, alpha=1.0, zorder=4),
    "REINFORCE (N=256)":  dict(color="#2979FF", ls="-",  lw=3.0, alpha=1.0, zorder=4),
    "L1":                 dict(color="#a04000", ls="--", lw=2.6, alpha=1.0, zorder=3),
    "MSE":                dict(color="#8c564b", ls="--", lw=2.6, alpha=1.0, zorder=3),
    "GIoU":               dict(color="#B8860B", ls="--", lw=2.6, alpha=1.0, zorder=3),
    "L1+GIoU":            dict(color="#556B2F", ls="--", lw=2.6, alpha=1.0, zorder=3),
    EXP_IOU_LABEL:        dict(color="black",   ls="--", lw=4.0, alpha=1.0, zorder=12,
                               dashes=ORD_DASHES),
}


def _draw(ax, series, overrides=None):
    for name, epochs, mean, std in series:
        s = dict(_SHARED.get(name, {}))
        if overrides and name in overrides:
            s.update(overrides[name])
        c  = s.get("color", "black")
        ls = s.get("ls", "-")
        lw = s.get("lw", 1.5)
        a  = s.get("alpha", 1.0)
        zo = s.get("zorder", 2)

        kw = dict(label=name, color=c, ls=ls, lw=lw, alpha=a, zorder=zo)
        if "dashes" in s:
            kw["dashes"] = s["dashes"]
        if "marker" in s:
            kw.update(marker=s["marker"], markevery=s.get("markevery", 5),
                      markersize=s.get("markersize", 5),
                      markeredgecolor="white", markeredgewidth=0.5)
        ax.plot(epochs, mean, **kw)
        ax.fill_between(epochs, mean - std, mean + std,
                        alpha=a * 0.15, color=c, zorder=zo - 1)


def _make_handle(name, overrides=None):
    s = dict(_SHARED.get(name, {}))
    if overrides and name in overrides:
        s.update(overrides[name])
    kw = dict(color=s.get("color", "k"), ls=s.get("ls", "-"),
              lw=s.get("lw", 1.5), alpha=s.get("alpha", 1.0), label=name)
    if "dashes" in s:
        kw["dashes"] = s["dashes"]
    if "marker" in s:
        kw["marker"] = s["marker"]
        kw["markersize"] = s.get("markersize", 5)
    return Line2D([0], [0], **kw)


def _add_bottom_legend(fig, ax, series, overrides, fontsize):
    short = {
        "REINFORCE (N=256)": "REINF.",
        "GRPO (N=256)": "GRPO",
        "RLOO (N=256)": "RLOO",
    }
    # Only legend the arms that actually produced a curve. Without this filter
    # every name listed below gets a handle even when load_metrics found no
    # results for it, so a not-yet-run arm shows up as a phantom entry.
    present = {name for name, *_ in series}
    row1 = [n for n in ["REINFORCE (N=256)", "GRPO (N=256)", "RLOO (N=256)",
                        "L1", "MSE", "GIoU", "L1+GIoU", EXP_IOU_LABEL]
            if n in present]
    row2 = [n for n in ["TailRL (N=16)", "TailRL (N=64)", "TailRL (N=256)", "TailRL (N=1024)"]
            if n in present]

    h1 = []
    for n in row1:
        h = _make_handle(n, overrides)
        h.set_label(short.get(n, n))
        h1.append(h)
    h2 = [_make_handle(n, overrides) for n in row2]

    if h1:
        fig.legend(handles=h1, loc="upper center", bbox_to_anchor=(0.5, 0.13),
                   ncol=len(row1), frameon=False, fontsize=fontsize,
                   handlelength=1.5, columnspacing=0.6)
    if h2:
        fig.legend(handles=h2, loc="upper center", bbox_to_anchor=(0.5, 0.06),
                   ncol=len(row2), frameon=False, fontsize=fontsize,
                   handlelength=1.5, columnspacing=0.6)


def _style_ax(ax):
    ax.set_facecolor(BG_COLOR)
    ax.set_box_aspect(1 / 1.25)


# ── V1: NeurIPS ─────────────────────────────────────────────────────────────

def plot_v1_neurips(series):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 11, "axes.labelsize": 24,
        "xtick.labelsize": 16.8, "ytick.labelsize": 16.8,
    })
    fig, ax = plt.subplots(figsize=(10, 7.5))
    _style_ax(ax)
    _draw(ax, series)
    ax.set_xlabel("Epoch", labelpad=10)
    ax.set_ylabel(YLABEL, labelpad=10)
    _add_bottom_legend(fig, ax, series, None, fontsize=15)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_xlim(left=0)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.30)
    return fig


# ── V2: bold presentation ───────────────────────────────────────────────────

def plot_v2_presentation(series):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 14, "axes.labelsize": 32,
        "xtick.labelsize": 21.7, "ytick.labelsize": 21.7,
    })
    ov = {EXP_IOU_LABEL: dict(lw=3.5)}
    fig, ax = plt.subplots(figsize=(11, 9.5))
    _style_ax(ax)
    _draw(ax, series, overrides=ov)
    ax.set_xlabel("Epoch", fontweight="bold", labelpad=10)
    ax.set_ylabel(YLABEL, fontweight="bold", labelpad=10)
    _add_bottom_legend(fig, ax, series, ov, fontsize=17)
    ax.set_xlim(left=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(direction="out", length=5)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.30)
    return fig


# ── V3: minimalist ──────────────────────────────────────────────────────────

def plot_v3_minimalist(series):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.labelsize": 22,
        "xtick.labelsize": 15.4, "ytick.labelsize": 15.4,
    })
    ov = {
        "REINFORCE (N=256)":   dict(marker="v", markevery=5, markersize=6),
        "GRPO (N=256)":        dict(marker="D", markevery=5, markersize=6),
        "RLOO (N=256)":        dict(marker="P", markevery=5, markersize=6),
        "TailRL (N=16)":          dict(marker="^", markevery=5, markersize=6),
        "TailRL (N=64)":          dict(marker="s", markevery=5, markersize=6),
        "TailRL (N=256)":         dict(marker="o", markevery=5, markersize=6),
        "TailRL (N=1024)":        dict(marker="X", markevery=5, markersize=6),
        "L1":                  dict(marker="x", markevery=5, markersize=7),
        "MSE":                 dict(marker="+", markevery=5, markersize=7),
        EXP_IOU_LABEL:         dict(marker="*", markevery=5, markersize=8),
    }
    fig, ax = plt.subplots(figsize=(9, 6.5))
    _style_ax(ax)
    _draw(ax, series, overrides=ov)
    ax.set_xlabel("Epoch", labelpad=10)
    ax.set_ylabel(YLABEL, labelpad=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.8)
    ax.yaxis.grid(True, alpha=0.2, linewidth=0.5, linestyle=":")
    ax.xaxis.grid(False)
    _add_bottom_legend(fig, ax, series, ov, fontsize=13)
    ax.set_xlim(left=0)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.32)
    return fig


# ── V4: grouped + inset zoom ────────────────────────────────────────────────

def plot_v4_grouped(series):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 11, "axes.labelsize": 24,
        "xtick.labelsize": 16.8, "ytick.labelsize": 16.8,
    })
    fig, ax = plt.subplots(figsize=(11, 8.5))
    _style_ax(ax)
    _draw(ax, series)
    ax.set_xlabel("Epoch", labelpad=10)
    ax.set_ylabel(YLABEL, labelpad=10)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_xlim(left=0)
    _add_bottom_legend(fig, ax, series, None, fontsize=16)

    axins = ax.inset_axes([0.55, 0.10, 0.40, 0.40])
    axins.set_facecolor(BG_COLOR)
    zoom_lo = int(0.5 * ax.get_xlim()[1])
    for name, epochs, mean, std in series:
        s = dict(_SHARED.get(name, {}))
        mask = epochs >= zoom_lo
        if mask.sum() == 0:
            continue
        inset_kw = dict(color=s["color"], ls=s["ls"], lw=s["lw"] * 0.8,
                        alpha=s.get("alpha", 1.0))
        if "dashes" in s:
            inset_kw["dashes"] = s["dashes"]
        axins.plot(epochs[mask], mean[mask], **inset_kw)
        axins.fill_between(epochs[mask], (mean - std)[mask], (mean + std)[mask],
                           alpha=s.get("alpha", 1.0) * 0.12, color=s["color"])
    axins.set_xlim(zoom_lo, ax.get_xlim()[1])
    axins.set_title(f"Epochs {zoom_lo}–{int(ax.get_xlim()[1])} (zoom)", fontsize=12)
    axins.tick_params(labelsize=10)
    axins.grid(True, alpha=0.2, linewidth=0.4)
    ax.indicate_inset_zoom(axins, edgecolor="0.5", linewidth=0.8)

    fig.tight_layout(); fig.subplots_adjust(bottom=0.28)
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data...")
    series = _load_all()
    if not series:
        print("No data found!")
        return
    print(f"Loaded {len(series)} methods: {[s[0] for s in series]}")

    versions = [
        ("v1_neurips",       plot_v1_neurips),
        ("v2_presentation",  plot_v2_presentation),
        ("v3_minimalist",    plot_v3_minimalist),
        ("v4_grouped_zoom",  plot_v4_grouped),
    ]
    for tag, fn in versions:
        fig = fn(series)
        for ext in ("png", "pdf"):
            out = OUTPUT_DIR / f"val_iou_comparison_{tag}.{ext}"
            fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        print(f"  Saved val_iou_comparison_{tag}.{{png,pdf}}")
    print("Done.")


if __name__ == "__main__":
    main()
