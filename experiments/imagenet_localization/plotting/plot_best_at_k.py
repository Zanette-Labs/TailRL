"""Plot best@k vs k for a set of runs produced by eval_best_at_k.py.

Reads the merged JSON ({"runs": [{"label", "ks", "best_at_k"}, ...]}) and draws
one curve per run: best@k (y) vs k (log x). Vanilla-IoU runs are solid, the
percentile-reward runs are dashed; colour groups by estimator (tailrl/grpo/rloo).

    python -m experiments.imagenet_localization.plotting.plot_best_at_k \
        --in_json bestk.json --out_png bestk.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_COLORS = {"tailrl": "tab:green", "grpo": "tab:blue", "rloo": "tab:orange"}


def _estimator_of(label: str) -> str:
    for est in ("tailrl", "grpo", "rloo"):
        if label.startswith(est) or f"_{est}_" in label or label == est:
            return est
    return "other"


def main() -> None:
    p = argparse.ArgumentParser(description="Plot best@k vs k.")
    p.add_argument("--in_json", required=True)
    p.add_argument("--out_png", required=True)
    p.add_argument("--title", default="Unbiased best@k IoU on val")
    args = p.parse_args()

    with open(args.in_json) as f:
        data = json.load(f)
    runs = data["runs"]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for r in runs:
        label = r["label"]
        est = _estimator_of(label)
        is_pct = "percentile" in label
        color = _COLORS.get(est, "tab:gray")
        style = "--" if is_pct else "-"
        marker = "o" if is_pct else "s"
        ax.plot(r["ks"], r["best_at_k"], style, marker=marker, color=color,
                label=label, linewidth=2, markersize=6)

    ax.set_xscale("log", base=2)
    ax.set_xticks(runs[0]["ks"])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("k (number of rollouts)")
    ax.set_ylabel("best@k IoU (unbiased)")
    ax.set_title(f"{args.title}  (M={data.get('M', '?')} samples/image)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
