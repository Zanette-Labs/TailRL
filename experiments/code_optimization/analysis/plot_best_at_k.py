"""Draw the best-of-k and pass@k curves from a results JSON.

    python3 analysis/plot_best_at_k.py                          # the shipped results
    python3 analysis/plot_best_at_k.py --results "$CODEOPT_EVAL_DIR/metrics/summary.json"

Accepts either shape: the `paper_results/best_at_k_n4096.json` written for release, or
the `summary.json` the aggregator writes at the end of an evaluation run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLOR = {"tailrl": "#d62728", "grpo": "#1f77b4", "rloo": "#2ca02c", "base": "#7f7f7f"}
LABEL = {"tailrl": "TailRL", "grpo": "GRPO", "rloo": "RLOO", "base": "untrained"}


def load(path: Path):
    """Return (ks, [(method, step, best_at_k, pass_at_k)], n_rollouts)."""
    d = json.loads(path.read_text())
    if "estimators" in d:                      # the released results file
        ks = d["k_values"]
        n = d["n_rollouts_per_prompt"]
        rows = [(v["method"], v["checkpoint_step"], v["best_reward_at_k"], v["pass_at_k"])
                for v in d["estimators"].values()]
        return ks, rows, n
    # the aggregator's summary.json
    first = next(iter(d.values()))
    ks = sorted((int(k) for k in first["best_at_k"]))
    n = max(v["n_completions"] for v in d.values())
    rows = [(v["arm"], int(v["step"]), v["best_at_k"], v["pass_at_k"]) for v in d.values()]
    return ks, rows, n


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(here.parent / "paper_results" / "best_at_k_n4096.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_prompts", type=int, default=878)
    a = ap.parse_args()

    src = Path(a.results)
    ks, rows, n = load(src)
    out = Path(a.out or src.with_suffix(".png"))

    order = {"tailrl": 0, "grpo": 1, "rloo": 2, "base": 3}
    rows.sort(key=lambda r: order.get(r[0], 9))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for method, step, bestk, passk in rows:
        lab = LABEL.get(method, method) + (f" @ step {step}" if step else "")
        c = COLOR.get(method)
        axes[0].plot(ks, [bestk[str(k)] for k in ks], marker="o", color=c, label=lab)
        axes[1].plot(ks, [passk[str(k)] for k in ks], marker="o", color=c, label=lab)

    # A reward of exactly 1.0 means the rollout reproduced the source program: correct,
    # and not one cycle faster. It is the line the collapsed arms sit on.
    axes[0].axhline(1.0, color="k", lw=0.8, ls=":")
    axes[0].text(ks[0] * 1.05, 1.03, "no speedup (source reproduced)", fontsize=8)

    for ax, title, ylab in ((axes[0], "Best-of-$k$ speedup", "E[max gem5 speedup over $k$ rollouts]"),
                            (axes[1], "Pass@$k$", "pass@$k$")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks); ax.set_xticklabels(ks)
        ax.set_xlabel("$k$ (inference rollouts)"); ax.set_ylabel(ylab); ax.set_title(title)
        ax.grid(alpha=0.3); ax.legend(fontsize=9)

    fig.suptitle(f"PIE C++ runtime optimization — {a.n_prompts} held-out programs, "
                 f"$n={n}$ rollouts/prompt, unbiased estimators, gem5 reward",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
