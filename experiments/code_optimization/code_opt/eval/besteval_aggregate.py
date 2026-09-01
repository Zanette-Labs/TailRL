"""Aggregate the sharded best_reward@k test eval into per-arm curves + a plot.

Contiguous prompt-sharding means each prompt's full n completions live in ONE
shard, so each shard already holds that prompt's full-n best@k. This just
averages per-prompt best@k / pass@k across all 878 test prompts per arm, checks
coverage, and plots TailRL vs GRPO vs RLOO. Flags incompleteness rather than
reporting partial numbers as final.

Usage:
  python -m code_opt.eval.besteval_aggregate \
    --eval_root "$EVAL_ROOT" \
    --runs tailrl_step300 grpo_step500 rloo_step500 base_step0 \
    --k_values 1 4 16 64 256 1024 --expect_prompts 878
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from code_opt.eval.post_hoc_eval import best_at_k, pass_at_k  # noqa: E402

K_DEFAULT = [1, 4, 16, 64, 256]
COLOR = {"tailrl": "#d62728", "grpo": "#1f77b4", "rloo": "#2ca02c", "base": "#7f7f7f"}
LABEL = {"tailrl": "TailRL", "grpo": "GRPO", "rloo": "RLOO", "base": "base model"}


def resolve_run_dir(root: Path, run: str) -> Path:
    """Cells write to <MODEL_TAG>_<run> (e.g. qwen3_1.7b_tailrl_step300); accept both the
    bare name and the model-tag-prefixed dir."""
    d = root / run
    if d.is_dir():
        return d
    cands = sorted(glob.glob(str(root / f"*_{run}")))
    return Path(cands[0]) if cands else d


def load_run(run_dir: Path):
    recs, shards_done, shards_expected = [], 0, None
    for d in sorted(glob.glob(str(run_dir / "shard_*.done.json"))):
        shards_done += 1
        info = json.loads(open(d).read())
        if shards_expected is None and "n_shards" in info:
            shards_expected = info["n_shards"]
    seen = set()
    for f in sorted(glob.glob(str(run_dir / "shard_*.jsonl"))):
        if f.endswith(".gens.jsonl"):
            continue  # rollout-dump sidecars, not score records
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r.get("problem_id"), r.get("src_id"))
            if key in seen:
                continue  # defensive dedup (of16/of128 overlap, re-run races)
            seen.add(key)
            recs.append(r)
    return recs, shards_done, shards_expected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="<arm>_step<step> subdir names under eval_root")
    ap.add_argument("--k_values", type=int, nargs="+", default=K_DEFAULT)
    ap.add_argument("--expect_prompts", type=int, default=878)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.eval_root)
    out = Path(args.out or (root / "besteval_summary"))
    out.mkdir(parents=True, exist_ok=True)
    ks = args.k_values

    summary = {}
    for run in args.runs:
        arm = run.split("_step")[0]
        step = run.split("_step")[1] if "_step" in run else "?"
        recs, sd, se = load_run(resolve_run_dir(root, run))
        npr = len(recs)
        complete = (npr >= args.expect_prompts) and (se is None or sd >= se)
        # Recompute from the raw per-prompt `scores` (source of truth) rather than trusting
        # the shard's precomputed metrics — robust to any shard-side estimator issue.
        bestk = {str(k): mean([best_at_k(r["scores"], k) for r in recs]) for k in ks} if recs else {}
        passk = {str(k): mean([pass_at_k(r["n"], r["n_pass"], k) for r in recs]) for k in ks} if recs else {}
        n_comp = recs[0]["n"] if recs else 0

        # per-problem breakdown: EVERYTHING per problem (all k + full reward/speedup
        # distributions + improvement fractions + the raw per-completion scores) -> own
        # file per arm. All derived from the stored `scores` (reward = speedup if the
        # rollout passes ALL usable cases else 0), so acc[i]=(score>0), correct-speedup=score.
        def _pct(srt, q):
            if not srt:
                return 0.0
            return srt[min(len(srt) - 1, max(0, int(round(q * (len(srt) - 1)))))]
        per_problem = []
        for r in recs:
            sc = r["scores"]; m = len(sc); srt = sorted(sc)
            corr = sorted(x for x in sc if x > 0.0)          # correct-only speedups
            rmean = sum(sc) / m
            std = (sum((x - rmean) ** 2 for x in sc) / m) ** 0.5
            per_problem.append({
                "problem_id": r["problem_id"], "src_id": r["src_id"],
                "global_idx": r.get("global_idx"), "n_usable": r.get("n_usable"),
                "n": r["n"], "n_pass": r["n_pass"], "pass_rate": r["n_pass"] / m,
                "best_at_k": {str(k): best_at_k(sc, k) for k in ks},
                "pass_at_k": {str(k): pass_at_k(r["n"], r["n_pass"], k) for k in ks},
                "reward": {"mean": rmean, "std": std, "min": min(sc), "max": max(sc),
                           "p10": _pct(srt, .10), "p25": _pct(srt, .25), "p50": _pct(srt, .50),
                           "p75": _pct(srt, .75), "p90": _pct(srt, .90), "p99": _pct(srt, .99)},
                "correct_speedup": {"n": len(corr),
                                    "mean": (sum(corr) / len(corr)) if corr else 0.0,
                                    "max": (corr[-1] if corr else 0.0),
                                    "p50": _pct(corr, .50), "p90": _pct(corr, .90),
                                    "p99": _pct(corr, .99)},
                "frac_ge": {"1x": sum(x > 1.0 for x in sc) / m,
                            "1.1x": sum(x >= 1.1 for x in sc) / m,
                            "2x": sum(x >= 2.0 for x in sc) / m,
                            "5x": sum(x >= 5.0 for x in sc) / m},
                "scores": sc,                                 # full per-completion reward array
            })
        per_problem.sort(key=lambda d: -d["best_at_k"][str(ks[-1])])   # most headroom first
        # compact (the 4096-long scores arrays make indent=1 explode to millions of lines)
        (out / f"per_problem_{run}.json").write_text(
            json.dumps(per_problem, separators=(",", ":")))

        summary[run] = {"arm": arm, "step": step, "n_prompts": npr,
                        "expect_prompts": args.expect_prompts,
                        "shards_done": sd, "shards_expected": se,
                        "complete": complete, "n_completions": n_comp,
                        "best_at_k": bestk, "pass_at_k": passk}
        flag = "" if complete else "  <-- INCOMPLETE (partial; not final)"
        print(f"\n=== {run} (n={n_comp}, prompts={npr}/{args.expect_prompts}, "
              f"shards={sd}/{se}){flag} ===")
        print("  best_reward@k: " + "  ".join(f"k={k}:{bestk.get(str(k),0):.3f}" for k in ks))
        print("  pass@k:        " + "  ".join(f"k={k}:{passk.get(str(k),0):.3f}" for k in ks))

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[summary] -> {out}/summary.json")

    # plot best@k + pass@k
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        n_any = max((s["n_completions"] for s in summary.values()), default=0)
        for run, s in summary.items():
            if not s["best_at_k"]:
                continue
            c = COLOR.get(s["arm"], None)
            lab = LABEL.get(s["arm"], s["arm"])
            if s["step"] not in ("0", "?"):
                lab += f" @ step {s['step']}"
            if not s["complete"]:
                lab += " (partial)"
            ls = "-" if s["complete"] else "--"
            axes[0].plot(ks, [s["best_at_k"][str(k)] for k in ks], marker="o", color=c, ls=ls, label=lab)
            axes[1].plot(ks, [s["pass_at_k"][str(k)] for k in ks], marker="o", color=c, ls=ls, label=lab)
        axes[0].axhline(1.0, color="k", lw=0.8, ls=":")
        for ax, t, yl in ((axes[0], "Best-of-$k$ speedup", "E[max gem5 speedup over $k$ rollouts]"),
                          (axes[1], "Pass@$k$ (correctness)", "pass@$k$")):
            ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks)
            ax.set_xlabel("$k$ (inference rollouts)"); ax.set_ylabel(yl); ax.set_title(t)
            ax.grid(alpha=0.3); ax.legend()
        fig.suptitle(f"PIE C++ runtime optimization — {args.expect_prompts} held-out programs, "
                     f"$n={n_any}$ rollouts/prompt, unbiased estimators, gem5 reward",
                     fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        p = out / "besteval_test.png"; fig.savefig(p, dpi=130, bbox_inches="tight")
        print(f"[plot] -> {p}")
    except Exception as e:
        print(f"[plot] skipped: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
