#!/usr/bin/env python3
"""pass@k and best_reward@k curves with bootstrap CIs, from persisted per-sample eval records.

Consumes the parquet written by `eval_screenspot_pro.py --dump_samples`
(columns: idx, category, ui_type, sample_idx, reward, hit) and produces, per (arm, step):

  * unbiased pass@k          = 1 - C(n-c, k) / C(n, k)                       (Chen et al. 2021)
  * unbiased best_reward@k   = sum_i C(i-1, k-1)/C(n, k) * r_(i)             (ascending r)
  * percentile-bootstrap 95% CIs over ITEMS (1,000 resamples by default)

Both estimators are per-item and then averaged, so resampling items is the right bootstrap: it
carries the item-difficulty variance that dominates a 1,581-item benchmark. Resampling *samples*
would understate it badly.

Usage:
  python3 analysis/curves.py --glob 'evals/*/*.parquet' --out analysis/outputs
  python3 analysis/curves.py --glob '...' --out ... --ks 1 2 4 8 16 32 64 --bootstrap 1000

Path convention: .../<arm>/<step>.parquet, or any path containing 'arm=<x>' and 'step=<n>'.
Falls back to the file stem when it cannot parse one.
"""
import argparse
import glob as globmod
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------------
# estimators (kept identical in form to scripts/eval_screenspot_pro.py, which the tests pin)
# --------------------------------------------------------------------------------------------
def pass_at_k(n, c, k):
    """Unbiased P(at least one hit among k draws without replacement) from n samples with c hits."""
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def _logC(n, k):
    from scipy.special import gammaln

    n = np.asarray(n, dtype=float)
    out = np.full(n.shape, -np.inf)
    ok = (k >= 0) & (n >= k)
    out[ok] = gammaln(n[ok] + 1) - gammaln(k + 1) - gammaln(n[ok] - k + 1)
    return out


def best_reward_at_k(rewards, k):
    """Unbiased E[max reward over a uniformly random size-k subset] of the n observed samples.

    Weight of the i-th smallest reward is C(i-1, k-1)/C(n, k): the probability it is the maximum of
    the subset, i.e. that the other k-1 draws all come from the i-1 strictly smaller ones.
    Reduces to pass@k when the reward is 0/1.
    """
    r = np.sort(np.asarray(rewards, dtype=float))
    n = r.size
    k = min(k, n)
    i = np.arange(1, n + 1)
    w = np.exp(_logC(i - 1, k - 1) - _logC(np.array(n), k))
    return float(np.dot(w, r))


# --------------------------------------------------------------------------------------------
def per_item_curves(df, ks):
    """-> (item_ids, {k: pass@k per item}, {k: best_reward@k per item}). One row per item."""
    ids, pass_k, rew_k = [], defaultdict(list), defaultdict(list)
    for idx, g in df.groupby("idx", sort=True):
        rewards = g["reward"].to_numpy(dtype=float)
        n = rewards.size
        c = int(round(g["hit"].sum()))
        ids.append(int(idx))
        for k in ks:
            pass_k[k].append(pass_at_k(n, c, min(k, n)))
            rew_k[k].append(best_reward_at_k(rewards, k))
    return np.asarray(ids), {k: np.asarray(v) for k, v in pass_k.items()}, {k: np.asarray(v) for k, v in rew_k.items()}


def bootstrap_ci(per_item, n_boot=1000, alpha=0.05, seed=0):
    """Percentile bootstrap over ITEMS -> (mean, lo, hi)."""
    per_item = np.asarray(per_item, dtype=float)
    m = float(per_item.mean()) if per_item.size else float("nan")
    if per_item.size < 2 or n_boot <= 0:
        return m, m, m
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, per_item.size, size=(n_boot, per_item.size))
    means = per_item[draws].mean(axis=1)
    return m, float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def parse_arm_step(path):
    """-> (arm, step). Handles both `.../<arm>/<step>.parquet` and the concatenated `.../gos700/`
    directory style. step is -1 when unknown: NOT None, because pandas groupby drops NaN keys and
    a None here silently produces zero charts."""
    s = path.replace("\\", "/")
    # concatenated form first: a path component that is exactly <arm><digits>, e.g. "gos700"
    m = re.search(r"(^|/)(grpo|tailrl|rloo|base)(\d+)(/|$)", s)
    if m:
        return m.group(2), int(m.group(3))
    arm = next((a for a in ("grpo", "tailrl", "rloo", "base") if re.search(rf"(^|[/_=]){a}([/_.]|$)", s)), None)
    m = re.search(r"step[=_-]?(\d+)", s) or re.search(r"/(\d+)\.parquet$", s)
    step = int(m.group(1)) if m else -1
    if arm is None:
        arm = os.path.basename(os.path.dirname(s)) or "unknown"
    return arm, step


# --------------------------------------------------------------------------------------------
def _svg(series, ks, title, ylabel, path, logx=True):
    """Dependency-free SVG line chart with CI bands (matplotlib is not importable on the login node).

    series: {label: (means, los, his)} aligned with ks.
    """
    W, H, PAD = 860, 520, 78
    xs = np.log2(np.asarray(ks, dtype=float)) if logx else np.asarray(ks, dtype=float)
    x0, x1 = xs.min(), xs.max()
    lo = min(min(l) for _, l, _ in series.values())
    hi = max(max(h) for _, _, h in series.values())
    pad = 0.06 * (hi - lo or 1.0)
    y0, y1 = max(0.0, lo - pad), hi + pad
    px = lambda x: PAD + (x - x0) / ((x1 - x0) or 1) * (W - 1.5 * PAD)  # noqa: E731
    py = lambda y: H - PAD - (y - y0) / ((y1 - y0) or 1) * (H - 1.9 * PAD)  # noqa: E731
    # colour-blind-safe, distinguishable in greyscale by order
    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="Inter,Helvetica,Arial,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{W / 2}" y="30" text-anchor="middle" font-size="17" font-weight="600">{title}</text>']
    for f in np.linspace(0, 1, 6):
        y = y0 + f * (y1 - y0)
        o.append(f'<line x1="{PAD}" y1="{py(y):.1f}" x2="{W - 0.5 * PAD}" y2="{py(y):.1f}" stroke="#E6E6E6"/>')
        o.append(f'<text x="{PAD - 10}" y="{py(y) + 4:.1f}" text-anchor="end" font-size="12" fill="#555">{y:.3f}</text>')
    for k, x in zip(ks, xs):
        o.append(f'<text x="{px(x):.1f}" y="{H - PAD + 20}" text-anchor="middle" font-size="12" fill="#555">{k}</text>')
    o.append(f'<text x="{W / 2}" y="{H - 14}" text-anchor="middle" font-size="13" fill="#333">k (samples drawn)</text>')
    o.append(f'<text x="18" y="{H / 2}" text-anchor="middle" font-size="13" fill="#333" '
             f'transform="rotate(-90 18 {H / 2})">{ylabel}</text>')

    for i, (label, (m, l, h)) in enumerate(series.items()):
        c = colours[i % len(colours)]
        band = " ".join(f"{px(x):.1f},{py(v):.1f}" for x, v in zip(xs, h))
        band += " " + " ".join(f"{px(x):.1f},{py(v):.1f}" for x, v in zip(xs[::-1], np.asarray(l)[::-1]))
        o.append(f'<polygon points="{band}" fill="{c}" opacity="0.13"/>')
        pts = " ".join(f"{px(x):.1f},{py(v):.1f}" for x, v in zip(xs, m))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2.4"/>')
        for x, v in zip(xs, m):
            o.append(f'<circle cx="{px(x):.1f}" cy="{py(v):.1f}" r="3.4" fill="{c}"/>')
        ly = 56 + 20 * i
        o.append(f'<line x1="{W - 210}" y1="{ly}" x2="{W - 182}" y2="{ly}" stroke="{c}" stroke-width="2.4"/>')
        o.append(f'<text x="{W - 176}" y="{ly + 4}" font-size="12.5" fill="#333">{label}</text>')
    o.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(o))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="glob for per-sample parquet files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--by_category", action="store_true")
    args = ap.parse_args()

    files = sorted(globmod.glob(args.glob))
    if not files:
        raise SystemExit(f"no files matched {args.glob!r}")
    os.makedirs(args.out, exist_ok=True)

    # shards of the same (arm, step) hold DISJOINT items, so concatenating is exactly the full set
    groups = defaultdict(list)
    for path in files:
        arm, step = parse_arm_step(path)
        groups[(arm, step)].append(path)

    records = []
    for (arm, step), paths in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        n_items = df["idx"].nunique()
        n_samp = int(df.groupby("idx").size().min())
        print(f"{arm} step={step}: {n_items} items x >={n_samp} samples ({len(paths)} shard(s))")
        slices = {"overall": df}
        if args.by_category:
            for cat, g in df.groupby("category"):
                slices[f"cat:{cat}"] = g
        for slice_name, sdf in slices.items():
            _, pk, rk = per_item_curves(sdf, args.ks)
            for k in args.ks:
                pm, pl, ph = bootstrap_ci(pk[k], args.bootstrap, seed=args.seed)
                rm, rl, rh = bootstrap_ci(rk[k], args.bootstrap, seed=args.seed)
                records.append({
                    "arm": arm, "step": step, "slice": slice_name, "k": k,
                    "n_items": sdf["idx"].nunique(), "n_samples": n_samp,
                    "pass_at_k": pm, "pass_at_k_lo": pl, "pass_at_k_hi": ph,
                    "best_reward_at_k": rm, "best_reward_at_k_lo": rl, "best_reward_at_k_hi": rh,
                })

    out = pd.DataFrame.from_records(records)
    csv = os.path.join(args.out, "curves.csv")
    out.to_csv(csv, index=False)
    print("wrote", csv)

    # one chart per (step, slice) comparing the arms -- that is the comparison the experiment is about
    made = []
    for (step, slice_name), g in out.groupby(["step", "slice"], dropna=False):
        if g["arm"].nunique() < 2:
            continue
        safe = str(slice_name).replace(":", "_")
        for metric, ylabel in (("pass_at_k", "unbiased pass@k"), ("best_reward_at_k", "unbiased best reward@k")):
            series = {}
            for arm, ga in g.groupby("arm"):
                ga = ga.sort_values("k")
                series[arm] = (ga[metric].to_numpy(), ga[f"{metric}_lo"].to_numpy(), ga[f"{metric}_hi"].to_numpy())
            path = os.path.join(args.out, f"{metric}_step{step}_{safe}.svg")
            made.append(_svg(series, sorted(g["k"].unique()),
                             f"{ylabel} — step {step} — {slice_name}", ylabel, path))
    for p in made:
        print("wrote", p)

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"files": files, "groups": [f"{a}@{s}" for a, s in groups], "charts": made}, f, indent=2)


if __name__ == "__main__":
    main()
