#!/usr/bin/env python3
"""Final pass@k / best_reward@k results -> one self-describing JSON per line.

  ./incontainer.sh python3 analysis/dump_bestk_json.py \
      --set 3B:26448:$EVAL_OUT/bestk_3b_final \
      --out results_3b_final.json

Emits, for every arm, both estimators at every k, OVERALL and broken down by ScreenSpot-Pro
`category` and by `ui_type`, each with a percentile-bootstrap 95% CI over items, plus paired
arm-vs-arm differences.

Three things are deliberate:

* Per-slice CIs are computed on that slice's own items. A CI taken over all items and drawn against
  a single category's curve describes a different population than the number it sits next to --
  which looks entirely plausible and is wrong. (Same bug, once, in the HTML renderer.)

* Every arm is checked to cover the SAME item set before anything is computed. Arms measured on
  different subsets still produce three tidy curves; nothing downstream can detect it.

* The `protocol` block is copied out of the shard results, not re-derived from flags here. It is the
  record of what actually produced the numbers -- reward module, format_weight, temperature, top_p,
  top_k, n. A `format_weight` mismatch went unnoticed for a whole eval round before that block
  existed.
"""
import argparse
import glob
import importlib.util
import json
import os

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "curves", os.path.join(os.path.dirname(os.path.abspath(__file__)), "curves.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

ARMS = ("tailrl", "grpo", "rloo")


def _slice_metrics(df, ks, n_boot, seed):
    """-> {pass_at_k, best_reward_at_k, n_items} with [mean, lo, hi] per k."""
    _, pk, rk = C.per_item_curves(df, ks)
    out = {"n_items": int(df["idx"].nunique()), "pass_at_k": {}, "best_reward_at_k": {}}
    for k in ks:
        m, lo, hi = C.bootstrap_ci(pk[k], n_boot, seed=seed)
        out["pass_at_k"][str(k)] = {"mean": m, "ci_lo": lo, "ci_hi": hi}
        m, lo, hi = C.bootstrap_ci(rk[k], n_boot, seed=seed)
        out["best_reward_at_k"][str(k)] = {"mean": m, "ci_lo": lo, "ci_hi": hi}
    return out, pk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="LABEL:STEP:DIR (DIR holds <arm>/samples.shard*.parquet)")
    ap.add_argument("--base", default=None,
                    help="dir holding base/samples.shard*.parquet for the UNTRAINED model. Included as "
                         "a fourth arm and as the reference for arm-minus-base differences -- without "
                         "it there is no way to tell whether RL expanded coverage or narrowed it.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    label, step, base = args.set.split(":", 2)

    arms = list(ARMS) + (["base"] if args.base else [])
    data, protocol = {}, None
    for arm in arms:
        if arm == "base":
            files = sorted(glob.glob(os.path.join(args.base, "base", "samples.shard*.parquet")))
            if not files:
                raise SystemExit(f"ABORT: no per-sample parquets under {args.base}/base")
            data[arm] = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            continue
        files = sorted(glob.glob(os.path.join(base, arm, "samples.shard*.parquet")))
        if not files:
            raise SystemExit(f"ABORT: no per-sample parquets for {arm} under {base}")
        data[arm] = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        js = sorted(glob.glob(os.path.join(base, arm, "eval_results.shard*.json")))
        if js and protocol is None:
            protocol = (json.load(open(js[0])) or {}).get("scoring")

    # every arm must cover the same items, or the comparison is not a comparison
    sets = [set(df["idx"].unique()) for df in data.values()]
    if len(set(map(frozenset, sets))) != 1:
        raise SystemExit("ABORT: arms cover DIFFERENT item sets: %s" % [len(s) for s in sets])
    n_samples = {a: int(df.groupby("idx").size().min()) for a, df in data.items()}
    if len(set(n_samples.values())) != 1:
        raise SystemExit(f"ABORT: arms have different samples/item: {n_samples}")

    result = {
        "line": label, "step": int(step), "benchmark": "ScreenSpot-Pro",
        "n_items": len(sets[0]), "n_samples_per_item": next(iter(n_samples.values())),
        "ks": args.ks, "bootstrap_resamples": args.bootstrap,
        "ci": "percentile bootstrap 95% over ITEMS (item difficulty dominates the variance)",
        "estimators": {
            "pass_at_k": "unbiased 1 - C(n-c,k)/C(n,k)  (Chen et al. 2021)",
            "best_reward_at_k": "unbiased sum_i C(i-1,k-1)/C(n,k) * r_(i); reward scale [0, 2.5]",
        },
        "protocol": protocol,
        "arms": {},
    }

    per_item_256 = {}
    for arm, df in data.items():
        entry = {}
        entry["overall"], pk = _slice_metrics(df, args.ks, args.bootstrap, args.seed)
        per_item_256[arm] = (sorted(df["idx"].unique()), pk[max(args.ks)])
        for field, key in (("category", "by_category"), ("ui_type", "by_ui_type")):
            if field not in df.columns:
                continue
            entry[key] = {}
            for name, g in df.groupby(field):
                entry[key][str(name)], _ = _slice_metrics(g, args.ks, args.bootstrap, args.seed)
        result["arms"][arm] = entry

    kmax = max(args.ks)
    result["paired_differences"] = {"k": kmax, "metric": "pass_at_k", "pairs": {}}
    ids = sorted(sets[0])
    pairs = [("tailrl", "grpo"), ("tailrl", "rloo"), ("rloo", "grpo")]
    if args.base:
        # arm - base is the control that decides whether RL EXPANDED or NARROWED coverage
        pairs += [("tailrl", "base"), ("rloo", "base"), ("grpo", "base")]
    for x, y in pairs:
        ax = dict(zip(*per_item_256[x])); ay = dict(zip(*per_item_256[y]))
        m, lo, hi = C.bootstrap_ci(np.array([ax[i] - ay[i] for i in ids]), args.bootstrap, seed=args.seed)
        result["paired_differences"]["pairs"][f"{x}_minus_{y}"] = {
            "mean": m, "ci_lo": lo, "ci_hi": hi, "significant": (lo > 0) or (hi < 0)}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out}  ({label} @ {step}: {result['n_items']} items x "
          f"{result['n_samples_per_item']} samples, {len(result['arms'])} arms)")
    cats = sorted(result["arms"]["tailrl"].get("by_category", {}))
    print(f"  categories: {', '.join(cats) if cats else '(none)'}")
    for p, v in result["paired_differences"]["pairs"].items():
        print("  %-16s %+.4f [%+.4f, %+.4f] %s" % (p, v["mean"], v["ci_lo"], v["ci_hi"],
                                                   "significant" if v["significant"] else "n.s."))


if __name__ == "__main__":
    main()
