#!/usr/bin/env python3
"""pass@k / best_reward@k over the whole SE-GUI checkpoint ladder -> one JSON (+ a flat CSV).

  ./incontainer.sh python3 analysis/dump_ladder_json.py \
      --sweep $EVAL_OUT/bestk_sweep_n512 \
      --final 3b:$EVAL_OUT/bestk_3b_k1024 \
      --final 7b:$EVAL_OUT/bestk_7b_k1024 \
      --base  3b:$EVAL_OUT/bestk_3b_base \
      --base  7b:$EVAL_OUT/bestk_7b_base \
      --out results_ladder_n512.json

Emits, for every (line, arm, step) and for the untrained base model at step 0, both estimators at
every k <= 128, OVERALL and split by `category` and `ui_type`, each with a percentile-bootstrap 95%
CI over items.

Four things are deliberate:

* `--final` and `--base` sources were generated at n=4096 under the identical protocol. They are
  SUBSAMPLED to n, not used whole: an arm measured at 4096 samples has a visibly tighter best@k
  than the same arm at 512 (the estimator's variance falls with n even though its mean does not),
  and a ladder whose last rung is measured differently from the other twenty produces an upturn at
  the right edge that is pure methodology. The subsample is `sample_idx < n`, which is deterministic
  and needs no seed because samples are i.i.d. draws from one policy.

* Per-slice CIs are computed on that slice's own items, never sliced out of an overall CI.

* An incomplete shard set is refused rather than silently concatenated. A 3-of-4 merge is a result
  over 3/4 of the benchmark that looks exactly like a full one.

* Every arm within a line must cover the SAME items at the SAME step. Arms measured on different
  subsets still produce tidy curves and nothing downstream can detect it.

The estimators are vectorised here (per-(n,k) weight tables reused across items) rather than looped
per item as in curves.py -- 128 checkpoints x 8 slices x 1581 items is ~20 minutes the slow way.
`--verify` asserts the fast path reproduces curves.py exactly before anything is written.
"""
import argparse
import glob
import importlib.util
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "curves", os.path.join(os.path.dirname(os.path.abspath(__file__)), "curves.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


# ------------------------------------------------------------------------------------------------
# vectorised estimators. Both are functions of (n, k) times per-item data, and n is constant across
# every item in this sweep, so the (n, k) part is computed once and reused.
# ------------------------------------------------------------------------------------------------
class Estimators:
    def __init__(self, n, ks):
        self.n, self.ks = n, list(ks)
        # pass@k depends on the item ONLY through its hit count c -> a length-(n+1) lookup per k.
        self.pass_tab = {k: np.array([C.pass_at_k(n, c, min(k, n)) for c in range(n + 1)])
                         for k in self.ks}
        # best@k weights the i-th smallest reward by C(i-1,k-1)/C(n,k); independent of the rewards.
        i = np.arange(1, n + 1)
        self.w = np.stack([np.exp(C._logC(i - 1, min(k, n) - 1) - C._logC(np.array(n), min(k, n)))
                           for k in self.ks], axis=1)               # (n, n_ks)

    def curves(self, df):
        """-> (item_ids, {k: pass@k per item}, {k: best@k per item}) for one slice."""
        g = df.sort_values(["idx", "reward"])
        ids = g["idx"].to_numpy()
        first = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
        counts = np.diff(np.r_[first, ids.size])
        if not np.all(counts == self.n):
            raise SystemExit(f"ABORT: items with != {self.n} samples: "
                             f"{sorted(set(counts.tolist()) - {self.n})}")
        item_ids = ids[first]
        R = g["reward"].to_numpy(dtype=float).reshape(-1, self.n)   # already ascending within item
        c = df.groupby("idx", sort=True)["hit"].sum().to_numpy().round().astype(int)
        best = R @ self.w                                           # (n_items, n_ks)
        return (item_ids,
                {k: self.pass_tab[k][c] for k in self.ks},
                {k: best[:, j] for j, k in enumerate(self.ks)})


def _slice_metrics(est, df, n_boot, seed):
    _, pk, rk = est.curves(df)
    out = {"n_items": int(df["idx"].nunique()), "pass_at_k": {}, "best_reward_at_k": {}}
    for k in est.ks:
        for key, vals in (("pass_at_k", pk[k]), ("best_reward_at_k", rk[k])):
            m, lo, hi = C.bootstrap_ci(vals, n_boot, seed=seed)
            out[key][str(k)] = {"mean": m, "ci_lo": lo, "ci_hi": hi}
    return out


def _read(files, n):
    """Concat shards and subsample to exactly n samples/item (see module docstring)."""
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    have = df.groupby("idx").size()
    if have.min() < n:
        raise SystemExit(f"ABORT: only {have.min()} samples/item, need {n}: {files[0]}")
    if have.max() > n:
        df = df[df["sample_idx"] < n]
    return df.reset_index(drop=True)


def _protocol(d):
    for j in sorted(glob.glob(os.path.join(d, "eval_results.shard*.json"))):
        sc = (json.load(open(j)) or {}).get("scoring")
        if sc:
            return sc
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="root holding <line>/<arm>/s<step>/samples.shard*.parquet")
    ap.add_argument("--final", action="append", default=[], metavar="LINE:DIR",
                    help="n=4096 dir for the last rung; subsampled to --n")
    ap.add_argument("--base", action="append", default=[], metavar="LINE:DIR",
                    help="n=4096 dir for the UNTRAINED model; emitted as step 0")
    ap.add_argument("--final_step", type=int, default=26448)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--nshards", type=int, default=4, help="expected shards per checkpoint")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow_partial", action="store_true",
                    help="emit checkpoints with a missing shard; the result is over a SUBSET")
    ap.add_argument("--verify", action="store_true", default=True)
    args = ap.parse_args()

    if max(args.ks) > args.n:
        raise SystemExit(f"ABORT: k={max(args.ks)} > n={args.n}; pass@k is not estimable there.")
    est = Estimators(args.n, args.ks)

    # ---- collect sources -------------------------------------------------------------------
    # (line, arm, step) -> shard files. The sweep supplies rungs 100..24000; --final and --base
    # supply the two rungs that already exist at n=4096.
    src, protocol = {}, {}
    for p in sorted(glob.glob(os.path.join(args.sweep, "*", "*", "s*"))):
        m = re.search(r"/([^/]+)/([^/]+)/s(\d+)$", p)
        if not m:
            continue
        line, arm, step = m.group(1), m.group(2), int(m.group(3))
        files = sorted(glob.glob(os.path.join(p, "samples.shard*.parquet")))
        if not files:
            continue
        src[(line, arm, step)] = files
        protocol.setdefault(line, _protocol(p))
    for spec, step in [(s, args.final_step) for s in args.final] + [(s, 0) for s in args.base]:
        line, d = spec.split(":", 1)
        arms = ["base"] if step == 0 else sorted(
            x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))
        for arm in arms:
            files = sorted(glob.glob(os.path.join(d, arm, "samples.shard*.parquet")))
            if not files:
                raise SystemExit(f"ABORT: no parquets under {d}/{arm}")
            src[(line, arm, step)] = files
            protocol.setdefault(line, _protocol(os.path.join(d, arm)))

    if not src:
        raise SystemExit(f"ABORT: nothing under {args.sweep}")

    # ---- completeness ----------------------------------------------------------------------
    incomplete = [(k, len(v)) for k, v in src.items()
                  if k[2] not in (0, args.final_step) and len(v) != args.nshards]
    if incomplete and not args.allow_partial:
        for k, nf in sorted(incomplete):
            print("  INCOMPLETE %s/%s@%d: %d/%d shards" % (k[0], k[1], k[2], nf, args.nshards))
        raise SystemExit(f"ABORT: {len(incomplete)} checkpoint(s) with a partial shard set. "
                         "Re-run them, or pass --allow_partial (results are then over a SUBSET).")

    # ---- load ------------------------------------------------------------------------------
    data = {k: _read(v, args.n) for k, v in sorted(src.items())}
    for (line, arm, step), df in data.items():
        if df["idx"].nunique() * args.n != len(df):
            raise SystemExit(f"ABORT: ragged sample counts for {line}/{arm}@{step}")

    # every arm at a given (line, step) must cover the same items, or it is not a comparison
    byls = defaultdict(dict)
    for (line, arm, step), df in data.items():
        byls[(line, step)][arm] = frozenset(df["idx"].unique())
    for (line, step), arms in sorted(byls.items()):
        if len(set(arms.values())) != 1:
            raise SystemExit("ABORT: %s@%d arms cover different item sets: %s" %
                             (line, step, {a: len(s) for a, s in arms.items()}))

    # protocol must be identical across everything being compared
    for line, sc in protocol.items():
        ref = {k: sc.get(k) for k in ("reward_module", "format_weight", "format_prompt",
                                      "bestk_temperature", "bestk_top_p", "bestk_top_k")} if sc else None
        if ref is None:
            print(f"  WARNING: no scoring block found for line {line}")
    if len({json.dumps({k: v.get(k) for k in ("reward_module", "format_weight", "format_prompt",
                                              "bestk_temperature", "bestk_top_p", "bestk_top_k")},
                       sort_keys=True)
            for v in protocol.values() if v}) > 1:
        raise SystemExit(f"ABORT: lines were scored under DIFFERENT protocols: {protocol}")

    # ---- verify the fast estimators against curves.py ---------------------------------------
    if args.verify:
        probe = data[sorted(data)[0]]
        sub = probe[probe["idx"].isin(sorted(probe["idx"].unique())[:24])]
        _, pk_ref, rk_ref = C.per_item_curves(sub, args.ks)
        _, pk, rk = est.curves(sub)
        for k in args.ks:
            assert np.allclose(pk[k], pk_ref[k], atol=1e-12), f"pass@{k} mismatch"
            assert np.allclose(rk[k], rk_ref[k], atol=1e-9), f"best@{k} mismatch"
        print(f"  verified vectorised estimators == curves.py on {sub['idx'].nunique()} items, "
              f"ks={args.ks}")

    # ---- compute ----------------------------------------------------------------------------
    result = {
        "benchmark": "ScreenSpot-Pro", "n_samples_per_item": args.n, "ks": args.ks,
        "bootstrap_resamples": args.bootstrap,
        "ci": "percentile bootstrap 95% over ITEMS (item difficulty dominates the variance)",
        "estimators": {
            "pass_at_k": "unbiased 1 - C(n-c,k)/C(n,k)  (Chen et al. 2021)",
            "best_reward_at_k": "unbiased sum_i C(i-1,k-1)/C(n,k) * r_(i); reward scale [0, 2.5]",
        },
        "notes": {
            "step_0": "the UNTRAINED base model, the shared initialisation of all three arms",
            f"step_{args.final_step}": f"generated at n=4096, subsampled to n={args.n} so the last "
                                       "rung is measured identically to the other twenty",
        },
        "protocol": protocol, "lines": {},
    }
    rows = []
    for (line, arm, step), df in sorted(data.items()):
        entry = {"n_items": int(df["idx"].nunique()),
                 "overall": _slice_metrics(est, df, args.bootstrap, args.seed)}
        for field, key in (("category", "by_category"), ("ui_type", "by_ui_type")):
            if field not in df.columns:
                continue
            entry[key] = {str(name): _slice_metrics(est, g, args.bootstrap, args.seed)
                          for name, g in df.groupby(field)}
        result["lines"].setdefault(line, {}).setdefault(arm, {})[str(step)] = entry
        for sl, blk in [("overall", entry["overall"])] + \
                       [(f"{f}:{nm}", b) for f, key in (("category", "by_category"), ("ui_type", "by_ui_type"))
                        for nm, b in entry.get(key, {}).items()]:
            for k in args.ks:
                rows.append({"line": line, "arm": arm, "step": step, "slice": sl,
                             "n_items": blk["n_items"], "k": k,
                             **{f"pass_at_k_{s}": blk["pass_at_k"][str(k)][x]
                                for s, x in (("", "mean"), ("lo", "ci_lo"), ("hi", "ci_hi"))},
                             **{f"best_reward_at_k_{s}": blk["best_reward_at_k"][str(k)][x]
                                for s, x in (("", "mean"), ("lo", "ci_lo"), ("hi", "ci_hi"))}})
        print("  %-3s %-5s step %-6d  %d items  pass@1 %.4f  pass@%d %.4f" %
              (line, arm, step, entry["n_items"],
               entry["overall"]["pass_at_k"]["1"]["mean"], max(args.ks),
               entry["overall"]["pass_at_k"][str(max(args.ks))]["mean"]))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    csv = os.path.splitext(args.out)[0] + ".csv"
    pd.DataFrame.from_records(rows).to_csv(csv, index=False)
    n_ck = sum(len(s) for a in result["lines"].values() for s in a.values())
    print(f"\nwrote {args.out}  ({n_ck} checkpoints, {len(rows)} rows)")
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
