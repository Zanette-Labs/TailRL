#!/usr/bin/env python3
"""Compile the --val-only best-of-K eval console logs into ONE JSON of all val metrics.

Reuses the reeval_and_log.py de-wrap logic (strip ANSI + Ray pid prefixes, collapse
newlines) so wrapped 'key: value' pairs are recovered, then extracts every
maze_17_continuous/* metric (mean@N, best@K, worst@K, trajectory/*, reward_composite/*).

Usage:  python scripts/eval_logs_to_json.py <out.json> <logdir> [glob]
        default glob: eval_*.log   (names: eval_<method>_ck<step>_G<N>_s<seed>.log)
"""
import glob
import json
import os
import re
import statistics
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PID = re.compile(r"\((?:TaskRunner|WorkerDict) pid=\d+\)")
# match BOTH verl print formats after de-wrap: quoted dict ('key': val) AND the
# inline step-summary line (key:val). The inline form is the reliable one; the dict
# form occasionally survives de-wrap poorly (footgun #3), which dropped ~130 metrics
# on some arms. Optional surrounding quotes handle both.
KV = re.compile(r"([A-Za-z@][\w@/\-]*maze_17_continuous[\w@/\-]*)'?:\s*'?(-?[0-9][0-9.eE+\-]*)")
ARM = re.compile(r"eval_([a-z_]+)_ck(\d+)_G(\d+)_s(\d+)")


def parse_metrics(path):
    t = open(path, errors="ignore").read()
    t = ANSI.sub("", t)
    t = PID.sub("", t)
    t = t.replace('"', "")
    t = re.sub(r"\s*\n\s*", " ", t)          # de-wrap: rejoin key and value
    out = {}
    for k, v in KV.findall(t):
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def main():
    outjson = sys.argv[1]
    logdir = sys.argv[2]
    pat = sys.argv[3] if len(sys.argv) > 3 else "eval_*.log"
    result = {}
    for f in sorted(glob.glob(os.path.join(logdir, pat))):
        m = ARM.search(os.path.basename(f))
        if not m:
            continue
        method, ckpt, g, seed = m.groups()
        arm = f"ckpt{ckpt}_{method}_G{g}_seed{seed}"
        metrics = parse_metrics(f)
        if not metrics:
            print(f"  WARN no metrics parsed from {os.path.basename(f)}", file=sys.stderr)
            continue
        # keep the newest log per arm if multiple (jobid suffix); parse gives full set
        result.setdefault(arm, {}).update(metrics)
    # ---- aggregate across seeds -------------------------------------------------
    # Arms differing only in seed are replicates of one configuration, so report
    # mean +/- std over them. Sample std (ddof=1, the n-1 estimator) is the right
    # one for "spread across seeds"; population std would understate it at n=3.
    # Emitted under "_aggregate" so the per-arm keys stay at the top level and any
    # existing consumer of this file keeps working.
    groups = {}
    for arm, metrics in result.items():
        base = re.sub(r"_seed\d+$", "", arm)          # ckpt3000_pkpo_G16_seed0 -> ckpt3000_pkpo_G16
        groups.setdefault(base, {})[arm] = metrics
    agg = {}
    for base, arms in sorted(groups.items()):
        keys = set()
        for m in arms.values():
            keys |= set(m)
        entry = {"n_seeds": len(arms), "seeds": sorted(arms), "metrics": {}}
        for k in sorted(keys):
            vals = [arms[a][k] for a in sorted(arms) if k in arms[a]]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            entry["metrics"][k] = {
                "mean": mean, "std": std, "n": len(vals),
                "values": vals,          # per-seed, in sorted-arm order
            }
        agg[base] = entry
    result["_aggregate"] = agg

    json.dump(result, open(outjson, "w"), indent=2, sort_keys=True)
    total = sum(len(v) for k, v in result.items() if k != "_aggregate")
    print(f"wrote {outjson}: {len(result)-1} arms, {total} metric values, "
          f"{len(agg)} aggregated configs (mean/std over seeds)")
    arms_only = [a for a in sorted(result) if a != "_aggregate"]
    if arms_only:
        a = arms_only[0]
        ks = sorted({int(x) for x in re.findall(r"best@(\d+)", " ".join(result[a]))})
        print(f"best@K present (e.g. {a}): {ks}")
        print("headline is_shortest/best@1024, mean +/- std over seeds:")
        for base in sorted(agg):
            m = agg[base]["metrics"]
            key = next((k for k in m if "is_shortest/best@1024/mean" in k), None)
            if key is None:
                key = next((k for k in m if "is_shortest/best@1024" in k), None)
            if key is None:
                print(f"  {base}: n/a")
                continue
            e = m[key]
            print(f"  {base}: {e['mean']:.4f} +/- {e['std']:.4f}  (n={e['n']}, seeds={[round(v,4) for v in e['values']]})")


if __name__ == "__main__":
    main()
