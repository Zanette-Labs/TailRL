"""Merge the per-shard rollout-dump sidecars (shard_*.gens.jsonl) of each arm into
one generations JSON per arm, plus a combined file. Dump policy is set upstream by
the cell (random 64 completions for a fixed 64-prompt set); this just concatenates
and dedups by (problem_id, src_id). Metrics live in a SEPARATE file (the aggregator's
summary.json) -- this file holds ONLY the raw generations.

Usage:
  python -m code_opt.eval.besteval_gens_merge \
    --eval_root "$EVAL_ROOT" \
    --runs tailrl_step300 grpo_step500 rloo_step500 base_step0
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path


def resolve_run_dir(root: Path, run: str) -> Path:
    d = root / run
    if d.is_dir():
        return d
    cands = sorted(glob.glob(str(root / f"*_{run}")))
    return Path(cands[0]) if cands else d


def merge_run(run_dir: Path):
    seen, prompts = set(), []
    for f in sorted(glob.glob(str(run_dir / "shard_*.gens.jsonl"))):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r.get("problem_id"), r.get("src_id"))
            if key in seen:
                continue
            seen.add(key)
            prompts.append(r)
    prompts.sort(key=lambda r: r.get("global_idx", 1 << 30))
    return prompts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.eval_root)
    out = Path(a.out or (root / "generations")); out.mkdir(parents=True, exist_ok=True)

    combined = {}
    for run in a.runs:
        arm = run.split("_step")[0]
        step = run.split("_step")[1] if "_step" in run else "?"
        prompts = merge_run(resolve_run_dir(root, run))
        n_gens = sum(len(p.get("gens", [])) for p in prompts)
        rec = {"run": run, "arm": arm, "step": step,
               "n_prompts_dumped": len(prompts), "n_gens_total": n_gens,
               "prompts": prompts}
        (out / f"generations_{run}.json").write_text(json.dumps(rec, indent=1))
        combined[run] = {"arm": arm, "step": step, "n_prompts_dumped": len(prompts),
                         "n_gens_total": n_gens}
        print(f"[{run}] {len(prompts)} prompts dumped, {n_gens} generations "
              f"-> {out}/generations_{run}.json")
    (out / "generations_index.json").write_text(json.dumps(combined, indent=1))
    print(f"[index] -> {out}/generations_index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
