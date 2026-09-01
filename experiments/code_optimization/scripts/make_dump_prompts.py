"""Choose the fixed set of test programs whose raw generations get dumped.

The evaluation stores per-completion *scores* for all 878 programs, but the
completions themselves are far too large to keep for 878 x 4096 rollouts x 4 arms.
Instead a fixed random subset of programs is dumped, the SAME subset for every arm,
so completions can be read side by side -- which is how you see that GRPO and RLOO
converge to reproducing the source program verbatim while TailRL rewrites it.

    python3 scripts/make_dump_prompts.py --n 64 --seed 20260817

Writes $CODEOPT_EVAL_DIR/dump_prompts.json, which scripts/eval_shard.sh passes to
the shard runner. Deterministic given (parquet, n, seed).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_parquet",
                    default=os.path.join(os.environ.get("PIE_PARQUET_ROOT", ""),
                                         "pie_gem5_test.parquet"))
    ap.add_argument("--n", type=int, default=64, help="how many programs to dump")
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = Path(a.out or os.path.join(os.environ.get("CODEOPT_EVAL_DIR", "."),
                                     "dump_prompts.json"))
    df = pd.read_parquet(a.test_parquet).reset_index(drop=True)
    n = min(a.n, len(df))
    # Index into the parquet, not problem_id: the shard runner identifies prompts by
    # their global row index, which is also what makes the choice arm-independent.
    chosen = sorted(random.Random(a.seed).sample(range(len(df)), n))

    meta = []
    for i in chosen:
        gt = json.loads(df.iloc[i]["reward_model"]["ground_truth"])
        meta.append({"global_idx": i, "problem_id": gt["problem_id"], "src_id": gt["src_id"]})

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_prompts_total": len(df), "n_chosen": n, "seed": a.seed,
        "test_parquet": str(a.test_parquet),
        "chosen_global_idx": chosen, "chosen": meta,
    }, indent=1))
    print(f"[dump] {n} of {len(df)} programs, seed {a.seed} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
