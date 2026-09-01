"""Build the verl train/val/test parquets from the gem5-timed PIE dataset on the
Hugging Face Hub (``stablegradients/pie-gem5-bysrc``).

The Hub files cannot be handed to verl directly: they carry ``src_code`` plus 51
provenance columns, and verl's ``RLHFDataset`` wants a rendered prompt, a
``data_source`` and a ``reward_model`` struct. This is that conversion. One row per
unique (problem, src):

    data_source: "pie"
    prompt: [{"role": "user", "content": PROMPT_TEMPLATE.format(src_code=...)}]
    reward_model: {"ground_truth": json.dumps({
        "problem_id", "src_id", "split",
        "usable_case_ids":       [int, ...],     # the post-filter case set
        "gem5_src_per_tc_ticks": {str(id): int}, # ticks for the usable cases
    })}

ground_truth carries exactly what reward/gem5_reward.py::_job_from_ground_truth
consumes: the native gate runs ALL usable_case_ids, gem5 times K of them against
these src reference ticks (ticks are trimmed to usable — every usable case has a
tick by dataset construction; the full blobs live on HF). src_id is the group
key for group-shared K sampling.

The only filter on top of the (already fully filtered) HF dataset is the same
tokenized-prompt-length cap used by build_pie_parquet.py, so batching behaves
identically.

Usage -- normally through ``scripts/prepare_dataset.sh``:

    python -m code_opt.build_pie_gem5_parquet --output_root "$PIE_PARQUET_ROOT"

Takes a few minutes, needs network access to the Hub, and no GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

HF_DATASET = "stablegradients/pie-gem5-bysrc"
# Where the converted parquets land. Keep them out of the checkout by default.
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("PIE_PARQUET_ROOT")
    or (Path(__file__).resolve().parents[1] / "data" / "parquet")
)

# 'minimal' is what every run in the paper used: persona + one instruction, with the
# "### Optimized Version:" anchor PREFILLED at the end of the user turn, so the model
# emits code directly. 'p3' instead asks for step-by-step reasoning and REQUESTS the
# anchor rather than prefilling it, leaving room for a chain of thought before the
# program. Both anchor on "### Optimized Version:", which is what the reward's code
# extractor parses.
PROMPT_TEMPLATE = (
    "You are a C++ optimization expert. Below is a working C++ program. "
    "Rewrite it to run faster while producing identical output for all valid "
    "inputs. Keep the same input/output format and only change the algorithm/"
    "implementation.\n\n"
    "### Slow Version:\n"
    "```cpp\n{src_code}\n```\n\n"
    "### Optimized Version:\n"
)

P3_PROMPT_TEMPLATE = (
    "You are a C++ optimization expert. Below is a working C++ program. Think step by step "
    "about how to make it run faster — its time complexity, the main bottlenecks, the data "
    "structures and I/O — and then write the optimized program. It must produce identical "
    "output for every valid input.\n\n"
    "### Slow Version:\n"
    "```cpp\n{src_code}\n```\n\n"
    "After your reasoning, give your final program under a line that reads exactly "
    "`### Optimized Version:`, as a single ```cpp code block."
)

PROMPT_TEMPLATES = {"minimal": PROMPT_TEMPLATE, "p3": P3_PROMPT_TEMPLATE}


def _format_record(row, split: str, template: str = PROMPT_TEMPLATE) -> dict:
    usable = [int(c) for c in json.loads(row["usable_case_ids"])]
    ticks_all = json.loads(row["gem5_src_per_tc_ticks"])
    ticks = {str(c): int(ticks_all[str(c)]) for c in usable}
    return {
        "data_source": "pie",
        "prompt": [
            {"role": "user",
             "content": template.format(src_code=row["src_code"])},
        ],
        "reward_model": {
            "ground_truth": json.dumps({
                "problem_id": row["problem_id"],
                "src_id": row["src_id"],
                "split": split,
                "usable_case_ids": usable,
                "gem5_src_per_tc_ticks": ticks,
            }),
        },
    }


def build_split(ds, split: str, tokenizer, max_prompt_tokens: int,
                template: str = PROMPT_TEMPLATE):
    kept, n_overlong, n_bad = [], 0, 0
    overlong_examples = []
    for row in ds:
        try:
            rec = _format_record(row, split, template)
        except (KeyError, TypeError, ValueError) as e:
            n_bad += 1
            print(f"  [warn] {split} row {row.get('problem_id')}/{row.get('src_id')}: "
                  f"{type(e).__name__}: {e}")
            continue
        n_tok = len(tokenizer(rec["prompt"][0]["content"],
                              add_special_tokens=False)["input_ids"])
        if n_tok > max_prompt_tokens:
            n_overlong += 1
            if len(overlong_examples) < 5:
                overlong_examples.append((row["problem_id"], n_tok))
            continue
        kept.append(rec)
    return kept, {"n_total": len(ds), "n_kept": len(kept),
                  "n_overlong": n_overlong, "n_bad": n_bad,
                  "overlong_examples": overlong_examples}


def write_parquet(records: list[dict], path: Path) -> None:
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--hf_dataset", default=HF_DATASET)
    ap.add_argument("--val_subsample", type=int, default=200)
    ap.add_argument("--val_seed", type=int, default=0)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B",
                    help="Prompt-length filter tokenizer (bare Qwen3 repo name "
                         "IS the instruct model).")
    ap.add_argument("--max_prompt_tokens", type=int, default=6144)
    ap.add_argument("--prompt", choices=list(PROMPT_TEMPLATES), default="minimal",
                    help="prompt template: 'minimal' (legacy 1.7B run) or 'p3' "
                         "(step-by-step, pre-RL-recommended for the 4B run)")
    args = ap.parse_args()
    output_root = Path(args.output_root)
    template = PROMPT_TEMPLATES[args.prompt]
    print(f"[prompt] using '{args.prompt}' template ({len(template)} chars)")

    print(f"[hf] loading {args.hf_dataset} ...")
    from datasets import load_dataset
    dsd = load_dataset(args.hf_dataset)
    print(f"[hf] splits: { {k: len(v) for k, v in dsd.items()} }")

    print(f"[tokenizer] loading {args.tokenizer} ...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    summary = {}
    for hf_split, split, out_name in [
        ("train", "train", "pie_gem5_train.parquet"),
        ("validation", "val", "pie_gem5_val_full.parquet"),
        ("test", "test", "pie_gem5_test.parquet"),
    ]:
        records, stats = build_split(dsd[hf_split], split, tokenizer,
                                     args.max_prompt_tokens, template)
        out_path = output_root / out_name
        write_parquet(records, out_path)
        summary[split] = {**stats, "out_path": str(out_path)}
        print(f"[{split}] kept {stats['n_kept']:>6} / {stats['n_total']:>6} "
              f"(overlong>{args.max_prompt_tokens}={stats['n_overlong']}, "
              f"bad={stats['n_bad']}) -> {out_path}")
        if stats["overlong_examples"]:
            print(f"  overlong examples: {stats['overlong_examples']}")

    # fixed val subsample for in-situ eval, mirroring build_pie_parquet.py
    import pandas as pd
    df_val = pd.read_parquet(output_root / "pie_gem5_val_full.parquet")
    recs = df_val.to_dict(orient="records")
    if len(recs) > args.val_subsample:
        recs = random.Random(args.val_seed).sample(recs, args.val_subsample)
    val_sub = output_root / f"pie_gem5_val_{args.val_subsample}.parquet"
    write_parquet(recs, val_sub)
    summary["val_subsample"] = {"n_rows": len(recs), "out_path": str(val_sub)}
    print(f"[val_subsample] {len(recs)} rows (seed={args.val_seed}) -> {val_sub}")

    with open(output_root / "MANIFEST_GEM5.json", "w") as f:
        json.dump({"schema_version": 1, "hf_dataset": args.hf_dataset,
                   "tokenizer": args.tokenizer, "prompt_variant": args.prompt,
                   "max_prompt_tokens": args.max_prompt_tokens,
                   "reward_engine": "code_opt/reward/gem5_reward.py",
                   "splits": summary}, f, indent=2)
    print(f"[done] MANIFEST -> {output_root}/MANIFEST_GEM5.json")

    print("\nNext: export these for the RL launch")
    print(f"  export PIE_TRAIN_PARQUET={output_root}/pie_gem5_train.parquet")
    print(f"  export PIE_VAL_PARQUET={output_root}/pie_gem5_val_{args.val_subsample}.parquet")
    n_train = summary["train"]["n_kept"]
    spe = math.ceil(n_train / 256)
    print(f"\nEpoch arithmetic (B=256, N_train={n_train}):")
    print(f"  export PIE_STEPS_PER_EPOCH={spe}")
    print(f"  export PIE_TOTAL_STEPS={10 * spe}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
