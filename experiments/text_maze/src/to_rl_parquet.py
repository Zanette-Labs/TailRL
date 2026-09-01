"""Convert v2 JSONL maze dataset -> RL-training parquet (verl format).

verl row schema (matches maze/prepare_maze_parquet.py):
    data_source: str
    prompt: list[{"role": "user", "content": <prompt_text>}]
    reward_model: {"style": "rule", "ground_truth": <full_sequence>}
    extra_info: {answer, question, optimal_path_length, index}

For our dataset, the maze grid is in the prompt; the model generates a path
from PATH_START to DONE. Multiple paths exist per maze in the v2 dataset, but
RL training generates from scratch so 1 row per maze suffices. We pick the
shortest sample (sample_id=0) as the ground_truth (only the maze grid in it
matters for the reward function).

Test split uses the same prompt_id-based numpy RNG as to_sft_json.py, so with
the same --seed and --test_prompts the SFT and RL test sets contain the same
mazes (this lets SFT eval and RL eval be compared apples-to-apples).

Generates the same 4 reward variants (data_source) as the legacy script.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd


SIZE = 17
START = (1, 1)
GOAL = (15, 15)
ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT")

REWARD_VARIANTS = [
    "maze_17",
    "maze_17_binary_shortest",
    "maze_17_continuous",
    "maze_17_composite",
]


def grid_to_tokens(grid: list[list[int]]) -> list[str]:
    toks: list[str] = []
    h = len(grid)
    w = len(grid[0])
    for r in range(h):
        for c in range(w):
            pos = (r, c)
            if pos == START:
                toks.append("START")
            elif pos == GOAL:
                toks.append("GOAL")
            elif grid[r][c] == 1:
                toks.append("WALL")
            else:
                toks.append("PATH")
        toks.append("NEWLINE")
    return toks


def build_rows(items, data_source: str, idx_offset: int = 0):
    rows = []
    for idx, obj in enumerate(items, start=idx_offset):
        grid = obj["grid"]
        grid_tokens = grid_to_tokens(grid)
        # Pick the L*-bucket sample for ground_truth
        s = sorted(obj["samples"], key=lambda x: x["L"])[0]
        action_tokens = [ACTION_NAMES[a] for a in s["actions"]]
        seq_tokens = (
            ["<bos>", "GRID_START"]
            + grid_tokens
            + ["GRID_END", "PATH_START"]
            + action_tokens
            + ["DONE", "<eos>"]
        )
        seq = " ".join(seq_tokens)

        path_start_pos = seq.find("PATH_START")
        prompt_str = seq[: path_start_pos + len("PATH_START")].strip()

        rows.append({
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt_str}],
            "reward_model": {"style": "rule", "ground_truth": seq},
            "extra_info": {
                "answer": seq,
                "question": prompt_str,
                "optimal_path_length": obj["L_star"],
                "index": idx,
            },
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, help="JSONL produced by build_dataset.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_prompts", type=int, default=2048,
                    help="reserve this many DISTINCT mazes for test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", nargs="+", default=REWARD_VARIANTS,
                    help="reward variants to emit; pick a subset of "
                         "maze_17 / maze_17_binary_shortest / maze_17_continuous / maze_17_composite")
    ap.add_argument("--canonical", action="store_true",
                    help="when exactly one variant is requested, write files as "
                         "train.parquet / test.parquet (no variant suffix)")
    args = ap.parse_args()

    if args.canonical and len(args.variants) != 1:
        ap.error("--canonical requires exactly one --variants entry")

    unknown = set(args.variants) - set(REWARD_VARIANTS)
    if unknown:
        ap.error(f"unknown variant(s): {sorted(unknown)}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Match to_sft_json.py: split on prompt_id with numpy default_rng(seed).
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    items = []
    with open(args.in_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"Read {len(items)} prompts in {time.time()-t0:.1f}s")

    prompt_ids = [obj["prompt_id"] for obj in items]
    shuffled = list(prompt_ids)
    rng.shuffle(shuffled)
    test_set = set(shuffled[: args.test_prompts])

    train_items = [obj for obj in items if obj["prompt_id"] not in test_set]
    test_items = [obj for obj in items if obj["prompt_id"] in test_set]
    print(f"  train mazes: {len(train_items)}   test mazes: {len(test_items)}")

    for variant in args.variants:
        t = time.time()
        train_rows = build_rows(train_items, variant)
        test_rows = build_rows(test_items, variant, idx_offset=len(train_rows))
        suffix = "" if args.canonical else f"_{variant}"
        train_path = os.path.join(args.out_dir, f"train{suffix}.parquet")
        test_path = os.path.join(args.out_dir, f"test{suffix}.parquet")
        pd.DataFrame(train_rows).to_parquet(train_path)
        pd.DataFrame(test_rows).to_parquet(test_path)
        print(f"  {variant}: train={len(train_rows)} test={len(test_rows)}  "
              f"({time.time()-t:.1f}s) -> {train_path}, {test_path}")

    print(f"Total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
