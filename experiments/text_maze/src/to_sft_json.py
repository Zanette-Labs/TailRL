"""Convert v2 JSONL maze dataset -> SFT-compatible JSON.

Output format matches the legacy maze SFT loader (maze/sft.py): a JSON list of
items, each with a 'sequence' string and 'optimal_path_length'.

Per-prompt path selection (--mode):
  * all       (default) — one item per (prompt, path); M*N items
  * random    — one item per prompt; randomly choose 1 of N paths
  * shortest  — one item per prompt; choose the lowest-L path (sample_id=0)

Use 'random' for the diverse-paths SFT and 'shortest' for the baseline SFT.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


SIZE = 17
START = (1, 1)
GOAL = (15, 15)
ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT")


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


def make_item(grid_tokens: list[str], obj: dict, sample: dict) -> dict:
    action_tokens = [ACTION_NAMES[a] for a in sample["actions"]]
    seq_tokens = (
        ["<bos>", "GRID_START"]
        + grid_tokens
        + ["GRID_END", "PATH_START"]
        + action_tokens
        + ["DONE", "<eos>"]
    )
    return {
        "sequence": " ".join(seq_tokens),
        "optimal_path_length": obj["L_star"],
        "path_length": sample["L"],
        "reward_continuous": sample["reward_continuous"],
        "prompt_id": obj["prompt_id"],
        "sample_id": sample["sample_id"],
        "ub": obj["ub"],
        "k_frac": obj["k_frac"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, help="JSONL produced by build_dataset.py")
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_test", required=True)
    ap.add_argument("--test_prompts", type=int, default=256,
                    help="reserve this many DISTINCT mazes for the test set "
                         "(test is always 1 path per prompt = the shortest)")
    ap.add_argument(
        "--mode",
        choices=("all", "random", "shortest"),
        default="random",
        help="train-set path selection per prompt: "
             "'all' = M*N items; "
             "'random' = M items, one randomly-picked path per prompt; "
             "'shortest' = M items, the lowest-L path per prompt.",
    )
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for test-split AND random-path-per-prompt selection")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_train) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_test) or ".", exist_ok=True)

    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    # Pass 1: collect prompt_ids to pick the test set deterministically.
    prompt_ids = []
    with open(args.in_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt_ids.append(obj["prompt_id"])
    print(f"Read {len(prompt_ids)} prompts in {time.time()-t0:.1f}s")

    shuffled = list(prompt_ids)
    rng.shuffle(shuffled)
    test_set = set(shuffled[: args.test_prompts])

    train_items: list = []
    test_items: list = []

    # Pass 2: emit training items per --mode policy. Test items always use the
    # shortest sample (sample_id=0 — lowest L) as ground_truth, since at eval
    # time only the maze grid matters for reward computation.
    t1 = time.time()
    with open(args.in_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            grid_tokens = grid_to_tokens(obj["grid"])
            samples_sorted = sorted(obj["samples"], key=lambda x: x["L"])
            shortest_sample = samples_sorted[0]

            if obj["prompt_id"] in test_set:
                test_items.append(make_item(grid_tokens, obj, shortest_sample))
                continue

            if args.mode == "all":
                for s in obj["samples"]:
                    train_items.append(make_item(grid_tokens, obj, s))
            elif args.mode == "random":
                pick = obj["samples"][rng.integers(len(obj["samples"]))]
                train_items.append(make_item(grid_tokens, obj, pick))
            elif args.mode == "shortest":
                train_items.append(make_item(grid_tokens, obj, shortest_sample))
            else:
                raise ValueError(f"unknown mode: {args.mode}")

    print(f"Built items in {time.time()-t1:.1f}s. "
          f"train={len(train_items)}  test={len(test_items)}  mode={args.mode}")

    t2 = time.time()
    with open(args.out_train, "w") as f:
        json.dump(train_items, f)
    with open(args.out_test, "w") as f:
        json.dump(test_items, f)
    print(f"Wrote {args.out_train} and {args.out_test} in {time.time()-t2:.1f}s")
    print(f"Total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
