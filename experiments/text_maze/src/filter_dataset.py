"""Filter a JSONL maze dataset down to top-K by per-prompt L_std (descending).

Two-pass approach to keep memory bounded:
  Pass 1: scan, collect (offset_in_file, L_std) pairs.
  Pass 2: sort, take top K, re-emit those lines in their original order.
"""

from __future__ import annotations

import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, help="input JSONL")
    ap.add_argument("--out_path", required=True, help="output JSONL")
    ap.add_argument("--k", type=int, required=True, help="keep top K by L_std")
    ap.add_argument("--metric", choices=["L_std", "reward_std"], default="L_std")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    t0 = time.time()
    # Pass 1: build (line_idx, file_offset, line_len, metric)
    keys = []  # list of (metric_value, line_idx, offset, length)
    with open(args.in_path, "rb") as f:
        line_idx = 0
        offset = 0
        for raw in f:
            length = len(raw)
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                offset += length
                line_idx += 1
                continue
            if args.metric == "L_std":
                metric = obj.get("L_std", 0.0)
            else:
                rewards = [s["reward_continuous"] for s in obj.get("samples", [])]
                if rewards:
                    n = len(rewards)
                    mean = sum(rewards) / n
                    metric = (sum((r - mean) ** 2 for r in rewards) / n) ** 0.5
                else:
                    metric = 0.0
            keys.append((metric, line_idx, offset, length))
            offset += length
            line_idx += 1
    n_total = len(keys)
    print(f"Pass 1: scanned {n_total} prompts in {time.time()-t0:.1f}s")

    # Sort by metric desc, take top K
    keys.sort(key=lambda x: -x[0])
    if args.k > n_total:
        print(f"WARNING: requested {args.k} but only have {n_total}; keeping all.")
        keep = keys
    else:
        keep = keys[: args.k]
    threshold = keep[-1][0] if keep else 0.0
    print(f"  metric={args.metric} threshold(top-{len(keep)}) = {threshold:.4f}  "
          f"max={keep[0][0]:.4f}  median={keep[len(keep)//2][0]:.4f}")

    # Re-sort by line_idx so we read sequentially in pass 2
    keep.sort(key=lambda x: x[1])

    # Pass 2: read kept lines by offset, write to out_path
    t1 = time.time()
    with open(args.in_path, "rb") as fin, open(args.out_path, "wb") as fout:
        for metric, line_idx, offset, length in keep:
            fin.seek(offset)
            raw = fin.read(length)
            fout.write(raw)
    print(f"Pass 2: wrote {len(keep)} lines in {time.time()-t1:.1f}s")
    print(f"Total {time.time()-t0:.1f}s")
    print(f"Wrote {args.out_path}")


if __name__ == "__main__":
    main()
