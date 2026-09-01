"""Build a diverse maze SFT dataset for GOS research.

Per maze: N=16 simple paths whose continuous-reward values are spread across (0,1].
Mazes built via Prim + random wall-knockout (K in [Kmin, Kmax]) so different mazes
have different "path-pool sizes" and contribute to a roughly uniform GLOBAL reward
distribution while each prompt itself still has some spread.

Continuous reward (matching judge_maze_continuous in verl/utils/reward_score/maze.py):
    R = max(0, (UB - L) / (UB - L*))    (0 if not goal-reached)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict, deque
from multiprocessing import Pool

import numpy as np

SIZE = 17
START = (1, 1)
GOAL = (SIZE - 2, SIZE - 2)

DR = (-1, 1, 0, 0)
DC = (0, 0, -1, 1)
ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT")


def gen_prim_maze(size: int, rng: random.Random) -> np.ndarray:
    """Perfect Prim maze. Cells live at odd coords; walls fill the rest."""
    grid = np.ones((size, size), dtype=np.int8)
    sx, sy = START
    grid[sx, sy] = 0
    cell_dirs = [(0, 2), (2, 0), (0, -2), (-2, 0)]

    frontier = []
    in_frontier = set()
    for dx, dy in cell_dirs:
        nx, ny = sx + dx, sy + dy
        if 0 < nx < size - 1 and 0 < ny < size - 1 and grid[nx, ny] == 1:
            frontier.append((nx, ny))
            in_frontier.add((nx, ny))

    while frontier:
        idx = rng.randrange(len(frontier))
        f = frontier[idx]
        frontier[idx] = frontier[-1]
        frontier.pop()
        in_frontier.discard(f)
        fx, fy = f

        neighbors = []
        for dx, dy in cell_dirs:
            nx, ny = fx + dx, fy + dy
            if 0 <= nx < size and 0 <= ny < size and grid[nx, ny] == 0:
                neighbors.append((nx, ny))

        if neighbors:
            nx, ny = rng.choice(neighbors)
            mx, my = (fx + nx) // 2, (fy + ny) // 2
            grid[mx, my] = 0
            grid[fx, fy] = 0
            for dx, dy in cell_dirs:
                wx, wy = fx + dx, fy + dy
                if (
                    0 < wx < size - 1
                    and 0 < wy < size - 1
                    and grid[wx, wy] == 1
                    and (wx, wy) not in in_frontier
                ):
                    frontier.append((wx, wy))
                    in_frontier.add((wx, wy))

    grid[GOAL[0], GOAL[1]] = 0
    return grid


def knockout_walls(grid: np.ndarray, k_frac: float, rng: random.Random) -> int:
    """Knock out a random k_frac of inter-cell walls (creates loops). Returns count."""
    size = grid.shape[0]
    cands = []
    for r in range(1, size - 1):
        for c in range(1, size - 1):
            if grid[r, c] != 1:
                continue
            # An inter-cell wall has two open cells on opposite sides:
            if (grid[r - 1, c] == 0 and grid[r + 1, c] == 0) or (
                grid[r, c - 1] == 0 and grid[r, c + 1] == 0
            ):
                cands.append((r, c))
    n = int(round(len(cands) * k_frac))
    rng.shuffle(cands)
    for r, c in cands[:n]:
        grid[r, c] = 0
    return n


def bfs_dist(grid: np.ndarray, source: tuple) -> np.ndarray:
    """BFS distance grid from source. Unreachable cells get a sentinel."""
    size = grid.shape[0]
    INF = np.int32(10**8)
    dist = np.full((size, size), INF, dtype=np.int32)
    sr, sc = source
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        d = dist[r, c]
        for ai in range(4):
            nr, nc = r + DR[ai], c + DC[ai]
            if 0 <= nr < size and 0 <= nc < size and grid[nr, nc] == 0 and dist[nr, nc] == INF:
                dist[nr, nc] = d + 1
                q.append((nr, nc))
    return dist


def harvest_simple_paths(
    grid: np.ndarray,
    start: tuple,
    goal: tuple,
    ub: int,
    dist_to_goal: np.ndarray,
    rng: random.Random,
    dfs_budget: int = 200_000,
    max_per_bucket: int = 4,
) -> dict:
    """DFS to enumerate simple paths from start to goal of length L < ub.

    Path length range: [L_star, ub - 1]. Reward = (ub - L) / (ub - L_star) ∈ (0, 1].

    Bucket kept per length uses **reservoir sampling** (Vitter algorithm R) so the
    final ``max_per_bucket`` items are an unbiased uniform sample over ALL paths of
    that length the DFS encountered, not just the first ``max_per_bucket`` found.
    This breaks the spatial bias that comes from DFS committing to whichever
    direction it happened to expand first at the root.

    Returns: {length: [action_tuple, ...]} (post-reservoir).
    """
    size = grid.shape[0]
    visited = np.zeros((size, size), dtype=bool)
    sr, sc = start
    gr, gc = goal
    visited[sr, sc] = True

    # length -> [items_kept, seen_count]
    buckets: dict = {}
    actions: list = []
    explore = [0]

    sys.setrecursionlimit(2000)

    L_star = int(dist_to_goal[sr, sc])

    def dfs(r: int, c: int, plen: int) -> None:
        if explore[0] >= dfs_budget:
            return
        if r == gr and c == gc:
            if L_star <= plen < ub:
                entry = buckets.get(plen)
                if entry is None:
                    entry = [[], 0]
                    buckets[plen] = entry
                items, seen = entry
                seen += 1
                if len(items) < max_per_bucket:
                    items.append(tuple(actions))
                else:
                    # Vitter reservoir replacement: each new item has
                    # probability max_per_bucket / seen of staying.
                    j = rng.randrange(seen)
                    if j < max_per_bucket:
                        items[j] = tuple(actions)
                entry[1] = seen
            return
        # Pruning: must be able to reach goal in <= ub-1 total steps
        if plen + dist_to_goal[r, c] >= ub:
            return

        order = [0, 1, 2, 3]
        rng.shuffle(order)
        for ai in order:
            nr = r + DR[ai]
            nc = c + DC[ai]
            if 0 <= nr < size and 0 <= nc < size and grid[nr, nc] == 0 and not visited[nr, nc]:
                visited[nr, nc] = True
                actions.append(ai)
                explore[0] += 1
                dfs(nr, nc, plen + 1)
                actions.pop()
                visited[nr, nc] = False

    dfs(sr, sc, 0)
    return {length: items for length, (items, _seen) in buckets.items()}


def _path_to_cellmask(start: tuple, action_tuple, size: int) -> int:
    """Encode visited cells as a Python int bitmask. Bit (r*size + c) is set iff visited.

    Python ints support &, |, .bit_count() natively; for 17*17=289 bits this is
    ~10x faster than building a Python set of (r,c) tuples.
    """
    r, c = start
    mask = 1 << (r * size + c)
    for a in action_tuple:
        r += DR[a]
        c += DC[a]
        mask |= 1 << (r * size + c)
    return mask


def _diversity_score(cand_mask: int, chosen_masks: list) -> float:
    """Lower max-Jaccard with already-chosen paths => higher diversity (more negative)."""
    if not chosen_masks:
        return 0.0
    max_j = 0.0
    for cc in chosen_masks:
        inter = (cand_mask & cc).bit_count()
        union = (cand_mask | cc).bit_count()
        if union == 0:
            continue
        j = inter / union
        if j > max_j:
            max_j = j
    return -max_j


def stratified_sample(
    buckets: dict, l_star: int, ub: int, n_paths: int, rng: random.Random,
    start: tuple = START, size: int = SIZE,
) -> list:
    """Pick n_paths spread across reward (0,1] AND spatially diverse.

    Per reward-bucket: walk outward to find candidates; pick the one that
    minimizes max-Jaccard with already-selected paths (greedy farthest-point).
    """
    pool = []  # (path, length, reward, cellmask)
    for length, paths in buckets.items():
        if not (l_star <= length < ub):
            continue
        if ub == l_star:
            r = 1.0
        else:
            r = max(0.0, (ub - length) / (ub - l_star))
        for p in paths:
            pool.append((p, length, r, _path_to_cellmask(start, p, size)))

    if not pool:
        return []

    rbuckets: list[list] = [[] for _ in range(n_paths)]
    for entry in pool:
        r = entry[2]
        if r <= 0:
            idx = 0
        elif r >= 1.0:
            idx = n_paths - 1
        else:
            idx = min(int(r * n_paths), n_paths - 1)
        rbuckets[idx].append(entry)

    chosen: list = []
    chosen_masks: list = []
    seen_paths: set = set()

    bucket_visit = list(range(n_paths))
    rng.shuffle(bucket_visit)

    for bucket_idx in bucket_visit:
        cands = []
        for off in range(n_paths):
            offsets = (0,) if off == 0 else (off, -off)
            for sign in offsets:
                j = bucket_idx + sign
                if 0 <= j < n_paths:
                    for entry in rbuckets[j]:
                        if entry[0] not in seen_paths:
                            cands.append(entry)
            if cands:
                break

        if not cands:
            continue

        if not chosen_masks:
            pick = rng.choice(cands)
        else:
            best_score = float("-inf")
            best_pick = None
            for cand in cands:
                s = _diversity_score(cand[3], chosen_masks) + rng.random() * 1e-6
                if s > best_score:
                    best_score = s
                    best_pick = cand
            pick = best_pick

        chosen.append((pick[0], pick[1], pick[2]))
        chosen_masks.append(pick[3])
        seen_paths.add(pick[0])

    chosen.sort(key=lambda x: x[1])
    return chosen


def grid_to_token_prefix(grid: np.ndarray) -> str:
    """Tokenized maze prompt: '<bos> GRID_START ... GRID_END PATH_START'."""
    size = grid.shape[0]
    toks = ["<bos>", "GRID_START"]
    for r in range(size):
        for c in range(size):
            pos = (r, c)
            if pos == START:
                toks.append("START")
            elif pos == GOAL:
                toks.append("GOAL")
            elif grid[r, c] == 1:
                toks.append("WALL")
            else:
                toks.append("PATH")
        toks.append("NEWLINE")
    toks += ["GRID_END", "PATH_START"]
    return " ".join(toks)


def grid_to_full_sequence(prompt_prefix: str, action_ids: list[int]) -> str:
    """prompt prefix + ' UP DOWN ... DONE <eos>' (concatenated full sequence)."""
    action_toks = [ACTION_NAMES[a] for a in action_ids]
    return prompt_prefix + " " + " ".join(action_toks) + " DONE <eos>"


def build_one_maze(seed: int, ub: int = 60, n_paths: int = 16,
                   dfs_budget: int = 200_000, k_min: float = 0.05,
                   k_max: float = 0.30, max_per_bucket: int = 4) -> dict | None:
    """Build one maze + n_paths samples. Returns None on failure."""
    rng = random.Random(seed)
    k_frac = rng.uniform(k_min, k_max)

    grid = gen_prim_maze(SIZE, rng)
    n_knockouts = knockout_walls(grid, k_frac, rng)

    dist_to_goal = bfs_dist(grid, GOAL)
    if dist_to_goal[START[0], START[1]] >= 10**8:
        return None  # unreachable; should never happen with our generator

    L_star = int(dist_to_goal[START[0], START[1]])
    if L_star >= ub:
        return None

    buckets = harvest_simple_paths(
        grid, START, GOAL, ub, dist_to_goal, rng,
        dfs_budget=dfs_budget, max_per_bucket=max_per_bucket,
    )

    chosen = stratified_sample(buckets, L_star, ub, n_paths, rng)
    if len(chosen) == 0:
        return None

    samples = []
    Ls = []
    for sample_id, (actions, L, r) in enumerate(chosen):
        samples.append({
            "sample_id": sample_id,
            "L": L,
            "reward_continuous": float(r),
            "actions": list(actions),
        })
        Ls.append(L)

    return {
        "prompt_id": seed,
        "grid": grid.tolist(),
        "L_star": L_star,
        "k_frac": float(k_frac),
        "n_knockouts": n_knockouts,
        "ub": ub,
        "n_samples": len(samples),
        "harvest_pool_size": sum(len(v) for v in buckets.values()),
        "L_std": float(np.std(Ls)) if len(Ls) > 1 else 0.0,
        "L_min": int(min(Ls)),
        "L_max": int(max(Ls)),
        "samples": samples,
    }


def _worker(args):
    seed, kwargs = args
    return build_one_maze(seed, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1000, help="number of mazes")
    ap.add_argument("--n", type=int, default=16, help="paths per maze")
    ap.add_argument("--ub", type=int, default=60, help="continuous-reward UB")
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--out", type=str, required=True,
                    help="JSONL output path (one prompt per line)")
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--dfs_budget", type=int, default=200_000)
    ap.add_argument("--k_min", type=float, default=0.05)
    ap.add_argument("--k_max", type=float, default=0.30)
    ap.add_argument("--max_per_bucket", type=int, default=4)
    ap.add_argument("--chunksize", type=int, default=64,
                    help="multiprocessing chunksize")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    kwargs = dict(
        ub=args.ub, n_paths=args.n, dfs_budget=args.dfs_budget,
        k_min=args.k_min, k_max=args.k_max, max_per_bucket=args.max_per_bucket,
    )
    seeds = range(args.seed_start, args.seed_start + args.m)
    work = ((s, kwargs) for s in seeds)

    t0 = time.time()
    kept = 0
    progress_every = max(1, args.m // 50)

    # Streaming write: one prompt per line. Crash-safe.
    meta = {
        "size": SIZE, "start": START, "goal": GOAL, "ub": args.ub,
        "m_target": args.m, "n": args.n,
        "k_min": args.k_min, "k_max": args.k_max,
        "dfs_budget": args.dfs_budget, "max_per_bucket": args.max_per_bucket,
        "seed_start": args.seed_start,
    }
    meta_path = args.out + ".meta.json"

    with open(args.out, "w") as f, Pool(args.workers) as pool:
        for i, item in enumerate(pool.imap_unordered(_worker, work, chunksize=args.chunksize)):
            if item is not None:
                f.write(json.dumps(item, separators=(",", ":")))
                f.write("\n")
                kept += 1
            if (i + 1) % progress_every == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (args.m - i - 1) / rate if rate > 0 else 0
                print(
                    f"  [{i+1:>8}/{args.m}] kept={kept:>8} "
                    f"elapsed={elapsed:7.1f}s rate={rate:7.1f}/s "
                    f"eta={eta/60:6.1f}min",
                    flush=True,
                )
                f.flush()

    elapsed = time.time() - t0
    meta["kept"] = kept
    meta["elapsed_seconds"] = elapsed
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Built {kept}/{args.m} mazes in {elapsed:.1f}s "
          f"({kept/max(1, elapsed):.1f}/s).")
    print(f"Wrote {args.out}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
