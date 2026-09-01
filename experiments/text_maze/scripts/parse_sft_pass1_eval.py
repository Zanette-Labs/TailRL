#!/usr/bin/env python3
"""Parse standard maze eval logs into a pass@1 CSV."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
NUM_RE = re.compile(
    r"(?:np\.float64\()?([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\)?"
)
RAY_PREFIX_RE = re.compile(r"\([^)]* pid=\d+\)\s*")


def clean_text(path: Path) -> str:
    return ANSI_RE.sub("", path.read_text(errors="replace"))


def metric(text: str, name: str) -> float | None:
    search_text = text
    step_idx = text.rfind("step:0 -")
    if step_idx > 0:
        search_text = text[:step_idx]
    pattern = re.compile(
        rf"{re.escape(name)}(?:@\d+)?['\"]?\s*:"
    )
    matches = list(pattern.finditer(search_text))
    if not matches and search_text is not text:
        matches = list(pattern.finditer(text))
    for match in reversed(matches):
        window = search_text[match.end() : match.end() + 500]
        window = RAY_PREFIX_RE.sub("", window).replace('"', " ").replace("'", " ")
        num_match = NUM_RE.search(window)
        if num_match:
            return float(num_match.group(1))
    return None


def ckpt_from_log(path: Path) -> int | None:
    match = re.search(r"ckpt[-_]?(\d+)", path.name)
    if match:
        return int(match.group(1))
    text = clean_text(path)
    match = re.search(r"ckpt-(\d+)", text)
    return int(match.group(1)) if match else None


def row_from_log(path: Path) -> dict[str, object]:
    text = clean_text(path)
    ckpt = ckpt_from_log(path)
    val_n_match = re.search(r"val_n=(\d+)", text) or re.search(r"_G(\d+)", path.name)
    val_n = int(val_n_match.group(1)) if val_n_match else None

    reward_mean = metric(text, "mean_accuracies/maze_17/reward/mean")
    goal_mean = metric(text, "val-aux/maze_17/goal_reached/mean")
    aggregate_goal = metric(text, "val/maze_17/trajectory/goal_reached")
    shortest_mean = metric(text, "val-aux/maze_17/is_shortest/mean")
    wall_collision = metric(text, "val/maze_17/trajectory/wall_collision")
    valid_format = metric(text, "val/maze_17/trajectory/valid_format")
    done_token = metric(text, "val/maze_17/trajectory/done_token_generated")
    num_examples = metric(text, "validation_num_examples/all")

    pass1 = goal_mean
    if pass1 is None:
        pass1 = aggregate_goal
    if pass1 is None:
        pass1 = reward_mean

    return {
        "ckpt_step": ckpt,
        "pass1": pass1,
        "pass1_pct": None if pass1 is None else 100.0 * pass1,
        "reward_mean": reward_mean,
        "goal_reached_mean": goal_mean,
        "aggregate_goal_reached": aggregate_goal,
        "is_shortest_mean": shortest_mean,
        "wall_collision": wall_collision,
        "valid_format": valid_format,
        "done_token_generated": done_token,
        "validation_num_examples": num_examples,
        "val_n": val_n,
        "log_path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    rows = [row_from_log(p) for p in sorted(log_dir.glob("ckpt-*_G*.log"))]
    rows = [r for r in rows if r["ckpt_step"] is not None]
    rows.sort(key=lambda r: int(r["ckpt_step"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "ckpt_step",
        "pass1",
        "pass1_pct",
        "reward_mean",
        "goal_reached_mean",
        "aggregate_goal_reached",
        "is_shortest_mean",
        "wall_collision",
        "valid_format",
        "done_token_generated",
        "validation_num_examples",
        "val_n",
        "log_path",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
