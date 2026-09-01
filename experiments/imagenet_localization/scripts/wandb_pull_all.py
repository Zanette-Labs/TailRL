"""Pull every relevant run's per-epoch history from W&B and write local
metrics.json files. Use as the source of truth for plotting; overwrites any
existing local copy so the plotter always sees the full history.

Usage (from the repo root):
    python experiments/imagenet_localization/scripts/wandb_pull_all.py

The project comes from $WANDB_PROJECT, the entity from $WANDB_ENTITY (unset =
your default entity) and the destination from $TAILRL_RESULTS_DIR. Auth is the
one the `wandb` CLI already has: `wandb login` writes it to ~/.netrc.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import wandb

# Also runnable as a plain file path, which needs the repo root on sys.path
# before the package import below.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.imagenet_localization import paths  # noqa: E402

ENTITY = paths.wandb_entity()
PROJECT = paths.wandb_project()
RESULTS_DIR = Path(paths.results_dir())

# (display_name -> wandb run id) for the sweep behind the paper's figures.
# Replace these with the ids of your own runs — they are specific to the W&B
# project they were logged to. Generated from the matching W&B inventory
# (state in {"finished", "crashed", "failed"} — we still take whatever
# epochs the run did log before dying, so even failed runs contribute).
TARGETS: dict[str, str] = {
    "tailrl_population_K50_N64_seed42":  "tuyfv5ag",
    "tailrl_population_K50_N64_seed43":  "66xm5j23",
    "tailrl_population_K50_N64_seed44":  "6n2qe3vx",
    "tailrl_population_K50_N64_seed45":  "kqgsjsnp",
    "tailrl_K50_N1024_seed42":           "gd7qai6o",
    "tailrl_K50_N1024_seed43":           "6ix07l5e",
    "tailrl_K50_N1024_seed44":           "a6aguwg8",
    "tailrl_K50_N1024_seed45":           "to73q4dg",
    "tailrl_K50_N16_seed42":             "5t2v0la3",
    "tailrl_K50_N16_seed43":             "80bxuown",
    "tailrl_K50_N16_seed44":             "dnxav8bs",
    "tailrl_K50_N16_seed45":             "f71avnrc",
    "tailrl_K50_N256_seed42":            "214altic",
    "tailrl_K50_N256_seed43":            "rdv3ljff",
    "tailrl_K50_N256_seed44":            "r28lwtna",
    "tailrl_K50_N256_seed45":            "eep4snqo",
    "tailrl_K50_N64_seed42":             "7w6d84zc",
    "tailrl_K50_N64_seed43":             "jefk1lti",
    "tailrl_K50_N64_seed44":             "7ehnq3ww",
    "tailrl_K50_N64_seed45":             "5vycn9nw",
    "grpo_K50_N256_seed42":           "hx15ijnv",
    "grpo_K50_N256_seed43":           "1oh44eul",
    "grpo_K50_N256_seed44":           "n967qd7v",
    "grpo_K50_N256_seed45":           "jarvmuwg",
    "l1_centroid_match_K50_N64_seed42":  "2ldhl83a",
    "l1_centroid_match_K50_N64_seed43":  "podathk8",
    "l1_centroid_match_K50_N64_seed44":  "cvlerobw",
    "l1_centroid_match_K50_N64_seed45":  "b9mcphk5",
    "mse_centroid_match_K50_N64_seed42": "0atx49kb",
    "mse_centroid_match_K50_N64_seed43": "s69ya8xg",
    "mse_centroid_match_K50_N64_seed44": "up3rmip4",
    "mse_centroid_match_K50_N64_seed45": "gfb880dr",
    "ordinal_ce_K50_N64_seed42":      "h3emalg1",
    "reinforce_K50_N256_seed42":      "9xw6usop",
    "reinforce_K50_N256_seed43":      "6pkv0s1m",
    "reinforce_K50_N256_seed44":      "jcw7sq8c",
    "reinforce_K50_N256_seed45":      "7bb99dl1",
    "rloo_K50_N256_seed42":           "hdu08ddh",
    "rloo_K50_N256_seed43":           "v8ztt1mx",
    "rloo_K50_N256_seed44":           "53e95d9a",
    "rloo_K50_N256_seed45":           "mc3vjkr8",
}

# W&B's history(keys=...) is row-intersection: a row is returned only when
# every requested key is non-null at that step. So we split keys into a
# core set (always logged on eval rows for every run) and an optional set
# (per-run availability — pulled separately and merged on epoch).
CORE_KEYS = [
    "epoch",
    "val/iou_at_50", "val/iou_at_75", "val/iou_at_90",
    "val/iou_greedy",
    "val_easy/iou_greedy", "val_medium/iou_greedy", "val_hard/iou_greedy",
    "val_easy/count", "val_medium/count", "val_hard/count",
]
OPTIONAL_KEY_GROUPS = [
    # Each inner list is pulled together (atomic group). Groups whose run
    # didn't log them all are silently skipped.
    ["epoch", "val/iou_best_of_N", "val/iou_expected"],
    ["epoch", "val/ordinal_ce_objective", "val/cross_entropy_objective"],
    ["epoch", "val/entropy_joint_mean",
     "val/entropy_xc", "val/entropy_yc", "val/entropy_w", "val/entropy_h"],
]


def _coerce(key: str, value) -> object | None:
    """Return JSON-serialisable form of `value`, or None if not finite."""
    if value is None:
        return None
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    if fv != fv:  # NaN
        return None
    return int(fv) if key.endswith("/count") else fv


def _pull_keys(run, keys: list[str]) -> dict[int, dict]:
    """Pull `keys` together (so wandb only emits eval rows where all are
    non-null). Returns {epoch_int: {key: value, ...}} merged-by-epoch.
    """
    df = run.history(keys=keys, pandas=True, samples=10000)
    out: dict[int, dict] = {}
    if df is None or df.empty or "epoch" not in df.columns:
        return out
    df = df.dropna(subset=["epoch"]).sort_values("epoch")
    df = df.drop_duplicates(subset="epoch", keep="last")
    for _, r in df.iterrows():
        ep = int(r["epoch"])
        row = out.setdefault(ep, {"epoch": ep})
        for k in keys:
            if k == "epoch":
                continue
            cv = _coerce(k, r.get(k))
            if cv is not None:
                row[k] = cv
    return out


def _run_path(run_id: str) -> str:
    """W&B path for `run_id`. Without an entity the API resolves the default
    one from your login, so an unset WANDB_ENTITY still works.
    """
    return f"{ENTITY}/{PROJECT}/{run_id}" if ENTITY else f"{PROJECT}/{run_id}"


def pull(api: wandb.Api, run_id: str, display_name: str) -> int:
    run = api.run(_run_path(run_id))

    # Core keys: required. If this fails, the run has no usable history.
    by_epoch = _pull_keys(run, CORE_KEYS)
    if not by_epoch:
        print(f"  [skip] {display_name}: no core history")
        return 0

    # Optional groups: best-effort.
    for group in OPTIONAL_KEY_GROUPS:
        try:
            extra = _pull_keys(run, group)
        except Exception:
            continue
        for ep, row in extra.items():
            target = by_epoch.setdefault(ep, {"epoch": ep})
            target.update({k: v for k, v in row.items() if k != "epoch"})

    rows = [by_epoch[e] for e in sorted(by_epoch)]
    out_dir = RESULTS_DIR / display_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    return len(rows)


def main() -> None:
    api = wandb.Api()
    n_ok = n_skip = 0
    total_epochs = 0
    for display_name, run_id in sorted(TARGETS.items()):
        try:
            n = pull(api, run_id, display_name)
        except Exception as exc:
            print(f"  [error] {display_name}: {exc}", file=sys.stderr)
            n = 0
        if n == 0:
            n_skip += 1
        else:
            n_ok += 1
            total_epochs += n
            print(f"  [ok]   {display_name}: {n} epochs")
    print(f"\nDone. ok={n_ok} skip={n_skip} total_epochs={total_epochs}")


if __name__ == "__main__":
    main()
