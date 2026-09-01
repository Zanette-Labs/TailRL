"""Compute epoch-0 val IoU{@50, @75, @90} for both architectures, before any training.

RL arch (LocalizationPolicy with K=50, softmax over bins): the prior at init is uniform
over bins, so iou_greedy = argmax-bin → centered box. Used by all RL methods + the
supervised softmax baselines (tailrl_population, ordinal_ce, cross_entropy).

Supervised arch (LocalizationRegressor): direct sigmoid xywh.  Used by mse, l1_*,
mse_iou_match, etc.

The two architectures share the ResNet-50 backbone but differ in the head, so
the epoch-0 numbers really do differ (head init randomness + sigmoid vs argmax).

Usage (from the repo root, on one GPU):
    python experiments/imagenet_localization/scripts/compute_epoch0_iou.py

Outputs:
  - $TAILRL_RESULTS_DIR/epoch0_iou.json   (per-seed values + mean/std)
  - stdout: a printed summary of the 6 numbers
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Also runnable as a plain file path, which needs the repo root on sys.path
# before the package imports below.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.imagenet_localization import paths  # noqa: E402
from experiments.imagenet_localization.datasets.data import ImageNetLocDataset, build_collate_fn  # noqa: E402
from experiments.imagenet_localization.evaluation.evaluate import evaluate_localization  # noqa: E402
from experiments.imagenet_localization.models.model import LocalizationPolicy, LocalizationRegressor  # noqa: E402
from experiments.imagenet_localization.run import evaluate_mse_regressor  # noqa: E402

DATA_DIR = paths.require_imagenet_dir()
K = 50
SEEDS = [42, 43, 44, 45]
BATCH_SIZE = 128
NUM_WORKERS = 8
OUT_PATH = Path(paths.results_dir()) / "epoch0_iou.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Build val loader once (it's deterministic; seed only affects the model init).
val_dataset = ImageNetLocDataset(
    root_dir=DATA_DIR, split="val", K=K, subsample=None, seed=42, train_aug=False,
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    pin_memory=True, collate_fn=build_collate_fn(), drop_last=False,
)
print(f"Val dataset: {len(val_dataset):,} images, {len(val_loader):,} batches")

results: dict = {"rl": {}, "regressor": {}}

# ── RL arch: LocalizationPolicy ────────────────────────────────────────────
print("\n=== RL arch (LocalizationPolicy, K=50, softmax) ===")
for seed in SEEDS:
    t0 = time.time()
    model = LocalizationPolicy(K=K, pretrained=True, seed=seed).to(device)
    model = model.to(memory_format=torch.channels_last)
    m = evaluate_localization(model, val_loader, K=K, device=device,
                              N_eval_samples=64, compute_expected=False)
    rec = {
        "iou_at_50": float(m["val/iou_at_50"]),
        "iou_at_75": float(m["val/iou_at_75"]),
        "iou_at_90": float(m["val/iou_at_90"]),
        "iou_greedy": float(m["val/iou_greedy"]),
    }
    results["rl"][str(seed)] = rec
    dt = time.time() - t0
    print(f"  seed={seed}  iou@50={rec['iou_at_50']:.4f}  iou@75={rec['iou_at_75']:.4f}  "
          f"iou@90={rec['iou_at_90']:.4f}  greedy={rec['iou_greedy']:.4f}  ({dt:.1f}s)")
    del model
    torch.cuda.empty_cache()

# ── Supervised regressor arch: LocalizationRegressor ───────────────────────
print("\n=== Supervised arch (LocalizationRegressor, sigmoid xywh) ===")
for seed in SEEDS:
    t0 = time.time()
    model = LocalizationRegressor(pretrained=True, seed=seed).to(device)
    model = model.to(memory_format=torch.channels_last)
    m = evaluate_mse_regressor(model, val_loader, device=device)
    rec = {
        "iou_at_50": float(m["val/iou_at_50"]),
        "iou_at_75": float(m["val/iou_at_75"]),
        "iou_at_90": float(m["val/iou_at_90"]),
        "iou_greedy": float(m["val/iou_greedy"]),
    }
    results["regressor"][str(seed)] = rec
    dt = time.time() - t0
    print(f"  seed={seed}  iou@50={rec['iou_at_50']:.4f}  iou@75={rec['iou_at_75']:.4f}  "
          f"iou@90={rec['iou_at_90']:.4f}  greedy={rec['iou_greedy']:.4f}  ({dt:.1f}s)")
    del model
    torch.cuda.empty_cache()

# ── Aggregate ──────────────────────────────────────────────────────────────
def agg(arch_dict, key):
    vals = [arch_dict[str(s)][key] for s in SEEDS]
    return float(np.mean(vals)), float(np.std(vals))

summary: dict = {}
for arch in ("rl", "regressor"):
    summary[arch] = {}
    for k in ("iou_at_50", "iou_at_75", "iou_at_90", "iou_greedy"):
        m, s = agg(results[arch], k)
        summary[arch][k] = {"mean": m, "std": s}
results["summary"] = summary

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {OUT_PATH}")
print("\n══ Summary (mean ± std over seeds 42-45) ══")
for arch_label, arch_key in [("RL arch        ", "rl"), ("Supervised arch", "regressor")]:
    s = summary[arch_key]
    print(f"  {arch_label}:  iou@50={s['iou_at_50']['mean']:.4f}±{s['iou_at_50']['std']:.4f}  "
          f"iou@75={s['iou_at_75']['mean']:.4f}±{s['iou_at_75']['std']:.4f}  "
          f"iou@90={s['iou_at_90']['mean']:.4f}±{s['iou_at_90']['std']:.4f}")
