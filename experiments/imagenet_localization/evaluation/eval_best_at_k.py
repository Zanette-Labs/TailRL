"""Compute unbiased best@k (best-of-k) IoU on the val set for one or more runs.

For each run we reload the *final* checkpoint, draw M iid rollouts per val image
from the policy, score each rollout's max-IoU vs the GT boxes, and reduce to the
unbiased best@k estimate (see ``bestk.best_at_k_unbiased``) for a grid of k.
Results (best@k per run, averaged over the val set) are written to JSON + CSV so
``plot_best_at_k.py`` can overlay vanilla-IoU vs percentile-reward runs.

Usage (module form, repo root on PYTHONPATH):

    python -m experiments.imagenet_localization.evaluation.eval_best_at_k \
        --run_dirs results/tailrl_K50_N256_seed43 results/grpo_K50_N256_seed43 ... \
        --data_dir "$IMAGENET_DIR" \
        --K 50 --M 4096 --ks 1,4,16,64,256,1024 \
        --out_json bestk.json --out_csv bestk.csv

The label for each run defaults to the run-dir basename; pass --labels to override.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import torch
from torch.utils.data import DataLoader

from experiments.imagenet_localization import paths
from experiments.imagenet_localization.core.bestk import best_at_k_unbiased
from experiments.imagenet_localization.datasets.data import ImageNetLocDataset, build_collate_fn
from experiments.imagenet_localization.evaluation.evaluate import sample_boxes_from_logits
from experiments.imagenet_localization.core.iou import batched_max_iou
from experiments.imagenet_localization.models.model import LocalizationPolicy


def resolve_final_ckpt(run_dir: str) -> str:
    """Return the path to the run's final checkpoint.

    Preference: checkpoints/final.pt (written at the last epoch), then a
    top-level final.pt, then last.pt (which is the epoch-30 model for a
    completed 30-epoch run). Raises if none exist.
    """
    candidates = [
        os.path.join(run_dir, "checkpoints", "final.pt"),
        os.path.join(run_dir, "final.pt"),
        os.path.join(run_dir, "last.pt"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"no final/last checkpoint found under {run_dir} (looked for {candidates})"
    )


def load_policy(ckpt_path: str, K: int, device: torch.device) -> LocalizationPolicy:
    """Build a LocalizationPolicy and load the checkpoint's model weights."""
    # pretrained=False: the saved state_dict fully overwrites the backbone, so
    # there is no need to download ImageNet weights here.
    model = LocalizationPolicy(K=K, pretrained=False)
    payload = torch.load(ckpt_path, map_location="cpu")
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.to(device).eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


@torch.no_grad()
def eval_best_at_k_over_val(
    model: LocalizationPolicy,
    val_loader: DataLoader,
    K: int,
    device: torch.device,
    M: int,
    ks: list[int],
    max_images: int | None = None,
) -> dict:
    """Stream over the val set, accumulating the unbiased best@k IoU.

    Returns a dict with best@k (list, aligned to ks), best@1 sanity (== mean IoU),
    the realized max over the M samples, and the number of val images used.
    """
    model.eval()
    ks_sorted = sorted(set(int(k) for k in ks))
    sum_bk = torch.zeros(len(ks_sorted), dtype=torch.float64)
    sum_mean = 0.0      # == best@1, sanity
    sum_realized_max = 0.0
    n_images = 0

    for batch in val_loader:
        images = batch["images"].to(device)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        gt_boxes = batch["gt_boxes"].to(device)
        gt_mask = batch["gt_mask"].to(device)
        B = images.shape[0]

        logits = model(images)                                   # dict (B, K)
        sampled = sample_boxes_from_logits(logits, M, K)         # (B, M, 4)
        ious = batched_max_iou(sampled, gt_boxes, gt_mask)       # (B, M)

        bk = best_at_k_unbiased(ious, ks_sorted)                 # (B, len(ks))
        sum_bk += bk.double().sum(dim=0).cpu()
        sum_mean += ious.mean(dim=1).double().sum().item()
        sum_realized_max += ious.max(dim=1).values.double().sum().item()
        n_images += B

        if max_images is not None and n_images >= max_images:
            break

    best_at_k = (sum_bk / n_images).tolist()
    return {
        "ks": ks_sorted,
        "best_at_k": best_at_k,
        "best_at_1_check_mean_iou": sum_mean / n_images,   # must match best_at_k[0]
        "realized_best_of_M": sum_realized_max / n_images,  # max over all M samples
        "M": M,
        "n_images": n_images,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unbiased best@k IoU on val for a set of runs.")
    p.add_argument("--run_dirs", nargs="+", required=True,
                   help="One or more run output dirs (final ckpt auto-resolved).")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Optional labels (default: run-dir basename).")
    p.add_argument("--data_dir", type=str, default=paths.imagenet_dir() or None,
                   help="Root ImageNet directory (default: $IMAGENET_DIR).")
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--M", type=int, default=4096, help="Rollouts/image for the unbiased estimate.")
    p.add_argument("--ks", type=str, default="1,4,16,64,256,1024")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0, help="Seed for reproducible rollout sampling.")
    p.add_argument("--max_images", type=int, default=None, help="Cap val images (smoke tests).")
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument("--out_csv", type=str, default=None)
    args = p.parse_args()
    if not args.data_dir:
        args.data_dir = paths.require_imagenet_dir()
    return args


def main() -> None:
    args = parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    assert args.M >= max(ks), f"M={args.M} must be >= max k={max(ks)} for an unbiased best@k"

    labels = args.labels or [os.path.basename(os.path.normpath(d)) for d in args.run_dirs]
    assert len(labels) == len(args.run_dirs), "labels must match run_dirs"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the val loader ONCE; reuse across all checkpoints (same val set).
    val_dataset = ImageNetLocDataset(root_dir=args.data_dir, split="val", K=args.K)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=build_collate_fn(),
        pin_memory=(device.type == "cuda"),
    )
    print(f"val images: {len(val_dataset)} | device: {device} | M={args.M} | ks={ks}", flush=True)

    results = []
    for label, run_dir in zip(labels, args.run_dirs):
        ckpt = resolve_final_ckpt(run_dir)
        print(f"\n[{label}] ckpt={ckpt}", flush=True)
        torch.manual_seed(args.seed)  # reproducible sampling; same RNG for every run
        model = load_policy(ckpt, args.K, device)
        res = eval_best_at_k_over_val(
            model, val_loader, args.K, device, args.M, ks, max_images=args.max_images,
        )
        res.update({"label": label, "run_dir": run_dir, "ckpt": ckpt})
        bk_str = "  ".join(f"@{k}={v:.4f}" for k, v in zip(res["ks"], res["best_at_k"]))
        print(f"[{label}] best@k: {bk_str}", flush=True)
        print(f"[{label}] sanity best@1={res['best_at_k'][0]:.4f} vs mean IoU={res['best_at_1_check_mean_iou']:.4f}; "
              f"realized best-of-{args.M}={res['realized_best_of_M']:.4f}; n={res['n_images']}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        results.append(res)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"ks": ks, "M": args.M, "runs": results}, f, indent=2)
    print(f"\nwrote {args.out_json}", flush=True)

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label", "k", "best_at_k"])
            for r in results:
                for k, v in zip(r["ks"], r["best_at_k"]):
                    w.writerow([r["label"], k, v])
        print(f"wrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
