"""Day-2 pilot smoke test — overfit TailRL on a single val image.

Picks a val image with:
- exactly 1 GT box of the one class
- "easy" size tier (primary box area ∈ [0.30, 0.70] of the image)

Runs 1000 TailRL training steps at K=10, N=64, Adam lr=3e-3 (no scheduler, no warmup).
Asserts final `greedy_iou > 0.8`. Prints `PILOT PASSED` on success.

If no easy-tier single-GT image can be found within the first 100 val entries,
the script reports PILOT FAILED with the reason.

Usage:
    python -m experiments.imagenet_localization.pilot [--data_dir PATH] [--k K] [--n N] [--steps S] [--lr LR] [--seed S]

Defaults match the spec; --data_dir falls back to $IMAGENET_DIR.
"""

import argparse
import sys
import torch
import torch.nn.functional as F

from experiments.imagenet_localization import paths
from experiments.imagenet_localization.datasets.data import (
    ImageNetLocDataset,
    MAX_M,
)
from experiments.imagenet_localization.core.iou import batched_max_iou, clamp_boxes_to_image
from experiments.imagenet_localization.models.model import LocalizationPolicy
from experiments.imagenet_localization.core.advantages import tailrl_advantage
from experiments.imagenet_localization.core.losses import factored_sample_log_prob
from experiments.imagenet_localization.evaluation.evaluate import greedy_box_from_logits
from experiments.imagenet_localization.core.iou import box_iou_xywh

HEAD_NAMES = ("x_c", "y_c", "w", "h")


def parse_args():
    p = argparse.ArgumentParser(
        description="Day-2 pilot smoke test: overfit TailRL on a single val image."
    )
    p.add_argument("--data_dir", default=paths.imagenet_dir() or None,
                   help="Root ImageNet directory (default: $IMAGENET_DIR)")
    p.add_argument("--K", type=int, default=10,
                   help="Number of bins per coordinate (default: 10)")
    p.add_argument("--N", type=int, default=64,
                   help="Number of rollout samples per step (default: 64)")
    p.add_argument("--steps", type=int, default=1000,
                   help="Number of TailRL training steps (default: 1000)")
    p.add_argument("--lr", type=float, default=3e-3,
                   help="Adam learning rate (default: 3e-3)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("--search_limit", type=int, default=100,
                   help="Max val images to scan for an easy single-GT image (default: 100)")
    p.add_argument("--target_iou", type=float, default=0.8,
                   help="Target greedy IoU to declare PILOT PASSED (default: 0.8)")
    p.add_argument("--device", default=None,
                   help="torch device; autodetect if omitted")
    p.add_argument("--image_id", default=None,
                   help="Override: use this specific val image_id instead of dynamic selection")
    args = p.parse_args()
    if not args.data_dir:
        args.data_dir = paths.require_imagenet_dir()
    return args


def pick_pilot_image(dataset, search_limit: int):
    """Return (image, gt_boxes, gt_mask, image_id) for the first val image
    with exactly 1 real GT box and 'easy' size tier (area ∈ [0.3, 0.7]).

    Searches through the first `search_limit` entries in `dataset` (in order).
    The dataset is already shuffled deterministically via its seed, so this
    gives a reproducible selection.

    Raises RuntimeError if no such image is found within `search_limit`.
    """
    found = 0
    limit = min(search_limit, len(dataset))

    for idx in range(limit):
        sample = dataset[idx]
        gt_mask = sample["gt_mask"]          # (MAX_M,) bool
        gt_boxes = sample["gt_boxes"]        # (MAX_M, 4) xywh

        # Filter 1: exactly 1 real GT box
        num_real = gt_mask.sum().item()
        if num_real != 1:
            continue

        # Filter 2: easy-tier — primary box area ∈ [0.30, 0.70]
        # Primary box is at index 0 (always real when num_real >= 1)
        w = gt_boxes[0, 2].item()
        h = gt_boxes[0, 3].item()
        area = w * h
        if not (0.30 <= area <= 0.70):
            continue

        # Passed both filters
        return (
            sample["image"],
            sample["gt_boxes"],
            sample["gt_mask"],
            sample["image_id"],
        )

    raise RuntimeError(
        f"No easy-tier single-GT val image found within the first {search_limit} entries. "
        f"Try increasing --search_limit."
    )


def _find_image_by_id(dataset, image_id: str):
    """Linear scan for a specific image_id. Returns the sample dict or raises RuntimeError."""
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample["image_id"] == image_id:
            return sample
    raise RuntimeError(
        f"image_id {image_id!r} not found in the val dataset. "
        f"Check the spelling and ensure the dataset is loaded without subsample."
    )


def run_pilot(args, wandb_run=None) -> tuple[bool, float, str]:
    """Run the pilot. Returns (passed, final_iou, reason)."""
    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # --- Find a suitable image ---
    val_dataset = ImageNetLocDataset(
        root_dir=args.data_dir, split="val", K=args.K,
        subsample=None, seed=args.seed, train_aug=False,
    )

    if args.image_id is not None:
        # Use the caller-specified image, bypassing dynamic selection
        sample = _find_image_by_id(val_dataset, args.image_id)
        image    = sample["image"]
        gt_boxes = sample["gt_boxes"]
        gt_mask  = sample["gt_mask"]
        image_id = sample["image_id"]
    else:
        image, gt_boxes, gt_mask, image_id = pick_pilot_image(val_dataset, args.search_limit)

    image    = image.unsqueeze(0).to(device)        # (1, 3, 224, 224)
    gt_boxes = gt_boxes.unsqueeze(0).to(device)     # (1, MAX_M, 4)
    gt_mask  = gt_mask.unsqueeze(0).to(device)      # (1, MAX_M)

    print(f"Pilot image: {image_id}")

    # --- Build model ---
    model = LocalizationPolicy(K=args.K, pretrained=True, seed=args.seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Training loop ---
    model.train()
    for step in range(1, args.steps + 1):
        logits = model(image)                                 # dict (1, K)
        log_probs = {h: F.log_softmax(logits[h], -1) for h in HEAD_NAMES}

        with torch.no_grad():
            probs = {h: log_probs[h].exp() for h in HEAD_NAMES}
            samples = {h: torch.multinomial(probs[h], args.N, replacement=True) for h in HEAD_NAMES}
            coords = torch.stack(
                [(samples[h].float() + 0.5) / args.K for h in HEAD_NAMES],
                dim=-1,
            )                                                  # (1, N, 4)
            sampled_boxes = clamp_boxes_to_image(coords)
            rewards = batched_max_iou(sampled_boxes, gt_boxes, gt_mask)  # (1, N)
            advantages = tailrl_advantage(rewards[0]).unsqueeze(0)          # (1, N)

        sample_log_probs = factored_sample_log_prob(log_probs, samples)  # (1, N)

        loss = -(advantages.detach() * sample_log_probs).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        mean_reward = rewards.mean().item()
        if step % 100 == 0 or step == 1:
            print(f"step {step:4d}  loss={loss.item():.4f}  reward_mean={mean_reward:.3f}")

    # --- Greedy IoU at the end ---
    model.eval()
    with torch.no_grad():
        final_logits = model(image)
        greedy = greedy_box_from_logits(final_logits, args.K)  # (1, 4)
        greedy_clamped = clamp_boxes_to_image(greedy)
        real_gt = gt_boxes[0][gt_mask[0]]                      # (num_real, 4)
        # Max IoU against real GT (single-GT image so real_gt has 1 row)
        ious = box_iou_xywh(greedy_clamped, real_gt[0])
        final_iou = ious.max().item()

    passed = final_iou > args.target_iou
    reason = (
        f"final_greedy_iou={final_iou:.3f} {'>' if passed else '<='} target={args.target_iou}"
    )
    return passed, final_iou, reason


def main():
    args = parse_args()
    try:
        passed, final_iou, reason = run_pilot(args)
    except RuntimeError as e:
        print(f"PILOT FAILED: {e}")
        sys.exit(1)

    if passed:
        print(f"PILOT PASSED: {reason}")
        sys.exit(0)
    else:
        print(f"PILOT FAILED: {reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()
