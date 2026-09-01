"""Evaluation utilities for the ImageNet localization RL experiment.

Runs full val-set evaluation and returns a flat dict of metrics with
``val/`` prefix (overall) and ``val_{tier}/`` prefix (per-size tier).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.imagenet_localization.core.iou import (
    batched_max_iou,
    box_iou_xywh,
    clamp_boxes_to_image,
)
from experiments.imagenet_localization.core.losses import (
    localization_ordinal_ce_loss,
    localization_cross_entropy_loss,
)

HEAD_NAMES: tuple[str, ...] = ("x_c", "y_c", "w", "h")

SIZE_TIERS = ("easy", "medium", "hard")


# ---------------------------------------------------------------------------
# Box prediction helpers
# ---------------------------------------------------------------------------


def greedy_box_from_logits(
    logits: dict[str, torch.Tensor],  # each (B, K)
    K: int,
) -> torch.Tensor:
    """Argmax bin per head → bin center → (B, 4) xywh box, clamped to [0, 1].

    The clamp mirrors clamp_boxes_to_image used in the RL training path and in
    sample_boxes_from_logits, so the box geometry seen at eval matches the box
    geometry seen during training.
    """
    coords = []
    for h in HEAD_NAMES:
        idx = logits[h].argmax(dim=-1)  # (B,)
        coord = (idx.float() + 0.5) / K  # (B,)
        coords.append(coord)
    boxes = torch.stack(coords, dim=-1)  # (B, 4)
    return clamp_boxes_to_image(boxes)


def sample_boxes_from_logits(
    logits: dict[str, torch.Tensor],  # each (B, K)
    N: int,
    K: int,
) -> torch.Tensor:
    """Sample N boxes per image via multinomial. Returns (B, N, 4) clamped xywh."""
    probs = {h: F.softmax(logits[h], dim=-1) for h in HEAD_NAMES}
    samples = {h: torch.multinomial(probs[h], N, replacement=True) for h in HEAD_NAMES}
    coords = torch.stack(
        [(samples[h].float() + 0.5) / K for h in HEAD_NAMES],
        dim=-1,
    )  # (B, N, 4)
    return clamp_boxes_to_image(coords)


# ---------------------------------------------------------------------------
# Size tier assignment
# ---------------------------------------------------------------------------


def size_tier_of_primary_box(gt_boxes_primary: torch.Tensor) -> list[str]:
    """Given (B, 4) primary GT boxes in xywh, return a list of tier labels
    ('easy' | 'medium' | 'hard') of length B.

    Tier boundaries (based on area = w * h):
      - easy:   0.30 <= area <= 0.70
      - medium: (0.10 <= area < 0.30) OR (0.70 < area <= 0.95)
      - hard:   area < 0.10 OR area > 0.95

    Boundary convention (inclusive at both ends of each interval):
      - area == 0.30  -> easy   (lower easy boundary, inclusive)
      - area == 0.70  -> easy   (upper easy boundary, inclusive)
      - area == 0.10  -> medium (lower medium boundary, inclusive)
      - area == 0.95  -> medium (upper medium boundary, inclusive)
      - area < 0.10   -> hard
      - area > 0.95   -> hard
    """
    # area = w * h
    areas = gt_boxes_primary[:, 2] * gt_boxes_primary[:, 3]  # (B,)
    tiers: list[str] = []
    for a in areas.tolist():
        if 0.30 <= a <= 0.70:
            tiers.append("easy")
        elif (0.10 <= a < 0.30) or (0.70 < a <= 0.95):
            tiers.append("medium")
        else:
            # a < 0.10 or a > 0.95
            tiers.append("hard")
    return tiers


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_localization(
    model,                          # LocalizationPolicy on device
    val_loader,                     # DataLoader yielding collate-fn batches
    K: int,
    device: torch.device,
    N_eval_samples: int = 1024,
    compute_expected: bool = True,  # set False for speed-tests
) -> dict:
    """Run full val evaluation.

    Returns a FLAT dict with keys:

      Overall
      - val/iou_greedy             (argmax -> box -> max IoU vs GT)
      - val/iou_expected           (sampling estimate with N_eval_samples)
      - val/iou_best_of_N          (max over the sampled N rollouts)
      - val/iou_at_50              (CorLoc@0.5: mean(iou_greedy > 0.5))
      - val/iou_at_75              (CorLoc@0.75)
      - val/iou_at_90              (CorLoc@0.9)
      - val/ordinal_ce_objective   (scalar, averaged over val set)
      - val/cross_entropy_objective     (scalar, averaged over val set)
      - val/entropy_xc, /entropy_yc, /entropy_w, /entropy_h
      - val/entropy_joint_mean    (mean of the four)

      Per-tier (for tier in easy / medium / hard)
      - val_easy/iou_greedy, val_medium/iou_greedy, val_hard/iou_greedy
      - val_easy/count, val_medium/count, val_hard/count

    The function iterates over val_loader ONCE; uses torch.no_grad; moves
    everything to CPU before aggregating to keep peak GPU memory bounded.
    """
    model.eval()

    # Accumulators
    greedy_ious: list[torch.Tensor] = []          # (B,) per batch
    expected_ious: list[torch.Tensor] = []        # (B,) per batch
    best_of_n_ious: list[torch.Tensor] = []       # (B,) per batch
    ordinal_ce_vals: list[float] = []
    cross_entropy_vals: list[float] = []

    # Per-head entropy accumulators: list of (B,) tensors
    entropy_accum: dict[str, list[torch.Tensor]] = {h: [] for h in HEAD_NAMES}

    # Per-tier IoU and count
    tier_iou_accum: dict[str, list[torch.Tensor]] = {t: [] for t in SIZE_TIERS}
    tier_count: dict[str, int] = {t: 0 for t in SIZE_TIERS}

    for batch in val_loader:
        images = batch["images"].to(device)        # (B, 3, H, W)
        gt_boxes = batch["gt_boxes"].to(device)    # (B, M, 4)
        gt_mask = batch["gt_mask"].to(device)      # (B, M)
        target_bins = {h: batch["target_bins"][h].to(device) for h in HEAD_NAMES}

        B = images.shape[0]

        # Forward pass
        logits = model(images)  # dict of (B, K)

        # ----------------------------------------------------------------
        # 1. Greedy IoU
        # ----------------------------------------------------------------
        greedy_boxes = greedy_box_from_logits(logits, K)  # (B, 4)
        # Expand to (B, 1, 4) for batched_max_iou
        greedy_iou_batch = batched_max_iou(
            greedy_boxes.unsqueeze(1), gt_boxes, gt_mask
        ).squeeze(1)  # (B,)
        greedy_ious.append(greedy_iou_batch.cpu())

        # ----------------------------------------------------------------
        # 2. Sampled IoU (expected and best-of-N)
        # ----------------------------------------------------------------
        if compute_expected:
            sampled_boxes = sample_boxes_from_logits(logits, N_eval_samples, K)  # (B, N, 4)
            # (B, N) rewards
            sample_ious = batched_max_iou(sampled_boxes, gt_boxes, gt_mask)  # (B, N)
            expected_iou_batch = sample_ious.mean(dim=1)   # (B,)
            best_of_n_iou_batch = sample_ious.max(dim=1).values  # (B,)
            expected_ious.append(expected_iou_batch.cpu())
            best_of_n_ious.append(best_of_n_iou_batch.cpu())

        # ----------------------------------------------------------------
        # 3. Entropy per head
        # ----------------------------------------------------------------
        for h in HEAD_NAMES:
            probs_h = F.softmax(logits[h], dim=-1)  # (B, K)
            # Entropy: -sum_k p_k * log(p_k), clamped to avoid log(0)
            log_probs_h = torch.log(probs_h.clamp(min=1e-12))
            entropy_h = -(probs_h * log_probs_h).sum(dim=-1)  # (B,)
            entropy_accum[h].append(entropy_h.cpu())

        # ----------------------------------------------------------------
        # 4. CE objectives
        # ----------------------------------------------------------------
        ordinal_ce = localization_ordinal_ce_loss(logits, target_bins, K)
        cross_entropy = localization_cross_entropy_loss(logits, target_bins)
        ordinal_ce_vals.append(ordinal_ce.item())
        cross_entropy_vals.append(cross_entropy.item())

        # ----------------------------------------------------------------
        # 5. Per-tier IoU (based on greedy IoU)
        # ----------------------------------------------------------------
        primary_boxes = gt_boxes[:, 0, :].cpu()  # (B, 4) — first GT box
        tiers = size_tier_of_primary_box(primary_boxes)
        greedy_iou_cpu = greedy_iou_batch.cpu()
        for b_idx, tier in enumerate(tiers):
            tier_iou_accum[tier].append(greedy_iou_cpu[b_idx].unsqueeze(0))
            tier_count[tier] += 1

    # ----------------------------------------------------------------
    # Aggregate
    # ----------------------------------------------------------------
    all_greedy = torch.cat(greedy_ious)  # (total_images,)

    metrics: dict = {}

    # Overall greedy
    metrics["val/iou_greedy"] = all_greedy.mean().item()

    # Expected and best-of-N
    if compute_expected and expected_ious:
        all_expected = torch.cat(expected_ious)
        all_best_of_n = torch.cat(best_of_n_ious)
        metrics["val/iou_expected"] = all_expected.mean().item()
        metrics["val/iou_best_of_N"] = all_best_of_n.mean().item()
    else:
        metrics["val/iou_expected"] = float("nan")
        metrics["val/iou_best_of_N"] = float("nan")

    # CorLoc
    metrics["val/iou_at_50"] = (all_greedy > 0.5).float().mean().item()
    metrics["val/iou_at_75"] = (all_greedy > 0.75).float().mean().item()
    metrics["val/iou_at_90"] = (all_greedy > 0.9).float().mean().item()

    # CE objectives — simple mean of per-batch averages
    # (each batch contributes one averaged scalar; for exact per-image average
    # we'd need to weight by batch size, but batches are uniform so this is fine)
    metrics["val/ordinal_ce_objective"] = float(
        sum(ordinal_ce_vals) / len(ordinal_ce_vals)
    ) if ordinal_ce_vals else float("nan")
    metrics["val/cross_entropy_objective"] = float(
        sum(cross_entropy_vals) / len(cross_entropy_vals)
    ) if cross_entropy_vals else float("nan")

    # Entropy per head and joint mean
    head_key_map = {"x_c": "xc", "y_c": "yc", "w": "w", "h": "h"}
    entropy_means = []
    for h in HEAD_NAMES:
        all_entropy_h = torch.cat(entropy_accum[h])
        mean_ent = all_entropy_h.mean().item()
        metrics[f"val/entropy_{head_key_map[h]}"] = mean_ent
        entropy_means.append(mean_ent)
    metrics["val/entropy_joint_mean"] = float(sum(entropy_means) / len(entropy_means))

    # Per-tier metrics
    for tier in SIZE_TIERS:
        count = tier_count[tier]
        metrics[f"val_{tier}/count"] = count
        if count > 0:
            all_tier_iou = torch.cat(tier_iou_accum[tier])
            metrics[f"val_{tier}/iou_greedy"] = all_tier_iou.mean().item()
        else:
            metrics[f"val_{tier}/iou_greedy"] = float("nan")

    return metrics


# ---------------------------------------------------------------------------
# Reward histogram helper
# ---------------------------------------------------------------------------


def build_reward_histogram(
    rewards: torch.Tensor,  # (total_rollouts,) flat
    num_bins: int = 20,
) -> dict:
    """Build a dict suitable for wandb.Histogram. If wandb is not available,
    returns a plain dict with 'counts' (list[int]) and 'edges' (list[float])
    so the caller can still log it as a table.

    Edge cases handled:
    - Empty tensor: returns counts=[0]*num_bins with edges linspace(0,1).
    - All identical values: all mass in one bin; neighbouring bins have 0 count.
    - Non-flat rewards: standard histogram binning via torch.histc.

    Note: this helper is decoupled from wandb so it can be unit-tested.
    """
    if rewards.numel() == 0:
        # Return empty histogram with sensible edges
        edges = torch.linspace(0.0, 1.0, num_bins + 1).tolist()
        counts = [0] * num_bins
        return {"counts": counts, "edges": edges}

    r_min = rewards.min().item()
    r_max = rewards.max().item()

    # When all values are identical, torch.histc requires min < max.
    # We widen the range slightly so all mass lands in the centre bin.
    if r_min == r_max:
        r_min = r_min - 0.5
        r_max = r_max + 0.5

    hist = torch.histc(rewards.float(), bins=num_bins, min=r_min, max=r_max)
    edges = torch.linspace(r_min, r_max, num_bins + 1).tolist()
    counts = hist.long().tolist()

    try:
        import wandb  # optional dependency
        return wandb.Histogram(np_histogram=(counts, edges))
    except ImportError:
        return {"counts": counts, "edges": edges}
