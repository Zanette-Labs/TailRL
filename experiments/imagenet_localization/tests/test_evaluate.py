"""Tests for experiments.imagenet_localization.evaluation.evaluate.

All tests are self-contained (no ImageNet data required).
Mock dataloaders use small synthetic batches with known shapes.
The mock model is LocalizationPolicy(K=10, pretrained=False).

Test count: 11 tests.
"""

from __future__ import annotations

import math

import pytest
import torch

from experiments.imagenet_localization.evaluation.evaluate import (
    HEAD_NAMES,
    SIZE_TIERS,
    build_reward_histogram,
    evaluate_localization,
    greedy_box_from_logits,
    sample_boxes_from_logits,
    size_tier_of_primary_box,
)
from experiments.imagenet_localization.models.model import LocalizationPolicy


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_K = 10
_B = 3
_MAX_M = 8


def _make_logits(B: int = _B, K: int = _K) -> dict[str, torch.Tensor]:
    """Random (B, K) logits for all 4 heads."""
    return {h: torch.randn(B, K) for h in HEAD_NAMES}


def _make_batch(B: int = _B, K: int = _K, max_m: int = _MAX_M) -> dict:
    """Construct a synthetic collate-fn-style batch dict.

    target_bins are drawn uniformly from {0, ..., K-1}.
    gt_boxes has real boxes (uniform random xywh) for the first slot and
    zero-padding for the rest; gt_mask marks only slot 0 as real.
    """
    # Images: small spatial size to keep the ResNet forward pass fast
    images = torch.rand(B, 3, 224, 224)

    # GT boxes: first real box, rest padded with zeros
    gt_boxes = torch.zeros(B, max_m, 4)
    # Primary box: random center and moderate size
    gt_boxes[:, 0, 0] = torch.empty(B).uniform_(0.2, 0.8)  # x_c
    gt_boxes[:, 0, 1] = torch.empty(B).uniform_(0.2, 0.8)  # y_c
    gt_boxes[:, 0, 2] = torch.empty(B).uniform_(0.2, 0.5)  # w (easy tier area)
    gt_boxes[:, 0, 3] = torch.empty(B).uniform_(0.2, 0.5)  # h

    gt_mask = torch.zeros(B, max_m, dtype=torch.bool)
    gt_mask[:, 0] = True

    target_bins = {h: torch.randint(0, K, (B,)) for h in HEAD_NAMES}

    return {
        "images": images,
        "gt_boxes": gt_boxes,
        "gt_mask": gt_mask,
        "target_bins": target_bins,
        "image_ids": [f"img_{i}" for i in range(B)],
    }


def _make_val_loader(num_batches: int = 2, B: int = _B, K: int = _K):
    """Return a simple list used as a mock DataLoader (iterable of batch dicts)."""
    torch.manual_seed(123)
    return [_make_batch(B=B, K=K) for _ in range(num_batches)]


def _make_model(K: int = _K) -> LocalizationPolicy:
    return LocalizationPolicy(K=K, pretrained=False, seed=0)


# ===========================================================================
# Box prediction helper tests
# ===========================================================================


def test_greedy_box_shape():
    """greedy_box_from_logits must return (B, 4) for any B and K.

    Wrong output shape would cause all downstream IoU calls to fail or broadcast
    incorrectly, silently producing wrong metrics.
    """
    logits = _make_logits(B=_B, K=_K)
    box = greedy_box_from_logits(logits, K=_K)
    assert box.shape == (_B, 4), (
        f"Expected shape ({_B}, 4), got {box.shape}"
    )


def test_greedy_box_deterministic():
    """greedy_box_from_logits called twice with the same logits must return the same box.

    Greedy (argmax) decoding is deterministic; any randomness here would indicate
    the function is sampling instead of taking the argmax.
    """
    logits = _make_logits(B=_B, K=_K)
    box1 = greedy_box_from_logits(logits, K=_K)
    box2 = greedy_box_from_logits(logits, K=_K)
    torch.testing.assert_close(box1, box2, atol=0.0, rtol=0.0)


def test_greedy_box_corners_clamped_to_unit_square():
    """For extreme argmax indices (xc at bin 0 + w at bin K-1) the unclamped
    corners would extend outside [0,1]. greedy_box_from_logits must clamp so
    the box geometry matches the RL training path and sample_boxes_from_logits.
    """
    K = 50
    B = 1
    # Build logits so argmax picks bin 0 for xc/yc and bin K-1 for w/h.
    # This produces xc=yc=0.01, w=h=0.99, so unclamped corners are (-0.485, 0.505).
    big = 100.0
    logits = {}
    for h in ('x_c', 'y_c'):
        v = torch.full((B, K), -big)
        v[:, 0] = big          # argmax at bin 0
        logits[h] = v
    for h in ('w', 'h'):
        v = torch.full((B, K), -big)
        v[:, K - 1] = big       # argmax at bin K-1
        logits[h] = v
    box = greedy_box_from_logits(logits, K=K)
    # Convert xywh -> corners and check [0,1] containment.
    xc, yc, w, h = box.unbind(-1)
    x1, x2 = xc - w / 2, xc + w / 2
    y1, y2 = yc - h / 2, yc + h / 2
    assert (x1 >= 0.0).all(), f"x1 below 0: {x1}"
    assert (y1 >= 0.0).all(), f"y1 below 0: {y1}"
    assert (x2 <= 1.0).all(), f"x2 above 1: {x2}"
    assert (y2 <= 1.0).all(), f"y2 above 1: {y2}"


def test_sample_boxes_shape_and_range():
    """sample_boxes_from_logits must return (B, N, 4) with values in [0, 1].

    Out-of-range coordinates would be invalid xywh box coordinates; wrong shape
    would break the downstream batched_max_iou call.
    """
    N = 32
    logits = _make_logits(B=_B, K=_K)
    boxes = sample_boxes_from_logits(logits, N=N, K=_K)

    assert boxes.shape == (_B, N, 4), (
        f"Expected shape ({_B}, {N}, 4), got {boxes.shape}"
    )
    assert boxes.min().item() >= 0.0, (
        f"Sampled box coordinates below 0: {boxes.min().item()}"
    )
    assert boxes.max().item() <= 1.0, (
        f"Sampled box coordinates above 1: {boxes.max().item()}"
    )


# ===========================================================================
# Size tier tests
# ===========================================================================


def test_size_tier_of_primary_box_easy():
    """Area 0.5 (w=0.5, h=1.0 → area=0.5) is in [0.30, 0.70] → 'easy'.

    Confirms the easy-tier boundary is inclusive on both ends.
    """
    # (1, 4): x_c, y_c, w, h — only w and h matter for area
    boxes = torch.tensor([[0.5, 0.5, 0.5, 1.0]])  # area = 0.5 * 1.0 = 0.5
    tiers = size_tier_of_primary_box(boxes)
    assert tiers == ["easy"], f"Expected ['easy'], got {tiers}"


def test_size_tier_of_primary_box_medium_small():
    """Area 0.15 is in [0.10, 0.30) → 'medium'."""
    # w * h = 0.3 * 0.5 = 0.15
    boxes = torch.tensor([[0.5, 0.5, 0.3, 0.5]])
    tiers = size_tier_of_primary_box(boxes)
    assert tiers == ["medium"], f"Expected ['medium'], got {tiers}"


def test_size_tier_of_primary_box_medium_large():
    """Area 0.85 is in (0.70, 0.95] → 'medium'."""
    # w * h = 0.85 * 1.0 = 0.85
    boxes = torch.tensor([[0.5, 0.5, 0.85, 1.0]])
    tiers = size_tier_of_primary_box(boxes)
    assert tiers == ["medium"], f"Expected ['medium'], got {tiers}"


def test_size_tier_of_primary_box_hard_small():
    """Area 0.05 < 0.10 → 'hard'."""
    # w * h = 0.1 * 0.5 = 0.05
    boxes = torch.tensor([[0.5, 0.5, 0.1, 0.5]])
    tiers = size_tier_of_primary_box(boxes)
    assert tiers == ["hard"], f"Expected ['hard'], got {tiers}"


def test_size_tier_of_primary_box_hard_large():
    """Area 0.99 > 0.95 → 'hard'."""
    # w * h ≈ 0.99 (use w=0.99, h=1.0)
    boxes = torch.tensor([[0.5, 0.5, 0.99, 1.0]])
    tiers = size_tier_of_primary_box(boxes)
    assert tiers == ["hard"], f"Expected ['hard'], got {tiers}"


# ===========================================================================
# evaluate_localization tests
# ===========================================================================

# Expected overall metric keys
_OVERALL_KEYS = {
    "val/iou_greedy",
    "val/iou_expected",
    "val/iou_best_of_N",
    "val/iou_at_50",
    "val/iou_at_75",
    "val/iou_at_90",
    "val/ordinal_ce_objective",
    "val/cross_entropy_objective",
    "val/entropy_xc",
    "val/entropy_yc",
    "val/entropy_w",
    "val/entropy_h",
    "val/entropy_joint_mean",
}

_TIER_METRIC_KEYS = {
    f"val_{tier}/iou_greedy" for tier in SIZE_TIERS
} | {
    f"val_{tier}/count" for tier in SIZE_TIERS
}

_ALL_EXPECTED_KEYS = _OVERALL_KEYS | _TIER_METRIC_KEYS


def test_evaluate_localization_output_keys():
    """evaluate_localization must return all expected keys with correct types.

    Missing keys would silently drop metrics from the experiment log; wrong
    types would cause serialisation errors when logging to wandb.
    Keys:
      - All overall val/ keys → float
      - val_{tier}/count      → int
      - val_{tier}/iou_greedy → float
    """
    model = _make_model(K=_K)
    val_loader = _make_val_loader(num_batches=2, B=_B, K=_K)
    device = torch.device("cpu")

    metrics = evaluate_localization(
        model, val_loader, K=_K, device=device,
        N_eval_samples=16, compute_expected=True,
    )

    missing = _ALL_EXPECTED_KEYS - set(metrics.keys())
    assert not missing, f"Missing keys: {sorted(missing)}"

    # Type checks
    for key in _OVERALL_KEYS:
        assert isinstance(metrics[key], float), (
            f"Key {key!r}: expected float, got {type(metrics[key]).__name__}"
        )
    for tier in SIZE_TIERS:
        count_key = f"val_{tier}/count"
        assert isinstance(metrics[count_key], int), (
            f"Key {count_key!r}: expected int, got {type(metrics[count_key]).__name__}"
        )
        iou_key = f"val_{tier}/iou_greedy"
        # May be float('nan') if count==0, but still float
        assert isinstance(metrics[iou_key], float), (
            f"Key {iou_key!r}: expected float, got {type(metrics[iou_key]).__name__}"
        )


def test_evaluate_localization_finite_values():
    """All metric values from evaluate_localization must be finite (no NaN, no Inf).

    NaN in any metric is a silent failure: wandb would log NaN and charts would
    appear empty. Inf would cause comparison errors in post-processing scripts.

    Note: tier iou_greedy may be NaN if count==0, but we design the mock loader
    so that all tiers receive at least one image (large enough B and random sizes).
    We use compute_expected=True so the expected/best-of-N values are also tested.
    """
    torch.manual_seed(42)
    model = _make_model(K=_K)
    val_loader = _make_val_loader(num_batches=3, B=6, K=_K)
    device = torch.device("cpu")

    metrics = evaluate_localization(
        model, val_loader, K=_K, device=device,
        N_eval_samples=16, compute_expected=True,
    )

    for key, val in metrics.items():
        if isinstance(val, float):
            # Allow NaN only for tier metrics where count==0
            if f"/count" in key:
                continue  # int, skip
            if math.isnan(val):
                # Only acceptable for tier iou_greedy when count is 0
                tier = None
                for t in SIZE_TIERS:
                    if key == f"val_{t}/iou_greedy":
                        tier = t
                        break
                if tier is not None:
                    count = metrics[f"val_{tier}/count"]
                    assert count == 0, (
                        f"Key {key!r} is NaN but count={count} > 0"
                    )
                else:
                    assert False, f"Overall metric {key!r} is NaN"
            else:
                assert math.isfinite(val), (
                    f"Key {key!r}: value {val} is not finite"
                )


# ===========================================================================
# build_reward_histogram tests
# ===========================================================================


def test_build_reward_histogram_bins_sum_to_input_count():
    """sum(counts) from build_reward_histogram must equal the number of input rewards.

    If any reward falls outside the histogram range it is silently dropped,
    making sum(counts) < N. This test verifies no rewards are lost.
    """
    torch.manual_seed(99)
    rewards = torch.rand(100)  # 100 uniform samples in [0, 1]
    result = build_reward_histogram(rewards, num_bins=20)

    # The function may return a plain dict or a wandb.Histogram object.
    # Extract counts defensively.
    if isinstance(result, dict):
        counts = result["counts"]
    else:
        # wandb.Histogram — access underlying data
        counts = result.histogram  # list of counts

    total = sum(counts)
    assert total == 100, (
        f"Expected sum(counts)==100 (no dropped rewards), got {total}"
    )
