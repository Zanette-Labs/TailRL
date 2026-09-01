"""Tests for experiments.imagenet_localization.core.iou.

Each test has a docstring stating the invariant and why failure matters.
Hand-computed expected values are derived in comments directly above
the relevant assertion.

All tests run on CPU and complete well under 10 seconds.
"""

from __future__ import annotations

import torch
import pytest

from experiments.imagenet_localization.core.iou import (
    EPS,
    batched_max_iou,
    box_iou_xywh,
    clamp_boxes_to_image,
    corners_to_xywh,
    max_iou_over_gt,
    xywh_to_corners,
)


# ===========================================================================
# xywh <-> corners roundtrip
# ===========================================================================


def test_xywh_corners_roundtrip():
    """xywh -> corners -> xywh must be a lossless roundtrip.

    Failure means either conversion direction loses information, which would
    corrupt every downstream IoU calculation.
    """
    # 50 random boxes with coordinates in (0, 1); w, h in (0, 0.5) to avoid
    # negative corners after conversion.
    boxes = torch.rand(50, 4)
    # Keep w, h positive and small enough to stay reasonable.
    boxes[:, 2:] = boxes[:, 2:].abs() * 0.5 + 0.01

    corners = xywh_to_corners(boxes)
    recovered = corners_to_xywh(corners)
    torch.testing.assert_close(recovered, boxes)


def test_xywh_to_corners_values():
    """xywh_to_corners must produce the correct corner coordinates.

    The formula x1=x_c-w/2, y1=y_c-h/2, x2=x_c+w/2, y2=y_c+h/2 is the
    definition of the format; any error here cascades to all IoU results.
    """
    # Box: center=(0.5, 0.5), w=1.0, h=1.0 (unit square centered at origin)
    # x1 = 0.5 - 0.5 = 0.0, y1 = 0.5 - 0.5 = 0.0
    # x2 = 0.5 + 0.5 = 1.0, y2 = 0.5 + 0.5 = 1.0
    box = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
    expected = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    torch.testing.assert_close(xywh_to_corners(box), expected)


# ===========================================================================
# Basic IoU correctness
# ===========================================================================


def test_identical_boxes_iou_is_1():
    """IoU of a box with itself must be exactly 1.

    Failure indicates xywh-vs-corner confusion: if the code passes corners
    through the xywh area formula (w*h), it produces wrong areas.
    """
    box = torch.tensor([0.5, 0.5, 0.4, 0.3])
    boxes = box.unsqueeze(0)  # (1, 4)
    iou = box_iou_xywh(boxes, box)
    torch.testing.assert_close(iou, torch.tensor([1.0]))


def test_disjoint_boxes_iou_is_0():
    """Fully disjoint boxes must yield IoU = 0, not NaN.

    A NaN here would propagate through reward signals and corrupt training.
    """
    # Box A at top-left corner, Box B at bottom-right corner, no overlap.
    box_a = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    box_b = torch.tensor([0.9, 0.9, 0.1, 0.1])
    iou = box_iou_xywh(box_a, box_b)
    assert not torch.isnan(iou).any(), "IoU must not be NaN for disjoint boxes"
    torch.testing.assert_close(iou, torch.tensor([0.0]))


def test_half_overlap_hand_computed():
    """Boxes with half-overlap must yield IoU = 1/3 (hand-verified).

    Validates the intersection-over-union formula for partial overlaps, which
    is the most common case in localization training.

    Box A: center=(0.5, 0.5), w=1.0, h=1.0  -> corners (0.0, 0.0, 1.0, 1.0)
    Box B: center=(1.0, 0.5), w=1.0, h=1.0  -> corners (0.5, 0.0, 1.5, 1.0)

    Intersection:
      x1_inter = max(0.0, 0.5) = 0.5,  x2_inter = min(1.0, 1.5) = 1.0
      y1_inter = max(0.0, 0.0) = 0.0,  y2_inter = min(1.0, 1.0) = 1.0
      area_inter = (1.0 - 0.5) * (1.0 - 0.0) = 0.5 * 1.0 = 0.5

    Areas:
      area_A = 1.0 * 1.0 = 1.0
      area_B = 1.0 * 1.0 = 1.0

    Union = 1.0 + 1.0 - 0.5 = 1.5
    IoU   = 0.5 / 1.5       = 1/3 ≈ 0.33333...
    """
    box_a = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
    box_b = torch.tensor([1.0, 0.5, 1.0, 1.0])
    iou = box_iou_xywh(box_a, box_b)
    # 0.5 / 1.5 = 1/3
    torch.testing.assert_close(iou, torch.tensor([1.0 / 3.0]))


def test_nested_boxes_hand_computed():
    """Nested boxes: IoU = inner_area / outer_area (hand-verified).

    Validates that the formula correctly handles containment, where union
    equals the outer area.

    Outer: center=(0.5, 0.5), w=1.0, h=1.0  -> corners (0.0, 0.0, 1.0, 1.0), area=1.0
    Inner: center=(0.5, 0.5), w=0.2, h=0.2  -> corners (0.4, 0.4, 0.6, 0.6), area=0.04

    Intersection:
      x1_inter = max(0.0, 0.4) = 0.4,  x2_inter = min(1.0, 0.6) = 0.6
      y1_inter = max(0.0, 0.4) = 0.4,  y2_inter = min(1.0, 0.6) = 0.6
      area_inter = 0.2 * 0.2 = 0.04

    Union = 1.0 + 0.04 - 0.04 = 1.0
    IoU   = 0.04 / 1.0 = 0.04
    """
    outer = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
    inner = torch.tensor([0.5, 0.5, 0.2, 0.2])
    iou = box_iou_xywh(outer, inner)
    # area_inner / area_outer = 0.04 / 1.0 = 0.04
    torch.testing.assert_close(iou, torch.tensor([0.04]), atol=1e-6, rtol=1e-5)


def test_symmetry():
    """IoU(A, B) must equal IoU(B, A) for 20 random pairs.

    Asymmetry would indicate a bug in the intersection or area formula.
    """
    torch.manual_seed(42)
    for _ in range(20):
        a = torch.rand(1, 4)
        a[:, 2:] = a[:, 2:].abs() * 0.5 + 0.01   # positive w, h
        b = torch.rand(4)
        b[2:] = b[2:].abs() * 0.5 + 0.01

        iou_ab = box_iou_xywh(a, b)
        # Swap: boxes_a=(1,4) with b, box_b=a[0]
        iou_ba = box_iou_xywh(b.unsqueeze(0), a[0])
        torch.testing.assert_close(iou_ab, iou_ba, atol=1e-6, rtol=1e-5)


def test_zero_width_box_produces_zero_not_nan():
    """A box with w=0 must yield IoU=0, never NaN.

    Zero-width boxes arise from degenerate policy outputs at training start;
    NaN would propagate through the reward and break all parameter updates.
    """
    zero_w = torch.tensor([[0.5, 0.5, 0.0, 0.3]])   # zero width
    other  = torch.tensor([0.5, 0.5, 0.3, 0.3])
    iou = box_iou_xywh(zero_w, other)
    assert not torch.isnan(iou).any(), "IoU with zero-width box must not be NaN"
    torch.testing.assert_close(iou, torch.tensor([0.0]))


def test_zero_area_both_boxes_not_nan():
    """Two zero-area boxes must yield IoU=0, not NaN.

    The EPS guard in the union denominator must prevent 0/0 = NaN, and
    `torch.where` must select the zero branch in this degenerate case.
    """
    zero_a = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    zero_b = torch.tensor([0.5, 0.5, 0.0, 0.0])
    iou = box_iou_xywh(zero_a, zero_b)
    assert not torch.isnan(iou).any(), "IoU of two zero-area boxes must not be NaN"
    torch.testing.assert_close(iou, torch.tensor([0.0]))


def test_iou_bounds():
    """IoU must be in [0, 1] for 100 random box pairs.

    Values outside [0, 1] are invalid rewards that would destabilise training.
    """
    torch.manual_seed(7)
    boxes_a = torch.rand(100, 4)
    boxes_a[:, 2:] = boxes_a[:, 2:].abs() * 0.5 + 0.01
    box_b = torch.rand(4)
    box_b[2:] = box_b[2:].abs() * 0.5 + 0.01

    iou = box_iou_xywh(boxes_a, box_b)
    assert (iou >= 0.0).all() and (iou <= 1.0).all(), (
        f"IoU out of [0,1]: min={iou.min()}, max={iou.max()}"
    )


# ===========================================================================
# max_iou_over_gt
# ===========================================================================


def test_max_iou_over_gt_with_single_gt_matches_direct_iou():
    """max_iou_over_gt with M=1 must equal box_iou_xywh for that single GT.

    If they disagree, either the mask logic or the max aggregation is wrong.
    """
    sampled = torch.rand(10, 4)
    sampled[:, 2:] = sampled[:, 2:].abs() * 0.3 + 0.01
    gt_single = torch.rand(1, 4)
    gt_single[:, 2:] = gt_single[:, 2:].abs() * 0.3 + 0.01
    mask = torch.tensor([True])

    result = max_iou_over_gt(sampled, gt_single, mask)
    expected = box_iou_xywh(sampled, gt_single[0])
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)


def test_max_iou_takes_max_over_multiple_gt():
    """max_iou_over_gt must return 1.0 when sample perfectly matches any GT.

    Ensures the max is taken over all real GT rows, not just the first.
    A partial match with GT #1 and a perfect match with GT #2 must give 1.0.
    """
    # GT #1: some box the sample only partially overlaps.
    # GT #2: exactly the same as the sample.
    sample_box = torch.tensor([0.5, 0.5, 0.3, 0.3])
    gt = torch.stack([
        torch.tensor([0.2, 0.2, 0.3, 0.3]),   # GT #1 — partial overlap
        sample_box.clone(),                     # GT #2 — perfect match
    ])  # (2, 4)
    mask = torch.tensor([True, True])

    sampled = sample_box.unsqueeze(0)  # (1, 4)
    result = max_iou_over_gt(sampled, gt, mask)
    torch.testing.assert_close(result, torch.tensor([1.0]))


def test_mask_excludes_padded_gt():
    """Padded GT rows (mask=False) must be ignored even if they perfectly match.

    This is a critical correctness requirement: using a padded box as if it
    were a real annotation inflates the reward signal and corrupts training.
    """
    sample_box = torch.tensor([0.5, 0.5, 0.3, 0.3])
    gt = torch.stack([
        torch.tensor([0.9, 0.9, 0.05, 0.05]),  # real GT — no overlap with sample
        sample_box.clone(),                      # padded — perfect match but MASKED
    ])
    # Only the first row is real.
    mask = torch.tensor([True, False])

    sampled = sample_box.unsqueeze(0)  # (1, 4)
    result = max_iou_over_gt(sampled, gt, mask)

    # Must return the IoU against the real GT (near 0), NOT 1.0 from the padded row.
    assert result[0] < 0.1, (
        f"Padded GT must be masked out; got IoU={result[0].item():.4f}, expected ~0"
    )


def test_all_masked_out_returns_zero():
    """If gt_mask is all False, max_iou_over_gt must return zeros (not NaN, not -1).

    When no ground-truth annotations are available for an image, the reward
    must be 0 rather than an undefined or negative value.
    """
    sampled = torch.rand(5, 4)
    sampled[:, 2:] = 0.2
    gt = torch.rand(3, 4)
    gt[:, 2:] = 0.2
    mask = torch.zeros(3, dtype=torch.bool)  # all False

    result = max_iou_over_gt(sampled, gt, mask)
    assert not torch.isnan(result).any(), "All-masked result must not be NaN"
    torch.testing.assert_close(result, torch.zeros(5))


# ===========================================================================
# batched_max_iou
# ===========================================================================


def test_batched_max_iou_shape():
    """batched_max_iou must return a tensor of shape (B, N).

    A wrong output shape breaks any downstream loss computation immediately.
    """
    B, N, M = 4, 10, 5
    sampled = torch.rand(B, N, 4)
    sampled[..., 2:] = 0.2
    gt = torch.rand(B, M, 4)
    gt[..., 2:] = 0.2
    mask = torch.ones(B, M, dtype=torch.bool)

    result = batched_max_iou(sampled, gt, mask)
    assert result.shape == (B, N), f"Expected ({B}, {N}), got {result.shape}"


def test_batched_max_iou_matches_unbatched():
    """batched_max_iou must match looped max_iou_over_gt for each batch item.

    Ensures the batching wrapper does not mix up batch indices or introduce
    off-by-one errors relative to the single-image implementation.
    """
    B, N, M = 4, 8, 3
    torch.manual_seed(99)
    sampled = torch.rand(B, N, 4)
    sampled[..., 2:] = torch.rand(B, N, 2) * 0.3 + 0.05
    gt = torch.rand(B, M, 4)
    gt[..., 2:] = torch.rand(B, M, 2) * 0.3 + 0.05
    mask = torch.ones(B, M, dtype=torch.bool)
    # Randomly mask out one GT per batch item.
    mask[:, -1] = False

    batched = batched_max_iou(sampled, gt, mask)

    for b in range(B):
        unbatched = max_iou_over_gt(sampled[b], gt[b], mask[b])
        torch.testing.assert_close(batched[b], unbatched, atol=1e-6, rtol=1e-5)


# ===========================================================================
# clamp_boxes_to_image
# ===========================================================================


def test_clamp_valid_box_unchanged():
    """A box entirely inside [0, 1] must be returned unchanged.

    If valid boxes are perturbed by clamping, the policy gradient receives
    wrong IoU values for correctly-predicted boxes.
    """
    # Center at (0.5, 0.5), width=0.4, height=0.3:
    # corners = (0.3, 0.35, 0.7, 0.65) — all in [0, 1].
    box = torch.tensor([[0.5, 0.5, 0.4, 0.3]])
    result = clamp_boxes_to_image(box)
    torch.testing.assert_close(result, box)


def test_clamp_left_overflow_hand_computed():
    """Box overflowing on the left must clamp only the left edge (hand-verified).

    Selective clamping ensures partial-overflow boxes are corrected minimally
    without distorting the part of the box that is inside the image.

    Input xywh: (0.1, 0.5, 0.4, 0.2)
    Corners:
      x1 = 0.1 - 0.4/2 = 0.1 - 0.2 = -0.1
      y1 = 0.5 - 0.2/2 = 0.5 - 0.1 =  0.4
      x2 = 0.1 + 0.2   =              0.3
      y2 = 0.5 + 0.1   =              0.6

    After clamp([0,1]):
      x1 = clamp(-0.1, 0, 1) = 0.0
      y1 = clamp(0.4,  0, 1) = 0.4
      x2 = clamp(0.3,  0, 1) = 0.3
      y2 = clamp(0.6,  0, 1) = 0.6

    Back to xywh:
      x_c = (0.0 + 0.3) / 2 = 0.15
      y_c = (0.4 + 0.6) / 2 = 0.5
      w   = 0.3 - 0.0        = 0.3
      h   = 0.6 - 0.4        = 0.2

    Expected xywh: (0.15, 0.5, 0.3, 0.2)
    """
    box = torch.tensor([[0.1, 0.5, 0.4, 0.2]])
    expected = torch.tensor([[0.15, 0.5, 0.3, 0.2]])
    result = clamp_boxes_to_image(box)
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)


def test_clamp_preserves_corners_in_unit_square():
    """20 random valid boxes (corners in [0,1]) must be unchanged after clamping.

    Ensures clamp is a no-op for the common case of in-bounds boxes.
    """
    torch.manual_seed(13)
    # Construct boxes whose corners are guaranteed to be in (0.05, 0.95).
    x1 = torch.rand(20) * 0.4 + 0.05    # in [0.05, 0.45]
    y1 = torch.rand(20) * 0.4 + 0.05
    x2 = x1 + torch.rand(20) * 0.4 + 0.05   # x2 in [0.1, 0.9]
    y2 = y1 + torch.rand(20) * 0.4 + 0.05
    # Clip to [0, 1] to guarantee valid corners.
    x2 = x2.clamp(max=0.95)
    y2 = y2.clamp(max=0.95)
    corners = torch.stack([x1, y1, x2, y2], dim=1)  # (20, 4)
    boxes = corners_to_xywh(corners)

    clamped = clamp_boxes_to_image(boxes)
    torch.testing.assert_close(clamped, boxes, atol=1e-6, rtol=1e-5)


def test_clamp_fully_outside_gives_zero_width():
    """A box fully outside the unit square must get w=0, h=0 (never negative).

    Negative width/height would cause negative IoU contributions and corrupt
    reward computation in an undetectable way.

    Input xywh: (2.0, 2.0, 0.5, 0.5)
    Corners:
      x1 = 2.0 - 0.25 = 1.75,  y1 = 2.0 - 0.25 = 1.75
      x2 = 2.0 + 0.25 = 2.25,  y2 = 2.0 + 0.25 = 2.25

    After clamp to [0, 1]:
      x1 = 1.0, y1 = 1.0, x2 = 1.0, y2 = 1.0

    xywh: x_c = 1.0, y_c = 1.0, w = 0.0, h = 0.0
    """
    box = torch.tensor([[2.0, 2.0, 0.5, 0.5]])
    result = clamp_boxes_to_image(box)
    # Width and height must be zero (or non-negative).
    assert result[0, 2] >= 0.0, f"w must be >= 0, got {result[0, 2].item()}"
    assert result[0, 3] >= 0.0, f"h must be >= 0, got {result[0, 3].item()}"
    # Specifically, both collapse to 0 for a fully-outside box.
    torch.testing.assert_close(result[0, 2], torch.tensor(0.0))
    torch.testing.assert_close(result[0, 3], torch.tensor(0.0))


# ===========================================================================
# Dtype preservation
# ===========================================================================


def test_float32_in_float32_out():
    """float32 inputs must produce float32 outputs (no silent upcasts).

    Silent dtype promotion to float64 would cause shape/type mismatches
    with the rest of the model (weights are float32).
    """
    box = torch.tensor([[0.5, 0.5, 0.3, 0.3]], dtype=torch.float32)
    ref = torch.tensor([0.4, 0.4, 0.2, 0.2], dtype=torch.float32)
    iou = box_iou_xywh(box, ref)
    assert iou.dtype == torch.float32, f"Expected float32, got {iou.dtype}"


def test_float16_supported():
    """float16 inputs must produce float16 IoU output (no silent upcasts).

    Half-precision inference is common for GPU deployment; silent casting
    to float32 would break downstream dtype checks and may double memory use.
    """
    box = torch.tensor([[0.5, 0.5, 0.3, 0.3]], dtype=torch.float16)
    ref = torch.tensor([0.4, 0.4, 0.2, 0.2], dtype=torch.float16)
    iou = box_iou_xywh(box, ref)
    assert iou.dtype == torch.float16, f"Expected float16, got {iou.dtype}"
