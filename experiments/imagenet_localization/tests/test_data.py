"""Tests for experiments.imagenet_localization.datasets.data.

All pure-function tests (parsing, normalization, flipping, bin discretization)
run without the ImageNet dataset present and complete well under 10 seconds.
Tests that require actual JPEG images are marked with @requires_data and will
be skipped in environments without the dataset.

Each test has a docstring explaining the invariant being checked and why it matters.
"""

from __future__ import annotations

import math
import random

import pytest
import torch

from experiments.imagenet_localization.datasets.data import (
    MAX_M,
    apply_horizontal_flip,
    bin_discretize_coord,
    normalize_box,
    parse_prediction_string,
)

# Import the skip marker from conftest (available via pytest's conftest mechanism)
from experiments.imagenet_localization.tests.conftest import requires_data


# ===========================================================================
# parse_prediction_string — parsing
# ===========================================================================


def test_single_box_row_parsed():
    """A single-box PredictionString parses to exactly 1 box with the correct wnid.

    This is the most common case in the dataset; any error here breaks all
    single-instance training examples.
    """
    wnid, boxes = parse_prediction_string("n02017213 115 49 448 294")
    assert wnid == "n02017213"
    assert len(boxes) == 1
    assert boxes[0] == (115, 49, 448, 294)


def test_multi_box_row_same_class_parsed():
    """Two boxes with the same wnid in a PredictionString parse to 2 boxes.

    Multi-instance rows are used for images with multiple annotations of the
    same class. Both boxes must be returned (the wnid appearing twice is valid).
    """
    pred_str = "n02017213 115 49 448 294 n02017213 91 42 330 432"
    wnid, boxes = parse_prediction_string(pred_str)
    assert wnid == "n02017213"
    assert len(boxes) == 2
    assert boxes[0] == (115, 49, 448, 294)
    assert boxes[1] == (91, 42, 330, 432)


def test_three_box_row_parsed():
    """A PredictionString with three boxes of the same wnid parses to 3 boxes.

    Validates that the stride-5 parsing loop correctly handles an arbitrary
    number of repeat boxes, not just 1 or 2.
    """
    pred_str = (
        "n01234567 10 20 100 200 "
        "n01234567 30 40 120 220 "
        "n01234567 50 60 140 240"
    )
    wnid, boxes = parse_prediction_string(pred_str)
    assert wnid == "n01234567"
    assert len(boxes) == 3
    assert boxes[0] == (10, 20, 100, 200)
    assert boxes[1] == (30, 40, 120, 220)
    assert boxes[2] == (50, 60, 140, 240)


def test_mixed_class_row_raises():
    """A PredictionString with two different wnids must raise AssertionError.

    The spec states all wnids in a single row are identical; mixing classes
    is a data format violation that should be caught immediately rather than
    silently producing a wrong supervision target.
    """
    pred_str = "n00000001 0 0 10 10 n00000002 2 2 3 3"
    with pytest.raises(AssertionError):
        parse_prediction_string(pred_str)


def test_malformed_row_raises():
    """A PredictionString whose token count is not divisible by 5 must raise AssertionError.

    A malformed row (e.g. truncated or with a stray extra token) would produce
    silently wrong box coordinates if parsed naively; an assertion makes the
    error explicit and visible during debugging.
    """
    # 6 tokens — not divisible by 5
    pred_str = "n01234567 10 20 100 200 42"
    with pytest.raises(AssertionError):
        parse_prediction_string(pred_str)


def test_extra_whitespace_tolerated():
    """Leading, trailing, and doubled spaces in a PredictionString are tolerated.

    The actual CSV contains trailing spaces in many rows. Failing here would
    crash dataset loading for a large fraction of the training set.
    """
    # Leading space, trailing space, double space between tokens
    pred_str = "  n02017213  115  49  448  294  "
    wnid, boxes = parse_prediction_string(pred_str)
    assert wnid == "n02017213"
    assert len(boxes) == 1
    assert boxes[0] == (115, 49, 448, 294)


# ===========================================================================
# normalize_box — coordinate normalization
# ===========================================================================


def test_basic_normalize_hand_computed():
    """normalize_box(0, 0, 500, 375, 500, 375) -> (0.5, 0.5, 1.0, 1.0).

    A box spanning the full 500x375 image:
      center_x = (0 + 500) / 2 = 250  -> 250/500 = 0.5
      center_y = (0 + 375) / 2 = 187.5 -> 187.5/375 = 0.5
      w = (500 - 0) / 500 = 1.0
      h = (375 - 0) / 375 = 1.0
    """
    xc, yc, w, h = normalize_box(0, 0, 500, 375, 500, 375)
    assert math.isclose(xc, 0.5, rel_tol=1e-6)
    assert math.isclose(yc, 0.5, rel_tol=1e-6)
    assert math.isclose(w,  1.0, rel_tol=1e-6)
    assert math.isclose(h,  1.0, rel_tol=1e-6)


def test_non_square_image_normalize_hand_computed():
    """normalize_box(100, 50, 300, 200, 400, 300) produces correct non-square normalization.

    Hand computation for a 400x300 image:
      center_x = (100 + 300) / 2 = 200   -> 200/400 = 0.5
      center_y = (50 + 200)  / 2 = 125   -> 125/300 ≈ 0.41667
      w = (300 - 100) / 400 = 200/400 = 0.5
      h = (200 - 50)  / 300 = 150/300 = 0.5

    Validates that width and height are normalized against THEIR OWN axis
    (w by orig_w, h by orig_h), not swapped or both divided by the same value.
    """
    xc, yc, w, h = normalize_box(100, 50, 300, 200, 400, 300)
    assert math.isclose(xc, 0.5,        rel_tol=1e-6)
    assert math.isclose(yc, 125 / 300,  rel_tol=1e-6)
    assert math.isclose(w,  0.5,        rel_tol=1e-6)
    assert math.isclose(h,  0.5,        rel_tol=1e-6)


def test_full_image_box_normalizes_to_ones():
    """A box spanning the full image at any resolution normalizes to (0.5, 0.5, 1.0, 1.0).

    Parametrized over several resolutions to confirm the formula is not
    accidentally hardcoded to a specific image size.
    """
    for orig_w, orig_h in [(224, 224), (640, 480), (1920, 1080), (375, 500)]:
        xc, yc, w, h = normalize_box(0, 0, orig_w, orig_h, orig_w, orig_h)
        assert math.isclose(xc, 0.5, rel_tol=1e-6), f"xc wrong for {orig_w}x{orig_h}"
        assert math.isclose(yc, 0.5, rel_tol=1e-6), f"yc wrong for {orig_w}x{orig_h}"
        assert math.isclose(w,  1.0, rel_tol=1e-6), f"w wrong for {orig_w}x{orig_h}"
        assert math.isclose(h,  1.0, rel_tol=1e-6), f"h wrong for {orig_w}x{orig_h}"


def test_normalized_coords_in_unit_range():
    """30 random valid bounding boxes normalize to coordinates in [0, 1].

    Normalized coordinates outside [0, 1] would cause IoU calculations to
    produce values outside [0, 1], corrupting the reward signal during training.
    """
    rng = random.Random(1337)
    for _ in range(30):
        orig_w = rng.randint(100, 2000)
        orig_h = rng.randint(100, 2000)
        x1 = rng.randint(0, orig_w - 1)
        y1 = rng.randint(0, orig_h - 1)
        x2 = rng.randint(x1 + 1, orig_w)
        y2 = rng.randint(y1 + 1, orig_h)
        xc, yc, w, h = normalize_box(x1, y1, x2, y2, orig_w, orig_h)
        assert 0.0 <= xc <= 1.0, f"xc={xc} out of [0,1] for ({x1},{y1},{x2},{y2}) in {orig_w}x{orig_h}"
        assert 0.0 <= yc <= 1.0, f"yc={yc} out of [0,1]"
        assert 0.0 <= w  <= 1.0, f"w={w} out of [0,1]"
        assert 0.0 <= h  <= 1.0, f"h={h} out of [0,1]"


# ===========================================================================
# apply_horizontal_flip — augmentation
# ===========================================================================


def test_double_flip_is_identity():
    """Applying horizontal flip twice returns the original image AND boxes exactly.

    A double flip that doesn't recover the original means the flip is lossy
    (e.g., using rounding or in-place modification). This would introduce
    systematic bias in augmented training data.
    """
    # Build a (3, 8, 8) image with distinct per-column values
    image = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8)
    boxes = torch.tensor([
        [0.3, 0.5, 0.2, 0.4],
        [0.7, 0.2, 0.1, 0.3],
    ])

    img1, boxes1 = apply_horizontal_flip(image, boxes)
    img2, boxes2 = apply_horizontal_flip(img1, boxes1)

    torch.testing.assert_close(img2, image)
    torch.testing.assert_close(boxes2, boxes)


def test_flip_moves_x_c_to_1_minus_x_c():
    """Horizontal flip maps x_c -> 1 - x_c (hand-computed).

    Box (0.3, 0.5, 0.2, 0.4) after flip:
      x_c = 1 - 0.3 = 0.7  (mirrored center)
      y_c = 0.5             (unchanged)
      w   = 0.2             (unchanged, width is symmetric)
      h   = 0.4             (unchanged)

    This is the exact formula required by the spec. A wrong formula (e.g.,
    flipping w or h, or forgetting to subtract from 1) would break supervision.
    """
    boxes = torch.tensor([[0.3, 0.5, 0.2, 0.4]])
    image = torch.zeros(3, 4, 4)
    _, flipped_boxes = apply_horizontal_flip(image, boxes)
    expected = torch.tensor([[0.7, 0.5, 0.2, 0.4]])
    torch.testing.assert_close(flipped_boxes, expected)


def test_flip_preserves_y_c_w_h():
    """Horizontal flip must not change y_c, w, or h — only x_c changes.

    y_c, w, and h are all symmetric under horizontal reflection. Accidentally
    modifying any of them would corrupt the ground-truth bounding box area
    and aspect ratio used for IoU reward computation.
    """
    torch.manual_seed(5)
    boxes = torch.rand(10, 4)
    # Ensure positive w, h; keep x_c and y_c in (0, 1)
    boxes[:, 2:] = boxes[:, 2:].abs() * 0.4 + 0.05
    boxes[:, :2] = boxes[:, :2] * 0.6 + 0.2

    image = torch.zeros(3, 8, 8)
    _, flipped_boxes = apply_horizontal_flip(image, boxes)

    # y_c, w, h columns (indices 1, 2, 3) must be unchanged
    torch.testing.assert_close(flipped_boxes[:, 1], boxes[:, 1])
    torch.testing.assert_close(flipped_boxes[:, 2], boxes[:, 2])
    torch.testing.assert_close(flipped_boxes[:, 3], boxes[:, 3])


def test_flip_image_pixels_actually_flipped():
    """Horizontal flip must physically reverse the pixel columns of the image tensor.

    Build a (3, 4, 4) image where column 0 = 1.0 and columns 1..3 = 0.0.
    After flip, column 3 should be 1.0 and columns 0..2 should be 0.0.

    Verifying actual pixel values ensures the flip is not a no-op that only
    updates metadata — the image must be truly reversed in the W dimension.
    """
    image = torch.zeros(3, 4, 4)
    image[:, :, 0] = 1.0  # set leftmost column to 1

    boxes = torch.zeros(1, 4)  # dummy box
    flipped_image, _ = apply_horizontal_flip(image, boxes)

    # After flip: rightmost column (index 3) should be 1; leftmost should be 0
    assert (flipped_image[:, :, 3] == 1.0).all(), "Rightmost column should be 1 after flip"
    assert (flipped_image[:, :, 0] == 0.0).all(), "Leftmost column should be 0 after flip"


def test_flip_multi_box():
    """Horizontal flip must update x_c for ALL rows in the (M, 4) box tensor, not just row 0.

    A naive implementation that only processes boxes[0] would corrupt all
    secondary annotations in multi-instance images.
    """
    M = 5
    boxes = torch.rand(M, 4)
    boxes[:, 2:] = 0.1  # small w, h for simplicity

    original_x_c = boxes[:, 0].clone()
    image = torch.zeros(3, 8, 8)

    _, flipped_boxes = apply_horizontal_flip(image, boxes)

    expected_x_c = 1.0 - original_x_c
    torch.testing.assert_close(flipped_boxes[:, 0], expected_x_c)


def test_flip_does_not_mutate_input():
    """apply_horizontal_flip must not modify the input image or boxes tensors in-place.

    In-place modification would corrupt the original sample when the augmentation
    is conditionally applied (e.g., 50% flip probability), because the caller
    still holds a reference to the original tensors.
    """
    image = torch.rand(3, 4, 4)
    boxes = torch.rand(2, 4)
    boxes[:, 2:] = 0.1

    image_orig = image.clone()
    boxes_orig = boxes.clone()

    apply_horizontal_flip(image, boxes)

    torch.testing.assert_close(image, image_orig, msg="Input image was mutated in-place")
    torch.testing.assert_close(boxes, boxes_orig, msg="Input boxes were mutated in-place")


# ===========================================================================
# bin_discretize_coord — bin assignment
# ===========================================================================


def test_bin_zero_coord_yields_bin_zero():
    """bin_discretize_coord(0.0, K) must return 0 for any K.

    The minimum coordinate should map to the minimum bin. Off-by-one here
    would mean the leftmost bin is never selected by a box at the image edge.
    """
    for K in [10, 25, 50, 100]:
        result = bin_discretize_coord(0.0, K)
        assert result == 0, f"Expected 0 for coord=0.0, K={K}, got {result}"


def test_bin_one_coord_yields_last_bin():
    """bin_discretize_coord(1.0, K) must return K-1, NOT K.

    coord=1.0 is the maximum valid coordinate (right/bottom edge of image).
    Without the clamp, int(1.0 * K) = K which is out-of-range for a length-K
    array and would cause IndexError in downstream embedding lookups.
    """
    for K in [10, 25, 50, 100]:
        result = bin_discretize_coord(1.0, K)
        assert result == K - 1, f"Expected {K - 1} for coord=1.0, K={K}, got {result}"


def test_bin_center_is_recoverable():
    """bin_discretize_coord((k + 0.5) / K, K) == k for all k in {0, ..., K-1}.

    The center of bin k must discretize back to bin k. This is the fundamental
    invertibility property of uniform bin discretization: a coordinate placed
    exactly at the bin center should round to that bin, not to an adjacent one.

    Tested for K in {10, 25, 50, 100} to catch scale-dependent failures.
    """
    for K in [10, 25, 50, 100]:
        for k in range(K):
            coord = (k + 0.5) / K
            result = bin_discretize_coord(coord, K)
            assert result == k, (
                f"Expected bin {k} for coord={coord:.6f}, K={K}, got {result}"
            )


def test_bin_out_of_range_clamped():
    """Coordinates below 0 and above 1 must clamp to 0 and K-1 respectively.

    Policy outputs can transiently produce coordinates outside [0, 1] (e.g.,
    in the early training regime). Silent wrapping or IndexError would corrupt
    training; explicit clamping keeps behavior well-defined and bounded.
    """
    K = 20
    assert bin_discretize_coord(-0.5, K) == 0,    "Negative coord must clamp to 0"
    assert bin_discretize_coord(-1.0, K) == 0,    "Coord = -1.0 must clamp to 0"
    assert bin_discretize_coord(1.01, K) == K - 1, "Coord > 1.0 must clamp to K-1"
    assert bin_discretize_coord(2.0,  K) == K - 1, "Coord = 2.0 must clamp to K-1"


def test_bin_monotone():
    """bin_discretize_coord must be non-decreasing in [0, 1] for a given K.

    Monotonicity ensures that increasing the coordinate value never decreases
    the bin index, which is required for bins to have consistent spatial meaning.
    """
    K = 50
    coords = [i / 200.0 for i in range(201)]  # 0.0, 0.005, 0.01, ..., 1.0
    bins = [bin_discretize_coord(c, K) for c in coords]
    for i in range(1, len(bins)):
        assert bins[i] >= bins[i - 1], (
            f"Non-monotone at coord {coords[i]:.4f}: bins[{i}]={bins[i]} < bins[{i-1}]={bins[i-1]}"
        )


def test_bin_all_values_in_range():
    """For 200 evenly-spaced coords in [0, 1], all bin indices must be in {0, ..., K-1}.

    Out-of-range bin indices would cause IndexError in embedding lookup tables
    during training, crashing the run.
    """
    for K in [10, 50, 100]:
        for i in range(201):
            coord = i / 200.0
            b = bin_discretize_coord(coord, K)
            assert 0 <= b < K, f"Bin {b} out of range [0, {K-1}] for coord={coord:.4f}, K={K}"


# ===========================================================================
# MAX_M constant
# ===========================================================================


def test_max_m_value():
    """MAX_M must equal 8 as required by the spec.

    The constant is used for padding gt_boxes tensors; any other value would
    break shape assumptions in downstream model code.
    """
    assert MAX_M == 8, f"Expected MAX_M=8, got MAX_M={MAX_M}"
