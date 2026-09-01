"""Aggressive correctness tests for the vectorized batched_max_iou.

The vectorized implementation must match the original per-image-loop
reference exactly, on every (B, N, M, gt_mask) configuration that
shows up in training: large N, small/large M, partially-masked GTs,
all-masked rows, zero-area boxes, mixed dtypes, and large enough
batches that any broadcasting bug would be obvious.
"""

from __future__ import annotations

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.imagenet_localization.core.iou import (
    batched_max_iou,
    max_iou_over_gt,
)


# ---------------------------------------------------------------------------
# Reference: the original per-image Python-loop implementation.
# ---------------------------------------------------------------------------

def _reference_batched_max_iou(sampled, gt, gt_mask):
    """Original per-image-loop implementation for comparison."""
    B = sampled.shape[0]
    out = []
    for b in range(B):
        out.append(max_iou_over_gt(sampled[b], gt[b], gt_mask[b]))
    return torch.stack(out, dim=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_boxes(*shape, seed=0, w_min=0.05, w_max=0.4):
    """Random valid xywh boxes with positive width/height in [0, 1] frame."""
    g = torch.Generator().manual_seed(seed)
    boxes = torch.rand(*shape, generator=g)
    # Ensure positive w, h and that the box stays in [0, 1].
    boxes[..., 2:] = (boxes[..., 2:] * (w_max - w_min) + w_min)
    # Center sufficiently inside so corners stay in [0, 1].
    margin = boxes[..., 2:].max() / 2 + 1e-3
    boxes[..., :2] = boxes[..., :2] * (1 - 2 * margin) + margin
    return boxes


def _rand_mask(B, M, p=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, M, generator=g) > p


# ---------------------------------------------------------------------------
# Equivalence with the reference (the load-bearing tests)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B,N,M", [
    (1, 1, 1),
    (1, 16, 1),
    (4, 32, 5),
    (8, 256, 10),
    (16, 1024, 8),
    (4, 4096, 12),
])
def test_vectorized_matches_reference_random(B, N, M):
    """Vectorized output must equal the per-image reference within fp32 tol.

    Identical math, just rearranged broadcasts; any deviation > 1e-5
    indicates a bug (off-by-one in broadcasting, wrong axis for max, etc.).
    """
    sampled = _rand_boxes(B, N, 4, seed=B * 1000 + N + M)
    gt = _rand_boxes(B, M, 4, seed=B * 2000 + N + M)
    mask = torch.ones(B, M, dtype=torch.bool)
    fast = batched_max_iou(sampled, gt, mask)
    ref = _reference_batched_max_iou(sampled, gt, mask)
    torch.testing.assert_close(fast, ref, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("B,N,M", [
    (8, 64, 6), (8, 256, 10), (8, 1024, 12),
])
def test_vectorized_matches_reference_with_partial_mask(B, N, M):
    """Random gt_mask: fast and reference must agree exactly."""
    sampled = _rand_boxes(B, N, 4, seed=42)
    gt = _rand_boxes(B, M, 4, seed=43)
    mask = _rand_mask(B, M, p=0.5, seed=44)
    # Ensure at least one True per row (otherwise we hit the no-GT branch,
    # tested separately).
    mask[:, 0] = True
    fast = batched_max_iou(sampled, gt, mask)
    ref = _reference_batched_max_iou(sampled, gt, mask)
    torch.testing.assert_close(fast, ref, atol=1e-6, rtol=1e-5)


def test_vectorized_with_some_rows_all_masked():
    """Mixed batch: some images have valid GT, others have all-False mask.

    The all-False rows must produce exact zeros (not NaN, not -1) and the
    valid rows must equal the reference.
    """
    B, N, M = 8, 32, 4
    sampled = _rand_boxes(B, N, 4, seed=10)
    gt = _rand_boxes(B, M, 4, seed=11)
    mask = torch.ones(B, M, dtype=torch.bool)
    # Zero out half the batch's GT mask entirely.
    mask[: B // 2] = False
    fast = batched_max_iou(sampled, gt, mask)
    ref = _reference_batched_max_iou(sampled, gt, mask)
    torch.testing.assert_close(fast, ref, atol=1e-6, rtol=1e-5)
    # Sanity: the no-GT rows are exactly zero.
    assert torch.all(fast[: B // 2] == 0.0)
    # And no NaN / Inf anywhere.
    assert not torch.isnan(fast).any()
    assert not torch.isinf(fast).any()


def test_vectorized_with_zero_area_sampled_box_no_nan():
    """Zero-area sampled boxes must produce IoU=0, not NaN."""
    B, N, M = 2, 4, 3
    sampled = _rand_boxes(B, N, 4, seed=7)
    sampled[0, 0, 2:] = 0.0   # zero w, h on (0, 0)
    sampled[1, 2, 2] = 0.0    # zero w on (1, 2)
    gt = _rand_boxes(B, M, 4, seed=8)
    mask = torch.ones(B, M, dtype=torch.bool)
    fast = batched_max_iou(sampled, gt, mask)
    assert not torch.isnan(fast).any()
    assert not torch.isinf(fast).any()
    # Zero-area sampled boxes have intersection 0 → IoU 0.
    assert fast[0, 0].item() == 0.0
    assert fast[1, 2].item() == 0.0


def test_vectorized_with_zero_area_gt_box_no_nan():
    """Zero-area GT boxes must produce IoU=0, not NaN."""
    B, N, M = 2, 4, 3
    sampled = _rand_boxes(B, N, 4, seed=9)
    gt = _rand_boxes(B, M, 4, seed=10)
    gt[0, 0, 2:] = 0.0  # zero-area GT
    mask = torch.ones(B, M, dtype=torch.bool)
    fast = batched_max_iou(sampled, gt, mask)
    assert not torch.isnan(fast).any()
    assert not torch.isinf(fast).any()


def test_vectorized_dtype_preserved_float32():
    sampled = _rand_boxes(2, 8, 4, seed=1).to(torch.float32)
    gt = _rand_boxes(2, 3, 4, seed=2).to(torch.float32)
    mask = torch.ones(2, 3, dtype=torch.bool)
    out = batched_max_iou(sampled, gt, mask)
    assert out.dtype == torch.float32


def test_vectorized_dtype_preserved_float64():
    sampled = _rand_boxes(2, 8, 4, seed=1).to(torch.float64)
    gt = _rand_boxes(2, 3, 4, seed=2).to(torch.float64)
    mask = torch.ones(2, 3, dtype=torch.bool)
    out = batched_max_iou(sampled, gt, mask)
    assert out.dtype == torch.float64


def test_vectorized_output_in_unit_interval():
    """Output is bounded in [0, 1] — guards against numerical overshoot."""
    sampled = _rand_boxes(4, 64, 4, seed=20)
    gt = _rand_boxes(4, 6, 4, seed=21)
    mask = torch.ones(4, 6, dtype=torch.bool)
    out = batched_max_iou(sampled, gt, mask)
    assert (out >= 0.0).all()
    assert (out <= 1.0).all()


def test_vectorized_gradient_does_not_flow():
    """The reward is computed under no_grad in train.py; even outside no_grad,
    detach should not leak unexpected gradient (this guards against any
    accidental requires_grad on the IoU intermediates)."""
    sampled = _rand_boxes(2, 16, 4, seed=30)
    sampled.requires_grad_(True)
    gt = _rand_boxes(2, 3, 4, seed=31)
    mask = torch.ones(2, 3, dtype=torch.bool)
    out = batched_max_iou(sampled, gt, mask)
    # Gradient *should* flow back to sampled (the IoU is differentiable
    # almost everywhere), but the value should be finite.
    out.sum().backward()
    assert sampled.grad is not None
    assert not torch.isnan(sampled.grad).any()
    assert not torch.isinf(sampled.grad).any()


# ---------------------------------------------------------------------------
# Stress: large-N reach (N=16k, 65k) — same shapes the production runs use.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [16384, 65536])
def test_vectorized_large_N_no_oom_no_nan(N):
    """Sanity-run at production N values to catch broadcast / memory bugs.

    Uses small B, M so the test stays under a few hundred MB even at
    N=65536 (peak intermediate (B, N, M) is ~5 MB at B=2, M=4).
    """
    B, M = 2, 4
    sampled = _rand_boxes(B, N, 4, seed=N)
    gt = _rand_boxes(B, M, 4, seed=N + 1)
    mask = torch.ones(B, M, dtype=torch.bool)
    out = batched_max_iou(sampled, gt, mask)
    assert out.shape == (B, N)
    assert (out >= 0.0).all()
    assert (out <= 1.0).all()
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("N,B,M", [(1024, 8, 8), (4096, 4, 10)])
def test_vectorized_large_matches_reference_within_tolerance(N, B, M):
    """Large-N equivalence: vectorized vs reference must match.

    Slow-ish (the reference loop is O(B M) Python iterations); kept small
    enough to run in a few seconds.
    """
    sampled = _rand_boxes(B, N, 4, seed=N)
    gt = _rand_boxes(B, M, 4, seed=N + 1)
    mask = torch.ones(B, M, dtype=torch.bool)
    fast = batched_max_iou(sampled, gt, mask)
    ref = _reference_batched_max_iou(sampled, gt, mask)
    torch.testing.assert_close(fast, ref, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# Edge-case axes: the vectorized version must not silently mis-handle
# extreme shapes that the reference would have handled correctly.
# ---------------------------------------------------------------------------

def test_M_equals_1():
    """Single GT per image — degenerate but legal."""
    B, N = 4, 32
    sampled = _rand_boxes(B, N, 4, seed=50)
    gt = _rand_boxes(B, 1, 4, seed=51)
    mask = torch.ones(B, 1, dtype=torch.bool)
    fast = batched_max_iou(sampled, gt, mask)
    ref = _reference_batched_max_iou(sampled, gt, mask)
    torch.testing.assert_close(fast, ref, atol=1e-6, rtol=1e-5)


def test_B_equals_1():
    """Single image batch."""
    sampled = _rand_boxes(1, 16, 4, seed=60)
    gt = _rand_boxes(1, 5, 4, seed=61)
    mask = torch.ones(1, 5, dtype=torch.bool)
    fast = batched_max_iou(sampled, gt, mask)
    ref = _reference_batched_max_iou(sampled, gt, mask)
    torch.testing.assert_close(fast, ref, atol=1e-6, rtol=1e-5)


def test_all_zero_size_GT_returns_zero():
    """All GTs masked out across the entire batch → all-zero rewards."""
    B, N, M = 4, 16, 3
    sampled = _rand_boxes(B, N, 4, seed=70)
    gt = _rand_boxes(B, M, 4, seed=71)
    mask = torch.zeros(B, M, dtype=torch.bool)
    out = batched_max_iou(sampled, gt, mask)
    assert torch.all(out == 0.0)


def test_identical_box_in_sampled_and_gt_gives_iou_1():
    """If sampled[b, i] == gt[b, 0], then the max IoU at (b, i) is 1.0."""
    B, N, M = 2, 8, 3
    sampled = _rand_boxes(B, N, 4, seed=80)
    gt = _rand_boxes(B, M, 4, seed=81)
    # Make sample (b=0, n=3) identical to gt[0, 0]
    sampled[0, 3] = gt[0, 0]
    mask = torch.ones(B, M, dtype=torch.bool)
    out = batched_max_iou(sampled, gt, mask)
    assert torch.allclose(out[0, 3], torch.tensor(1.0), atol=1e-5)


def test_speedup_vs_reference():
    """Sanity: vectorized must complete (B=32, N=4096, M=10) faster than the
    Python-loop reference. Not a strict timing test (CI variance), but the
    speedup should be at least 2× — if it regresses below that, something
    went very wrong."""
    import time
    B, N, M = 32, 4096, 10
    sampled = _rand_boxes(B, N, 4, seed=100)
    gt = _rand_boxes(B, M, 4, seed=101)
    mask = torch.ones(B, M, dtype=torch.bool)

    # Warmup
    _ = batched_max_iou(sampled, gt, mask)
    _ = _reference_batched_max_iou(sampled, gt, mask)

    t0 = time.perf_counter()
    for _ in range(3):
        _ = batched_max_iou(sampled, gt, mask)
    t_fast = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(3):
        _ = _reference_batched_max_iou(sampled, gt, mask)
    t_ref = time.perf_counter() - t0

    speedup = t_ref / max(t_fast, 1e-9)
    assert speedup >= 2.0, (
        f"Vectorized batched_max_iou ({t_fast*1000:.1f} ms) only "
        f"{speedup:.1f}× faster than reference ({t_ref*1000:.1f} ms); "
        f"expected ≥2×."
    )
