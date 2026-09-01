"""Tests for experiments.imagenet_localization.analysis.gradient_analysis.

All tests use mock in-memory data — no ImageNet dataset required.
Model is always pretrained=False with K=10.

Test count: 8 tests.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from experiments.imagenet_localization.analysis.gradient_analysis import (
    HEAD_NAMES,
    cosine,
    compute_supervised_head_gradient,
    compute_rl_head_gradient,
    flatten_head_grads,
    head_params,
    run_analysis,
)
from experiments.imagenet_localization.models.model import LocalizationPolicy

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

K = 10
MAX_M = 4   # GT boxes per image (padded tensor height)


def _make_model(K: int = K) -> LocalizationPolicy:
    """Construct a CPU model with no pretrained weights."""
    return LocalizationPolicy(K=K, pretrained=False, seed=0)


def _random_image(B: int = 1) -> torch.Tensor:
    """Return a (B, 3, 224, 224) float tensor."""
    return torch.rand(B, 3, 224, 224)


def _random_gt(B: int = 1, M: int = MAX_M) -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded gt_boxes (B, M, 4) and gt_mask (B, M) with 1 real box each."""
    gt_boxes = torch.zeros(B, M, 4)
    gt_mask = torch.zeros(B, M, dtype=torch.bool)
    for b in range(B):
        # A single valid GT box: center (0.5, 0.5), w=0.4, h=0.4
        gt_boxes[b, 0] = torch.tensor([0.5, 0.5, 0.4, 0.4])
        gt_mask[b, 0] = True
    return gt_boxes, gt_mask


def _random_target_bins(B: int = 1) -> dict[str, torch.Tensor]:
    """Return a dict of per-head target bin tensors, each (B,) long."""
    return {h: torch.randint(0, K, (B,)) for h in HEAD_NAMES}


# ---------------------------------------------------------------------------
# 1. head_params / flatten_head_grads
# ---------------------------------------------------------------------------


def test_flatten_head_grads_shape():
    """After a backward pass, flatten_head_grads returns a 1D tensor.

    The output must be exactly 1D; a 2D or ND result would break the cosine
    similarity computation (torch dot product requires 1D inputs).
    """
    model = _make_model()
    model.train()

    # Trigger a backward pass so all head params have .grad
    image = _random_image()
    logits = model(image)
    loss = sum(logits[h].mean() for h in HEAD_NAMES)
    loss.backward()

    params = head_params(model)
    flat = flatten_head_grads(params)

    assert flat.dim() == 1, (
        f"Expected 1D tensor, got {flat.dim()}D with shape {flat.shape}"
    )


def test_flatten_head_grads_matches_count():
    """Sum of numel across head params must equal the flattened tensor's numel.

    If any parameter is dropped or double-counted, the gradient vector would
    have the wrong size and give incorrect cosine similarities.
    """
    model = _make_model()
    model.train()

    image = _random_image()
    logits = model(image)
    loss = sum(logits[h].mean() for h in HEAD_NAMES)
    loss.backward()

    params = head_params(model)
    expected_numel = sum(p.numel() for p in params)
    flat = flatten_head_grads(params)

    assert flat.numel() == expected_numel, (
        f"Expected {expected_numel} elements in flattened grads, got {flat.numel()}"
    )


# ---------------------------------------------------------------------------
# 2. compute_supervised_head_gradient
# ---------------------------------------------------------------------------


def test_compute_supervised_head_gradient_runs():
    """compute_supervised_head_gradient on a 1-image dummy batch returns a finite 1D tensor.

    If the function crashes or returns NaN, the cosine analysis cannot run at all.
    A finite result confirms the ordinal CE loss backward pass is numerically stable.
    """
    model = _make_model()
    model.train()

    image = _random_image(B=1)
    target_bins = _random_target_bins(B=1)

    grad = compute_supervised_head_gradient(model, image, target_bins, K)

    assert grad.dim() == 1, f"Expected 1D gradient, got shape {grad.shape}"
    assert torch.isfinite(grad).all(), "Supervised gradient contains NaN or Inf"
    assert grad.numel() > 0, "Gradient tensor is empty"


# ---------------------------------------------------------------------------
# 3. compute_rl_head_gradient
# ---------------------------------------------------------------------------


def test_compute_rl_head_gradient_runs():
    """compute_rl_head_gradient with method='tailrl' and G=4 returns a finite 1D tensor.

    This tests the full RL gradient pipeline: forward, sampling, IoU reward,
    TailRL advantage, and policy-gradient backward. Any crash or NaN indicates a
    pipeline bug that would silently corrupt all gradient analysis results.
    """
    model = _make_model()
    model.train()

    image = _random_image(B=1)
    gt_boxes, gt_mask = _random_gt(B=1)

    from experiments.imagenet_localization.core.advantages import ADVANTAGE_FNS
    advantage_fn = ADVANTAGE_FNS["tailrl"]

    grad = compute_rl_head_gradient(model, image, gt_boxes, gt_mask, K, G=4, advantage_fn=advantage_fn)

    assert grad.dim() == 1, f"Expected 1D gradient, got shape {grad.shape}"
    assert torch.isfinite(grad).all(), "RL gradient contains NaN or Inf"
    assert grad.numel() > 0, "Gradient tensor is empty"


# ---------------------------------------------------------------------------
# 4. cosine
# ---------------------------------------------------------------------------


def test_cosine_identity_equals_one():
    """cosine(a, a) must equal 1.0 (up to floating-point tolerance).

    A vector is perfectly aligned with itself; cosine similarity = 1.0.
    If this fails, the cosine function has a bug (e.g., wrong normalization).
    """
    a = torch.randn(128)
    result = cosine(a, a)
    assert abs(result - 1.0) < 1e-5, (
        f"Expected cosine(a, a) ≈ 1.0, got {result}"
    )


def test_cosine_orthogonal_is_zero():
    """cosine of two orthogonal standard-basis vectors must equal 0.0.

    e1 = [1, 0, ...] and e2 = [0, 1, ...] have dot product 0.
    If the cosine is nonzero, the normalization or dot-product is wrong.
    """
    n = 64
    e1 = torch.zeros(n)
    e2 = torch.zeros(n)
    e1[0] = 1.0
    e2[1] = 1.0

    result = cosine(e1, e2)
    assert abs(result) < 1e-6, (
        f"Expected cosine(e1, e2) = 0.0 for orthogonal vectors, got {result}"
    )


def test_cosine_zero_vector_returns_zero():
    """cosine with a zero-norm vector must return 0.0, not NaN.

    Division by zero (||a|| = 0) must be guarded explicitly. If unguarded,
    the result would be NaN, which would corrupt all downstream statistics.
    """
    a = torch.zeros(64)
    b = torch.randn(64)

    result_ab = cosine(a, b)
    result_ba = cosine(b, a)

    assert result_ab == 0.0, (
        f"Expected cosine(zero, b) = 0.0, got {result_ab}"
    )
    assert result_ba == 0.0, (
        f"Expected cosine(b, zero) = 0.0, got {result_ba}"
    )
    assert not math.isnan(result_ab), "cosine(zero, b) returned NaN"
    assert not math.isnan(result_ba), "cosine(b, zero) returned NaN"


# ---------------------------------------------------------------------------
# 5. run_analysis smoke test
# ---------------------------------------------------------------------------


def test_run_analysis_smoke():
    """run_analysis with 3 mock images, 2 methods, 2 G values returns a correctly structured dict.

    Verifies: (a) the output dict has entries for every requested method and G,
    (b) each value is a finite Python float, (c) the function does not crash.
    We do NOT assert specific cosine values — they depend on random sampling.
    """
    torch.manual_seed(7)
    n_images = 3
    methods = ["tailrl", "reinforce"]
    Gs = [4, 8]

    model = _make_model()
    model.train()

    images = _random_image(B=n_images)
    gt_boxes, gt_mask = _random_gt(B=n_images)
    target_bins = _random_target_bins(B=n_images)

    results = run_analysis(
        model=model,
        images=images,
        gt_boxes=gt_boxes,
        gt_mask=gt_mask,
        target_bins=target_bins,
        K=K,
        methods=methods,
        Gs=Gs,
    )

    # Structural checks
    assert isinstance(results, dict), "run_analysis must return a dict"
    assert set(results.keys()) == set(methods), (
        f"Expected methods {methods}, got keys {list(results.keys())}"
    )

    for method in methods:
        assert isinstance(results[method], dict), (
            f"results['{method}'] must be a dict"
        )
        assert set(results[method].keys()) == set(Gs), (
            f"For method '{method}', expected G keys {Gs}, got {list(results[method].keys())}"
        )
        for G in Gs:
            val = results[method][G]
            assert isinstance(val, float), (
                f"results['{method}'][{G}] must be a float, got {type(val)}"
            )
            assert math.isfinite(val), (
                f"results['{method}'][{G}] = {val} is not finite"
            )
