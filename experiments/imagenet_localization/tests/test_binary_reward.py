"""Tests for the binary IoU-threshold reward transforms.

These pair a {0,1} success reward (IoU > tau) with the ``bmaxrl_adv_est``
estimator ``(r - mu) / (mu + eps)``. Together they implement binary MaxRL at a
chosen CorLoc operating point, so the tests pin down both the thresholding
semantics and the estimator algebra that follows from it.
"""

from __future__ import annotations

import pytest
import torch

from experiments.imagenet_localization.core.advantages import (
    ADVANTAGE_FNS,
    REWARD_TRANSFORMS,
    binary_iou_50_transform,
    binary_iou_75_transform,
    binary_maxrl_advantage,
    bmaxrl_adv_est_advantage,
)
from experiments.imagenet_localization.models.model import LocalizationPolicy
from experiments.imagenet_localization.training.train import rl_training_step

MAX_M = 8


# ---------------------------------------------------------------------------
# Thresholding semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn,tau", [
    (binary_iou_50_transform, 0.5),
    (binary_iou_75_transform, 0.75),
])
def test_threshold_is_strictly_greater(fn, tau):
    """r == tau must map to 0.0; anything above to 1.0.

    Strict `>` matches the spec and the existing reward_at_0_5 / reward_at_0_75
    metrics, which also use `>`. An off-by-one here silently shifts the
    operating point the arm is optimising.
    """
    r = torch.tensor([[0.0, tau - 1e-6, tau, tau + 1e-6, 1.0]])
    out = fn(r)
    assert out.tolist() == [[0.0, 0.0, 0.0, 1.0, 1.0]]


@pytest.mark.parametrize("fn", [binary_iou_50_transform, binary_iou_75_transform])
def test_output_is_float_zero_one_same_shape_dtype(fn):
    """Result must be a {0.0,1.0} float tensor, not a bool mask."""
    r = torch.rand(3, 7, dtype=torch.float64)
    out = fn(r)
    assert out.shape == r.shape
    assert out.dtype == r.dtype
    assert set(out.unique().tolist()) <= {0.0, 1.0}


def test_the_two_thresholds_differ_where_expected():
    """0.5 and 0.75 must disagree exactly on the band (0.5, 0.75]."""
    r = torch.tensor([[0.4, 0.6, 0.75, 0.8]])
    assert binary_iou_50_transform(r).tolist() == [[0.0, 1.0, 1.0, 1.0]]
    assert binary_iou_75_transform(r).tolist() == [[0.0, 0.0, 0.0, 1.0]]


# ---------------------------------------------------------------------------
# Estimator algebra: binary reward + bmaxrl_adv_est
# ---------------------------------------------------------------------------

def test_advantage_matches_closed_form_success_and_failure():
    """success -> (1-mu)/(mu+eps), failure -> -mu/(mu+eps), for r in {0,1}.

    In the eps -> 0 limit these are the familiar (N-K)/K and -1. We assert the
    EXACT eps-aware form, then separately confirm the idealised form holds to
    the relative accuracy eps/mu that the regulariser permits -- being explicit
    that (r-mu)/(mu+eps) is only equal to binary MaxRL up to that eps.
    """
    eps = 1e-8
    # 8 rollouts, 3 clear 0.5 -> K=3, mu=3/8
    r = torch.tensor([[0.9, 0.1, 0.8, 0.2, 0.95, 0.3, 0.05, 0.4]], dtype=torch.float64)
    b = binary_iou_50_transform(r)
    N = b.shape[-1]
    K = int(b.sum().item())
    assert K == 3
    mu = K / N
    adv = bmaxrl_adv_est_advantage(b, eps=eps)

    for i in range(N):
        exact = (1.0 - mu) / (mu + eps) if b[0, i] == 1.0 else -mu / (mu + eps)
        assert abs(adv[0, i].item() - exact) < 1e-12, f"index {i}"

    # Idealised (eps -> 0) form, to the accuracy eps/mu allows.
    for i in range(N):
        ideal = (N - K) / K if b[0, i] == 1.0 else -1.0
        assert abs(adv[0, i].item() - ideal) <= abs(ideal) * (eps / mu) + 1e-12


@pytest.mark.parametrize("fn", [binary_iou_50_transform, binary_iou_75_transform])
def test_bmaxrl_adv_est_equals_binary_maxrl_on_binary_rewards(fn):
    """On {0,1} rewards the continuous estimator reduces to binary_maxrl exactly.

    bmaxrl_adv_est is (r-mu)/(mu+eps); binary_maxrl binarizes at r == 1.0. After
    the transform the rewards ARE exactly 0.0/1.0, so the two must coincide.
    This is the identity the arm's interpretation rests on.
    """
    torch.manual_seed(0)
    eps = 1e-8
    r = torch.rand(16, 32, dtype=torch.float64)
    b = fn(r)
    a1 = bmaxrl_adv_est_advantage(b, eps=eps)
    a2 = binary_maxrl_advantage(b)
    # Not bit-exact: bmaxrl_adv_est carries eps in the denominator, so it agrees
    # with binary_maxrl only to relative order eps/mu (~8e-8 here). Asserting
    # exact equality would be asserting something false.
    torch.testing.assert_close(a1, a2, rtol=1e-6, atol=1e-9)


def test_all_failure_group_yields_zero_advantage():
    """No rollout clears the bar -> mu=0 -> zero advantage (no gradient).

    This is the degenerate regime whose probability (1-p)^N shrinks with N,
    and is the reason the rollout count is the axis of interest for this arm.
    """
    r = torch.full((2, 8), 0.1, dtype=torch.float64)     # nothing above 0.5
    adv = bmaxrl_adv_est_advantage(binary_iou_50_transform(r))
    assert torch.all(adv == 0.0)


def test_all_success_group_yields_zero_advantage():
    """Everything clears the bar -> mu=1 -> (1-1)/1 = 0 for every rollout."""
    r = torch.full((2, 8), 0.99, dtype=torch.float64)
    adv = bmaxrl_adv_est_advantage(binary_iou_50_transform(r))
    torch.testing.assert_close(adv, torch.zeros_like(adv), atol=1e-9, rtol=0)


def test_advantages_are_mean_zero_per_group():
    """Advantages sum to zero over a non-degenerate group.

    K successes at (N-K)/K plus (N-K) failures at -1 gives
    (N-K) - (N-K) = 0, so the estimator adds no net drift to the score
    function; it only reweights within the group.
    """
    torch.manual_seed(1)
    r = torch.rand(8, 64, dtype=torch.float64)
    for fn in (binary_iou_50_transform, binary_iou_75_transform):
        b = fn(r)
        adv = bmaxrl_adv_est_advantage(b)
        K = b.sum(dim=-1)
        nondegenerate = (K > 0) & (K < b.shape[-1])
        assert bool(nondegenerate.any()), "test data produced no usable group"
        sums = adv[nondegenerate].sum(dim=-1)
        torch.testing.assert_close(
            sums, torch.zeros_like(sums), atol=1e-8, rtol=0,
        )


# ---------------------------------------------------------------------------
# Registry + end-to-end wiring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["binary_0.5", "binary_0.75"])
def test_registered_in_reward_transforms(key):
    assert key in REWARD_TRANSFORMS
    out = REWARD_TRANSFORMS[key](torch.tensor([[0.0, 1.0]]))
    assert out.tolist() == [[0.0, 1.0]]


@pytest.mark.parametrize("key", ["binary_0.5", "binary_0.75"])
def test_accepted_by_run_argparse(key, monkeypatch):
    from experiments.imagenet_localization.run import parse_args
    monkeypatch.setattr("sys.argv", [
        "run.py", "--method", "bmaxrl_adv_est", "--reward_transform", key,
        "--data_dir", "/tmp", "--output_dir", "/tmp",
    ])
    assert parse_args().reward_transform == key


@pytest.mark.parametrize("key", ["binary_0.5", "binary_0.75"])
def test_training_step_end_to_end_keeps_raw_iou_in_metrics(key):
    """One rl_training_step with the transform: finite loss, and the LOGGED
    rewards must remain continuous IoU (not binarised), so reward_mean and
    val metrics stay comparable across arms."""
    torch.manual_seed(0)
    B, N, K = 2, 16, 10
    model = LocalizationPolicy(K=K, pretrained=False)
    gt = torch.zeros(B, MAX_M, 4)
    gt[:, :, :2] = 0.5
    gt[:, :, 2:] = 0.3
    batch = {
        "images": torch.randn(B, 3, 224, 224),
        "gt_boxes": gt,
        "gt_mask": torch.ones(B, MAX_M, dtype=torch.bool),
    }
    out = rl_training_step(
        model, batch, method="bmaxrl_adv_est", N=N, K=K,
        device=torch.device("cpu"), reward_transform=key,
    )
    assert torch.isfinite(out["loss"])
    assert not torch.isnan(out["advantages"]).any()
    r = out["rewards"]
    assert r.shape == (B, N)
    assert ((r >= 0.0) & (r <= 1.0)).all()
    # Raw IoU must NOT have been binarised in the logged tensor.
    assert not set(r.unique().tolist()) <= {0.0, 1.0}, (
        "logged rewards were binarised; metrics would no longer report true IoU"
    )
