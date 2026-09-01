"""Tests for the mask-based TailRL survival floor fix."""

import torch
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.imagenet_localization.core.advantages import tailrl_advantage


def _tailrl_no_mask(rewards):
    """Baseline: TailRL with NO mask (original commit 0f6613f behavior)."""
    N = rewards.shape[-1]
    if N <= 1:
        return torch.zeros_like(rewards)
    sorted_rewards, sort_indices = torch.sort(rewards, dim=-1, stable=True)
    batch_shape = sorted_rewards.shape[:-1]
    zero_col = torch.zeros(*batch_shape, 1, device=rewards.device, dtype=rewards.dtype)
    prev = torch.cat([zero_col, sorted_rewards[..., :-1]], dim=-1)
    gaps = sorted_rewards - prev
    survivors = torch.arange(N, 0, -1, device=rewards.device, dtype=rewards.dtype)
    increments = gaps / survivors
    weights = torch.cumsum(increments, dim=-1) * N
    weights = weights - weights.mean(dim=-1, keepdim=True)
    advantages = torch.zeros_like(rewards)
    advantages.scatter_(dim=-1, index=sort_indices, src=weights)
    return advantages


@pytest.mark.parametrize("N", [16, 64, 256, 1024, 4096, 10000])
def test_mask_noop_small_N(N):
    """At N ≤ 10000 with ε=1e-4, mask is all-True → identical to no mask."""
    torch.manual_seed(42)
    rewards = torch.rand(8, N)
    assert torch.allclose(
        tailrl_advantage(rewards, survival_eps=1e-4),
        _tailrl_no_mask(rewards),
        atol=1e-10,
    )


@pytest.mark.parametrize("N,expected_masked", [
    (16384, 1),
    (65536, 6),
    (131072, 13),
])
def test_mask_count(N, expected_masked):
    """At N > 10000, the expected number of increments are masked."""
    survivors = torch.arange(N, 0, -1, dtype=torch.float32)
    n_masked = (survivors / N < 1e-4).sum().item()
    assert n_masked == expected_masked


@pytest.mark.parametrize("N", [16, 64, 256, 1024])
def test_binary_recovery(N):
    """With binary rewards {0,1}, all successes get equal advantage,
    all failures get equal advantage."""
    torch.manual_seed(42)
    K = max(2, N // 4)
    rewards = torch.zeros(N)
    rewards[:K] = 1.0
    rewards = rewards[torch.randperm(N)]
    adv = tailrl_advantage(rewards.unsqueeze(0), survival_eps=1e-4).squeeze(0)
    successes = (rewards == 1.0)
    s_vals = adv[successes]
    f_vals = adv[~successes]
    assert (s_vals - s_vals.mean()).abs().max() < 1e-5, "Successes should have equal advantage"
    assert (f_vals - f_vals.mean()).abs().max() < 1e-5, "Failures should have equal advantage"


def test_eps_zero_disables_mask():
    """survival_eps=0 → no mask → identical to baseline."""
    torch.manual_seed(42)
    rewards = torch.rand(4, 16384)
    assert torch.allclose(
        tailrl_advantage(rewards, survival_eps=0),
        _tailrl_no_mask(rewards),
        atol=1e-10,
    )


def test_mask_domain_matches_tailrl_population():
    """Masked increments = quality levels where S_emp < ε
    = quality levels where tailrl_population's log(clamp(P, ε)) is constant."""
    N, eps = 131072, 1e-4
    survivors = torch.arange(N, 0, -1, dtype=torch.float32)
    mask = (survivors / N >= eps)
    n_false = (~mask).sum().item()
    assert n_false == int(N * eps)
    # boundary: last unmasked position has fraction just ≥ ε
    boundary_idx = N - n_false - 1
    assert survivors[boundary_idx] / N >= eps
    assert survivors[boundary_idx + 1] / N < eps


def test_advantages_mean_zero():
    """Mean-centering: advantages should sum to zero along last dim.

    Tolerance scales with N: float32 mean-center has residual sum bounded
    by ~N · ε_machine · max|weight|, and weights here are O(N).
    """
    torch.manual_seed(42)
    for N in [64, 1024, 16384]:
        rewards = torch.rand(4, N)
        adv = tailrl_advantage(rewards, survival_eps=1e-4)
        tol = max(1e-4, 1e-7 * N)
        assert adv.sum(dim=-1).abs().max() < tol, (
            f"N={N}: advantages not mean-zero (tol={tol})"
        )
