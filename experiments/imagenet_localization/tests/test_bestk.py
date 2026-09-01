"""Tests for the unbiased best@k order-statistic estimator (bestk.py).

best@k = expected max reward among k rollouts, estimated unbiasedly from M >= k
iid rollout rewards via the order-statistic U-statistic:

    best@k = sum_{i=k}^{M} [C(i-1, k-1) / C(M, k)] * r_(i)      (r ascending)

Each test states the invariant and, where applicable, the hand-computed value.
All tests run on CPU in well under a second.
"""

from __future__ import annotations

import math

import pytest
import torch

from experiments.imagenet_localization.core.bestk import (
    best_at_k_unbiased,
    best_at_k_weights,
)


def test_weights_sum_to_one():
    """The order-statistic weights are a convex combination (hockey-stick identity)."""
    for M, k in [(4, 2), (10, 1), (10, 10), (4096, 1024), (256, 64)]:
        w = best_at_k_weights(M, k)
        assert w.shape == (M,)
        assert torch.isclose(w.sum(), torch.tensor(1.0, dtype=w.dtype), atol=1e-9), (
            f"weights for M={M}, k={k} sum to {w.sum().item()}, expected 1.0"
        )
        # Weights below rank k must be exactly zero.
        assert torch.all(w[: k - 1] == 0.0)


def test_best_at_1_is_mean():
    """best@1 = E[max of 1 sample] = mean of the rewards."""
    r = torch.tensor([[0.1, 0.7, 0.3, 0.9, 0.5], [0.0, 0.0, 1.0, 1.0, 0.5]])
    out = best_at_k_unbiased(r, [1])  # (2, 1)
    assert torch.allclose(out[:, 0], r.mean(dim=-1), atol=1e-6)


def test_best_at_M_is_max():
    """best@M (k == number of samples) = the maximum reward."""
    r = torch.tensor([[0.1, 0.7, 0.3, 0.9, 0.5], [0.0, 0.0, 1.0, 1.0, 0.5]])
    out = best_at_k_unbiased(r, [5])  # (2, 1)
    assert torch.allclose(out[:, 0], r.max(dim=-1).values, atol=1e-6)


def test_hand_computed_M4_k2():
    """M=4, k=2, r=[0,1,2,3]: average max over all C(4,2)=6 pairs = 14/6.

    pairs maxes: {0,1}->1 {0,2}->2 {0,3}->3 {1,2}->2 {1,3}->3 {2,3}->3 => 14/6.
    """
    r = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    out = best_at_k_unbiased(r, [2])
    assert torch.allclose(out[:, 0], torch.tensor([14.0 / 6.0]), atol=1e-6)


def test_hand_computed_matches_bruteforce():
    """Unbiased estimator equals the exact average-over-all-k-subsets, on a random small case."""
    torch.manual_seed(0)
    r = torch.rand(7)
    M = r.numel()
    from itertools import combinations
    for k in range(1, M + 1):
        subsets = list(combinations(range(M), k))
        brute = sum(r[list(s)].max().item() for s in subsets) / len(subsets)
        est = best_at_k_unbiased(r.unsqueeze(0), [k])[0, 0].item()
        assert abs(est - brute) < 1e-6, f"k={k}: est={est}, brute={brute}"


def test_batched_rows_independent():
    """(B, M) input: each row's best@k is computed independently over its own samples."""
    r = torch.tensor([[0.0, 1.0, 2.0, 3.0], [3.0, 3.0, 3.0, 3.0]])
    out = best_at_k_unbiased(r, [2])  # (2, 1)
    # Row 0 -> 14/6 ; row 1 (all equal) -> 3.0 for any k.
    assert torch.allclose(out[:, 0], torch.tensor([14.0 / 6.0, 3.0]), atol=1e-6)


def test_monotonic_increasing_in_k():
    """best@k is non-decreasing in k (more tries can only help)."""
    torch.manual_seed(1)
    r = torch.rand(1, 4096)
    ks = [1, 4, 16, 64, 256, 1024]
    out = best_at_k_unbiased(r, ks)[0]  # (6,)
    diffs = out[1:] - out[:-1]
    assert torch.all(diffs >= -1e-7), f"best@k not monotone: {out.tolist()}"


def test_all_equal_rewards_constant():
    """If all M rewards are identical, best@k equals that value for every k."""
    r = torch.full((1, 256), 0.42)
    out = best_at_k_unbiased(r, [1, 4, 16, 64, 256])[0]
    assert torch.allclose(out, torch.full((5,), 0.42), atol=1e-6)


def test_invalid_k_raises():
    """k must satisfy 1 <= k <= M."""
    with pytest.raises(ValueError):
        best_at_k_weights(M=10, k=11)
    with pytest.raises(ValueError):
        best_at_k_weights(M=10, k=0)


def test_large_M_k_numerically_stable():
    """M=4096, k=1024 (the production grid) yields a finite value in [0,1] for IoU-like input."""
    torch.manual_seed(2)
    r = torch.rand(3, 4096)  # IoU-like rewards in [0,1]
    out = best_at_k_unbiased(r, [1, 4, 16, 64, 256, 1024])
    assert torch.isfinite(out).all()
    assert (out >= 0.0).all() and (out <= 1.0).all()
