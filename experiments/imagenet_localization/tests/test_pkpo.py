"""Tests for pkpo_advantage in experiments.imagenet_localization.core.advantages.

Reference: Walder & Karkhanis 2025, NeurIPS, arXiv:2505.15201, Listing 1.

The PyTorch implementation is verified bit-exact (within float64 epsilon)
against the paper's numpy reference for several (N, K) pairs and for
batched inputs.
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest
import torch

# Make sure the package is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.imagenet_localization.core.advantages import (
    ADVANTAGE_FNS,
    pkpo_advantage,
)


# ---------------------------------------------------------------------------
# Reference implementation — verbatim port of the paper's Listing 1.
# ---------------------------------------------------------------------------

def _ref_m_normed(N, K, i, j):
    if i == j and i >= K - 1:
        return (
            K / (N - K + 1)
            * np.prod(np.arange(i - K + 2, i + 1) / np.arange(N - K + 2, N + 1))
        )
    elif j > i and j >= K - 1 and K >= 2:
        return (
            K / (N - K + 1) * (K - 1) / N
            * np.prod(np.arange(j - K + 2, j) / np.arange(N - K + 2, N))
        )
    return 0.0


def _ref_m_diagonal(N, K):
    return np.array([_ref_m_normed(N, K, i, i) for i in range(N)])


def _ref_delta(N, K, i):
    return _ref_m_normed(N, K, i, i + 1) - _ref_m_normed(N, K, i + 1, i + 1)


def _ref_deltas(N, K):
    return np.array([_ref_delta(N - 1, K, i) for i in range(N - 2)])


def _ref_sorted_apply(func):
    def inner(x, *args, **kwargs):
        i_sort = np.argsort(x)
        out = np.zeros_like(x)
        out[i_sort] = func(x[i_sort], *args, **kwargs)
        return out
    return inner


@_ref_sorted_apply
def _ref_s(g, K):
    N = len(g)
    c = g * _ref_m_diagonal(N, K)
    c[: (N - 1)] += g[1:] * _ref_deltas(N + 1, K)
    return np.cumsum(c[::-1])[::-1]


@_ref_sorted_apply
def _ref_b(g, K):
    N = len(g)
    w = (_ref_m_diagonal(N - 1, K) * np.arange(1, N)).astype(float)
    w[1:] += _ref_deltas(N, K) * np.arange(1, N - 1)
    c1 = np.array([(w * g[1:]).sum()])
    c2 = (g[:-1] - g[1:]) * w
    return np.cumsum(np.concatenate((c1, c2)))


def _ref_sloo_minus_one(g, K):
    return _ref_s(g, K) - _ref_b(g, K - 1) * K / (K - 1) / len(g)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N,K", [
    (4, 2), (8, 4), (16, 8), (32, 16), (64, 16),
    (128, 16), (256, 16), (16, 16), (16, 2),
])
def test_pkpo_matches_paper_reference(N, K):
    """PyTorch impl must equal the numpy reference within float64 epsilon."""
    np.random.seed(42)
    g_np = np.random.uniform(0, 1, size=N)
    ref = _ref_sloo_minus_one(g_np, K)
    # paper_scale=True -> raw Eq-33 (sum-convention) values, which is what the
    # numpy reference computes. The default return is N x these (see the
    # convention note in pkpo_advantage): that rescaling is what makes the
    # estimator match the true gradient under train.py's .mean(-1) loss.
    out = pkpo_advantage(
        torch.tensor(g_np, dtype=torch.float64), k=K, paper_scale=True,
    ).numpy()
    err = np.abs(ref - out).max()
    assert err < 1e-9, f"N={N} K={K}: max_err={err:.2e}"


def test_pkpo_batched_matches_per_row_reference():
    """Batched (B, N) call must equal per-row numpy reference."""
    np.random.seed(0)
    B, N, K = 8, 64, 16
    g_np = np.random.uniform(0, 1, size=(B, N))
    ref = np.stack([_ref_sloo_minus_one(g_np[b], K) for b in range(B)])
    out = pkpo_advantage(
        torch.tensor(g_np, dtype=torch.float64), k=K, paper_scale=True,
    ).numpy()
    err = np.abs(ref - out).max()
    assert err < 1e-9, f"batched N={N} K={K}: max_err={err:.2e}"


def test_pkpo_default_k_from_env():
    """Calling pkpo_advantage(rewards) without k must use PKPO_K_DEFAULT."""
    from experiments.imagenet_localization.core.advantages import PKPO_K_DEFAULT
    np.random.seed(1)
    rewards = torch.rand(32, dtype=torch.float64)
    out_default = pkpo_advantage(rewards)
    out_explicit = pkpo_advantage(rewards, k=PKPO_K_DEFAULT)
    assert torch.allclose(out_default, out_explicit, atol=1e-12)


def test_pkpo_k1_falls_back_to_reinforce():
    """k=1 has K-1=0 in denominator (Eq 33). Paper says k=1 is no transform.
    Our impl returns mean-centered rewards (== reinforce_advantage)."""
    rewards = torch.rand(64)
    out = pkpo_advantage(rewards, k=1)
    expected = rewards - rewards.mean()
    assert torch.allclose(out, expected, atol=1e-6)


def test_pkpo_dispatch_via_advantage_fns():
    """ADVANTAGE_FNS['pkpo'] is callable as fn(rewards) and matches the
    direct call (uses PKPO_K_DEFAULT)."""
    rewards = torch.rand(32)
    out_dispatch = ADVANTAGE_FNS["pkpo"](rewards)
    out_direct = pkpo_advantage(rewards)
    assert torch.allclose(out_dispatch, out_direct, atol=1e-7)


def test_pkpo_shape_dtype_preserved():
    """Output shape == input shape, dtype preserved (no silent upcast)."""
    rewards32 = torch.rand(4, 32, dtype=torch.float32)
    out32 = pkpo_advantage(rewards32, k=8)
    assert out32.shape == (4, 32)
    assert out32.dtype == torch.float32

    rewards64 = torch.rand(4, 32, dtype=torch.float64)
    out64 = pkpo_advantage(rewards64, k=8)
    assert out64.shape == (4, 32)
    assert out64.dtype == torch.float64


def test_pkpo_no_nan_or_inf():
    """No NaN or ±inf advantages on random inputs across several (N, K)."""
    for N, K in [(16, 2), (32, 8), (256, 16), (256, 128)]:
        torch.manual_seed(N + K)
        rewards = torch.rand(8, N)
        out = pkpo_advantage(rewards, k=K)
        assert not torch.isnan(out).any(), f"NaN at N={N} K={K}"
        assert not torch.isinf(out).any(), f"Inf at N={N} K={K}"


def test_pkpo_invalid_k_raises():
    """k outside [1, N] must raise ValueError."""
    rewards = torch.rand(16)
    with pytest.raises(ValueError):
        pkpo_advantage(rewards, k=0)
    with pytest.raises(ValueError):
        pkpo_advantage(rewards, k=17)


def test_pkpo_constant_rewards_zero_advantage():
    """When all rewards are equal, sloo_minus_one weight on each sample
    is the same constant; the advantage is zero (or numerically negligible).

    Reasoning: when g_i are identical, the sort is degenerate and every
    element gets the same effective reward, so the gradient contribution
    is zero in expectation.  The estimator is exact (not just unbiased)
    in this case because every k-subset has the same max.
    """
    rewards = torch.full((4, 32), 0.5)
    out = pkpo_advantage(rewards, k=8)
    assert out.abs().max().item() < 1e-6


def test_pkpo_eq_n_matches_tzsm25_max_form():
    """At k=N, sloo_minus_one reduces to (max(all) - max(without_i)).

    For non-largest samples i: max(all) - max(without_i) = max - max = 0.
    For the largest sample i*:  max(all) - max(without_i*) = top1 - top2.
    This is the special case (k=n) cited as TZSM25 in the paper.
    """
    torch.manual_seed(7)
    N = 8
    rewards = torch.rand(N)
    # raw Eq-33 values: the top1-top2 identity is a statement about the
    # paper's estimator, before the policy-gradient rescaling.
    out = pkpo_advantage(rewards, k=N, paper_scale=True)

    sorted_rewards, _ = torch.sort(rewards)        # ascending
    largest_idx = torch.argmax(rewards)
    expected_largest = sorted_rewards[-1] - sorted_rewards[-2]   # top1 - top2

    others_mask = torch.ones(N, dtype=torch.bool)
    others_mask[largest_idx] = False
    assert out[others_mask].abs().max().item() < 1e-6
    assert torch.allclose(out[largest_idx], expected_largest, atol=1e-6)


# ---------------------------------------------------------------------------
# Regression tests for the sum-vs-mean convention bug (advantages were 1/N too
# small, so the arm silently failed to train: grad_norm ~2.5e-4 vs ~3 for tailrl).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N,K", [(64, 16), (256, 16), (1024, 16), (1024, 64), (1024, 256)])
def test_pkpo_default_is_n_times_paper_scale(N, K):
    """The dispatched value must be exactly N x the raw Eq-33 value."""
    torch.manual_seed(0)
    r = torch.rand(2, N, dtype=torch.float64)
    raw = pkpo_advantage(r, k=K, paper_scale=True)
    pg = pkpo_advantage(r, k=K)
    torch.testing.assert_close(pg, raw * float(N), rtol=1e-12, atol=0)


@pytest.mark.parametrize("N,K", [(256, 16), (1024, 64), (1024, 256)])
def test_pkpo_advantage_scale_is_not_1_over_n_too_small(N, K):
    """Guard the sum-vs-mean convention bug without asserting a false claim.

    Before the fix mean|adv| was ~1/N of what it should be (1.4e-4 at
    N=256,K=16 vs tailrl 4.4e-1), putting the gradient below the weight-decay term
    and freezing training.

    The band is deliberately loose (tailrl/1000). PKPO advantages are NOT expected
    to match tailrl in magnitude: the true gradient of E[max@k] genuinely shrinks
    as k grows because the objective saturates (exact-gradient norms measured on
    a non-saturating categorical toy at N=1024: 4.07e-3 at k=16, 2.26e-3 at
    k=64, 1.15e-3 at k=256). Demanding parity with tailrl would assert something
    false. The 1/N regression this guards against is ~3 orders of magnitude,
    far outside the band.
    """
    from experiments.imagenet_localization.core.advantages import ADVANTAGE_FNS
    torch.manual_seed(0)
    r = torch.rand(32, N, dtype=torch.float64) * 0.6 + 0.2
    pk = pkpo_advantage(r, k=K).abs().mean().item()
    raw = pkpo_advantage(r, k=K, paper_scale=True).abs().mean().item()
    tailrl = ADVANTAGE_FNS["tailrl"](r).abs().mean().item()
    assert pk > tailrl / 1000.0, (
        f"N={N} K={K}: pkpo mean|adv|={pk:.3e} vs tailrl {tailrl:.3e} -- "
        "the 1/N convention bug is back"
    )
    assert abs(pk - raw * N) < 1e-9 * max(1.0, pk)


def test_pkpo_k1_is_not_rescaled():
    """k=1 is already mean-convention (REINFORCE) and must stay unscaled."""
    torch.manual_seed(0)
    r = torch.rand(1, 32, dtype=torch.float64)
    out = pkpo_advantage(r, k=1)
    torch.testing.assert_close(out, r - r.mean(dim=-1, keepdim=True),
                               rtol=1e-12, atol=0)


def test_pkpo_k1_and_k2_scales_are_continuous():
    """k=1 and k=2 must not differ by orders of magnitude.

    The original code returned mean-convention at k=1 and sum-convention at
    k>=2, producing a ~200x cliff between adjacent k. After the fix the two
    should be within a small factor of each other.
    """
    torch.manual_seed(0)
    r = torch.rand(16, 64, dtype=torch.float64)
    a1 = pkpo_advantage(r, k=1).abs().mean().item()
    a2 = pkpo_advantage(r, k=2).abs().mean().item()
    assert 0.05 < a2 / a1 < 20.0, f"k=1 {a1:.3e} vs k=2 {a2:.3e} (ratio {a2/a1:.1f})"
