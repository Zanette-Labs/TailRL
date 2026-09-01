"""Per-group advantage estimators. Pure torch, no verl dependency.

Every function takes a 1-D tensor of shape ``[G]`` -- one scalar reward per rollout
for a single prompt -- and returns advantages of the same shape. ``verl_register``
wraps them into the batched signature verl calls; keeping the maths here means the
estimators can be unit-tested without Ray, vLLM, or a GPU (see ``tests/``).

The three estimators compared in this experiment are :func:`tailrl_advantage`,
:func:`grpo_advantage` and :func:`rloo_advantage`. Everything else in this file is a
baseline or an ablation knob and is not part of the headline comparison.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# TailRL
# ---------------------------------------------------------------------------

def tailrl_advantage(
    rewards: torch.Tensor,
    scale_by_group_size: bool = True,
) -> torch.Tensor:
    """TailRL: tail-likelihood advantage, computed as gap-over-survivors.

    Sort the group ascending to get order statistics ``r_(1) <= ... <= r_(G)``, take
    the gap between consecutive order statistics, and divide each gap by the number
    of rollouts at or above it::

        gap_k       = r_(k) - r_(k-1)            with r_(0) := 0
        survivors_k = G - k + 1                  (rank k counts itself)
        A_(k)       = G * sum_{j <= k} gap_j / survivors_j     (then mean-centred
                                                                and unsorted)

    The sum is a discrete integral over reward thresholds: the gap between two
    adjacent order statistics is a band of thresholds that exactly the survivors
    clear, and ``1/survivors`` is the derivative of the log tail probability
    estimated on that band. Summing the bands gives the finite-``G`` estimator of the
    average log-tail-likelihood. Mean-centring is the usual variance-reduction
    baseline and leaves the gradient unbiased.

    The leading ``G`` is the convention used throughout this repository: it is what
    makes the reduction to binary MaxRL exact -- see :func:`binary_maxrl_advantage`,
    an identity pinned by ``tests/test_advantages.py``. Being a uniform rescaling of
    every advantage, AdamW's second-moment normalization absorbs it, so it does not
    change the effective learning rate. ``scale_by_group_size=False`` drops it.

    Args:
        rewards: shape ``[G]``, one scalar reward per rollout of one prompt.
        scale_by_group_size: multiply by ``G`` before mean-centring (default on).

    Returns:
        Advantages of shape ``[G]``, summing to zero. A group of size 1 gets zeros
        (there is no tail to integrate over).
    """
    G = rewards.numel()
    if G <= 1:
        return torch.zeros_like(rewards)

    sorted_rewards, sort_indices = torch.sort(rewards)

    prev = torch.cat([
        torch.zeros(1, dtype=rewards.dtype, device=rewards.device),
        sorted_rewards[:-1],
    ])
    gaps = sorted_rewards - prev                                    # [G]

    survivors = torch.arange(G, 0, -1, dtype=rewards.dtype, device=rewards.device)
    w_sorted = torch.cumsum(gaps / survivors, dim=0)                # [G]

    if scale_by_group_size:
        w_sorted = w_sorted * G

    w_centered = w_sorted - w_sorted.mean()

    advantages = torch.zeros_like(rewards)
    advantages[sort_indices] = w_centered
    return advantages


# ---------------------------------------------------------------------------
# Expected-reward baselines (the two arms TailRL is compared against)
# ---------------------------------------------------------------------------

def grpo_advantage(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """GRPO: within-group z-score.

        A_i = (r_i - mean(r)) / (std(r) + eps)

    An all-equal group carries no signal and gets zeros rather than a division by a
    near-zero standard deviation.
    """
    G = rewards.numel()
    if G <= 1:
        return torch.zeros_like(rewards)
    std = rewards.std(unbiased=False)
    if std < eps:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (std + eps)


def rloo_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """RLOO: leave-one-out baseline (Kool et al. 2019; Ahmadian et al. 2024).

        A_i = r_i - mean_{j != i}(r_j) = G/(G-1) * (r_i - mean(r))
    """
    G = rewards.numel()
    if G <= 1:
        return torch.zeros_like(rewards)
    return (G / (G - 1)) * (rewards - rewards.mean())


def reinforce_baseline_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """REINFORCE with a per-group mean baseline: ``A_i = r_i - mean(r)``."""
    return rewards - rewards.mean()


# ---------------------------------------------------------------------------
# Baselines from prior work
# ---------------------------------------------------------------------------

def binary_maxrl_advantage(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Binary MaxRL (Tajwar et al., 2026): ``A_i = (r_i - mean(r)) / mean(r)``.

    On binary rewards with ``C`` successes out of ``G`` this is ``(G-C)/C`` for a
    success and ``-1`` for a failure, and it is *exactly* what
    :func:`tailrl_advantage` returns on the same input -- the binary-recovery
    proposition. ``tests/test_advantages.py`` pins the identity.

    Included as a baseline, not as this paper's estimator.
    """
    G = rewards.numel()
    if G <= 1:
        return torch.zeros_like(rewards)
    mean = rewards.mean()
    if mean < eps:
        return torch.zeros_like(rewards)
    return (rewards - mean) / mean


def _pkpo_m_normed(N: int, K: int, i: int, j: int) -> float:
    """Normalized weight ``m_ij / C(N,K)``. Direct port of Listing 1, ``_m_normed``."""
    if i == j and i >= K - 1:
        num = 1.0
        den = 1.0
        for a in range(i - K + 2, i + 1):
            num *= a
        for a in range(N - K + 2, N + 1):
            den *= a
        return (K / (N - K + 1)) * (num / den)
    if j > i and j >= K - 1 and K >= 2:
        num = 1.0
        den = 1.0
        for a in range(j - K + 2, j):
            num *= a
        for a in range(N - K + 2, N):
            den *= a
        return (K / (N - K + 1)) * ((K - 1) / N) * (num / den)
    return 0.0


def _pkpo_m_diagonal(N: int, K: int, dtype, device) -> torch.Tensor:
    return torch.tensor(
        [_pkpo_m_normed(N, K, i, i) for i in range(N)], dtype=dtype, device=device,
    )


def _pkpo_deltas(N: int, K: int, dtype, device) -> torch.Tensor:
    # Listing 1 returns length N-2, not N-1.
    return torch.tensor(
        [
            _pkpo_m_normed(N - 1, K, i, i + 1) - _pkpo_m_normed(N - 1, K, i + 1, i + 1)
            for i in range(N - 2)
        ],
        dtype=dtype, device=device,
    )


def _pkpo_s_sorted(g_sorted: torch.Tensor, K: int) -> torch.Tensor:
    """Eq. (19), the raw ``s_i``. ``g_sorted`` must already be ascending."""
    N = g_sorted.numel()
    dtype, device = g_sorted.dtype, g_sorted.device
    diag = _pkpo_m_diagonal(N, K, dtype, device)
    c = (g_sorted * diag).clone()
    if N >= 2:
        deltas = _pkpo_deltas(N + 1, K, dtype, device)
        c[: N - 1] = c[: N - 1] + g_sorted[1:] * deltas
    return torch.flip(torch.cumsum(torch.flip(c, [0]), dim=0), [0])


def _pkpo_b_sorted(g_sorted: torch.Tensor, K: int) -> torch.Tensor:
    """Baseline ``b`` used by the ``loo`` and ``loo_minus_one`` variants."""
    N = g_sorted.numel()
    dtype, device = g_sorted.dtype, g_sorted.device
    if N <= 1 or K < 1:
        return torch.zeros(N, dtype=dtype, device=device)
    diag_nm1 = _pkpo_m_diagonal(N - 1, K, dtype, device)
    w = (diag_nm1 * torch.arange(1, N, dtype=dtype, device=device)).clone()
    if N >= 3:
        deltas = _pkpo_deltas(N, K, dtype, device)
        w[1:] = w[1:] + deltas * torch.arange(1, N - 1, dtype=dtype, device=device)
    c1 = (w * g_sorted[1:]).sum().unsqueeze(0)
    c2 = (g_sorted[:-1] - g_sorted[1:]) * w
    return torch.cumsum(torch.cat([c1, c2], dim=0), dim=0)


def pkpo_advantage(
    rewards: torch.Tensor,
    k_opt: int = 8,
    variant: str = "loo_minus_one",
) -> torch.Tensor:
    """Pass@K Policy Optimization (Walder & Karkhanis, 2025, arXiv:2505.15201).

    A reward transformation ``[G] -> [G]`` that is unbiased for the gradient of
    ``max_g@k_opt``. Ported from the paper's Listing 1; included as a baseline.
    """
    G = rewards.numel()
    if G <= 1:
        return torch.zeros_like(rewards)
    if not 1 <= k_opt <= G:
        raise ValueError(f"pkpo_advantage: k_opt={k_opt} must be in [1, G={G}]")
    if variant not in ("raw", "loo", "loo_minus_one"):
        raise ValueError(f"pkpo_advantage: unknown variant {variant!r}")

    sorted_rewards, sort_idx = torch.sort(rewards)

    if k_opt == 1:
        s_sorted = sorted_rewards - sorted_rewards.mean()
    else:
        s_sorted = _pkpo_s_sorted(sorted_rewards, k_opt)
        if variant == "loo":
            b = torch.zeros(G, dtype=rewards.dtype, device=rewards.device)
            for i in range(G):
                mask = torch.ones(G, dtype=torch.bool, device=rewards.device)
                mask[i] = False
                s_minus = _pkpo_s_sorted(sorted_rewards[mask], min(k_opt, G - 1))
                b[i] = s_minus.sum() / (G - 1)
            s_sorted = s_sorted - b
        elif variant == "loo_minus_one":
            b = _pkpo_b_sorted(sorted_rewards, k_opt - 1)
            s_sorted = s_sorted - b * k_opt / (k_opt - 1) / G

    s_sorted = s_sorted * G

    advantages = torch.zeros_like(rewards)
    advantages[sort_idx] = s_sorted
    return advantages


# ---------------------------------------------------------------------------
# Reward transform (an ablation, applied to the group's rewards BEFORE an
# estimator sees them -- not an advantage itself)
# ---------------------------------------------------------------------------

def empirical_cdf_transform(rewards: torch.Tensor) -> torch.Tensor:
    """Within-group empirical-CDF (rank) transform.

        F(r_i) = (1/G) * #{ j : r_j <= r_i }   in (0, 1]

    A monotone map into ``(0, 1]`` that discards reward magnitudes and keeps only
    within-group rank. Ties share one value (max-rank convention), the group maximum
    always maps to ``1.0``, and an all-equal group maps to all-ones, which any
    mean-centred estimator then reduces to zero advantage.

    Under this transform every order-statistic gap becomes a uniform ``1/G``, so a
    lone heavy-tail rollout can no longer dominate the group. It exists to ablate
    exactly that: how much of TailRL's behaviour is the reward magnitudes versus the
    ranking. Off by default; enable with ``PIE_REWARD_CDF_TRANSFORM=1``, which
    ``verl_register`` reads once per step.
    """
    G = rewards.numel()
    if G <= 1:
        # Degenerate group: F(r_1) = 1. Downstream estimators return zeros anyway.
        return torch.ones_like(rewards)
    le = rewards.unsqueeze(0) <= rewards.unsqueeze(1)      # [G, G]
    return le.sum(dim=1).to(rewards.dtype) / G


__all__ = [
    "tailrl_advantage",
    "grpo_advantage",
    "rloo_advantage",
    "reinforce_baseline_advantage",
    "binary_maxrl_advantage",
    "pkpo_advantage",
    "empirical_cdf_transform",
]
