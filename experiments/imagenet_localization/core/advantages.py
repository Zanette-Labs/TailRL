
"""Advantage estimators for ImageNet localization RL experiment.

All functions are **polymorphic over the last tensor dimension**:

    rewards.shape == (N,)      -> advantages.shape == (N,)
    rewards.shape == (B, N)    -> advantages.shape == (B, N)
    rewards.shape == (*batch, N) -> advantages.shape == (*batch, N)

The per-image loop over the batch dim (Python for b in range(B)) is
therefore unnecessary — just call `advantage_fn(rewards)` once with the
full (B, N) tensor. This matters at large N (e.g. N=16384): the dispatch
overhead of 128 sequential torch calls per step becomes the bottleneck.

Methods:
- tailrl: TailRL — this paper's estimator (gap-over-survivors form)
- binary_maxrl: Tajwar et al. binarized at r = 1.0 (exact equality)
- bmaxrl_adv_est: continuous extension (r - mu) / (mu + eps)
- grpo: z-score normalized expected reward baseline
- rloo: Leave-One-Out baseline
- reinforce: mean-centered baseline (no std normalization)
- pkpo: Pass@K Policy Optimization (Walder & Karkhanis 2025, Eq 33)
"""

import os

import torch


TAILRL_SURVIVAL_EPS: float = 1e-4
"""Survival-floor ε for the TailRL increment mask.

At each quality-level interval j in the TailRL sum, the empirical survival
fraction is (N − j) / N.  Intervals where this fraction < ε are masked
(their increment is zeroed), mirroring tailrl_population_loss's
clamp(min=ε) on P(IoU > τ): thresholds where P(IoU > τ) ≤ ε contribute
zero gradient in the supervised loss, so the RL estimator must skip them
too.  In the N → ∞ limit the mask converges to 𝟙(S(τ) ≥ ε), and the
TailRL gradient converges to ∇ L_ε exactly.

Practical effect: at ε = 1e-4, the mask only binds when N > 1/ε = 10000;
at N = 16384 it zeros a single increment (survivors = 1), at N = 131072
about 13 increments.  For N ≤ 10000 the mask is all-True (no-op).
"""


def tailrl_advantage(
    rewards: torch.Tensor,
    survival_eps: float = TAILRL_SURVIVAL_EPS,
) -> torch.Tensor:
    """TailRL advantage for continuous rewards.

    Operates on the last dim: (N,) → (N,) or (*batch, N) → (*batch, N).

    Algorithm (per-row, last dim):
      1. Sort rewards ascending.
      2. Gaps: gap_i = r_(i) - r_(i-1), with r_(0) = 0.
      3. Increments: gap_i / survivors_i, where survivors_i = N − i + 1.
         Increments at positions where the empirical survival fraction
         survivors_i / N < survival_eps are zeroed (masked out).
         This is the exact finite-N analog of tailrl_population_loss's
         clamp(min=eps) on P(IoU > τ): both omit quality levels where
         the survival probability is below ε.
      4. Cumulative sum, scale by N, mean-center.
      5. Scatter back to original order.

    Args:
        rewards: tensor shaped (*batch, N) of reward values in [0, 1].
        survival_eps: floor on the empirical survival fraction.  Intervals
            where (N − j) / N < survival_eps are masked out.  Must match
            the ``eps`` used in ``localization_tailrl_population_loss`` for
            the N → ∞ identity to hold.  Default 1e-4.

    Returns:
        Mean-centered advantages, same shape / dtype / device as input.
        Returns zeros when the last dim is empty or of size ≤ 1.
    """
    if rewards.numel() == 0:
        return torch.zeros_like(rewards)
    N = rewards.shape[-1]
    if N <= 1:
        return torch.zeros_like(rewards)

    sorted_rewards, sort_indices = torch.sort(rewards, dim=-1, stable=True)

    # Shift sorted_rewards right by 1 along last dim, padding with 0.
    batch_shape = sorted_rewards.shape[:-1]
    zero_col = torch.zeros(
        *batch_shape, 1, device=rewards.device, dtype=rewards.dtype,
    )
    prev = torch.cat([zero_col, sorted_rewards[..., :-1]], dim=-1)
    gaps = sorted_rewards - prev

    # Survivors: [N, N-1, ..., 1].  No clamp on the counts themselves.
    survivors = torch.arange(
        N, 0, -1, device=rewards.device, dtype=rewards.dtype,
    )
    increments = gaps / survivors

    # Mask out quality-level intervals where the empirical survival
    # fraction drops below ε, matching tailrl_population_loss's clamp.
    # survivors is shape (N,); broadcasts against (*batch, N).
    if survival_eps > 0:
        mask = (survivors / N >= survival_eps)          # (N,) bool
        increments = increments * mask.to(increments.dtype)

    weights = torch.cumsum(increments, dim=-1) * N
    weights = weights - weights.mean(dim=-1, keepdim=True)

    advantages = torch.zeros_like(rewards)
    advantages.scatter_(dim=-1, index=sort_indices, src=weights)
    return advantages


def binary_maxrl_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """Binary MaxRL advantage (Tajwar et al.), binarized at r == 1.0 (exact).

    Operates on the last dim. Success = exact match (r == 1.0); failure = anything
    else. For rows where all samples succeed or all fail, returns zeros.

    Scaled so that (1/N)·Σ adv_i · score_i has O(1) magnitude:
      success advantage = +(N - K) / K
      failure advantage = -1

    Args:
        rewards: tensor shaped (*batch, N).

    Returns:
        Advantages, same shape / dtype / device as input.
    """
    if rewards.numel() == 0:
        return torch.zeros_like(rewards)
    N = rewards.shape[-1]
    if N <= 1:
        return torch.zeros_like(rewards)

    successes = (rewards == 1.0)                                   # (*, N) bool
    K = successes.sum(dim=-1, keepdim=True).to(rewards.dtype)      # (*, 1)

    safe_K = K.clamp(min=1)                                         # avoid div-by-0
    adv_success = (N - K) / safe_K                                 # (*, 1)

    advantages = torch.where(
        successes,
        adv_success.expand_as(rewards),
        torch.full_like(rewards, -1.0),
    )
    degenerate = (K == 0) | (K == N)                               # (*, 1) bool
    return torch.where(
        degenerate.expand_as(rewards),
        torch.zeros_like(rewards),
        advantages,
    )


def bmaxrl_adv_est_advantage(
    rewards: torch.Tensor, eps: float = 1e-8,
) -> torch.Tensor:
    """B-MaxRL continuous advantage: ``(r - mu) / (mu + eps)``.

    Continuous extension of ``binary_maxrl``. For binary rewards r ∈ {0, 1}
    with success rate p = K/N, this reduces exactly to ``binary_maxrl``:
      success (r=1): (1 - p)/p = (N - K)/K
      failure (r=0): -p/p     = -1

    For continuous IoU rewards in [0, 1] the same structure is kept without
    binarizing. Rows with mean(r) < ``eps`` return zeros (degenerate, mirrors
    ``binary_maxrl``'s K=0 case).

    Operates on the last dim: (N,) → (N,) or (*batch, N) → (*batch, N).

    Args:
        rewards: tensor shaped (*batch, N).
        eps: numerical stability constant added to the denominator and used
            as the degeneracy threshold on mean(r).
    """
    if rewards.numel() == 0:
        return torch.zeros_like(rewards)
    N = rewards.shape[-1]
    if N <= 1:
        return torch.zeros_like(rewards)

    mu = rewards.mean(dim=-1, keepdim=True)
    advantages = (rewards - mu) / (mu + eps)
    degenerate = (mu < eps).expand_as(rewards)
    return torch.where(degenerate, torch.zeros_like(rewards), advantages)


def grpo_advantage(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """GRPO advantage: z-score normalized rewards.

    Operates on the last dim. Rows with near-zero std (< 1e-4) get zero
    advantages to avoid division instability.

    Args:
        rewards: tensor shaped (*batch, N).
        eps: numerical stability constant added to std denominator.
    """
    if rewards.numel() == 0:
        return torch.zeros_like(rewards)
    N = rewards.shape[-1]
    if N <= 1:
        return torch.zeros_like(rewards)

    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    normalized = (rewards - mean) / (std + eps)
    zero_mask = (std < 1e-4).expand_as(rewards)
    return torch.where(zero_mask, torch.zeros_like(rewards), normalized)


def rloo_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """Leave-One-Out (RLOO) baseline advantage.

    A_i = (N · r_i - Σ r) / (N - 1)

    Operates on the last dim. Returns zeros when the last dim has ≤ 1 element.
    """
    if rewards.numel() == 0:
        return torch.zeros_like(rewards)
    N = rewards.shape[-1]
    if N <= 1:
        return torch.zeros_like(rewards)

    total = rewards.sum(dim=-1, keepdim=True)
    return (N * rewards - total) / (N - 1)


def reinforce_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """REINFORCE with baseline: mean-centered along the last dim.

    A_i = r_i - mean(r)
    """
    if rewards.numel() == 0:
        return torch.zeros_like(rewards)
    N = rewards.shape[-1]
    if N <= 1:
        return torch.zeros_like(rewards)

    return rewards - rewards.mean(dim=-1, keepdim=True)




# ---------------------------------------------------------------------------
# Pass@K Policy Optimization (PKPO) — Walder & Karkhanis 2025 (arXiv:2505.15201)
# Implements s_i^(loo-1) (Eq 33), the recommended estimator from the paper.
# Reference: Listing 1 in the paper. Verified bit-exact (1e-16) vs the numpy
# reference for N up to 256 and K up to 16.
# ---------------------------------------------------------------------------

PKPO_K_DEFAULT: int = int(os.environ.get("PKPO_K", "8"))
"""Default pass@k threshold for ``pkpo_advantage``.

Set via the ``PKPO_K`` env var so different runs in a sweep can target
different k values without code changes. The valid range is 2..N. At
k=1, the paper says no transformation (== mean-centered REINFORCE);
``pkpo_advantage`` falls back to mean-centering for k=1 only.
"""


# Cache of (m_diag_outer, deltas_outer, m_diag_inner, deltas_inner) tensors,
# keyed by (N, K). All four are invariant given (N, K) and broadcast against
# (*batch, N) sorted-rewards tensors.
_PKPO_WEIGHT_CACHE: dict = {}


def _pkpo_m_diagonal_np(N: int, K: int) -> torch.Tensor:
    """``_m_diagonal(N, K)`` from the paper's Listing 1.

    m_ii^(normed) = K/(N-K+1) * prod_{l=1..K-1} (i-K+1+l)/(N-K+1+l) for i ≥ K-1,
    else 0.  Indices are zero-based (i ∈ {0, .., N-1}).
    """
    out = torch.zeros(N, dtype=torch.float64)
    if K < 1 or N < 1:
        return out
    for i in range(N):
        if i < K - 1:
            continue
        if K == 1:
            out[i] = 1.0 / N
            continue
        num = torch.arange(i - K + 2, i + 1, dtype=torch.float64)
        den = torch.arange(N - K + 2, N + 1, dtype=torch.float64)
        out[i] = K / (N - K + 1) * (num / den).prod()
    return out


def _pkpo_deltas_np(N: int, K: int) -> torch.Tensor:
    """``_deltas(N, K)`` from the paper's Listing 1, length N-2.

    delta_i = m_normed(N-1, K, i, i+1) - m_normed(N-1, K, i+1, i+1).
    """
    if N <= 2:
        return torch.zeros(0, dtype=torch.float64)
    out = torch.zeros(N - 2, dtype=torch.float64)
    M = N - 1
    for i in range(N - 2):
        # off-diagonal (i, i+1) at size-M arr
        j = i + 1
        if j > i and j >= K - 1 and K >= 2:
            if K == 2:
                v_off = float(K / (M - K + 1) * (K - 1) / M)
            else:
                num = torch.arange(j - K + 2, j, dtype=torch.float64)
                den = torch.arange(M - K + 2, M, dtype=torch.float64)
                v_off = float(
                    K / (M - K + 1) * (K - 1) / M * (num / den).prod()
                )
        else:
            v_off = 0.0
        # diagonal at i+1 in size-M arr
        if j >= K - 1 and K >= 1:
            if K == 1:
                ratio = 1.0
            else:
                num = torch.arange(j - K + 2, j + 1, dtype=torch.float64)
                den = torch.arange(M - K + 2, M + 1, dtype=torch.float64)
                ratio = float((num / den).prod())
            v_diag = K / (M - K + 1) * ratio
        else:
            v_diag = 0.0
        out[i] = v_off - v_diag
    return out


def _pkpo_get_weights(N: int, K: int, device, dtype):
    """Return (m_diag_outer, deltas_outer, m_diag_inner, deltas_inner) for sloo_minus_one(g, K).

    Computed once per (N, K) in float64 and cached. Cast to the requested
    dtype/device at retrieval (cast cost is negligible vs construction).
    """
    cache_key = (N, K)
    if cache_key not in _PKPO_WEIGHT_CACHE:
        # s(g, K) needs m_diagonal(N, K) and deltas(N+1, K).
        m_diag_outer = _pkpo_m_diagonal_np(N, K)
        deltas_outer = _pkpo_deltas_np(N + 1, K)
        # _b(g, K-1) needs m_diagonal(N-1, K-1) and deltas(N, K-1).
        m_diag_inner = _pkpo_m_diagonal_np(N - 1, K - 1)
        deltas_inner = _pkpo_deltas_np(N, K - 1)
        _PKPO_WEIGHT_CACHE[cache_key] = (
            m_diag_outer, deltas_outer, m_diag_inner, deltas_inner,
        )
    m_diag_outer, deltas_outer, m_diag_inner, deltas_inner = _PKPO_WEIGHT_CACHE[cache_key]
    return (
        m_diag_outer.to(device=device, dtype=dtype),
        deltas_outer.to(device=device, dtype=dtype),
        m_diag_inner.to(device=device, dtype=dtype),
        deltas_inner.to(device=device, dtype=dtype),
    )


def pkpo_advantage(
    rewards: torch.Tensor, k: int | None = None, *, paper_scale: bool = False,
) -> torch.Tensor:
    """Pass@K Policy Optimization (PKPO) advantage — s_i^(loo-1) of Eq 33.

    Operates on the last dim. Implements the leave-one-out-minus-one baseline
    from Walder & Karkhanis 2025, the paper's recommended pkpo estimator. The
    weight on each sample, as a function of its sorted rank, interpolates
    between uniform (k=1, == REINFORCE) and "max only" (k=N, == TZSM25).

    Algorithm (per-row, last dim):
      1. Sort rewards ascending.
      2. s_sorted = s(sorted_g, K)        — Eq 19 (cumsum-from-right).
      3. b_sorted = _b(sorted_g, K-1)     — auxiliary cumsum, Eq 32 recursion.
      4. sloo_m1_sorted = s_sorted - b_sorted * K / ((K-1) * N)   — Eq 33.
      5. Scatter back to the original order.

    Internally computes in float64 for the binomial-ratio precision (the
    weights span ~1e-13 .. 1e0 at N=256, K=16, which is fine for float64
    cumsums but tight for float32). Result is cast back to ``rewards.dtype``.

    CONVENTION (important): Eq 33 is a *sum*-convention estimator -- the true
    gradient is ``sum_i s_i * grad log pi(g_i)``, NOT the mean. Every other
    estimator here is mean-convention, because ``train.py`` forms the loss as
    ``-(adv * logp).mean(-1)``, i.e. it divides by N. Returning Eq 33 unscaled
    therefore injects a spurious 1/N and the arm does not train (at N=1024 the
    gradient is 1024x too small; measured grad_norm ~2.5e-4 vs ~3 for tailrl).

    So by default this returns ``N * s_i^(loo-1)``, putting PKPO on the same
    footing as tailrl/grpo/rloo. Verified against the exact gradient of
    ``E[max@k] = sum_j v_j (F_j^k - F_{j-1}^k)`` on a non-saturating categorical
    toy: cosine 0.996-0.999 and magnitude ratio 1.00 +- 0.01 for
    (k, N) = (16, 1024), (64, 1024), (256, 1024).

    Pass ``paper_scale=True`` to get the unscaled Eq-33 values (what the paper's
    Listing 1 prints); the reference tests use that.

    k=1 is special-cased to mean-centered rewards (==REINFORCE), which is
    ALREADY mean-convention, so it is never rescaled. Before this fix k=1 and
    k>=2 silently used different conventions (a ~200x discontinuity in scale).

    Args:
        rewards: tensor shaped (*batch, N) of reward values (any range OK).
        k: pass@k threshold; valid range 1..N.  Defaults to ``PKPO_K_DEFAULT``
            (env-var-controlled).  At k=1 returns mean-centered rewards
            (==REINFORCE) since the loo-1 baseline divides by k-1=0.
        paper_scale: return raw Eq-33 (sum-convention) values instead of the
            policy-gradient-scaled ones. For reference checks only.

    Returns:
        Advantages, same shape / dtype / device as input.
    """
    if rewards.numel() == 0:
        return torch.zeros_like(rewards)
    N = rewards.shape[-1]
    if N <= 1:
        return torch.zeros_like(rewards)

    if k is None:
        # Falling back to the global env default: clamp rather than raise, since
        # the caller never chose this k. Without this, any group smaller than
        # PKPO_K_DEFAULT (e.g. the N=4 unit tests vs the default k=8) blows up.
        k = min(PKPO_K_DEFAULT, N)
    elif k < 1 or k > N:
        # An explicitly requested k that cannot be satisfied is a real error.
        raise ValueError(f"pkpo_advantage: k={k} out of range [1, N={N}]")
    k = max(1, min(k, N))
    if k == 1:
        # Eq 33 has K-1 in the denominator → undefined.  Paper notes that
        # k=1 corresponds to "no reward transformation"; the unbiased
        # mean-centered REINFORCE baseline is the natural choice.
        return rewards - rewards.mean(dim=-1, keepdim=True)

    # Compute in float64 for the binomial-ratio precision.
    sorted_g64, sort_idx = torch.sort(rewards.to(torch.float64), dim=-1, stable=True)

    m_diag_outer, deltas_outer, m_diag_inner, deltas_inner = _pkpo_get_weights(
        N, k, device=sorted_g64.device, dtype=torch.float64,
    )

    # --- s(sorted_g, k) — Eq 19 ---
    c = sorted_g64 * m_diag_outer                          # (..., N)
    update = sorted_g64[..., 1:] * deltas_outer            # (..., N-1)
    zero_pad = torch.zeros(
        *sorted_g64.shape[:-1], 1, dtype=torch.float64, device=sorted_g64.device,
    )
    c = c + torch.cat([update, zero_pad], dim=-1)          # add to first N-1 entries
    s_sorted = torch.flip(torch.cumsum(torch.flip(c, dims=[-1]), dim=-1), dims=[-1])

    # --- _b(sorted_g, k-1) — Eq 32 recursion in cumsum form ---
    arange_1_N = torch.arange(
        1, N, dtype=torch.float64, device=sorted_g64.device,
    )                                                       # (N-1,)
    arange_1_Nm1 = torch.arange(
        1, N - 1, dtype=torch.float64, device=sorted_g64.device,
    )                                                       # (N-2,)
    w = m_diag_inner * arange_1_N                           # (N-1,)
    w = w + torch.cat(
        [torch.zeros(1, dtype=torch.float64, device=w.device), deltas_inner * arange_1_Nm1],
        dim=0,
    )                                                       # (N-1,)
    c1 = (w * sorted_g64[..., 1:]).sum(dim=-1, keepdim=True)         # (..., 1)
    c2 = (sorted_g64[..., :-1] - sorted_g64[..., 1:]) * w            # (..., N-1)
    b_sorted = torch.cumsum(torch.cat([c1, c2], dim=-1), dim=-1)     # (..., N)

    # --- sloo_minus_one (Eq 33): s - _b * k / ((k-1) * N) ---
    sloo_m1_sorted = s_sorted - b_sorted * (k / ((k - 1) * N))

    # Eq 33 is sum-convention; train.py's loss takes .mean(-1) over N. Rescale
    # to mean-convention so the estimator matches the true gradient magnitude.
    if not paper_scale:
        sloo_m1_sorted = sloo_m1_sorted * float(N)

    # Scatter back to original order, restore input dtype.
    advantages_sorted = sloo_m1_sorted.to(rewards.dtype)
    advantages = torch.zeros_like(rewards)
    advantages.scatter_(dim=-1, index=sort_idx, src=advantages_sorted)
    return advantages


# ---------------------------------------------------------------------------
# Reward shaping — applied to the raw IoU rewards *before* the advantage
# estimator.  Orthogonal to the advantage estimator (tailrl/grpo/rloo/...):
# select via the --reward_transform flag, dispatched through REWARD_TRANSFORMS.
# ---------------------------------------------------------------------------

def identity_transform(rewards: torch.Tensor) -> torch.Tensor:
    """No-op reward shaping (the default).  Returns rewards unchanged."""
    return rewards


def percentile_transform(rewards: torch.Tensor) -> torch.Tensor:
    """Average-percentile-rank reward shaping, per-row over the last dim.

    Maps each rollout's reward to its percentile rank in [0, 1] within the
    group of N rollouts:

      - the largest reward -> 1.0, the smallest -> 0.0;
      - ties: all rollouts sharing a value receive the *median* (average) of
        the percentiles they would otherwise span.

    Concretely, percentile_i = avg_rank_i / (N - 1), where avg_rank_i is the
    average 0-indexed sorted rank of value r_i.  With L_i = #{j : r_j < r_i}
    and U_i = #{j : r_j <= r_i}, the sorted positions of r_i are L_i .. U_i-1,
    whose midpoint is (L_i + U_i - 1) / 2.  This is the standard "average"
    tie-break (scipy rankdata method='average'), rescaled to [0, 1].

    Computed with two batched binary searches (O(N log N), no Python loop and
    no scatter — the result is already in the original rollout order).

    N <= 1 has no spread, so we return zeros (and never divide by N - 1 = 0).
    """
    N = rewards.shape[-1]
    if rewards.numel() == 0 or N <= 1:
        return torch.zeros_like(rewards)

    sorted_vals, _ = torch.sort(rewards, dim=-1)
    # Counts in the original rollout order: strictly-less and less-or-equal.
    lower = torch.searchsorted(sorted_vals, rewards, side="left").to(rewards.dtype)
    upper = torch.searchsorted(sorted_vals, rewards, side="right").to(rewards.dtype)
    avg_rank = (lower + upper - 1.0) / 2.0
    return avg_rank / (N - 1)


def _binary_threshold_transform(
    rewards: torch.Tensor, tau: float,
) -> torch.Tensor:
    """Binary success indicator: 1.0 where ``reward > tau``, else 0.0.

    Strictly greater, matching the ``reward_at_0_5`` / ``reward_at_0_75``
    metrics in ``train.py`` so the shaped reward agrees with the fraction those
    metrics already report. Returns the input dtype/device, so the result is a
    genuine {0.0, 1.0} float tensor rather than a bool mask.
    """
    return (rewards > tau).to(rewards.dtype)


def binary_iou_50_transform(rewards: torch.Tensor) -> torch.Tensor:
    """Binary reward at IoU > 0.5 (the CorLoc@0.5 operating point).

    Turns the continuous max-over-GT IoU into the success indicator that
    CorLoc@0.5 -- the headline localization metric -- actually scores. Paired
    with the ``bmaxrl_adv_est`` estimator ``(r - mu) / (mu + eps)`` this yields
    exactly the binary-MaxRL advantage: with success rate p = K/N,

        success (r=1): (1 - p) / p = (N - K) / K
        failure (r=0): -p / p      = -1

    i.e. it reduces analytically to ``binary_maxrl_advantage``, which binarizes
    at ``r == 1.0`` -- after this transform the rewards *are* exactly 0.0/1.0,
    so the two estimators coincide (there is a unit test pinning that).

    Degeneracy note: when no rollout clears the threshold, mu = 0 and
    ``bmaxrl_adv_est`` returns zeros, so that image contributes no gradient.
    The probability of an all-failure group falls as (1 - p)^N, which is
    precisely why the rollout count N is the interesting axis for this arm.
    """
    return _binary_threshold_transform(rewards, 0.5)


def binary_iou_75_transform(rewards: torch.Tensor) -> torch.Tensor:
    """Binary reward at IoU > 0.75 (the stricter CorLoc@0.75 operating point).

    Same construction as `binary_iou_50_transform` at a harder threshold, so
    the success rate p is lower and all-failure groups are correspondingly more
    likely at small N. See that function for the estimator algebra and the
    degeneracy note.
    """
    return _binary_threshold_transform(rewards, 0.75)


# ---------------------------------------------------------------------------
# Dispatch dict for convenience.
# Keys in the canonical order for this experiment.
# ---------------------------------------------------------------------------

ADVANTAGE_FNS = {
    "tailrl": tailrl_advantage,
    "binary_maxrl": binary_maxrl_advantage,
    "bmaxrl_adv_est": bmaxrl_adv_est_advantage,
    "grpo": grpo_advantage,
    "rloo": rloo_advantage,
    "reinforce": reinforce_advantage,
    "pkpo": pkpo_advantage,
}


# ---------------------------------------------------------------------------
# Reward-transform dispatch — orthogonal to ADVANTAGE_FNS.  The selected
# transform is applied to the raw IoU rewards before the advantage estimator.
# ---------------------------------------------------------------------------

REWARD_TRANSFORMS = {
    "none": identity_transform,
    "percentile": percentile_transform,
    "binary_0.5": binary_iou_50_transform,
    "binary_0.75": binary_iou_75_transform,
}
