"""Unbiased best@k (best-of-k) reward estimator for the localization experiment.

Given M iid rollout rewards per image, the expected maximum reward over a
*random* k-subset of those M samples — i.e. the best-of-k reward — has the
unbiased order-statistic (U-statistic) estimator

    best@k = sum_{i=k}^{M} w_i^{(k)} * r_(i),     r_(1) <= ... <= r_(M),
    w_i^{(k)} = C(i-1, k-1) / C(M, k).

r_(i) is the max of a chosen k-subset iff its other k-1 members all come from
the i-1 strictly-smaller samples, giving C(i-1, k-1) favourable subsets out of
C(M, k); hence the weights. They satisfy sum_i w_i = 1 (hockey-stick identity),
so best@k is a convex combination of the sorted rewards (in [min, max]).

This is the continuous-reward generalisation of the Chen et al. (2021) pass@k
estimator and shares the binomial-weight structure used by ``pkpo_advantage``
in ``advantages.py``.  Requires M >= k; with M >> k the estimate is low-variance.
"""

from __future__ import annotations

from typing import Iterable

import torch


def best_at_k_weights(
    M: int,
    k: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Order-statistic weights ``w_i`` (i = 1..M) for the unbiased best@k estimator.

    ``w_i = C(i-1, k-1) / C(M, k)`` for i >= k, else 0.  Returns a length-M
    tensor that sums to 1.  Computed in log-gamma space for numerical stability
    at large M/k (e.g. M=4096, k=1024).
    """
    if not (1 <= k <= M):
        raise ValueError(f"best_at_k_weights: need 1 <= k <= M, got k={k}, M={M}")

    i = torch.arange(1, M + 1, device=device, dtype=dtype)          # ranks 1..M
    k_t = torch.tensor(float(k), device=device, dtype=dtype)

    # log C(i-1, k-1) = lgamma(i) - lgamma(k) - lgamma(i-k+1), valid for i >= k.
    # Clamp the (i-k+1) argument to >= 1 so lgamma stays finite for i < k; those
    # ranks are masked to zero below, so the clamped (wrong) value is discarded.
    arg = (i - k + 1).clamp(min=1.0)
    log_num = torch.lgamma(i) - torch.lgamma(k_t) - torch.lgamma(arg)

    M_t = torch.tensor(float(M), device=device, dtype=dtype)
    log_den = (
        torch.lgamma(M_t + 1.0)
        - torch.lgamma(k_t + 1.0)
        - torch.lgamma(M_t - k_t + 1.0)
    )

    w = torch.exp(log_num - log_den)
    w = torch.where(i >= k, w, torch.zeros_like(w))
    return w


def best_at_k_unbiased(rewards: torch.Tensor, ks: Iterable[int]) -> torch.Tensor:
    """Unbiased best@k over the last dim.

    Args:
        rewards: (..., M) tensor of M iid rollout rewards per row.
        ks: iterable of k values (each 1 <= k <= M).

    Returns:
        (..., len(ks)) tensor; entry [..., j] is best@ks[j] for that row.
    """
    M = rewards.shape[-1]
    sorted_r, _ = torch.sort(rewards, dim=-1)            # ascending, (..., M)
    sorted_r = sorted_r.to(torch.float64)

    cols = []
    for k in ks:
        w = best_at_k_weights(M, int(k), device=rewards.device, dtype=torch.float64)
        cols.append((sorted_r * w).sum(dim=-1))          # (...,)
    return torch.stack(cols, dim=-1).to(rewards.dtype)   # (..., len(ks))
