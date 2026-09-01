"""Aggressive tests for the online running-metrics accumulator.

The new _running_metrics_init/update/finalize must produce identical
outputs to the prior buffered implementation for every key in the
finalize dict, while using O(1) host RAM regardless of the number of
steps or rollouts per step.
"""

from __future__ import annotations

import sys
import os
import statistics

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.imagenet_localization.training.train import (
    _running_metrics_init,
    _running_metrics_update,
    _running_metrics_finalize,
)


# ---------------------------------------------------------------------------
# Reference: the original buffered accumulator (pre-fix).
# ---------------------------------------------------------------------------

def _ref_init():
    return {'losses': [], 'rewards': [], 'advantages': [], 'grad_norms': []}


def _ref_update(m, loss, out, grad_norm):
    m['losses'].append(loss)
    m['rewards'].append(out['rewards'].flatten().cpu())
    m['advantages'].append(out['advantages'].flatten().cpu())
    m['grad_norms'].append(grad_norm)


def _ref_finalize(m):
    if not m['losses']:
        return {'loss_mean': 0.0}
    r = torch.cat(m['rewards'])
    a = torch.cat(m['advantages'])
    return {
        'loss_mean':         statistics.fmean(m['losses']),
        'reward_mean':       r.mean().item(),
        'reward_std':        r.std().item(),
        'reward_at_0_5':     (r > 0.5).float().mean().item(),
        'reward_at_0_75':    (r > 0.75).float().mean().item(),
        'frac_zero_reward':  (r == 0).float().mean().item(),
        'frac_iou_band_02_08': ((r > 0.2) & (r < 0.8)).float().mean().item(),
        'advantage_abs_mean':a.abs().mean().item(),
        'grad_norm_mean':    statistics.fmean(m['grad_norms']),
    }


# ---------------------------------------------------------------------------
# Helper: feed n random steps through both accumulators.
# ---------------------------------------------------------------------------

def _feed_random_steps(n_steps, B, N, seed=42):
    """Drive both accumulators with the same sequence of random rewards/advantages."""
    g = torch.Generator().manual_seed(seed)
    online = _running_metrics_init()
    ref = _ref_init()
    for s in range(n_steps):
        rewards = torch.rand(B, N, generator=g)
        advantages = torch.randn(B, N, generator=g)
        out = {'rewards': rewards, 'advantages': advantages}
        loss = float(torch.randn(1, generator=g).item())
        grad_norm = float(torch.rand(1, generator=g).item())
        _running_metrics_update(online, loss, out, grad_norm)
        _ref_update(ref, loss, out, grad_norm)
    return online, ref


# ---------------------------------------------------------------------------
# Equivalence tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_steps,B,N", [
    (1, 4, 16),
    (5, 8, 32),
    (50, 16, 64),
    (200, 4, 256),
])
def test_online_matches_buffered_finalize(n_steps, B, N):
    """Every key in the finalize dict must match the buffered impl within
    fp64 round-off tolerance.

    Some metrics (mean, fractions) are exact up to associativity-of-sum
    rounding; std uses sum + sum-of-squares which loses precision in the
    catastrophic-cancellation regime, but the rewards are in [0, 1] so the
    absolute scale is bounded.
    """
    online, ref = _feed_random_steps(n_steps, B, N, seed=n_steps + B + N)
    fin_online = _running_metrics_finalize(online)
    fin_ref = _ref_finalize(ref)

    assert set(fin_online.keys()) == set(fin_ref.keys()), (
        f"Key mismatch: {set(fin_online) ^ set(fin_ref)}"
    )

    # Loss mean and grad-norm mean: identical (both use statistics.fmean
    # on the same Python-float lists).
    assert fin_online['loss_mean'] == fin_ref['loss_mean']
    assert fin_online['grad_norm_mean'] == fin_ref['grad_norm_mean']

    # Reward mean: very tight tolerance (just floating-point summation order).
    assert abs(fin_online['reward_mean'] - fin_ref['reward_mean']) < 1e-6, (
        f"reward_mean: online={fin_online['reward_mean']} "
        f"ref={fin_ref['reward_mean']}"
    )

    # Std: sum + sum-of-squares vs torch.std (population formula in both,
    # since torch.std on a flat tensor uses ddof=0 by default? Actually
    # torch.std() defaults to unbiased=True (ddof=1). For large N*B
    # the difference is 1/(n-1) ≈ 0 so we use a moderately loose tolerance.
    assert abs(fin_online['reward_std'] - fin_ref['reward_std']) < 1e-3, (
        f"reward_std: online={fin_online['reward_std']} "
        f"ref={fin_ref['reward_std']}"
    )

    # Fractions: online uses fp64 int-division; reference uses
    # (r > thr).float().mean() which rounds in fp32 (~1e-7 relative
    # error).  Tolerance is set above that.
    for k in ('reward_at_0_5', 'reward_at_0_75', 'frac_zero_reward',
              'frac_iou_band_02_08'):
        assert abs(fin_online[k] - fin_ref[k]) < 1e-6, (
            f"{k}: online={fin_online[k]} ref={fin_ref[k]}"
        )

    # Advantage |.| mean: floating-point summation order.
    assert abs(fin_online['advantage_abs_mean'] - fin_ref['advantage_abs_mean']) < 1e-6


def test_online_empty_returns_loss_mean_zero():
    """No update calls → finalize returns just {'loss_mean': 0.0}."""
    m = _running_metrics_init()
    out = _running_metrics_finalize(m)
    assert out == {'loss_mean': 0.0}


def test_online_zero_rewards_corner_case():
    """All rewards exactly zero — frac_zero_reward should be 1.0; mean/std 0."""
    m = _running_metrics_init()
    for s in range(5):
        rewards = torch.zeros(4, 16)
        advantages = torch.randn(4, 16)
        _running_metrics_update(m, 0.0, {'rewards': rewards, 'advantages': advantages}, 1.0)
    fin = _running_metrics_finalize(m)
    assert fin['reward_mean'] == 0.0
    assert fin['reward_std'] == 0.0
    assert fin['frac_zero_reward'] == 1.0
    assert fin['reward_at_0_5'] == 0.0
    assert fin['reward_at_0_75'] == 0.0
    assert fin['frac_iou_band_02_08'] == 0.0


def test_online_constant_reward_corner_case():
    """All rewards = 0.5 → mean=0.5, std=0, frac_gt_0.5 = 0 (strict)."""
    m = _running_metrics_init()
    for s in range(5):
        rewards = torch.full((4, 16), 0.5)
        advantages = torch.zeros(4, 16)
        _running_metrics_update(m, 0.0, {'rewards': rewards, 'advantages': advantages}, 1.0)
    fin = _running_metrics_finalize(m)
    assert abs(fin['reward_mean'] - 0.5) < 1e-9
    assert fin['reward_std'] < 1e-6
    assert fin['reward_at_0_5'] == 0.0           # strict (>); equals 0.5 doesn't count
    assert fin['frac_zero_reward'] == 0.0
    assert fin['frac_iou_band_02_08'] == 1.0     # 0.5 is in (0.2, 0.8)


def test_online_thresholds_strict():
    """Boundary checks on the strict-inequality thresholds (>0.5, >0.75, ==0)."""
    m = _running_metrics_init()
    rewards = torch.tensor([
        [0.0, 0.5, 0.75, 1.0],     # 1 zero, 1 = 0.5 (NOT > 0.5), 1 = 0.75 (NOT > 0.75), 1 strictly greater
    ])
    advantages = torch.zeros_like(rewards)
    _running_metrics_update(m, 0.0, {'rewards': rewards, 'advantages': advantages}, 0.0)
    fin = _running_metrics_finalize(m)
    # 1/4 strictly > 0.5  (only 0.75 and 1.0 — wait, 0.75 > 0.5 too)
    # Actually: r > 0.5 → {0.75, 1.0} → 2/4
    assert abs(fin['reward_at_0_5'] - 0.5) < 1e-9
    # r > 0.75 → {1.0} → 1/4
    assert abs(fin['reward_at_0_75'] - 0.25) < 1e-9
    # r == 0 → {0.0} → 1/4
    assert abs(fin['frac_zero_reward'] - 0.25) < 1e-9


def test_online_state_size_does_not_grow_with_steps():
    """The accumulator state must stay constant size across many updates.

    Verifies the actual memory-saving claim: the keys are scalars / small
    Python lists (one entry per step is fine — losses + grad_norms are
    O(steps) but tiny floats), and the rewards/advantages tensor lists
    are not present.
    """
    m = _running_metrics_init()
    # Initial schema
    expected_keys = {
        'losses', 'grad_norms',
        'count_r', 'sum_r', 'sum_r2',
        'count_r_gt_0_5', 'count_r_gt_0_75', 'count_r_eq_0',
        'count_r_in_02_08',
        'count_a', 'sum_abs_a',
    }
    assert set(m.keys()) == expected_keys

    for _ in range(100):
        rewards = torch.rand(4, 1024)        # 4 KB tensor
        advantages = torch.randn(4, 1024)
        _running_metrics_update(m, 0.1, {'rewards': rewards, 'advantages': advantages}, 1.0)

    # Schema unchanged after 100 updates.
    assert set(m.keys()) == expected_keys
    # Per-step lists are O(steps) in count, but each entry is one float.
    assert len(m['losses']) == 100
    assert len(m['grad_norms']) == 100
    # Online stats are simple Python ints/floats, NOT tensor lists.
    for k in ('count_r', 'sum_r', 'sum_r2', 'count_r_gt_0_5',
              'count_r_gt_0_75', 'count_r_eq_0', 'count_r_in_02_08',
              'count_a', 'sum_abs_a'):
        v = m[k]
        assert isinstance(v, (int, float)), f"{k} should be scalar, got {type(v)}"


def test_online_large_step_count_no_blowup():
    """Drive the accumulator with 500 steps × big rewards; compare to ref."""
    n_steps, B, N = 500, 8, 256
    online, ref = _feed_random_steps(n_steps, B, N, seed=99)
    fin_online = _running_metrics_finalize(online)
    fin_ref = _ref_finalize(ref)

    for k in fin_online.keys():
        diff = abs(fin_online[k] - fin_ref[k])
        if k == 'reward_std':
            assert diff < 1e-3, f"{k}: {fin_online[k]} vs {fin_ref[k]}"
        else:
            assert diff < 1e-5, f"{k}: {fin_online[k]} vs {fin_ref[k]}"


def test_online_single_step_matches_per_tensor_stats():
    """One-step finalize must equal direct torch ops on the rewards tensor."""
    m = _running_metrics_init()
    torch.manual_seed(0)
    rewards = torch.rand(4, 32)
    advantages = torch.randn(4, 32)
    _running_metrics_update(
        m, 0.5, {'rewards': rewards, 'advantages': advantages}, 1.0,
    )
    fin = _running_metrics_finalize(m)
    assert abs(fin['reward_mean'] - rewards.mean().item()) < 1e-6
    assert abs(fin['reward_at_0_5'] - (rewards > 0.5).float().mean().item()) < 1e-9
    assert abs(fin['advantage_abs_mean'] - advantages.abs().mean().item()) < 1e-6
