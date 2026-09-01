"""Tests for the eval best@k / maxreward@k estimators and the in-box helper.

Covers spec section 5d:
  - eval_screenspot_pro.bestk_unbiased(n, c, k)  (unbiased pass@k)
  - eval_screenspot_pro.max_reward_at_k(rewards, k)  (continuous generalisation)
  - gui_grounding._in_box(px, py, x1, y1, x2, y2)  (inclusive-boundary in-box flag)

All data is synthetic / in-memory. Imports are CPU-safe (eval_screenspot_pro imports
vLLM/transformers only inside main()).
"""
import math
from itertools import combinations

import numpy as np
import pytest

from eval_screenspot_pro import bestk_unbiased, max_reward_at_k
from gui_grounding import _in_box


# --------------------------------------------------------------------------- #
# bestk_unbiased
# --------------------------------------------------------------------------- #

def test_bestk_unbiased_no_hits_is_zero():
    # c == 0 (and c < 0) -> guaranteed miss -> 0.0
    for n in (1, 3, 10, 50):
        for k in (1, 2, n):
            assert bestk_unbiased(n, 0, k) == 0.0


def test_bestk_unbiased_all_hits_is_one():
    # c == n -> n - c == 0 < k (k >= 1) -> 1.0
    for n in (1, 3, 10, 50):
        for k in (1, 2, n):
            assert bestk_unbiased(n, n, k) == 1.0


def test_bestk_unbiased_few_misses_is_one():
    # n - c < k -> impossible to draw k samples that are all misses -> 1.0
    # e.g. n=5, c=3 (2 misses), k=3 > 2 misses.
    assert bestk_unbiased(5, 3, 3) == 1.0
    assert bestk_unbiased(5, 3, 4) == 1.0
    assert bestk_unbiased(5, 3, 5) == 1.0
    assert bestk_unbiased(10, 8, 3) == 1.0  # 2 misses < 3


def test_bestk_unbiased_k1_equals_c_over_n():
    # k == 1 -> probability the single sample is a hit == c / n (telescoping product).
    for n in range(1, 21):
        for c in range(0, n + 1):
            assert bestk_unbiased(n, c, 1) == pytest.approx(c / n, abs=1e-12)


def test_bestk_unbiased_known_values():
    # bestk_unbiased(4, 1, 2): 1 - (1 - 2/4) = 0.5
    assert bestk_unbiased(4, 1, 2) == pytest.approx(0.5, abs=1e-12)
    # bestk_unbiased(10, 1, 2) == 1 - C(9,2)/C(10,2) == 1 - 36/45 == 0.2
    assert bestk_unbiased(10, 1, 2) == pytest.approx(0.2, abs=1e-12)
    expected = 1.0 - (math.comb(9, 2) / math.comb(10, 2))
    assert bestk_unbiased(10, 1, 2) == pytest.approx(expected, abs=1e-12)


def test_bestk_unbiased_matches_hypergeometric_formula():
    # General closed form: pass@k = 1 - C(n-c, k) / C(n, k) for n - c >= k.
    for n in range(2, 16):
        for c in range(0, n + 1):
            for k in range(1, n + 1):
                got = bestk_unbiased(n, c, k)
                if c == 0:
                    expected = 0.0
                elif n - c < k:
                    expected = 1.0
                else:
                    expected = 1.0 - (math.comb(n - c, k) / math.comb(n, k))
                assert got == pytest.approx(expected, abs=1e-9), (n, c, k)


def test_bestk_unbiased_monotonic_in_k():
    # For fixed n, c the estimator is non-decreasing in k.
    for n in (8, 13, 20):
        for c in range(0, n + 1):
            vals = [bestk_unbiased(n, c, k) for k in range(1, n + 1)]
            for a, b in zip(vals, vals[1:]):
                assert b >= a - 1e-12, (n, c)


def test_bestk_unbiased_monotonic_in_c():
    # For fixed n, k the estimator is non-decreasing in c.
    for n in (8, 13, 20):
        for k in (1, 2, 4, n):
            vals = [bestk_unbiased(n, c, k) for c in range(0, n + 1)]
            for a, b in zip(vals, vals[1:]):
                assert b >= a - 1e-12, (n, k)


def test_bestk_unbiased_in_unit_interval():
    for n in range(1, 21):
        for c in range(0, n + 1):
            for k in range(1, n + 1):
                v = bestk_unbiased(n, c, k)
                assert -1e-12 <= v <= 1.0 + 1e-12, (n, c, k, v)


def test_bestk_unbiased_monte_carlo_agreement():
    # Empirical hit-rate of ">=1 of k samples (drawn without replacement) is a hit"
    # must match bestk_unbiased within 3 standard errors.
    rng = np.random.default_rng(20240617)
    trials = 20000
    for (n, c, k) in [(10, 3, 4), (20, 5, 8), (16, 1, 2), (12, 7, 3), (30, 2, 10)]:
        pop = np.zeros(n, dtype=int)
        pop[:c] = 1  # c hits, rest misses
        hits = 0
        for _ in range(trials):
            draw = rng.choice(n, size=k, replace=False)
            if pop[draw].sum() >= 1:
                hits += 1
        emp = hits / trials
        p = bestk_unbiased(n, c, k)
        se = math.sqrt(max(p * (1.0 - p), 1e-9) / trials)
        assert abs(emp - p) <= 3.0 * se + 1e-6, (n, c, k, emp, p, se)


# --------------------------------------------------------------------------- #
# max_reward_at_k
# --------------------------------------------------------------------------- #

def test_max_reward_at_k_reduces_to_bestk_for_binary():
    # Binary 0/1 rewards: max over a size-k subset == "any hit in the subset",
    # so the unbiased expectation equals bestk_unbiased(n, c, k).
    for n in (6, 10, 15):
        for c in range(0, n + 1):
            rewards = [0.0] * (n - c) + [1.0] * c
            for k in range(1, n + 1):
                assert max_reward_at_k(rewards, k) == pytest.approx(
                    bestk_unbiased(n, c, k), abs=1e-9
                ), (n, c, k)


def test_max_reward_at_k_k1_is_mean():
    rng = np.random.default_rng(7)
    for _ in range(20):
        rewards = rng.random(rng.integers(2, 30))
        assert max_reward_at_k(rewards, 1) == pytest.approx(float(np.mean(rewards)), abs=1e-12)


def test_max_reward_at_k_kN_is_max():
    rng = np.random.default_rng(8)
    for _ in range(20):
        rewards = rng.random(rng.integers(2, 30))
        N = rewards.size
        assert max_reward_at_k(rewards, N) == pytest.approx(float(np.max(rewards)), abs=1e-12)
        # k > N is clamped to N -> still the max.
        assert max_reward_at_k(rewards, N + 5) == pytest.approx(float(np.max(rewards)), abs=1e-12)


def test_max_reward_at_k_monotonic_in_k():
    rng = np.random.default_rng(9)
    for _ in range(10):
        rewards = rng.random(rng.integers(3, 25))
        N = rewards.size
        vals = [max_reward_at_k(rewards, k) for k in range(1, N + 1)]
        for a, b in zip(vals, vals[1:]):
            assert b >= a - 1e-12


def test_max_reward_at_k_matches_brute_force_expectation():
    # Exact: E[max over a uniformly random size-k subset] computed by enumeration.
    rng = np.random.default_rng(11)
    for _ in range(8):
        rewards = rng.random(rng.integers(3, 9))
        N = rewards.size
        for k in range(1, N + 1):
            subsets = list(combinations(range(N), k))
            brute = float(np.mean([max(rewards[list(s)]) for s in subsets]))
            assert max_reward_at_k(rewards, k) == pytest.approx(brute, abs=1e-9), (N, k)


# --------------------------------------------------------------------------- #
# _in_box  (inclusive boundary: x1 <= px <= x2 and y1 <= py <= y2)
# --------------------------------------------------------------------------- #

BOX = (100.0, 100.0, 900.0, 900.0)  # (x1, y1, x2, y2)


def test_in_box_strictly_inside():
    x1, y1, x2, y2 = BOX
    assert _in_box(500.0, 500.0, x1, y1, x2, y2) == 1.0
    assert _in_box(101.0, 899.0, x1, y1, x2, y2) == 1.0


def test_in_box_outside():
    x1, y1, x2, y2 = BOX
    assert _in_box(99.0, 500.0, x1, y1, x2, y2) == 0.0    # left of x1
    assert _in_box(901.0, 500.0, x1, y1, x2, y2) == 0.0   # right of x2
    assert _in_box(500.0, 99.0, x1, y1, x2, y2) == 0.0    # above y1
    assert _in_box(500.0, 901.0, x1, y1, x2, y2) == 0.0   # below y2
    assert _in_box(50.0, 50.0, x1, y1, x2, y2) == 0.0     # outside corner-ward
    assert _in_box(1000.0, 1000.0, x1, y1, x2, y2) == 0.0


def test_in_box_edges_inclusive():
    x1, y1, x2, y2 = BOX
    mid_x, mid_y = 500.0, 500.0
    # A point on each of the 4 edges (boundaries are inclusive).
    assert _in_box(x1, mid_y, x1, y1, x2, y2) == 1.0   # left edge
    assert _in_box(x2, mid_y, x1, y1, x2, y2) == 1.0   # right edge
    assert _in_box(mid_x, y1, x1, y1, x2, y2) == 1.0   # top edge
    assert _in_box(mid_x, y2, x1, y1, x2, y2) == 1.0   # bottom edge


def test_in_box_corners_inclusive():
    x1, y1, x2, y2 = BOX
    for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        assert _in_box(cx, cy, x1, y1, x2, y2) == 1.0


def test_in_box_returns_float():
    x1, y1, x2, y2 = BOX
    inside = _in_box(500.0, 500.0, x1, y1, x2, y2)
    outside = _in_box(0.0, 0.0, x1, y1, x2, y2)
    assert isinstance(inside, float) and inside == 1.0
    assert isinstance(outside, float) and outside == 0.0
