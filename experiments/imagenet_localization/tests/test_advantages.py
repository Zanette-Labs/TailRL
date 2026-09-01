"""Tests for experiments.imagenet_localization.core.advantages.

Each test has a docstring stating the invariant and why failure matters.
Hand-computed expected values are derived in comments directly above
the relevant assertion.

All tests run on CPU and complete well under 10 seconds.

Test count: 31 tests (18 parametrized sanity × 3 shapes, plus targeted tests).
"""

from __future__ import annotations

import pytest
import torch

from experiments.imagenet_localization.core.advantages import (
    ADVANTAGE_FNS,
    REWARD_TRANSFORMS,
    binary_maxrl_advantage,
    tailrl_advantage,
    grpo_advantage,
    percentile_transform,
    reinforce_advantage,
    rloo_advantage,
)


# ===========================================================================
# Parametrized per-method sanity tests (6 methods × 3 tests = 18 tests)
# ===========================================================================

_ALL_METHODS = [
    ("tailrl", tailrl_advantage),
    ("binary_maxrl", binary_maxrl_advantage),
    ("grpo", grpo_advantage),
    ("rloo", rloo_advantage),
    ("reinforce", reinforce_advantage),
]

_METHOD_IDS = [name for name, _ in _ALL_METHODS]


@pytest.mark.parametrize("method_name,method_fn", _ALL_METHODS, ids=_METHOD_IDS)
def test_output_shape(method_name, method_fn):
    """(N=16,) input must produce (16,) output for every advantage method.

    A shape mismatch would crash the policy gradient computation immediately.
    """
    rewards = torch.rand(16)
    advantages = method_fn(rewards)
    assert advantages.shape == (16,), (
        f"{method_name}: expected shape (16,), got {advantages.shape}"
    )


@pytest.mark.parametrize("method_name,method_fn", _ALL_METHODS, ids=_METHOD_IDS)
def test_output_dtype_preserved(method_name, method_fn):
    """float32 input must produce float32 output (no silent dtype promotion).

    Silent upcast to float64 would cause dtype mismatches with model weights
    and break the policy gradient loss computation.
    """
    rewards = torch.rand(16, dtype=torch.float32)
    advantages = method_fn(rewards)
    assert advantages.dtype == torch.float32, (
        f"{method_name}: expected float32, got {advantages.dtype}"
    )


@pytest.mark.parametrize("method_name,method_fn", _ALL_METHODS, ids=_METHOD_IDS)
def test_no_nan_on_random(method_name, method_fn):
    """10 random reward vectors must produce no NaN or ±inf advantages.

    NaN or inf advantages would produce undefined gradient updates and silently
    corrupt model weights — the training run would appear to continue normally
    but produce garbage outputs.
    """
    for seed in range(10):
        torch.manual_seed(seed)
        rewards = torch.rand(16)
        advantages = method_fn(rewards)
        assert not torch.isnan(advantages).any(), (
            f"{method_name}: NaN in advantages for seed={seed}"
        )
        assert not torch.isinf(advantages).any(), (
            f"{method_name}: Inf in advantages for seed={seed}"
        )


# ===========================================================================
# TailRL-specific tests
# ===========================================================================


def test_tailrl_hand_computed_N3():
    """TailRL on rewards [0.2, 0.5, 0.8] must match the hand-derived values.

    Derivation (ascending sort = identity here):
      sorted:    [0.2,   0.5,  0.8]
      prev:      [0.0,   0.2,  0.5]
      gaps:      [0.2,   0.3,  0.3]
      survivors: [3,     2,    1  ]
      increments:[0.2/3, 0.3/2, 0.3/1] = [0.0667, 0.15, 0.3]
      cumsum:    [0.0667, 0.2167, 0.5167]
      ×N=3:      [0.2,    0.65,   1.55  ]
      mean = (0.2 + 0.65 + 1.55) / 3 = 2.4 / 3 = 0.8
      centered:  [-0.6,  -0.15,   0.75 ]
      unsorted:  identity → [-0.6, -0.15, 0.75]

    Failure here means the TailRL formula is wrong, which invalidates Proposition 3.3.
    """
    rewards = torch.tensor([0.2, 0.5, 0.8])
    expected = torch.tensor([-0.6, -0.15, 0.75])
    advantages = tailrl_advantage(rewards)
    torch.testing.assert_close(advantages, expected, atol=1e-5, rtol=0)


def test_tailrl_mean_zero_after_centering():
    """TailRL advantages must have mean ~ 0 (to numerical precision) on random rewards.

    Non-zero mean advantages introduce a bias into the policy gradient that
    drives the policy away from the true reward gradient.
    """
    for seed in range(20):
        torch.manual_seed(seed)
        rewards = torch.rand(32)
        advantages = tailrl_advantage(rewards)
        mean_adv = advantages.mean().item()
        assert abs(mean_adv) < 1e-5, (
            f"TailRL mean advantage = {mean_adv:.2e}, expected ~0 (seed={seed})"
        )


def test_tailrl_rank_function_property():
    """Permuting rewards must permute advantages the same way.

    TailRL is a rank-based function: A_i depends only on the rank of r_i.
    Violation would mean the mapping is not permutation-equivariant, breaking
    the fairness of the advantage assignment.
    """
    torch.manual_seed(42)
    rewards = torch.rand(16)
    perm = torch.randperm(16)
    rewards_perm = rewards[perm]

    adv_orig = tailrl_advantage(rewards)
    adv_perm = tailrl_advantage(rewards_perm)

    # Advantages of permuted rewards must equal permuted advantages of original.
    torch.testing.assert_close(adv_perm, adv_orig[perm], atol=1e-5, rtol=0)


def test_tailrl_all_equal_rewards_zero_advantage():
    """All-equal rewards must give all-zero TailRL advantages.

    If equal rewards produce non-zero advantages, the policy is trained to
    prefer certain actions over identical ones — a spurious gradient signal.
    """
    rewards = torch.full((16,), 0.5)
    advantages = tailrl_advantage(rewards)
    torch.testing.assert_close(
        advantages, torch.zeros(16), atol=1e-6, rtol=0
    )


def test_tailrl_single_sample_zero():
    """TailRL on a single reward must return zeros (N=1 edge case).

    Leave-one-out and relative-comparison semantics are undefined for a single
    sample; returning zeros ensures no spurious gradient update occurs.
    """
    rewards = torch.tensor([0.7])
    advantages = tailrl_advantage(rewards)
    torch.testing.assert_close(advantages, torch.zeros(1), atol=1e-7, rtol=0)


def test_tailrl_highest_reward_highest_advantage():
    """For any reward vector, argmax(reward) must equal argmax(advantage) under TailRL.

    TailRL is monotone in rank: the sample with the highest reward always gets the
    highest advantage. Failure breaks the alignment between reward and learning signal.
    """
    for seed in range(10):
        torch.manual_seed(seed)
        rewards = torch.rand(16)
        # Ensure unique maximum to avoid tie-breaking ambiguity.
        rewards[0] = 1.1  # strictly above [0,1]
        advantages = tailrl_advantage(rewards)
        assert advantages.argmax() == rewards.argmax(), (
            f"seed={seed}: argmax mismatch — "
            f"reward argmax={rewards.argmax()}, adv argmax={advantages.argmax()}"
        )


# ===========================================================================
# Binary MaxRL tests
# ===========================================================================


def test_binary_maxrl_tajwar_formula_hand_computed():
    """Binary MaxRL on 5 failures + 3 successes must match the Tajwar formula.

    Derivation (N=8, K=3):
      success advantage = (N - K) / K = (8 - 3) / 3 = 5/3
      failure advantage = -1
      Expected: [-1, -1, -1, -1, -1, 5/3, 5/3, 5/3]

    Failure indicates the Tajwar scaling is not applied correctly, which would
    break the TailRL ↔ binary_maxrl equivalence on binary rewards (Proposition 3.3).
    """
    rewards = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    K = 3
    N = 8
    expected = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, 5.0 / 3.0, 5.0 / 3.0, 5.0 / 3.0])
    advantages = binary_maxrl_advantage(rewards)
    torch.testing.assert_close(advantages, expected, atol=1e-5, rtol=0)


def test_binary_maxrl_all_success_zero():
    """All-success rewards (all 1.0) must give zero advantages under binary_maxrl.

    When all outputs are successes, there is no relative signal; a non-zero
    advantage would create a spurious gradient that destabilises the policy.
    """
    rewards = torch.ones(16)
    advantages = binary_maxrl_advantage(rewards)
    torch.testing.assert_close(advantages, torch.zeros(16), atol=1e-6, rtol=0)


def test_binary_maxrl_all_failure_zero():
    """All-failure rewards (all 0.0) must give zero advantages under binary_maxrl.

    When all outputs fail, there is no relative signal; any non-zero advantage
    would introduce a biased gradient with no correct direction to follow.
    """
    rewards = torch.zeros(16)
    advantages = binary_maxrl_advantage(rewards)
    torch.testing.assert_close(advantages, torch.zeros(16), atol=1e-6, rtol=0)


# ===========================================================================
# Equivalence test: TailRL == binary_maxrl on binary rewards (Proposition 3.3)
# ===========================================================================


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("N", [4, 16, 64])
def test_tailrl_equals_binary_maxrl_on_binary_rewards(seed, N):
    """TailRL and binary_maxrl must produce identical advantages on {0,1} rewards.

    This is Proposition 3.3 at the advantage level: for Bernoulli rewards,
    the continuous TailRL formula reduces algebraically to the Tajwar binary formula.
    Failure means TailRL is not a correct generalization of binary_maxrl.

    All-same rewards (all 0 or all 1) are a valid degenerate case — both
    functions must return zeros, which is also consistent.
    """
    torch.manual_seed(seed * 100 + N)
    rewards = torch.bernoulli(torch.ones(N) * 0.5)

    adv_tailrl = tailrl_advantage(rewards)
    adv_bin = binary_maxrl_advantage(rewards)

    torch.testing.assert_close(adv_tailrl, adv_bin, atol=1e-5, rtol=0)


# ===========================================================================
# GRPO tests
# ===========================================================================


def test_grpo_mean_zero_std_one():
    """GRPO advantages on 20 distinct rewards must have mean ~0 and std ~1.

    GRPO is a z-score normalization; any deviation from mean=0, std=1 indicates
    the normalization formula is incorrect and the advantages are miscalibrated.
    """
    for seed in range(20):
        torch.manual_seed(seed)
        rewards = torch.rand(32)
        # Ensure rewards are distinct (no all-equal degenerate case).
        rewards = rewards + torch.arange(32, dtype=torch.float32) * 1e-5
        advantages = grpo_advantage(rewards)
        mean_adv = advantages.mean().item()
        std_adv = advantages.std().item()
        assert abs(mean_adv) < 0.01, (
            f"GRPO mean={mean_adv:.4f}, expected ~0 (seed={seed})"
        )
        assert abs(std_adv - 1.0) < 0.01, (
            f"GRPO std={std_adv:.4f}, expected ~1 (seed={seed})"
        )


def test_grpo_all_equal_no_nan():
    """All-equal rewards must give zero advantages (not NaN from std division).

    GRPO divides by std; when std=0 (all rewards equal), the formula would
    produce 0/0 = NaN, which must be caught by the near-zero std guard.
    """
    rewards = torch.full((16,), 0.42)
    advantages = grpo_advantage(rewards)
    assert not torch.isnan(advantages).any(), "GRPO must not produce NaN for equal rewards"
    torch.testing.assert_close(advantages, torch.zeros(16), atol=1e-6, rtol=0)


# ===========================================================================
# RLOO tests
# ===========================================================================


def test_rloo_leave_one_out_hand_computed():
    """RLOO on [1, 2, 3, 4] must match the hand-derived leave-one-out values.

    Derivation (N=4, sum=10):
      A_0 = (4*1 - 10) / (4-1) = (4  - 10) / 3 = -6/3 = -2
      A_1 = (4*2 - 10) / (4-1) = (8  - 10) / 3 = -2/3
      A_2 = (4*3 - 10) / (4-1) = (12 - 10) / 3 =  2/3
      A_3 = (4*4 - 10) / (4-1) = (16 - 10) / 3 =  2

    Failure means the leave-one-out subtraction is wrong, producing biased
    baseline estimates that skew the policy gradient.
    """
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    expected = torch.tensor([-2.0, -2.0 / 3.0, 2.0 / 3.0, 2.0])
    advantages = rloo_advantage(rewards)
    torch.testing.assert_close(advantages, expected, atol=1e-5, rtol=0)


def test_rloo_sum_to_zero():
    """Sum of RLOO advantages must be ~ 0 for any random reward vector.

    RLOO advantages sum to zero by construction: each A_i uses a leave-one-out
    baseline that balances out over the full batch. A non-zero sum indicates a
    formula error.
    """
    for seed in range(10):
        torch.manual_seed(seed)
        rewards = torch.rand(16)
        advantages = rloo_advantage(rewards)
        total = advantages.sum().item()
        assert abs(total) < 1e-5, (
            f"RLOO advantages must sum to ~0, got {total:.2e} (seed={seed})"
        )


def test_rloo_degenerate_N1_zero():
    """RLOO with N=1 must return zeros (leave-one-out is undefined for one sample).

    With only one sample, subtracting the leave-one-out mean is undefined (0/0).
    Returning zeros prevents a NaN from propagating into the gradient.
    """
    rewards = torch.tensor([0.9])
    advantages = rloo_advantage(rewards)
    torch.testing.assert_close(advantages, torch.zeros(1), atol=1e-7, rtol=0)


# ===========================================================================
# REINFORCE tests
# ===========================================================================


def test_reinforce_mean_centered():
    """REINFORCE advantages must have mean ~ 0 on any random reward vector.

    Mean centering removes the constant baseline, ensuring the gradient points
    in the direction of above-average rewards. Non-zero mean re-introduces bias.
    """
    for seed in range(20):
        torch.manual_seed(seed)
        rewards = torch.rand(32)
        advantages = reinforce_advantage(rewards)
        mean_adv = advantages.mean().item()
        assert abs(mean_adv) < 1e-5, (
            f"REINFORCE mean={mean_adv:.2e}, expected ~0 (seed={seed})"
        )


# ===========================================================================
# DG tests
# ===========================================================================






def test_all_centered_methods_produce_centered_advantages():
    """TailRL, GRPO, RLOO, and REINFORCE must produce zero-mean advantages.

    These four methods all use mean-centering (or equivalent) as part of their
    formulation. Any non-zero mean introduces a systematic bias into the gradient
    that can cause the policy to drift regardless of reward signal.

    NOTE: binary_maxrl is deliberately excluded because it only centers when
    K > 0 and K < N.
    """
    centered_methods = [
        ("tailrl", tailrl_advantage),
        ("grpo", grpo_advantage),
        ("rloo", rloo_advantage),
        ("reinforce", reinforce_advantage),
    ]

    for seed in range(10):
        torch.manual_seed(seed)
        rewards = torch.rand(32)
        for name, fn in centered_methods:
            advantages = fn(rewards)
            mean_adv = advantages.mean().item()
            assert abs(mean_adv) < 1e-5, (
                f"{name}: mean advantage = {mean_adv:.2e} (seed={seed}), expected ~0"
            )


def test_grpo_scale_invariant_but_tailrl_not():
    """GRPO advantages are scale-invariant; TailRL advantages are not.

    GRPO computes a z-score which cancels multiplicative scaling of rewards.
    TailRL advantages scale with the reward magnitude because the gaps (and thus
    the cumulative weights) scale proportionally while *N does not cancel this.

    This test validates that the two methods behave differently under reward
    scaling, which is an important property distinguishing them for research.
    """
    torch.manual_seed(7)
    r = torch.rand(16)
    r_scaled = r * 10.0

    # GRPO: scale-invariant (z-score cancels multiplicative scaling)
    grpo_r = grpo_advantage(r)
    grpo_r10 = grpo_advantage(r_scaled)
    torch.testing.assert_close(grpo_r, grpo_r10, atol=1e-5, rtol=0)

    # TailRL: scale-sensitive (gaps scale with reward magnitude)
    tailrl_r = tailrl_advantage(r)
    tailrl_r10 = tailrl_advantage(r_scaled)
    diff_norm = (tailrl_r - tailrl_r10).norm().item()
    assert diff_norm > 1e-3, (
        f"TailRL should differ under 10x reward scaling, but ||diff||={diff_norm:.2e}"
    )


# ===========================================================================
# ADVANTAGE_FNS dict tests
# ===========================================================================


def test_advantage_fns_keys_and_order():
    """ADVANTAGE_FNS must have exactly the 8 expected keys in canonical order.

    The dict serves as the central dispatch table for experiment config; any
    missing or misspelled key (e.g., 'cont_maxrl' instead of 'tailrl') would cause
    a KeyError when loading a saved experiment config.
    """
    expected_keys = [
        "tailrl", "binary_maxrl", "bmaxrl_adv_est",
        "grpo", "rloo", "reinforce", "pkpo",
    ]
    actual_keys = list(ADVANTAGE_FNS.keys())
    assert actual_keys == expected_keys, (
        f"ADVANTAGE_FNS keys mismatch: expected {expected_keys}, got {actual_keys}"
    )


def test_advantage_fns_callable():
    """Every entry in ADVANTAGE_FNS must be callable and produce valid output.

    A non-callable entry (e.g., imported string by mistake) would crash at
    training time in a way that may not surface until the experiment starts.
    """
    rewards = torch.rand(8)
    for key, fn in ADVANTAGE_FNS.items():
        assert callable(fn), f"ADVANTAGE_FNS['{key}'] is not callable"
        out = fn(rewards)
        assert out.shape == (8,), (
            f"ADVANTAGE_FNS['{key}']: expected shape (8,), got {out.shape}"
        )
        assert not torch.isnan(out).any(), (
            f"ADVANTAGE_FNS['{key}']: produced NaN on random rewards"
        )


# ===========================================================================
# percentile_transform — average-percentile-rank reward shaping
#
# Maps each rollout's IoU reward to its percentile rank in [0, 1] within the
# group: highest -> 1.0, lowest -> 0.0; tied rollouts all receive the median
# (average) of the percentiles they would otherwise span. This is the reward
# shaping fed into grpo/rloo/tailrl for the percentile ablation.
# ===========================================================================

def test_percentile_transform_distinct_endpoints():
    """Distinct rewards: min -> 0.0, max -> 1.0, ordering preserved.

    For N distinct values the percentile is rank/(N-1).
    rewards = [0.1, 0.7, 0.3, 0.9, 0.5] (N=5) sorts to ranks
    0.1->0/4, 0.3->1/4, 0.5->2/4, 0.7->3/4, 0.9->4/4, i.e. in the original
    order [0.0, 0.75, 0.25, 1.0, 0.5].
    """
    rewards = torch.tensor([0.1, 0.7, 0.3, 0.9, 0.5])
    expected = torch.tensor([0.0, 0.75, 0.25, 1.0, 0.5])
    out = percentile_transform(rewards)
    assert torch.allclose(out, expected), f"got {out.tolist()}"
    assert out.min().item() == 0.0
    assert out.max().item() == 1.0


def test_percentile_transform_ties_get_median():
    """Tied rewards all receive the median (average) percentile they span.

    rewards = [0.5, 0.5, 0.9] (N=3): the two 0.5's occupy sorted ranks
    {0, 1}; average rank = 0.5 -> 0.5/(3-1) = 0.25. The 0.9 is rank 2 ->
    2/2 = 1.0. Expected [0.25, 0.25, 1.0].
    """
    rewards = torch.tensor([0.5, 0.5, 0.9])
    expected = torch.tensor([0.25, 0.25, 1.0])
    out = percentile_transform(rewards)
    assert torch.allclose(out, expected), f"got {out.tolist()}"


def test_percentile_transform_tie_block_bottom():
    """A 3-way tie block at the bottom shares one median percentile.

    rewards = [0.2, 0.2, 0.2, 0.8] (N=4): the three 0.2's occupy ranks
    {0, 1, 2}; average rank = 1 -> 1/(4-1) = 1/3. The 0.8 is rank 3 ->
    3/3 = 1.0. Expected [1/3, 1/3, 1/3, 1.0].
    """
    rewards = torch.tensor([0.2, 0.2, 0.2, 0.8])
    third = 1.0 / 3.0
    expected = torch.tensor([third, third, third, 1.0])
    out = percentile_transform(rewards)
    assert torch.allclose(out, expected), f"got {out.tolist()}"


def test_percentile_transform_all_equal_is_half():
    """All-equal rewards -> every percentile is 0.5 (median of the full span).

    With all values tied, average rank = (N-1)/2 -> ((N-1)/2)/(N-1) = 0.5.
    The downstream estimators then map this degenerate input to ~0 advantage
    (grpo: std=0 -> 0; rloo: equal rewards -> 0; tailrl: zero gaps -> 0).
    """
    rewards = torch.full((6,), 0.4)
    out = percentile_transform(rewards)
    assert torch.allclose(out, torch.full((6,), 0.5)), f"got {out.tolist()}"


def test_percentile_transform_per_row_independent():
    """(B, N) input: transform is computed independently per row, shape preserved."""
    rewards = torch.tensor([
        [0.1, 0.7, 0.3, 0.9, 0.5],   # distinct -> [0, .75, .25, 1, .5]
        [0.5, 0.5, 0.9, 0.9, 0.1],   # ties on both 0.5 and 0.9
    ])
    out = percentile_transform(rewards)
    assert out.shape == rewards.shape
    # Row 0 matches the 1-D distinct case.
    assert torch.allclose(out[0], torch.tensor([0.0, 0.75, 0.25, 1.0, 0.5]))
    # Row 1: sorted [0.1, 0.5, 0.5, 0.9, 0.9]; 0.1 -> rank 0 = 0.0;
    # 0.5's ranks {1,2} -> avg 1.5/4 = 0.375; 0.9's ranks {3,4} -> avg 3.5/4 = 0.875.
    assert torch.allclose(out[1], torch.tensor([0.375, 0.375, 0.875, 0.875, 0.0]))


def test_percentile_transform_single_element_is_zero():
    """N=1 has no spread; return 0.0 (and never divide by N-1 = 0)."""
    out = percentile_transform(torch.tensor([[0.5], [0.2]]))
    assert out.shape == (2, 1)
    assert torch.allclose(out, torch.zeros(2, 1))


def test_reward_transforms_registry():
    """REWARD_TRANSFORMS exposes the identity, percentile and binary shapers.

    Kept as an exact-set assertion (not a superset check) so an accidentally
    registered transform still trips it; update deliberately when adding one.
    """
    assert set(REWARD_TRANSFORMS.keys()) == {
        "none", "percentile", "binary_0.5", "binary_0.75",
    }
    r = torch.rand(8)
    # 'none' is the identity transform — values unchanged.
    assert torch.equal(REWARD_TRANSFORMS["none"](r), r)
    # 'percentile' matches the standalone function.
    assert torch.allclose(REWARD_TRANSFORMS["percentile"](r), percentile_transform(r))
