"""Estimator maths for ``code_opt.advantages``.

Pure torch on CPU: no GPU, no verl, no dataset. These tests are as much a
statement of what the estimators *are* as a check that they still are it.
"""
from __future__ import annotations

import math

import pytest
import torch

from code_opt.advantages import (
    binary_maxrl_advantage,
    empirical_cdf_transform,
    grpo_advantage,
    pkpo_advantage,
    reinforce_baseline_advantage,
    rloo_advantage,
    tailrl_advantage,
)


def t(values) -> torch.Tensor:
    """float64 so the hand-computed constants below are exact to ~1e-15."""
    return torch.tensor(values, dtype=torch.float64)


#: Estimators that mean-centre their output. ``pkpo_advantage`` is deliberately
#: absent -- it is not mean-centred (see ``test_pkpo_is_not_mean_centred``).
MEAN_CENTRED = {
    "tailrl": tailrl_advantage,
    "grpo": grpo_advantage,
    "rloo": rloo_advantage,
    "reinforce_baseline": reinforce_baseline_advantage,
    "binary_maxrl": binary_maxrl_advantage,
}


# ---------------------------------------------------------------------------
# TailRL: the worked example
# ---------------------------------------------------------------------------

def test_tailrl_worked_example():
    """The whole estimator on one group, by hand.

    rewards      = [0, 1, 3]                      (already ascending, G = 3)
    prev         = [0, 0, 1]                      (r_(0) := 0, then r_(k-1))
    gaps         = [0, 1, 2]                      (r_(k) - r_(k-1))
    survivors    = [3, 2, 1]                      (G - k + 1, rank counts itself)
    gap/survivor = [0, 0.5, 2]
    cumsum       = [0, 0.5, 2.5]                  (the log-tail integral)
    x G = 3      = [0, 1.5, 7.5]
    mean         = 3.0
    centred      = [-3, -1.5, 4.5]
    """
    got = tailrl_advantage(t([0.0, 1.0, 3.0]))
    assert torch.allclose(got, t([-3.0, -1.5, 4.5]), atol=1e-12)


@pytest.mark.parametrize("rewards", [
    [0.0, 1.0, 3.0],
    [2.0, 2.0, 5.0, 9.0],                 # a tie: one gap is exactly 0
    [-4.0, 0.5, 0.5, 7.25, 100.0],        # negatives and a heavy tail
    [1.0] * 6 + [40.0],                   # the lone-outlier shape TailRL is for
])
def test_tailrl_matches_the_formula_recomputed_in_plain_python(rewards):
    """Independent re-implementation of gap-over-survivors, no torch ops shared."""
    G = len(rewards)
    srt = sorted(rewards)
    prev = [0.0] + srt[:-1]
    gaps = [a - b for a, b in zip(srt, prev)]
    survivors = [G - k for k in range(G)]          # G, G-1, ..., 1
    acc, cum = 0.0, []
    for g, s in zip(gaps, survivors):
        acc += g / s
        cum.append(acc * G)
    mean = sum(cum) / G
    want_sorted = [c - mean for c in cum]
    # Scatter back to the original (unsorted) positions the way the function does.
    order = sorted(range(G), key=lambda i: rewards[i])
    want = [0.0] * G
    for rank, i in enumerate(order):
        want[i] = want_sorted[rank]
    assert torch.allclose(tailrl_advantage(t(rewards)), t(want), atol=1e-12)


# ---------------------------------------------------------------------------
# The binary-recovery identity -- the load-bearing property
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("G", [2, 4, 8, 16, 32])
def test_binary_recovery_closed_form(G):
    """On binary rewards TailRL *is* binary MaxRL, exactly: (G-C)/C and -1.

    This is what fixes the leading factor of G in ``tailrl_advantage``. Sketch:
    with C successes the only non-zero gap is the 0->1 jump, which sits at rank
    G-C+1 and therefore has exactly C survivors. So the unscaled cumsum is 0 on
    the failures and 1/C on the successes; multiplying by G and subtracting the
    mean (which is C*(G/C)/G = 1) gives -1 and G/C - 1 = (G-C)/C. Any other
    leading constant would break the identity, so it is pinned here for every
    (G, C) rather than spot-checked.
    """
    for C in range(1, G):
        rewards = t([0.0] * (G - C) + [1.0] * C)
        got = tailrl_advantage(rewards)
        want = t([-1.0] * (G - C) + [(G - C) / C] * C)
        # (a) the closed form itself, asserted directly
        assert torch.allclose(got, want, atol=1e-12), f"G={G} C={C}"
        # (b) and, as a cross-check, agreement with the baseline it recovers
        assert torch.allclose(got, binary_maxrl_advantage(rewards), atol=1e-12)


def test_binary_recovery_survives_shuffling():
    """The identity is about the multiset of rewards, not their order."""
    rewards = t([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0])   # G=8, C=4
    got = tailrl_advantage(rewards)
    want = t([1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0])  # (G-C)/C = 1
    assert torch.allclose(got, want, atol=1e-12)


def test_tailrl_equals_rloo_at_group_size_two():
    """G=2 is the one size where the tail integral has nothing to weight.

    gaps = [a, b-a], survivors = [2, 1] -> 2*[a/2, a/2 + (b-a)] = [a, 2b-a],
    mean b, centred [a-b, b-a] -- which is exactly RLOO's 2/(2-1)*(r - mean).
    """
    for a, b in [(0.0, 1.0), (-3.0, 2.5), (7.0, 7.0)]:
        r = t([a, b])
        assert torch.allclose(tailrl_advantage(r), rloo_advantage(r), atol=1e-12)


# ---------------------------------------------------------------------------
# Structural properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("G", [2, 3, 5, 16])
def test_scale_by_group_size_is_exactly_a_factor_of_G(G):
    """``scale_by_group_size`` is a uniform rescale, so it commutes with centring."""
    torch.manual_seed(G)
    rewards = torch.rand(G, dtype=torch.float64) * 10.0
    scaled = tailrl_advantage(rewards, scale_by_group_size=True)
    unscaled = tailrl_advantage(rewards, scale_by_group_size=False)
    assert torch.allclose(unscaled, scaled / G, atol=1e-12)


@pytest.mark.parametrize("name", sorted(MEAN_CENTRED))
def test_permutation_equivariance(name):
    """Advantages are a function of the rollout's own reward and the group's
    multiset, never of the rollout's position in the batch. Permuting the input
    must permute the output identically -- otherwise the estimator would depend
    on vLLM's return order."""
    fn = MEAN_CENTRED[name]
    rewards = t([3.0, 0.5, 0.5, 9.0, 1.25, 2.0])
    perm = torch.tensor([4, 0, 5, 2, 1, 3])
    assert torch.allclose(fn(rewards)[perm], fn(rewards[perm]), atol=1e-12)


@pytest.mark.parametrize("name", sorted(MEAN_CENTRED))
@pytest.mark.parametrize("rewards", [
    [0.0, 1.0, 3.0],
    [1.0, 1.0, 1.0, 4.0],
    [0.25, 0.5, 0.75, 1.0, 8.0, 8.0],
])
def test_every_mean_centred_estimator_sums_to_zero(name, rewards):
    """Sum-zero is the variance-reduction baseline: the estimator adds no drift
    to the policy-gradient direction, it only redistributes weight in the group.
    """
    assert abs(MEAN_CENTRED[name](t(rewards)).sum().item()) < 1e-10


def test_pkpo_is_not_mean_centred():
    """PKPO's transform targets the pass@k gradient, not a centred baseline, so
    its output does NOT sum to zero. Recorded here so the sum-zero sweep above
    is understood as deliberate about which estimators it covers."""
    got = pkpo_advantage(t([0.5, 2.0, 1.0, 4.0]), k_opt=2)
    assert abs(got.sum().item()) > 1e-3


@pytest.mark.parametrize("name", sorted(MEAN_CENTRED) + ["pkpo"])
def test_degenerate_group_returns_zeros(name):
    """A group of one carries no within-group signal: there is no tail to
    integrate, no leave-one-out partner and no spread to divide by."""
    fn = pkpo_advantage if name == "pkpo" else MEAN_CENTRED[name]
    got = fn(t([4.2]))
    assert got.shape == (1,)
    assert torch.allclose(got, torch.zeros(1, dtype=torch.float64))


@pytest.mark.parametrize("name", sorted(MEAN_CENTRED))
@pytest.mark.parametrize("value", [0.0, 1.0, 7.5])
def test_all_equal_group_gives_zeros(name, value):
    """Every rollout equally good -> no gradient. For TailRL the single non-zero
    gap is r_(1) - 0, which lands in every partial sum identically and is removed
    by centring; GRPO gets there via its std guard rather than a 0/0."""
    got = MEAN_CENTRED[name](t([value] * 5))
    assert torch.allclose(got, torch.zeros(5, dtype=torch.float64), atol=1e-12)


def test_tailrl_is_rank_preserving():
    """A larger reward never gets a smaller advantage: the partial sums are
    non-decreasing in rank because every gap after the first is >= 0."""
    rewards = t([-2.0, 0.0, 0.3, 0.3, 11.0])
    adv = tailrl_advantage(rewards)
    assert torch.all(adv[1:] >= adv[:-1] - 1e-12)


# ---------------------------------------------------------------------------
# Shift / scale behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shift", [-10.0, -0.5, 3.0, 100.0])
@pytest.mark.parametrize("name", ["tailrl", "rloo", "reinforce_baseline", "grpo"])
def test_shift_invariance(name, shift):
    """Adding a constant to every reward in a group changes nothing.

    TailRL is included, and this is worth stating precisely because the docstring's
    ``r_(0) := 0`` makes it look like the absolute level matters. It does not:
    a shift of c moves ONLY the first gap (r_(1) - 0), whose survivor count is G,
    so every partial sum moves by exactly c/G -- a uniform offset that mean-centring
    removes. So TailRL is exactly shift-invariant, at either setting of
    ``scale_by_group_size``. What it is NOT is scale-invariant (below): the level
    of the rewards is irrelevant, their spacing is not.
    """
    fn = MEAN_CENTRED[name]
    rewards = t([0.0, 1.0, 2.5, 9.0])
    assert torch.allclose(fn(rewards), fn(rewards + shift), atol=1e-10)


def test_binary_maxrl_is_not_shift_invariant():
    """The baseline TailRL reduces to divides by the group MEAN, so it is tied to
    the reward's zero point -- which is why the binary-recovery identity holds at
    rewards {0,1} specifically and not at {c, c+1}."""
    rewards = t([0.0, 0.0, 1.0, 1.0])
    assert not torch.allclose(binary_maxrl_advantage(rewards),
                              binary_maxrl_advantage(rewards + 3.0), atol=1e-6)


@pytest.mark.parametrize("scale", [0.5, 2.0, 1000.0])
def test_grpo_is_scale_invariant(scale):
    """GRPO divides by the group std, so it sees only the shape of the group.
    That is the property that makes it insensitive to a heavy tail -- and the
    reason it is the interesting contrast to TailRL."""
    rewards = t([1.0, 2.0, 4.0, 30.0])
    # eps=1e-8 in the denominator means this is invariant only to ~1e-8 relative.
    assert torch.allclose(grpo_advantage(rewards), grpo_advantage(rewards * scale),
                          atol=1e-6)


@pytest.mark.parametrize("scale", [0.5, 2.0, 7.0])
@pytest.mark.parametrize("name", ["tailrl", "rloo", "reinforce_baseline"])
def test_the_non_grpo_estimators_scale_linearly(name, scale):
    """They are homogeneous of degree 1 in the reward, so a reward transform that
    stretches the tail (ratio vs log-ratio) genuinely changes the update."""
    fn = MEAN_CENTRED[name]
    rewards = t([1.0, 2.0, 4.0, 30.0])
    assert torch.allclose(fn(rewards * scale), fn(rewards) * scale, atol=1e-10)


# ---------------------------------------------------------------------------
# Baselines: RLOO, GRPO closed forms
# ---------------------------------------------------------------------------

def test_rloo_equals_leave_one_out_baseline():
    """A_i = r_i - mean of the OTHER G-1 rollouts, computed the slow way."""
    rewards = [1.0, 4.0, 6.0, 9.0]
    want = [r - (sum(rewards) - r) / (len(rewards) - 1) for r in rewards]
    assert torch.allclose(rloo_advantage(t(rewards)), t(want), atol=1e-12)


def test_grpo_is_the_population_z_score():
    """std is the biased (population) one -- unbiased=False -- so the advantages
    have unit population variance, not unit sample variance."""
    rewards = t([1.0, 2.0, 4.0, 8.0])
    want = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)
    assert torch.allclose(grpo_advantage(rewards), want, atol=1e-12)


def test_grpo_std_guard_fires_before_the_division():
    """Rewards that differ by less than eps would otherwise produce advantages of
    order (r-mean)/1e-8, i.e. an arbitrarily large update from numerical noise."""
    got = grpo_advantage(t([1.0, 1.0, 1.0 + 1e-12]))
    assert torch.allclose(got, torch.zeros(3, dtype=torch.float64))


def test_binary_maxrl_zero_mean_guard():
    """An all-failure group has mean 0; dividing by it would be a NaN broadcast
    into the loss, so it returns zeros."""
    got = binary_maxrl_advantage(t([0.0, 0.0, 0.0, 0.0]))
    assert torch.all(torch.isfinite(got)) and torch.all(got == 0)


# ---------------------------------------------------------------------------
# PKPO baseline
# ---------------------------------------------------------------------------

def test_pkpo_k1_is_G_times_mean_centred_rewards():
    """pass@1 is just the mean reward, so the k=1 branch degenerates to the
    REINFORCE baseline -- up to the same leading G every estimator here carries
    (``s_sorted = s_sorted * G`` at the end of the function). Note the factor:
    it is proportional to, not equal to, the mean-centred rewards."""
    rewards = t([0.5, 2.0, 1.0, 4.0])
    G = rewards.numel()
    got = pkpo_advantage(rewards, k_opt=1)
    assert torch.allclose(got, G * (rewards - rewards.mean()), atol=1e-12)
    # ...and therefore it IS mean-centred at k=1, unlike k>1.
    assert abs(got.sum().item()) < 1e-10


@pytest.mark.parametrize("k_opt", [0, -1, 5, 99])
def test_pkpo_rejects_out_of_range_k(k_opt):
    """k_opt is a pass@k target; k > G is unestimable from G samples and k < 1 is
    meaningless. A silent clamp would report an unbiased estimate of the wrong k."""
    with pytest.raises(ValueError, match="k_opt"):
        pkpo_advantage(t([1.0, 2.0, 3.0, 4.0]), k_opt=k_opt)


def test_pkpo_rejects_unknown_variant():
    with pytest.raises(ValueError, match="variant"):
        pkpo_advantage(t([1.0, 2.0, 3.0, 4.0]), k_opt=2, variant="not_a_variant")


@pytest.mark.parametrize("variant", ["raw", "loo", "loo_minus_one"])
def test_pkpo_variants_run_and_are_permutation_equivariant(variant):
    """All three baselines differ only by the control variate subtracted, so all
    three must still be a pure function of (own reward, group multiset)."""
    rewards = t([0.5, 2.0, 1.0, 4.0, 0.75])
    perm = torch.tensor([3, 1, 4, 0, 2])
    a = pkpo_advantage(rewards, k_opt=2, variant=variant)
    b = pkpo_advantage(rewards[perm], k_opt=2, variant=variant)
    assert torch.allclose(a[perm], b, atol=1e-10)
    assert torch.all(torch.isfinite(a))


# ---------------------------------------------------------------------------
# Empirical-CDF reward transform (ablation)
# ---------------------------------------------------------------------------

def test_empirical_cdf_values_and_ties():
    """F(r_i) = #{j : r_j <= r_i} / G. Max-rank convention: ties share the value
    of the LAST tied position, so the group maximum always lands on exactly 1.0
    and the transform stays a valid CDF."""
    got = empirical_cdf_transform(t([3.0, 1.0, 1.0, 7.0]))
    assert torch.allclose(got, t([0.75, 0.5, 0.5, 1.0]), atol=1e-12)


@pytest.mark.parametrize("rewards", [
    [0.0, 1.0, 3.0],
    [-5.0, -5.0, 0.0, 2.0, 2.0, 900.0],
    [1e-9, 1.0],
])
def test_empirical_cdf_range_and_max(rewards):
    """Range (0, 1]: 0 is impossible because r_i <= r_i always counts itself."""
    got = empirical_cdf_transform(t(rewards))
    assert torch.all(got > 0.0) and torch.all(got <= 1.0)
    assert math.isclose(got.max().item(), 1.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(got[int(torch.argmax(t(rewards)))].item(), 1.0, abs_tol=1e-12)


def test_empirical_cdf_all_equal_group_is_all_ones():
    """Every element ties with every other, so all G of them are the max. Feeding
    a constant vector to any mean-centred estimator then yields zero advantage --
    the transform cannot manufacture signal where the group had none."""
    got = empirical_cdf_transform(t([2.0] * 4))
    assert torch.allclose(got, torch.ones(4, dtype=torch.float64))
    assert torch.allclose(tailrl_advantage(got), torch.zeros(4, dtype=torch.float64))


def test_empirical_cdf_discards_magnitudes_and_keeps_only_rank():
    """The point of the ablation: after the transform every distinct order-
    statistic gap is exactly 1/G, so the advantages depend on the group's ORDER
    and nothing else. Two groups with the same ranking but wildly different
    spreads become indistinguishable -- which is precisely the degree of freedom
    TailRL otherwise uses, so the ablation isolates it."""
    heavy = t([1.0, 1.1, 1.2, 1000.0])
    flat = t([1.0, 2.0, 3.0, 4.0])
    assert torch.allclose(tailrl_advantage(empirical_cdf_transform(heavy)),
                          tailrl_advantage(empirical_cdf_transform(flat)), atol=1e-12)
    # Without the transform the same two groups give completely different updates.
    assert not torch.allclose(tailrl_advantage(heavy), tailrl_advantage(flat), atol=1e-6)
    # And the outlier's share of the total advantage mass drops.
    def top_share(adv):
        return (adv.max() / adv.abs().sum()).item()

    assert top_share(tailrl_advantage(heavy)) > top_share(
        tailrl_advantage(empirical_cdf_transform(heavy)))
