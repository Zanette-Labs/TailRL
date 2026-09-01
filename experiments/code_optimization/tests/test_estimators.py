"""Unbiasedness of the pass@k / best@k estimators in ``code_opt.eval.post_hoc_eval``.

These two functions produce every number in the test-set table, so they get both a
closed-form check and a Monte-Carlo check against the sampling process they claim to
estimate. Fixed seeds throughout: a flaky headline metric is worse than none.

Only the two pure estimators are exercised here. ``generate_completions`` needs vLLM
and a GPU and ``score_completions`` needs gem5 and the test-case corpus, so neither is
imported at module scope.
"""
from __future__ import annotations

import itertools
import math
import random

import pytest

from code_opt.eval.post_hoc_eval import best_at_k, pass_at_k


# ---------------------------------------------------------------------------
# pass@k
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,k", [(10, 1), (10, 3), (4096, 1024), (7, 7)])
def test_pass_at_k_is_zero_with_no_correct_samples(n, k):
    """c=0: no k-subset can contain a success, so the estimator must be exactly 0
    (not a tiny positive from float error -- these get averaged over programs)."""
    assert pass_at_k(n, 0, k) == 0.0


@pytest.mark.parametrize("n,k", [(10, 1), (10, 3), (10, 10), (4096, 1024)])
def test_pass_at_k_is_one_when_everything_is_correct(n, k):
    assert pass_at_k(n, n, k) == 1.0


@pytest.mark.parametrize("n,c,k", [(10, 8, 3), (10, 6, 5), (5, 1, 5), (100, 99, 2)])
def test_pass_at_k_is_one_when_failures_cannot_fill_a_subset(n, c, k):
    """n-c < k means every k-subset is forced to include a success. This is the
    branch that also keeps math.comb out of the k>n-c regime."""
    assert n - c < k
    assert pass_at_k(n, c, k) == 1.0


def test_pass_at_k_closed_form():
    """1 - C(n-c, k)/C(n, k): the probability a uniform k-subset is all failures."""
    assert pass_at_k(12, 4, 3) == pytest.approx(1.0 - math.comb(8, 3) / math.comb(12, 3))
    assert pass_at_k(12, 4, 1) == pytest.approx(4 / 12)   # pass@1 is the success rate


@pytest.mark.parametrize("n,c,k", [(12, 4, 3), (20, 1, 5), (8, 6, 2), (30, 10, 7)])
def test_pass_at_k_matches_the_sampling_process(n, c, k):
    """Monte Carlo: draw k of the n completions without replacement and ask how
    often the draw contains at least one success. The estimator is meant to BE that
    probability, so the two must agree to sampling error (30k draws, ~0.003 sd)."""
    rng = random.Random(1234)
    population = [1] * c + [0] * (n - c)
    trials = 30_000
    hits = sum(1 for _ in range(trials) if any(rng.sample(population, k)))
    assert hits / trials == pytest.approx(pass_at_k(n, c, k), abs=0.01)


# ---------------------------------------------------------------------------
# best@k
# ---------------------------------------------------------------------------

def test_best_at_k_on_an_empty_score_list():
    """A program with no scored completions contributes 0, not a crash."""
    assert best_at_k([], 4) == 0.0


@pytest.mark.parametrize("scores", [
    [0.0, 1.0, 2.0, 5.0, 7.0],
    [1.0, 1.0, 1.0],
    [3.5],
    [-2.0, 0.0, 4.0],
])
def test_best_at_k_at_k_equals_one_is_the_mean(scores):
    """One draw from n, so E[max] is just E[score]. This is the anchor the whole
    best@k curve starts from."""
    assert best_at_k(scores, 1) == pytest.approx(sum(scores) / len(scores))


@pytest.mark.parametrize("scores", [
    [0.0, 1.0, 2.0, 5.0, 7.0],
    [4.0, 4.0, 4.0],
    [-2.0, 0.0, 4.0],
])
@pytest.mark.parametrize("k_offset", [0, 1, 10])
def test_best_at_k_at_k_equals_n_or_more_is_the_max(scores, k_offset):
    """Drawing all n (or asking for more than exist) always yields the maximum."""
    assert best_at_k(scores, len(scores) + k_offset) == max(scores)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_best_at_k_matches_exhaustive_enumeration(k):
    """For n=5 every k-subset can be enumerated, so the estimator can be checked
    against the exact expectation with no sampling error at all."""
    scores = [0.0, 1.0, 2.0, 5.0, 7.0]
    exact = (sum(max(c) for c in itertools.combinations(scores, k))
             / math.comb(len(scores), k))
    assert best_at_k(scores, k) == pytest.approx(exact, abs=1e-12)


@pytest.mark.parametrize("k", [1, 2, 5, 9, 15])
def test_best_at_k_matches_the_sampling_process(k):
    """Monte Carlo against E[max of a uniform k-subset of the n scores], the
    quantity the paper's best@k curve reports. Fixed seeds for both the scores and
    the draws, so this is reproducible rather than merely usually-true."""
    score_rng = random.Random(7)
    scores = [round(score_rng.random() * 10, 4) for _ in range(20)]
    draw_rng = random.Random(4321)
    trials = 30_000
    empirical = sum(max(draw_rng.sample(scores, k)) for _ in range(trials)) / trials
    assert empirical == pytest.approx(best_at_k(scores, k), abs=0.04)


def test_best_at_k_is_monotone_non_decreasing_in_k():
    """More draws can only help: max over a superset-sized sample is never worse.
    A best@k curve that dipped would mean the estimator, not the model, moved."""
    score_rng = random.Random(11)
    scores = [score_rng.random() * 10 for _ in range(64)]
    curve = [best_at_k(scores, k) for k in range(1, 70)]
    assert all(b >= a - 1e-12 for a, b in zip(curve, curve[1:]))


@pytest.mark.parametrize("k", [1, 2, 8, 64, 1023, 1024])
def test_best_at_k_is_bracketed_by_the_mean_and_the_max(k):
    """E[max of k] is between E[max of 1] (the mean) and the max, for any k."""
    score_rng = random.Random(3)
    scores = [score_rng.random() * 10 for _ in range(1024)]
    got = best_at_k(scores, k)
    assert sum(scores) / len(scores) - 1e-12 <= got <= max(scores) + 1e-12


# ---------------------------------------------------------------------------
# The n=4096 overflow regression
# ---------------------------------------------------------------------------

def test_the_naive_closed_form_really_does_overflow_at_n_4096():
    """Documents the bug this regression guards against.

    The textbook weight is ``score * C(n-i, k-1) / C(n, k)``. At n=4096, k=1024 the
    numerator C(4095, 1023) is a ~10^1200 integer, and the float64 multiply forces a
    cast that raises OverflowError. That is what the whole n=4096 test evaluation was
    silently losing its compute to: 4096 gem5-scored completions per program were
    generated and then could not be turned into a best@k number at the k values the
    run existed to measure.
    """
    with pytest.raises(OverflowError):
        1.0 * math.comb(4095, 1023)
    with pytest.raises(OverflowError):
        float(math.comb(4096, 1024))


def test_best_at_k_survives_n_4096_k_1024():
    """The O(n) float recurrence never materialises the binomial, so the exact
    configuration the evaluation runs at returns a finite, in-range number.

    ``w_1 = k/n`` and ``w_{i+1} = w_i * (rem-k+1)/rem`` with ``rem = n-i`` is an
    algebraic rewrite of ``C(n-i, k-1)/C(n, k)``, so the value is the same unbiased
    estimator -- it just stays inside float64 the whole way.
    """
    score_rng = random.Random(0)
    scores = [score_rng.random() * 10 for _ in range(4096)]
    got = best_at_k(scores, 1024)
    assert math.isfinite(got)
    assert sum(scores) / len(scores) < got < max(scores)


@pytest.mark.parametrize("k", [1, 16, 256, 1024, 4095])
def test_best_at_k_is_finite_across_the_reported_k_ladder(k):
    """Every k the evaluation reports at n=4096, not just the one that broke."""
    score_rng = random.Random(k)
    scores = [score_rng.random() * 10 for _ in range(4096)]
    got = best_at_k(scores, k)
    assert math.isfinite(got)
    assert 0.0 <= got <= max(scores) + 1e-12


def test_best_at_k_recurrence_agrees_with_the_big_int_closed_form():
    """Same estimator, computed two ways, at a size where the big-int form still
    fits: int/int division in Python is correctly rounded, so this is an exact
    cross-check of the recurrence rather than of the docstring."""
    score_rng = random.Random(5)
    n, k = 60, 17
    scores = sorted((score_rng.random() * 10 for _ in range(n)), reverse=True)
    exact = sum(x * (math.comb(n - i, k - 1) / math.comb(n, k))
                for i, x in enumerate(scores, start=1) if n - i >= k - 1)
    assert best_at_k(scores, k) == pytest.approx(exact, rel=1e-9)
