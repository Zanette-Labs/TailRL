"""Unbiased pass@k / best_reward@k estimators, plus the generate-and-score helpers
the sharded test evaluation is built on.

``besteval_shard.py`` drives this: it generates ``n`` completions per held-out program
with vLLM, scores every one of them with the gem5 speedup reward, and stores the raw
per-completion score array so best@k and pass@k can be recomputed offline at any k
without re-running gem5. ``besteval_aggregate.py`` then averages across programs.

Both estimators are the exact unbiased ones, not the "sample k of n and take the max"
approximation:

    pass@k     = 1 - C(n-c, k) / C(n, k)
    best@k     = sum_i  s_(i) * P(s_(i) is the max of a uniform k-subset)
               = sum_i  s_(i) * C(n-i, k-1) / C(n, k)     with s sorted descending

``best_at_k`` evaluates that sum with an O(n) floating-point recurrence rather than
the closed form. At n=4096, k=1024 the binomial C(4095, 1023) is a ~10^1200 integer
and ``float * C(...)`` overflows float64; the recurrence never materialises it.
"""

from __future__ import annotations

import math
import os


# ----------------------------------------------------------------------------
# best@K / pass@K
# ----------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator. n = total samples, c = #correct, k = pick size."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def best_at_k(scores: list[float], k: int) -> float:
    """Unbiased estimator of E[max(K random samples from n scores)].

    Sort scores descending; for the i-th score (1-indexed), the probability it
    is the max in a uniform K-subset of n is C(n-i, k-1)/C(n, k). Numerically
    this collapses scores below n-k+1 (probability 0). For k > n, returns max.
    """
    n = len(scores)
    if n == 0:
        return 0.0
    if k >= n:
        return max(scores)
    s = sorted(scores, reverse=True)
    # weight_i = P(i-th largest is the max of a uniform k-subset) = C(n-i,k-1)/C(n,k),
    # a probability in [0,1]. Computing C(n-i,k-1) and C(n,k) as huge ints (~10^1200 at
    # n=4096) and casting to float overflows float64. Use the exact recurrence instead —
    # pure float, no big ints:  w_1 = k/n;  w_{i+1} = w_i * (rem-k+1)/rem  where rem=n-i.
    total, w = 0.0, k / n
    for i, x in enumerate(s, start=1):
        rem = n - i
        if rem < k - 1:
            break
        total += x * w
        if rem:
            w *= (rem - k + 1) / rem
    return total


# ----------------------------------------------------------------------------
# Generation + scoring
# ----------------------------------------------------------------------------

def generate_completions(model_path: str, prompts: list[str], n_completions: int,
                         max_tokens: int = 4096, temperature: float = 0.6,
                         top_p: float = 0.95,
                         tensor_parallel_size: int = 4) -> list[list[str]]:
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.9,
        max_model_len=max_tokens + 2816,  # headroom for the up-to-2560-token PIE source-laden prompt (+256 slack)
        enforce_eager=True,
    )
    sp = SamplingParams(
        n=n_completions, temperature=temperature, top_p=top_p,
        max_tokens=max_tokens,
    )
    outputs = llm.generate(prompts, sp)
    return [[o.text for o in out.outputs] for out in outputs]


def score_completions(completions: list[list[str]], ground_truths: list[str],
                      test_case_dir: str) -> list[list[dict]]:
    """Score every completion of every problem with the same reward used in training.

    This has to be the *training* reward, or best@k measures something the run never
    optimized. Each ground_truth must carry ``usable_case_ids`` and
    ``gem5_src_per_tc_ticks``, so pass the parquet built by
    ``code_opt.build_pie_gem5_parquet``.

    Returns a list of per-problem lists of score dicts, aligned with ``completions``.
    """
    from code_opt.reward.gem5_reward import compute_scores_batch

    # Flatten: pair each completion with its ground truth, score in one big batch.
    flat_inputs = []
    flat_owner = []  # (problem_idx, completion_idx)
    for i, comps in enumerate(completions):
        for j, c in enumerate(comps):
            flat_inputs.append({"solution_str": c, "ground_truth": ground_truths[i]})
            flat_owner.append((i, j))

    # Everything else (reward kind, K, case selection, timeouts, worker count) is
    # read from the PIE_GEM5_* environment by gem5_reward itself.
    results = compute_scores_batch(flat_inputs, test_case_dir=test_case_dir)

    # Scatter back into per-problem lists.
    per_problem: list[list[dict]] = [[None] * len(comps) for comps in completions]
    for (i, j), r in zip(flat_owner, results):
        per_problem[i][j] = r
    return per_problem


# ----------------------------------------------------------------------------
