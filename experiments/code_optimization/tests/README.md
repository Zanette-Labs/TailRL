# Tests — code optimization

316 tests, pure CPU. No GPU, no gem5 build, no test-case corpus, no network. The
whole suite runs in about 3 seconds.

Run from the experiment root:

```bash
cd experiments/code_optimization
PYTHONPATH=. pytest tests/ -q                  # all
PYTHONPATH=. pytest tests/ -v                  # per-test names
PYTHONPATH=. pytest tests/test_advantages.py -v # one file
```

Everything under test here is deliberately the pure half of the experiment: the
estimator maths, the batching adapter, the offline pass@k / best@k estimators, the
code extractor, and the reward's configuration and case selection. The parts that
need real hardware — vLLM generation, the native correctness gate, gem5 timing — are
not imported at module scope and are validated separately on a compute node by
`python code_opt/reward/gem5_reward.py --selftest`.

## What each file covers

**`test_advantages.py`** (107 tests) pins the estimator maths in
`code_opt/advantages.py`. It works `tailrl_advantage` out by hand on `[0, 1, 3]`,
re-derives it in plain Python on four differently-shaped groups, and then pins the
binary-recovery identity exhaustively: for every `G` in {2, 4, 8, 16, 32} and every
`C` in `1..G-1`, TailRL on `[0]*(G-C) + [1]*C` equals `(G-C)/C` on the successes and
`-1` on the failures. That identity is what fixes the leading factor of `G` in the
estimator, so it is asserted against the closed form directly rather than only
against `binary_maxrl_advantage`. The rest covers the structural properties the
analysis assumes — permutation equivariance, sum-to-zero, zeros for `G <= 1` and for
an all-equal group, rank preservation, and the shift/scale behaviour of each
estimator — plus the PKPO baseline's argument validation and the empirical-CDF
reward transform used by the ablation.

**`test_verl_register.py`** (29 tests) pins the batching adapter in
`code_opt/verl_register.py`, the only place the estimators meet verl's flat
`[B*N, seq_len]` tensors. It builds a fake batch of three prompts by four rollouts
with varying response lengths and real padding, and asserts that the advantage is
constant across a row's response tokens and exactly zero on padding, that `returns`
is the same object as `advantages` (outcome reward, no value model), that each
group's values equal calling the 1-D estimator on that group's scalars, that groups
are independent, and that grouping is by uid rather than by row position. It also
pins the `PIE_REWARD_CDF_TRANSFORM` ablation switch: unset leaves the result
bit-identical to the direct estimator call, `1` routes through the CDF transform, and
anything else is off.

**`test_estimators.py`** (56 tests) pins the unbiasedness of `pass_at_k` and
`best_at_k` in `code_opt/eval/post_hoc_eval.py`, which produce every number in the
test-set table. Both get a closed-form check and a fixed-seed Monte-Carlo check
against the sampling process they claim to estimate, plus exhaustive enumeration at
`n=5`. The file also carries the `n=4096` overflow regression: the textbook weight
`score * C(n-i, k-1) / C(n, k)` cannot be evaluated in float64 at `k=1024` because
`C(4095, 1023)` is a ~10^1200 integer, and that is what the n=4096 evaluation was
silently losing its compute to — 4096 gem5-scored completions per program that could
not be turned into a best@k number at the k values the run existed to measure. The
tests assert both that the naive form really does raise `OverflowError` and that the
shipped O(n) recurrence returns a finite value bracketed by the mean and the max.

**`test_code_extraction.py`** (28 tests) pins `extract_code`, which decides what gets
compiled, gated and timed — so a change to it silently redefines the reward. It
covers the normal fenced-block case, multiple fences (the last complete one wins),
`<|...|>` special-token stripping, `</think>` prefix dropping, and the bare
post-anchor body when there is no fence at all. The load-bearing case is the anchor:
a model that echoes the *slow* program in its preamble must not have that echo
graded, because the echo passes correctness at a speedup of ~1.0 and would look like
a real reward. Two tests record surprising-but-real behaviour rather than assuming a
nicer rule — a trailing unterminated fence extracts the empty string, and a lone
unterminated fence returns its own backtick marker — both of which end as
score-0 rollouts downstream.

**`test_reward_config.py`** (96 tests) pins the pure parts of
`code_opt/reward/gem5_reward.py`. It checks each entry of `REWARD_KINDS` against its
documented formula at `s` in {1, 2, 0.5}, requires every registered kind to have a
documented formula, and pins which kinds can score a correct-but-slow rollout below
the 0 given to a gate failure. It asserts that a failed gate is exactly 0 whatever
the reward kind, that `Gem5RewardConfig` reads its defaults from the environment at
construction time rather than import time (so a var monkeypatched after import is
picked up), and that an invalid config — unknown `reward_kind`, `k_cases=0`,
`budget_mult<=0`, unknown `sample_mode`, unknown `case_select` — raises before
anything is graded, since a typo that merely produced zero rewards would be
indistinguishable from a failed training run. It also covers `select_cases`
(deterministic, sorted, a subset; `group` mode identical across a prompt's rollouts,
`rollout` mode independent, `fixed` mode nonce-free) and `nearest_pow2`'s
ties-resolve-down rule. Nothing in the file shells out to a compiler or to gem5.

## Conventions

- **No GPU, no gem5, no dataset, no network** in the default path, and no test is
  skipped on this machine. The two host preconditions are handled explicitly:
  `test_reward_config.py` skips at module level if the host g++ that
  `measurement.core` resolves at import time is missing, and its one
  missing-gem5-stack test skips when the stack *is* present.
- `PIE_TEST_CASE_DIR` is `setdefault`-ed to an empty temp directory so
  `Gem5RewardConfig()` constructs. No test reads a case file.
- Both env-reading modules are shielded by autouse fixtures that clear the relevant
  `PIE_*` variables, so the suite gives the same result inside a shell that has a
  real launcher environment exported.
- Monte-Carlo tests use fixed seeds. A flaky headline metric is worse than none.
- Every non-obvious assertion carries a one-line comment saying why it must hold;
  these tests are meant to document the maths as much as check it.
