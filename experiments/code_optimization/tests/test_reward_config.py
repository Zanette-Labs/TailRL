"""The pure parts of ``code_opt.reward.gem5_reward``: reward kinds, config, case
selection, and the up-front validation.

Nothing here compiles, runs or simulates anything. The grading path itself
(``_grade_one``, ``compute_rewards_batch``'s thread pool) needs a g++ toolchain, a
gem5 build and the ~3.4 GB test-case corpus, so it is exercised on a compute node by
``python code_opt/reward/gem5_reward.py --selftest``, not here.

Importing the module requires only the host g++ that ``measurement.core`` resolves at
import time; if that is missing the whole file skips.
"""
from __future__ import annotations

import math
import os
import tempfile

import pytest

# Gem5RewardConfig's test_case_dir default raises unless PIE_TEST_CASE_DIR points
# somewhere. Nothing here reads a case file, so an empty temp dir is enough -- and
# setdefault leaves a real corpus in place if the developer already exported one.
_FALLBACK_CASE_DIR = tempfile.mkdtemp(prefix="tailrl_cases_")
os.environ.setdefault("PIE_TEST_CASE_DIR", _FALLBACK_CASE_DIR)

# Importing the reward installs the checkpoint carve-out hook, which creates a
# directory and imports verl when this is set. Neither belongs in a unit test run.
os.environ.pop("PIE_ACTOR_CARVEOUT_DIR", None)

try:
    from code_opt.reward import gem5_reward as gr
except Exception as exc:  # pragma: no cover - depends on the host toolchain
    pytest.skip(
        f"cannot import code_opt.reward.gem5_reward ({type(exc).__name__}: {exc}); "
        "it resolves a host g++ at import time -- set PIE_GXX/PIE_GCC or install one",
        allow_module_level=True,
    )


#: Env vars the config reads at construction. Anything a developer happens to have
#: exported for a real run would otherwise change what "the default" means here.
_CONFIG_PREFIXES = ("PIE_GEM5_", "PIE_NAT_", "PIE_PROGRAM_", "PIE_FILE_")
_KEEP = {"PIE_TEST_CASE_DIR"}


@pytest.fixture(autouse=True)
def clean_reward_env(monkeypatch):
    """Every test starts from the module's declared defaults, not from whatever
    launcher environment the shell is carrying. Restored by monkeypatch afterwards.

    Note this does NOT change which gem5 paths ``gem5_backend`` resolved -- those are
    module-level constants fixed at import -- so the missing-stack skip stays honest.
    """
    for name in [k for k in os.environ
                 if k.startswith(_CONFIG_PREFIXES) and k not in _KEEP]:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Reward kinds
# ---------------------------------------------------------------------------

#: kind -> (formula as documented in the module docstring, callable on the speedup s)
#: Every entry of ``REWARD_KINDS`` must appear here; the sweep below enforces that, so
#: a new kind cannot be added without writing down what it computes.
DOCUMENTED = {
    "ratio":              ("s",              lambda s: s),
    "log1p_ratio":        ("1 + ln(1 + s)",  lambda s: 1.0 + math.log(1.0 + s)),
    "gated_log_ratio":    ("ln(s) if s > 1", lambda s: math.log(s) if s > 1.0 else 0.0),
    "log_ratio":          ("ln(s)",          lambda s: math.log(s)),
    "one_plus_log_ratio": ("1 + ln(s)",      lambda s: 1.0 + math.log(s)),
    "ln1p_ratio":         ("ln(1 + s)",      lambda s: math.log(1.0 + s)),
}


def test_every_registered_reward_kind_has_a_documented_formula():
    """Guards against a kind being wired into the launcher without anyone writing
    down (or checking) what it does to the reward's shape."""
    assert sorted(gr.REWARD_KINDS) == sorted(DOCUMENTED)


@pytest.mark.parametrize("s", [1.0, 2.0, 0.5])
@pytest.mark.parametrize("kind", sorted(DOCUMENTED))
def test_reward_kind_matches_its_documented_formula(kind, s):
    """s = src_time / rollout_time, so s=1 is 'no change', s=2 is 'twice as fast'
    and s=0.5 is 'half as fast' (reachable: budget_mult=3 only truncates s < 1/3)."""
    formula, expected = DOCUMENTED[kind]
    assert gr.REWARD_KINDS[kind](s) == pytest.approx(expected(s), rel=1e-12), formula


@pytest.mark.parametrize("kind", sorted(DOCUMENTED))
def test_reward_kinds_are_monotone_increasing_in_the_speedup(kind):
    """Whatever the shape, a faster program must never score lower -- otherwise the
    reward is not a speedup reward at all."""
    fn = gr.REWARD_KINDS[kind]
    xs = [0.4, 0.5, 0.9, 1.0, 1.5, 2.0, 10.0, 100.0]
    vals = [fn(s) for s in xs]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))


def test_ln1p_ratio_is_exactly_log1p_ratio_minus_one():
    """The two differ by a constant shift, which is the whole point of having both:
    log1p_ratio's floor is 1 + ln(4/3), ln1p_ratio's is ln(4/3). Under a mean-centred
    estimator the shift is invisible; the pair exists to make that explicit."""
    for s in (0.34, 1.0, 2.5, 40.0):
        assert (gr.REWARD_KINDS["log1p_ratio"](s)
                - gr.REWARD_KINDS["ln1p_ratio"](s)) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("kind", ["ratio", "log1p_ratio", "ln1p_ratio", "gated_log_ratio"])
def test_the_positive_kinds_never_score_a_pass_below_a_failure(kind):
    """A gate failure scores exactly 0. For these kinds a correct-but-slow rollout
    must still land at or above 0, so correctness always outranks failure. The
    reachable floor is s = 1/3 (budget_mult=3 zeroes anything slower)."""
    assert gr.REWARD_KINDS[kind](1.0 / 3.0) >= 0.0


@pytest.mark.parametrize("kind,s", [("log_ratio", 0.5), ("one_plus_log_ratio", 0.34)])
def test_the_negative_capable_kinds_do_score_below_a_failure(kind, s):
    """The deliberate ablation semantics, recorded so it is not read as a bug:
    ln(s) and 1 + ln(s) go negative for slow-but-correct rollouts, ranking them
    BELOW the 0 given to a rollout that failed correctness outright."""
    assert gr.REWARD_KINDS[kind](s) < 0.0


@pytest.mark.parametrize("phase,reason", [
    ("compile", "native compile: error"),
    ("native", "case 3: mismatch"),
    ("native_budget", "native phase over wall cap"),
    ("gem5_over_budget", "rollout > 3.0x src on case(s) [1]"),
    ("extract", "no C++ code in completion"),
    ("infra", "empty usable_case_ids"),
    ("deadline", "global deadline"),
])
def test_a_failed_gate_scores_exactly_zero_whatever_the_reward_kind(phase, reason):
    """Every transform is applied ONLY on the full-pass path; every other exit goes
    through ``_zero``, which never consults the reward kind. So the choice of kind
    can change the shape of the positive rewards but can never make a failure
    non-zero -- the property the whole 'gated' design rests on."""
    got = gr._zero(phase, reason)
    assert got["score"] == 0.0
    assert got["acc"] == 0.0
    assert got["speedup"] == 0.0
    assert got["phase"] == phase


def test_zero_accepts_extra_diagnostic_fields_without_touching_the_score():
    got = gr._zero("native", "case 7: timeout", n_native_pass=3, n_native_total=9)
    assert got["score"] == 0.0
    assert got["n_native_pass"] == 3 and got["n_native_total"] == 9


# ---------------------------------------------------------------------------
# Config: env is read at CONSTRUCTION time
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("env,value,field,expected", [
    ("PIE_GEM5_REWARD_KIND", "ln1p_ratio", "reward_kind", "ln1p_ratio"),
    ("PIE_GEM5_REWARD_K", "7", "k_cases", 7),
    ("PIE_GEM5_REWARD_K", "-1", "k_cases", -1),
    ("PIE_GEM5_REWARD_SAMPLE", "fixed", "sample_mode", "fixed"),
    ("PIE_GEM5_CASE_SELECT", "biggest", "case_select", "biggest"),
    ("PIE_GEM5_REWARD_SEED", "17", "seed", 17),
    ("PIE_GEM5_BUDGET_MULT", "5.5", "budget_mult", 5.5),
    ("PIE_NAT_TIMEOUT", "9", "native_timeout_s", 9.0),
    ("PIE_NAT_CORRECT_FLOOR", "120", "native_correct_floor_s", 120.0),
    ("PIE_NAT_CORRECT_MULT", "4", "native_correct_mult", 4.0),
    ("PIE_GEM5_REWARD_DEADLINE", "900", "global_deadline_s", 900.0),
    ("PIE_GEM5_REWARD_FSIZE_MB", "8", "fsize_limit_mb", 8),
    ("PIE_GEM5_REWARD_NPROC", "64", "nproc_limit", 64),
    ("PIE_GEM5_REWARD_PCH", "0", "use_pch", False),
])
def test_config_defaults_are_read_from_the_env_at_construction(monkeypatch, env, value,
                                                               field, expected):
    """``field(default_factory=...)`` rather than a module-level constant. This is
    what lets one SLURM script export the knobs and the reward, imported long
    before by verl's exec_module, still pick them up -- and it is why a test can
    monkeypatch an env var AFTER importing the module and see the change."""
    monkeypatch.setenv(env, value)
    assert getattr(gr.Gem5RewardConfig(), field) == expected


def test_config_env_changes_do_not_retroactively_move_an_existing_config(monkeypatch):
    """Construction time, not access time: an already-built config is frozen against
    later env edits, so a mid-run export cannot silently change a live batch."""
    monkeypatch.setenv("PIE_GEM5_REWARD_KIND", "ratio")
    before = gr.Gem5RewardConfig()
    monkeypatch.setenv("PIE_GEM5_REWARD_KIND", "log_ratio")
    after = gr.Gem5RewardConfig()
    assert before.reward_kind == "ratio"
    assert after.reward_kind == "log_ratio"


def test_config_memory_limit_is_gigabytes_in_the_env_and_megabytes_on_the_field(monkeypatch):
    """Unit change across the boundary; pinned because an off-by-1024 here would
    cap every rollout's address space at 1 MB and fail correct programs."""
    monkeypatch.setenv("PIE_PROGRAM_MEM_LIMIT_GB", "3")
    assert gr.Gem5RewardConfig().mem_limit_mb == 3 * 1024


def test_config_explicit_arguments_beat_the_env(monkeypatch):
    monkeypatch.setenv("PIE_GEM5_REWARD_KIND", "ratio")
    assert gr.Gem5RewardConfig(reward_kind="ln1p_ratio").reward_kind == "ln1p_ratio"


def test_test_case_dir_is_required_and_says_how_to_get_it(monkeypatch):
    """The corpus is downloaded, not shipped. Without this the reward would run and
    zero every rollout, which is indistinguishable from a model that never finds a
    speedup -- so it has to be a loud failure at construction."""
    monkeypatch.delenv("PIE_TEST_CASE_DIR", raising=False)
    with pytest.raises(RuntimeError, match="PIE_TEST_CASE_DIR"):
        gr.Gem5RewardConfig()


def test_test_case_dir_comes_from_the_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PIE_TEST_CASE_DIR", str(tmp_path))
    assert gr.Gem5RewardConfig().test_case_dir == str(tmp_path)


def test_cfg_from_kwargs_ignores_unknown_and_none_valued_keys(monkeypatch, tmp_path):
    """The verl entry point forwards a kwargs bag that also carries non-config keys;
    a None means 'not specified', which must fall through to the env default."""
    monkeypatch.setenv("PIE_GEM5_REWARD_KIND", "ratio")
    cfg = gr._cfg_from_kwargs({"reward_kind": None, "k_cases": 3,
                               "test_case_dir": str(tmp_path),
                               "data_source": "pie", "not_a_field": 1})
    assert cfg.reward_kind == "ratio"      # None -> env default, not None
    assert cfg.k_cases == 3
    assert cfg.test_case_dir == str(tmp_path)


# ---------------------------------------------------------------------------
# Validation: a config typo must raise, never silently zero a run
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,match", [
    ({"reward_kind": "not_a_kind"}, "unknown reward_kind"),
    ({"reward_kind": "Ratio"},      "unknown reward_kind"),   # case-sensitive
    ({"k_cases": 0},                r"k_cases must be -1"),
    ({"k_cases": -2},               r"k_cases must be -1"),
    ({"budget_mult": 0.0},          "budget_mult must be > 0"),
    ({"budget_mult": -1.0},         "budget_mult must be > 0"),
    ({"sample_mode": "random"},     "unknown sample_mode"),
    ({"case_select": "smallest"},   "unknown case_select"),
])
def test_invalid_config_raises_before_anything_is_graded(kwargs, match, tmp_path):
    """All validation happens at the top of ``compute_rewards_batch``, ahead of the
    gem5-stack check and the thread pool, so an empty job list is enough to reach
    it -- no compiler and no simulator are touched. A typo that merely produced
    zero rewards would look exactly like a failed training run."""
    cfg = gr.Gem5RewardConfig(test_case_dir=str(tmp_path), **kwargs)
    with pytest.raises(ValueError, match=match):
        gr.compute_rewards_batch([], cfg)


@pytest.mark.parametrize("k_cases", [-1, 1, 5, 1000])
def test_valid_k_cases_pass_validation(k_cases, tmp_path):
    """-1 means 'all usable cases'; any positive K is clamped later against the
    problem's case count rather than rejected here."""
    cfg = gr.Gem5RewardConfig(test_case_dir=str(tmp_path), k_cases=k_cases)
    try:
        gr.compute_rewards_batch([], cfg)
    except ValueError:                       # pragma: no cover
        pytest.fail(f"k_cases={k_cases} should be accepted")
    except RuntimeError:
        pass                                 # got past validation, hit the gem5 check


@pytest.mark.skipif(gr._g5.gem5_available(),
                    reason="the gem5 stack IS present on this host, so the "
                           "missing-stack failure path cannot be exercised")
def test_a_missing_gem5_stack_fails_loudly_rather_than_zeroing_everything(tmp_path):
    """Without the check the reward still 'works': every rollout fails to time,
    scores 0, and the run looks like a model that never finds a speedup. The check
    is four os.path.exists calls -- it shells out to nothing."""
    gr._gem5_checked[0] = False
    cfg = gr.Gem5RewardConfig(test_case_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="gem5"):
        gr.compute_rewards_batch([], cfg)


# ---------------------------------------------------------------------------
# Case selection (pure, deterministic)
# ---------------------------------------------------------------------------

USABLE = [0, 1, 2, 5, 8, 13]                 # deliberately non-contiguous ids


def test_case_selection_is_a_sorted_subset_of_the_usable_cases():
    """Sorted so the gem5 forkloop always gets the same input ordering, and a
    subset so a case the correctness screen rejected can never be timed."""
    got = gr.select_cases(USABLE, 3, "group", seed=0, nonce=0, group_key="p|s", uid_idx=0)
    assert got == sorted(got)
    assert len(got) == 3
    assert set(got) <= set(USABLE)


@pytest.mark.parametrize("k", [-1, 6, 7, 999])
def test_k_at_or_above_the_case_count_selects_them_all(k):
    """k_eff = min(K, n): a problem with fewer cases than K is timed on all of
    them, never padded, skipped or zeroed."""
    assert gr.select_cases(USABLE, k, "group", 0, 0, "p|s", 0) == sorted(USABLE)


def test_group_mode_gives_every_rollout_of_a_prompt_the_same_cases():
    """The advantage estimators compare rollouts within a group, so all G of them
    must be timed on the SAME cases or the comparison is not like-for-like."""
    picks = {tuple(gr.select_cases(USABLE, 3, "group", 0, 11, "p|s", uid))
             for uid in range(16)}
    assert len(picks) == 1


def test_rollout_mode_varies_per_rollout():
    """The independent-sampling ablation: same group, different case draw per
    rollout, which is exactly what 'group' mode exists to avoid."""
    picks = {tuple(gr.select_cases(USABLE, 3, "rollout", 0, 11, "p|s", uid))
             for uid in range(16)}
    assert len(picks) > 1


def test_group_mode_rotates_with_the_step_nonce():
    """Otherwise the run would optimise one fixed 3-case subset for 300 steps."""
    per_step = {tuple(gr.select_cases(USABLE, 3, "group", 0, step, "p|s", 0))
                for step in range(20)}
    assert len(per_step) > 1


def test_fixed_mode_ignores_the_nonce():
    """'fixed' is the stable-across-steps mode used by the golden self-test and by
    any evaluation that must be comparable between checkpoints."""
    picks = {tuple(gr.select_cases(USABLE, 3, "fixed", 0, step, "p|s", 0))
             for step in range(20)}
    assert len(picks) == 1


def test_different_prompts_get_independent_draws():
    """The group key is part of the hash, so two problems at the same step do not
    happen to select structurally identical case positions."""
    a = gr.select_cases(list(range(40)), 4, "group", 0, 3, "problem-a|src-1", 0)
    b = gr.select_cases(list(range(40)), 4, "group", 0, 3, "problem-b|src-1", 0)
    assert a != b


def test_the_seed_changes_the_draw():
    a = gr.select_cases(list(range(40)), 4, "group", 0, 3, "p|s", 0)
    b = gr.select_cases(list(range(40)), 4, "group", 99, 3, "p|s", 0)
    assert a != b


def test_case_selection_is_pure_and_repeatable():
    """Same arguments, same answer, in any process: it is a SHA-256 of the key, not
    a global RNG. A run that resumes must re-time the same cases."""
    args = (USABLE, 3, "group", 0, 5, "p|s", 0)
    assert gr.select_cases(*args) == gr.select_cases(*args)


def test_unknown_sample_mode_raises():
    with pytest.raises(ValueError, match="unknown sample_mode"):
        gr.select_cases(USABLE, 3, "shuffled", 0, 0, "p|s", 0)


# ---------------------------------------------------------------------------
# Worker sizing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,want", [
    (0, 1), (1, 1), (2, 2), (3, 2), (6, 4), (64, 64),
    (72, 64), (96, 64), (129, 128), (288, 256),
])
def test_nearest_pow2_resolves_ties_downwards(n, want):
    """96 -> 64, not 128. The pool drives one pinned subprocess each, so rounding UP
    on a 96-core node would oversubscribe it and inflate the native wall clock,
    which then trips the correctness timeouts and false-zeroes correct rollouts."""
    assert gr.nearest_pow2(n) == want
