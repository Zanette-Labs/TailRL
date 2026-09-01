"""Pin the offline curve estimators in analysis/curves.py.

`curves.py` re-implements pass@k and best_reward@k so the analysis stage does not need vLLM or a
GPU. That duplication is the risk: if it silently drifts from the online estimators in
`scripts/eval_screenspot_pro.py`, the published curves stop matching the numbers the eval printed.
These tests hold the two together and pin the closed forms independently.
"""
import importlib.util
import pathlib

import numpy as np
import pytest

import eval_screenspot_pro as E


ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_curves():
    spec = importlib.util.spec_from_file_location("curves", ROOT / "analysis/curves.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load_curves()


# ---------------------------------------------------------------------------
# agreement with the online estimators
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 8, 64, 256])
def test_pass_at_k_matches_the_eval_harness(n):
    for c in range(n + 1):
        for k in (1, 2, 4, 8, 16, 32, 64):
            if k > n:
                continue
            assert C.pass_at_k(n, c, k) == pytest.approx(E.bestk_unbiased(n, c, k), abs=1e-12)


def test_best_reward_at_k_matches_the_eval_harness():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(2, 80))
        rewards = rng.uniform(0, 2.5, size=n)  # SE-GUI's range, not [0,1]
        for k in (1, 2, 4, 8, 16, 64):
            assert C.best_reward_at_k(rewards, k) == pytest.approx(E.max_reward_at_k(rewards, k), abs=1e-9)


# ---------------------------------------------------------------------------
# closed forms, checked independently
# ---------------------------------------------------------------------------
def test_pass_at_k_edges():
    assert C.pass_at_k(10, 0, 5) == 0.0, "no hits => never"
    assert C.pass_at_k(10, 10, 1) == 1.0, "all hits => always"
    assert C.pass_at_k(10, 1, 1) == pytest.approx(0.1), "one hit, one draw"
    assert C.pass_at_k(10, 6, 5) == 1.0, "n-c < k => a hit is unavoidable"
    # k=1 is just the hit rate
    for c in range(11):
        assert C.pass_at_k(10, c, 1) == pytest.approx(c / 10)


def test_pass_at_k_is_monotone_in_k_and_c():
    prev = -1
    for k in range(1, 21):
        v = C.pass_at_k(20, 5, k)
        assert v >= prev
        prev = v
    prev = -1
    for c in range(21):
        v = C.pass_at_k(20, c, 4)
        assert v >= prev
        prev = v


def test_best_reward_at_k_edges():
    r = [0.0, 1.0, 2.0, 2.5]
    assert C.best_reward_at_k(r, 1) == pytest.approx(np.mean(r)), "k=1 is the plain mean"
    assert C.best_reward_at_k(r, len(r)) == pytest.approx(max(r)), "k=n is the max"
    assert C.best_reward_at_k(r, 99) == pytest.approx(max(r)), "k is clamped to n"
    assert C.best_reward_at_k([1.5] * 7, 3) == pytest.approx(1.5), "constant rewards"


def test_best_reward_at_k_reduces_to_pass_at_k_on_binary_rewards():
    """The continuous estimator must generalise the binary one, or the two curves in the same
    figure are measuring different things."""
    rng = np.random.default_rng(7)
    for _ in range(30):
        n = int(rng.integers(4, 40))
        c = int(rng.integers(0, n + 1))
        rewards = np.array([1.0] * c + [0.0] * (n - c))
        for k in (1, 2, 4, 8):
            if k > n:
                continue
            assert C.best_reward_at_k(rewards, k) == pytest.approx(C.pass_at_k(n, c, k), abs=1e-9)


def test_best_reward_at_k_is_monotone_in_k():
    rng = np.random.default_rng(3)
    rewards = rng.uniform(0, 2.5, size=40)
    prev = -np.inf
    for k in range(1, 41):
        v = C.best_reward_at_k(rewards, k)
        assert v >= prev - 1e-12
        prev = v


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, size=200)
    m, lo, hi = C.bootstrap_ci(x, n_boot=500, seed=0)
    assert m == pytest.approx(x.mean())
    assert lo < m < hi
    assert (m, lo, hi) == C.bootstrap_ci(x, n_boot=500, seed=0), "must be reproducible"


def test_bootstrap_ci_narrows_with_more_items():
    rng = np.random.default_rng(2)
    wide = C.bootstrap_ci(rng.uniform(0, 1, size=30), n_boot=500, seed=0)
    narrow = C.bootstrap_ci(rng.uniform(0, 1, size=3000), n_boot=500, seed=0)
    assert (narrow[2] - narrow[1]) < (wide[2] - wide[1])


def test_bootstrap_ci_degenerate_inputs():
    assert C.bootstrap_ci([0.5], n_boot=100) == (0.5, 0.5, 0.5)
    m, lo, hi = C.bootstrap_ci([0.25, 0.25, 0.25], n_boot=100)
    assert m == lo == hi == pytest.approx(0.25), "zero variance => zero-width interval"


# ---------------------------------------------------------------------------
# per-item aggregation + path parsing
# ---------------------------------------------------------------------------
def test_per_item_curves_groups_by_item_not_by_row():
    pd = pytest.importorskip("pandas")
    # item 0: 4 samples, 1 hit.  item 1: 4 samples, 4 hits.
    df = pd.DataFrame({
        "idx": [0, 0, 0, 0, 1, 1, 1, 1],
        "category": ["CAD"] * 8,
        "ui_type": ["icon"] * 8,
        "sample_idx": [0, 1, 2, 3, 0, 1, 2, 3],
        "reward": [0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 2.5, 2.5],
        "hit": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    })
    ids, pk, rk = C.per_item_curves(df, [1, 4])
    assert list(ids) == [0, 1]
    assert pk[1] == pytest.approx([0.25, 1.0])
    assert pk[4] == pytest.approx([1.0, 1.0]), "4 draws from 4 samples always includes the hit"
    assert rk[1] == pytest.approx([0.625, 2.5])
    assert rk[4] == pytest.approx([2.5, 2.5])


@pytest.mark.parametrize("path,arm,step", [
    ("evals/tailrl/2000.parquet", "tailrl", 2000),
    ("evals/grpo/step_8816.parquet", "grpo", 8816),
    ("out/arm=rloo/step=13000/samples.parquet", "rloo", 13000),
    ("evals/base/0.parquet", "base", 0),
    # concatenated <arm><step> directory, which is what the ladder sweep writes
    ("$EVAL_OUT/bestk_ladder/tailrl700/samples.shard0of32.parquet", "tailrl", 700),
    ("$EVAL_OUT/bestk_ladder/grpo700/samples.shard31of32.parquet", "grpo", 700),
])
def test_parse_arm_step(path, arm, step):
    assert C.parse_arm_step(path) == (arm, step)


def test_unknown_step_is_minus_one_not_none():
    """Regression: step=None silently produced ZERO charts, because pandas groupby drops NaN keys.
    An unparseable path must degrade to a real, groupable value."""
    arm, step = C.parse_arm_step("/tmp/whatever/samples.parquet")
    assert step == -1 and step is not None


def test_charts_are_emitted_for_unlabelled_steps():
    """The other half of that regression: the grouping must not drop rows whose step is unknown."""
    src = (ROOT / "analysis/curves.py").read_text()
    assert 'groupby(["step", "slice"], dropna=False)' in src, (
        "chart grouping must pass dropna=False or unlabelled steps vanish silently"
    )
