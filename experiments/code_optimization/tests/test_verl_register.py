"""The batching adapter in ``code_opt.verl_register``.

``advantages.py`` owns the maths; this file owns the plumbing between it and the
flat ``[B*N, seq_len]`` tensors verl hands over. Nothing here touches Ray, vLLM or
a GPU -- the whole point of the split is that this layer is testable on CPU.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from code_opt.advantages import (
    empirical_cdf_transform,
    rloo_advantage,
    tailrl_advantage,
)
from code_opt.verl_register import CUSTOM_ESTIMATORS


# ---------------------------------------------------------------------------
# Fake batch
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def cdf_ablation_off_by_default(monkeypatch):
    """The adapter reads ``PIE_REWARD_CDF_TRANSFORM`` from the live environment, so
    a shell that already exports it for a real run would otherwise silently change
    what every test in this file is measuring. Tests that want it on set it
    themselves, which runs after this fixture."""
    monkeypatch.delenv("PIE_REWARD_CDF_TRANSFORM", raising=False)


SEQ_LEN = 8

#: B=3 prompts x N=4 rollouts, laid out group-contiguous the way verl does.
#: Response lengths differ per row (2..6 tokens) so the padding columns are real.
GROUPS = {
    "prompt-a": [0.0, 1.0, 3.0, 3.0],
    "prompt-b": [2.0, 2.0, 2.0, 2.0],     # all-equal -> every estimator gives zeros
    "prompt-c": [0.5, 9.0, 1.0, 0.25],
}
RESP_LENS = [2, 3, 5, 6, 4, 4, 2, 6, 3, 5, 6, 2]


def make_batch(groups=None, resp_lens=None, dtype=torch.float64):
    """Return (token_level_rewards, response_mask, index) exactly as verl passes them.

    The scalar outcome reward sits on the LAST response position of each row; every
    other position is 0. ``response_mask`` is 1 on response tokens and 0 on padding.
    """
    groups = GROUPS if groups is None else groups
    resp_lens = RESP_LENS if resp_lens is None else resp_lens
    rewards, uids = [], []
    for uid, group in groups.items():
        rewards.extend(group)
        uids.extend([uid] * len(group))
    rows = len(rewards)
    assert len(resp_lens) == rows

    token_level_rewards = torch.zeros(rows, SEQ_LEN, dtype=dtype)
    response_mask = torch.zeros(rows, SEQ_LEN, dtype=dtype)
    for i, (r, L) in enumerate(zip(rewards, resp_lens)):
        response_mask[i, :L] = 1.0
        token_level_rewards[i, L - 1] = r
    return token_level_rewards, response_mask, np.array(uids), rewards, resp_lens


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_contains_the_estimators_the_experiment_selects():
    """``algorithm.adv_estimator=<key>`` in the launcher must resolve. ``rloo`` is
    re-registered even though verl ships one, because this fork's version drops
    verl's extra normalization -- so the key must come from HERE, not upstream."""
    keys = sorted(CUSTOM_ESTIMATORS)
    assert "tailrl" in keys
    assert "rloo" in keys


@pytest.mark.parametrize("name", sorted(CUSTOM_ESTIMATORS))
def test_every_registered_estimator_is_callable_on_a_batch(name):
    """Run at the production group size G=16: the ``pkpo_k*`` entries are bound to
    a fixed k_opt and raise unless G >= k_opt, so ``pkpo_k16`` is only valid at
    exactly the rollout count the experiment trains with."""
    groups = {
        "prompt-a": [float(i) for i in range(16)],
        "prompt-b": [1.0] * 15 + [40.0],
    }
    tlr, mask, index, _, _ = make_batch(groups=groups,
                                        resp_lens=[1 + (i % SEQ_LEN) for i in range(32)])
    adv, ret = CUSTOM_ESTIMATORS[name](tlr, mask, index)
    assert adv.shape == tlr.shape
    assert torch.all(torch.isfinite(adv))
    assert ret is adv


# ---------------------------------------------------------------------------
# Broadcast / masking
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["tailrl", "rloo"])
def test_advantage_is_constant_over_response_tokens_and_zero_on_padding(name):
    """This is an OUTCOME reward: one scalar per rollout, credited equally to every
    token the policy actually emitted. A non-zero value on a padding column would
    put gradient on positions the policy never produced."""
    tlr, mask, index, _, resp_lens = make_batch()
    adv, _ = CUSTOM_ESTIMATORS[name](tlr, mask, index)
    for i, L in enumerate(resp_lens):
        row = adv[i]
        assert torch.allclose(row[:L], row[0].expand(L), atol=1e-12)
        assert torch.all(row[L:] == 0.0)


def test_returns_is_the_same_tensor_as_advantages():
    """Outcome reward: there is no per-token value model, so returns ARE the
    advantages. verl expects both; handing back the same object is the contract."""
    tlr, mask, index, _, _ = make_batch()
    adv, ret = CUSTOM_ESTIMATORS["tailrl"](tlr, mask, index)
    assert ret is adv


def test_scalar_reduction_ignores_reward_outside_the_response_mask():
    """The scalar is sum(token_level_rewards * response_mask): anything the reward
    manager left on a masked-out position must not reach the estimator."""
    tlr, mask, index, rewards, resp_lens = make_batch()
    poisoned = tlr.clone()
    poisoned[:, -1] += 1000.0            # last column is padding for most rows
    for i, L in enumerate(resp_lens):
        if L == SEQ_LEN:                 # ...except any row that fills the sequence
            poisoned[i, -1] -= 1000.0
    clean, _ = CUSTOM_ESTIMATORS["tailrl"](tlr, mask, index)
    dirty, _ = CUSTOM_ESTIMATORS["tailrl"](poisoned, mask, index)
    assert torch.allclose(clean, dirty, atol=1e-12)


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,fn", [("tailrl", tailrl_advantage), ("rloo", rloo_advantage)])
def test_per_group_values_match_the_1d_estimator(name, fn):
    """The adapter must be a pure regrouping: each prompt's rows carry exactly what
    calling the 1-D estimator on that prompt's scalars returns, in row order."""
    tlr, mask, index, rewards, resp_lens = make_batch()
    adv, _ = CUSTOM_ESTIMATORS[name](tlr, mask, index)
    for uid in GROUPS:
        rows = [i for i, u in enumerate(index) if u == uid]
        want = fn(torch.tensor([rewards[i] for i in rows], dtype=torch.float64))
        got = torch.tensor([adv[i, 0].item() for i in rows], dtype=torch.float64)
        assert torch.allclose(got, want, atol=1e-12), uid


def test_groups_are_independent():
    """Advantages are computed within a prompt. Changing one prompt's rewards must
    not move another's -- if it did, the batch composition would leak into the
    update and the estimator would no longer be the per-group one we analysed."""
    tlr, mask, index, _, _ = make_batch()
    base, _ = CUSTOM_ESTIMATORS["tailrl"](tlr, mask, index)

    bumped = dict(GROUPS)
    bumped["prompt-c"] = [50.0, 900.0, 51.0, 0.25]
    tlr2, mask2, index2, _, _ = make_batch(groups=bumped)
    moved, _ = CUSTOM_ESTIMATORS["tailrl"](tlr2, mask2, index2)

    untouched = [i for i, u in enumerate(index) if u != "prompt-c"]
    assert torch.allclose(base[untouched], moved[untouched], atol=1e-12)
    touched = [i for i, u in enumerate(index) if u == "prompt-c"]
    assert not torch.allclose(base[touched], moved[touched], atol=1e-6)


def test_grouping_is_by_uid_not_by_position():
    """verl lays groups out contiguously, but the adapter groups through a dict
    keyed on ``index``, so a shuffled batch must give the same per-row answer."""
    tlr, mask, index, _, _ = make_batch()
    perm = np.array([7, 0, 11, 3, 5, 9, 1, 4, 10, 2, 8, 6])
    shuffled, _ = CUSTOM_ESTIMATORS["tailrl"](tlr[perm], mask[perm], index[perm])
    straight, _ = CUSTOM_ESTIMATORS["tailrl"](tlr, mask, index)
    assert torch.allclose(shuffled, straight[perm], atol=1e-12)


def test_an_all_equal_group_gets_exactly_zero_advantage():
    """prompt-b's four rollouts all scored 2.0: no within-group signal, so those
    rows must contribute nothing at all to the gradient."""
    tlr, mask, index, _, _ = make_batch()
    adv, _ = CUSTOM_ESTIMATORS["tailrl"](tlr, mask, index)
    rows = [i for i, u in enumerate(index) if u == "prompt-b"]
    assert torch.all(adv[rows] == 0.0)


# ---------------------------------------------------------------------------
# The CDF ablation switch
# ---------------------------------------------------------------------------

def test_cdf_transform_is_off_unless_the_env_var_is_exactly_1(monkeypatch):
    """Off by default, so the advantage path is bit-identical to calling the
    estimator directly. The comparison is bitwise (atol=0) on purpose: 'off'
    must mean untouched, not 'nearly untouched'."""
    monkeypatch.delenv("PIE_REWARD_CDF_TRANSFORM", raising=False)
    tlr, mask, index, rewards, _ = make_batch()
    adv, _ = CUSTOM_ESTIMATORS["tailrl"](tlr, mask, index)
    for uid in GROUPS:
        rows = [i for i, u in enumerate(index) if u == uid]
        want = tailrl_advantage(torch.tensor([rewards[i] for i in rows], dtype=torch.float64))
        got = torch.tensor([adv[i, 0].item() for i in rows], dtype=torch.float64)
        assert torch.equal(got, want), uid


def test_cdf_transform_routes_through_the_transform_when_enabled(monkeypatch):
    """With the ablation on, the estimator must see the ranks, not the rewards."""
    monkeypatch.setenv("PIE_REWARD_CDF_TRANSFORM", "1")
    tlr, mask, index, rewards, _ = make_batch()
    adv, _ = CUSTOM_ESTIMATORS["tailrl"](tlr, mask, index)
    for uid in GROUPS:
        rows = [i for i, u in enumerate(index) if u == uid]
        scalars = torch.tensor([rewards[i] for i in rows], dtype=torch.float64)
        want = tailrl_advantage(empirical_cdf_transform(scalars))
        got = torch.tensor([adv[i, 0].item() for i in rows], dtype=torch.float64)
        assert torch.allclose(got, want, atol=1e-12), uid


@pytest.mark.parametrize("value", ["0", "", "true", "yes", "01", "2"])
def test_cdf_transform_only_accepts_the_literal_1(monkeypatch, value):
    """An exact string compare, so a typo in the launcher silently leaves the
    ablation OFF rather than half-on. Pinned so the semantics are on record."""
    monkeypatch.setenv("PIE_REWARD_CDF_TRANSFORM", value)
    tlr, mask, index, _, _ = make_batch()
    on, _ = CUSTOM_ESTIMATORS["tailrl"](tlr, mask, index)
    monkeypatch.delenv("PIE_REWARD_CDF_TRANSFORM")
    off, _ = CUSTOM_ESTIMATORS["tailrl"](tlr, mask, index)
    assert torch.equal(on, off)


def test_cdf_transform_is_read_per_call_not_at_import(monkeypatch):
    """The flag is read inside the closure, once per training step, so flipping it
    mid-process (as these tests and any resumed run do) actually takes effect."""
    tlr, mask, index, _, _ = make_batch()
    monkeypatch.delenv("PIE_REWARD_CDF_TRANSFORM", raising=False)
    before = CUSTOM_ESTIMATORS["tailrl"](tlr, mask, index)[0].clone()
    monkeypatch.setenv("PIE_REWARD_CDF_TRANSFORM", "1")
    after = CUSTOM_ESTIMATORS["tailrl"](tlr, mask, index)[0].clone()
    assert not torch.allclose(before, after, atol=1e-6)


# ---------------------------------------------------------------------------
# Dtype / shape robustness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_is_preserved(dtype):
    """verl runs float32 in production; the adapter must not silently upcast."""
    tlr, mask, index, _, _ = make_batch(dtype=dtype)
    adv, _ = CUSTOM_ESTIMATORS["tailrl"](tlr, mask, index)
    assert adv.dtype == dtype


def test_single_rollout_group_is_zeroed_not_crashed():
    """N=1 has no group to compare against. It should not raise -- a stray
    singleton group in a ragged batch must cost zero gradient, not the step."""
    groups = {"solo": [4.0], "pair": [1.0, 3.0]}
    tlr, mask, index, _, _ = make_batch(groups=groups, resp_lens=[3, 4, 2])
    adv, _ = CUSTOM_ESTIMATORS["tailrl"](tlr, mask, index)
    assert torch.all(adv[0] == 0.0)
    assert not torch.all(adv[1:] == 0.0)
