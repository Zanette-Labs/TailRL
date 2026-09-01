"""Pin ``verl.trainer.metrics.compute_group_advantage_metrics``.

Within-group (per-prompt / per-uid) reward & advantage spread diagnostics used to tell TailRL
apart from GRPO on the money plot: GRPO pins Std(A) ~1 regardless of reward spread, TailRL/centered
lets Std(A) track the reward std. Plus degenerate/bimodal fraction drift checks.

Real implementation (metrics.py), reproduced here so expectations match exactly:
  scores      = token_level_rewards.sum(-1)                      # (bs,) scalar reward per rollout
  tok         = response_mask.float().sum(-1).clamp(min=1.0)
  adv_scalar  = (advantages * response_mask).sum(-1) / tok       # per-rollout advantage
  group by non_tensor_batch["uid"]; groups with < 2 members are SKIPPED
  per group:  rs   = pop-std(scores)          (unbiased=False, ddof=0)
              a_s  = pop-std(adv_scalar)
              mid  = mean( mid_lo <= r <= mid_hi )     (both bounds inclusive)
              degen= 1.0 if rs <  degenerate_std else 0.0        (STRICT <)
              bimod= 1.0 if (rs >= bimodal_std_min and mid < bimodal_mid_max) else 0.0
  return means over groups; if NO qualifying group, every metric is 0.0
Keys: group/reward_std_mean, group/advantage_std_mean, group/mid_frac_mean,
      group/degenerate_frac, group/bimodal_frac
CPU-only, no model / vLLM / GPU.
"""
import types

import numpy as np
import pytest
import torch

from verl.trainer.metrics import compute_group_advantage_metrics

EXPECTED_KEYS = {
    "group/reward_std_mean",
    "group/advantage_std_mean",
    "group/mid_frac_mean",
    "group/degenerate_frac",
    "group/bimodal_frac",
    "group/survivor_std_mean",
}

# The decile histogram is BATCH-level, not group-level: it is computed over every rollout, so
# unlike the group/ metrics it does NOT skip singleton groups. It exists because the scalar
# graded-ness summaries can all look healthy while the reward mass is piled at the two ends.
HIST_KEYS = {f"reward_hist/bin_{i}" for i in range(10)}


# --------------------------------------------------------------------------- helpers
def make_batch(rewards, adv_scalars, uids, L=3, mask=None):
    """Build a stub DataProto-like namespace.

    Per-rollout scalar reward `rewards[i]` is placed in column 0 of token_level_rewards so
    ``.sum(-1) == rewards[i]``. `adv_scalars[i]` is broadcast constant across the L response
    tokens so ``adv_scalar == adv_scalars[i]`` (constant-per-token is what outcome estimators
    like GRPO/TailRL produce). response_mask defaults to all-ones (bs, L).
    """
    rewards = [float(x) for x in rewards]
    adv_scalars = [float(x) for x in adv_scalars]
    bs = len(rewards)
    tlr = torch.zeros((bs, L), dtype=torch.float32)
    if bs > 0:
        tlr[:, 0] = torch.tensor(rewards, dtype=torch.float32)
    adv = torch.zeros((bs, L), dtype=torch.float32)
    for i in range(bs):
        adv[i, :] = adv_scalars[i]
    if mask is None:
        rmask = torch.ones((bs, L), dtype=torch.float32)
    else:
        rmask = torch.tensor(mask, dtype=torch.float32).reshape(bs, L)
    return types.SimpleNamespace(
        batch={
            "token_level_rewards": tlr,
            "advantages": adv,
            "response_mask": rmask,
        },
        non_tensor_batch={"uid": np.array(list(uids), dtype=object)},
    )


def centered_adv(rewards, uids):
    """TailRL/centered outcome advantage: r - group_mean (per uid)."""
    rewards = np.asarray(rewards, dtype=np.float64)
    uids = list(uids)
    out = np.zeros(len(rewards), dtype=np.float64)
    for g in set(uids):
        idx = [i for i, u in enumerate(uids) if u == g]
        out[idx] = rewards[idx] - rewards[idx].mean()
    return out.tolist()


def zscored_adv(rewards, uids):
    """GRPO outcome advantage: (r - group_mean) / group_pop_std (per uid). Requires std > 0."""
    rewards = np.asarray(rewards, dtype=np.float64)
    uids = list(uids)
    out = np.zeros(len(rewards), dtype=np.float64)
    for g in set(uids):
        idx = [i for i, u in enumerate(uids) if u == g]
        out[idx] = (rewards[idx] - rewards[idx].mean()) / rewards[idx].std()
    return out.tolist()


def numpy_reference(rewards, adv_scalars, uids, mid_lo=0.2, mid_hi=0.8,
                    degenerate_std=1e-3, bimodal_std_min=0.30, bimodal_mid_max=0.25):
    """Independent numpy reimplementation of the aggregation, for property tests."""
    rewards = np.asarray(rewards, dtype=np.float64)
    adv = np.asarray(adv_scalars, dtype=np.float64)
    uids = list(uids)
    r_stds, a_stds, mids, degen, bimod, surv = [], [], [], [], [], []
    for g in dict.fromkeys(uids):  # unique, insertion order
        idx = [i for i, u in enumerate(uids) if u == g]
        if len(idx) < 2:
            continue
        r = rewards[idx]
        a = adv[idx]
        rs = float(r.std())  # ddof=0
        r_stds.append(rs)
        a_stds.append(float(a.std()))
        mid = float(((r >= mid_lo) & (r <= mid_hi)).mean())
        mids.append(mid)
        degen.append(1.0 if rs < degenerate_std else 0.0)
        bimod.append(1.0 if (rs >= bimodal_std_min and mid < bimodal_mid_max) else 0.0)
        surv_r = r[r > 0.2]
        surv.append(float(surv_r.std()) if surv_r.size >= 2 else 0.0)  # ddof=0
    m = lambda x: float(sum(x) / len(x)) if x else 0.0  # noqa: E731
    return {
        "group/reward_std_mean": m(r_stds),
        "group/advantage_std_mean": m(a_stds),
        "group/mid_frac_mean": m(mids),
        "group/degenerate_frac": m(degen),
        "group/bimodal_frac": m(bimod),
        "group/survivor_std_mean": m(surv),
    }


POP_STD_QUAD = 0.1118033988749895  # np.std([0.1, 0.2, 0.3, 0.4])  ddof=0


# --------------------------------------------------------------------------- schema
def test_returns_exactly_six_keys():
    out = compute_group_advantage_metrics(
        make_batch([0.1, 0.2, 0.3, 0.4], [-0.15, -0.05, 0.05, 0.15], ["g0"] * 4)
    )
    assert set(out.keys()) == EXPECTED_KEYS | HIST_KEYS
    assert len(out) == 16
    # the histogram is a distribution over the batch's rollouts
    assert sum(out[k] for k in HIST_KEYS) == pytest.approx(1.0)


def test_reward_range_rescales_the_shape_tests():
    """The mid-band and bimodality thresholds are SHAPE tests, so they must read the same on a
    reward with different units. SE-GUI's `overall` spans [0.5, 2.5] once the format term is
    earned; left on the [0,1] default, a perfectly graded group would report mid_frac 0."""
    uids = ["g0"] * 4
    # deliberately mid-bin (0.05, not 0.1): a value sitting exactly on a histogram bin edge lands
    # on either side depending on float32 rounding, which would make this test flaky rather than wrong
    unit = [0.05, 0.35, 0.65, 0.95]                  # spread across the [0,1] mid band
    segui = [0.5 + 2.0 * r for r in unit]            # the identical shape in SE-GUI units

    base = compute_group_advantage_metrics(make_batch(unit, centered_adv(unit, uids), uids))
    naive = compute_group_advantage_metrics(make_batch(segui, centered_adv(segui, uids), uids))
    scaled = compute_group_advantage_metrics(
        make_batch(segui, centered_adv(segui, uids), uids), reward_range=(0.5, 2.5)
    )

    assert base["group/mid_frac_mean"] == pytest.approx(0.5)  # 0.4 and 0.6 are in [0.2, 0.8]
    assert scaled["group/mid_frac_mean"] == pytest.approx(0.5), "rescaled: same shape, same answer"
    # Unrescaled, the [0.2, 0.8] band lands on the BOTTOM of a [0.5, 2.5] reward: it reports 0.25
    # and the one rollout it counts (0.7, the group's WORST) is the one the rescaled band excludes.
    # The number is not merely low, it is measuring the opposite end of the distribution.
    assert naive["group/mid_frac_mean"] == pytest.approx(0.25)
    assert naive["group/mid_frac_mean"] != pytest.approx(base["group/mid_frac_mean"])
    # the histogram is rescaled by the same span, so the shape is identical in both unit systems
    for i in range(10):
        assert scaled[f"reward_hist/bin_{i}"] == pytest.approx(base[f"reward_hist/bin_{i}"], abs=1e-6)

    # reward_std_mean stays in RAW units so it remains comparable with the reward's own scale
    assert scaled["group/reward_std_mean"] == pytest.approx(2.0 * base["group/reward_std_mean"])


def test_reward_range_default_is_inert():
    """Every existing caller must be unaffected."""
    uids = ["g0"] * 4
    r = [0.0, 0.3, 0.7, 1.0]
    a = compute_group_advantage_metrics(make_batch(r, centered_adv(r, uids), uids))
    b = compute_group_advantage_metrics(make_batch(r, centered_adv(r, uids), uids), reward_range=(0.0, 1.0))
    assert a == b


def test_degenerate_frac_stays_on_the_raw_scale():
    """'This group is flat' is an absolute claim, so the 1e-3 threshold is deliberately NOT
    rescaled -- a flat group is flat in whatever units the reward uses."""
    uids = ["g0"] * 4
    flat = [1.5, 1.5, 1.5, 1.5]
    out = compute_group_advantage_metrics(
        make_batch(flat, [0.0] * 4, uids), reward_range=(0.5, 2.5)
    )
    assert out["group/degenerate_frac"] == 1.0


def test_survivor_std_mean_ignores_low_and_needs_two_survivors():
    # survivors = rewards > 0.2.  g0: {0.0,0.3,0.7} -> survivors {0.3,0.7}, pop-std 0.2
    #                            g1: {0.0,0.0,0.9} -> survivors {0.9} (<2) -> 0.0
    uids = ["g0", "g0", "g0", "g1", "g1", "g1"]
    r = [0.0, 0.3, 0.7, 0.0, 0.0, 0.9]
    out = compute_group_advantage_metrics(make_batch(r, centered_adv(r, uids), uids))
    expected = (float(np.std([0.3, 0.7])) + 0.0) / 2.0  # (0.2 + 0.0)/2
    assert out["group/survivor_std_mean"] == pytest.approx(expected, abs=1e-6)


def test_all_values_are_python_floats():
    out = compute_group_advantage_metrics(
        make_batch([0.0, 0.0, 1.0, 1.0], [-0.5, -0.5, 0.5, 0.5], ["g0"] * 4)
    )
    for k, v in out.items():
        assert type(v) is float, f"{k} -> {type(v)}"


# --------------------------------------------------- anchor: centered (TailRL) single group
def test_centered_advantage_std_equals_reward_std():
    # adv = r - mean  => Std(A) tracks the reward spread exactly (TailRL / centered outcome).
    r = [0.1, 0.2, 0.3, 0.4]
    out = compute_group_advantage_metrics(
        make_batch(r, centered_adv(r, ["g0"] * 4), ["g0"] * 4)
    )
    assert out["group/reward_std_mean"] == pytest.approx(POP_STD_QUAD, abs=1e-6)
    assert out["group/advantage_std_mean"] == pytest.approx(POP_STD_QUAD, abs=1e-6)
    assert out["group/advantage_std_mean"] == pytest.approx(out["group/reward_std_mean"], abs=1e-6)


def test_centered_single_group_mid_and_fracs():
    r = [0.1, 0.2, 0.3, 0.4]
    out = compute_group_advantage_metrics(
        make_batch(r, centered_adv(r, ["g0"] * 4), ["g0"] * 4)
    )
    # in [0.2, 0.8]: 0.2, 0.3, 0.4 -> 3/4 (0.1 excluded, lower bound inclusive)
    assert out["group/mid_frac_mean"] == pytest.approx(0.75)
    assert out["group/degenerate_frac"] == 0.0
    assert out["group/bimodal_frac"] == 0.0


# --------------------------------------------------- anchor: z-scored (GRPO) single group
def test_zscored_advantage_std_is_one():
    # adv = (r - mean)/std  => Std(A) == 1 regardless of reward spread (GRPO signature).
    r = [0.1, 0.2, 0.3, 0.4]
    out = compute_group_advantage_metrics(
        make_batch(r, zscored_adv(r, ["g0"] * 4), ["g0"] * 4)
    )
    assert out["group/advantage_std_mean"] == pytest.approx(1.0, abs=1e-5)
    assert out["group/reward_std_mean"] == pytest.approx(POP_STD_QUAD, abs=1e-6)
    # the whole point: advantage std decoupled from reward std under GRPO
    assert out["group/advantage_std_mean"] != pytest.approx(out["group/reward_std_mean"], abs=1e-2)


def test_zscored_preserves_reward_diagnostics():
    # z-scoring changes advantage std but NOT reward-side metrics (mid/degen/bimod use rewards).
    r = [0.1, 0.2, 0.3, 0.4]
    out = compute_group_advantage_metrics(
        make_batch(r, zscored_adv(r, ["g0"] * 4), ["g0"] * 4)
    )
    assert out["group/mid_frac_mean"] == pytest.approx(0.75)
    assert out["group/degenerate_frac"] == 0.0
    assert out["group/bimodal_frac"] == 0.0


# --------------------------------------------------- anchor: degenerate group
def test_degenerate_group_all_equal():
    r = [0.5, 0.5, 0.5, 0.5]
    out = compute_group_advantage_metrics(
        make_batch(r, [0.0] * 4, ["g0"] * 4)  # centered adv of equal rewards == 0
    )
    assert out["group/reward_std_mean"] == pytest.approx(0.0, abs=1e-7)
    assert out["group/advantage_std_mean"] == pytest.approx(0.0, abs=1e-7)
    assert out["group/degenerate_frac"] == 1.0
    assert out["group/mid_frac_mean"] == pytest.approx(1.0)  # 0.5 in [0.2, 0.8]
    assert out["group/bimodal_frac"] == 0.0


# --------------------------------------------------- anchor: bimodal group
def test_bimodal_group_0011():
    r = [0.0, 0.0, 1.0, 1.0]
    out = compute_group_advantage_metrics(
        make_batch(r, centered_adv(r, ["g0"] * 4), ["g0"] * 4)
    )
    assert out["group/reward_std_mean"] == pytest.approx(0.5)  # pop std of {0,0,1,1}
    assert out["group/bimodal_frac"] == 1.0                    # std>=0.30 AND mid<0.25 (mid==0)
    assert out["group/mid_frac_mean"] == pytest.approx(0.0)
    assert out["group/degenerate_frac"] == 0.0


# --------------------------------------------------- anchor: tight-low cluster (NOT bimodal)
def test_tight_low_cluster_not_bimodal():
    r = [0.10, 0.11, 0.12, 0.13]
    out = compute_group_advantage_metrics(
        make_batch(r, centered_adv(r, ["g0"] * 4), ["g0"] * 4)
    )
    expected_std = float(np.std(r))  # ~0.011 < bimodal_std_min(0.30)
    assert out["group/reward_std_mean"] == pytest.approx(expected_std, abs=1e-6)
    assert out["group/mid_frac_mean"] == pytest.approx(0.0)   # all below 0.2
    assert out["group/bimodal_frac"] == 0.0                   # empty middle but spread too small
    assert out["group/degenerate_frac"] == 0.0                # 0.011 >= degenerate_std(1e-3)


def test_high_spread_but_populated_middle_not_bimodal():
    # std >= 0.30 but the middle band is NOT empty -> not bimodal (needs BOTH conditions).
    r = [0.0, 0.5, 0.5, 1.0]
    out = compute_group_advantage_metrics(
        make_batch(r, centered_adv(r, ["g0"] * 4), ["g0"] * 4)
    )
    assert out["group/reward_std_mean"] == pytest.approx(float(np.std(r)), abs=1e-6)
    assert float(np.std(r)) >= 0.30
    assert out["group/mid_frac_mean"] == pytest.approx(0.5)   # >= bimodal_mid_max
    assert out["group/bimodal_frac"] == 0.0


# --------------------------------------------------- band-boundary inclusivity
def test_lower_bound_inclusive():
    r = [0.2, 0.2, 0.2, 0.2]  # exactly at mid_lo -> counted in band
    out = compute_group_advantage_metrics(make_batch(r, [0.0] * 4, ["g0"] * 4))
    assert out["group/mid_frac_mean"] == pytest.approx(1.0)


def test_upper_bound_inclusive():
    r = [0.8, 0.8, 0.8, 0.8]  # exactly at mid_hi -> counted in band
    out = compute_group_advantage_metrics(make_batch(r, [0.0] * 4, ["g0"] * 4))
    assert out["group/mid_frac_mean"] == pytest.approx(1.0)


def test_mixed_band_membership():
    r = [0.2, 0.5, 0.1, 0.9]  # in: 0.2, 0.5 ; out: 0.1, 0.9
    out = compute_group_advantage_metrics(
        make_batch(r, centered_adv(r, ["g0"] * 4), ["g0"] * 4)
    )
    assert out["group/mid_frac_mean"] == pytest.approx(0.5)


# --------------------------------------------------- multi-group averaging
def test_multi_group_reward_std_is_average_over_groups():
    uids = ["g0"] * 4 + ["g1"] * 4
    r = [0.1, 0.2, 0.3, 0.4] + [0.0, 0.0, 1.0, 1.0]
    out = compute_group_advantage_metrics(make_batch(r, centered_adv(r, uids), uids))
    expected = (POP_STD_QUAD + 0.5) / 2.0
    assert out["group/reward_std_mean"] == pytest.approx(expected, abs=1e-6)


def test_multi_group_centered_advantage_std_average():
    uids = ["g0"] * 4 + ["g1"] * 4
    r = [0.1, 0.2, 0.3, 0.4] + [0.0, 0.0, 1.0, 1.0]
    out = compute_group_advantage_metrics(make_batch(r, centered_adv(r, uids), uids))
    # centered -> per-group advantage std == per-group reward std
    expected = (POP_STD_QUAD + 0.5) / 2.0
    assert out["group/advantage_std_mean"] == pytest.approx(expected, abs=1e-6)


def test_multi_group_zscored_advantage_std_is_one():
    uids = ["g0"] * 4 + ["g1"] * 4
    r = [0.1, 0.2, 0.3, 0.4] + [0.0, 0.0, 1.0, 1.0]
    out = compute_group_advantage_metrics(make_batch(r, zscored_adv(r, uids), uids))
    assert out["group/advantage_std_mean"] == pytest.approx(1.0, abs=1e-5)


def test_multi_group_mid_frac_average():
    uids = ["g0"] * 4 + ["g1"] * 4
    r = [0.1, 0.2, 0.3, 0.4] + [0.0, 0.0, 1.0, 1.0]  # mids: 0.75 and 0.0
    out = compute_group_advantage_metrics(make_batch(r, centered_adv(r, uids), uids))
    assert out["group/mid_frac_mean"] == pytest.approx((0.75 + 0.0) / 2.0)


def test_multi_group_degenerate_frac_average():
    uids = ["g0"] * 4 + ["g1"] * 4
    r = [0.5, 0.5, 0.5, 0.5] + [0.1, 0.2, 0.3, 0.4]  # degenerate + not
    out = compute_group_advantage_metrics(make_batch(r, centered_adv(r, uids), uids))
    assert out["group/degenerate_frac"] == pytest.approx(0.5)


def test_multi_group_bimodal_frac_average():
    uids = ["g0"] * 4 + ["g1"] * 4
    r = [0.0, 0.0, 1.0, 1.0] + [0.1, 0.2, 0.3, 0.4]  # bimodal + not
    out = compute_group_advantage_metrics(make_batch(r, centered_adv(r, uids), uids))
    assert out["group/bimodal_frac"] == pytest.approx(0.5)


# --------------------------------------------------- group-size gating (< 2 skipped)
def test_all_singletons_return_zero():
    uids = ["a", "b", "c", "d"]  # every group size 1 -> all skipped
    r = [0.15, 0.45, 0.75, 0.95]  # mid-bin values, so the assertion is not float-edge sensitive
    out = compute_group_advantage_metrics(make_batch(r, [0.0] * 4, uids))
    assert {k: out[k] for k in EXPECTED_KEYS} == {k: 0.0 for k in EXPECTED_KEYS}
    # ...but the histogram is batch-level, so it still sees all four rollouts
    assert sum(out[k] for k in HIST_KEYS) == pytest.approx(1.0)
    for b in (1, 4, 7, 9):
        assert out[f"reward_hist/bin_{b}"] == pytest.approx(0.25)


def test_empty_batch_returns_zero():
    out = compute_group_advantage_metrics(make_batch([], [], []))
    assert out == {k: 0.0 for k in EXPECTED_KEYS | HIST_KEYS}


def test_singletons_mixed_with_one_real_group():
    # only the 3-member group counts; the two singletons are skipped.
    uids = ["g0", "g0", "g0", "solo1", "solo2"]
    r = [0.0, 0.5, 1.0, 0.42, 0.99]
    out = compute_group_advantage_metrics(make_batch(r, centered_adv(r, uids), uids))
    ref = numpy_reference(r, centered_adv(r, uids), uids)
    for k in EXPECTED_KEYS:
        assert out[k] == pytest.approx(ref[k], abs=1e-6)
    # sanity: reward std reflects only {0.0, 0.5, 1.0}
    assert out["group/reward_std_mean"] == pytest.approx(float(np.std([0.0, 0.5, 1.0])), abs=1e-6)


def test_two_member_group_is_kept():
    out = compute_group_advantage_metrics(
        make_batch([0.0, 1.0], centered_adv([0.0, 1.0], ["g0", "g0"]), ["g0", "g0"])
    )
    assert out["group/reward_std_mean"] == pytest.approx(0.5)  # pop std of {0,1}


# --------------------------------------------------- response_mask handling
def test_adv_scalar_averages_only_masked_tokens():
    # Unmasked positions carry garbage; adv_scalar must ignore them (mask-weighted mean).
    L = 4
    mask = [[1, 1, 0, 0], [1, 1, 0, 0]]
    adv = torch.tensor([[-0.1, -0.1, 999.0, 999.0],
                        [0.1, 0.1, -999.0, -999.0]], dtype=torch.float32)
    tlr = torch.zeros((2, L), dtype=torch.float32)
    tlr[:, 0] = torch.tensor([0.2, 0.4])  # rewards; scores sum over ALL tokens
    batch = types.SimpleNamespace(
        batch={
            "token_level_rewards": tlr,
            "advantages": adv,
            "response_mask": torch.tensor(mask, dtype=torch.float32),
        },
        non_tensor_batch={"uid": np.array(["g0", "g0"], dtype=object)},
    )
    out = compute_group_advantage_metrics(batch)
    # adv_scalar -> {-0.1, 0.1}; std == 0.1 ; reward std of {0.2,0.4} == 0.1
    assert out["group/advantage_std_mean"] == pytest.approx(0.1, abs=1e-6)
    assert out["group/reward_std_mean"] == pytest.approx(0.1, abs=1e-6)


def test_scores_sum_over_all_response_tokens():
    # reward is the FULL token-level sum regardless of how it is distributed across L.
    L = 3
    tlr = torch.tensor([[0.1, 0.05, 0.05],   # sum 0.2
                        [0.2, 0.1, 0.1]], dtype=torch.float32)  # sum 0.4
    batch = types.SimpleNamespace(
        batch={
            "token_level_rewards": tlr,
            "advantages": torch.zeros((2, L), dtype=torch.float32),
            "response_mask": torch.ones((2, L), dtype=torch.float32),
        },
        non_tensor_batch={"uid": np.array(["g0", "g0"], dtype=object)},
    )
    out = compute_group_advantage_metrics(batch)
    assert out["group/reward_std_mean"] == pytest.approx(0.1, abs=1e-6)  # std{0.2,0.4}


# --------------------------------------------------- custom thresholds
def test_custom_mid_band():
    r = [0.1, 0.2, 0.3, 0.4]
    out = compute_group_advantage_metrics(
        make_batch(r, centered_adv(r, ["g0"] * 4), ["g0"] * 4),
        mid_lo=0.15, mid_hi=0.35,  # in-band: 0.2, 0.3
    )
    assert out["group/mid_frac_mean"] == pytest.approx(0.5)


def test_custom_degenerate_std_flags_tight_cluster():
    r = [0.10, 0.11, 0.12, 0.13]  # std ~0.011
    out = compute_group_advantage_metrics(
        make_batch(r, [0.0] * 4, ["g0"] * 4),
        degenerate_std=0.05,  # now 0.011 < 0.05 -> degenerate
    )
    assert out["group/degenerate_frac"] == 1.0


def test_custom_bimodal_std_min_flags_tight_empty_middle():
    r = [0.10, 0.11, 0.12, 0.13]  # mid==0 (all below 0.2), std ~0.011
    out = compute_group_advantage_metrics(
        make_batch(r, [0.0] * 4, ["g0"] * 4),
        bimodal_std_min=0.005,  # 0.011 >= 0.005 AND mid 0 < 0.25 -> bimodal
    )
    assert out["group/bimodal_frac"] == 1.0


# --------------------------------------------------- threshold boundary semantics
def test_degenerate_is_strict_less_than():
    # rs == degenerate_std must NOT be flagged (comparison is strict <).
    r = [0.5, 0.5, 0.5, 0.5]  # rs == 0.0
    out = compute_group_advantage_metrics(
        make_batch(r, [0.0] * 4, ["g0"] * 4),
        degenerate_std=0.0,  # 0.0 < 0.0 is False
    )
    assert out["group/degenerate_frac"] == 0.0


def test_bimodal_std_min_boundary_is_inclusive():
    r = [0.0, 0.0, 1.0, 1.0]  # rs == 0.5 exactly (float32-exact)
    flagged = compute_group_advantage_metrics(
        make_batch(r, centered_adv(r, ["g0"] * 4), ["g0"] * 4),
        bimodal_std_min=0.5,  # rs >= 0.5 -> True
    )
    not_flagged = compute_group_advantage_metrics(
        make_batch(r, centered_adv(r, ["g0"] * 4), ["g0"] * 4),
        bimodal_std_min=0.5001,  # rs >= 0.5001 -> False
    )
    assert flagged["group/bimodal_frac"] == 1.0
    assert not_flagged["group/bimodal_frac"] == 0.0


def test_bimodal_mid_max_boundary_is_strict():
    # group with mid == 0.25 exactly (1 of 4 in band) and high spread.
    r = [0.5, 0.0, 0.0, 1.0]  # in-band: only 0.5 -> mid == 0.25 ; std >= 0.30
    assert float(np.std(r)) >= 0.30
    not_flagged = compute_group_advantage_metrics(
        make_batch(r, centered_adv(r, ["g0"] * 4), ["g0"] * 4),
        bimodal_mid_max=0.25,  # mid < 0.25 is False when mid == 0.25
    )
    flagged = compute_group_advantage_metrics(
        make_batch(r, centered_adv(r, ["g0"] * 4), ["g0"] * 4),
        bimodal_mid_max=0.30,  # mid 0.25 < 0.30 is True
    )
    assert not_flagged["group/bimodal_frac"] == 0.0
    assert flagged["group/bimodal_frac"] == 1.0


# --------------------------------------------------- invariances & non-string uids
def test_row_order_invariance():
    uids = ["g0", "g1", "g0", "g1", "g0", "g1"]
    r = [0.1, 0.9, 0.2, 0.8, 0.3, 0.05]
    adv = centered_adv(r, uids)
    base = compute_group_advantage_metrics(make_batch(r, adv, uids))

    perm = [5, 0, 3, 1, 4, 2]
    r_p = [r[i] for i in perm]
    adv_p = [adv[i] for i in perm]
    uids_p = [uids[i] for i in perm]
    shuffled = compute_group_advantage_metrics(make_batch(r_p, adv_p, uids_p))

    for k in EXPECTED_KEYS:
        assert shuffled[k] == pytest.approx(base[k], abs=1e-6)


def test_integer_uids_group_correctly():
    uids = [0, 0, 1, 1]  # non-string, object-dtype hashables
    r = [0.1, 0.2, 0.3, 0.4]
    out = compute_group_advantage_metrics(make_batch(r, centered_adv(r, uids), uids))
    ref = numpy_reference(r, centered_adv(r, uids), uids)
    for k in EXPECTED_KEYS:
        assert out[k] == pytest.approx(ref[k], abs=1e-6)


# --------------------------------------------------- property / random
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7])
def test_random_matches_numpy_reference_centered(seed):
    rng = np.random.default_rng(seed)
    n_groups = int(rng.integers(2, 6))
    uids, rewards = [], []
    for g in range(n_groups):
        size = int(rng.integers(2, 7))  # every group has >= 2 members
        uids += [f"grp{g}"] * size
        rewards += rng.uniform(0.0, 1.0, size=size).tolist()
    adv = centered_adv(rewards, uids)

    out = compute_group_advantage_metrics(make_batch(rewards, adv, uids))
    ref = numpy_reference(rewards, adv, uids)

    # continuous metrics: float32 vs float64, generous abs tol
    assert out["group/reward_std_mean"] == pytest.approx(ref["group/reward_std_mean"], abs=1e-4)
    assert out["group/advantage_std_mean"] == pytest.approx(ref["group/advantage_std_mean"], abs=1e-4)
    assert out["group/mid_frac_mean"] == pytest.approx(ref["group/mid_frac_mean"], abs=1e-6)


@pytest.mark.parametrize("seed", [11, 21, 31])
def test_random_low_reward_regime_never_degenerate_or_bimodal(seed):
    # rewards in [0, 0.15]: group stds are comfortably in (1e-3, 0.30) and all below the mid band,
    # so degenerate_frac and bimodal_frac are deterministically 0 (no float boundary flakiness).
    rng = np.random.default_rng(seed)
    uids, rewards = [], []
    for g in range(4):
        uids += [f"g{g}"] * 5
        rewards += rng.uniform(0.0, 0.15, size=5).tolist()
    adv = centered_adv(rewards, uids)
    out = compute_group_advantage_metrics(make_batch(rewards, adv, uids))
    assert out["group/degenerate_frac"] == 0.0
    assert out["group/bimodal_frac"] == 0.0
    assert out["group/mid_frac_mean"] == pytest.approx(0.0)


def test_centered_random_advantage_tracks_reward_std():
    # Property: under centered advantages, per-batch mean advantage std == mean reward std.
    rng = np.random.default_rng(123)
    uids, rewards = [], []
    for g in range(5):
        uids += [f"g{g}"] * 4
        rewards += rng.uniform(0.0, 1.0, size=4).tolist()
    out = compute_group_advantage_metrics(
        make_batch(rewards, centered_adv(rewards, uids), uids)
    )
    assert out["group/advantage_std_mean"] == pytest.approx(out["group/reward_std_mean"], abs=1e-5)


def test_zscored_random_advantage_std_is_one():
    # Property: under z-scored advantages (all groups non-degenerate), mean advantage std == 1.
    rng = np.random.default_rng(321)
    uids, rewards = [], []
    for g in range(5):
        uids += [f"g{g}"] * 4
        rewards += rng.uniform(0.0, 1.0, size=4).tolist()
    out = compute_group_advantage_metrics(
        make_batch(rewards, zscored_adv(rewards, uids), uids)
    )
    assert out["group/advantage_std_mean"] == pytest.approx(1.0, abs=1e-4)


def test_metrics_nonnegative_and_fracs_in_unit_interval():
    rng = np.random.default_rng(999)
    uids, rewards = [], []
    for g in range(6):
        uids += [f"g{g}"] * 4
        rewards += rng.uniform(0.0, 1.0, size=4).tolist()
    out = compute_group_advantage_metrics(
        make_batch(rewards, centered_adv(rewards, uids), uids)
    )
    assert out["group/reward_std_mean"] >= 0.0
    assert out["group/advantage_std_mean"] >= 0.0
    for k in ("group/mid_frac_mean", "group/degenerate_frac", "group/bimodal_frac"):
        assert 0.0 <= out[k] <= 1.0
