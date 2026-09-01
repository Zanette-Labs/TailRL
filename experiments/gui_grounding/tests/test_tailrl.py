# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit checks for the TailRL (Gap-Over-Survivors) advantage estimator.

Run in-container:  python3 tests/test_gos.py   (or: pytest tests/test_gos.py)
Checks: registration, mean-zero, monotonicity, a hand-computed continuous case, and the
binary-reward recovery of MaxRL (Prop 6.1): successes -> 1/K - meanA, failures -> -meanA
(meanA = 1/N). TailRL is magnitude-preserving: the per-gap weight is 1/S, not N/S.
"""
import numpy as np
import torch

from verl.trainer.core_algos import (
    ADV_ESTIMATOR_MAP,
    AdvantageEstimator,
    compute_tailrl_outcome_advantage,
)


def gos_adv(rewards, uids=None):
    """Return per-rollout TailRL advantages for a flat list of scalar rewards."""
    tlr = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)  # (bs, 1) outcome reward
    mask = torch.ones_like(tlr)
    if uids is None:
        uids = ["g"] * len(rewards)
    index = np.array(uids)
    returns, returns2 = compute_tailrl_outcome_advantage(tlr, mask, index)
    assert torch.equal(returns, returns2), "TailRL must return (advantages, returns) equal"
    return returns[:, 0].to(torch.float64).numpy()


def test_registered():
    assert AdvantageEstimator.TAILRL.value == "tailrl"
    assert "tailrl" in ADV_ESTIMATOR_MAP


def test_continuous_closed_form():
    # Hand-computed: rewards [0, 0.3, 0.7, 1.0], N=4. Survivor count S runs N..1, so the per-gap
    # weight is 1/S (NOT N/S -- TailRL is magnitude-preserving, no N scaling; see core_algos docstring).
    #   gaps=[0,.3,.4,.3]; inv_survival=[1/4,1/3,1/2,1]; gap*inv=[0,.1,.2,.3]
    #   cumsum=[0,.1,.3,.6]; mean=0.25 -> centered=[-0.25,-0.15,0.05,0.35]
    adv = gos_adv([0.0, 0.3, 0.7, 1.0])
    expected = np.array([-0.25, -0.15, 0.05, 0.35])
    assert np.allclose(adv, expected, atol=1e-5), (adv, expected)


def test_mean_zero():
    for rewards in ([0.1, 0.4, 0.4, 0.9, 1.0], list(np.linspace(0, 1, 16))):
        adv = gos_adv(rewards)
        assert abs(adv.sum()) < 1e-6, adv.sum()


def test_monotone_and_ties():
    rewards = [0.2, 0.2, 0.5, 0.9, 0.9, 1.0]
    adv = gos_adv(rewards)
    order = np.argsort(rewards)
    sorted_adv = adv[order]
    # non-decreasing in reward
    assert np.all(np.diff(sorted_adv) >= -1e-9), sorted_adv
    # equal rewards -> equal advantage
    assert abs(adv[0] - adv[1]) < 1e-9 and abs(adv[3] - adv[4]) < 1e-9, adv


def test_binary_recovery_prop61():
    # Binary group: K successes (r=1), N-K failures (r=0). TailRL weight w(1)=int_0^1 dtau/S = 1/K
    # (S=K on [0,1)), w(0)=0. meanA=1/N -> centered MaxRL: success=1/K-1/N, failure=-1/N (Prop 6.1).
    for n, k in [(8, 3), (16, 1), (16, 8), (4, 2)]:
        rewards = [1.0] * k + [0.0] * (n - k)
        adv = gos_adv(rewards)
        succ = adv[:k]
        fail = adv[k:]
        assert np.allclose(succ, 1.0 / k - 1.0 / n, atol=1e-5), (n, k, succ)
        assert np.allclose(fail, -1.0 / n, atol=1e-5), (n, k, fail)
        assert abs(adv.sum()) < 1e-6


def test_multi_group_independent():
    # Two groups, different uids -> computed independently.
    rewards = [0.0, 1.0, 0.0, 1.0]
    uids = ["a", "a", "b", "b"]
    adv = gos_adv(rewards, uids)
    # each group is binary K=1,N=2 -> success 1/K-1/N=0.5, failure -1/N=-0.5
    assert np.allclose(adv, [-0.5, 0.5, -0.5, 0.5], atol=1e-5), adv


def test_degenerate_all_equal():
    # All-equal rewards -> all advantages 0 (mean-centered constant).
    adv = gos_adv([0.5, 0.5, 0.5, 0.5])
    assert np.allclose(adv, 0.0, atol=1e-7), adv


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} TailRL unit checks passed.")
    raise SystemExit(1 if failed else 0)
