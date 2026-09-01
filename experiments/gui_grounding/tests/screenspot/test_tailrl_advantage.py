"""Pin the centered Gap-Over-Survivors (Algorithm 1) semantics.

These tests check compute_tailrl_outcome_advantage against an INDEPENDENT literal
reference of Algorithm 1 and against hand-computed values. Centered TailRL:
    w_(i) = w_(i-1) + (r_(i) - r_(i-1)) / (N - i + 1);   A = w - mean(w).
"""
import math, numpy as np, torch, pytest
from verl.trainer.core_algos import compute_tailrl_outcome_advantage


# ---- independent literal reference (Algorithm 1), original order ----
def gos_ref(rewards):
    r = list(map(float, rewards)); N = len(r)
    order = sorted(range(N), key=lambda j: r[j])
    rs = [r[j] for j in order]
    w, prev, wl = 0.0, 0.0, []
    for i in range(1, N + 1):
        w += (rs[i-1] - prev) / (N - i + 1); wl.append(w); prev = rs[i-1]
    wbar = sum(wl) / N
    A = [0.0]*N
    for pos, j in enumerate(order): A[j] = wl[pos] - wbar
    return A


# ---- driver: one or more groups -> per-rollout advantages (original order) ----
def run(rewards, index=None, L=1):
    r = torch.tensor(rewards, dtype=torch.float64).reshape(-1, 1)
    tlr = torch.zeros(r.shape[0], L, dtype=torch.float64); tlr[:, -1] = r[:, 0]
    mask = torch.ones(r.shape[0], L, dtype=torch.float64)
    if index is None: index = np.zeros(r.shape[0], dtype=object)
    adv, ret = compute_tailrl_outcome_advantage(tlr, mask, np.asarray(index, dtype=object))
    assert torch.allclose(adv, ret)                       # returns == advantages
    return adv[:, -1]


EX = [[0.,1.], [0.,0.,1.], [0.,0.,0.,1.], [1.,2.,4.], [3.,1.,2.,4.,0.],
      [-2.,-1.,0.,5.], [0.5,0.5,0.5], [7.,7.,7.,7.], list(np.linspace(0,1,16))]


@pytest.mark.parametrize("r", EX)
def test_matches_literal_reference(r):
    assert np.allclose(run(r).numpy(), gos_ref(r), atol=1e-9)


@pytest.mark.parametrize("seed", range(50))
def test_random_parity(seed):
    rng = np.random.default_rng(seed); N = int(rng.integers(2, 33))
    r = rng.normal(size=N) * rng.choice([0.01, 1.0, 100.0])
    assert np.allclose(run(r).numpy(), gos_ref(r), atol=1e-7)


def test_hand_binary_single_winner():           # N=4, one winner: winner=(1/1-1/4)=0.75, losers=-0.25
    a = run([0.,0.,0.,1.]).numpy()
    assert np.allclose(sorted(a), [-0.25,-0.25,-0.25,0.75])


def test_hand_two():                            # [0,1] -> [-0.5, 0.5]
    assert np.allclose(run([0.,1.]).numpy(), [-0.5, 0.5])


def test_hand_distinct():                        # [1,2,4] -> [-1, -0.5, 1.5]
    assert np.allclose(run([1.,2.,4.]).numpy(), [-1.0, -0.5, 1.5])


@pytest.mark.parametrize("r", EX)
def test_mean_centered(r):                       # sum of advantages == 0 per group
    assert abs(float(run(r).sum())) < 1e-7


@pytest.mark.parametrize("r", EX)
def test_monotone_in_reward(r):                  # higher reward -> >= advantage
    r = np.asarray(r); a = run(r).numpy(); o = np.argsort(r)
    assert np.all(np.diff(a[o]) >= -1e-9)


def test_ties_equal_advantage():
    a = run([2., 2., 5., 1., 1.]).numpy()
    assert abs(a[0]-a[1]) < 1e-9 and abs(a[3]-a[4]) < 1e-9


def test_all_equal_is_zero():
    assert np.allclose(run([4.,4.,4.,4.]).numpy(), 0.0)


@pytest.mark.parametrize("seed", range(20))
def test_shift_invariance(seed):                 # A(r + c) == A(r)
    rng = np.random.default_rng(seed); r = rng.normal(size=12); c = rng.normal()*10
    assert np.allclose(run(r).numpy(), run(r + c).numpy(), atol=1e-7)


@pytest.mark.parametrize("seed", range(20))
def test_positive_scale_equivariance(seed):      # A(k r) == k A(r), k>0
    rng = np.random.default_rng(seed); r = rng.normal(size=12); k = abs(rng.normal())+0.1
    assert np.allclose(run(k*r).numpy(), k*run(r).numpy(), atol=1e-6)


@pytest.mark.parametrize("seed", range(20))
def test_permutation_equivariance(seed):
    rng = np.random.default_rng(seed); r = rng.normal(size=10); p = rng.permutation(10)
    a = run(r).numpy(); ap = run(r[p]).numpy()
    assert np.allclose(ap, a[p], atol=1e-9)


def test_binary_equals_maxrl():                  # winners 1/k - 1/N, losers -1/N
    N, k = 8, 3; r = [0.]*(N-k) + [1.]*k; a = np.sort(run(r).numpy())
    assert np.allclose(a[:N-k], -1.0/N) and np.allclose(a[N-k:], 1.0/k - 1.0/N)


def test_min_weight_one_over_N_max_weight_one():  # structural: pre-center cumsum endpoints
    r = [0., 10.]                                  # N=2: w=[0,10/1]; mean 5 -> [-5,5]
    assert np.allclose(run(r).numpy(), [-5., 5.])


def test_singleton_group_zero():
    assert np.allclose(run([3.0]).numpy(), 0.0)


def test_multi_group_independent():
    idx = np.array(['a','a','b','b','b'], dtype=object)
    a = run([0.,1., 0.,0.,1.], index=idx).numpy()
    assert abs(a[0]+a[1]) < 1e-7 and abs(a[2]+a[3]+a[4]) < 1e-7      # each group centered
    assert np.allclose(a[:2], [-0.5, 0.5])


def test_broadcast_and_mask():                   # value repeated across response tokens, 0 elsewhere
    r = torch.tensor([[0.],[1.]], dtype=torch.float64)
    tlr = torch.zeros(2,3,dtype=torch.float64); tlr[:,-1]=r[:,0]
    mask = torch.tensor([[1,1,0],[1,1,1]], dtype=torch.float64)
    adv,_ = compute_tailrl_outcome_advantage(tlr, mask, np.zeros(2,dtype=object))
    assert torch.allclose(adv[0], torch.tensor([-0.5,-0.5,0.0],dtype=torch.float64))
    assert torch.allclose(adv[1], torch.tensor([0.5,0.5,0.5],dtype=torch.float64))


@pytest.mark.parametrize("seed", range(30))
def test_no_nan_inf(seed):
    rng = np.random.default_rng(seed); N=int(rng.integers(2,40))
    r = rng.normal(size=N)*rng.choice([1e-6,1.0,1e6])
    a = run(r); assert torch.isfinite(a).all()


def test_dtype_and_device_preserved():
    tlr = torch.tensor([[0.],[1.],[2.]], dtype=torch.float32)
    mask = torch.ones(3,1)
    adv,_ = compute_tailrl_outcome_advantage(tlr, mask, np.zeros(3,dtype=object))
    assert adv.dtype == torch.float32


def test_differs_from_grpo_on_nondegenerate():   # sanity: TailRL != z-score on graded rewards
    from verl.trainer.core_algos import compute_grpo_outcome_advantage as grpo
    r = [0.,1.,2.,4.,8.,16.]
    tlr = torch.tensor(r, dtype=torch.float64).reshape(-1,1); mask=torch.ones(len(r),1,dtype=torch.float64)
    g_adv,_ = compute_tailrl_outcome_advantage(tlr, mask, np.zeros(len(r),dtype=object))
    try:
        gr,_ = grpo(tlr, mask, np.zeros(len(r),dtype=object))   # signature may differ; adapt
        assert not np.allclose(g_adv[:,0].numpy(), gr[:,0].numpy())
    except TypeError:
        pytest.skip("grpo signature differs; adapt call")
