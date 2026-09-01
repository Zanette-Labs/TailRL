"""Pin the no-clip `vanilla` policy-gradient loss added to compute_policy_loss.

vanilla: pg_loss = average_loss(-A * ratio, mask, mode); ratio = exp(clamp(logp - old, -20, 20)).
NO clipping (distinguishes it from `default`); pg_clipfrac_* and entropy_loss reported as 0.
"""
import torch, pytest
from verl.trainer.core_algos import compute_policy_loss


def vanilla(old, new, adv, mask, mode="token"):
    return compute_policy_loss(
        old, new, adv, mask,
        clip_ratio_low=0.2, clip_ratio_high=0.2, clip_ratio_dual=3.0,
        tau_positive=1.0, tau_negative=1.0,
        loss_type="vanilla", loss_avg_mode=mode,
    )


def default(old, new, adv, mask, mode="token"):
    return compute_policy_loss(
        old, new, adv, mask,
        clip_ratio_low=0.2, clip_ratio_high=0.2, clip_ratio_dual=3.0,
        tau_positive=1.0, tau_negative=1.0,
        loss_type="default", loss_avg_mode=mode,
    )


def test_on_policy_equals_neg_mean_advantage():     # ratio==1 => loss == -masked_mean(A)
    torch.manual_seed(0)
    old = torch.randn(2, 3); new = old.clone(); adv = torch.randn(2, 3); mask = torch.ones(2, 3)
    loss, _ = vanilla(old, new, adv, mask, "token")
    expected = -(adv * mask).sum() / mask.sum()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_metrics_clipfrac_and_entropy_zero():
    old = torch.zeros(2, 3); new = torch.zeros(2, 3); adv = torch.randn(2, 3); mask = torch.ones(2, 3)
    _, m = vanilla(old, new, adv, mask)
    assert m["pg_clipfrac_higher"] == 0.0
    assert m["pg_clipfrac_lower"] == 0.0
    assert m["entropy_loss"] == 0.0
    assert abs(m["ppo_kl"]) < 1e-6                   # on-policy => zero KL
    assert set(m.keys()) == {"pg_clipfrac_higher", "pg_clipfrac_lower", "ppo_kl", "entropy_loss"}


def test_gradient_flows_finite():
    old = torch.zeros(2, 3); new = torch.zeros(2, 3, requires_grad=True)
    adv = torch.randn(2, 3); mask = torch.ones(2, 3)
    loss, _ = vanilla(old, new, adv, mask)
    loss.backward()
    assert new.grad is not None and torch.isfinite(new.grad).all()


def test_masked_tokens_do_not_contribute():
    torch.manual_seed(1)
    old = torch.randn(2, 3); new1 = torch.randn(2, 3); adv = torch.randn(2, 3)
    mask = torch.tensor([[1., 1., 0.], [1., 0., 1.]])
    new2 = new1.clone(); new2[0, 2] += 5.0; new2[1, 1] -= 3.0    # perturb only masked-out positions
    l1, _ = vanilla(old, new1, adv, mask)
    l2, _ = vanilla(old, new2, adv, mask)
    assert torch.allclose(l1, l2, atol=1e-6)


def test_token_vs_seq_hand_values():
    old = torch.zeros(2, 3); new = torch.zeros(2, 3)          # ratio == 1
    adv = torch.tensor([[1., 1., 1.], [2., 2., 2.]])
    mask = torch.tensor([[1., 1., 0.], [1., 1., 1.]])
    lt, _ = vanilla(old, new, adv, mask, "token")            # -(1+1+2+2+2)/5 = -1.6
    ls, _ = vanilla(old, new, adv, mask, "seq")              # mean(-(2)/2, -(6)/3) = mean(-1,-2) = -1.5
    assert torch.allclose(lt, torch.tensor(-1.6), atol=1e-6)
    assert torch.allclose(ls, torch.tensor(-1.5), atol=1e-6)


def test_positive_advantage_pushes_logprob_up():        # d loss / d logprob = -A < 0 at ratio 1
    adv = torch.ones(1, 1); old = torch.zeros(1, 1)
    new = torch.zeros(1, 1, requires_grad=True); mask = torch.ones(1, 1)
    loss, _ = vanilla(old, new, adv, mask)
    loss.backward()
    assert new.grad.item() < 0


def test_no_clipping_differs_from_default():
    old = torch.zeros(2, 3); new = torch.full((2, 3), 0.5)   # ratio = exp(0.5) ~ 1.6487 > 1+clip_high
    adv = torch.ones(2, 3); mask = torch.ones(2, 3)
    ratio = torch.exp(torch.clamp(new - old, -20.0, 20.0))
    lv, _ = vanilla(old, new, adv, mask, "token")
    expected = (-(adv * ratio) * mask).sum() / mask.sum()    # unclamped advantage term
    assert torch.allclose(lv, expected, atol=1e-6)
    ld, _ = default(old, new, adv, mask, "token")            # default clips ratio to 1.2 for A>0
    assert not torch.allclose(lv, ld, atol=1e-4)


@pytest.mark.parametrize("gap", [1000.0, -1000.0])
def test_nan_safety_on_extreme_logprob_gaps(gap):
    old = torch.zeros(1, 2); new = torch.full((1, 2), gap)
    adv = torch.ones(1, 2); mask = torch.ones(1, 2)
    loss, m = vanilla(old, new, adv, mask)
    assert torch.isfinite(loss)
    assert all(v == v for v in m.values())                   # no NaN metrics
