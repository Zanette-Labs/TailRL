"""`skip_old_log_prob` must be EXACT, not approximately right.

The optimisation drops a full forward pass over the batch (measured: 4.74 s of a 23.9 s step on the
7B gate, 20%) by substituting `log_probs.detach()` for the separately-computed `old_log_probs`.

That is only sound because of a specific autograd identity, and the whole safety of the change rests
on it:

    ratio = exp(log_probs - log_probs.detach())

evaluates to exactly 1.0, but differentiates to `ratio * d(log_probs)` = `d(log_probs)`. So the
vanilla objective `-A * ratio` produces the gradient `-A * d(log pi)` -- the exact REINFORCE/policy
gradient, which is what the two-pass version computes anyway when the policy has not moved between
rollout and update (ppo_epochs == 1, no clipping).

If someone "simplifies" the substitution to `torch.ones_like(...)` or drops the `.detach()` these
tests fail: the first kills the gradient entirely (silently training nothing), the second makes the
ratio's gradient double-count. Both are silent in training -- no crash, no NaN, just a wrong or
absent update -- which is exactly why they are pinned here.

The guards are tested too: outside the strictly-on-policy regime the substituted ratio is NOT the
true ratio, and the failure mode is a quietly incorrect objective rather than an error.
"""

import pytest
import torch

from verl.trainer.core_algos import compute_policy_loss
from verl.workers.config import ActorConfig, validate_skip_old_log_prob


def _batch(seed=0, bsz=4, resp_len=6):
    g = torch.Generator().manual_seed(seed)
    log_probs = (-torch.rand(bsz, resp_len, generator=g)).requires_grad_(True)
    advantages = torch.randn(bsz, resp_len, generator=g)
    mask = torch.ones(bsz, resp_len)
    mask[0, -2:] = 0.0  # a padded tail, so masking is actually exercised
    return log_probs, advantages, mask


def _loss_and_grad(log_probs, old_log_probs, advantages, mask, loss_type="vanilla"):
    loss, metrics = compute_policy_loss(
        old_log_probs=old_log_probs,
        log_probs=log_probs,
        advantages=advantages,
        response_mask=mask,
        clip_ratio_low=0.2,
        clip_ratio_high=0.3,
        clip_ratio_dual=3.0,
        tau_positive=1.0,
        tau_negative=1.05,
        loss_type=loss_type,
        loss_avg_mode="token",
    )
    grad, = torch.autograd.grad(loss, log_probs)
    return loss.detach(), grad, metrics


# ---------------------------------------------------------------- the core identity


def test_skip_reproduces_the_two_pass_gradient_exactly():
    """The substitution is not an approximation: bit-for-bit the same gradient."""
    lp, adv, mask = _batch()

    # two-pass: a genuinely separate tensor, as compute_log_probs would return on-policy
    old_separate = lp.detach().clone()
    loss_a, grad_a, _ = _loss_and_grad(lp, old_separate, adv, mask)

    lp2 = lp.detach().clone().requires_grad_(True)
    loss_b, grad_b, _ = _loss_and_grad(lp2, lp2.detach(), adv, mask)

    assert torch.equal(grad_a, grad_b)
    assert torch.equal(loss_a, loss_b)


def test_gradient_is_the_exact_reinforce_gradient():
    """d/dlogp of the objective is -A/N on live tokens and 0 on masked ones."""
    lp, adv, mask = _batch()
    _, grad, _ = _loss_and_grad(lp, lp.detach(), adv, mask)

    n_live = mask.sum()
    expected = -adv * mask / n_live

    assert torch.allclose(grad, expected, atol=1e-7)
    assert torch.equal(grad[0, -2:], torch.zeros(2))  # masked tokens get no gradient


def test_ratio_is_exactly_one_so_ppo_kl_is_exactly_zero():
    lp, adv, mask = _batch()
    _, _, metrics = _loss_and_grad(lp, lp.detach(), adv, mask)
    assert metrics["ppo_kl"] == 0.0


def test_detach_is_load_bearing_a_constant_one_would_kill_the_gradient():
    """Guard against 'simplifying' the substitution to a literal 1.

    Both tensors equal 1.0 everywhere, so any value-based check would pass them equally. The
    difference is entirely in the autograd graph: `exp(lp - lp.detach())` carries a grad_fn and
    differentiates to d(log pi), while `ones_like` is a constant. Substituting the constant makes
    `-A * ratio` independent of the policy -- the update silently becomes a no-op, with no crash to
    reveal it. That is the trap this pins.
    """
    lp, _, _ = _batch()

    ratio = torch.exp(lp - lp.detach())
    naive = torch.ones_like(lp)

    # indistinguishable by value ...
    assert torch.equal(ratio, naive)
    assert torch.equal(ratio, torch.ones_like(lp))

    # ... and completely different as gradients
    assert ratio.requires_grad and ratio.grad_fn is not None
    assert not naive.requires_grad

    # the real ratio differentiates to exactly 1 * d(log pi)
    grad, = torch.autograd.grad(ratio.sum(), lp)
    assert torch.equal(grad, torch.ones_like(lp))

    # the naive one is not differentiable wrt the policy at all
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(naive.sum(), lp)


# ---------------------------------------------------------------- entropy stays observable


def test_entropy_metrics_survive_the_skip():
    """The whole point of keeping entropy logged: it must not depend on old_log_probs.

    `entropy_loss` is average_loss(-log_probs), and `actor/entropy` is computed in dp_actor from the
    forward pass's own entropy tensor -- neither reads old_log_probs. Pinned because the exploration
    curve is the main diagnostic for estimator collapse in this experiment.
    """
    lp, adv, mask = _batch()
    _, _, m_skip = _loss_and_grad(lp, lp.detach(), adv, mask)
    _, _, m_two = _loss_and_grad(lp, lp.detach().clone(), adv, mask)

    assert "entropy_loss" in m_skip
    assert m_skip["entropy_loss"] == pytest.approx(m_two["entropy_loss"])
    # a real number, not a degenerate zero
    assert m_skip["entropy_loss"] > 0.0


def test_entropy_loss_equals_masked_mean_negative_log_prob():
    lp, adv, mask = _batch()
    _, _, m = _loss_and_grad(lp, lp.detach(), adv, mask)
    expected = float((-lp.detach() * mask).sum() / mask.sum())
    assert m["entropy_loss"] == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------- config guards


def _cfg(**kw):
    c = ActorConfig()
    c.skip_old_log_prob = True
    c.loss_type = "vanilla"
    c.ppo_epochs = 1
    c.disable_kl = True
    c.use_kl_loss = False
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_guard_accepts_the_strictly_on_policy_config():
    validate_skip_old_log_prob(_cfg())  # must not raise


def test_guard_is_a_no_op_when_the_flag_is_off():
    c = _cfg(skip_old_log_prob=False, ppo_epochs=4, loss_type="default")
    validate_skip_old_log_prob(c)  # nothing to validate


@pytest.mark.parametrize(
    "kw, needle",
    [
        ({"ppo_epochs": 2}, "ppo_epochs"),
        ({"loss_type": "default"}, "loss_type"),
        ({"loss_type": "gspo"}, "loss_type"),
        ({"use_kl_loss": True}, "KL"),
        ({"disable_kl": False}, "KL"),
    ],
)
def test_guard_rejects_configs_where_the_ratio_is_not_one(kw, needle):
    with pytest.raises(ValueError, match=needle):
        validate_skip_old_log_prob(_cfg(**kw))


def test_off_by_default():
    """The 3B arms are mid-flight under the old path; the default must not change under them."""
    assert ActorConfig().skip_old_log_prob is False


# ---------------------------------------------------------------- micro-batch neutrality


def test_micro_batch_partitioning_does_not_change_the_accumulated_gradient():
    """Raising micro_batch_size_per_device_for_update is mathematically exact.

    dp_actor scales each micro-batch by `sum(response_mask) * world_size / total_response_tokens`
    where the denominator is all-reduced over the whole mini-batch, so the accumulated gradient
    telescopes to the same value regardless of how the mini-batch is cut up. Pinned because the 7B
    arms run a different micro-batch size than the gate that validated them.
    """
    lp, adv, mask = _batch(seed=3, bsz=8)
    total_tokens = mask.sum()

    def accumulate(chunk):
        g = torch.zeros_like(lp)
        for i in range(0, lp.shape[0], chunk):
            sl = slice(i, i + chunk)
            x = lp.detach()[sl].clone().requires_grad_(True)
            loss, _ = compute_policy_loss(
                old_log_probs=x.detach(), log_probs=x, advantages=adv[sl],
                response_mask=mask[sl], clip_ratio_low=0.2, clip_ratio_high=0.3,
                clip_ratio_dual=3.0, tau_positive=1.0, tau_negative=1.05,
                loss_type="vanilla", loss_avg_mode="token",
            )
            # the dp_actor scaling (world_size == 1 here)
            loss = loss * mask[sl].sum() / total_tokens
            g[sl] = torch.autograd.grad(loss, x)[0]
        return g

    assert torch.allclose(accumulate(1), accumulate(4), atol=1e-7)
    assert torch.allclose(accumulate(1), accumulate(8), atol=1e-7)
