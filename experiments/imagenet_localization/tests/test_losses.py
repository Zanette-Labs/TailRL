"""Tests for experiments.imagenet_localization.core.losses.

Each test has a docstring stating the invariant being checked and why failure matters.
Hand-computed expected values are derived in comments directly above assertions.

All tests run on CPU and complete well under 10 seconds.

Test count: 27 tests.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from experiments.imagenet_localization.core.losses import (
    HEAD_NAMES,
    factored_sample_log_prob,
    localization_tailrl_population_loss,
    localization_giou_loss,
    localization_mse_loss,
    localization_ordinal_ce_loss,
    localization_cross_entropy_loss,
    ordinal_ce_single_head,
)


# ===========================================================================
# tailrl_population tests (CPU-safe, small K)
# ===========================================================================

def _random_logits(B, K, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {h: torch.randn(B, K, generator=g) for h in HEAD_NAMES}


def test_tailrl_population_runs_and_is_positive():
    """Basic smoke: shape OK, finite, positive. K=5, B=2, M=3, T=10 (small for CPU)."""
    B, K = 2, 5
    logits = _random_logits(B, K)
    gt_boxes = torch.tensor([
        [[0.5, 0.5, 0.4, 0.4], [0.2, 0.3, 0.1, 0.2], [0, 0, 0, 0]],
        [[0.3, 0.7, 0.5, 0.3], [0, 0, 0, 0],          [0, 0, 0, 0]],
    ])
    gt_mask = torch.tensor([[True, True, False], [True, False, False]])
    loss = localization_tailrl_population_loss(
        logits, gt_boxes, gt_mask, K=K, n_thresholds=10,
    )
    assert loss.dim() == 0, "loss must be a scalar"
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    assert loss.item() >= 0.0, f"loss must be nonneg, got {loss}"


def test_tailrl_population_gradient_flows_to_all_heads():
    """Gradient must flow back to every one of the 4 head logits.

    If any head gets zero gradient we've broken the factored joint product.
    """
    B, K = 2, 5
    logits = {h: torch.randn(B, K, requires_grad=True) for h in HEAD_NAMES}
    gt_boxes = torch.tensor([
        [[0.5, 0.5, 0.3, 0.3], [0, 0, 0, 0]],
        [[0.2, 0.3, 0.4, 0.5], [0, 0, 0, 0]],
    ])
    gt_mask = torch.tensor([[True, False], [True, False]])
    loss = localization_tailrl_population_loss(
        logits, gt_boxes, gt_mask, K=K, n_thresholds=10,
    )
    loss.backward()
    for h in HEAD_NAMES:
        g = logits[h].grad
        assert g is not None, f"head {h!r}: no gradient"
        assert g.abs().sum().item() > 0, f"head {h!r}: zero gradient"


def test_tailrl_population_lower_loss_when_policy_matches_gt():
    """Policy concentrating on the correct bins must yield lower loss than uniform.

    At K=5, if GT is at x_c=0.5 (bin 2), y_c=0.5 (bin 2), w=0.4 (bin 1), h=0.4 (bin 1),
    a policy that puts all mass on bin (2,2,1,1) should get near-zero loss
    vs a uniform policy which gets a much larger loss.
    """
    B, K = 1, 5
    gt_boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4], [0, 0, 0, 0]]])
    gt_mask = torch.tensor([[True, False]])

    # Peaked policy on the correct bins
    peaked = {h: torch.full((B, K), -10.0) for h in HEAD_NAMES}
    peaked['x_c'][0, 2] = 10.0
    peaked['y_c'][0, 2] = 10.0
    peaked['w'][0, 1]   = 10.0   # w=0.3 (bin 1 center)
    peaked['h'][0, 1]   = 10.0
    loss_peaked = localization_tailrl_population_loss(
        peaked, gt_boxes, gt_mask, K=K, n_thresholds=10,
    )

    # Uniform policy
    uniform = {h: torch.zeros(B, K) for h in HEAD_NAMES}
    loss_uniform = localization_tailrl_population_loss(
        uniform, gt_boxes, gt_mask, K=K, n_thresholds=10,
    )

    assert loss_peaked.item() < loss_uniform.item() - 0.1, (
        f"peaked loss {loss_peaked.item():.4f} not meaningfully below "
        f"uniform {loss_uniform.item():.4f} — loss does not reward correct bin mass"
    )


def test_tailrl_population_handles_invalid_gt_rows():
    """gt_mask should correctly ignore padded GT rows (all-zeros boxes).

    Loss with a real GT + padding rows should equal loss with just the real GT.
    """
    B, K = 1, 5
    logits = _random_logits(B, K, seed=42)
    gt_boxes_padded = torch.tensor([[[0.4, 0.5, 0.3, 0.3], [0, 0, 0, 0], [0.9, 0.9, 0.1, 0.1]]])
    mask_pad1 = torch.tensor([[True, False, False]])   # only real GT
    mask_pad2 = torch.tensor([[True, False, True]])    # real + garbage at idx 2

    loss_just_real = localization_tailrl_population_loss(
        logits, gt_boxes_padded[:, :1], torch.tensor([[True]]), K=K, n_thresholds=10,
    )
    loss_masked = localization_tailrl_population_loss(
        logits, gt_boxes_padded, mask_pad1, K=K, n_thresholds=10,
    )
    assert abs(loss_just_real.item() - loss_masked.item()) < 1e-4, (
        "Padded invalid rows are being counted despite gt_mask=False"
    )


def test_tailrl_population_clamp_pred_false_matches_baseline():
    """clamp_pred=False must produce bit-identical output to omitting the
    kwarg, so the existing tailrl_population baseline is unaffected."""
    B, K = 2, 5
    logits = _random_logits(B, K, seed=7)
    gt_boxes = torch.tensor([
        [[0.5, 0.5, 0.4, 0.4], [0.2, 0.3, 0.1, 0.2], [0, 0, 0, 0]],
        [[0.3, 0.7, 0.5, 0.3], [0, 0, 0, 0],          [0, 0, 0, 0]],
    ])
    gt_mask = torch.tensor([[True, True, False], [True, False, False]])
    base = localization_tailrl_population_loss(
        logits, gt_boxes, gt_mask, K=K, n_thresholds=10,
    )
    explicit = localization_tailrl_population_loss(
        logits, gt_boxes, gt_mask, K=K, n_thresholds=10, clamp_pred=False,
    )
    assert torch.equal(base, explicit), (
        f"clamp_pred=False must not change behavior; got {base} vs {explicit}"
    )


def test_tailrl_population_clamp_pred_true_finite_grad_and_differs():
    """clamp_pred=True must return finite/positive loss with grad to all heads
    and must differ from clamp_pred=False on logits where the predicted box
    can extend outside [0,1]."""
    B, K = 2, 5
    g = torch.Generator().manual_seed(11)
    logits = {h: torch.randn(B, K, generator=g, requires_grad=True)
              for h in HEAD_NAMES}
    gt_boxes = torch.tensor([
        [[0.5, 0.5, 0.6, 0.6], [0, 0, 0, 0]],
        [[0.2, 0.8, 0.5, 0.3], [0, 0, 0, 0]],
    ])
    gt_mask = torch.tensor([[True, False], [True, False]])

    loss_unclamped = localization_tailrl_population_loss(
        logits, gt_boxes, gt_mask, K=K, n_thresholds=10, clamp_pred=False,
    )
    loss_clamped = localization_tailrl_population_loss(
        logits, gt_boxes, gt_mask, K=K, n_thresholds=10, clamp_pred=True,
    )

    assert torch.isfinite(loss_clamped), f"clamped loss not finite: {loss_clamped}"
    assert loss_clamped.item() >= 0.0, f"clamped loss must be nonneg, got {loss_clamped}"
    assert not torch.allclose(loss_unclamped, loss_clamped), (
        f"clamp_pred=True did not change anything: "
        f"unclamped={loss_unclamped.item()} clamped={loss_clamped.item()}"
    )
    loss_clamped.backward()
    for h in HEAD_NAMES:
        grad = logits[h].grad
        assert grad is not None, f"no grad reached head {h!r}"
        assert torch.isfinite(grad).all(), f"non-finite grad on head {h!r}: {grad}"
        assert grad.abs().sum().item() > 0.0, f"zero grad on head {h!r}"


# ===========================================================================
# Factored log-prob tests
# ===========================================================================


def test_factored_sum_shape_is_BN_not_BN4():
    """Output of factored_sample_log_prob must be (B, N), not (B, N, 4).

    The #1 trap: gathering per-head log-probs without summing leaves a (B, N, 4)
    tensor that silently broadcasts-wrong in all downstream policy-gradient losses.
    A (B, N, 4) shape would make the REINFORCE loss sum over the last dim instead
    of the sample dim, producing a completely wrong gradient signal.
    """
    B, K, N = 2, 4, 5
    torch.manual_seed(0)
    log_probs = {h: torch.log_softmax(torch.randn(B, K), dim=-1) for h in HEAD_NAMES}
    samples = {h: torch.randint(0, K, (B, N)) for h in HEAD_NAMES}

    out = factored_sample_log_prob(log_probs, samples)

    assert out.shape == (B, N), (
        f"Expected shape ({B}, {N}), got {out.shape}. "
        "The function must SUM across heads, not stack them."
    )


def test_factored_sum_equals_sum_of_gather():
    """factored_sample_log_prob must equal the hand-computed sum of four gathers.

    This verifies the actual arithmetic: gather each head's log-prob at the
    sampled index and sum. Any indexing error (e.g., wrong dim, wrong gather
    axis) would produce a different numerical value here.
    """
    B, K, N = 3, 6, 4
    torch.manual_seed(1)
    log_probs = {h: torch.log_softmax(torch.randn(B, K), dim=-1) for h in HEAD_NAMES}
    samples = {h: torch.randint(0, K, (B, N)) for h in HEAD_NAMES}

    # Hand-compute: sum of per-head gather results
    expected = sum(log_probs[h].gather(1, samples[h]) for h in HEAD_NAMES)  # (B, N)

    out = factored_sample_log_prob(log_probs, samples)

    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0)


def test_single_sample_per_image():
    """factored_sample_log_prob with N=1 must return shape (B, 1).

    The single-sample case (N=1) is the common inference path. A shape of (B,)
    instead of (B, 1) would break broadcasting in the policy gradient loss.
    """
    B, K, N = 3, 8, 1
    torch.manual_seed(2)
    log_probs = {h: torch.log_softmax(torch.randn(B, K), dim=-1) for h in HEAD_NAMES}
    samples = {h: torch.randint(0, K, (B, N)) for h in HEAD_NAMES}

    out = factored_sample_log_prob(log_probs, samples)

    assert out.shape == (B, N), (
        f"Expected shape ({B}, {N}), got {out.shape}."
    )


# ===========================================================================
# Supervised loss tests
# ===========================================================================


def test_ordinal_ce_reduces_to_mean_of_4_head_losses():
    """localization_ordinal_ce_loss must equal the mean of 4 per-head ordinal CE losses.

    This checks the aggregation logic: the multi-head function must divide the
    sum of per-head losses by 4.0, not just return the sum. A sum-only return
    would inflate the loss scale by 4× and cause gradient magnitude issues.
    """
    B, K = 5, 8
    torch.manual_seed(3)
    logits = {h: torch.randn(B, K) for h in HEAD_NAMES}
    targets = {h: torch.randint(0, K, (B,)) for h in HEAD_NAMES}

    multi = localization_ordinal_ce_loss(logits, targets, K)
    per_head_mean = sum(
        ordinal_ce_single_head(logits[h], targets[h], K) for h in HEAD_NAMES
    ) / 4.0

    torch.testing.assert_close(multi, per_head_mean, atol=1e-6, rtol=0)


def test_perfectly_peaked_logits_small_loss():
    """Logits [100, 0, 0, 0] with target 0 must give near-zero ordinal CE loss.

    When the model is completely confident and correct, the softmax places
    virtually all mass on bin 0, making W_d ≈ 1 for all d, so -log W_d ≈ 0.
    A non-zero loss here would indicate numerical errors in the log computation.
    """
    B, K = 4, 4
    # logits[0] = 100, rest = 0; target = 0 for every sample in every head
    logits = {h: torch.tensor([[100.0, 0.0, 0.0, 0.0]] * B) for h in HEAD_NAMES}
    targets = {h: torch.zeros(B, dtype=torch.long) for h in HEAD_NAMES}

    loss = localization_ordinal_ce_loss(logits, targets, K)

    assert loss.item() < 1e-6, (
        f"Expected near-zero loss for perfectly peaked logits, got {loss.item():.2e}"
    )


def test_uniform_logits_finite_loss():
    """Uniform logits must produce a finite, positive ordinal CE loss.

    With uniform logits (all zeros), softmax gives equal probability to every
    bin. The loss must be finite (not Inf/NaN) and positive. Inf would indicate
    a -log(0) numerical issue; NaN would indicate divide-by-zero somewhere.
    """
    B, K = 4, 4
    logits = {h: torch.zeros(B, K) for h in HEAD_NAMES}
    targets = {h: torch.ones(B, dtype=torch.long) for h in HEAD_NAMES}  # target = 1

    loss = localization_ordinal_ce_loss(logits, targets, K)

    assert torch.isfinite(loss), f"Expected finite loss, got {loss.item()}"
    assert loss.item() > 0.0, f"Expected positive loss, got {loss.item()}"


def test_cross_entropy_equals_mean_of_f_cross_entropy():
    """localization_cross_entropy_loss must equal mean([F.cross_entropy(logits[h], targets[h])]).

    This verifies the aggregation formula: sum divided by 4. If the function
    returns the raw sum, the loss would be 4× too large and destabilise training.
    """
    B, K = 6, 10
    torch.manual_seed(4)
    logits = {h: torch.randn(B, K) for h in HEAD_NAMES}
    targets = {h: torch.randint(0, K, (B,)) for h in HEAD_NAMES}

    multi = localization_cross_entropy_loss(logits, targets)
    expected = sum(F.cross_entropy(logits[h], targets[h]) for h in HEAD_NAMES) / 4.0

    torch.testing.assert_close(multi, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ordinal_ce_equals_cross_entropy_at_K2(seed):
    """At K=2, ordinal_ce_single_head must equal F.cross_entropy exactly.

    At K=2 there is only one threshold (d=0). W_0(b, y) = probs[b, y].
    So the ordinal CE = mean_b[-log probs[b, y]] = F.cross_entropy.
    The (K-1)=1 normalization factor is a no-op.

    This is the binary-recovery identity — if it fails, the ordinal loss is
    not a proper generalization of standard CE, and the K=2 baseline is invalid.
    """
    torch.manual_seed(seed * 17 + 5)
    B, K = 8, 2
    logits = torch.randn(B, K)
    targets = torch.randint(0, K, (B,))

    ordinal_loss = ordinal_ce_single_head(logits, targets, K)
    plain_loss = F.cross_entropy(logits, targets)

    torch.testing.assert_close(ordinal_loss, plain_loss, atol=1e-5, rtol=0)


# ===========================================================================
# MSE baseline tests
# ===========================================================================


def test_mse_zero_when_pred_matches_target():
    """localization_mse_loss must be ~0 when pred exactly equals gt_primary_box.

    If the predicted box is identical to the GT box, the MSE is defined to be 0.
    Any non-zero value indicates an implementation error (e.g., computing loss
    against the wrong tensor or adding an unintended offset).
    """
    B = 5
    torch.manual_seed(6)
    gt = torch.rand(B, 4)
    pred = gt.clone()

    loss = localization_mse_loss(pred, gt)

    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-7, rtol=0)


def test_mse_positive_when_pred_differs():
    """localization_mse_loss must be strictly positive when pred != gt_primary_box.

    MSE is only zero when inputs are identical; any difference must produce a
    strictly positive value. A zero loss on mismatched inputs would silently
    suppress all gradient updates to the regression head.
    """
    B = 5
    torch.manual_seed(7)
    gt = torch.rand(B, 4)
    pred = gt + 0.1  # perturb by a fixed offset

    loss = localization_mse_loss(pred, gt)

    assert loss.item() > 0.0, (
        f"Expected positive MSE for perturbed predictions, got {loss.item()}"
    )


# ===========================================================================
# GIoU tests (Rezatofighi et al., CVPR 2019)
# ===========================================================================


def _giou_loss_1v1(pred, gt, dtype=torch.float64):
    """L_GIoU for a single (pred, gt) pair, through the public loss API.

    `pred` and `gt` are 4-lists of xywh floats. Returns a Python float.
    B = 1, M = 1, one valid GT — so the mean and the matcher are both trivial
    and the returned value is exactly 1 - GIoU(pred, gt).
    """
    p = torch.tensor([pred], dtype=dtype)                # (1, 4)
    g = torch.tensor([[gt]], dtype=dtype)                # (1, 1, 4)
    m = torch.ones(1, 1, dtype=torch.bool)
    return localization_giou_loss(p, g, m).item()


def _plain_iou(pred, gt):
    """Differentiable IoU between two (B, 4) xywh tensors — the naive baseline.

    Used only to contrast GIoU's behaviour against `1 - IoU` in the
    zero-overlap regime; deliberately written out here rather than imported so
    the contrast test does not depend on any production code path.
    """
    px1, py1 = pred[:, 0] - pred[:, 2] / 2, pred[:, 1] - pred[:, 3] / 2
    px2, py2 = pred[:, 0] + pred[:, 2] / 2, pred[:, 1] + pred[:, 3] / 2
    gx1, gy1 = gt[:, 0] - gt[:, 2] / 2, gt[:, 1] - gt[:, 3] / 2
    gx2, gy2 = gt[:, 0] + gt[:, 2] / 2, gt[:, 1] + gt[:, 3] / 2
    inter = (
        (torch.minimum(px2, gx2) - torch.maximum(px1, gx1)).clamp(min=0.0)
        * (torch.minimum(py2, gy2) - torch.maximum(py1, gy1)).clamp(min=0.0)
    )
    union = pred[:, 2] * pred[:, 3] + gt[:, 2] * gt[:, 3] - inter
    return inter / union.clamp(min=1e-10)


def test_giou_identical_boxes_zero_loss():
    """L_GIoU must be exactly 0 when pred equals the GT box.

    Hand-computed, pred = gt = [0.3, 0.4, 0.2, 0.5]:
      corners x in [0.2, 0.4], y in [0.15, 0.65]
      A_p = A_g = 0.2 * 0.5 = 0.1
      I = 0.1 ; U = 0.1 + 0.1 - 0.1 = 0.1 ; IoU = 1.0
      C = 0.2 * 0.5 = 0.1 -> penalty = (0.1 - 0.1) / 0.1 = 0
      GIoU = 1.0 - 0 = 1.0 ; L = 1 - 1.0 = 0.0
    A non-zero loss at a perfect prediction means the optimum of the arm is not
    the GT box, so the regressor would be pulled away from the right answer.
    """
    loss = _giou_loss_1v1([0.3, 0.4, 0.2, 0.5], [0.3, 0.4, 0.2, 0.5])

    # GIoU = 1.0 -> L = 0.0
    assert abs(loss - 0.0) < 1e-12, f"expected L=0 for identical boxes, got {loss}"


def test_giou_edge_touching_equals_iou_hand_computed():
    """Edge-to-edge boxes with equal y-extent give GIoU == IoU == 0, so L == 1.

    Hand-computed, pred = [0.25, 0.5, 0.5, 0.5], gt = [0.75, 0.5, 0.5, 0.5]:
      pred corners x in [0.0, 0.5], y in [0.25, 0.75]
      gt   corners x in [0.5, 1.0], y in [0.25, 0.75]
      iw = min(0.5, 1.0) - max(0.0, 0.5) = 0 -> I = 0
      A_p = A_g = 0.25 ; U = 0.5 ; IoU = 0
      C = (1.0 - 0.0) * (0.75 - 0.25) = 1.0 * 0.5 = 0.5 -> penalty = 0
      GIoU = 0 ; L = 1.0
    This is the canonical GIoU == IoU witness at zero overlap (the union
    exactly fills the enclosing box). If the enclosing-box term were computed
    from the wrong corners, C != U here and the value would drift off 1.0.
    """
    loss = _giou_loss_1v1([0.25, 0.5, 0.5, 0.5], [0.75, 0.5, 0.5, 0.5])

    # C == U == 0.5 -> penalty 0 -> GIoU = IoU = 0 -> L = 1.0
    assert abs(loss - 1.0) < 1e-12, f"expected L=1.0 for abutting boxes, got {loss}"


def test_giou_far_disjoint_hand_computed():
    """Fully disjoint corner-to-corner boxes must give L = 1.92, not 1.0.

    Hand-computed, pred = [0.1, 0.1, 0.2, 0.2], gt = [0.9, 0.9, 0.2, 0.2]:
      pred corners x, y in [0.0, 0.2] ; gt corners x, y in [0.8, 1.0]
      iw = 0.2 - 0.8 = -0.6 -> clamped to 0 -> I = 0
      A_p = A_g = 0.04 ; U = 0.08 ; IoU = 0
      C = 1.0 * 1.0 = 1.0 -> penalty = (1.0 - 0.08) / 1.0 = 0.92
      GIoU = 0 - 0.92 = -0.92 ; L = 1.92 = 48/25
    A result of exactly 1.0 here would mean the enclosing-box penalty is
    missing entirely and the arm has silently degenerated to `1 - IoU`.
    """
    loss = _giou_loss_1v1([0.1, 0.1, 0.2, 0.2], [0.9, 0.9, 0.2, 0.2])

    # 1 - (-23/25) = 48/25 = 1.92
    assert abs(loss - 48.0 / 25.0) < 1e-12, f"expected L=1.92, got {loss}"


def test_giou_nested_boxes_hand_computed():
    """A prediction fully inside the GT gives GIoU == IoU (C == U), L = 8/9.

    Hand-computed, pred = [0.5, 0.5, 0.2, 0.2], gt = [0.5, 0.5, 0.6, 0.6]:
      pred corners x, y in [0.4, 0.6] ; gt corners x, y in [0.2, 0.8]
      I = 0.2 * 0.2 = 0.04 (pred entirely contained)
      A_p = 0.04, A_g = 0.36 ; U = 0.36 ; IoU = 0.04 / 0.36 = 1/9
      C = 0.36 (the enclosing box IS the gt box) -> penalty = 0
      GIoU = 1/9 ; L = 8/9 = 0.888888...
    Nesting is the second GIoU == IoU witness; together with the abutting case
    it pins down the exact equality condition area(C) == area(U).
    """
    loss = _giou_loss_1v1([0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6])

    # 1 - 1/9 = 8/9
    assert abs(loss - 8.0 / 9.0) < 1e-12, f"expected L=8/9, got {loss}"


def test_giou_loss_is_scalar_finite_and_in_range():
    """L_GIoU must be a finite scalar in [0, 2] for arbitrary random boxes.

    GIoU = IoU + U/C - 1 with IoU in [0, 1] and U/C in [0, 1], so GIoU is in
    [-1, 1] and L = 1 - GIoU is in [0, 2]. A value outside that range means a
    sign error or an unguarded division, and would make the loss scale
    incomparable to the other regression arms.
    """
    torch.manual_seed(101)
    B, M = 16, 4
    pred = torch.rand(B, 4)
    gt_boxes = torch.rand(B, M, 4)
    gt_mask = torch.rand(B, M) > 0.4
    gt_mask[:, 0] = True   # guarantee at least one valid GT per row

    loss = localization_giou_loss(pred, gt_boxes, gt_mask)

    assert loss.dim() == 0, f"loss must be a scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    assert 0.0 <= loss.item() <= 2.0, f"loss must lie in [0, 2], got {loss.item()}"


def test_giou_never_exceeds_iou():
    """GIoU <= IoU always, because C >= U makes the penalty non-negative.

    This is the defining property of the metric. If it is ever violated the
    enclosing box is being computed smaller than the union, which would make
    the loss reward separation instead of penalising it.
    """
    torch.manual_seed(102)
    n = 200
    pred = torch.rand(n, 4, dtype=torch.float64)
    gt = torch.rand(n, 4, dtype=torch.float64)

    mask = torch.ones(n, 1, dtype=torch.bool)
    giou = 1.0 - torch.stack([
        localization_giou_loss(pred[i:i + 1], gt[i:i + 1].unsqueeze(1), mask[i:i + 1])
        for i in range(n)
    ])
    iou = _plain_iou(pred, gt)

    assert bool((giou <= iou + 1e-9).all()), (
        f"GIoU exceeded IoU; max violation = {(giou - iou).max().item()}"
    )
    assert bool((giou >= -1.0 - 1e-9).all()), f"GIoU below -1: {giou.min().item()}"


def test_giou_gradient_nonzero_for_disjoint_boxes_unlike_iou():
    """Disjoint boxes must still yield a non-zero gradient on all 4 coords.

    This is the entire raison d'etre of GIoU. At pred = [0.1, 0.1, 0.2, 0.2]
    vs gt = [0.9, 0.9, 0.2, 0.2] the intersection is identically 0 on an open
    neighbourhood, so `1 - IoU` has exactly zero gradient and a regressor that
    misses the object never learns to move. With I = 0 the GIoU loss reduces to
    L = 2 - U/C with U = w*h + 0.04 and C = (1 - cx + w/2)(1 - cy + h/2);
    at (0.1, 0.1, 0.2, 0.2) we have C = 1, U = 0.08, dC/dcx = dC/dcy = -1,
    dC/dw = dC/dh = 0.5, dU/dw = dU/dh = 0.2, giving
      dL/dcx = dL/dcy = -(0 - 0.08 * -1) = -0.08
      dL/dw  = dL/dh  = -(0.2 - 0.08 * 0.5) = -0.16
    All negative, so descent increases cx, cy (moves pred toward the GT) and
    increases w, h (grows the box to close the gap).
    """
    gt_boxes = torch.tensor([[[0.9, 0.9, 0.2, 0.2]]], dtype=torch.float64)
    gt_mask = torch.ones(1, 1, dtype=torch.bool)

    pred_g = torch.tensor([[0.1, 0.1, 0.2, 0.2]], dtype=torch.float64,
                          requires_grad=True)
    localization_giou_loss(pred_g, gt_boxes, gt_mask).backward()

    pred_i = torch.tensor([[0.1, 0.1, 0.2, 0.2]], dtype=torch.float64,
                          requires_grad=True)
    (1.0 - _plain_iou(pred_i, gt_boxes[:, 0, :])).sum().backward()

    assert pred_g.grad is not None, "no gradient reached pred"
    assert torch.isfinite(pred_g.grad).all(), f"non-finite grad: {pred_g.grad}"
    assert pred_g.grad.abs().max().item() > 0.0, (
        "GIoU loss gave zero gradient on disjoint boxes — it has degenerated "
        "to a plain IoU loss"
    )
    assert pred_i.grad.abs().max().item() == 0.0, (
        f"the 1-IoU contrast baseline should have zero grad here, got {pred_i.grad}"
    )

    # dL/d(cx, cy, w, h) = (-0.08, -0.08, -0.16, -0.16)
    expected = torch.tensor([[-0.08, -0.08, -0.16, -0.16]], dtype=torch.float64)
    torch.testing.assert_close(pred_g.grad, expected, atol=1e-12, rtol=0)


def test_giou_padded_gt_slot_never_selected():
    """A padded GT must never win the argmax, even at a perfect GIoU of 1.

    Slot 1 is an exact copy of the prediction (GIoU = 1, the best possible
    match) but is masked out; slot 0 is the only valid GT. If the mask is
    ignored the loss collapses to 0 and the arm trains on padding instead of
    real boxes.
    """
    pred = torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float64)
    gt_boxes = torch.tensor([[
        [0.5, 0.5, 0.6, 0.6],   # valid — nested case, L = 8/9
        [0.5, 0.5, 0.2, 0.2],   # padding — identical to pred, GIoU = 1
    ]], dtype=torch.float64)
    gt_mask = torch.tensor([[True, False]])

    loss = localization_giou_loss(pred, gt_boxes, gt_mask)

    # Must equal the valid-GT closed form 8/9, not 0.0.
    assert abs(loss.item() - 8.0 / 9.0) < 1e-12, (
        f"expected the valid GT (L=8/9), got {loss.item()} — padded slot won "
        "the argmax"
    )


def test_giou_argmax_selects_nearest_gt_where_argmax_iou_ties():
    """With two zero-IoU GTs, argmax-GIoU must pick the nearer one.

    pred = [0.1, 0.1, 0.2, 0.2] overlaps neither GT, so every IoU is exactly 0
    and argmax-IoU ties, silently falling back to slot 0 (the far box).
      slot 0 = [0.9, 0.9, 0.2, 0.2]  -> C = 1.0,  U = 0.08, GIoU = -0.92
      slot 1 = [0.35, 0.1, 0.2, 0.2] -> corners x in [0.25, 0.45], y in [0, 0.2]
                                        I = 0, U = 0.08, C = 0.45 * 0.2 = 0.09
                                        penalty = 0.01/0.09 = 1/9, GIoU = -1/9
    GIoU is strictly ordered at zero overlap, so slot 1 wins and L = 10/9.
    This single case is the whole justification for matching on GIoU rather
    than IoU; a value of 1.92 means the matcher fell back to the far GT.
    """
    pred = torch.tensor([[0.1, 0.1, 0.2, 0.2]], dtype=torch.float64)
    gt_boxes = torch.tensor([[
        [0.9, 0.9, 0.2, 0.2],
        [0.35, 0.1, 0.2, 0.2],
    ]], dtype=torch.float64)
    gt_mask = torch.tensor([[True, True]])

    loss = localization_giou_loss(pred, gt_boxes, gt_mask)

    # 1 - (-1/9) = 10/9 = 1.111...  (argmax-IoU would have given 48/25 = 1.92)
    assert abs(loss.item() - 10.0 / 9.0) < 1e-12, (
        f"expected L=10/9 from the near GT, got {loss.item()}"
    )


def test_giou_loss_strictly_increases_with_separation():
    """L must keep growing after the boxes separate, where 1-IoU is pinned at 1.

    gt = [0.5, 0.5, 0.2, 0.2], pred = [0.5 - t, 0.5, 0.2, 0.2]. Exact values:
      t = 0.00 -> I=0.04 U=0.04 C=0.04 -> L = 0
      t = 0.05 -> I=0.03 U=0.05 C=0.05 -> L = 0.4
      t = 0.10 -> I=0.02 U=0.06 C=0.06 -> L = 2/3
      t = 0.20 -> I=0.00 U=0.08 C=0.08 -> L = 1.0
      t = 0.30 -> I=0.00 U=0.08 C=0.10 -> L = 1.2
      t = 0.40 -> I=0.00 U=0.08 C=0.12 -> L = 4/3
    Beyond t = 0.2 the boxes are disjoint and `1 - IoU` is constant at 1.0, so
    a flat tail here would mean the loss carries no information about how badly
    a miss missed.
    """
    ts = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
    expected = [0.0, 0.4, 2.0 / 3.0, 1.0, 1.2, 4.0 / 3.0]

    losses = [_giou_loss_1v1([0.5 - t, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2])
              for t in ts]

    for t, got, want in zip(ts, losses, expected):
        assert abs(got - want) < 1e-12, f"t={t}: expected L={want}, got {got}"
    for a, b in zip(losses, losses[1:]):
        assert b > a, f"loss not strictly increasing with separation: {losses}"
    # Disjoint tail (t >= 0.2): 1 - IoU would be exactly 1.0 at all three.
    assert losses[5] > losses[4] > 1.0 + 1e-6, (
        f"loss is flat once the boxes are disjoint: {losses[3:]}"
    )


def test_giou_is_symmetric_in_its_two_boxes():
    """GIoU(A, B) == GIoU(B, A) — every primitive (I, U, C) is symmetric.

    An asymmetry would mean pred and GT enter the formula differently, which
    for a regression loss shows up as a systematic bias toward predicting boxes
    larger (or smaller) than the target.
    """
    torch.manual_seed(103)
    for _ in range(20):
        a = torch.rand(4, dtype=torch.float64).tolist()
        b = torch.rand(4, dtype=torch.float64).tolist()
        forward = _giou_loss_1v1(a, b)
        backward = _giou_loss_1v1(b, a)
        assert abs(forward - backward) < 1e-12, (
            f"asymmetric GIoU: L(a,b)={forward} vs L(b,a)={backward}"
        )


def test_giou_degenerate_boxes_produce_no_nan_or_inf():
    """Zero-area predictions and all-zero padded GT rows must not produce NaN.

    Both divisions can hit 0/0: U = 0 when both boxes have zero area, and
    C = 0 when they are additionally coincident (pred and a padded GT both
    exactly [0, 0, 0, 0]). A NaN here poisons the whole batch gradient, and
    `torch.where(mask, f(x), const)` still backprops NaN from the discarded
    branch, so the guards must live inside the division itself.
    """
    pred = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],     # exactly degenerate, coincident with padding
        [0.5, 0.5, 1e-12, 1e-12],  # near-degenerate
    ], dtype=torch.float64, requires_grad=True)
    gt_boxes = torch.tensor([
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        [[0.4, 0.4, 0.2, 0.2], [0.0, 0.0, 0.0, 0.0]],
    ], dtype=torch.float64)
    gt_mask = torch.tensor([[True, False], [True, False]])

    loss = localization_giou_loss(pred, gt_boxes, gt_mask)
    loss.backward()

    assert torch.isfinite(loss), f"loss not finite on degenerate boxes: {loss}"
    assert torch.isfinite(pred.grad).all(), (
        f"non-finite gradient on degenerate boxes: {pred.grad}"
    )


def test_giou_rows_with_no_valid_gt_are_excluded_from_the_mean():
    """Images whose gt_mask is all-False must not contribute to the loss.

    argmax over an all-sentinel row returns slot 0, i.e. padding, so including
    such a row would regress the prediction toward a zero box at the origin.
    Row 0 here is the nested case (L = 8/9) and row 1 has no GT at all, so the
    batch loss must be exactly 8/9 — not the 2-row average.
    """
    pred = torch.tensor([
        [0.5, 0.5, 0.2, 0.2],
        [0.5, 0.5, 0.2, 0.2],
    ], dtype=torch.float64)
    gt_boxes = torch.tensor([
        [[0.5, 0.5, 0.6, 0.6], [0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    ], dtype=torch.float64)
    gt_mask = torch.tensor([[True, False], [False, False]])

    loss = localization_giou_loss(pred, gt_boxes, gt_mask)

    # Only row 0 counts: 1 - 1/9 = 8/9
    assert abs(loss.item() - 8.0 / 9.0) < 1e-12, (
        f"expected 8/9 from the single valid row, got {loss.item()}"
    )
