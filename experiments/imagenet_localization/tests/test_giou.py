"""Tests for the GIoU-family regression arms.

`tests/test_losses.py` already pins down `localization_giou_loss` itself (the
argmax-GIoU-matched arm). This file covers what that one does not:

  - the worked numeric cases from the GIoU baseline spec, in spec coordinates
  - an independent oracle cross-check against torchvision
  - the corner-conversion helpers
  - the five arms added alongside `giou`: the two matcher variants and the
    three DETR-style L1 + (1 - GIoU) combinations
  - end-to-end wiring (argparse choices, regressor selection, train dispatch)
    and a short overfitting smoke test through the real supervised epoch loop

CPU-safe throughout; the smoke test freezes the ResNet-50 backbone so the
20-step overfit stays quick.
"""

from __future__ import annotations

import pytest
import torch

from experiments.imagenet_localization.core.iou import corners_to_xywh, xywh_to_corners
from experiments.imagenet_localization.core.losses import (
    _giou_xywh,
    localization_giou_centroid_match_loss,
    localization_giou_iou_match_loss,
    localization_giou_loss,
    localization_l1_giou_centroid_match_loss,
    localization_l1_giou_iou_match_loss,
    localization_l1_giou_loss,
)
from experiments.imagenet_localization.models.model import LocalizationRegressor
from experiments.imagenet_localization.run import ALL_METHODS, REGRESSION_METHODS, parse_args
from experiments.imagenet_localization.training.train import (
    _GIOU_FAMILY_LOSSES,
    train_supervised_epoch_localization,
)

MAX_M = 8  # mirrors data.py

# The five arms added on top of the pre-existing `giou`.
NEW_GIOU_METHODS = [
    "giou_iou_match",
    "giou_centroid_match",
    "l1_giou",
    "l1_giou_iou_match",
    "l1_giou_centroid_match",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _giou_1v1(pred, target, dtype=torch.float64) -> float:
    """GIoU between a single pred/target xywh pair, via the public loss API.

    B = 1, M = 1, one valid GT, so both the matcher and the batch mean are
    trivial and `1 - loss` is exactly GIoU(pred, target).
    """
    p = torch.tensor([pred], dtype=dtype)          # (1, 4)
    g = torch.tensor([[target]], dtype=dtype)      # (1, 1, 4)
    m = torch.ones(1, 1, dtype=torch.bool)
    return 1.0 - localization_giou_loss(p, g, m).item()


def _single_gt(boxes, dtype=torch.float64):
    """(B, 4) xywh list -> ((B, 1, 4) gt_boxes, (B, 1) all-True mask)."""
    g = torch.tensor([[b] for b in boxes], dtype=dtype)
    m = torch.ones(len(boxes), 1, dtype=torch.bool)
    return g, m


def _make_batch(B: int = 4, M: int = MAX_M, seed: int = 0) -> dict:
    """Synthetic batch matching the collate_fn contract (see test_training_step)."""
    gen = torch.Generator().manual_seed(seed)
    images = torch.randn(B, 3, 224, 224, generator=gen)
    gt_boxes = torch.zeros(B, M, 4)
    for b in range(B):
        for m in range(M):
            gt_boxes[b, m] = torch.tensor([
                torch.rand(1, generator=gen).item() * 0.8 + 0.1,   # x_c 0.1..0.9
                torch.rand(1, generator=gen).item() * 0.8 + 0.1,   # y_c
                torch.rand(1, generator=gen).item() * 0.4 + 0.1,   # w 0.1..0.5
                torch.rand(1, generator=gen).item() * 0.4 + 0.1,   # h
            ])
    return {
        'images': images,
        'gt_boxes': gt_boxes,
        'gt_mask': torch.ones(B, M, dtype=torch.bool),
    }


class _LossRecorder:
    """Stand-in for a wandb run that just records the logged train losses."""

    def __init__(self):
        self.losses: list[float] = []

    def log(self, payload: dict) -> None:
        if 'train/loss' in payload:
            self.losses.append(payload['train/loss'])


# ===========================================================================
# 1. Worked numeric cases from the spec
# ===========================================================================


def test_identical_boxes_give_giou_one_and_zero_loss():
    """P == T implies I = U = C, so GIoU = 1 and the loss vanishes.

    A non-zero loss at a perfect prediction would mean the arm's optimum is
    not the ground-truth box.
    """
    box = [0.4, 0.6, 0.3, 0.2]
    giou = _giou_1v1(box, box)

    assert abs(giou - 1.0) < 1e-5, f"expected GIoU=1, got {giou}"

    p = torch.tensor([box], dtype=torch.float64)
    g, m = _single_gt([box])
    assert abs(localization_giou_loss(p, g, m).item()) < 1e-5


def test_worked_case_a_corner_touching_squares():
    """Corner-touching half-unit squares: I=0, U=0.5, C=1, GIoU=-0.5, L=1.5.

    pred = (0.25, 0.25, 0.5, 0.5) -> corners x, y in [0.0, 0.5]
    tgt  = (0.75, 0.75, 0.5, 0.5) -> corners x, y in [0.5, 1.0]
      I = 0 (they meet only at the point (0.5, 0.5))
      A_P = A_T = 0.25 -> U = 0.5 ; IoU = 0
      C = 1.0 * 1.0 = 1.0 -> penalty = (1.0 - 0.5) / 1.0 = 0.5
      GIoU = 0 - 0.5 = -0.5 ; L = 1.5
    """
    pred = [0.25, 0.25, 0.5, 0.5]
    tgt = [0.75, 0.75, 0.5, 0.5]

    giou = _giou_1v1(pred, tgt)
    assert abs(giou - (-0.5)) < 1e-5, f"expected GIoU=-0.5, got {giou}"
    assert abs((1.0 - giou) - 1.5) < 1e-5

    # GIoU is symmetric in its two boxes: I, U and C are all symmetric.
    assert abs(giou - _giou_1v1(tgt, pred)) < 1e-5, "GIoU is not symmetric"


def test_worked_case_b_containment_gives_giou_equal_to_iou():
    """Containment forces C == U, so GIoU collapses to IoU exactly.

    pred = (0.5, 0.5, 0.5, 0.5) -> corners [0.25, 0.75]^2, area 0.25
    tgt  = (0.5, 0.5, 1.0, 1.0) -> corners [0.0, 1.0]^2,  area 1.0
      pred is fully inside tgt -> I = 0.25 ; U = 0.25 + 1.0 - 0.25 = 1.0
      IoU = 0.25 / 1.0 = 0.25
      C = the enclosing box IS tgt = 1.0 == U -> penalty = 0
      GIoU = IoU = 0.25 ; L = 0.75
    """
    giou = _giou_1v1([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0])

    assert abs(giou - 0.25) < 1e-5, f"expected GIoU=IoU=0.25, got {giou}"
    assert abs((1.0 - giou) - 0.75) < 1e-5


def test_far_disjoint_boxes_keep_a_useful_gradient():
    """Disjoint boxes: GIoU well below 0, loss > 1, and descent moves pred in.

    pred = (0.1, 0.1, 0.1, 0.1) -> corners [0.05, 0.15]^2, area 0.01
    tgt  = (0.9, 0.9, 0.1, 0.1) -> corners [0.85, 0.95]^2, area 0.01
      I = 0 ; U = 0.02
      C = (0.95 - 0.05)^2 = 0.81 -> penalty = (0.81 - 0.02) / 0.81
      GIoU = -0.79 / 0.81 = -0.9753...
    `1 - IoU` is exactly flat here, so the non-zero gradient is entirely the
    hull term. That is the whole reason GIoU replaced a plain IoU loss.
    """
    pred_l = [0.1, 0.1, 0.1, 0.1]
    tgt_l = [0.9, 0.9, 0.1, 0.1]

    giou = _giou_1v1(pred_l, tgt_l)
    assert giou < -0.9, f"expected GIoU well below -0.9, got {giou}"
    assert abs(giou - (-0.79 / 0.81)) < 1e-5
    assert (1.0 - giou) > 1.0

    g, m = _single_gt([tgt_l])
    pred = torch.tensor([pred_l], dtype=torch.float64, requires_grad=True)
    loss = localization_giou_loss(pred, g, m)
    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().max().item() > 0.0, (
        "zero gradient on disjoint boxes: the hull term is missing"
    )

    # A small step along the descent direction must strictly reduce the loss.
    with torch.no_grad():
        stepped = torch.tensor([pred_l], dtype=torch.float64) - 0.01 * pred.grad
    stepped_loss = localization_giou_loss(stepped, g, m)
    assert stepped_loss.item() < loss.item(), (
        f"descent step did not reduce the loss: {loss.item()} -> {stepped_loss.item()}"
    )


# ===========================================================================
# 2. Oracle cross-check and corner helpers
# ===========================================================================


def _has_tv_giou() -> bool:
    try:
        from torchvision.ops import generalized_box_iou  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


@pytest.mark.skipif(
    not _has_tv_giou(),
    reason="installed torchvision lacks torchvision.ops.generalized_box_iou",
)
def test_agrees_with_torchvision_generalized_box_iou():
    """Our GIoU must match torchvision's on random valid boxes, atol 1e-5.

    torchvision returns the full pairwise matrix; the elementwise values we
    want are its diagonal. This is an independent implementation, so it
    catches sign errors and wrong-corner bugs that hand-computed cases in the
    same coordinate convention could share.
    """
    from torchvision.ops import generalized_box_iou

    torch.manual_seed(11)
    n = 128
    # Build valid xywh boxes: centers in [0.2, 0.8], sizes in [0.05, 0.45].
    a = torch.cat([torch.rand(n, 2) * 0.6 + 0.2, torch.rand(n, 2) * 0.4 + 0.05], dim=1)
    b = torch.cat([torch.rand(n, 2) * 0.6 + 0.2, torch.rand(n, 2) * 0.4 + 0.05], dim=1)

    ours = _giou_xywh(a.double(), b.double())                       # (n,)
    theirs = generalized_box_iou(
        xywh_to_corners(a).double(), xywh_to_corners(b).double()
    ).diagonal()                                                     # (n,)

    torch.testing.assert_close(ours, theirs, atol=1e-5, rtol=0)


def test_xywh_corner_roundtrip_and_shapes():
    """xywh -> corners -> xywh is the identity, and shapes are preserved.

    The spec calls this helper `boxes_cxcywh_to_xyxy`; the repo already ships
    the identical conversion as `iou.xywh_to_corners`, which is what the GIoU
    code uses, so that is what is pinned here.
    """
    torch.manual_seed(12)
    boxes = torch.cat([
        torch.rand(7, 2) * 0.6 + 0.2, torch.rand(7, 2) * 0.4 + 0.05,
    ], dim=1).double()

    corners = xywh_to_corners(boxes)
    assert corners.shape == boxes.shape == (7, 4)

    # x1 < x2 and y1 < y2 for every positive-size box.
    assert bool((corners[:, 0] < corners[:, 2]).all())
    assert bool((corners[:, 1] < corners[:, 3]).all())

    torch.testing.assert_close(corners_to_xywh(corners), boxes, atol=1e-12, rtol=0)

    # Batched shapes survive too.
    batched = boxes.unsqueeze(0).expand(3, 7, 4)
    assert xywh_to_corners(batched).shape == (3, 7, 4)


def test_hand_checked_corner_conversion():
    """One explicit conversion, so a swapped w/h could not pass silently."""
    got = xywh_to_corners(torch.tensor([[0.5, 0.25, 0.4, 0.1]], dtype=torch.float64))
    want = torch.tensor([[0.3, 0.2, 0.7, 0.3]], dtype=torch.float64)
    torch.testing.assert_close(got, want, atol=1e-12, rtol=0)


# ===========================================================================
# 3. The L1 + GIoU combination
# ===========================================================================


def test_l1_giou_reduces_to_pure_giou_when_l1_weight_is_zero():
    """At l1_weight=0, giou_weight=1 the combo must equal the plain GIoU arm.

    Same matcher (argmax-GIoU) on both sides, so any difference would be in
    the reduction rather than the matching.
    """
    torch.manual_seed(13)
    B, M = 8, 4
    pred = torch.rand(B, 4, dtype=torch.float64) * 0.5 + 0.25
    gt_boxes = torch.rand(B, M, 4, dtype=torch.float64) * 0.5 + 0.25
    gt_mask = torch.ones(B, M, dtype=torch.bool)

    combo = localization_l1_giou_loss(
        pred, gt_boxes, gt_mask, l1_weight=0.0, giou_weight=1.0,
    )
    pure = localization_giou_loss(pred, gt_boxes, gt_mask)

    torch.testing.assert_close(combo, pure, atol=1e-12, rtol=0)


def test_l1_giou_reduces_to_weighted_l1_when_giou_weight_is_zero():
    """At giou_weight=0 the combo must equal l1_weight * mean|pred - target|."""
    torch.manual_seed(14)
    B = 6
    pred = torch.rand(B, 4, dtype=torch.float64) * 0.5 + 0.25
    target = torch.rand(B, 4, dtype=torch.float64) * 0.5 + 0.25
    gt_boxes = target.unsqueeze(1)                     # (B, 1, 4), matcher trivial
    gt_mask = torch.ones(B, 1, dtype=torch.bool)

    got = localization_l1_giou_loss(
        pred, gt_boxes, gt_mask, l1_weight=3.0, giou_weight=0.0,
    )
    want = 3.0 * (pred - target).abs().mean(dim=-1).mean()

    torch.testing.assert_close(got, want, atol=1e-12, rtol=0)


def test_l1_giou_default_weights_hand_computed():
    """Default DETR weights: L = 5 * L1 + 2 * (1 - GIoU), checked in closed form.

    pred = (0.25, 0.25, 0.5, 0.5), target = (0.75, 0.75, 0.5, 0.5) is worked
    case A, so 1 - GIoU = 1.5. The L1 term is mean(|0.5|, |0.5|, 0, 0) = 0.25.
      L = 5 * 0.25 + 2 * 1.5 = 1.25 + 3.0 = 4.25
    """
    pred = torch.tensor([[0.25, 0.25, 0.5, 0.5]], dtype=torch.float64)
    gt_boxes = torch.tensor([[[0.75, 0.75, 0.5, 0.5]]], dtype=torch.float64)
    gt_mask = torch.ones(1, 1, dtype=torch.bool)

    got = localization_l1_giou_loss(pred, gt_boxes, gt_mask).item()

    assert abs(got - 4.25) < 1e-12, f"expected 4.25, got {got}"


# ===========================================================================
# 4. Matcher variants
# ===========================================================================


def test_matchers_agree_when_there_is_only_one_gt():
    """With a single valid GT all three matchers must select it, so the three
    pure-GIoU arms coincide. Any disagreement here is a matcher bug, not a
    matching-policy difference."""
    torch.manual_seed(15)
    B = 8
    pred = torch.rand(B, 4, dtype=torch.float64) * 0.5 + 0.25
    gt_boxes = torch.rand(B, 1, 4, dtype=torch.float64) * 0.5 + 0.25
    gt_mask = torch.ones(B, 1, dtype=torch.bool)

    base = localization_giou_loss(pred, gt_boxes, gt_mask)
    for fn in (localization_giou_iou_match_loss, localization_giou_centroid_match_loss):
        torch.testing.assert_close(fn(pred, gt_boxes, gt_mask), base, atol=1e-12, rtol=0)


def test_iou_matcher_ties_to_first_gt_where_giou_matcher_picks_the_nearest():
    """The matchers must actually differ in the zero-overlap regime.

    pred = [0.1, 0.1, 0.2, 0.2] overlaps neither GT, so every IoU is 0 and
    argmax-IoU ties, falling back to slot 0 (the far box, L = 48/25 = 1.92).
    GIoU stays strictly ordered at zero overlap, so argmax-GIoU picks the near
    slot 1 instead (L = 10/9). This asymmetry is the reason `giou` matches on
    GIoU; the `_iou_match` arm exists to quantify what that choice buys.
    """
    pred = torch.tensor([[0.1, 0.1, 0.2, 0.2]], dtype=torch.float64)
    gt_boxes = torch.tensor([[
        [0.9, 0.9, 0.2, 0.2],     # far
        [0.35, 0.1, 0.2, 0.2],    # near
    ]], dtype=torch.float64)
    gt_mask = torch.ones(1, 2, dtype=torch.bool)

    giou_matched = localization_giou_loss(pred, gt_boxes, gt_mask).item()
    iou_matched = localization_giou_iou_match_loss(pred, gt_boxes, gt_mask).item()

    assert abs(giou_matched - 10.0 / 9.0) < 1e-12, (
        f"argmax-GIoU should pick the near GT (10/9), got {giou_matched}"
    )
    assert abs(iou_matched - 48.0 / 25.0) < 1e-12, (
        f"argmax-IoU should tie to slot 0 (48/25), got {iou_matched}"
    )


def test_centroid_matcher_selects_the_closest_centroid():
    """Centroid matching must pick slot 1 here, where argmax-IoU would tie.

    pred centre (0.1, 0.1). Slot 0 centre (0.9, 0.9), squared distance 1.28;
    slot 1 centre (0.35, 0.1), squared distance 0.0625. Slot 1 wins, giving
    the same L = 10/9 as the GIoU matcher on this configuration.
    """
    pred = torch.tensor([[0.1, 0.1, 0.2, 0.2]], dtype=torch.float64)
    gt_boxes = torch.tensor([[
        [0.9, 0.9, 0.2, 0.2],
        [0.35, 0.1, 0.2, 0.2],
    ]], dtype=torch.float64)
    gt_mask = torch.ones(1, 2, dtype=torch.bool)

    got = localization_giou_centroid_match_loss(pred, gt_boxes, gt_mask).item()

    assert abs(got - 10.0 / 9.0) < 1e-12, f"expected 10/9 from the near GT, got {got}"


def test_padded_gt_slots_never_win_any_matcher():
    """A masked-out perfect match must be ignored by all five new arms.

    Slot 1 is an exact copy of the prediction (the best possible target) but
    is padding; slot 0 is the only valid GT, giving the nested closed form
    L = 8/9 for the pure arms. A loss of 0 would mean the arm trains on
    padding.
    """
    pred = torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float64)
    gt_boxes = torch.tensor([[
        [0.5, 0.5, 0.6, 0.6],   # valid, nested -> L = 8/9
        [0.5, 0.5, 0.2, 0.2],   # padding, identical to pred
    ]], dtype=torch.float64)
    gt_mask = torch.tensor([[True, False]])

    for fn in (localization_giou_iou_match_loss, localization_giou_centroid_match_loss):
        got = fn(pred, gt_boxes, gt_mask).item()
        assert abs(got - 8.0 / 9.0) < 1e-12, (
            f"{fn.__name__}: padded slot won the match (got {got})"
        )

    # The L1+GIoU arms must select the same target; L1 against slot 0 is
    # mean(0, 0, 0.4, 0.4) = 0.2, so L = 5 * 0.2 + 2 * 8/9 = 1 + 16/9.
    want = 1.0 + 16.0 / 9.0
    for fn in (
        localization_l1_giou_loss,
        localization_l1_giou_iou_match_loss,
        localization_l1_giou_centroid_match_loss,
    ):
        got = fn(pred, gt_boxes, gt_mask).item()
        assert abs(got - want) < 1e-12, f"{fn.__name__}: expected {want}, got {got}"


def test_rows_with_no_valid_gt_are_excluded_from_every_new_arm():
    """All-False gt_mask rows must not drag the mean toward a zero box.

    Row 0 is the nested case (L = 8/9) and row 1 has no GT at all, so the
    batch loss must be exactly 8/9 rather than a two-row average.
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

    for fn in (localization_giou_iou_match_loss, localization_giou_centroid_match_loss):
        got = fn(pred, gt_boxes, gt_mask).item()
        assert abs(got - 8.0 / 9.0) < 1e-12, f"{fn.__name__}: got {got}"


# ===========================================================================
# 5. Numerical robustness
# ===========================================================================


@pytest.mark.parametrize("fn", [
    localization_giou_iou_match_loss,
    localization_giou_centroid_match_loss,
    localization_l1_giou_loss,
    localization_l1_giou_iou_match_loss,
    localization_l1_giou_centroid_match_loss,
])
def test_gradients_finite_on_near_degenerate_tiny_boxes(fn):
    """w = h = 1e-4 boxes (and a padded all-zero GT) must not yield NaN.

    Both divisions in the GIoU can approach 0/0 for tiny or coincident boxes.
    A NaN here poisons the whole batch gradient, and `torch.where` still
    backprops NaN from its discarded branch, so the guards have to sit inside
    the divisions themselves.
    """
    pred = torch.tensor([
        [0.5, 0.5, 1e-4, 1e-4],
        [0.2, 0.2, 1e-4, 1e-4],
        [0.0, 0.0, 0.0, 0.0],
    ], dtype=torch.float64, requires_grad=True)
    gt_boxes = torch.tensor([
        [[0.5, 0.5, 1e-4, 1e-4], [0.0, 0.0, 0.0, 0.0]],
        [[0.8, 0.8, 0.2, 0.2], [0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    ], dtype=torch.float64)
    gt_mask = torch.tensor([[True, False], [True, False], [True, False]])

    loss = fn(pred, gt_boxes, gt_mask)
    loss.backward()

    assert torch.isfinite(loss), f"{fn.__name__}: non-finite loss {loss}"
    assert torch.isfinite(pred.grad).all(), (
        f"{fn.__name__}: non-finite gradient {pred.grad}"
    )


@pytest.mark.parametrize("fn", [
    localization_giou_iou_match_loss,
    localization_giou_centroid_match_loss,
])
def test_pure_giou_arms_stay_in_the_zero_two_range(fn):
    """GIoU is in [-1, 1], so 1 - GIoU must land in [0, 2] for random boxes."""
    torch.manual_seed(16)
    B, M = 32, 4
    pred = torch.rand(B, 4)
    gt_boxes = torch.rand(B, M, 4)
    gt_mask = torch.rand(B, M) > 0.4
    gt_mask[:, 0] = True

    loss = fn(pred, gt_boxes, gt_mask)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert 0.0 <= loss.item() <= 2.0, f"{fn.__name__}: loss {loss.item()} outside [0, 2]"


# ===========================================================================
# 6. Wiring
# ===========================================================================


@pytest.mark.parametrize("method", NEW_GIOU_METHODS)
def test_new_methods_are_accepted_by_argparse(method, monkeypatch):
    """Each new arm must be a valid --method choice."""
    monkeypatch.setattr("sys.argv", [
        "run.py", "--method", method, "--data_dir", "/tmp", "--output_dir", "/tmp",
    ])
    args = parse_args()
    assert args.method == method
    assert method in ALL_METHODS


@pytest.mark.parametrize("method", NEW_GIOU_METHODS)
def test_new_methods_use_the_regressor_and_have_a_loss(method):
    """Every new arm must select LocalizationRegressor and have a dispatch entry."""
    assert method in REGRESSION_METHODS, (
        f"{method} would build LocalizationPolicy and skip evaluate_mse_regressor"
    )
    assert method in _GIOU_FAMILY_LOSSES, f"{method} has no train.py dispatch entry"


def test_l1_giou_weight_flags_default_to_detr_values(monkeypatch):
    """--l1_weight / --giou_weight must exist and default to DETR's 5.0 / 2.0."""
    monkeypatch.setattr("sys.argv", [
        "run.py", "--method", "l1_giou", "--data_dir", "/tmp", "--output_dir", "/tmp",
    ])
    args = parse_args()
    assert args.l1_weight == 5.0
    assert args.giou_weight == 2.0

    monkeypatch.setattr("sys.argv", [
        "run.py", "--method", "l1_giou", "--data_dir", "/tmp", "--output_dir", "/tmp",
        "--l1_weight", "1.5", "--giou_weight", "4.0",
    ])
    args = parse_args()
    assert args.l1_weight == 1.5
    assert args.giou_weight == 4.0


# ===========================================================================
# 7. Training smoke test
# ===========================================================================


@pytest.mark.parametrize("method", ["giou", "l1_giou"])
def test_supervised_epoch_overfits_a_fixed_batch(method):
    """30 steps on one fixed batch through the real supervised epoch loop.

    Exercises the full path the sweep scripts use: LocalizationRegressor
    forward, the train.py method dispatch (including the l1_weight/giou_weight
    threading for l1_giou), backward, grad clip and optimizer.step. The
    backbone is frozen so the run stays quick on CPU; the trainable
    spatial_reduce + 4-way head read fixed features and have ample capacity to
    fit four boxes.

    One valid GT per image (the rest padded), so the argmax-GIoU matcher is
    stationary and this is a clean regression toward a fixed target. That is
    deliberate: the matcher's target-selection behaviour is pinned by the
    deterministic tests above; here we only want to see that the objective
    actually descends. LR is the sweep default 5e-4 (1e-3 Adam overshoots the
    flatter pure-GIoU landscape). We assert the loss *trends* down -- the mean
    over the last third below the mean over the first third -- rather than a
    single-step endpoint inequality, which Adam's early transient can violate
    even on a cleanly-decreasing run.
    """
    torch.manual_seed(0)
    device = torch.device("cpu")

    model = LocalizationRegressor(pretrained=False, seed=0)
    model.freeze_backbone()
    model.to(device)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=5e-4,
    )

    # Single valid GT per image -> stationary matcher (see docstring).
    batch = _make_batch(B=4, seed=0)
    batch['gt_mask'] = torch.zeros_like(batch['gt_mask'])
    batch['gt_mask'][:, 0] = True

    n_steps = 30
    loader = [batch] * n_steps        # same batch every step -> pure overfitting
    recorder = _LossRecorder()

    out = train_supervised_epoch_localization(
        model, loader, optimizer, scheduler=None,
        method=method, K=50, device=device, epoch=1,
        wandb_run=recorder, log_every=1,
    )

    assert 'loss_mean' in out
    assert torch.isfinite(torch.tensor(out['loss_mean'])), (
        f"{method}: non-finite epoch loss {out['loss_mean']}"
    )
    assert len(recorder.losses) == n_steps, (
        f"expected {n_steps} logged steps, got {len(recorder.losses)}"
    )
    assert all(torch.isfinite(torch.tensor(v)) for v in recorder.losses)

    window = n_steps // 3
    first = sum(recorder.losses[:window]) / window
    last = sum(recorder.losses[-window:]) / window
    assert last < first, (
        f"{method}: loss did not trend down over {n_steps} steps "
        f"(first-third mean {first:.4f} -> last-third mean {last:.4f}; "
        f"trajectory {recorder.losses[0]:.4f} .. {recorder.losses[-1]:.4f})"
    )
