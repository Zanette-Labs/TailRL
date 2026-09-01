"""Loss functions for experiments.imagenet_localization.

- factored_sample_log_prob: sum of per-head log-probs at sampled indices
- ordinal_ce_single_head: ordinal cross-entropy (window-survival form, per head)
- localization_ordinal_ce_loss: mean of per-head ordinal CE over 4 coordinate heads
- localization_cross_entropy_loss: mean of per-head F.cross_entropy over 4 heads
- localization_mse_loss: MSE against the primary GT box (regression baseline)
- localization_giou_loss: 1 - GIoU against the argmax-GIoU-matched GT box
- localization_giou_{iou,centroid}_match_loss: same objective, alternative matchers
- localization_l1_giou[_{iou,centroid}_match]_loss: DETR-style L1 + (1 - GIoU)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.imagenet_localization.core.iou import xywh_to_corners


HEAD_NAMES: tuple[str, ...] = ("x_c", "y_c", "w", "h")


# ---------------------------------------------------------------------------
# 1. Factored sample log-prob (the #1 trap for newcomers)
# ---------------------------------------------------------------------------

def factored_sample_log_prob(
    log_probs: dict[str, torch.Tensor],   # each (B, K)
    samples:   dict[str, torch.Tensor],   # each (B, N) long
) -> torch.Tensor:
    """Sum of per-head log-probs at sampled bin indices.

    log π(sample | I) = Σ_h log π_h(sample_h | I)

    Args:
        log_probs: dict keyed by HEAD_NAMES, each value shape (B, K)
                   — log-probabilities (log_softmax output) from one head.
        samples:   dict keyed by HEAD_NAMES, each value shape (B, N) long
                   — sampled bin indices per head.

    Returns:
        (B, N) tensor of summed log-probs. NOT (B, N, 4): summed across heads.

    The #1 trap: gathering across heads without summing leaves a (B, N, 4)
    tensor that silently broadcasts-wrong in downstream losses.
    """
    # For each head h: log_probs[h] has shape (B, K); samples[h] has shape (B, N).
    # gather(1, samples[h]) -> (B, N).
    # Sum across heads.
    out = None
    for h in HEAD_NAMES:
        per_head = log_probs[h].gather(1, samples[h])    # (B, N)
        out = per_head if out is None else out + per_head
    return out  # (B, N)


# ---------------------------------------------------------------------------
# 2. Single-head ordinal CE (window-survival form — from experiment5_ordinal)
# ---------------------------------------------------------------------------

def ordinal_ce_single_head(
    logits: torch.Tensor,     # (B, K)
    targets: torch.Tensor,    # (B,) long in {0, ..., K-1}
    K: int,
) -> torch.Tensor:
    """Single-head ordinal cross-entropy (window-survival form).

    L = mean_b [ (1/(K-1)) * sum_{d=0}^{K-2} -log W_d(b, target_b) ]
    where W_d(b, y) = P( |bin - y| <= d | b ) = sum of softmax probs in the window.

    At K=2, this reduces to standard cross-entropy — which is the ordinal_ce
    ≡ cross_entropy binary-recovery identity Task 6 has a test for.

    Args:
        logits:   (B, K) raw logits
        targets:  (B,)    long bin indices
        K:        number of bins

    Returns:
        scalar loss, positive value, averaged over batch.
    """
    # Adapt from experiment5_ordinal/losses.py::ordinal_ce_loss.
    B = logits.size(0)
    probs = F.softmax(logits, dim=-1)           # (B, K)
    cum_probs = probs.cumsum(dim=-1)            # (B, K)

    total_neg_log_survival = torch.zeros(B, device=logits.device, dtype=logits.dtype)

    for d in range(K - 1):  # d = 0, 1, ..., K-2
        # Window bounds: [max(0, y*-d), min(K-1, y*+d)]
        low = (targets - d).clamp(min=0)        # (B,)
        high = (targets + d).clamp(max=K - 1)   # (B,)

        # W_d = P(low <= b <= high) via cumulative sum
        upper = cum_probs.gather(1, high.unsqueeze(1)).squeeze(1)   # (B,)
        lower = torch.where(
            low > 0,
            cum_probs.gather(1, (low - 1).clamp(min=0).unsqueeze(1)).squeeze(1),
            torch.zeros(B, device=logits.device, dtype=logits.dtype),
        )
        W_d = (upper - lower).clamp(min=1e-12)  # (B,)
        total_neg_log_survival += -torch.log(W_d)

    # Average over (K-1) thresholds, then average over batch
    loss = total_neg_log_survival / (K - 1)     # (B,)
    return loss.mean()


# ---------------------------------------------------------------------------
# 3. Factored supervised losses (sum over 4 heads, divided by 4)
# ---------------------------------------------------------------------------

def localization_ordinal_ce_loss(
    logits: dict[str, torch.Tensor],    # each (B, K)
    targets: dict[str, torch.Tensor],   # each (B,) long
    K: int,
) -> torch.Tensor:
    """Mean of per-head ordinal CE losses over the 4 coordinate heads.

    Args:
        logits:  dict keyed by HEAD_NAMES, each (B, K)
        targets: dict keyed by HEAD_NAMES, each (B,) long
        K:       number of bins

    Returns:
        scalar loss — mean over the 4 heads.
    """
    losses = [
        ordinal_ce_single_head(logits[h], targets[h], K)
        for h in HEAD_NAMES
    ]
    return sum(losses) / 4.0


def localization_cross_entropy_loss(
    logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Mean of per-head `F.cross_entropy` losses over 4 coordinate heads."""
    losses = [
        F.cross_entropy(logits[h], targets[h])
        for h in HEAD_NAMES
    ]
    return sum(losses) / 4.0


def localization_mse_loss(
    pred: torch.Tensor,        # (B, 4) in [0, 1]
    gt_primary_box: torch.Tensor,  # (B, 4) in [0, 1]
) -> torch.Tensor:
    """MSE loss against the PRIMARY GT box (legacy — kept for reproducibility).

    Uses `gt_boxes[:, 0, :]` (first slot in CSV). This is biased on multi-GT
    images — use `localization_mse_iou_match_loss` instead for new runs.
    """
    return F.mse_loss(pred, gt_primary_box)


def _centroid_match_target(
    pred: torch.Tensor,       # (B, 4) xywh
    gt_boxes: torch.Tensor,   # (B, M, 4) xywh
    gt_mask: torch.Tensor,    # (B, M) bool
) -> torch.Tensor:
    """Select each image's GT target by minimum centroid L2 distance.

    For each image, picks j* = argmin_j ||pred_center - GT_j_center||_2 among
    valid GTs. Returns the (B, 4) target box.

    Motivation: max-IoU matching breaks down when the prediction overlaps no
    GT (IoU==0 for all j). In that regime argmax-IoU is undefined up to ties
    and silently falls back to the first valid GT. Centroid L2 matching gives
    a meaningful signal even when boxes don't overlap, making it a more
    stable matcher early in training or for hard (small/off-center) images.
    """
    B, M, _ = gt_boxes.shape

    with torch.no_grad():
        pred_c = pred[:, None, :2]             # (B, 1, 2)
        gt_c   = gt_boxes[..., :2]             # (B, M, 2)
        d2 = ((pred_c - gt_c) ** 2).sum(dim=-1)   # (B, M) squared L2
        # Mask padded rows so they lose the argmin
        d2 = torch.where(gt_mask, d2, torch.full_like(d2, float("inf")))
        j_star = d2.argmin(dim=-1)             # (B,)

    target = gt_boxes.gather(
        1, j_star.view(B, 1, 1).expand(-1, 1, 4)
    ).squeeze(1)
    return target


def localization_mse_centroid_match_loss(
    pred: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss against the GT with closest centroid to the prediction.

    Matcher: min-L2-distance of (x_c, y_c) — see `_centroid_match_target`.
    Loss: F.mse_loss(pred, target).

    Use this when max-IoU matching is unstable (e.g., early training when
    predictions overlap no GT, so IoU matching falls back to first-GT).
    """
    target = _centroid_match_target(pred, gt_boxes, gt_mask)
    return F.mse_loss(pred, target)


def localization_l1_centroid_match_loss(
    pred: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_mask: torch.Tensor,
) -> torch.Tensor:
    """L1 loss against the GT with closest centroid to the prediction.

    Matcher: min-L2-distance of (x_c, y_c) — see `_centroid_match_target`.
    Loss: F.l1_loss(pred, target).
    """
    target = _centroid_match_target(pred, gt_boxes, gt_mask)
    return F.l1_loss(pred, target)


def localization_l1_iou_match_loss(
    pred: torch.Tensor,       # (B, 4) in [0, 1]  — xywh
    gt_boxes: torch.Tensor,   # (B, M, 4)         — xywh, padded
    gt_mask: torch.Tensor,    # (B, M) bool       — valid GT rows
    eps: float = 1e-10,
) -> torch.Tensor:
    """L1 (mean absolute error) against the GT with highest IoU vs the prediction.

    Same matching rule as `localization_mse_iou_match_loss` (argmax-IoU over
    valid GTs, recomputed each batch, no gradient through argmax), but applies
    F.l1_loss instead of F.mse_loss on the residual. L1 is more robust to
    outliers and generally yields slightly tighter localization on small boxes
    since large coordinate errors aren't squared.
    """
    B, M, _ = gt_boxes.shape

    with torch.no_grad():
        pred_exp = pred.unsqueeze(1).expand(-1, M, -1)
        pc1 = pred_exp[..., :2] - pred_exp[..., 2:] / 2
        pc2 = pred_exp[..., :2] + pred_exp[..., 2:] / 2
        gc1 = gt_boxes[..., :2] - gt_boxes[..., 2:] / 2
        gc2 = gt_boxes[..., :2] + gt_boxes[..., 2:] / 2
        i1 = torch.maximum(pc1, gc1)
        i2 = torch.minimum(pc2, gc2)
        inter_wh = (i2 - i1).clamp(min=0.0)
        inter = inter_wh[..., 0] * inter_wh[..., 1]
        pa = pred_exp[..., 2] * pred_exp[..., 3]
        ga = gt_boxes[..., 2] * gt_boxes[..., 3]
        union = (pa + ga - inter).clamp(min=eps)
        iou = inter / union
        iou = torch.where(gt_mask, iou, torch.full_like(iou, -1.0))
        j_star = iou.argmax(dim=-1)

    target = gt_boxes.gather(
        1, j_star.view(B, 1, 1).expand(-1, 1, 4)
    ).squeeze(1)

    return F.l1_loss(pred, target)


def localization_mse_iou_match_loss(
    pred: torch.Tensor,       # (B, 4) in [0, 1]  — xywh
    gt_boxes: torch.Tensor,   # (B, M, 4)         — xywh, padded
    gt_mask: torch.Tensor,    # (B, M) bool       — valid GT rows
    eps: float = 1e-10,
) -> torch.Tensor:
    """MSE against the GT with highest IoU vs the prediction.

    For each image, picks j* = argmax_j IoU(pred, GT_j) among valid GTs
    (masked by gt_mask), then applies F.mse_loss against GT_{j*}.

    Why: a deterministic regressor outputs one box per image. The val metric
    takes max-IoU over all GTs. Using max-IoU-match as the training target
    aligns the loss with the eval metric. Matching is recomputed every batch
    under `torch.no_grad()`, so no gradient flows through argmax — only
    through the L2 residual against the selected target.

    Args:
        pred:     (B, 4) xywh in [0, 1] (sigmoid output)
        gt_boxes: (B, M, 4) xywh in [0, 1], padded
        gt_mask:  (B, M) bool — True for valid GT slots

    Returns:
        scalar F.mse_loss(pred, target) where target is the max-IoU GT box.
    """
    B, M, _ = gt_boxes.shape

    with torch.no_grad():
        # Per-GT pairwise IoU (B, M) via xywh → corners
        pred_exp = pred.unsqueeze(1).expand(-1, M, -1)            # (B, M, 4)
        pc1 = pred_exp[..., :2] - pred_exp[..., 2:] / 2           # (B, M, 2)
        pc2 = pred_exp[..., :2] + pred_exp[..., 2:] / 2
        gc1 = gt_boxes[..., :2] - gt_boxes[..., 2:] / 2
        gc2 = gt_boxes[..., :2] + gt_boxes[..., 2:] / 2
        i1 = torch.maximum(pc1, gc1)
        i2 = torch.minimum(pc2, gc2)
        inter_wh = (i2 - i1).clamp(min=0.0)
        inter = inter_wh[..., 0] * inter_wh[..., 1]               # (B, M)
        pa = pred_exp[..., 2] * pred_exp[..., 3]
        ga = gt_boxes[..., 2] * gt_boxes[..., 3]
        union = (pa + ga - inter).clamp(min=eps)
        iou = inter / union                                        # (B, M)
        # Mask out padded rows → they lose the argmax
        iou = torch.where(gt_mask, iou, torch.full_like(iou, -1.0))
        j_star = iou.argmax(dim=-1)                                # (B,)

    # Gather per-image target (B, 4)
    target = gt_boxes.gather(
        1, j_star.view(B, 1, 1).expand(-1, 1, 4)
    ).squeeze(1)

    return F.mse_loss(pred, target)


# ---------------------------------------------------------------------------
# 3b. Generalized IoU (Rezatofighi et al., CVPR 2019)
# ---------------------------------------------------------------------------

def _giou_xywh(
    boxes_a: torch.Tensor,   # (..., 4) xywh
    boxes_b: torch.Tensor,   # (..., 4) xywh — broadcastable against boxes_a
    eps: float = 1e-10,
) -> torch.Tensor:
    """Generalized IoU between two broadcastable sets of xywh boxes.

    Rezatofighi et al., "Generalized Intersection over Union: A Metric and
    A Loss for Bounding Box Regression", CVPR 2019.

        I    = area(A ∩ B)
        U    = area(A) + area(B) - I
        IoU  = I / U
        C    = area of the smallest axis-aligned box enclosing BOTH A and B
        GIoU = IoU - (C - U) / C            ∈ [-1, 1]

    Areas are computed from *clamped corner extents* (`clamp(x2-x1, min=0)`),
    not from `w * h`. For a well-formed box the two agree exactly; for an
    inverted box (w < 0 and h < 0) `w * h` is spuriously positive, which would
    corrupt U, allow U > C, and break the defining `GIoU <= IoU` invariant.
    The matching branch only feeds an argmax so it could tolerate that, but the
    GIoU value here *is* the loss, so it is computed robustly.

    Both divisions guard their denominator with `clamp(min=eps)`:
      - U = 0 iff both boxes have zero area (U >= max(area_a, area_b)). Reachable
        here: a padded GT row of [0,0,0,0] has zero area.
      - C = 0 iff both boxes are degenerate *and* coincident (e.g. pred and a
        padded GT both exactly [0,0,0,0]). Unlike IoU — where I = 0 makes the
        numerator vanish safely — the GIoU penalty divides by a quantity with no
        lower bound of its own, so this second guard is mandatory. Without it
        the padded slots produce NaN, and `torch.where(mask, f(x), const)` still
        backprops NaN from the discarded branch.

    Convention that falls out of the two clamps (documented so nobody "fixes"
    it): two coincident zero-area boxes give GIoU = 0 (not 1), matching
    `iou.box_iou_xywh`, which also returns 0 for zero-area boxes. Two separated
    zero-area boxes give GIoU = -1.

    `(C - U)` is deliberately NOT clamped to min=0: it is mathematically
    non-negative, and floating point may render it as ~-1e-9 when C == U
    exactly (nested or abutting equal-extent boxes), where the true penalty
    gradient is already zero anyway.

    Args:
        boxes_a: (..., 4) xywh
        boxes_b: (..., 4) xywh, broadcastable against boxes_a
        eps:     denominator floor for the U and C divisions

    Returns:
        (...) tensor of GIoU values in [-1, 1]. Differentiable w.r.t. both
        arguments; the caller decides which side carries gradient.
    """
    a = xywh_to_corners(boxes_a)
    b = xywh_to_corners(boxes_b)

    ax1, ay1, ax2, ay2 = a.unbind(-1)
    bx1, by1, bx2, by2 = b.unbind(-1)

    # Intersection — clamped, so disjoint boxes give exactly 0, never negative.
    ix1 = torch.maximum(ax1, bx1)
    iy1 = torch.maximum(ay1, by1)
    ix2 = torch.minimum(ax2, bx2)
    iy2 = torch.minimum(ay2, by2)
    inter = (ix2 - ix1).clamp(min=0.0) * (iy2 - iy1).clamp(min=0.0)

    # Areas from clamped extents (see docstring).
    area_a = (ax2 - ax1).clamp(min=0.0) * (ay2 - ay1).clamp(min=0.0)
    area_b = (bx2 - bx1).clamp(min=0.0) * (by2 - by1).clamp(min=0.0)

    union = area_a + area_b - inter
    iou = inter / union.clamp(min=eps)

    # Smallest axis-aligned box enclosing both.
    ex1 = torch.minimum(ax1, bx1)
    ey1 = torch.minimum(ay1, by1)
    ex2 = torch.maximum(ax2, bx2)
    ey2 = torch.maximum(ay2, by2)
    enclose = (ex2 - ex1).clamp(min=0.0) * (ey2 - ey1).clamp(min=0.0)

    return iou - (enclose - union) / enclose.clamp(min=eps)


def localization_giou_loss(
    pred: torch.Tensor,       # (B, 4) in [0, 1]  — xywh
    gt_boxes: torch.Tensor,   # (B, M, 4)         — xywh, padded
    gt_mask: torch.Tensor,    # (B, M) bool       — valid GT rows
    eps: float = 1e-10,
) -> torch.Tensor:
    """Generalized-IoU loss against the GT with highest GIoU vs the prediction.

    L = mean_b [ 1 - GIoU(pred_b, GT_{b, j*}) ],  j* = argmax_j GIoU(pred_b, GT_j)

    GIoU ∈ [-1, 1] so L ∈ [0, 2], and L = 0 iff the boxes coincide.
    See Rezatofighi et al., "Generalized Intersection over Union: A Metric and
    A Loss for Bounding Box Regression", CVPR 2019.

    Why GIoU rather than plain 1 - IoU: when pred and GT do not overlap, IoU is
    identically 0 on an open neighbourhood, so `1 - IoU` has *exactly zero*
    gradient and the regressor gets no learning signal at all from a miss. GIoU
    adds the enclosing-box penalty (C - U) / C, which keeps growing as the boxes
    separate, so the gradient still points pred toward the GT (and grows it to
    close the gap) in the zero-overlap regime.

    Why argmax-GIoU matching rather than argmax-IoU (the rule used by
    `localization_mse_iou_match_loss` / `localization_l1_iou_match_loss`):
    argmax-IoU is degenerate exactly where it matters most. On an image whose
    prediction overlaps no GT, every IoU is 0, the argmax ties, and PyTorch
    silently returns slot 0 — the loss then regresses toward whichever box the
    CSV happened to list first, which may be the farthest one. GIoU is strictly
    ordered even at zero overlap (it keeps decreasing with separation), so
    argmax-GIoU always selects the *nearest* GT and the matcher stays consistent
    with the objective it feeds. Using the same quantity for both matching and
    loss also means the matched target is the one the loss can most cheaply
    improve.

    Padded GT slots are filled with the sentinel -2.0 before the argmax. -2.0,
    not the -1.0 used by `iou.py`: IoU >= 0 there, but GIoU >= -1 with -1
    attainable, and argmax breaks ties toward the lowest index, so a -1.0
    sentinel could let a padded slot beat a legitimately -1.0 real GT.

    Matching runs under `torch.no_grad()` and the GIoU is then *recomputed*
    against the gathered target, so gradient flows only through `pred` — never
    through the argmax, the sentinel `where`, or the padded slots. `target` is
    gathered from `gt_boxes` (a constant), so the gather carries no gradient.

    Images with no valid GT at all (`gt_mask[b]` all False) are excluded from
    the mean rather than being regressed toward a padded zero box; this mirrors
    `iou.batched_max_iou`, which handles the same case explicitly.

    Args:
        pred:     (B, 4) xywh in [0, 1] (sigmoid output of LocalizationRegressor)
        gt_boxes: (B, M, 4) xywh in [0, 1], padded
        gt_mask:  (B, M) bool — True for valid GT slots
        eps:      denominator floor for the union / enclosing-box divisions

    Returns:
        scalar mean of (1 - GIoU) over the images that have at least one GT.
    """
    B, M, _ = gt_boxes.shape

    with torch.no_grad():
        # (B, M) GIoU matrix — thrown away after the argmax.
        pred_exp = pred.unsqueeze(1).expand(-1, M, -1)             # (B, M, 4)
        giou_mat = _giou_xywh(pred_exp, gt_boxes, eps=eps)         # (B, M)
        # Mask padded rows so they can never win the argmax (-2.0 < min GIoU).
        giou_mat = torch.where(gt_mask, giou_mat, torch.full_like(giou_mat, -2.0))
        j_star = giou_mat.argmax(dim=-1)                           # (B,)

    # Gather per-image target (B, 4) — constant, no gradient path.
    target = gt_boxes.gather(
        1, j_star.view(B, 1, 1).expand(-1, 1, 4)
    ).squeeze(1)

    # Recompute GIoU with gradient, pred side only.
    per_sample = 1.0 - _giou_xywh(pred, target, eps=eps)           # (B,)

    # Drop rows with no valid GT (argmax would have returned slot 0 = padding).
    has_gt = gt_mask.any(dim=-1).to(per_sample.dtype)              # (B,)
    return (per_sample * has_gt).sum() / has_gt.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# 3c. GIoU-family arms: alternative GT matchers and the DETR L1+GIoU combo
#
# `localization_giou_loss` above bundles one specific matcher (argmax-GIoU)
# into the loss. The arms below factor the matcher out so the same GIoU
# objective can be paired with the two matchers the mse / l1 family already
# uses (argmax-IoU, min-centroid-L2), and so the L1 term can be added on top.
# The functions above are left byte-for-byte unchanged: `giou` runs already on
# disk stay reproducible.
# ---------------------------------------------------------------------------

def _iou_match_target(
    pred: torch.Tensor,       # (B, 4) xywh
    gt_boxes: torch.Tensor,   # (B, M, 4) xywh
    gt_mask: torch.Tensor,    # (B, M) bool
    eps: float = 1e-10,
) -> torch.Tensor:
    """Select each image's GT target by maximum IoU against the prediction.

    Same rule (and same -1.0 padding sentinel) as the matcher inlined in
    `localization_mse_iou_match_loss` / `localization_l1_iou_match_loss`,
    lifted into a helper so the GIoU arms can reuse it without touching those
    functions.

    Note the known degeneracy this matcher carries: when the prediction
    overlaps no GT every IoU is exactly 0, the argmax ties, and torch returns
    slot 0, so the target becomes whichever box the CSV listed first rather
    than the nearest one. That is precisely the failure `_giou_match_target`
    avoids; it is reproduced faithfully here so the `*_iou_match` arms remain
    a like-for-like comparison against the mse / l1 arms.

    Runs under `torch.no_grad()`; the returned target is gathered from
    `gt_boxes` (a constant), so it carries no gradient.
    """
    B, M, _ = gt_boxes.shape

    with torch.no_grad():
        pred_exp = pred.unsqueeze(1).expand(-1, M, -1)            # (B, M, 4)
        pc1 = pred_exp[..., :2] - pred_exp[..., 2:] / 2
        pc2 = pred_exp[..., :2] + pred_exp[..., 2:] / 2
        gc1 = gt_boxes[..., :2] - gt_boxes[..., 2:] / 2
        gc2 = gt_boxes[..., :2] + gt_boxes[..., 2:] / 2
        i1 = torch.maximum(pc1, gc1)
        i2 = torch.minimum(pc2, gc2)
        inter_wh = (i2 - i1).clamp(min=0.0)
        inter = inter_wh[..., 0] * inter_wh[..., 1]               # (B, M)
        pa = pred_exp[..., 2] * pred_exp[..., 3]
        ga = gt_boxes[..., 2] * gt_boxes[..., 3]
        union = (pa + ga - inter).clamp(min=eps)
        iou = inter / union                                        # (B, M)
        iou = torch.where(gt_mask, iou, torch.full_like(iou, -1.0))
        j_star = iou.argmax(dim=-1)                                # (B,)

    return gt_boxes.gather(
        1, j_star.view(B, 1, 1).expand(-1, 1, 4)
    ).squeeze(1)


def _giou_match_target(
    pred: torch.Tensor,       # (B, 4) xywh
    gt_boxes: torch.Tensor,   # (B, M, 4) xywh
    gt_mask: torch.Tensor,    # (B, M) bool
    eps: float = 1e-10,
) -> torch.Tensor:
    """Select each image's GT target by maximum GIoU against the prediction.

    Identical rule to the matcher inlined in `localization_giou_loss`,
    including the -2.0 padding sentinel (chosen below GIoU's attainable -1.0
    floor so a padded slot can never beat a legitimately -1.0 real GT).

    Runs under `torch.no_grad()`; the gathered target carries no gradient.
    """
    B, M, _ = gt_boxes.shape

    with torch.no_grad():
        pred_exp = pred.unsqueeze(1).expand(-1, M, -1)             # (B, M, 4)
        giou_mat = _giou_xywh(pred_exp, gt_boxes, eps=eps)         # (B, M)
        giou_mat = torch.where(gt_mask, giou_mat, torch.full_like(giou_mat, -2.0))
        j_star = giou_mat.argmax(dim=-1)                           # (B,)

    return gt_boxes.gather(
        1, j_star.view(B, 1, 1).expand(-1, 1, 4)
    ).squeeze(1)


def _l1_giou_reduce(
    pred: torch.Tensor,       # (B, 4) xywh
    target: torch.Tensor,     # (B, 4) xywh, already matched
    gt_mask: torch.Tensor,    # (B, M) bool
    l1_weight: float,
    giou_weight: float,
    eps: float,
) -> torch.Tensor:
    """Combine the L1 and (1 - GIoU) terms and average over rows with a GT.

    L_b = l1_weight * mean_coord |pred_b - target_b|
          + giou_weight * (1 - GIoU(pred_b, target_b))

    The L1 term is taken on the xywh parameterization, matching DETR (Carion
    et al., ECCV 2020), whose box loss is exactly this pair. The two terms are
    complementary: L1 is scale-sensitive (a fixed coordinate error costs the
    same on a large and a small box) while GIoU is scale-invariant, so the sum
    keeps a usable gradient at both extremes.

    Images with no valid GT are dropped from the mean rather than regressed
    toward a padded zero box, matching `localization_giou_loss` and
    `iou.batched_max_iou`. Note this differs from the mse / l1 arms, which run
    `F.mse_loss` / `F.l1_loss` over the whole batch including such rows; the
    GIoU family is kept internally consistent instead so its arms stay
    comparable to each other.
    """
    per_sample = giou_weight * (1.0 - _giou_xywh(pred, target, eps=eps))   # (B,)
    if l1_weight != 0.0:
        per_sample = per_sample + l1_weight * (pred - target).abs().mean(dim=-1)

    has_gt = gt_mask.any(dim=-1).to(per_sample.dtype)                      # (B,)
    return (per_sample * has_gt).sum() / has_gt.sum().clamp(min=1.0)


def localization_giou_iou_match_loss(
    pred: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_mask: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """1 - GIoU against the GT with highest *IoU* vs the prediction.

    The matching ablation for `localization_giou_loss`: same objective, but
    the target is chosen by argmax-IoU (the rule `mse_iou_match` uses) instead
    of argmax-GIoU. Isolates how much of the `giou` arm's behaviour comes from
    the loss and how much from the matcher.
    """
    target = _iou_match_target(pred, gt_boxes, gt_mask, eps=eps)
    return _l1_giou_reduce(pred, target, gt_mask, 0.0, 1.0, eps)


def localization_giou_centroid_match_loss(
    pred: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_mask: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """1 - GIoU against the GT with the closest centroid to the prediction.

    Matcher: min-L2 on (x_c, y_c), see `_centroid_match_target`. This is the
    matcher the paper's L1 / MSE curves use, so this arm is the closest
    like-for-like GIoU counterpart to them.
    """
    target = _centroid_match_target(pred, gt_boxes, gt_mask)
    return _l1_giou_reduce(pred, target, gt_mask, 0.0, 1.0, eps)


def localization_l1_giou_loss(
    pred: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_mask: torch.Tensor,
    l1_weight: float = 5.0,
    giou_weight: float = 2.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """DETR-style box loss (L1 on xywh plus 1 - GIoU), argmax-GIoU matched.

    Default weights 5.0 / 2.0 are DETR's. Matches `localization_giou_loss`'s
    argmax-GIoU rule so the two differ only by the added L1 term.
    """
    target = _giou_match_target(pred, gt_boxes, gt_mask, eps=eps)
    return _l1_giou_reduce(pred, target, gt_mask, l1_weight, giou_weight, eps)


def localization_l1_giou_iou_match_loss(
    pred: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_mask: torch.Tensor,
    l1_weight: float = 5.0,
    giou_weight: float = 2.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """DETR-style L1 + (1 - GIoU), argmax-IoU matched."""
    target = _iou_match_target(pred, gt_boxes, gt_mask, eps=eps)
    return _l1_giou_reduce(pred, target, gt_mask, l1_weight, giou_weight, eps)


def localization_l1_giou_centroid_match_loss(
    pred: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_mask: torch.Tensor,
    l1_weight: float = 5.0,
    giou_weight: float = 2.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """DETR-style L1 + (1 - GIoU), min-centroid-L2 matched."""
    target = _centroid_match_target(pred, gt_boxes, gt_mask)
    return _l1_giou_reduce(pred, target, gt_mask, l1_weight, giou_weight, eps)


# ---------------------------------------------------------------------------
# 4. Expected-IoU ordinal CE — exact supervised analog of TailRL for IoU reward
# ---------------------------------------------------------------------------

def localization_tailrl_population_loss(
    logits: dict[str, torch.Tensor],     # each (B, K)
    gt_boxes: torch.Tensor,              # (B, M, 4) xywh in [0, 1]
    gt_mask: torch.Tensor,               # (B, M) bool — valid GT rows
    K: int,
    n_thresholds: int = 100,
    eps: float = 1e-4,
    clamp_pred: bool = False,
) -> torch.Tensor:
    """Expected-IoU ordinal CE loss — exact infinite-N limit of TailRL for IoU.

    L = -∫_0^1 log P_π(max_j IoU(b_hat, GT_j) > τ) dτ
      ≈ -Σ_{j=1}^{T-1} (1/T) log P_π(max-over-M IoU > j/T)   [left-Riemann]

    Derivation: for the reward r = max_j IoU, the score-function policy-gradient
    identity gives ∇_θ E[w(r)] = 0 (cumulative-hazard weight has policy-
    invariant expectation). Decomposing r as an integral of threshold
    indicators and using the binary-MaxRL Prop 3.3 per threshold, the TailRL
    gradient at N→∞ equals -∇_θ L_ord-IoU.

    Implementation:
      1. Compute max-over-M-GT IoU tensor (B, K, K, K, K) via factored int_x,
         int_y per GT slot; elementwise-max across slots.
      2. Bucketize IoU into T = n_thresholds buckets.
      3. Materialize joint probability tensor (B, K, K, K, K) via outer product
         of the 4 softmax vectors.
      4. Scatter-add joint probs into per-bucket histogram.
      5. Cumulative sum → P(IoU > τ_j) = 1 - cum[j-1] for j=1..T-1.
      6. L = -(1/T) Σ log P(τ_j).

    Memory: O(B K^4) peak (IoU tensor + joint tensor + bucket indices).
    Compute: O(B K^4) per step. At B=128, K=50, this is ~800M ops.

    Args:
        logits:        dict keyed by HEAD_NAMES, each (B, K)
        gt_boxes:      (B, M, 4) normalized xywh coordinates
        gt_mask:       (B, M) bool — which rows are real GT (vs padding)
        K:             number of bins per coordinate
        n_thresholds:  T; number of grid points for the τ integral
        eps:           clamp for log(P) when P ≈ 0 and for zero-area union.
                       Must match ``TAILRL_SURVIVAL_EPS`` in ``advantages.py``
                       for the TailRL → tailrl_population identity (N → ∞) to hold.
        clamp_pred:    if True, clamp predicted box corners to [0, 1] before
                       computing intersection/union, matching the RL path's
                       clamp_boxes_to_image. The predicted area becomes 4-D
                       over (xc, yc, w, h) because the clamped width depends
                       on xc (and clamped height on yc). Default False
                       (baseline behavior — bit-identical to the pre-flag version).

    Returns:
        scalar loss, positive value, averaged over batch.
    """
    device = logits['x_c'].device
    dtype = logits['x_c'].dtype
    B = logits['x_c'].size(0)
    M = gt_boxes.size(1)
    T = n_thresholds

    # 1. Softmax per head
    p_xc = F.softmax(logits['x_c'], dim=-1)   # (B, K)
    p_yc = F.softmax(logits['y_c'], dim=-1)
    p_w  = F.softmax(logits['w'],   dim=-1)
    p_h  = F.softmax(logits['h'],   dim=-1)

    # 2. Bin centers
    bc = (torch.arange(K, device=device, dtype=dtype) + 0.5) / K   # (K,)

    # 3. Predicted corner intervals per (k1, k3) / (k2, k4)
    px1 = bc.view(K, 1) - bc.view(1, K) / 2    # (K, K)
    px2 = bc.view(K, 1) + bc.view(1, K) / 2
    py1 = bc.view(K, 1) - bc.view(1, K) / 2
    py2 = bc.view(K, 1) + bc.view(1, K) / 2

    if clamp_pred:
        # Mirror clamp_boxes_to_image in the RL path: corners pinned to [0, 1].
        # After clamping, the predicted width depends on (xc, w) jointly, so
        # pred_area becomes 4-D over (xc, yc, w, h).
        px1 = px1.clamp(0.0, 1.0)
        px2 = px2.clamp(0.0, 1.0)
        py1 = py1.clamp(0.0, 1.0)
        py2 = py2.clamp(0.0, 1.0)
        pred_w = (px2 - px1).clamp(min=0.0)                              # (K_xc, K_w)
        pred_h = (py2 - py1).clamp(min=0.0)                              # (K_yc, K_h)
        # axis order matches iou_m: (xc, yc, w, h)
        pred_area = pred_w[:, None, :, None] * pred_h[None, :, None, :]  # (K_xc, K_yc, K_w, K_h)
    else:
        # Predicted area (K, K) over (k3, k4) = w × h (full, unclamped)
        pred_area = bc[:, None] * bc[None, :]                            # (K, K)

    # 4. Max-over-M-GT IoU tensor (B, K, K, K, K)
    iou_max = torch.full(
        (B, K, K, K, K), -1.0, device=device, dtype=dtype
    )
    for m in range(M):
        gt_m    = gt_boxes[:, m]                 # (B, 4)
        valid   = gt_mask[:, m]                  # (B,) bool
        gt_xc, gt_yc, gt_w, gt_h = gt_m.unbind(-1)
        gt_x1 = gt_xc - gt_w / 2
        gt_x2 = gt_xc + gt_w / 2
        gt_y1 = gt_yc - gt_h / 2
        gt_y2 = gt_yc + gt_h / 2
        gt_area = (gt_w * gt_h).clamp(min=eps)

        # int_x: (B, K, K) [k1, k3]
        int_x_l = torch.maximum(px1[None], gt_x1[:, None, None])
        int_x_r = torch.minimum(px2[None], gt_x2[:, None, None])
        int_x = (int_x_r - int_x_l).clamp(min=0.0)

        # int_y: (B, K, K) [k2, k4]
        int_y_t = torch.maximum(py1[None], gt_y1[:, None, None])
        int_y_b = torch.minimum(py2[None], gt_y2[:, None, None])
        int_y = (int_y_b - int_y_t).clamp(min=0.0)

        # IoU (B, K, K, K, K) = inter / (pred_area + gt_area - inter)
        inter = int_x[:, :, None, :, None] * int_y[:, None, :, None, :]
        if clamp_pred:
            # pred_area shape (K_xc, K_yc, K_w, K_h) → broadcast as (1, ...)
            union = (
                pred_area[None]
                + gt_area[:, None, None, None, None]
                - inter
            )
        else:
            # pred_area shape (K_w, K_h) → broadcast as (1, 1, 1, K_w, K_h)
            union = (
                pred_area[None, None, None, :, :]
                + gt_area[:, None, None, None, None]
                - inter
            )
        iou_m = inter / union.clamp(min=eps)

        # Zero-out (actually set to -1) for invalid GT rows
        iou_m = torch.where(
            valid[:, None, None, None, None],
            iou_m,
            torch.full_like(iou_m, -1.0),
        )

        iou_max = torch.maximum(iou_max, iou_m)

        # Explicit free of per-slot tensors
        del int_x, int_y, inter, union, iou_m

    # If a whole image had zero valid GTs, iou_max stays at -1; clip to 0.
    iou_max = iou_max.clamp(min=0.0, max=1.0)
    iou_max = iou_max.detach()   # no gradient flows through IoU values

    # 5. Bucketize into T buckets
    bucket_idx = (iou_max * T).clamp(max=T - 1).long()   # (B, K, K, K, K)

    # 6. Joint probability (B, K, K, K, K) via outer product
    joint = (
        p_xc[:, :, None, None, None] *
        p_yc[:, None, :, None, None] *
        p_w [:, None, None, :, None] *
        p_h [:, None, None, None, :]
    )

    # 7. Scatter-add joint into per-bucket histogram
    joint_flat = joint.reshape(B, -1)                    # (B, K^4)
    bucket_flat = bucket_idx.reshape(B, -1)              # (B, K^4)
    hist = torch.zeros(B, T, device=device, dtype=dtype)
    hist.scatter_add_(1, bucket_flat, joint_flat)        # (B, T)

    # 8. Cumulative sum — cum[b, j] = P(IoU <= j/T)
    cum = hist.cumsum(dim=-1)                            # (B, T)

    # 9. P(τ_j) = P(IoU > τ_j) for τ_j = j/T, j=1..T-1:
    #    P(τ_j) = 1 - cum[j-1]
    # For j=0, P = 1 so log(P)=0 (no contribution).
    P_tau = (1.0 - cum[:, :-1]).clamp(min=eps)           # (B, T-1)

    # 10. Left-Riemann sum with dtau = 1/T
    loss = -(1.0 / T) * torch.log(P_tau).sum(dim=-1).mean()

    return loss
