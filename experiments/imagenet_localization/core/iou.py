"""IoU utilities for the ImageNet localization RL experiment.

All boxes are in xywh format: (center_x, center_y, width, height) with
coordinates in [0, 1]. All computations are in continuous space (never bin-space).
"""

from __future__ import annotations

import torch

# Guard constant — prevents division-by-zero in union denominator.
EPS = 1e-10


# ---------------------------------------------------------------------------
# Format conversion helpers
# ---------------------------------------------------------------------------


def xywh_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (..., 4) xywh -> (..., 4) (x1, y1, x2, y2).

    x1 = x_c - w/2,  y1 = y_c - h/2
    x2 = x_c + w/2,  y2 = y_c + h/2
    """
    x_c = boxes[..., 0]
    y_c = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def corners_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (..., 4) (x1, y1, x2, y2) -> (..., 4) xywh.

    x_c = (x1 + x2) / 2,  y_c = (y1 + y2) / 2
    w   = x2 - x1,         h   = y2 - y1
    """
    x1 = boxes[..., 0]
    y1 = boxes[..., 1]
    x2 = boxes[..., 2]
    y2 = boxes[..., 3]
    x_c = (x1 + x2) / 2
    y_c = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([x_c, y_c, w, h], dim=-1)


# ---------------------------------------------------------------------------
# Box clamping
# ---------------------------------------------------------------------------


def clamp_boxes_to_image(boxes: torch.Tensor) -> torch.Tensor:
    """Clamp (..., 4) xywh boxes to the unit square [0, 1].

    Algorithm:
      1. Convert xywh -> corners (x1, y1, x2, y2).
      2. Clamp each corner coordinate independently to [0, 1].
      3. Enforce x2 >= x1 and y2 >= y1 (both already guaranteed by step 2
         since both are clamped to the same range, but a box fully outside
         collapses to a point on the border — zero width/height, never negative).
      4. Convert clamped corners back to xywh.

    A box fully outside the unit square (e.g. x_c=2.0) will have both
    x1 and x2 clamped to 1.0, resulting in w=0 (not negative).
    Returns a tensor with the same dtype and device as `boxes`.
    """
    corners = xywh_to_corners(boxes)  # (..., 4)
    # Clamp every corner independently to [0, 1].
    corners = corners.clamp(0.0, 1.0)
    # After clamping, x2 >= x1 and y2 >= y1 are guaranteed because both were
    # clamped into the same [0,1] interval.  For a box far outside the image,
    # both corners collapse to the same edge value → w=h=0.
    return corners_to_xywh(corners)


# ---------------------------------------------------------------------------
# Pairwise IoU (vectorized)
# ---------------------------------------------------------------------------


def box_iou_xywh(boxes_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """Compute IoU between N boxes and a single reference box.

    Args:
        boxes_a: (N, 4) xywh
        box_b:   (4,)   xywh

    Returns:
        (N,) IoU values in [0, 1].  Zero-area boxes yield 0, never NaN.
    """
    # Convert to corners for easy intersection computation.
    # boxes_a corners: (N, 4)
    a = xywh_to_corners(boxes_a)   # (N, 4)
    b = xywh_to_corners(box_b)     # (4,) -> broadcast against N

    # Intersection
    x1_inter = torch.max(a[..., 0], b[0])
    y1_inter = torch.max(a[..., 1], b[1])
    x2_inter = torch.min(a[..., 2], b[2])
    y2_inter = torch.min(a[..., 3], b[3])

    inter_w = (x2_inter - x1_inter).clamp(min=0)
    inter_h = (y2_inter - y1_inter).clamp(min=0)
    intersection = inter_w * inter_h  # (N,)

    # Areas
    area_a = boxes_a[..., 2] * boxes_a[..., 3]  # w * h, (N,)
    area_b = box_b[2] * box_b[3]                 # scalar

    union = area_a + area_b - intersection  # (N,)

    # Guard: if union <= EPS (covers zero-area cases), return 0 explicitly.
    iou = torch.where(
        union > EPS,
        intersection / (union + EPS),
        torch.zeros_like(intersection),
    )
    return iou


# ---------------------------------------------------------------------------
# Max IoU over ground-truth instances
# ---------------------------------------------------------------------------


def max_iou_over_gt(
    sampled: torch.Tensor,   # (N, 4) xywh
    gt: torch.Tensor,        # (M, 4) xywh, padded rows are ignored
    gt_mask: torch.Tensor,   # (M,) bool, True = real GT row
) -> torch.Tensor:
    """Compute the max IoU of each sampled box against the real GT boxes.

    Args:
        sampled:  (N, 4) xywh — the N candidate predicted boxes.
        gt:       (M, 4) xywh — ground-truth boxes (some may be padding).
        gt_mask:  (M,)  bool  — True for real GT rows, False for padding.

    Returns:
        (N,) tensor of max IoU values in [0, 1].
        If gt_mask is all False (no real GT), returns zeros (not NaN, not -1).
    """
    M = gt.shape[0]
    N = sampled.shape[0]

    # Build (N, M) IoU matrix without an explicit Python loop.
    # Expand sampled to (N, 1, 4) and gt to (1, M, 4), then reshape for
    # box_iou_xywh which expects (N, 4) vs (4,).
    # We compute IoU for each GT box separately and stack.
    # This is fully vectorized in the N dimension per GT, O(M) Python steps max.
    iou_rows = []
    for m in range(M):
        # box_iou_xywh: (N, 4) vs (4,) -> (N,)
        iou_rows.append(box_iou_xywh(sampled, gt[m]))  # (N,)
    iou_matrix = torch.stack(iou_rows, dim=1)  # (N, M)

    # Mask out padded GT rows: set their IoU to -1 so torch.max skips them.
    # gt_mask: (M,) bool -> broadcast to (N, M)
    mask = gt_mask.unsqueeze(0).expand(N, M)  # (N, M)
    iou_matrix = iou_matrix.masked_fill(~mask, -1.0)

    # Edge case: all GT rows padded → every entry is -1 → we must return zeros.
    if not gt_mask.any():
        return torch.zeros(N, dtype=sampled.dtype, device=sampled.device)

    max_iou, _ = iou_matrix.max(dim=1)  # (N,)
    # Clamp to [0, 1] (real GT rows produce IoU in [0, 1] already, but
    # defensive clamp guards against any numerical overshoot).
    return max_iou.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Batched wrapper — fully vectorized over (B, N, M)
# ---------------------------------------------------------------------------


def batched_max_iou(
    sampled: torch.Tensor,   # (B, N, 4) xywh
    gt: torch.Tensor,        # (B, M, 4) xywh, padded
    gt_mask: torch.Tensor,   # (B, M) bool
) -> torch.Tensor:
    """Compute per-image max IoU reward matrix.

    Fully vectorized across (B, N, M) — no Python loop. The intermediate
    (B, N, M) IoU tensor is materialized once and reduced along the M axis.

    Memory: O(B · N · M) for the IoU matrix. At B=128, N=16384, M=10 this is
    ~80 MB; at N=65536 it is ~320 MB. Both fit easily on A5000 (24 GB) /
    A6000 (48 GB).

    Args:
        sampled:  (B, N, 4) xywh in [0, 1].
        gt:       (B, M, 4) xywh in [0, 1], padded rows ignored via gt_mask.
        gt_mask:  (B, M) bool — True for real GT rows.

    Returns:
        (B, N) reward matrix, each entry in [0, 1]. Images with all-False
        gt_mask (no real GTs) contribute zeros, never -1 or NaN.
    """
    # Convert both sets to corners. xywh_to_corners is broadcast-friendly.
    a = xywh_to_corners(sampled)   # (B, N, 4)
    b = xywh_to_corners(gt)        # (B, M, 4)

    # Broadcast: a -> (B, N, 1, 4), b -> (B, 1, M, 4)
    a_e = a.unsqueeze(2)           # (B, N, 1, 4)
    b_e = b.unsqueeze(1)           # (B, 1, M, 4)

    x1_int = torch.maximum(a_e[..., 0], b_e[..., 0])
    y1_int = torch.maximum(a_e[..., 1], b_e[..., 1])
    x2_int = torch.minimum(a_e[..., 2], b_e[..., 2])
    y2_int = torch.minimum(a_e[..., 3], b_e[..., 3])

    inter_w = (x2_int - x1_int).clamp(min=0.0)
    inter_h = (y2_int - y1_int).clamp(min=0.0)
    inter = inter_w * inter_h                                   # (B, N, M)

    area_a = sampled[..., 2] * sampled[..., 3]                  # (B, N)
    area_b = gt[..., 2] * gt[..., 3]                            # (B, M)
    union = area_a.unsqueeze(2) + area_b.unsqueeze(1) - inter   # (B, N, M)

    iou = torch.where(
        union > EPS,
        inter / (union + EPS),
        torch.zeros_like(inter),
    )                                                           # (B, N, M)

    # Mask invalid GT rows so they cannot win the max; -1 < 0 so any real
    # GT (IoU >= 0) outranks them.
    iou = iou.masked_fill(~gt_mask.unsqueeze(1), -1.0)          # (B, N, M)

    max_iou, _ = iou.max(dim=2)                                 # (B, N)

    # Images with no real GT have all-False gt_mask → max=-1; replace with 0.
    no_gt = ~gt_mask.any(dim=1, keepdim=True)                   # (B, 1)
    max_iou = torch.where(no_gt, torch.zeros_like(max_iou), max_iou)

    # Defensive clamp against numerical overshoot (real IoU is in [0, 1]).
    return max_iou.clamp(0.0, 1.0)
