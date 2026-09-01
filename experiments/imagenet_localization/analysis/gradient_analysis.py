"""Gradient + cosine analysis for the ImageNet localization RL experiment.

Mirrors experiment5_ordinal/gradient_analysis.py so the same plotting code
can consume both experiments' outputs. Three analysis modes, each writing a
JSON file whose schema matches experiment5's:

  ``per_image``  → gradient_per_image_epoch{E}.json
      List[dict] with one row per val image: gradient L2 norm + analytical
      quantities (iou_greedy, expected_iou, ordinal_ce_value, entropy).

  ``per_threshold`` → gradient_per_threshold_epoch{E}.json
      List[dict] over (image, τ) pairs: P(max-over-GT IoU > τ) and
      ‖∇_heads -log P(IoU > τ)‖_2. Only makes sense for policy methods
      whose loss decomposes into an integral over thresholds (tailrl,
      tailrl_population, ordinal_ce).

  ``cosine``  → cosine_vs_G_epoch{E}[_ref_{cross_entropy|mse}].json
      {epoch, n_images, n_trials, ref_method, results:
       {method: {G: {mean, std, values, mag_ratio_mean, mag_ratio_std,
                     mag_ratio_values}}}} — cosine similarity and magnitude
      ratio between the policy-gradient estimator at G rollouts and the
      supervised reference gradient.

All gradients are computed over the 4 classification heads only (the
``model.heads`` parameters) — the shared backbone is excluded because it
otherwise dominates the norms and conflates policy/supervised comparisons.

CLI:
  # per-image + (when eligible) per-threshold:
  python -m experiments.imagenet_localization.analysis.gradient_analysis per_image \
      --checkpoint PATH --method tailrl --K 50 --output_dir DIR

  # cosine vs G:
  python -m experiments.imagenet_localization.analysis.gradient_analysis cosine \
      --checkpoint PATH --K 50 --ref_method ordinal_ce --output_dir DIR
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.imagenet_localization import paths
from experiments.imagenet_localization.core.advantages import ADVANTAGE_FNS
from experiments.imagenet_localization.datasets.data import (
    ImageNetLocDataset, build_collate_fn,
)
from experiments.imagenet_localization.evaluation.evaluate import (
    SIZE_TIERS, size_tier_of_primary_box,
)
from experiments.imagenet_localization.core.iou import (
    batched_max_iou, clamp_boxes_to_image,
)
from experiments.imagenet_localization.core.losses import (
    factored_sample_log_prob,
    localization_ordinal_ce_loss,
    localization_cross_entropy_loss,
    localization_tailrl_population_loss,
    localization_mse_iou_match_loss,
)
from experiments.imagenet_localization.models.model import LocalizationPolicy


HEAD_NAMES: tuple[str, ...] = ("x_c", "y_c", "w", "h")

# Methods for which per-threshold decomposition is meaningful (the loss can
# be written as an integral of -log survival). Other methods are skipped.
PER_THRESHOLD_METHODS: frozenset[str] = frozenset({
    "ordinal_ce", "tailrl_population", "tailrl",
})


# =============================================================================
# Gradient helpers (head-only)
# =============================================================================

def head_params(model: LocalizationPolicy) -> list[torch.nn.Parameter]:
    return [p for p in model.heads.parameters() if p.requires_grad]


def flatten_head_grads(params) -> torch.Tensor:
    """Concat `p.grad` across `params` into a 1D tensor."""
    return torch.cat([
        (p.grad.detach() if p.grad is not None else torch.zeros_like(p)).flatten()
        for p in params
    ])


def zero_head_grads(params) -> None:
    for p in params:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


# =============================================================================
# Single-image loss helpers — return scalar loss WITH grad so caller can .backward()
# =============================================================================

def _ordinal_ce_loss_single(
    logits: dict[str, torch.Tensor], target_bins: dict[str, torch.Tensor], K: int,
) -> torch.Tensor:
    return localization_ordinal_ce_loss(logits, target_bins, K)


def _cross_entropy_loss_single(
    logits: dict[str, torch.Tensor], target_bins: dict[str, torch.Tensor],
) -> torch.Tensor:
    return localization_cross_entropy_loss(logits, target_bins)


def _mse_ref_loss_single(
    logits: dict[str, torch.Tensor],
    gt_boxes: torch.Tensor, gt_mask: torch.Tensor,
    K: int,
) -> torch.Tensor:
    """MSE reference loss for cosine analysis on a policy model.

    The MSE regressor baseline uses a 1-output regression head, incompatible
    with the 4 × K softmax heads. To compare gradients on the SAME parameter
    space, take the policy's *expected box* (soft argmax over bin centers) as
    a point prediction and apply MSE (IoU-match variant) against the GT boxes.
    """
    bin_centers = (torch.arange(K, device=logits["x_c"].device,
                                dtype=logits["x_c"].dtype) + 0.5) / K  # (K,)
    pred_parts = []
    for h in HEAD_NAMES:
        probs = F.softmax(logits[h], dim=-1)        # (1, K)
        pred_parts.append((probs * bin_centers).sum(dim=-1))  # (1,)
    pred = torch.stack(pred_parts, dim=-1)           # (1, 4)
    return localization_mse_iou_match_loss(pred, gt_boxes, gt_mask)


def _rl_loss_single(
    logits: dict[str, torch.Tensor],
    gt_boxes: torch.Tensor, gt_mask: torch.Tensor,
    K: int, G: int, advantage_fn,
) -> torch.Tensor:
    """One-image policy-gradient loss with G rollouts."""
    log_probs = {h: F.log_softmax(logits[h], dim=-1) for h in HEAD_NAMES}
    with torch.no_grad():
        probs = {h: log_probs[h].exp() for h in HEAD_NAMES}
        samples = {h: torch.multinomial(probs[h], G, replacement=True)
                   for h in HEAD_NAMES}
        coords = torch.stack(
            [(samples[h].float() + 0.5) / K for h in HEAD_NAMES], dim=-1,
        )  # (1, G, 4)
        sampled_boxes = clamp_boxes_to_image(coords)
        rewards = batched_max_iou(sampled_boxes, gt_boxes, gt_mask)  # (1, G)
        advantages = advantage_fn(rewards[0]).unsqueeze(0)            # (1, G)
    sample_log_probs = factored_sample_log_prob(log_probs, samples)   # (1, G)
    return -(advantages.detach() * sample_log_probs).mean()


# =============================================================================
# Analytical per-image quantities (iou_greedy, expected_iou, ...)
# =============================================================================

@torch.no_grad()
def _analytical_quantities(
    logits: dict[str, torch.Tensor],   # each (1, K)
    gt_boxes: torch.Tensor,            # (1, MAX_M, 4)
    gt_mask: torch.Tensor,             # (1, MAX_M)
    target_bins: dict[str, torch.Tensor],  # each (1,) long
    K: int,
    N_eval_samples: int = 1024,
) -> dict:
    """Compute greedy IoU, expected IoU, entropy, ordinal_ce_value for ONE image."""
    # Greedy box → max IoU
    greedy_coords = torch.stack(
        [(logits[h].argmax(dim=-1).float() + 0.5) / K for h in HEAD_NAMES],
        dim=-1,
    )  # (1, 4)
    greedy_iou = batched_max_iou(
        greedy_coords.unsqueeze(1), gt_boxes, gt_mask
    ).squeeze().item()

    # Sampled IoU → expected + CorLoc tail frequencies
    probs = {h: F.softmax(logits[h], dim=-1) for h in HEAD_NAMES}
    samples = {h: torch.multinomial(probs[h], N_eval_samples, replacement=True)
               for h in HEAD_NAMES}
    coords = torch.stack(
        [(samples[h].float() + 0.5) / K for h in HEAD_NAMES], dim=-1,
    )  # (1, N, 4)
    sampled_boxes = clamp_boxes_to_image(coords)
    sample_ious = batched_max_iou(sampled_boxes, gt_boxes, gt_mask).squeeze(0)  # (N,)
    expected_iou = sample_ious.mean().item()
    p_iou_at_50 = (sample_ious > 0.5).float().mean().item()
    p_iou_at_75 = (sample_ious > 0.75).float().mean().item()
    p_iou_at_90 = (sample_ious > 0.9).float().mean().item()

    # Entropy (joint across 4 heads — independent factors, so sum entropies)
    entropies = []
    for h in HEAD_NAMES:
        p = probs[h].clamp(min=1e-12)
        entropies.append(-(p * p.log()).sum().item())
    entropy_joint = sum(entropies)

    # Ordinal CE scalar for this image (positive loss, higher = worse)
    ordinal_ce_val = localization_ordinal_ce_loss(logits, target_bins, K).item()
    cross_entropy_val = localization_cross_entropy_loss(logits, target_bins).item()

    return {
        "iou_greedy": float(greedy_iou),
        "expected_iou": float(expected_iou),
        "p_iou_at_50": float(p_iou_at_50),
        "p_iou_at_75": float(p_iou_at_75),
        "p_iou_at_90": float(p_iou_at_90),
        "entropy_joint": float(entropy_joint),
        "ordinal_ce_value": float(ordinal_ce_val),
        "cross_entropy_value": float(cross_entropy_val),
    }


# =============================================================================
# Per-image gradient analysis
# =============================================================================

def compute_per_image(
    model: LocalizationPolicy,
    images: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_mask: torch.Tensor,
    target_bins: dict[str, torch.Tensor],
    K: int,
    device: torch.device,
    method: str,
    G_for_rl: int = 4096,
) -> list[dict]:
    """One gradient per image, plus analytical quantities.

    Supervised methods use their own loss (ordinal_ce / cross_entropy). RL methods
    use a policy-gradient estimate with G_for_rl rollouts (a large G so the
    per-image gradient estimate is close to the expected-reward gradient).
    """
    model.eval()
    params = head_params(model)
    n = images.size(0)
    out: list[dict] = []

    advantage_fn = ADVANTAGE_FNS.get(method)

    for i in range(n):
        im = images[i:i+1]
        gb = gt_boxes[i:i+1]
        gm = gt_mask[i:i+1]
        tb = {h: target_bins[h][i:i+1] for h in HEAD_NAMES}

        model.zero_grad()
        logits = model(im)

        if method == "ordinal_ce":
            loss = _ordinal_ce_loss_single(logits, tb, K)
        elif method == "cross_entropy":
            loss = _cross_entropy_loss_single(logits, tb)
        elif method == "tailrl_population":
            loss = localization_tailrl_population_loss(logits, gb, gm, K)
        elif method in ADVANTAGE_FNS:
            loss = _rl_loss_single(logits, gb, gm, K, G_for_rl, advantage_fn)
        else:
            raise ValueError(
                f"Unsupported method for per-image analysis: {method!r}"
            )

        loss.backward()
        g_norm = flatten_head_grads(params).norm().item()

        row = {
            "image_idx": int(i),
            "gradient_l2_norm": float(g_norm),
            "method": method,
        }
        # Recompute for analytical stats under no_grad (fresh forward pass).
        with torch.no_grad():
            logits_eval = model(im)
        row.update(_analytical_quantities(logits_eval, gb, gm, tb, K))
        out.append(row)

        model.zero_grad()

    return out


# =============================================================================
# Per-threshold gradient analysis (survival curves)
# =============================================================================

def _iou_max_tensor_single(
    gt_boxes: torch.Tensor,     # (1, M, 4)
    gt_mask:  torch.Tensor,     # (1, M)
    K: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Max-over-valid-GT IoU tensor of shape (K, K, K, K) for one image.

    Same derivation as in ``localization_tailrl_population_loss`` but specialized
    to batch=1; kept local to avoid round-tripping gradients through the full
    loss path in the per-threshold code.
    """
    bc = (torch.arange(K, device=device, dtype=dtype) + 0.5) / K   # (K,)
    px1 = bc.view(K, 1) - bc.view(1, K) / 2      # (K, K)
    px2 = bc.view(K, 1) + bc.view(1, K) / 2
    py1 = bc.view(K, 1) - bc.view(1, K) / 2
    py2 = bc.view(K, 1) + bc.view(1, K) / 2
    pred_area = bc[:, None] * bc[None, :]        # (K, K)

    iou_max = torch.full((K, K, K, K), -1.0, device=device, dtype=dtype)
    M = gt_boxes.size(1)
    for m in range(M):
        valid = gt_mask[0, m].item()
        if not valid:
            continue
        gt = gt_boxes[0, m]
        gt_xc, gt_yc, gt_w, gt_h = gt.unbind(-1)
        gt_x1 = gt_xc - gt_w / 2; gt_x2 = gt_xc + gt_w / 2
        gt_y1 = gt_yc - gt_h / 2; gt_y2 = gt_yc + gt_h / 2
        gt_area = (gt_w * gt_h).clamp(min=eps)

        ix_l = torch.maximum(px1, gt_x1); ix_r = torch.minimum(px2, gt_x2)
        iy_t = torch.maximum(py1, gt_y1); iy_b = torch.minimum(py2, gt_y2)
        int_x = (ix_r - ix_l).clamp(min=0.0)   # (K, K)
        int_y = (iy_b - iy_t).clamp(min=0.0)

        inter = int_x[:, None, :, None] * int_y[None, :, None, :]   # (K,K,K,K)
        union = (
            pred_area[None, None, :, :] + gt_area.view(1, 1, 1, 1) - inter
        )
        iou_m = inter / union.clamp(min=eps)
        iou_max = torch.maximum(iou_max, iou_m)

    iou_max = iou_max.clamp(min=0.0, max=1.0)
    return iou_max


def compute_per_threshold(
    model: LocalizationPolicy,
    images: torch.Tensor,          # (n, 3, H, W)
    gt_boxes: torch.Tensor,        # (n, M, 4)
    gt_mask: torch.Tensor,         # (n, M)
    K: int,
    device: torch.device,
    thresholds: Optional[list[float]] = None,
) -> list[dict]:
    """Per-(image, τ) survival + gradient norm of -log P(IoU > τ).

    Memory: O(K^4) per image for the IoU + joint tensors (~25 MB at K=50).
    """
    if thresholds is None:
        # 19 thresholds spanning the interior of [0, 1].
        thresholds = [round(0.05 * t, 2) for t in range(1, 20)]

    params = head_params(model)
    out: list[dict] = []
    dtype = next(model.parameters()).dtype

    for i in range(images.size(0)):
        im = images[i:i+1]
        gb = gt_boxes[i:i+1]
        gm = gt_mask[i:i+1]

        # IoU tensor depends only on GT + bin-centers; compute once per image.
        iou_max = _iou_max_tensor_single(gb, gm, K, device, dtype)  # (K,K,K,K)

        for tau in thresholds:
            model.zero_grad()
            logits = model(im)
            p_xc = F.softmax(logits['x_c'], dim=-1)[0]  # (K,)
            p_yc = F.softmax(logits['y_c'], dim=-1)[0]
            p_w  = F.softmax(logits['w'],   dim=-1)[0]
            p_h  = F.softmax(logits['h'],   dim=-1)[0]
            joint = (
                p_xc[:, None, None, None] *
                p_yc[None, :, None, None] *
                p_w [None, None, :, None] *
                p_h [None, None, None, :]
            )  # (K,K,K,K)
            mask = (iou_max > tau).to(joint.dtype)
            P_tau = (joint * mask).sum().clamp(min=1e-12)
            loss = -torch.log(P_tau)
            loss.backward()
            g_norm = flatten_head_grads(params).norm().item()
            survival = P_tau.item()

            out.append({
                "image_idx": int(i),
                "threshold_d": float(tau),
                "survival_prob": float(survival),
                "gradient_norm_at_threshold": float(g_norm),
                "log_survival": float(np.log(max(survival, 1e-12))),
            })

            model.zero_grad()
            del joint, mask, P_tau, loss, logits
        del iou_max

    return out


# =============================================================================
# Cosine similarity vs G
# =============================================================================

def _accumulate_ref_gradient(
    model: LocalizationPolicy,
    images: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_mask: torch.Tensor,
    target_bins: dict[str, torch.Tensor],
    K: int,
    ref_method: str,
) -> torch.Tensor:
    """Compute mean-over-images reference gradient (head params, flat)."""
    params = head_params(model)
    model.zero_grad()
    n = images.size(0)
    for i in range(n):
        im = images[i:i+1]
        gb = gt_boxes[i:i+1]
        gm = gt_mask[i:i+1]
        tb = {h: target_bins[h][i:i+1] for h in HEAD_NAMES}
        logits = model(im)
        if ref_method == "ordinal_ce":
            loss = _ordinal_ce_loss_single(logits, tb, K)
        elif ref_method == "cross_entropy":
            loss = _cross_entropy_loss_single(logits, tb)
        elif ref_method == "mse":
            loss = _mse_ref_loss_single(logits, gb, gm, K)
        elif ref_method == "tailrl_population":
            loss = localization_tailrl_population_loss(logits, gb, gm, K)
        elif ref_method == "tailrl_population_clamped":
            loss = localization_tailrl_population_loss(
                logits, gb, gm, K, clamp_pred=True,
            )
        else:
            raise ValueError(f"unknown ref_method: {ref_method!r}")
        (loss / n).backward()
    flat = flatten_head_grads(params).clone()
    model.zero_grad()
    return flat


def _accumulate_rl_gradient(
    model: LocalizationPolicy,
    images: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_mask: torch.Tensor,
    K: int,
    G: int,
    advantage_fn,
) -> torch.Tensor:
    """Mean-over-images RL gradient at G rollouts/image."""
    params = head_params(model)
    model.zero_grad()
    n = images.size(0)
    for i in range(n):
        im = images[i:i+1]
        gb = gt_boxes[i:i+1]
        gm = gt_mask[i:i+1]
        logits = model(im)
        loss = _rl_loss_single(logits, gb, gm, K, G, advantage_fn)
        (loss / n).backward()
    flat = flatten_head_grads(params).clone()
    model.zero_grad()
    return flat


def compute_cosine_vs_G(
    model: LocalizationPolicy,
    images: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_mask: torch.Tensor,
    target_bins: dict[str, torch.Tensor],
    K: int,
    device: torch.device,
    G_values: Optional[list[int]] = None,
    methods: Optional[list[str]] = None,
    ref_method: str = "ordinal_ce",
    n_trials: int = 10,
) -> dict:
    """{method: {G: {mean, std, values, mag_ratio_mean, mag_ratio_std,
                     mag_ratio_values}}}"""
    if G_values is None:
        G_values = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096]
    if methods is None:
        methods = list(ADVANTAGE_FNS.keys())

    ref_grad = _accumulate_ref_gradient(
        model, images, gt_boxes, gt_mask, target_bins, K, ref_method,
    )
    ref_norm = ref_grad.norm().item()
    print(f"  Reference ({ref_method}) head-grad norm: {ref_norm:.4f}")

    results: dict = {m: {} for m in methods}
    for method in methods:
        adv_fn = ADVANTAGE_FNS[method]
        for G in G_values:
            coss: list[float] = []
            mags: list[float] = []
            for _ in range(n_trials):
                rl_grad = _accumulate_rl_gradient(
                    model, images, gt_boxes, gt_mask, K, G, adv_fn,
                )
                cos = F.cosine_similarity(
                    ref_grad.unsqueeze(0), rl_grad.unsqueeze(0),
                ).item()
                rl_norm = rl_grad.norm().item()
                mag = rl_norm / ref_norm if ref_norm > 1e-12 else 0.0
                coss.append(float(cos))
                mags.append(float(mag))
            results[method][str(G)] = {
                "mean": float(np.mean(coss)),
                "std":  float(np.std(coss)),
                "values": coss,
                "mag_ratio_mean": float(np.mean(mags)),
                "mag_ratio_std":  float(np.std(mags)),
                "mag_ratio_values": mags,
            }
            print(f"  {method:>16s}  G={G:5d}  "
                  f"cos={results[method][str(G)]['mean']:.4f}  "
                  f"mag={results[method][str(G)]['mag_ratio_mean']:.4f}")
    return results


# =============================================================================
# Orchestrators — load checkpoint, run analysis, write JSON
# =============================================================================

def _load_policy_from_checkpoint(
    checkpoint_path: str, K: int, device: torch.device, seed: int,
) -> tuple[LocalizationPolicy, int]:
    """Return (model_on_device, epoch)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    epoch = int(ckpt.get("epoch", -1))
    # Construct an un-pretrained policy shell; state_dict brings weights.
    model = LocalizationPolicy(K=K, pretrained=False, seed=seed).to(device)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=True)
    return model, epoch


def _load_val_subset(
    data_dir: str, K: int, seed: int, n_images: int,
    batch_size: int = 32, num_workers: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return (images, gt_boxes, gt_mask, target_bins) all on CPU for ``n_images``."""
    ds = ImageNetLocDataset(
        root_dir=data_dir, split="val", K=K, subsample=n_images,
        seed=seed, train_aug=False,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=build_collate_fn(),
    )
    imgs, gbs, gms = [], [], []
    tbins: dict[str, list[torch.Tensor]] = {h: [] for h in HEAD_NAMES}
    for batch in loader:
        imgs.append(batch["images"])
        gbs.append(batch["gt_boxes"])
        gms.append(batch["gt_mask"])
        for h in HEAD_NAMES:
            tbins[h].append(batch["target_bins"][h])
    return (
        torch.cat(imgs), torch.cat(gbs), torch.cat(gms),
        {h: torch.cat(tbins[h]) for h in HEAD_NAMES},
    )


def run_per_image_and_threshold(
    checkpoint_path: str,
    data_dir: str,
    method: str,
    K: int,
    seed: int,
    output_dir: str,
    n_images: int = 500,
    n_threshold_images: int = 50,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, epoch = _load_policy_from_checkpoint(checkpoint_path, K, device, seed)
    images, gt_boxes, gt_mask, target_bins = _load_val_subset(
        data_dir, K, seed, max(n_images, n_threshold_images),
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- Per-image --------------------------------------------------------
    images_pi = images[:n_images].to(device)
    gb_pi = gt_boxes[:n_images].to(device)
    gm_pi = gt_mask[:n_images].to(device)
    tb_pi = {h: target_bins[h][:n_images].to(device) for h in HEAD_NAMES}

    print(f"[per_image] method={method}  epoch={epoch}  n_images={n_images}")
    per_image = compute_per_image(
        model, images_pi, gb_pi, gm_pi, tb_pi, K, device, method,
    )
    pi_file = out_path / f"gradient_per_image_epoch{epoch}.json"
    with open(pi_file, "w") as f:
        json.dump(per_image, f, indent=2)
    print(f"  wrote {pi_file}")

    # --- Per-threshold (only for methods that decompose this way) --------
    if method in PER_THRESHOLD_METHODS:
        n_thr = min(n_threshold_images, images.size(0))
        images_th = images[:n_thr].to(device)
        gb_th = gt_boxes[:n_thr].to(device)
        gm_th = gt_mask[:n_thr].to(device)
        print(f"[per_threshold] method={method}  epoch={epoch}  n_images={n_thr}")
        per_thr = compute_per_threshold(
            model, images_th, gb_th, gm_th, K, device,
        )
        pt_file = out_path / f"gradient_per_threshold_epoch{epoch}.json"
        with open(pt_file, "w") as f:
            json.dump(per_thr, f, indent=2)
        print(f"  wrote {pt_file}")


def run_cosine_analysis(
    checkpoint_path: str,
    data_dir: str,
    K: int,
    seed: int,
    output_dir: str,
    n_images: int = 200,
    n_trials: int = 10,
    ref_method: str = "ordinal_ce",
    methods: Optional[list[str]] = None,
    G_values: Optional[list[int]] = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, epoch = _load_policy_from_checkpoint(checkpoint_path, K, device, seed)
    images, gt_boxes, gt_mask, target_bins = _load_val_subset(
        data_dir, K, seed, n_images,
    )
    images = images.to(device); gt_boxes = gt_boxes.to(device)
    gt_mask = gt_mask.to(device)
    target_bins = {h: v.to(device) for h, v in target_bins.items()}

    print(f"[cosine] ref={ref_method}  epoch={epoch}  n_images={images.size(0)}  "
          f"n_trials={n_trials}")
    results = compute_cosine_vs_G(
        model, images, gt_boxes, gt_mask, target_bins, K, device,
        G_values=G_values, methods=methods, ref_method=ref_method,
        n_trials=n_trials,
    )
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    suffix = f"_ref_{ref_method}" if ref_method != "ordinal_ce" else ""
    out_file = out_path / f"cosine_vs_G_epoch{epoch}{suffix}.json"
    with open(out_file, "w") as f:
        json.dump({
            "epoch": epoch,
            "n_images": int(images.size(0)),
            "n_trials": int(n_trials),
            "ref_method": ref_method,
            "results": results,
        }, f, indent=2)
    print(f"  wrote {out_file}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("per_image", "cosine"),
                   help="per_image: gradient_per_image + per_threshold. "
                        "cosine: cosine_vs_G.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", default=paths.imagenet_dir() or None,
                   help="Root ImageNet directory (default: $IMAGENET_DIR).")
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", required=True,
                   help="Usually <run_dir>/gradient_analysis/")

    # per_image
    p.add_argument("--method", default=None,
                   help="per_image mode: the method whose gradient to compute.")
    p.add_argument("--n_images", type=int, default=500)
    p.add_argument("--n_threshold_images", type=int, default=50)

    # cosine
    p.add_argument("--ref_method", default="ordinal_ce",
                   choices=("ordinal_ce", "cross_entropy", "mse", "tailrl_population"))
    p.add_argument("--methods", default=None,
                   help="cosine mode: comma-separated list of RL methods "
                        "(default: all 6).")
    p.add_argument("--Gs", default="4,8,16,32,64,128,256,512,1024,4096",
                   help="Comma-separated rollout counts (cosine mode).")
    p.add_argument("--n_trials", type=int, default=10,
                   help="Trials per (method, G) for variance estimation.")
    args = p.parse_args()
    if not args.data_dir:
        args.data_dir = paths.require_imagenet_dir()
    return args


def main():
    args = parse_args()
    if args.mode == "per_image":
        if args.method is None:
            raise SystemExit("per_image mode requires --method")
        run_per_image_and_threshold(
            args.checkpoint, args.data_dir, args.method, args.K,
            args.seed, args.output_dir,
            n_images=args.n_images,
            n_threshold_images=args.n_threshold_images,
        )
    else:
        methods = None
        if args.methods:
            methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        G_values = [int(g) for g in args.Gs.split(",") if g.strip()]
        run_cosine_analysis(
            args.checkpoint, args.data_dir, args.K, args.seed,
            args.output_dir,
            n_images=args.n_images, n_trials=args.n_trials,
            ref_method=args.ref_method, methods=methods,
            G_values=G_values,
        )


if __name__ == "__main__":
    main()


# =============================================================================
# Legacy API (kept for tests/test_gradient_analysis.py).
# =============================================================================

def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Scalar cosine similarity. Returns 0.0 when either input is zero-norm."""
    norm_a = a.norm()
    norm_b = b.norm()
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return (a @ b / (norm_a * norm_b)).item()


def compute_supervised_head_gradient(
    model: LocalizationPolicy,
    image: torch.Tensor,                           # (1, 3, H, W)
    target_bins: dict[str, torch.Tensor],          # each (1,) long
    K: int,
) -> torch.Tensor:
    """Head-only gradient of the ordinal CE loss for a single image."""
    model.zero_grad()
    logits = model(image)
    loss = localization_ordinal_ce_loss(logits, target_bins, K)
    loss.backward()
    return flatten_head_grads(head_params(model))


def compute_rl_head_gradient(
    model: LocalizationPolicy,
    image: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_mask: torch.Tensor,
    K: int,
    G: int,
    advantage_fn,
) -> torch.Tensor:
    """Head-only policy-gradient of one image using G rollouts."""
    model.zero_grad()
    logits = model(image)
    loss = _rl_loss_single(logits, gt_boxes, gt_mask, K, G, advantage_fn)
    loss.backward()
    return flatten_head_grads(head_params(model))


def run_analysis(
    model: LocalizationPolicy,
    images: torch.Tensor,                          # (n, 3, H, W)
    gt_boxes: torch.Tensor,                        # (n, M, 4)
    gt_mask: torch.Tensor,                         # (n, M)
    target_bins: dict[str, torch.Tensor],          # each (n,) long
    K: int,
    methods: list[str],
    Gs: list[int],
) -> dict[str, dict[int, float]]:
    """Legacy: {method: {G: mean_cosine}} over the provided images.

    For each (method, G) pair: for each image, compute supervised head-grad
    and RL head-grad, take cosine similarity, then average over images.
    """
    results: dict[str, dict[int, float]] = {m: {} for m in methods}
    n = images.shape[0]
    for method in methods:
        adv_fn = ADVANTAGE_FNS[method]
        for G in Gs:
            coss: list[float] = []
            for i in range(n):
                im = images[i:i+1]
                gb = gt_boxes[i:i+1]
                gm = gt_mask[i:i+1]
                tb = {h: target_bins[h][i:i+1] for h in HEAD_NAMES}
                g_sup = compute_supervised_head_gradient(model, im, tb, K)
                g_rl = compute_rl_head_gradient(model, im, gb, gm, K, G, adv_fn)
                coss.append(cosine(g_sup, g_rl))
            results[method][G] = float(sum(coss) / max(1, len(coss)))
    return results
