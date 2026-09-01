from __future__ import annotations

from typing import Optional, Callable

import os
import torch
import torch.nn.functional as F

from experiments.imagenet_localization.core.advantages import ADVANTAGE_FNS, REWARD_TRANSFORMS
from experiments.imagenet_localization.core.iou import batched_max_iou, clamp_boxes_to_image
from experiments.imagenet_localization.core.losses import (
    factored_sample_log_prob,
    localization_tailrl_population_loss,
    localization_giou_loss,
    localization_giou_iou_match_loss,
    localization_giou_centroid_match_loss,
    localization_l1_giou_loss,
    localization_l1_giou_iou_match_loss,
    localization_l1_giou_centroid_match_loss,
    localization_ordinal_ce_loss,
    localization_cross_entropy_loss,
    localization_mse_loss,
    localization_mse_iou_match_loss,
    localization_mse_centroid_match_loss,
    localization_l1_iou_match_loss,
    localization_l1_centroid_match_loss,
)
from experiments.imagenet_localization.models.model import LocalizationPolicy  # type hint

HEAD_NAMES: tuple[str, ...] = ("x_c", "y_c", "w", "h")

# GIoU-objective supervised arms keyed by --method. `giou` itself is dispatched
# separately above because it predates this table and its call signature takes
# no weight kwargs. Entries whose name starts with "l1_giou" additionally
# receive l1_weight / giou_weight.
_GIOU_FAMILY_LOSSES = {
    "giou_iou_match":           localization_giou_iou_match_loss,
    "giou_centroid_match":      localization_giou_centroid_match_loss,
    "l1_giou":                  localization_l1_giou_loss,
    "l1_giou_iou_match":        localization_l1_giou_iou_match_loss,
    "l1_giou_centroid_match":   localization_l1_giou_centroid_match_loss,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sample_boxes_from_policy(
    logits: dict[str, torch.Tensor],    # each (B, K)
    N: int,
    K: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Sample N boxes per image via multinomial over each head.

    Returns:
        samples: dict keyed by HEAD_NAMES, each (B, N) long.
        sampled_boxes: (B, N, 4) float, xywh in [0, 1], already clamp_boxes_to_image-ed.
    """
    log_probs = {h: F.log_softmax(logits[h], dim=-1) for h in HEAD_NAMES}
    probs     = {h: log_probs[h].exp()               for h in HEAD_NAMES}

    samples: dict[str, torch.Tensor] = {}
    for h in HEAD_NAMES:
        # probs[h]: (B, K); torch.multinomial on (B, K) with N samples -> (B, N)
        samples[h] = torch.multinomial(probs[h], N, replacement=True)

    # Convert bin indices to box coordinates (bin centers).
    coords = []
    for h in HEAD_NAMES:
        coords.append((samples[h].float() + 0.5) / K)   # (B, N)
    sampled_boxes = torch.stack(coords, dim=-1)         # (B, N, 4)
    sampled_boxes = clamp_boxes_to_image(sampled_boxes) # safe-guard (already in [0,1])

    return samples, sampled_boxes


# ---------------------------------------------------------------------------
# One RL training step (per-batch) — exposed so tests can call it directly.
# ---------------------------------------------------------------------------

def rl_training_step(
    model: LocalizationPolicy,
    batch: dict,
    method: str,
    N: int,
    K: int,
    device: torch.device,
    reward_transform: str = "none",
    pkpo_k: int | None = None,
) -> dict:
    """Run a single RL training step (no optimizer.step — test helper).

    Args:
        model: the policy. Must be on `device` already.
        batch: dict from the collate fn, keys 'images', 'gt_boxes', 'gt_mask'.
        method: one of the ADVANTAGE_FNS keys.
        N: number of rollouts per image.
        K: number of bins per coordinate.
        device: torch device.
        reward_transform: one of the REWARD_TRANSFORMS keys ('none' |
            'percentile'). Applied to the raw IoU rewards before the advantage
            estimator. 'none' is the identity (raw-IoU baseline behaviour).

    Returns:
        dict with keys:
        - 'loss': scalar tensor (with grad).
        - 'rewards': (B, N) float tensor of *raw* IoU (no grad — metrics use this).
        - 'advantages': (B, N) float tensor (no grad).
        - 'sample_log_probs': (B, N) float tensor (has grad).
    """
    advantage_fn = ADVANTAGE_FNS[method]
    transform_fn = REWARD_TRANSFORMS[reward_transform]

    images   = batch['images'].to(device)               # (B, 3, 224, 224)
    # channels_last layout: matches the cuDNN Tensor-Core code path on
    # Ampere (a6000/a5000), gives ~1.5× on ResNet conv. No-op if the model
    # wasn't converted (PyTorch silently falls back).
    if device.type == 'cuda':
        images = images.contiguous(memory_format=torch.channels_last)
    gt_boxes = batch['gt_boxes'].to(device)             # (B, MAX_M, 4)
    gt_mask  = batch['gt_mask'].to(device)              # (B, MAX_M) bool

    # Forward pass: optionally bf16 autocast (toggle via BF16=1 env var).
    # Default off because earlier comparisons showed ~5% train reward drop
    # at end of epoch 1 vs fp32 — likely from the gradient on logits being
    # downcast to bf16 at the autocast boundary.
    use_bf16 = bool(int(os.environ.get("BF16", "0"))) and device.type == 'cuda'
    if use_bf16:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(images)
        log_probs = {h: F.log_softmax(logits[h].float(), dim=-1) for h in HEAD_NAMES}
    else:
        logits = model(images)                           # dict of (B, K)
        log_probs = {h: F.log_softmax(logits[h], dim=-1) for h in HEAD_NAMES}

    # Stack the 4 head probabilities into one (4·B, K) tensor so we can
    # do a single multinomial / single gather instead of 4 separate kernel
    # launches per step. Saves a few % at small N, more at large N where
    # kernel-launch overhead matters less than the memory traffic.
    B = log_probs['x_c'].size(0)
    log_probs_stacked = torch.stack([log_probs[h] for h in HEAD_NAMES], dim=0)  # (4, B, K)

    with torch.no_grad():
        probs_flat = log_probs_stacked.exp().reshape(4 * B, K)              # (4·B, K)
        # One fused multinomial draws N samples for every (head, image).
        samples_flat = torch.multinomial(probs_flat, N, replacement=True)   # (4·B, N)
        samples_stacked = samples_flat.view(4, B, N)                         # (4, B, N)

        # coords: (B, N, 4) — note we permute (4, B, N) → (B, N, 4)
        coords = ((samples_stacked.float() + 0.5) / K).permute(1, 2, 0)
        sampled_boxes = clamp_boxes_to_image(coords)
        rewards = batched_max_iou(sampled_boxes, gt_boxes, gt_mask)         # (B, N)

        # Reward shaping (e.g. percentile) is applied to the raw IoU rewards
        # *before* the advantage estimator. We still return the raw IoU as
        # 'rewards' so the logged metrics keep reporting true IoU — only the
        # advantage path sees the shaped rewards.
        shaped_rewards = transform_fn(rewards)                             # (B, N)

        # advantage_fn operates on the last dim — (B, N) in → (B, N) out.
        # pkpo needs its max@k threshold; every other estimator takes none.
        if method == "pkpo" and pkpo_k is not None:
            advantages = advantage_fn(shaped_rewards, k=pkpo_k)           # (B, N)
        else:
            advantages = advantage_fn(shaped_rewards)                     # (B, N)

    # Fused gather across heads: per_head_lp[h] = log_probs[h].gather(1, samples[h])
    # → stacked: (4, B, N) gathered from (4, B, K) at indices (4, B, N)
    sample_log_probs = log_probs_stacked.gather(2, samples_stacked).sum(dim=0)  # (B, N)

    # PG loss: -mean_b mean_n (advantages[b, n] * log_prob[b, n])
    per_image = -(advantages.detach() * sample_log_probs).mean(dim=1)  # (B,)
    loss = per_image.mean()

    return {
        'loss': loss,
        'rewards': rewards.detach(),
        'advantages': advantages.detach(),
        'sample_log_probs': sample_log_probs.detach(),
    }


# ---------------------------------------------------------------------------
# Epoch loops — the shapes used by run.py
# ---------------------------------------------------------------------------

def train_rl_epoch_localization(
    model: LocalizationPolicy,
    dataloader,
    optimizer,
    scheduler,
    method: str,
    N: int,
    K: int,
    device: torch.device,
    epoch: int,
    grad_clip: float = 10.0,
    reward_transform: str = "none",
    pkpo_k: int | None = None,
    wandb_run=None,
    log_every: int = 50,
    max_steps: int | None = None,
) -> dict:
    """Run one RL training epoch.

    ``max_steps`` stops the epoch after that many optimizer steps. It does NOT
    touch the LR schedule, which is built from the full run length, so a capped
    run sees exactly the learning rates the uncapped run would have seen over
    its first ``max_steps`` steps.

    Returns a dict of running metrics (aggregated over the epoch):
      - loss_mean, reward_mean, reward_std, reward_at_0_5, reward_at_0_75,
        frac_zero_reward, advantage_abs_mean, grad_norm_mean.
    Optionally logs per-step metrics to `wandb_run`.
    """
    model.train()
    metrics = _running_metrics_init()
    step = 0
    for batch in dataloader:
        optimizer.zero_grad()
        out = rl_training_step(
            model, batch, method, N, K, device, reward_transform, pkpo_k=pkpo_k,
        )
        loss = out['loss']
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=grad_clip
        ).item()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        _running_metrics_update(metrics, loss.item(), out, grad_norm)

        if wandb_run is not None and step % log_every == 0:
            r_step = out['rewards']
            wandb_run.log({
                'train/loss': loss.item(),
                'train/grad_norm_global': grad_norm,
                'train/lr': optimizer.param_groups[0]['lr'],
                'train_rl/reward_mean': r_step.mean().item(),
                'train_rl/reward_std':  r_step.std().item(),
                'train_rl/advantage_abs_mean': out['advantages'].abs().mean().item(),
                'train_rl/frac_iou_band_02_08': (
                    ((r_step > 0.2) & (r_step < 0.8)).float().mean().item()
                ),
                'epoch': epoch,
                'step': step,
            })
        step += 1
        if max_steps is not None and step >= max_steps:
            break

    return _running_metrics_finalize(metrics)


def train_supervised_epoch_localization(
    model,  # LocalizationPolicy (for ordinal/plain CE) OR LocalizationRegressor (for MSE)
    dataloader,
    optimizer,
    scheduler,
    method: str,   # "ordinal_ce" | "cross_entropy" | "mse"
    K: int,
    device: torch.device,
    epoch: int,
    grad_clip: float = 10.0,
    wandb_run=None,
    log_every: int = 50,
    l1_weight: float = 5.0,
    giou_weight: float = 2.0,
    max_steps: int | None = None,
) -> dict:
    """Run one supervised training epoch for the chosen baseline method.

    Methods:
    - "ordinal_ce": uses LocalizationPolicy (4 heads, K bins each) +
       localization_ordinal_ce_loss.
    - "cross_entropy": LocalizationPolicy + localization_cross_entropy_loss.
    - "mse":       LocalizationRegressor + localization_mse_loss.

    Args:
        l1_weight, giou_weight: term weights for the "l1_giou*" arms only
            (DETR defaults 5.0 / 2.0). Ignored by every other method, so
            existing callers keep their exact behaviour without passing them.
    """
    model.train()
    losses = []
    step = 0
    for batch in dataloader:
        images = batch['images'].to(device)
        optimizer.zero_grad()

        if method == "mse":
            # Legacy primary-GT MSE (kept for reproducibility; biased on multi-GT).
            pred = model(images)
            gt_primary = batch['gt_boxes'][:, 0, :].to(device)
            loss = localization_mse_loss(pred, gt_primary)
        elif method == "mse_iou_match":
            # MSE against the GT with highest IoU vs the prediction.
            pred = model(images)                                # (B, 4)
            gt_boxes = batch['gt_boxes'].to(device)             # (B, M, 4)
            gt_mask  = batch['gt_mask'].to(device)              # (B, M) bool
            loss = localization_mse_iou_match_loss(pred, gt_boxes, gt_mask)
        elif method == "l1_iou_match":
            # L1 against the GT with highest IoU vs the prediction.
            pred = model(images)                                # (B, 4)
            gt_boxes = batch['gt_boxes'].to(device)             # (B, M, 4)
            gt_mask  = batch['gt_mask'].to(device)              # (B, M) bool
            loss = localization_l1_iou_match_loss(pred, gt_boxes, gt_mask)
        elif method == "mse_centroid_match":
            pred = model(images)
            gt_boxes = batch['gt_boxes'].to(device)
            gt_mask  = batch['gt_mask'].to(device)
            loss = localization_mse_centroid_match_loss(pred, gt_boxes, gt_mask)
        elif method == "l1_centroid_match":
            pred = model(images)
            gt_boxes = batch['gt_boxes'].to(device)
            gt_mask  = batch['gt_mask'].to(device)
            loss = localization_l1_centroid_match_loss(pred, gt_boxes, gt_mask)
        elif method == "giou":
            # 1 - GIoU against the GT with highest GIoU vs the prediction.
            pred = model(images)                                # (B, 4)
            gt_boxes = batch['gt_boxes'].to(device)             # (B, M, 4)
            gt_mask  = batch['gt_mask'].to(device)              # (B, M) bool
            loss = localization_giou_loss(pred, gt_boxes, gt_mask)
        elif method in _GIOU_FAMILY_LOSSES:
            # GIoU-objective arms that differ only in the GT matcher and in
            # whether the DETR L1 term is added. l1_weight / giou_weight are
            # ignored by the pure-GIoU entries (they take no such kwarg).
            pred = model(images)                                # (B, 4)
            gt_boxes = batch['gt_boxes'].to(device)             # (B, M, 4)
            gt_mask  = batch['gt_mask'].to(device)              # (B, M) bool
            loss_fn = _GIOU_FAMILY_LOSSES[method]
            if method.startswith("l1_giou"):
                loss = loss_fn(
                    pred, gt_boxes, gt_mask,
                    l1_weight=l1_weight, giou_weight=giou_weight,
                )
            else:
                loss = loss_fn(pred, gt_boxes, gt_mask)
        elif method in ("tailrl_population", "tailrl_population_clamped"):
            logits = model(images)                             # dict of (B, K)
            gt_boxes = batch['gt_boxes'].to(device)             # (B, M, 4)
            gt_mask  = batch['gt_mask'].to(device)              # (B, M) bool
            loss = localization_tailrl_population_loss(
                logits, gt_boxes, gt_mask, K,
                n_thresholds=100,
                clamp_pred=(method == "tailrl_population_clamped"),
            )
        else:
            logits = model(images)                             # dict of (B, K)
            targets = {h: batch['target_bins'][h].to(device) for h in HEAD_NAMES}
            if method == "ordinal_ce":
                loss = localization_ordinal_ce_loss(logits, targets, K)
            elif method == "cross_entropy":
                loss = localization_cross_entropy_loss(logits, targets)
            else:
                raise ValueError(f"unknown supervised method: {method}")

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=grad_clip
        ).item()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
        if wandb_run is not None and step % log_every == 0:
            wandb_run.log({
                'train/loss': loss.item(),
                'train/grad_norm_global': grad_norm,
                'train/lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'step': step,
            })
        step += 1
        if max_steps is not None and step >= max_steps:
            break

    import statistics
    return {'loss_mean': statistics.fmean(losses) if losses else 0.0}


# ---------------------------------------------------------------------------
# Running-metrics helpers — online (constant-memory) accumulator.
#
# The original version buffered every step's flattened rewards + advantages
# on CPU (out['rewards'].flatten().cpu() append).  At N=16k, B=128, ~4254
# steps/epoch that is ~68 GB host RAM per epoch (272 GB at N=65k), which
# OOM-killed jobs even at 96 GB / 128 GB SLURM allocations.
#
# All finalized metrics are linear functionals of the per-step reward /
# advantage tensors (mean, std via sum + sum-of-squares, per-threshold
# fractions, mean |advantage|), so they can be computed online with O(1)
# state per stat.  After this fix, single-GPU N=16k / N=65k jobs run in
# ~5 GB of host RAM regardless of epoch size.
# ---------------------------------------------------------------------------

def _running_metrics_init() -> dict:
    return {
        # per-step scalars (still small): keep as lists for fmean.
        'losses': [],
        'grad_norms': [],
        # online reward stats
        'count_r': 0,
        'sum_r': 0.0,
        'sum_r2': 0.0,
        'count_r_gt_0_5': 0,
        'count_r_gt_0_75': 0,
        'count_r_eq_0': 0,
        'count_r_in_02_08': 0,
        # online advantage stats
        'count_a': 0,
        'sum_abs_a': 0.0,
    }


def _running_metrics_update(m: dict, loss: float, out: dict, grad_norm: float) -> None:
    m['losses'].append(loss)
    m['grad_norms'].append(grad_norm)

    r = out['rewards']
    m['count_r'] += int(r.numel())
    m['sum_r'] += float(r.sum().item())
    m['sum_r2'] += float((r * r).sum().item())
    m['count_r_gt_0_5'] += int((r > 0.5).sum().item())
    m['count_r_gt_0_75'] += int((r > 0.75).sum().item())
    m['count_r_eq_0'] += int((r == 0).sum().item())
    m['count_r_in_02_08'] += int(((r > 0.2) & (r < 0.8)).sum().item())

    a = out['advantages']
    m['count_a'] += int(a.numel())
    m['sum_abs_a'] += float(a.abs().sum().item())


def _running_metrics_finalize(m: dict) -> dict:
    if not m['losses']:
        return {'loss_mean': 0.0}

    n = m['count_r']
    if n > 0:
        mean_r = m['sum_r'] / n
        if n > 1:
            # Sample variance (Bessel's correction) — matches torch.std()'s
            # default unbiased=True. Clamp at 0 for fp64 catastrophic
            # cancellation when sum_r2 ≈ n·mean².
            var_r = max(
                (m['sum_r2'] - n * mean_r * mean_r) / (n - 1), 0.0,
            )
            std_r = var_r ** 0.5
        else:
            std_r = 0.0  # single sample → torch.std returns NaN; we return 0.
        frac_gt_05 = m['count_r_gt_0_5'] / n
        frac_gt_075 = m['count_r_gt_0_75'] / n
        frac_eq_0 = m['count_r_eq_0'] / n
        frac_in_02_08 = m['count_r_in_02_08'] / n
    else:
        mean_r = std_r = 0.0
        frac_gt_05 = frac_gt_075 = frac_eq_0 = frac_in_02_08 = 0.0

    abs_a_mean = m['sum_abs_a'] / m['count_a'] if m['count_a'] > 0 else 0.0

    import statistics
    return {
        'loss_mean':         statistics.fmean(m['losses']),
        'reward_mean':       mean_r,
        'reward_std':        std_r,
        'reward_at_0_5':     frac_gt_05,
        'reward_at_0_75':    frac_gt_075,
        'frac_zero_reward':  frac_eq_0,
        # Day-2 go/no-go gate (spec §9.1): fraction of rollouts with IoU strictly
        # in (0.2, 0.8). Must be >= 0.10 after epoch 1 for TailRL vs binary_maxrl to
        # be separable; otherwise the reward distribution is effectively binary.
        'frac_iou_band_02_08': frac_in_02_08,
        'advantage_abs_mean':abs_a_mean,
        'grad_norm_mean':    statistics.fmean(m['grad_norms']),
    }
