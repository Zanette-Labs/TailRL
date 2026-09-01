"""Quick A/B benchmark: rl_training_step throughput for TailRL at various N,
comparing baseline (no bf16, no compile, per-head loop) vs optimized
(bf16 autocast, fused heads, optionally torch.compile).

Run on a single GPU. Mocks a batch from random images so we don't rely
on the full ImageNet train pipeline.
"""
from __future__ import annotations

import time
import torch
import torch.nn.functional as F

from experiments.imagenet_localization.models.model import LocalizationPolicy
from experiments.imagenet_localization.core.advantages import tailrl_advantage
from experiments.imagenet_localization.core.iou import (
    batched_max_iou, clamp_boxes_to_image,
)

HEAD_NAMES = ("x_c", "y_c", "w", "h")


def fake_batch(B: int, device: torch.device) -> dict:
    return {
        'images':   torch.randn(B, 3, 224, 224, device=device),
        'gt_boxes': torch.tensor(
            [[[0.5, 0.5, 0.4, 0.4], [0.3, 0.3, 0.2, 0.2]]] * B,
            dtype=torch.float, device=device,
        ),
        'gt_mask':  torch.ones(B, 2, dtype=torch.bool, device=device),
    }


def step_baseline(model, batch, N, K):
    """Original per-head loop, no bf16, no compile."""
    images = batch['images']; gt_boxes = batch['gt_boxes']; gt_mask = batch['gt_mask']
    logits = model(images)
    log_probs = {h: F.log_softmax(logits[h], dim=-1) for h in HEAD_NAMES}
    with torch.no_grad():
        probs = {h: log_probs[h].exp() for h in HEAD_NAMES}
        samples = {h: torch.multinomial(probs[h], N, replacement=True)
                   for h in HEAD_NAMES}
        coords = torch.stack(
            [(samples[h].float() + 0.5) / K for h in HEAD_NAMES], dim=-1)
        boxes = clamp_boxes_to_image(coords)
        rewards = batched_max_iou(boxes, gt_boxes, gt_mask)
        adv = tailrl_advantage(rewards)
    sample_log_probs = sum(log_probs[h].gather(1, samples[h]) for h in HEAD_NAMES)
    return -(adv.detach() * sample_log_probs).mean(dim=1).mean()


def step_optimized(model, batch, N, K):
    """bf16 autocast + fused per-head ops."""
    images = batch['images']; gt_boxes = batch['gt_boxes']; gt_mask = batch['gt_mask']
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = model(images)
    log_probs = {h: F.log_softmax(logits[h].float(), dim=-1) for h in HEAD_NAMES}
    B = log_probs['x_c'].size(0)
    lp_stack = torch.stack([log_probs[h] for h in HEAD_NAMES], dim=0)  # (4, B, K)
    with torch.no_grad():
        probs_flat = lp_stack.exp().reshape(4 * B, K)
        samples_flat = torch.multinomial(probs_flat, N, replacement=True)
        s_stack = samples_flat.view(4, B, N)
        coords = ((s_stack.float() + 0.5) / K).permute(1, 2, 0)
        boxes = clamp_boxes_to_image(coords)
        rewards = batched_max_iou(boxes, gt_boxes, gt_mask)
        adv = tailrl_advantage(rewards)
    sample_log_probs = lp_stack.gather(2, s_stack).sum(dim=0)
    return -(adv.detach() * sample_log_probs).mean(dim=1).mean()


def time_loop(step_fn, model, batch, N, K, n_warmup=3, n_measure=20):
    opt = torch.optim.SGD(model.parameters(), lr=1e-5)
    for _ in range(n_warmup):
        opt.zero_grad()
        loss = step_fn(model, batch, N, K)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        opt.zero_grad()
        loss = step_fn(model, batch, N, K)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / n_measure


def main():
    device = torch.device("cuda")
    K = 50
    B = 32  # match the per-GPU batch in 4-GPU DDP runs
    n_measure = 20
    n_warmup = 3
    print(f"K={K}, B={B}, warmup={n_warmup}, measure={n_measure}")
    print(f"{'N':>8s}  {'baseline ms':>12s}  {'optimized ms':>13s}  {'speedup':>8s}")
    for N in [4096, 16384, 65536]:
        torch.manual_seed(0)
        model_a = LocalizationPolicy(K=K, pretrained=False, seed=0).to(device)
        torch.manual_seed(0)
        model_b = LocalizationPolicy(K=K, pretrained=False, seed=0).to(device)
        batch = fake_batch(B, device)
        t_baseline = time_loop(step_baseline, model_a, batch, N, K, n_warmup, n_measure)
        t_opt      = time_loop(step_optimized, model_b, batch, N, K, n_warmup, n_measure)
        print(f"{N:>8}  {t_baseline*1000:>11.1f}  {t_opt*1000:>12.1f}  "
              f"{t_baseline/t_opt:>7.2f}x")
        del model_a, model_b


if __name__ == "__main__":
    main()
