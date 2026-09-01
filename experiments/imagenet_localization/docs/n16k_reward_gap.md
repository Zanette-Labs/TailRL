# TailRL Train Reward Gap at N=16384 (vs N=1024)

## Observation

When training the localization policy with TailRL at increasing rollout counts
N (with all other hyperparameters held constant — eff_batch=128, lr=5e-4,
30 epochs, full ImageNet train set), the **train reward at end of epoch 1
drops noticeably at N=16384** compared to smaller N values.

| Run group                                      | N      | Code path                | step-85 reward (mean over 3 seeds) |
| ---------------------------------------------- | ------ | ------------------------ | ---------------------------------- |
| `tailrl_K50_N1024`                                | 1024   | original (no bf16/fused) | **0.642**                          |
| `tailrl_K50_N4096_survFloor`                      | 4096   | bf16 + fused-heads       | **0.638**                          |
| `tailrl_K50_N16384_survFloor`                     | 16384  | bf16 + fused-heads       | **0.594** ⚠️                        |

Step 85 ≈ end of epoch 1 (with `log_every=50` batches per wandb step and
~4254 batches/epoch at eff_batch=128 / full 544K-image train).

The gap is **~0.05 reward** (≈5% relative). This is well above seed-to-seed
noise (per-seed std ≈ 0.03 at N=1024, ≈ 0.005 at N=16384).

## First (incorrect) hypothesis: bf16 autocast

A recent code change wrapped the model forward in
`torch.autocast(device_type='cuda', dtype=torch.bfloat16)` to speed up the
backbone. This means:

- Forward activations are bf16 (≈3 decimal digits of precision).
- The autocast boundary downcast the **gradient on logits to bf16** during
  backward (because we used `.float()` on the logits before
  `F.log_softmax`, but the cast's backward goes the other direction).

This was suspicious — bf16 gradients on the policy could plausibly degrade
training. So bf16 was disabled and a controlled toy A/B was run.

### A/B test (CPU, B=8, K=50, N=4096, 5 seeds)

A small `Toy(Linear→ReLU→Linear→ReLU + 4 head Linears)` model was used to
mimic the localization policy. The exact training-step pipeline (factored
log-softmax, fused multinomial, TailRL advantage, factored sample log-prob,
`(adv * logp).mean()` loss) was reproduced. For each random seed the same
random sampling was forced; only the `with torch.autocast(...)` wrapper
was toggled.

```
seed   loss_fp32   loss_bf16   cos(g_fp32, g_bf16)   ||Δg||/||g||
   0   0.00154    0.00154            1.0000             0.80%
   1   0.00551    0.00550            1.0000             0.66%
   2  -0.00722   -0.00713            1.0000             0.60%
   3   0.00008    0.00013            1.0000             0.57%
   4   0.00533    0.00531            1.0000             0.60%
```

**Result:** bf16 gradient is essentially identical to fp32 gradient
(cosine = 1.0000, magnitude diff < 1%). Loss diff < 1% per step.

This level of per-step deviation cannot accumulate to a 5% reward gap over
~4250 optimization steps in a single epoch.

→ **bf16 is NOT the dominant cause.**

## Refined diagnosis

Cross-checking the data, the gap is **N-specific**, not code-specific:

- `N=4096_survFloor` (new bf16+fused code) = 0.638
- `N=1024` (original code, no bf16/fused) = 0.642
- `N=16384_survFloor` (same new code as N=4096) = 0.594

The new code **reproduces the original N=1024 numbers at N=4096** within
noise, then drops only at N=16384.

## Better candidate causes (untested)

1. **Linear-scaling-rule violation.** Larger N → lower-variance gradient
   estimator → effectively larger update per step. The fixed lr=5e-4 was
   tuned for noisier (smaller-N) gradients; it may overshoot at N=16384.

2. **Survival floor begins to bind at N=16384.** The clamp
   `min_survivor = max(1, N · 1e-4)` evaluates to 1 for N≤4096 (no-op),
   but to 2 for N=16384 — so the cumulative-hazard contribution from the
   single highest-reward sample is halved (gap_top/2 instead of gap_top/1).
   Mathematically the effect is small (one Riemann term out of N), but it
   does introduce a directional bias the smaller-N runs don't have.

3. **At larger N the policy collapses faster** to a peakier distribution —
   peakier policies have lower mean reward under stochastic rollouts even
   if they have higher greedy IoU. This is consistent with the "mean
   reward" being lower at N=16384 while greedy `val/iou_at_50` may not be.

## Status

- bf16 autocast is currently **off** (toggleable via `BF16=1` env var
  if we want to re-enable later).
- Fused per-head ops are **on** (mathematically identical to per-head
  loop, marginal speedup).
- Survival floor `eps = 1e-4` is **on** (`SURVIVAL_FLOOR_EPS` in
  `advantages.py`).
- `torch.compile` is **off** in the live runs (caused RAM growth → OOM at
  the previous 128 GB cap; mem now bumped to 480 GB).

## Next steps

To pin down the actual cause:

1. Compare per-step `train/grad_norm_global` between N=1024, 4096, 16384 —
   if grad_norm ∝ √(1/N), the linear-scaling-rule explanation is strongly
   supported.
2. Run a smoke job with N=16384 and lr halved (2.5e-4) to test the
   linear-scaling fix.
3. Check `val/iou_at_50` (greedy decoding) at end of epoch 1 across N — if
   greedy IoU is *not* lower at N=16384 even though mean rollout reward
   is, that supports the "peakier policy" explanation.
