"""Main entry point for the ImageNet localization RL experiment.

Single run: trains one model with one method, one K, one N, one seed.
Handles W&B logging, checkpointing, and per-epoch evaluation.

Usage:
    python -m experiments.imagenet_localization.run \
        --method tailrl --K 50 --N 64 --seed 42 \
        --epochs 30 --batch_size 128 --lr 5e-4 \
        --data_dir "$IMAGENET_DIR" \
        --output_dir "$TAILRL_RESULTS_DIR/tailrl_K50_N64_seed42" \
        --num_workers 8 --train_subsample 100000 --wandb

--data_dir and --output_dir may be omitted once IMAGENET_DIR / TAILRL_RESULTS_DIR
are exported (see paths.py); the flags always win over the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from experiments.imagenet_localization import config as config_mod
from experiments.imagenet_localization import paths
from experiments.imagenet_localization.core.advantages import REWARD_TRANSFORMS
from experiments.imagenet_localization.datasets.data import ImageNetLocDataset, build_collate_fn
from experiments.imagenet_localization.evaluation.evaluate import evaluate_localization
from experiments.imagenet_localization.core.iou import batched_max_iou
from experiments.imagenet_localization.models.model import LocalizationPolicy, LocalizationRegressor
from experiments.imagenet_localization.training.train import (
    train_rl_epoch_localization,
    train_supervised_epoch_localization,
)

# ---------------------------------------------------------------------------
# Method groupings
# ---------------------------------------------------------------------------

RL_METHODS = {"tailrl", "binary_maxrl", "bmaxrl_adv_est", "grpo", "rloo", "reinforce", "pkpo"}
SUPERVISED_METHODS = {
    "ordinal_ce", "cross_entropy",
    "mse", "mse_iou_match", "mse_centroid_match",
    "l1_iou_match", "l1_centroid_match",
    "giou", "giou_iou_match", "giou_centroid_match",
    "l1_giou", "l1_giou_iou_match", "l1_giou_centroid_match",
    "tailrl_population", "tailrl_population_clamped",
}
ALL_METHODS = sorted(RL_METHODS | SUPERVISED_METHODS)

# Supervised arms backed by LocalizationRegressor (one deterministic (B, 4)
# box) rather than LocalizationPolicy. Also selects evaluate_mse_regressor.
REGRESSION_METHODS = {
    "mse", "mse_iou_match", "mse_centroid_match",
    "l1_iou_match", "l1_centroid_match",
    "giou", "giou_iou_match", "giou_centroid_match",
    "l1_giou", "l1_giou_iou_match", "l1_giou_centroid_match",
}


# ---------------------------------------------------------------------------
# DDP helpers — activated when torchrun/srun sets RANK/WORLD_SIZE/LOCAL_RANK
# ---------------------------------------------------------------------------

def _ddp_info() -> tuple[int, int, int, bool]:
    """Read DDP env vars. Returns (rank, world_size, local_rank, is_ddp)."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank, world_size > 1


def _ddp_init(local_rank: int) -> None:
    """Init the default process group (NCCL). Idempotent."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)


def _ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

# Global tracker for best checkpoint
_BEST_VAL_IOU: float = -1.0

# Epochs at which we dump a per-epoch checkpoint for offline gradient / cosine
# analysis. Mirrors experiment5_ordinal's {1, 10, 25, 50} schedule. The loop
# additionally writes ``final.pt`` at the last epoch and keeps ``best.pt`` /
# ``last.pt`` as before. Epochs that exceed the configured run length are
# skipped silently.
GRADIENT_ANALYSIS_EPOCHS: tuple[int, ...] = (1, 10, 25, 50)


def _atomic_write_json(path: str, payload) -> None:
    """Write ``payload`` as JSON to ``path`` atomically (fsync + rename).

    Guarantees that a crashed mid-write does not truncate the existing file —
    important because we re-write the whole metrics history each epoch.
    """
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(
        prefix=".metrics.", suffix=".json.tmp", dir=directory,
    )
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _jsonable(metrics: dict) -> dict:
    """Shallow copy of ``metrics`` with non-JSON-native scalars coerced.

    torch scalars / numpy scalars / dicts-of-floats slip in from wandb logs;
    cast them to plain floats. Non-scalar values (tensors, ndarrays) are
    dropped with a warning — they should not appear here but we guard anyway.
    """
    out: dict = {}
    for k, v in metrics.items():
        if v is None:
            out[k] = None
            continue
        if hasattr(v, "item") and not isinstance(v, (str, bytes)):
            try:
                out[k] = v.item()
                continue
            except Exception:
                pass
        if isinstance(v, (int, float, bool, str)):
            out[k] = v
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            # Skip unserialisable entry; log once so we notice.
            print(f"[metrics.json] dropping non-scalar key {k!r} (type={type(v).__name__})")
    return out


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def build_scheduler(optimizer, warmup_epochs: int, epochs: int, steps_per_epoch: int):
    """Linear warmup into cosine decay, stepped once per optimizer step.

    The warmup ramp is clamped to the length of the run: with
    ``warmup_epochs >= epochs`` (e.g. ``--epochs 1`` against the default
    ``warmup_epochs: 1``) the cosine phase would get ``T_max <= 0`` and
    CosineAnnealingLR divides by it, so the whole run is warmup instead.
    """
    total_steps = epochs * steps_per_epoch
    warmup_steps = min(warmup_epochs * steps_per_epoch, total_steps)

    if warmup_steps >= total_steps:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0,
            total_iters=max(total_steps, 1),
        )
        desc = (f"linear warmup over all {total_steps} steps "
                f"(run too short for a cosine phase)")
    elif warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-3, end_factor=1.0,
                    total_iters=warmup_steps,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=total_steps - warmup_steps, eta_min=0.0,
                ),
            ],
            milestones=[warmup_steps],
        )
        desc = (f"linear warmup {warmup_steps} steps → cosine to 0 over "
                f"remaining {total_steps - warmup_steps} steps")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=0.0,
        )
        desc = f"cosine to 0 over {total_steps} steps"
    return scheduler, desc


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an ImageNet localization model (single run).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config", type=str, default=str(config_mod.DEFAULT_CONFIG_PATH),
        help="YAML file supplying the hyperparameter defaults. Flags override it.",
    )

    # Required (--data_dir / --output_dir fall back to the environment)
    parser.add_argument(
        "--method", type=str, required=True, choices=ALL_METHODS,
        help="Training method / loss.",
    )
    parser.add_argument(
        "--data_dir", type=str, default=paths.imagenet_dir() or None,
        help="Root directory of the ImageNet dataset (with LOC_*_solution.csv "
             "files). Defaults to $IMAGENET_DIR.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=paths.results_dir(),
        help="Directory under which run-specific sub-directories will be "
             "created. Defaults to $TAILRL_RESULTS_DIR.",
    )

    # Core hyperparameters
    parser.add_argument("--K", type=int, default=50,
                        help="Number of bins per coordinate (policy heads).")
    parser.add_argument("--N", type=int, default=64,
                        help="Number of rollouts per image (RL methods).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global RNG seed.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Total training epochs.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size per GPU.")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Peak learning rate after linear warmup.")
    parser.add_argument("--warmup_epochs", type=int, default=1,
                        help="Number of epochs over which to linearly warm up "
                             "the LR from ~0 to --lr. Backbone is NOT frozen "
                             "during this phase. Set 0 to disable warmup.")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Coupled L2, applied to every parameter.")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Stop after this many optimizer steps. The LR "
                             "schedule is still built from --epochs over the "
                             "full split, so a capped run sees exactly the "
                             "learning rates an uncapped run would over its "
                             "first --max_steps steps. Use it to sample a "
                             "prefix of the full recipe.")
    parser.add_argument("--grad_clip", type=float, default=10.0,
                        help="Global gradient norm clip value.")
    parser.add_argument("--pkpo_k", type=int, default=None,
                        help="pass@k / max@k threshold for --method pkpo "
                             "(valid range 1..N). Defaults to the PKPO_K env "
                             "var. Included in the run name so different k at "
                             "the same N do not collide on disk or in W&B.")
    parser.add_argument("--l1_weight", type=float, default=5.0,
                        help="Weight on the L1 term of the 'l1_giou*' arms "
                             "(DETR default 5.0). Ignored by other methods.")
    parser.add_argument("--giou_weight", type=float, default=2.0,
                        help="Weight on the (1 - GIoU) term of the 'l1_giou*' "
                             "arms (DETR default 2.0). Ignored by other methods.")
    parser.add_argument("--reward_transform", type=str, default="none",
                        choices=sorted(REWARD_TRANSFORMS.keys()),
                        help="Reward shaping applied to per-rollout IoU before "
                             "the advantage estimator (RL methods only). 'none' "
                             "= raw IoU; 'percentile' = average-percentile-rank "
                             "in [0,1] (highest->1.0, lowest->0.0, ties->median).")

    # Data
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader worker processes.")
    parser.add_argument("--train_subsample", type=int, default=None,
                        help="If set, subsample this many training images.")

    # Eval
    parser.add_argument("--eval_every", type=int, default=1,
                        help="Evaluate on val every this many epochs.")
    parser.add_argument("--N_eval_samples", type=int, default=1024,
                        help="Number of samples per image during val evaluation.")

    # W&B
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default=paths.wandb_project(),
                        help="W&B project name (default: $WANDB_PROJECT).")

    # Misc
    # NB: there is deliberately no --include_mse here. It exists only on
    # sweep.py, where it gates whether the MSE arm joins the generated job
    # matrix. On a single run the arm is selected with --method mse.
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Disable pretrained backbone weights (useful for unit tests).")

    # Two-pass parse: resolve --config first, apply its values as defaults, then
    # parse again so an explicit flag still beats the file.
    pre, _ = parser.parse_known_args()
    file_defaults = config_mod.argparse_defaults(pre.config)
    known = {a.dest for a in parser._actions}
    unknown = sorted(set(file_defaults) - known)
    if unknown:
        raise SystemExit(
            f"{pre.config}: no such option(s): {', '.join(unknown)}. "
            "Every config key must match a run.py flag."
        )
    parser.set_defaults(**file_defaults)

    args = parser.parse_args()
    # --data_dir has no repo-relative default: it comes from the flag or from
    # $IMAGENET_DIR, and it is an error for both to be missing.
    if not args.data_dir:
        args.data_dir = paths.require_imagenet_dir()
    return args


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    output_dir: str,
    *,
    best_val_iou_tracker: list,  # single-element list used as mutable container
    total_epochs: Optional[int] = None,
    milestone_epochs: tuple[int, ...] = GRADIENT_ANALYSIS_EPOCHS,
) -> None:
    """Save last.pt every epoch and best.pt when val/iou_greedy improves.

    Additionally saves a fresh checkpoint named ``epoch_{epoch}.pt`` whenever
    ``epoch`` is in ``milestone_epochs`` and ``final.pt`` when the epoch equals
    ``total_epochs`` — mirrors experiment5_ordinal's post-hoc analysis schedule
    so gradient / cosine analyses can reuse the same code.

    Args:
        best_val_iou_tracker: a one-element list holding the best val IoU seen
            so far (float). Mutated in-place when a new best is found. Pass a
            reference to ``[float('-inf')]`` from the caller.
        total_epochs: if provided and ``epoch == total_epochs``, also write
            ``final.pt``. Left optional for callers that don't need it.
        milestone_epochs: epochs that trigger ``epoch_{epoch}.pt`` copies.
    """
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    last_path = os.path.join(output_dir, "last.pt")
    torch.save(payload, last_path)

    # Save best.pt when val/iou_greedy improves.
    current_iou = metrics.get("val/iou_greedy", None)
    if current_iou is not None and current_iou > best_val_iou_tracker[0]:
        best_val_iou_tracker[0] = current_iou
        best_path = os.path.join(output_dir, "best.pt")
        torch.save(payload, best_path)
        print(f"  [checkpoint] New best val/iou_greedy={current_iou:.4f} -> saved best.pt")

    # Milestone per-epoch copy for gradient / cosine analysis.
    if epoch in milestone_epochs:
        epoch_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
        torch.save(payload, epoch_path)
        print(f"  [checkpoint] Saved milestone epoch_{epoch}.pt")

    # Final checkpoint.
    if total_epochs is not None and epoch == total_epochs:
        final_path = os.path.join(ckpt_dir, "final.pt")
        torch.save(payload, final_path)
        print(f"  [checkpoint] Saved final.pt (epoch={epoch})")


# ---------------------------------------------------------------------------
# metrics.json persistence
# ---------------------------------------------------------------------------


def _append_metrics_record(
    history: list[dict],
    epoch: int,
    metrics: dict,
    output_dir: str,
) -> None:
    """Append this epoch's metrics record to ``history`` and persist to disk.

    The on-disk file ``metrics.json`` matches experiment5_ordinal's layout: a
    list of dicts, one per epoch, each with an ``epoch`` key plus every logged
    metric. Written atomically so partially-completed jobs leave the file in
    a coherent state.
    """
    record = _jsonable(metrics)
    record["epoch"] = epoch
    history.append(record)
    _atomic_write_json(os.path.join(output_dir, "metrics.json"), history)


# ---------------------------------------------------------------------------
# MSE regressor evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_mse_regressor(
    model: LocalizationRegressor,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate the MSE regression baseline.

    MSE regressor outputs one deterministic point prediction per image (no
    sampling), so sample-based metrics (iou_expected, iou_best_of_N, entropy)
    don't apply. We log everything else that does apply, matching
    `evaluate_localization`'s schema as closely as possible:

      - val/iou_greedy:  mean max-IoU(pred, GT_j) over batch
      - val/iou_at_50, val/iou_at_75, val/iou_at_90: CorLoc fractions
      - val/mse_loss:    mean MSE vs primary GT (for historical comparison)
      - val_{tier}/iou_greedy, val_{tier}/count: per-size-tier splits
    """
    from experiments.imagenet_localization.evaluation.evaluate import (
        SIZE_TIERS,
        size_tier_of_primary_box,
    )

    model.eval()
    ious_all: list[torch.Tensor] = []
    mse_losses: list[float] = []
    tier_iou: dict[str, list[torch.Tensor]] = {t: [] for t in SIZE_TIERS}
    tier_count: dict[str, int] = {t: 0 for t in SIZE_TIERS}

    for batch in val_loader:
        images = batch["images"].to(device)            # (B, 3, H, W)
        gt_boxes = batch["gt_boxes"].to(device)        # (B, M, 4)
        gt_mask = batch["gt_mask"].to(device)          # (B, M) bool

        pred = model(images)                           # (B, 4) in [0, 1]

        # Max-IoU over all valid GTs (B,)
        iou_batch = batched_max_iou(
            pred.unsqueeze(1), gt_boxes, gt_mask
        ).squeeze(1).cpu()
        ious_all.append(iou_batch)

        # MSE against primary GT — kept for backwards comparability
        gt_primary = gt_boxes[:, 0, :]
        mse = F.mse_loss(pred, gt_primary, reduction="mean")
        mse_losses.append(mse.item())

        # Per-tier bookkeeping (tier determined by primary GT area)
        primary_cpu = gt_boxes[:, 0, :].cpu()
        tiers = size_tier_of_primary_box(primary_cpu)
        for b_idx, tier in enumerate(tiers):
            tier_iou[tier].append(iou_batch[b_idx].unsqueeze(0))
            tier_count[tier] += 1

    ious_flat = torch.cat(ious_all)
    metrics: dict = {
        "val/iou_greedy": ious_flat.mean().item(),
        "val/iou_at_50":  (ious_flat >= 0.5 ).float().mean().item(),
        "val/iou_at_75":  (ious_flat >= 0.75).float().mean().item(),
        "val/iou_at_90":  (ious_flat >= 0.9 ).float().mean().item(),
        "val/mse_loss":   float(sum(mse_losses) / len(mse_losses)) if mse_losses else float("nan"),
    }
    for tier in SIZE_TIERS:
        if tier_iou[tier]:
            metrics[f"val_{tier}/iou_greedy"] = torch.cat(tier_iou[tier]).mean().item()
        else:
            metrics[f"val_{tier}/iou_greedy"] = float("nan")
        metrics[f"val_{tier}/count"] = tier_count[tier]
    return metrics


# ---------------------------------------------------------------------------
# Configuration banner
# ---------------------------------------------------------------------------


def _print_banner(args: argparse.Namespace, run_name: str, output_dir: str) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  ImageNet Localization RL Experiment")
    print(f"  Run:        {run_name}")
    print(sep)
    print(f"  Method:          {args.method}")
    print(f"  K (bins):        {args.K}")
    print(f"  N (rollouts):    {args.N}")
    print(f"  Seed:            {args.seed}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  LR (peak):       {args.lr}")
    print(f"  Warmup epochs:   {args.warmup_epochs}  (linear LR ramp, no backbone freeze)")
    print(f"  Grad clip:       {args.grad_clip}")
    if args.method == "pkpo":
        print(f"  PKPO k (max@k):  {args.pkpo_k if args.pkpo_k else '(PKPO_K env)'}")
    if args.method.startswith("l1_giou"):
        print(f"  L1 weight:       {args.l1_weight}")
        print(f"  GIoU weight:     {args.giou_weight}")
    print(f"  Eval every:      {args.eval_every} epoch(s)")
    print(f"  N_eval_samples:  {args.N_eval_samples}")
    print(f"  Train subsample: {args.train_subsample}")
    print(f"  Num workers:     {args.num_workers}")
    print(f"  Pretrained:      {not args.no_pretrained}")
    print(f"  Data dir:        {args.data_dir}")
    print(f"  Output dir:      {output_dir}")
    print(f"  W&B:             {args.wandb}")
    if args.wandb:
        print(f"  W&B project:     {args.wandb_project}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Train one model for one configuration.

    Args:
        args: pre-built argparse.Namespace (useful for programmatic/test
              invocation). If None, will call parse_args() to read sys.argv.
    """
    if args is None:
        args = parse_args()

    # --- DDP detection & init ---
    rank, world_size, local_rank, is_ddp = _ddp_info()
    if is_ddp:
        _ddp_init(local_rank)
    is_main = (rank == 0)

    # --- Seeding (use same base seed across ranks for reproducible model init) ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # --- Run identity + output dir ---
    # Tag the run with the reward transform when it is not the raw-IoU default,
    # so percentile-ablation runs don't collide with the raw baselines.
    xform_tag = "" if args.reward_transform == "none" else f"_{args.reward_transform}"
    # pkpo's k is a distinct axis from --K (bin count): without it in the name,
    # k=16/64/256 at one N would all write to the same dir and W&B group.
    k_tag = f"_k{args.pkpo_k}" if (args.method == "pkpo" and args.pkpo_k) else ""
    run_name = f"{args.method}{xform_tag}{k_tag}_K{args.K}_N{args.N}_seed{args.seed}"
    # Allow the sweep launcher to override the W&B group via env var
    # (e.g. to tag a re-run with `_fixedN`). Falls back to the canonical
    # group name when the env var is unset.
    run_group = os.environ.get(
        "WANDB_RUN_GROUP", f"{args.method}{xform_tag}{k_tag}_K{args.K}_N{args.N}"
    )
    # Use args.output_dir directly. Sweep scripts pass the per-run path
    # (e.g. results/<run_name>) — appending run_name again would create
    # the legacy results/<run>/<run>/ nested structure.
    output_dir = args.output_dir
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
        _print_banner(args, run_name, output_dir)
        if is_ddp:
            print(f"DDP: rank={rank}/{world_size}, local_rank={local_rank}\n")

    # --- W&B init (rank 0 only) ---
    wandb_run = None
    if args.wandb and is_main:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            group=run_group,
            config=vars(args),
        )

    # --- Datasets / loaders ---
    train_dataset = ImageNetLocDataset(
        root_dir=args.data_dir, split="train", K=args.K,
        subsample=args.train_subsample, seed=args.seed, train_aug=True,
    )
    val_dataset = ImageNetLocDataset(
        root_dir=args.data_dir, split="val", K=args.K,
        subsample=None, seed=args.seed, train_aug=False,
    )
    collate_fn = build_collate_fn()
    if is_ddp:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank,
            shuffle=True, seed=args.seed, drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=train_sampler, num_workers=args.num_workers,
            pin_memory=True, collate_fn=collate_fn, drop_last=True,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers,
            pin_memory=True, collate_fn=collate_fn, drop_last=True,
        )
    # Val loader: only used on rank 0, non-distributed
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_fn, drop_last=False,
    )

    # --- Model ---
    # Arms that use LocalizationRegressor (single (B, 4) box) rather than
    # LocalizationPolicy. Also selects evaluate_mse_regressor below.
    is_mse = args.method in REGRESSION_METHODS
    if is_mse:
        model = LocalizationRegressor(pretrained=not args.no_pretrained, seed=args.seed)
    else:
        model = LocalizationPolicy(K=args.K, pretrained=not args.no_pretrained, seed=args.seed)

    if is_ddp:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # channels_last: makes the ResNet backbone use cuDNN's NHWC Tensor-Core
    # path. ~1.5× faster forward+backward on a6000/a5000. No-op for ops
    # without a channels_last kernel — they auto-fallback to NCHW.
    if torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)
    if is_main:
        print(f"Device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Keep a reference to the unwrapped model for eval/checkpointing
    raw_model = model
    if is_ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])

    # torch.compile fuses the backbone + head ops; gives 1.3-1.8× speedup
    # on a6000/a5000 typically. Suppress recompilations and gracefully fall
    # back if the compiler trips on a graph break.
    if torch.cuda.is_available() and not bool(int(os.environ.get("DISABLE_COMPILE", "0"))):
        try:
            # mode='default' (the standard inductor pipeline) — avoids CUDA
            # graphs which were causing slow CPU-RAM growth → OOM in our
            # earlier runs. Still gives ~1.3× via op fusion / kernel selection.
            model = torch.compile(model, mode="default", dynamic=False)
            if is_main:
                print("torch.compile: enabled (mode=default)")
        except Exception as e:  # noqa: BLE001
            if is_main:
                print(f"torch.compile: skipped ({e})")

    # --- Method classification ---
    is_rl = args.method in RL_METHODS

    # --- Optimizer + schedule ---
    # Single optimizer over all params from step 0 (no backbone freeze).
    # Schedule: linear LR warmup from ~0 → args.lr over the first
    # `warmup_epochs` epochs, then cosine decay to 0 over the remainder.
    # Matches the standard detection-model recipe (DETR, RetinaNet, etc.).
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    steps_per_epoch = len(train_loader)
    scheduler, sched_desc = build_scheduler(
        optimizer, args.warmup_epochs, args.epochs, steps_per_epoch,
    )
    opt_desc = f"Adam(lr={args.lr}, wd={args.weight_decay})"
    print(f"Optimizer: {opt_desc}; {sched_desc}.")

    # Mutable tracker for best checkpoint (single-element list acts as pointer)
    best_val_iou_tracker = [float("-inf")]

    # Store last epoch's metrics for final W&B summary
    train_metrics: dict = {}

    # In-memory history mirrored to metrics.json after each epoch. Resumed
    # from disk if a previous run left one behind — useful when a job is
    # requeued on SLURM pre-emption.
    metrics_history: list[dict] = []
    metrics_path = os.path.join(output_dir, "metrics.json")
    if is_main and os.path.exists(metrics_path):
        try:
            with open(metrics_path) as fh:
                loaded = json.load(fh)
            if isinstance(loaded, list):
                metrics_history = loaded
                print(f"Loaded existing metrics.json with {len(metrics_history)} epochs.")
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[metrics.json] existing file unreadable ({exc}); starting fresh.")

    # --- Training loop ---
    for epoch in range(1, args.epochs + 1):

        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # --- Train ---
        if is_rl:
            train_metrics = train_rl_epoch_localization(
                model, train_loader, optimizer, scheduler,
                method=args.method, N=args.N, K=args.K,
                device=device, epoch=epoch, grad_clip=args.grad_clip,
                max_steps=args.max_steps,
                reward_transform=args.reward_transform,
                pkpo_k=args.pkpo_k,
                wandb_run=wandb_run,
            )
        else:
            train_metrics = train_supervised_epoch_localization(
                model, train_loader, optimizer, scheduler,
                method=args.method, K=args.K,
                device=device, epoch=epoch, grad_clip=args.grad_clip,
                max_steps=args.max_steps,
                wandb_run=wandb_run,
                l1_weight=args.l1_weight, giou_weight=args.giou_weight,
            )

        # --- Eval (rank 0 only; other ranks wait at barrier below) ---
        if is_main:
            if is_mse:
                val_metrics = evaluate_mse_regressor(raw_model, val_loader, device=device)
                train_metrics.update(val_metrics)
            elif epoch % args.eval_every == 0:
                val_metrics = evaluate_localization(
                    raw_model, val_loader, K=args.K, device=device,
                    N_eval_samples=args.N_eval_samples,
                )
                train_metrics.update(val_metrics)

            # --- W&B logging ---
            if wandb_run is not None:
                wandb_run.log({**train_metrics, "epoch": epoch})

            # --- Persist per-epoch metrics.json (mirror of the W&B dict) ---
            _append_metrics_record(
                metrics_history, epoch, train_metrics, output_dir,
            )

            # --- Checkpoint ---
            save_checkpoint(
                raw_model, optimizer, epoch, train_metrics, output_dir,
                best_val_iou_tracker=best_val_iou_tracker,
                total_epochs=args.epochs,
            )

        if is_ddp:
            dist.barrier()

        # --- Epoch summary (rank 0 only) ---
        if is_main:
            loss_str = f"loss={train_metrics.get('loss_mean', float('nan')):.4f}"
            iou_str = (
                f"val/iou_greedy={train_metrics['val/iou_greedy']:.4f}"
                if "val/iou_greedy" in train_metrics else ""
            )
            print(f"[epoch {epoch:3d}/{args.epochs}] {loss_str}  {iou_str}")

    # --- Final W&B summary (rank 0 only) ---
    if wandb_run is not None:
        final = {
            f"final/{k.split('/', 1)[-1]}": v
            for k, v in train_metrics.items()
            if k.startswith("val/")
        }
        wandb_run.log(final)
        wandb_run.finish()

    if is_main:
        print(f"\nTraining complete. Checkpoints saved to: {output_dir}")

    # --- DDP cleanup ---
    if is_ddp:
        _ddp_cleanup()


if __name__ == "__main__":
    main()
