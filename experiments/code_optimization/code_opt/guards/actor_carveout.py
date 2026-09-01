"""
Per-epoch actor carve-out.

verl's training loop keeps `max_actor_ckpt_to_keep` checkpoints in its
rolling buffer (set to 1 in the launch scripts to bound disk use).
That preserves resume integrity but loses the per-epoch trajectory, and the
post-hoc best@K eval needs every saved step to still be on disk.

This module installs a save-hook by monkey-patching
`RayPPOTrainer._save_checkpoint`. After each verl save, the driver
HARD-LINKS the saved files into `$PIE_ACTOR_CARVEOUT_DIR/step_<N>/`.
Hard-links are inode-level: zero extra disk space, instant, and they
survive verl's later `remove_previous_save_local_path`, because the
underlying inode stays alive while ANY link references it.

Call `install_actor_carveout_hook()` at training entry, AFTER importing
verl. `code_opt/train.py` does this when `PIE_ACTOR_CARVEOUT_DIR` is set,
and `code_opt/reward/gem5_reward.py` repeats the (idempotent) install so
the hook is also present in the Ray actor that executes the trainer.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


_INSTALLED = False


def _hardlink_tree(
    src: str,
    dst: str,
    skip_predicate=None,
) -> None:
    """Mirror `src` to `dst` using os.link() on files (zero-copy, same FS
    only) and os.makedirs on directories. Falls back to copytree if the
    destination filesystem doesn't support hardlinks.

    `skip_predicate(filename) -> bool` skips files when truthy. Used by
    the actor carve-out to drop ~14GB of optimizer shards per save (not
    needed for post-hoc eval or vLLM-load), keeping only the resume-
    irrelevant `model_*.pt`, `huggingface/`, and tokenizer/config files.
    """
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)

    try:
        for root, dirs, files in os.walk(src):
            rel = os.path.relpath(root, src)
            tgt_root = os.path.join(dst, rel) if rel != "." else dst
            os.makedirs(tgt_root, exist_ok=True)
            for fn in files:
                if skip_predicate is not None and skip_predicate(fn):
                    continue
                src_p = os.path.join(root, fn)
                tgt_p = os.path.join(tgt_root, fn)
                os.link(src_p, tgt_p)
    except OSError as e:
        # Cross-filesystem or permission issue — fall back to copy.
        print(
            f"[carveout] hardlink failed ({e}); falling back to copytree",
            file=sys.stderr,
        )
        shutil.rmtree(dst, ignore_errors=True)
        if skip_predicate is None:
            shutil.copytree(src, dst)
        else:
            # Recursive copy honoring the same skip predicate.
            for root, _dirs, files in os.walk(src):
                rel = os.path.relpath(root, src)
                tgt_root = os.path.join(dst, rel) if rel != "." else dst
                os.makedirs(tgt_root, exist_ok=True)
                for fn in files:
                    if skip_predicate(fn):
                        continue
                    shutil.copy2(os.path.join(root, fn),
                                 os.path.join(tgt_root, fn))


def _skip_optimizer_shards(filename: str) -> bool:
    """Skip verl's FSDP shard .pt files during carve-out, keeping only the
    huggingface/ subdir (+ tokenizer/config) — which is exactly what post-hoc
    vLLM eval loads. The FSDP optim_/model_/extra_state shards are redundant
    with the rolling full-state ckpt kept for resume, and dominate size
    (~8GB model + ~14GB optim per 1.7B save), so dropping them cuts each
    long-lived snapshot from ~16GB to ~3.4GB (huggingface safetensors)."""
    return filename.startswith(
        ("optim_world_size_", "model_world_size_", "extra_state_world_size_")
    ) and filename.endswith(".pt")


def install_actor_carveout_hook() -> bool:
    """Install the save-hook. Returns True if installed, False if env not
    set or already installed. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return False

    carveout_dir = os.environ.get("PIE_ACTOR_CARVEOUT_DIR")
    if not carveout_dir:
        return False

    try:
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    except ImportError as e:
        print(
            f"[carveout] WARN: could not import RayPPOTrainer ({e}); "
            f"hook NOT installed",
            file=sys.stderr,
        )
        return False

    os.makedirs(carveout_dir, exist_ok=True)
    # Hook the DRIVER-side trainer save. Wrapping
    # FSDPCheckpointManager.save_checkpoint instead does not work: it executes
    # inside a Ray WORKER process, where a monkeypatch applied here never lands,
    # so the carve-out directory silently stays empty.
    # RayPPOTrainer._save_checkpoint runs in the driver (where train.py installs
    # this hook), and by the time it returns
    # default_local_dir/global_step_<N>/actor is on the shared filesystem and
    # ready to mirror.
    original_save = RayPPOTrainer._save_checkpoint
    # Mirror only every PIE_CARVEOUT_EVERY steps (default 1 = every save). For
    # example save_freq=50 with carve_every=100 on a 2570-step run yields about
    # 26 snapshots per run.
    carve_every = int(os.environ.get("PIE_CARVEOUT_EVERY", "1"))

    def save_checkpoint_with_carveout(self):
        original_save(self)
        try:
            global_step = int(getattr(self, "global_steps", 0))
        except (TypeError, ValueError):
            global_step = 0
        # Step-interval filter: carve actor-only snapshots every N steps, AND
        # ALWAYS carve the final step (== total_training_steps) so the most-
        # trained checkpoint is preserved for the post-hoc best@K eval even when
        # total_training_steps is not a multiple of carve_every (e.g. 2570 % 250).
        try:
            total_steps = int(self.config.trainer.total_training_steps)
        except (AttributeError, TypeError, ValueError):
            total_steps = -1
        is_last = total_steps > 0 and global_step >= total_steps
        if carve_every > 1 and (global_step % carve_every) != 0 and not is_last:
            return
        local_path = os.path.join(
            self.config.trainer.default_local_dir,
            f"global_step_{global_step}", "actor")
        if not os.path.isdir(local_path):
            print(f"[carveout] WARN: actor dir missing, skip step={global_step}: "
                  f"{local_path}", file=sys.stderr)
            return
        target = os.path.join(carveout_dir, f"step_{global_step}")
        try:
            _hardlink_tree(local_path, target, skip_predicate=_skip_optimizer_shards)
            print(
                f"[carveout] step={global_step} -> {target} (hardlinked)",
                file=sys.stderr,
            )
            # Sidecar info.json records provenance (run / algo / init / step)
            # so any downstream consumer of step_<N>/ knows where it came from
            # without having to reconstruct the job's environment.
            import json as _json
            from datetime import datetime as _dt
            info = {
                "global_step": int(global_step),
                "run_id": os.environ.get("PIE_RUN_ID"),
                "algorithm": os.environ.get("PIE_ALGO"),
                "init_ckpt": os.environ.get("PIE_INIT_CKPT"),
                "seed": os.environ.get("SEED") or os.environ.get("PIE_SEED"),
                "wandb_run_group": os.environ.get("WANDB_RUN_GROUP"),
                "experiment_name": os.environ.get("EXPERIMENT_NAME"),
                "saved_at": _dt.utcnow().isoformat() + "Z",
                "source_local_path": local_path,
            }
            try:
                with open(os.path.join(target, "info.json"), "w") as f:
                    _json.dump(info, f, indent=2)
            except OSError as e:
                print(f"[carveout] WARN: failed to write info.json: {e}",
                      file=sys.stderr)
            # Sentinel — written LAST so post-hoc eval can filter half-written
            # carve-outs on resume after a crash.
            Path(target, ".carveout_complete").touch()
        except (OSError, RuntimeError) as e:
            print(
                f"[carveout] WARN: failed to mirror step={global_step}: {e}",
                file=sys.stderr,
            )

    RayPPOTrainer._save_checkpoint = save_checkpoint_with_carveout
    _INSTALLED = True
    print(
        f"[carveout] installed: driver-side mirror of global_step_*/actor -> "
        f"{carveout_dir}/step_<N>/ (every {carve_every} steps)",
        file=sys.stderr,
    )
    return True
