# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import random
import re
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from filelock import FileLock
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedTokenizer, ProcessorMixin


CHECKPOINT_TRACKER = "checkpoint_tracker.json"


class BaseCheckpointManager(ABC):
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin],
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.processing_class = processing_class

        assert isinstance(self.model, FSDP)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    @abstractmethod
    def load_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def local_mkdir(path: str) -> str:
        if not os.path.isabs(path):
            working_dir = os.getcwd()
            path = os.path.join(working_dir, path)

        # Using hash value of path as lock file name to avoid long file name
        lock_filename = f"ckpt_{hash(path) & 0xFFFFFFFF:08x}.lock"
        lock_path = os.path.join(tempfile.gettempdir(), lock_filename)

        try:
            with FileLock(lock_path, timeout=60):
                os.makedirs(path, exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to acquire lock for {path}: {e}")
            os.makedirs(path, exist_ok=True)  # even if the lock is not acquired, try to create the directory

        return path

    @staticmethod
    def get_rng_state() -> dict[str, Any]:
        rng_state = {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }
        return rng_state

    @staticmethod
    def load_rng_state(rng_state: dict[str, Any]):
        torch.set_rng_state(rng_state["cpu"])
        torch.cuda.set_rng_state(rng_state["cuda"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])


def get_checkpoint_tracker_filename(root_path: str) -> str:
    """
    Tracker file rescords the latest chckpoint during training to restart from.
    """
    return os.path.join(root_path, CHECKPOINT_TRACKER)


CHECKPOINT_COMPLETE_MARKER = ".complete"


def _actor_shards_complete(ckpt_path: str) -> bool:
    """A step dir is complete iff its actor/ holds the FULL per-rank model-shard set
    (model_world_size_{W}_rank_{0..W-1}.pt). W is read from the shard filenames themselves, so
    this is world-size-agnostic. Catches an interrupted save (missing shards) or a deleted/partial
    shard (errata C1 "expected shard count"). Backward-compatible: pre-sentinel checkpoints that
    have a full shard set still pass."""
    actor_dir = os.path.join(ckpt_path, "actor")
    if not os.path.isdir(actor_dir):
        return False
    shard_re = re.compile(r"^model_world_size_(\d+)_rank_(\d+)\.pt$")
    world_size, ranks = None, set()
    for fname in os.listdir(actor_dir):
        m = shard_re.match(fname)
        if m:
            w, r = int(m.group(1)), int(m.group(2))
            world_size = w if world_size is None else world_size
            if w == world_size:
                ranks.add(r)
    if world_size is None:
        return False
    return ranks == set(range(world_size))


def _ckpt_is_complete(ckpt_path: str) -> bool:
    """errata C1: COMPLETE == full actor shard set present. A `.complete` sentinel (written LAST
    by the trainer on a fully-committed save) is an additional positive signal but is not required,
    so checkpoints written before the sentinel existed still resume."""
    if not os.path.isdir(ckpt_path):
        return False
    return _actor_shards_complete(ckpt_path)


def should_save_at(
    global_step: int, save_freq: int, save_paired: bool = False, extra_steps: Optional[Iterable[int]] = None
) -> bool:
    """Save-cadence policy. Plain: save at global_step % save_freq == 0. With save_paired ALSO at
    % save_freq == save_freq - 1, i.e. the step immediately BEFORE each plain save (9,10 / 19,20
    with save_freq=10): the two retained checkpoints (save_limit=2, protect_best off) are adjacent,
    so if the newest fails to LOAD on resume the fallback loses 1 step, not save_freq steps.

    `extra_steps` adds explicitly-listed global steps on top of that periodic cadence, giving a
    NON-UNIFORM schedule from a single knob: a dense early ladder (100, 200, 300, ...) for a
    separation-window analysis plus a sparse periodic one for the long tail. It fires regardless of
    save_freq -- including save_freq <= 0 -- so an explicit list is never silently dropped."""
    if extra_steps and global_step in set(extra_steps):
        return True
    if save_freq <= 0:
        return False
    if global_step % save_freq == 0:
        return True
    return save_paired and save_freq > 1 and global_step % save_freq == save_freq - 1


def find_complete_ckpts(
    path: str, directory_format: str = "global_step_{}"
) -> tuple[list[str], Optional[dict[str, Any]]]:
    """ALL complete checkpoints under `path`, NEWEST FIRST, plus the tracker info.

    The ordered list is the trainer's load-fallback chain: candidates are tried in order and a
    checkpoint whose LOAD raises (a shard set can be complete on disk yet fail to deserialize —
    torn write, bad block) is skipped in favour of the next one. With save_paired the top two
    candidates are adjacent steps, so a fallback costs 1 step. Incomplete dirs (C1) are excluded."""
    if not os.path.exists(path):
        return [], None

    tracker_file = get_checkpoint_tracker_filename(path)
    tracker_info: Optional[dict[str, Any]] = None
    if os.path.exists(tracker_file):
        with open(tracker_file, "rb") as f:
            tracker_info = json.load(f)

    pattern = re.compile(re.escape(directory_format).replace(r"\{\}", r"(\d+)"))
    steps = sorted(
        (int(m.group(1)) for folder in os.listdir(path) if (m := pattern.match(folder))),
        reverse=True,
    )

    complete = []
    for step in steps:
        ckpt_path = os.path.join(path, directory_format.format(step))
        if _ckpt_is_complete(ckpt_path):
            complete.append(ckpt_path)
        else:
            print(f"[resume] skipping incomplete checkpoint: {ckpt_path}")

    return complete, tracker_info


def find_latest_ckpt(
    path: str, directory_format: str = "global_step_{}"
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """Find the newest COMPLETE checkpoint to resume from (errata C1 "malformed-latest revert").

    Instead of blindly trusting checkpoint_tracker.json's last_global_step (which crashes the run if
    the newest save was interrupted or a model shard is truncated/missing), enumerate every
    global_step_{N} dir newest-first and return the first one that passes the completeness check;
    a malformed newest checkpoint is skipped in favour of the older complete one. The tracker is
    still read for best_val bookkeeping and returned as-is (the resumed global_step is derived from
    the chosen dir name, not the tracker, in RayPPOTrainer._load_checkpoint)."""
    complete, tracker_info = find_complete_ckpts(path, directory_format)
    if not complete:
        print(f"No complete checkpoint found under {path}; starting from scratch.")
        return None, None

    ckpt_path = complete[0]
    step = int(ckpt_path.rstrip(os.path.sep).split("global_step_")[-1])
    tracker_step = tracker_info.get("last_global_step") if tracker_info else None
    if tracker_step is not None and step != tracker_step:
        print(
            f"[resume] tracker names step {tracker_step} but the newest COMPLETE checkpoint "
            f"is step {step}; reverting to it (malformed-latest revert)."
        )
    print(
        f"Found latest checkpoint: {ckpt_path}, will resume from it. "
        f"Turn off `find_last_checkpoint` to disable it."
    )
    return ckpt_path, tracker_info


def remove_obsolete_ckpt(
    path: str,
    global_step: int,
    best_global_step: int,
    save_limit: int = -1,
    directory_format: str = "global_step_{}",
    protect_best: bool = True,
):
    """
    Remove the obsolete checkpoints that exceed the save limit.

    protect_best=False drops the keep-the-best-val exception so exactly the NEWEST `save_limit`
    dirs survive — required by the paired-save scheme (save_paired), where the retained set must
    be the two ADJACENT latest saves for the 1-step load-fallback to hold (a pinned old best would
    evict the newer pair partner). Best weights for eval live on the actor_only track anyway.
    """
    if save_limit <= 0 or not os.path.exists(path):
        return

    num_ckpt_to_keep = save_limit - 1  # exclude the current ckpt
    pattern = re.escape(directory_format).replace(r"\{\}", r"(\d+)")
    ckpt_global_steps = []
    for folder in os.listdir(path):
        if match := re.match(pattern, folder):
            step = int(match.group(1))
            if step < global_step:
                ckpt_global_steps.append(step)

    ckpt_global_steps.sort(reverse=True)
    if protect_best and best_global_step in ckpt_global_steps:  # do not remove the best ckpt
        ckpt_global_steps.remove(best_global_step)
        num_ckpt_to_keep = max(num_ckpt_to_keep - 1, 0)

    for step in ckpt_global_steps[num_ckpt_to_keep:]:
        folder_path = os.path.join(path, directory_format.format(step))
        try:
            shutil.rmtree(folder_path, ignore_errors=True)
            print(f"Removed obsolete checkpoint: {folder_path}")
        except Exception as e:
            print(f"Failed to remove {folder_path}: {e}")
