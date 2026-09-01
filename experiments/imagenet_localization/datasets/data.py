"""Dataset utilities for the ImageNet localization RL experiment.

Handles:
- Parsing LOC_{split}_solution.csv files
- Normalizing pixel-coordinate bounding boxes to unit-square xywh
- Bin-discretizing continuous coordinates for supervised CE baselines
- Horizontal flip augmentation (image + boxes in sync, avoiding silent bug)
- ImageNetLocDataset: the main Dataset class
- build_collate_fn: batching utility
"""

from __future__ import annotations

import os
import random
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

MAX_M = 8  # Max number of GT boxes per image; drop extras (<0.1% of images).

IMAGE_SIZE = 224
# ImageNet normalization (standard ResNet stats).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Reusable deterministic transforms (no augmentation)
# ---------------------------------------------------------------------------

_BASE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Pure parsing / conversion utilities (no I/O)
# ---------------------------------------------------------------------------


def parse_prediction_string(pred_str: str) -> tuple[str, list[tuple[int, int, int, int]]]:
    """Parse a PredictionString into (wnid, [(x1, y1, x2, y2), ...]).

    Multi-box rows MUST have all the same wnid; raise AssertionError otherwise.
    Malformed rows (token count not divisible by 5) raise AssertionError.
    Extra whitespace is tolerated.

    Format: '<wnid> <x1> <y1> <x2> <y2>' repeated for each instance,
    all separated by spaces (wnid is a WordNet synset ID string).
    """
    tokens = pred_str.strip().split()
    assert len(tokens) % 5 == 0, (
        f"PredictionString token count {len(tokens)} not divisible by 5: {pred_str!r}"
    )
    assert len(tokens) >= 5, (
        f"PredictionString is empty or has fewer than 5 tokens: {pred_str!r}"
    )

    first_wnid = tokens[0]
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(0, len(tokens), 5):
        wnid = tokens[i]
        assert wnid == first_wnid, (
            f"Mixed wnid in PredictionString: got {wnid!r} but first was {first_wnid!r}"
        )
        x1, y1, x2, y2 = int(tokens[i + 1]), int(tokens[i + 2]), int(tokens[i + 3]), int(tokens[i + 4])
        boxes.append((x1, y1, x2, y2))

    return first_wnid, boxes


def normalize_box(
    x1: int, y1: int, x2: int, y2: int,
    orig_w: int, orig_h: int,
) -> tuple[float, float, float, float]:
    """Convert corner pixel coords (in the original image frame) to
    normalized xywh (x_c, y_c, w, h) in [0, 1].

    The CSV provides absolute pixel coordinates in the ORIGINAL image resolution.
    This function converts them to the normalized unit-square xywh format used
    throughout the rest of the pipeline.
    """
    # Center coordinates in pixels
    x_c_px = (x1 + x2) / 2.0
    y_c_px = (y1 + y2) / 2.0
    # Width and height in pixels
    w_px = float(x2 - x1)
    h_px = float(y2 - y1)
    # Normalize by image dimensions
    x_c = x_c_px / orig_w
    y_c = y_c_px / orig_h
    w = w_px / orig_w
    h = h_px / orig_h
    return x_c, y_c, w, h


def bin_discretize_coord(coord: float, K: int) -> int:
    """Map a coordinate in [0, 1] (possibly out of range) to a bin index in {0, ..., K-1}.

    coord < 0  -> 0
    coord >= 1 -> K-1      (spec: exactly at 1.0 goes to K-1, NOT K)
    else       -> min(int(coord * K), K-1)

    The K-1 upper clamp is the critical off-by-one guard: without it,
    coord=1.0 would yield bin K which is out of range.
    """
    if coord < 0.0:
        return 0
    if coord >= 1.0:
        return K - 1
    return min(int(coord * K), K - 1)


def apply_horizontal_flip(
    image: torch.Tensor,  # (3, H, W)
    boxes: torch.Tensor,  # (M, 4) xywh in [0, 1]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flip image horizontally AND mirror x_c of every box.

    boxes: only x_c (column 0) is updated (x_c -> 1 - x_c); y_c, w, h preserved.
    Works on the full (M, 4) tensor (multi-box safe). Does NOT modify input tensors.

    This is the critical fix for the #1 silent bug: applying flip to image only
    but not to boxes creates inconsistent supervision targets.
    """
    # Flip image: reverse the width dimension (dim=-1 for (3, H, W))
    flipped_image = image.flip(dims=[-1])

    # Mirror x_c for all boxes: clone first to avoid in-place mutation
    flipped_boxes = boxes.clone()
    flipped_boxes[:, 0] = 1.0 - flipped_boxes[:, 0]

    return flipped_image, flipped_boxes


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ImageNetLocDataset(Dataset):
    def __init__(
        self,
        root_dir: str,           # ImageNet root, e.g. os.environ["IMAGENET_DIR"]
        split: str,              # "train" or "val"
        K: int,                  # number of bins per coordinate
        subsample: Optional[int] = None,   # keep first N rows of the CSV after shuffle
        seed: int = 42,
        train_aug: bool = True,  # apply random horizontal flip on train split
    ):
        """Load rows from LOC_{split}_solution.csv, shuffle deterministically,
        optionally subsample, and build per-row image paths.

        Train image path: {root_dir}/ILSVRC/Data/CLS-LOC/train/{wnid}/{image_id}.JPEG
        Val   image path: {root_dir}/ILSVRC/Data/CLS-LOC/val/{image_id}.JPEG

        The wnid for train images is extracted from the image_id itself
        (format: <wnid>_<index>).
        """
        assert split in ("train", "val"), f"split must be 'train' or 'val', got {split!r}"
        self.root_dir = root_dir
        self.split = split
        self.K = K
        self.train_aug = train_aug and (split == "train")

        # Load CSV
        csv_path = os.path.join(root_dir, f"LOC_{split}_solution.csv")
        df = pd.read_csv(csv_path)

        # Deterministic shuffle
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        # Optional subsample
        if subsample is not None:
            df = df.iloc[:subsample].reset_index(drop=True)

        self.image_ids: list[str] = df["ImageId"].tolist()
        self.pred_strings: list[str] = df["PredictionString"].tolist()

    def __len__(self) -> int:
        return len(self.image_ids)

    def _get_image_path(self, image_id: str) -> str:
        """Resolve the JPEG path for a given image_id."""
        if self.split == "val":
            return os.path.join(
                self.root_dir, "ILSVRC", "Data", "CLS-LOC", "val",
                f"{image_id}.JPEG"
            )
        else:
            # Train: image_id is <wnid>_<index>
            wnid = image_id.rsplit("_", 1)[0]
            return os.path.join(
                self.root_dir, "ILSVRC", "Data", "CLS-LOC", "train",
                wnid, f"{image_id}.JPEG"
            )

    def __getitem__(self, idx: int) -> dict:
        """Return a dict with:
        - 'image':       (3, 224, 224) float tensor, ImageNet-normalized.
        - 'gt_boxes':    (MAX_M, 4) float tensor — normalized xywh, padded with zeros.
        - 'gt_mask':     (MAX_M,) bool tensor — True where the GT row is real.
        - 'target_bins': dict with keys 'x_c', 'y_c', 'w', 'h' -> int,
                         the PRIMARY box's bin indices (for supervised CE baselines).
        - 'image_id':    str.
        """
        image_id = self.image_ids[idx]
        pred_str = self.pred_strings[idx]

        # --- Load image ---
        img_path = self._get_image_path(image_id)
        pil_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_img.size  # PIL gives (width, height)

        # --- Parse boxes (use original resolution for normalization) ---
        _wnid, raw_boxes = parse_prediction_string(pred_str)

        # Normalize all boxes to unit-square xywh
        norm_boxes: list[tuple[float, float, float, float]] = []
        for (x1, y1, x2, y2) in raw_boxes:
            norm_boxes.append(normalize_box(x1, y1, x2, y2, orig_w, orig_h))

        # Truncate to MAX_M
        norm_boxes = norm_boxes[:MAX_M]
        num_real = len(norm_boxes)

        # Build padded (MAX_M, 4) tensor
        gt_boxes = torch.zeros(MAX_M, 4, dtype=torch.float32)
        for i, (xc, yc, w, h) in enumerate(norm_boxes):
            gt_boxes[i] = torch.tensor([xc, yc, w, h], dtype=torch.float32)

        gt_mask = torch.zeros(MAX_M, dtype=torch.bool)
        gt_mask[:num_real] = True

        # --- Apply base transforms (resize + to-tensor + normalize) ---
        image = _BASE_TRANSFORM(pil_img)  # (3, 224, 224)

        # --- Random horizontal flip (train only, applied to BOTH image and boxes) ---
        if self.train_aug and random.random() < 0.5:
            image, gt_boxes = apply_horizontal_flip(image, gt_boxes)

        # --- Bin-discretize the PRIMARY box (index 0) ---
        primary = gt_boxes[0]  # (4,) xywh of the real primary box
        target_bins = {
            "x_c": bin_discretize_coord(primary[0].item(), self.K),
            "y_c": bin_discretize_coord(primary[1].item(), self.K),
            "w":   bin_discretize_coord(primary[2].item(), self.K),
            "h":   bin_discretize_coord(primary[3].item(), self.K),
        }

        return {
            "image":       image,
            "gt_boxes":    gt_boxes,
            "gt_mask":     gt_mask,
            "target_bins": target_bins,
            "image_id":    image_id,
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def build_collate_fn():
    """Return a collate function that stacks per-sample tensors into a batch dict.

    Output keys:
    - 'images'      (B, 3, H, W) float tensor
    - 'gt_boxes'    (B, MAX_M, 4) float tensor
    - 'gt_mask'     (B, MAX_M) bool tensor
    - 'target_bins' dict of head -> (B,) long tensor  (keys: 'x_c', 'y_c', 'w', 'h')
    - 'image_ids'   list of str, length B
    """
    def collate_fn(samples: list[dict]) -> dict:
        images = torch.stack([s["image"] for s in samples], dim=0)
        gt_boxes = torch.stack([s["gt_boxes"] for s in samples], dim=0)
        gt_mask = torch.stack([s["gt_mask"] for s in samples], dim=0)
        image_ids = [s["image_id"] for s in samples]

        # target_bins: list of dicts -> dict of lists -> dict of tensors
        target_bins: dict[str, torch.Tensor] = {}
        for head in ("x_c", "y_c", "w", "h"):
            target_bins[head] = torch.tensor(
                [s["target_bins"][head] for s in samples],
                dtype=torch.long,
            )

        return {
            "images":      images,
            "gt_boxes":    gt_boxes,
            "gt_mask":     gt_mask,
            "target_bins": target_bins,
            "image_ids":   image_ids,
        }

    return collate_fn
