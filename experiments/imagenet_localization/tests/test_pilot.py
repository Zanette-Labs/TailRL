"""Tests for experiments.imagenet_localization.pilot.

Does NOT require the ImageNet dataset for the pure-function tests.
Integration tests that touch the dataset or GPU are guarded by
requires_data and requires_gpu markers from conftest.
"""

from __future__ import annotations

import argparse

import pytest
import torch

from experiments.imagenet_localization.tests.conftest import (
    requires_data,
    requires_gpu,
)
from experiments.imagenet_localization.pilot import (
    parse_args,
    pick_pilot_image,
    run_pilot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockSample:
    """Minimal dict-like object representing one dataset sample."""

    def __init__(self, image_id: str, gt_boxes: torch.Tensor, gt_mask: torch.Tensor):
        self.image_id = image_id
        self.gt_boxes = gt_boxes    # (MAX_M, 4)
        self.gt_mask  = gt_mask     # (MAX_M,) bool
        # Dummy image: all-zeros (3, 224, 224)
        self.image = torch.zeros(3, 224, 224)

    def __getitem__(self, key: str):
        return getattr(self, key)


def _make_mock_dataset(entries: list[dict]) -> list[_MockSample]:
    """Build a list of _MockSample objects from a list of spec dicts.

    Each spec dict must have:
        image_id:  str
        boxes:     list of (xc, yc, w, h) tuples for REAL GT boxes
    """
    MAX_M = 8
    samples = []
    for spec in entries:
        boxes = spec["boxes"]
        num_real = min(len(boxes), MAX_M)
        gt_boxes = torch.zeros(MAX_M, 4, dtype=torch.float32)
        for i, (xc, yc, w, h) in enumerate(boxes[:MAX_M]):
            gt_boxes[i] = torch.tensor([xc, yc, w, h], dtype=torch.float32)
        gt_mask = torch.zeros(MAX_M, dtype=torch.bool)
        gt_mask[:num_real] = True
        samples.append(_MockSample(spec["image_id"], gt_boxes, gt_mask))
    return samples


# ---------------------------------------------------------------------------
# Test 1: parse_args defaults
# ---------------------------------------------------------------------------


def test_pilot_parse_args_defaults(monkeypatch, tmp_path):
    """Verify all default values from the spec.

    --data_dir has no baked-in path any more, so IMAGENET_DIR has to supply one
    before parse_args() will return at all.
    """
    monkeypatch.setenv("IMAGENET_DIR", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["pilot.py"])
    args = parse_args()

    assert args.K == 10, f"K default should be 10, got {args.K}"
    assert args.N == 64, f"N default should be 64, got {args.N}"
    assert args.steps == 1000, f"steps default should be 1000, got {args.steps}"
    assert abs(args.lr - 3e-3) < 1e-10, f"lr default should be 3e-3, got {args.lr}"
    assert abs(args.target_iou - 0.8) < 1e-10, f"target_iou default should be 0.8, got {args.target_iou}"
    assert args.seed == 42, f"seed default should be 42, got {args.seed}"
    assert args.search_limit == 100, f"search_limit default should be 100, got {args.search_limit}"
    assert args.data_dir == str(tmp_path.resolve()), (
        f"data_dir default mismatch: {args.data_dir!r}"
    )
    assert args.image_id is None, f"image_id default should be None, got {args.image_id!r}"
    assert args.device is None, f"device default should be None, got {args.device!r}"


def test_pilot_data_dir_follows_imagenet_dir(monkeypatch, tmp_path):
    """--data_dir defaults to $IMAGENET_DIR, and an explicit flag still wins."""
    explicit = tmp_path / "elsewhere"
    explicit.mkdir()
    monkeypatch.setenv("IMAGENET_DIR", str(tmp_path))

    monkeypatch.setattr("sys.argv", ["pilot.py"])
    assert parse_args().data_dir == str(tmp_path.resolve()), (
        "data_dir default should follow $IMAGENET_DIR"
    )

    monkeypatch.setattr("sys.argv", ["pilot.py", "--data_dir", str(explicit)])
    assert parse_args().data_dir == str(explicit), (
        "an explicit --data_dir must win over $IMAGENET_DIR"
    )


def test_pilot_data_dir_requires_imagenet_dir(monkeypatch):
    """With neither $IMAGENET_DIR nor --data_dir there is nothing to fall back on.

    parse_args() must exit with a message naming the variable rather than
    silently using some machine-specific path.
    """
    monkeypatch.delenv("IMAGENET_DIR", raising=False)
    monkeypatch.setattr("sys.argv", ["pilot.py"])

    with pytest.raises(SystemExit) as excinfo:
        parse_args()

    message = str(excinfo.value)
    assert "IMAGENET_DIR" in message, f"error should name IMAGENET_DIR, got {message!r}"
    assert "--data_dir" in message, f"error should mention --data_dir, got {message!r}"


# ---------------------------------------------------------------------------
# Test 2: pick_pilot_image selects the right entry on a mock dataset
# ---------------------------------------------------------------------------


def test_pick_pilot_image_selects_easy_single_gt():
    """pick_pilot_image picks the first entry with single-GT + easy-tier area.

    Mock dataset of 5 entries:
      0: multi-GT (2 boxes) — should be skipped
      1: single-GT, hard area (area=0.04, w=0.2, h=0.2) — should be skipped
      2: single-GT, easy area (area=0.50, w=0.5, h=1.0 -> clamped later) — PICK THIS
      3: single-GT, easy area — would also match but entry 2 is found first
      4: multi-GT — skipped
    """
    entries = [
        # Entry 0: 2 boxes (multi-GT)
        {"image_id": "multi_gt", "boxes": [(0.5, 0.5, 0.5, 0.5), (0.2, 0.2, 0.2, 0.2)]},
        # Entry 1: 1 box but hard tier (area = 0.2*0.2 = 0.04)
        {"image_id": "hard_tier", "boxes": [(0.5, 0.5, 0.2, 0.2)]},
        # Entry 2: 1 box, easy tier (area = 0.7*0.7 = 0.49, in [0.3, 0.7])
        {"image_id": "easy_target", "boxes": [(0.5, 0.5, 0.7, 0.7)]},
        # Entry 3: 1 box, easy tier too — but entry 2 wins
        {"image_id": "easy_second", "boxes": [(0.4, 0.4, 0.6, 0.6)]},
        # Entry 4: 3 boxes (multi-GT)
        {"image_id": "another_multi", "boxes": [(0.5, 0.5, 0.3, 0.3), (0.2, 0.2, 0.1, 0.1), (0.8, 0.8, 0.1, 0.1)]},
    ]
    dataset = _make_mock_dataset(entries)

    image, gt_boxes, gt_mask, image_id = pick_pilot_image(dataset, search_limit=10)

    assert image_id == "easy_target", (
        f"Expected 'easy_target', got {image_id!r}"
    )
    assert gt_mask.sum().item() == 1, (
        f"Should have exactly 1 real GT box, got {gt_mask.sum().item()}"
    )
    # Verify the area is in the easy tier
    w = gt_boxes[0, 2].item()
    h = gt_boxes[0, 3].item()
    area = w * h
    assert 0.30 <= area <= 0.70, (
        f"Selected box area {area:.4f} not in easy tier [0.30, 0.70]"
    )


def test_pick_pilot_image_picks_boundary_easy_area():
    """pick_pilot_image accepts area == 0.30 and area == 0.70 (boundaries inclusive)."""
    # Entry with area = 0.30 exactly: w * h = 0.30. E.g., w=0.6, h=0.5
    entries_low = [
        {"image_id": "boundary_low", "boxes": [(0.5, 0.5, 0.6, 0.5)]},  # area = 0.3
    ]
    entries_high = [
        {"image_id": "boundary_high", "boxes": [(0.5, 0.5, 0.7, 1.0)]},  # area = 0.7
    ]

    dataset_low = _make_mock_dataset(entries_low)
    _, _, _, image_id_low = pick_pilot_image(dataset_low, search_limit=5)
    assert image_id_low == "boundary_low"

    dataset_high = _make_mock_dataset(entries_high)
    _, _, _, image_id_high = pick_pilot_image(dataset_high, search_limit=5)
    assert image_id_high == "boundary_high"


# ---------------------------------------------------------------------------
# Test 3: pick_pilot_image raises RuntimeError when nothing qualifies
# ---------------------------------------------------------------------------


def test_pick_pilot_image_raises_when_none_found():
    """RuntimeError is raised if no image passes both filters within search_limit."""
    entries = [
        # All multi-GT
        {"image_id": "multi_1", "boxes": [(0.5, 0.5, 0.5, 0.5), (0.2, 0.2, 0.3, 0.3)]},
        {"image_id": "multi_2", "boxes": [(0.3, 0.3, 0.4, 0.4), (0.7, 0.7, 0.2, 0.2)]},
        # Single-GT but hard tier (area = 0.05*0.05 = 0.0025)
        {"image_id": "hard_1", "boxes": [(0.5, 0.5, 0.05, 0.05)]},
        # Single-GT but medium tier (area = 0.1*0.1 = 0.01 < 0.30)
        {"image_id": "medium_1", "boxes": [(0.5, 0.5, 0.1, 0.1)]},
    ]
    dataset = _make_mock_dataset(entries)

    with pytest.raises(RuntimeError, match="No easy-tier single-GT val image found"):
        pick_pilot_image(dataset, search_limit=10)


def test_pick_pilot_image_respects_search_limit():
    """pick_pilot_image stops at search_limit even if a qualifying image exists beyond it."""
    entries = [
        # Entries 0, 1: hard tier (should be skipped)
        {"image_id": "hard_0", "boxes": [(0.5, 0.5, 0.1, 0.1)]},
        {"image_id": "hard_1", "boxes": [(0.5, 0.5, 0.1, 0.1)]},
        # Entry 2: easy single-GT (but beyond search_limit=2)
        {"image_id": "easy_out_of_range", "boxes": [(0.5, 0.5, 0.7, 0.7)]},
    ]
    dataset = _make_mock_dataset(entries)

    # search_limit=2 means we only look at indices 0 and 1
    with pytest.raises(RuntimeError):
        pick_pilot_image(dataset, search_limit=2)


# ---------------------------------------------------------------------------
# Test 4: run_pilot happy path (requires_data + requires_gpu, short run)
# ---------------------------------------------------------------------------


@requires_data
@requires_gpu
def test_run_pilot_happy_path(data_dir):
    """End-to-end smoke: run_pilot with steps=50 and verify result is finite.

    Not a strict pass test — just confirms the code runs without error
    and returns a finite IoU.
    """
    args = argparse.Namespace(
        data_dir=data_dir,
        K=10,
        N=32,
        steps=50,
        lr=3e-3,
        seed=42,
        search_limit=100,
        target_iou=0.8,
        device="cuda",
        image_id=None,
    )

    passed, final_iou, reason = run_pilot(args)

    assert isinstance(passed, bool), "run_pilot must return a bool as first element"
    assert isinstance(final_iou, float), "run_pilot must return a float IoU as second element"
    assert isinstance(reason, str), "run_pilot must return a str reason as third element"
    assert 0.0 <= final_iou <= 1.0 or final_iou == final_iou, (
        f"final_iou={final_iou} is not finite"
    )
    import math
    assert math.isfinite(final_iou), f"final_iou={final_iou} is not finite"
    print(f"\nrun_pilot (50 steps) => passed={passed}, final_iou={final_iou:.3f}")
