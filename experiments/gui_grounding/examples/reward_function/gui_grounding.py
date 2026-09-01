"""Continuous Gaussian point-grounding reward for GUI click tasks (EasyR1 `sequential`).

Qwen2.5-VL emits click points in the coordinate frame of the image it actually SEES, i.e. the
smart-resized image (Qwen's smart_resize with the run's min/max_pixels), NOT a normalized
[0,1000] frame. So we score by mapping the model's point from that resized-pixel frame into the
GT's normalized [0,1000] frame:

    rW, rH = smart_resize(H, W, factor=28, min_pixels, max_pixels)      # the frame the model sees
    px01, py01 = px * 1000 / rW,  py * 1000 / rH                        # -> [0,1000]
    reward = Gaussian( (px01,py01) -> GT element center ),  sigma proportional to element size.

ground_truth is JSON {"bbox": [x1,y1,x2,y2] in [0,1000], "wh": [origW, origH]} written by the
converters. min_pixels/max_pixels MUST equal the run's data.min_pixels/data.max_pixels (training)
or the eval processor's (eval) — if they drift, the rescale is silently wrong and Diag 0's max@1
collapses (the tripwire). A bare-list ground_truth (no "wh") is scored directly in [0,1000] (used
by unit tests). GRADED reward (not 0/1 in-box) -- required so TailRL does not collapse to binary MaxRL.

Interface mirrors rec_iou.py: compute_score(reward_input, **kwargs) -> {overall, format, accuracy,
parse_error}. `parse_error` is 1.0 when GT/coord parsing raised (logged with the exception type), so
`reward/parse_error` in W&B is a systematic-bug tripwire rather than silent all-zero "hard prompts".
"""
import json
import logging
import math
import re
from typing import Any, Optional, Tuple

# Qwen's own smart_resize (do NOT reimplement) -- both import paths are Qwen's.
try:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
except Exception:  # pragma: no cover - fallback for other layouts
    from qwen_vl_utils import smart_resize

REWARD_NAME = "gui_gaussian"
REWARD_TYPE = "sequential"

_log = logging.getLogger("gui_grounding_reward")

_BBOX = re.compile(
    r"[\[(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\])]"
)
_PT = re.compile(r"[\[(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\])]")


def _pred_point(text: str) -> Optional[Tuple[float, float]]:
    """Predicted click point in the model's (resized-pixel) frame: a 4-tuple box -> its center,
    else the LAST (x, y) pair, else None. Shared by the eval harness."""
    boxes = _BBOX.findall(text)
    if boxes:
        x1, y1, x2, y2 = map(float, boxes[-1])
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    pts = _PT.findall(text)
    if pts:
        return (float(pts[-1][0]), float(pts[-1][1]))
    return None


def _parse_gt(ground_truth: Any) -> Tuple[list, Optional[list]]:
    """-> ([x1,y1,x2,y2] in [0,1000], [origW,origH] | None). Accepts the JSON dict
    {"bbox","wh"} the converters write, a dict under bbox/bbox_2d, or a bare list (no wh)."""
    v = ground_truth
    if isinstance(v, str):
        v = json.loads(v)
    if isinstance(v, dict):
        bbox = v.get("bbox", v.get("bbox_2d"))
        wh = v.get("wh", v.get("size"))
        wh = [float(x) for x in wh][:2] if wh else None
        return [float(x) for x in bbox][:4], wh
    return [float(x) for x in v][:4], None


def _process_image_dims(orig_w: float, orig_h: float, min_pixels: int, max_pixels: int) -> Tuple[int, int]:
    """Replicate verl.utils.dataset.process_image's AREA resize on dims only (this is NOT
    smart_resize): downscale to <= max_pixels then upscale to >= min_pixels, with int() truncation.
    Training feeds the processor the process_image'd image, so this runs before smart_resize."""
    w, h = float(orig_w), float(orig_h)
    if max_pixels is not None and w * h > max_pixels:
        f = math.sqrt(max_pixels / (w * h)); w, h = int(w * f), int(h * f)
    if min_pixels is not None and w * h < min_pixels:
        f = math.sqrt(min_pixels / (w * h)); w, h = int(w * f), int(h * f)
    return int(w), int(h)


def resized_frame(orig_w: float, orig_h: float, min_pixels: int, max_pixels: int) -> Tuple[int, int]:
    """(rW, rH): the resized-pixel dims the model actually sees in training = Qwen smart_resize
    applied AFTER verl's process_image area-resize. Bit-exact with processor(process_image(image))
    (verified by tests/screenspot/test_coord_frame.py, incl. the small-image upscale case)."""
    w, h = _process_image_dims(orig_w, orig_h, min_pixels, max_pixels)
    rh, rw = smart_resize(h, w, factor=28, min_pixels=min_pixels, max_pixels=max_pixels)
    return rw, rh


def _in_box(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    return 1.0 if (x1 <= px <= x2 and y1 <= py <= y2) else 0.0


def _to_norm(pred: Tuple[float, float], wh, min_pixels: int, max_pixels: int) -> Tuple[float, float]:
    """Map a predicted point from the model frame into [0,1000]. If wh is None the point is
    assumed already in [0,1000] (unit-test path)."""
    px, py = pred
    if wh is None:
        return px, py
    rw, rh = resized_frame(wh[0], wh[1], min_pixels, max_pixels)
    return px * 1000.0 / rw, py * 1000.0 / rh


def compute_score(
    reward_input: dict,
    sigma_scale: float = 0.5,
    sigma_min: float = 0.0,
    format_weight: float = 0.0,
    min_pixels: int = 3136,
    max_pixels: int = 12845056,
    **kwargs,
) -> dict:
    """EasyR1 `sequential` reward entry point.

    Args:
        reward_input: {"response": str, "response_length": int, "ground_truth": Any}
        sigma_scale/sigma_min: per-axis sigma = max(sigma_scale*element_side, sigma_min) in [0,1000].
        format_weight: weight on the 0/1 format flag in `overall` (default 0 -> overall == soft).
        min_pixels/max_pixels: MUST match the run's data.min_pixels/data.max_pixels (the frame the
            model saw) so the resized-frame rescale is correct.
    """
    # errata O6: response parse + point extraction live INSIDE the try so a missing/non-str
    # `response` (or any coord-parse failure) returns parse_error=1.0 instead of raising. fmt is set
    # after pred, and reset to 0.0 on the exception path (a row we could not score is not "formatted").
    err = 0.0
    fmt = 0.0
    try:
        resp = reward_input["response"]
        pred = _pred_point(resp)
        fmt = 1.0 if pred is not None else 0.0
        bbox, wh = _parse_gt(reward_input["ground_truth"])
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        if pred is None:
            soft, hit = 0.0, 0.0
        else:
            px, py = _to_norm(pred, wh, min_pixels, max_pixels)
            sx, sy = max(sigma_scale * w, sigma_min), max(sigma_scale * h, sigma_min)
            soft = math.exp(-0.5 * (((px - cx) / sx) ** 2 + ((py - cy) / sy) ** 2))
            hit = _in_box(px, py, x1, y1, x2, y2)
    except Exception as e:  # log the TYPE so a systematic format/coord bug surfaces (not silent zeros)
        _log.warning("gui_grounding reward parse failed: %s: %s", type(e).__name__, str(e)[:200])
        soft, hit, err, fmt = 0.0, 0.0, 1.0, 0.0
    return {
        "overall": (1.0 - format_weight) * soft + format_weight * fmt,
        "format": fmt,
        "accuracy": hit,
        "parse_error": err,
    }
