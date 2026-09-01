"""SE-GUI dense point reward (Yuan et al., NeurIPS 2025) on the vlm-rl frame machinery.

SE-GUI scores a predicted click with an IN-BOX indicator PLUS a distance-decay term that pays
graded credit everywhere in the image, so a group of rollouts is almost never flat:

    point   = base + decay,   base  = 1 iff the point lands in the GT box
                              decay = 1 - (d / d_max)^2      (0 if the raw distance d > 1)
    d       = || pred - box_center ||   in per-axis image-normalized units
    d_max   = max over the FOUR IMAGE CORNERS of || box_center - corner ||   (same units)
    overall = point + format_weight * format                 # ADDITIVE, range [0, 2.5]

There is NO shape knob: unlike the Gaussian reward in gui_grounding.py there is no sigma, and
the decay's spatial scale is set entirely by where the target sits in the image (d_max).

COORDINATE FRAME. Qwen2.5-VL emits click points in the frame of the image it actually SEES (the
smart-resized image), so the point is first mapped into the GT's normalized [0,1000] frame by the
SHARED helpers imported from gui_grounding (_pred_point / _parse_gt / _to_norm). The SE-GUI math
then runs in the unit square u = [0,1000] / 1000, which is EXACTLY SE-GUI's per-axis normalized
space (their x/W, y/H) with W = H = 1000: our GT box is already per-axis normalized to [0,1000],
and _to_norm divides the prediction per-axis by the model-frame width/height. So feeding their
reference implementation sol = [x1, y1, x2, y2, 1000, 1000] reproduces our `point` exactly -- this
is what tests/segui/test_segui_point.py asserts to 1e-6 against their vendored code.

min_pixels/max_pixels MUST equal the run's data.min_pixels/data.max_pixels (training) or the eval
processor's (eval); if they drift the rescale is silently wrong.

UPSTREAM DISCREPANCY (deliberate, see docs/segui/errata.md): the SE-GUI PAPER's Eq. 3 writes the
decay as (1 - d/d_max)^2; their RELEASED CODE computes 1 - (d/d_max)^2, guarded by `if d <= 1` on
the RAW distance. We match the CODE, because the code is what produced their checkpoints.

Interface mirrors gui_grounding.compute_score: `parse_error` is 1.0 when parsing raised (logged
with the exception type), so reward/parse_error in W&B is a systematic-bug tripwire rather than
silent all-zero "hard prompts".
"""
import logging
import math
import os
import sys
from typing import Tuple

# verl loads a reward file standalone via importlib.util.spec_from_file_location
# (verl/workers/reward/function.py), so this file's own directory is NOT on sys.path and a plain
# `import gui_grounding` would fail at train time. Put it there, then IMPORT the frame machinery
# instead of copying it: bit-parity with gui_grounding.resized_frame is load-bearing (a divergent
# copy would mis-map every predicted point and the failure would be silent).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from gui_grounding import _parse_gt, _pred_point, _to_norm  # noqa: E402

REWARD_NAME = "segui_point"
REWARD_TYPE = "sequential"

_log = logging.getLogger("segui_point_reward")


def _segui_decay(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """SE-GUI's distance-decay term for a point already in the [0,1000] frame.

    Transcribed operation-for-operation from SE-GUI's point_reward (their qwen_module.py) with
    W = H = 1000, including the `if d <= 1` guard on the RAW distance and their d1..d4 corner
    enumeration. The arithmetic is kept in their exact order/associativity so the parity test can
    hold to 1e-6 (and in practice to the last ulp).
    """
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    # per-axis image-normalized coordinates (their nx = x / img_width, ny = y / img_height)
    nx, ny = px / 1000.0, py / 1000.0
    ncx, ncy = cx / 1000.0, cy / 1000.0
    # normalized distance from the box center to each of the four image corners
    d1 = math.sqrt((ncx - 0) ** 2 + (ncy - 0) ** 2)
    d2 = math.sqrt((ncx - 1) ** 2 + (ncy - 0) ** 2)
    d3 = math.sqrt((ncx - 0) ** 2 + (ncy - 1) ** 2)
    d4 = math.sqrt((ncx - 1) ** 2 + (ncy - 1) ** 2)
    max_d = max(d1, d2, d3, d4)
    d = math.sqrt((nx - ncx) ** 2 + (ny - ncy) ** 2)
    d_normalized = d / max_d if max_d > 0 else 0
    return 1 - d_normalized**2 if d <= 1 else 0


def compute_score(
    reward_input: dict,
    format_weight: float = 0.5,
    min_pixels: int = 3136,
    max_pixels: int = 12845056,
    **kwargs,
) -> dict:
    """EasyR1 `sequential` reward entry point.

    Args:
        reward_input: {"response": str, "response_length": int, "ground_truth": Any}
        format_weight: ADDITIVE weight on the 0/1 format flag: overall = point + w * format.
            Range of `overall` is therefore [0, 2 + format_weight] = [0, 2.5] at the default.
        min_pixels/max_pixels: MUST match the run's data.min_pixels/data.max_pixels (the frame the
            model saw) so the resized-frame rescale is correct.

    Returns:
        {"overall", "point", "format", "accuracy", "parse_error"} where `point` is the pure SE-GUI
        term in [0, 2] and `accuracy` is the standard 0/1 in-box hit.
    """
    # Parsing lives INSIDE the try (errata O6) so a missing/non-str response or any coord-parse
    # failure returns parse_error=1.0 instead of raising. fmt is reset to 0.0 on the exception path
    # -- a row we could not score is not "formatted".
    err = 0.0
    fmt = 0.0
    try:
        resp = reward_input["response"]
        pred = _pred_point(resp)
        fmt = 1.0 if pred is not None else 0.0
        bbox, wh = _parse_gt(reward_input["ground_truth"])
        x1, y1, x2, y2 = bbox
        if pred is None:
            point, hit = 0.0, 0.0
        else:
            px, py = _to_norm(pred, wh, min_pixels, max_pixels)
            hit = 1.0 if (x1 <= px <= x2 and y1 <= py <= y2) else 0.0
            point = hit + _segui_decay(px, py, x1, y1, x2, y2)
    except Exception as e:  # log the TYPE so a systematic format/coord bug surfaces (not silent zeros)
        _log.warning("segui_point reward parse failed: %s: %s", type(e).__name__, str(e)[:200])
        point, hit, err, fmt = 0.0, 0.0, 1.0, 0.0
    return {
        "overall": point + format_weight * fmt,
        "point": point,
        "format": fmt,
        "accuracy": hit,
        "parse_error": err,
    }
