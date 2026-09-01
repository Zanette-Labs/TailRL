"""Tests for the coordinate seam that Diagnostic 0 surfaced: the model emits clicks in the
smart-resized pixel frame, and the reward maps them into the GT's [0,1000] frame. We pin:

  (A) PARITY: gui_grounding.resized_frame(W,H,min,max) (== Qwen smart_resize) equals the actual
      processor dims (image_grid_thw*14), INCLUDING through verl's process_image (the training
      path). If these drift, the reward's rescale is silently wrong.
  (B) ROUND-TRIP: a click placed at an element's center expressed in the MODEL frame is mapped
      back to that element's [0,1000] center and scores in-box.
  (C) wh-normalization actually changes scoring (a model-frame point that misses if read as raw
      [0,1000] hits once normalized).

These import the Qwen processor (CPU; cached in $HF_HOME) -- run inside the container.
"""
import importlib.util
import os

import pytest

import gui_grounding as R

MINP, MAXP = 262144, 1500000
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
SIZES = [(1960, 1092), (3840, 2160), (3840, 1080), (1280, 720), (800, 600), (500, 280), (1024, 768)]
# sizes already within [MINP, MAXP] -> verl process_image is a no-op, so resized_frame == processor(orig)
IN_RANGE = [(w, h) for (w, h) in SIZES if MINP <= w * h <= MAXP]


@pytest.fixture(scope="module")
def processor():
    """The real Qwen2.5-VL processor. This is the ONLY test file here that needs anything beyond a
    CPU: it downloads (or reads from $HF_HOME) the processor config, because the whole point is to
    check our frame arithmetic against the actual thing rather than against a re-implementation.
    Skip rather than fail when the Hub is unreachable, so `pytest tests` is green on a fresh clone
    with no network -- an environment problem must not look like a parity failure."""
    from transformers import AutoProcessor
    try:
        return AutoProcessor.from_pretrained(MODEL, min_pixels=MINP, max_pixels=MAXP)
    except Exception as e:                                            # network, auth, or cache miss
        pytest.skip(f"cannot load {MODEL} processor ({type(e).__name__}); "
                    f"set HF_HOME to a cache containing it, or run with network access")


def _grid_dims(processor, im):
    """(W,H) the processor actually produces = image_grid_thw * patch(14)."""
    g = processor.image_processor(images=im, return_tensors="pt")["image_grid_thw"][0].tolist()
    return g[2] * 14, g[1] * 14


def _blank(w, h):
    from PIL import Image
    return Image.new("RGB", (w, h), (123, 222, 64))


@pytest.mark.parametrize("w,h", IN_RANGE)
def test_resized_frame_matches_processor_in_range(processor, w, h):
    """For images already within [min,max]_pixels, process_image is a no-op, so resized_frame
    must equal the processor's dims on the original image (works without verl)."""
    rw, rh = R.resized_frame(w, h, MINP, MAXP)
    pw, ph = _grid_dims(processor, _blank(w, h))
    assert (rw, rh) == (pw, ph), f"resized_frame {(rw,rh)} != processor {(pw,ph)} for {(w,h)}"


@pytest.mark.parametrize("w,h", SIZES)
def test_parity_through_training_process_image(processor, w, h):
    """THE training-faithful parity: verl process_image (area resize) THEN the processor. The
    reward's resized_frame must match the dims the model ends up seeing -- INCLUDING the small-image
    upscale case (e.g. 500x280) where a bare smart_resize(original) would be wrong."""
    from verl.utils.dataset import process_image
    processed = process_image(_blank(w, h), MINP, MAXP)
    pw, ph = _grid_dims(processor, processed)
    rw, rh = R.resized_frame(w, h, MINP, MAXP)
    assert (rw, rh) == (pw, ph), f"training-path dims {(pw,ph)} != reward resized_frame {(rw,rh)}"


@pytest.mark.parametrize("w,h", SIZES)
def test_roundtrip_model_frame_center_hits(w, h):
    # element centered at [0,1000] (500,500), size 200x200
    bbox = [400, 400, 600, 600]
    rw, rh = R.resized_frame(w, h, MINP, MAXP)
    # the element center expressed in the model (resized-pixel) frame:
    mx, my = 500 / 1000 * rw, 500 / 1000 * rh
    gt = '{"bbox": [400,400,600,600], "wh": [%d, %d]}' % (w, h)
    out = R.compute_score({"response": f"({mx},{my})", "ground_truth": gt},
                          min_pixels=MINP, max_pixels=MAXP)
    assert out["accuracy"] == 1.0
    assert out["overall"] > 0.99            # lands on the center -> soft ~ 1
    # a model-frame point at the element's right edge (x2=600 in [0,1000]) maps to the edge
    ex = 600 / 1000 * rw
    edge = R.compute_score({"response": f"({ex},{my})", "ground_truth": gt}, min_pixels=MINP, max_pixels=MAXP)
    assert edge["accuracy"] == 1.0          # inclusive boundary


def test_wh_normalization_changes_scoring():
    # large image: model-frame click that is OUT of the [0,1000] box if read raw, but IN once normalized.
    w, h = 3840, 2160
    rw, rh = R.resized_frame(w, h, MINP, MAXP)          # ~ (1624, 896)
    bbox = [400, 400, 600, 600]
    mx, my = 500 / 1000 * rw, 500 / 1000 * rh           # element center in model frame (~812, 448)
    assert mx > 600                                     # raw-read would miss in x
    gt = '{"bbox": [400,400,600,600], "wh": [%d, %d]}' % (w, h)
    norm = R.compute_score({"response": f"({mx},{my})", "ground_truth": gt}, min_pixels=MINP, max_pixels=MAXP)
    assert norm["accuracy"] == 1.0                      # normalized -> hit
    raw = R.compute_score({"response": f"({mx},{my})", "ground_truth": "[400,400,600,600]"})  # bare list -> no wh
    assert raw["accuracy"] == 0.0                       # read as raw [0,1000] -> miss


def test_bare_list_gt_is_direct_no_normalization():
    # backward-compat: a bare [x1,y1,x2,y2] ground_truth is scored directly in [0,1000].
    out = R.compute_score({"response": "(500,500)", "ground_truth": "[400,400,600,600]"})
    assert out["accuracy"] == 1.0 and out["overall"] > 0.99
