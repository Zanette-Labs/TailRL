"""Coverage for examples/reward_function/gui_grounding.py.

Tests gui_grounding.compute_score and gui_grounding._pred_point against the
ACTUAL implemented behavior:

    compute_score(reward_input, sigma_scale=0.5, sigma_min=20.0, format_weight=0.0)
        -> {"overall", "format", "accuracy"}

    overall   = (1 - format_weight) * soft + format_weight * format
    soft      = exp(-0.5 * ((dx/sx)^2 + (dy/sy)^2)),  sx = max(sigma_scale*w, sigma_min)
    accuracy  = 1.0 if predicted point inside GT box else 0.0
    format    = 1.0 if a point parsed else 0.0

The GT is a JSON string "[x1,y1,x2,y2]" in [0,1000]. _pred_point prefers a
4-tuple box (returns its center), else the LAST (x, y) pair.

CPU-safe: gui_grounding imports only json/math/re.
"""
import json
import math

import numpy as np
import pytest

import gui_grounding


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _gt(box):
    """JSON-encode a box list as compute_score receives it."""
    return json.dumps(list(box))


def _score(response, box, **kw):
    return gui_grounding.compute_score(
        {"response": response, "ground_truth": _gt(box)}, **kw
    )


# ---------------------------------------------------------------------------
# _pred_point parsing
# ---------------------------------------------------------------------------
def test_pred_point_paren_pair():
    assert gui_grounding._pred_point("(100, 200)") == (100.0, 200.0)


def test_pred_point_bracket_pair():
    assert gui_grounding._pred_point("[100, 200]") == (100.0, 200.0)


def test_pred_point_spaced_pair():
    assert gui_grounding._pred_point("( 100 , 200 )") == (100.0, 200.0)


def test_pred_point_trailing_text():
    # text after the pair must not break parsing
    assert gui_grounding._pred_point("the click is at (300, 400) ok") == (300.0, 400.0)


def test_pred_point_leading_text():
    assert gui_grounding._pred_point("answer: (12.5, 34.5)") == (12.5, 34.5)


def test_pred_point_four_tuple_returns_center():
    # 4-tuple box -> center
    assert gui_grounding._pred_point("[100, 200, 300, 400]") == (200.0, 300.0)


def test_pred_point_box_preferred_over_pair():
    # a box present anywhere is preferred over a bare pair
    out = gui_grounding._pred_point("first (1, 2) then [10, 20, 30, 40]")
    assert out == (20.0, 30.0)


def test_pred_point_multiple_pairs_uses_last():
    assert gui_grounding._pred_point("(1, 2) and (3, 4) and (5, 6)") == (5.0, 6.0)


def test_pred_point_multiple_boxes_uses_last():
    out = gui_grounding._pred_point("[0,0,10,10] [100,100,300,300]")
    assert out == (200.0, 200.0)


def test_pred_point_none_when_no_point():
    assert gui_grounding._pred_point("no coordinates here") is None


def test_pred_point_none_when_empty():
    assert gui_grounding._pred_point("") is None


def test_pred_point_negative_and_float():
    assert gui_grounding._pred_point("(-5, 12.25)") == (-5.0, 12.25)


# ---------------------------------------------------------------------------
# accuracy (in-box hit) and soft reward at center
# ---------------------------------------------------------------------------
def test_center_hit_and_soft_one():
    # point exactly at the center -> hit and soft == 1.0
    box = [400, 400, 600, 600]  # center (500, 500)
    out = _score("(500, 500)", box, format_weight=0.0)
    assert out["accuracy"] == 1.0
    assert out["format"] == 1.0
    assert math.isclose(out["overall"], 1.0, abs_tol=1e-9)


def test_center_soft_is_exactly_one():
    box = [0, 0, 1000, 1000]
    out = _score("(500, 500)", box, format_weight=0.0)
    assert out["overall"] == 1.0


# ---------------------------------------------------------------------------
# edge soft value with sigma_min=0
# ---------------------------------------------------------------------------
def test_edge_soft_is_exp_minus_half():
    # box [400,400,600,600]: cx=cy=500, w=h=200, sx=0.5*200=100 (sigma_min=0).
    # point at (x2, cy) = (600, 500): dx=100 dy=0 -> soft = exp(-0.5).
    box = [400, 400, 600, 600]
    out = _score("(600, 500)", box, sigma_scale=0.5, sigma_min=0.0, format_weight=0.0)
    assert math.isclose(out["overall"], math.exp(-0.5), rel_tol=1e-9)
    # the point sits ON the boundary x2 -> inclusive in-box -> hit
    assert out["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# hit flips across the edge; soft is continuous across it
# ---------------------------------------------------------------------------
def test_hit_flips_across_edge():
    box = [400, 400, 600, 600]
    inside = _score("(599, 500)", box, sigma_scale=0.5, sigma_min=0.0)
    outside = _score("(601, 500)", box, sigma_scale=0.5, sigma_min=0.0)
    assert inside["accuracy"] == 1.0
    assert outside["accuracy"] == 0.0


def test_soft_continuous_across_edge():
    # soft just inside vs just outside the edge differs by a tiny amount,
    # while the binary hit flips. sigma_min=0 -> sx = 0.5*200 = 100.
    box = [400, 400, 600, 600]
    inside = _score("(599, 500)", box, sigma_scale=0.5, sigma_min=0.0)
    outside = _score("(601, 500)", box, sigma_scale=0.5, sigma_min=0.0)
    # both near exp(-0.5*(99/100)^2) and exp(-0.5*(101/100)^2)
    assert math.isclose(inside["overall"], math.exp(-0.5 * (99 / 100) ** 2), rel_tol=1e-9)
    assert math.isclose(outside["overall"], math.exp(-0.5 * (101 / 100) ** 2), rel_tol=1e-9)
    # continuity: outputs are close despite the binary flip
    assert abs(inside["overall"] - outside["overall"]) < 0.05


# ---------------------------------------------------------------------------
# sigma_min floor on tiny targets
# ---------------------------------------------------------------------------
def test_sigma_min_floor_prevents_zero_spike():
    # 2x2 box: w=h=2 -> sigma_scale*w = 1; floor sigma_min=20 dominates.
    # point ~50 units away -> soft = exp(-0.5*(50/20)^2) ~ 0.0439, clearly nonzero.
    box = [499, 499, 501, 501]                      # center (500, 500), w = h = 2
    out = _score("(550, 500)", box, sigma_scale=0.5, sigma_min=20.0, format_weight=0.0)
    assert out["overall"] > 0.02
    assert math.isclose(out["overall"], math.exp(-0.5 * (50 / 20) ** 2), rel_tol=1e-9)
    # without the floor (sigma_min=0) sigma would be 1 -> soft underflows to ~0
    no_floor = _score("(550, 500)", box, sigma_scale=0.5, sigma_min=0.0)
    assert no_floor["overall"] < 1e-100


# ---------------------------------------------------------------------------
# no parseable point / garbage / malformed GT
# ---------------------------------------------------------------------------
def test_no_point_all_zero():
    out = _score("I do not know", [0, 0, 100, 100])
    # no point parsed but GT is valid -> zeros, and NO parse error (the GT parsed fine)
    assert out["overall"] == 0.0 and out["format"] == 0.0 and out["accuracy"] == 0.0
    assert out["parse_error"] == 0.0


def test_empty_response_all_zero():
    out = _score("", [0, 0, 100, 100])
    assert out["overall"] == 0.0 and out["format"] == 0.0 and out["accuracy"] == 0.0
    assert out["parse_error"] == 0.0


def test_pure_garbage_all_zero():
    out = _score("!@#$%^&* no nums", [10, 10, 20, 20])
    assert out["overall"] == 0.0
    assert out["format"] == 0.0
    assert out["accuracy"] == 0.0


def test_malformed_gt_parse_error_and_format_zero():
    # errata O6: malformed GT -> exception -> parse_error 1; response read + _pred_point now live
    # INSIDE the try and fmt is RESET to 0 on the exception path (a row we could not score is not
    # "formatted"); soft/hit forced to 0. (Was: format stayed 1 with the pre-O6 outside-try layout.)
    out = gui_grounding.compute_score(
        {"response": "(500, 500)", "ground_truth": "not json at all"},
        format_weight=0.0,
    )
    assert out["format"] == 0.0
    assert out["overall"] == 0.0
    assert out["accuracy"] == 0.0
    assert out["parse_error"] == 1.0


def test_malformed_gt_and_no_point_all_zero():
    out = gui_grounding.compute_score(
        {"response": "nothing", "ground_truth": "{bad json"},
    )
    # malformed GT raises inside compute_score -> zeros AND parse_error flagged
    assert out["overall"] == 0.0 and out["format"] == 0.0 and out["accuracy"] == 0.0
    assert out["parse_error"] == 1.0


# ---------------------------------------------------------------------------
# format_weight blending
# ---------------------------------------------------------------------------
def test_format_weight_blend_far_miss():
    # correctly formatted but a far miss. format_weight=0.5 ->
    # overall = 0.5*soft + 0.5*1.0
    box = [400, 400, 600, 600]  # cx=cy=500, w=h=200, sx=sy=100 (default sigma_min=20<100)
    response = "(0, 0)"  # dx=dy=500 -> soft = exp(-0.5*(25+25)) ~ 0
    out = _score(response, box, sigma_scale=0.5, sigma_min=20.0, format_weight=0.5)
    soft = math.exp(-0.5 * ((500 / 100) ** 2 + (500 / 100) ** 2))
    assert math.isclose(out["overall"], 0.5 * soft + 0.5 * 1.0, rel_tol=1e-9)
    # soft is essentially 0, so overall ~ 0.5 driven by the format term
    assert math.isclose(out["overall"], 0.5, abs_tol=1e-3)
    assert out["format"] == 1.0
    assert out["accuracy"] == 0.0


def test_format_weight_full_center_hit():
    box = [0, 0, 1000, 1000]
    out = _score("(500, 500)", box, format_weight=0.5)
    # soft == 1, fmt == 1 -> overall == 1
    assert math.isclose(out["overall"], 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# bounds: soft & overall in [0,1], hit in {0,1} over random inputs
# ---------------------------------------------------------------------------
def test_bounds_over_random_inputs():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        px, py = rng.integers(0, 1001, size=2)
        x1, x2 = sorted(rng.integers(0, 1001, size=2).tolist())
        y1, y2 = sorted(rng.integers(0, 1001, size=2).tolist())
        out = _score(f"({px}, {py})", [x1, y1, x2, y2], format_weight=0.0)
        assert 0.0 <= out["overall"] <= 1.0
        assert out["accuracy"] in (0.0, 1.0)
        assert out["format"] == 1.0


def test_bounds_with_format_weight_random():
    rng = np.random.default_rng(12345)
    for _ in range(1000):
        px, py = rng.integers(-200, 1200, size=2)
        x1, x2 = sorted(rng.integers(0, 1001, size=2).tolist())
        y1, y2 = sorted(rng.integers(0, 1001, size=2).tolist())
        fw = float(rng.random())
        out = _score(f"({px}, {py})", [x1, y1, x2, y2], format_weight=fw)
        assert 0.0 <= out["overall"] <= 1.0
        assert out["accuracy"] in (0.0, 1.0)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def test_determinism():
    box = [123, 45, 678, 900]
    a = _score("(300, 400)", box, sigma_scale=0.3, sigma_min=15.0, format_weight=0.25)
    b = _score("(300, 400)", box, sigma_scale=0.3, sigma_min=15.0, format_weight=0.25)
    assert a == b


# ---------------------------------------------------------------------------
# exact bounds 0 and 1000, and out-of-range predictions
# ---------------------------------------------------------------------------
def test_exact_bounds_zero_and_thousand():
    box = [0, 0, 1000, 1000]
    out = _score("(0, 1000)", box, format_weight=0.0)
    # corner is inside the inclusive box
    assert out["accuracy"] == 1.0
    assert 0.0 <= out["overall"] <= 1.0


def test_point_at_full_corner_box():
    box = [990, 990, 1000, 1000]  # cx=cy=995, w=h=10, sx=sy=max(5,20)=20
    out = _score("(1000, 1000)", box, sigma_scale=0.5, sigma_min=20.0)
    assert out["accuracy"] == 1.0
    soft = math.exp(-0.5 * ((5 / 20) ** 2 + (5 / 20) ** 2))
    assert math.isclose(out["overall"], soft, rel_tol=1e-9)


def test_negative_prediction_no_crash_low_reward():
    box = [400, 400, 600, 600]
    out = _score("(-500, -500)", box, sigma_scale=0.5, sigma_min=20.0)
    assert out["accuracy"] == 0.0
    assert out["overall"] < 0.01
    assert out["format"] == 1.0


def test_over_thousand_prediction_no_crash_low_reward():
    box = [400, 400, 600, 600]
    out = _score("(5000, 5000)", box, sigma_scale=0.5, sigma_min=20.0)
    assert out["accuracy"] == 0.0
    assert out["overall"] < 0.01
    assert out["format"] == 1.0


# ---------------------------------------------------------------------------
# returned dict shape
# ---------------------------------------------------------------------------
def test_return_keys_subset_with_overall():
    out = _score("(500, 500)", [0, 0, 1000, 1000])
    assert set(out.keys()) <= {"overall", "format", "accuracy", "parse_error"}
    assert "overall" in out


def test_gt_accepts_list_directly():
    # _gt_box tolerates a raw list (not only a JSON string)
    out = gui_grounding.compute_score(
        {"response": "(500, 500)", "ground_truth": [0, 0, 1000, 1000]},
        format_weight=0.0,
    )
    assert out["accuracy"] == 1.0
    assert math.isclose(out["overall"], 1.0, rel_tol=1e-9)


def test_box_prediction_in_response_uses_center():
    # model emits a box; reward uses its center against the GT box
    box = [400, 400, 600, 600]
    out = _score("predicted region [450, 450, 550, 550]", box, format_weight=0.0)
    # center (500,500) is GT center -> hit + soft 1
    assert out["accuracy"] == 1.0
    assert math.isclose(out["overall"], 1.0, rel_tol=1e-9)
