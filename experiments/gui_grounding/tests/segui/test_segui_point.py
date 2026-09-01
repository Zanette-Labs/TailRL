"""Tests for the SE-GUI dense point reward (examples/reward_function/segui_point.py).

Three layers:
  (a) hand oracles      -- closed-form values worked out by hand, to 1e-9;
  (b) the PARITY HARNESS -- 500 randomised cases scored by BOTH our implementation and SE-GUI's
      vendored released code (tests/segui/segui_reference.py); they must agree to 1e-6. This is
      the load-bearing test: it is what licenses the claim "SE-GUI reward, bit-parity";
  (c) regression        -- gui_grounding's own oracles still hold and segui_point genuinely REUSES
      its helper objects rather than shadowing them with a copy.

The unit-square identity being exploited: our GT box is per-axis normalized to [0,1000] and
_to_norm divides the prediction per-axis by the model-frame w/h, so SE-GUI's per-axis
image-normalized space is reproduced exactly by passing them sol = [x1, y1, x2, y2, 1000, 1000].
"""
import json
import math
import random

import pytest

import gui_grounding
import segui_point
from segui_reference import point_reward as segui_ref


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _score(px, py, box, **kw):
    """Score a click at (px, py) in the [0,1000] frame against a bare-list GT box.

    A bare list (no "wh") is the _parse_gt path that skips the resized-frame rescale, so the
    numbers below are pure reward math with no image-geometry confound.
    """
    return segui_point.compute_score({"response": _fmt_response(px, py), "ground_truth": list(box)}, **kw)


def _fmt(v):
    """Decimal text for v that float() round-trips EXACTLY, so our string-parsing path and the
    reference's numeric path see the identical double."""
    s = repr(float(v))
    assert float(s) == float(v) and "e" not in s and "E" not in s, f"non-round-tripping literal {s}"
    return s


def _fmt_response(px, py):
    return f"({_fmt(px)}, {_fmt(py)})"


def _ref(px, py, box):
    """SE-GUI's released point_reward on the same case, via their tool_call interface."""
    completion = [[{"content": "<tool_call>" + json.dumps({"arguments": {"coordinate": [px, py]}}) + "</tool_call>"}]]
    return segui_ref(completion, [list(box) + [1000, 1000]])[0]


# --------------------------------------------------------------------------------------------
# (a) hand oracles
# --------------------------------------------------------------------------------------------
# box centred in the image: centre (500,500) -> all four corners are equidistant,
# d_max = sqrt(0.5^2 + 0.5^2) = 0.7071067811865476
CENTRED = [400.0, 400.0, 600.0, 600.0]
# box in the top-left corner: centre (50,50) -> the far corner dominates,
# d_max = sqrt(0.95^2 + 0.95^2) = 1.3435028842544403
CORNER = [0.0, 0.0, 100.0, 100.0]

ORACLES = [
    # (px, py, box, expected point, why)
    (500.0, 500.0, CENTRED, 2.0, "dead centre: base 1 + decay 1 -> the maximum"),
    (600.0, 500.0, CENTRED, 1.98, "on the box edge: d=0.1, (d/d_max)^2=0.02, in-box -> 1 + 0.98"),
    (800.0, 500.0, CENTRED, 0.82, "outside: d=0.3, (d/d_max)^2=0.18, no base -> 0.82"),
    (1000.0, 1000.0, CENTRED, 0.0, "image corner == the argmax corner: d == d_max -> decay 0"),
    (1000.0, 1000.0, CORNER, 0.0, "d=1.3435 > 1: the raw-d guard zeroes the decay"),
    (700.0, 50.0, CORNER, 0.7659279778393353, "d=0.65, d_max=1.3435 -> 1 - (0.65/1.3435)^2"),
]


@pytest.mark.parametrize("px,py,box,expected,why", ORACLES)
def test_hand_oracles(px, py, box, expected, why):
    assert segui_point.compute_score(
        {"response": _fmt_response(px, py), "ground_truth": list(box)}
    )["point"] == pytest.approx(expected, abs=1e-9), why


@pytest.mark.parametrize("px,py,box,expected,why", ORACLES)
def test_hand_oracles_match_upstream(px, py, box, expected, why):
    """The oracles are not just self-consistent -- SE-GUI's own code produces them."""
    assert _ref(px, py, box) == pytest.approx(expected, abs=1e-9), why


def test_raw_distance_guard_actually_bites():
    """The `if d <= 1` guard is on the RAW distance, so it can zero a decay that would otherwise
    be clearly positive. Without a case in this regime the guard is untested code."""
    got = _score(900.0, 900.0, CORNER)["point"]
    d_max = math.sqrt(0.95**2 + 0.95**2)
    d = math.sqrt(0.85**2 + 0.85**2)
    unguarded = 1 - (d / d_max) ** 2
    assert d > 1.0, "test case no longer exercises the guard"
    assert unguarded == pytest.approx(0.19944598, abs=1e-6), "unguarded value drifted"
    assert got == 0.0, "the raw-d guard must zero this decay"


# --------------------------------------------------------------------------------------------
# (a') the additive format term
# --------------------------------------------------------------------------------------------
def test_overall_is_point_plus_weighted_format():
    out = _score(500.0, 500.0, CENTRED)
    assert out["format"] == 1.0
    assert out["point"] == pytest.approx(2.0)
    assert out["overall"] == pytest.approx(2.5), "overall = point + 0.5 * format; max is 2.5, not 1.0"
    assert out["accuracy"] == 1.0
    assert out["parse_error"] == 0.0


def test_format_weight_is_additive_not_convex():
    """A convex blend would pull `overall` DOWN toward the format flag; SE-GUI's is additive."""
    a = _score(500.0, 500.0, CENTRED, format_weight=0.0)
    b = _score(500.0, 500.0, CENTRED, format_weight=0.5)
    assert a["overall"] == pytest.approx(2.0)
    assert b["overall"] == pytest.approx(2.5)
    assert b["overall"] - a["overall"] == pytest.approx(0.5)


def test_reward_range_is_zero_to_two_point_five():
    assert segui_point.compute_score({"response": "no coordinates here", "ground_truth": CENTRED})["overall"] == 0.0
    assert _score(500.0, 500.0, CENTRED)["overall"] == pytest.approx(2.5)


def test_unparseable_response_is_format_zero_not_parse_error():
    """A response with no point is a legitimate (bad) rollout, NOT a systematic bug."""
    out = segui_point.compute_score({"response": "I cannot find it.", "ground_truth": CENTRED})
    assert out == {"overall": 0.0, "point": 0.0, "format": 0.0, "accuracy": 0.0, "parse_error": 0.0}


def test_broken_ground_truth_trips_parse_error():
    out = segui_point.compute_score({"response": "(500, 500)", "ground_truth": "{not json"})
    assert out["parse_error"] == 1.0
    assert out["format"] == 0.0, "a row we could not score is not 'formatted'"
    assert out["overall"] == 0.0 and out["point"] == 0.0 and out["accuracy"] == 0.0


def test_missing_response_key_trips_parse_error():
    out = segui_point.compute_score({"ground_truth": CENTRED})
    assert out["parse_error"] == 1.0 and out["overall"] == 0.0


def test_accuracy_is_the_standard_in_box_hit():
    assert _score(400.0, 400.0, CENTRED)["accuracy"] == 1.0, "inclusive on the boundary"
    assert _score(600.0, 600.0, CENTRED)["accuracy"] == 1.0, "inclusive on the boundary"
    assert _score(399.9, 500.0, CENTRED)["accuracy"] == 0.0
    assert _score(500.0, 500.0, CENTRED)["accuracy"] == 1.0


# --------------------------------------------------------------------------------------------
# (b) PARITY HARNESS -- our `point` vs SE-GUI's released code
# --------------------------------------------------------------------------------------------
def _random_cases(n=500, seed=0):
    """Cases spanning the regimes that matter: in-box, near-miss, far-miss, off-image (guard),
    tiny boxes, huge boxes, corner boxes, integer and fractional coordinates."""
    rng = random.Random(seed)
    cases = []
    while len(cases) < n:
        w = rng.choice([1, 2, 5, 20, 80, 300, 900])
        h = rng.choice([1, 2, 5, 20, 80, 300, 900])
        x1 = rng.uniform(0, 1000 - w)
        y1 = rng.uniform(0, 1000 - h)
        box = [round(x1, 4), round(y1, 4), round(x1 + w, 4), round(y1 + h, 4)]
        mode = rng.choice(["in", "near", "far", "off"])
        if mode == "in":
            px, py = rng.uniform(box[0], box[2]), rng.uniform(box[1], box[3])
        elif mode == "near":
            px, py = (box[0] + box[2]) / 2 + rng.gauss(0, 30), (box[1] + box[3]) / 2 + rng.gauss(0, 30)
        elif mode == "far":
            px, py = rng.uniform(0, 1000), rng.uniform(0, 1000)
        else:  # off-image: exercises the raw-d > 1 guard
            px, py = rng.uniform(-400, 1400), rng.uniform(-400, 1400)
        if rng.random() < 0.5:
            px, py = float(round(px)), float(round(py))
        else:
            px, py = round(px, 4), round(py, 4)
        cases.append((px, py, box))
    return cases


PARITY_CASES = _random_cases()


def test_parity_with_segui_reference():
    """Our `point` term must equal SE-GUI's released point_reward EXACTLY (to 1e-6) on every case.

    This is the STOP condition in the spec: if it fails, the reward is not SE-GUI's.
    """
    worst, worst_case = 0.0, None
    for px, py, box in PARITY_CASES:
        ours = _score(px, py, box)["point"]
        theirs = _ref(px, py, box)
        delta = abs(ours - theirs)
        if delta > worst:
            worst, worst_case = delta, (px, py, box, ours, theirs)
    assert worst <= 1e-6, f"parity broken (max |delta| = {worst:.3e}) on {worst_case}"
    # in practice this is exact; assert it so a future refactor that costs precision is visible
    assert worst == 0.0, f"parity is no longer bit-exact (max |delta| = {worst:.3e}) on {worst_case}"


def test_parity_harness_covers_every_regime():
    """A harness that silently degenerates to one regime proves nothing. Assert the coverage."""
    n_hit = n_guard = n_zero_decay = 0
    for px, py, box in PARITY_CASES:
        out = _score(px, py, box)
        n_hit += out["accuracy"] == 1.0
        cx, cy = (box[0] + box[2]) / 2000.0, (box[1] + box[3]) / 2000.0
        if math.sqrt((px / 1000.0 - cx) ** 2 + (py / 1000.0 - cy) ** 2) > 1.0:
            n_guard += 1
        n_zero_decay += out["point"] == 0.0
    assert n_hit >= 20, f"only {n_hit} in-box cases"
    assert n_guard >= 20, f"only {n_guard} cases past the raw-d guard"
    assert n_zero_decay >= 20, f"only {n_zero_decay} zero-reward cases"


def test_parity_holds_through_the_resized_frame_path():
    """The training path goes through _to_norm (wh present -> smart_resize rescale). Parity must
    survive it: score the SAME physical click expressed in model-frame pixels, and in [0,1000]."""
    rng = random.Random(7)
    for _ in range(50):
        orig_w = rng.choice([1280, 1920, 2560, 3840, 800])
        orig_h = rng.choice([720, 1080, 1440, 2160, 600])
        rw, rh = gui_grounding.resized_frame(orig_w, orig_h, 3136, 12845056)
        mx, my = rng.uniform(0, rw), rng.uniform(0, rh)
        box = [100.0, 150.0, 400.0, 450.0]
        via_frame = segui_point.compute_score(
            {
                "response": _fmt_response(round(mx, 4), round(my, 4)),
                "ground_truth": json.dumps({"bbox": box, "wh": [orig_w, orig_h]}),
            }
        )["point"]
        px, py = round(mx, 4) * 1000.0 / rw, round(my, 4) * 1000.0 / rh
        assert via_frame == pytest.approx(_ref(px, py, box), abs=1e-9)


# --------------------------------------------------------------------------------------------
# (c) regression: the shared helpers are REUSED, not copied, and still behave
# --------------------------------------------------------------------------------------------
def test_segui_point_reuses_gui_grounding_helpers():
    """Identity, not equality: a divergent copy of the frame machinery would mis-map every point
    and fail silently, so assert we hold the very same function objects."""
    assert segui_point._pred_point is gui_grounding._pred_point
    assert segui_point._parse_gt is gui_grounding._parse_gt
    assert segui_point._to_norm is gui_grounding._to_norm


def test_gui_grounding_oracles_unchanged():
    """The Gaussian reward must be untouched by this work."""
    box = [400.0, 400.0, 600.0, 600.0]
    at_centre = gui_grounding.compute_score({"response": "(500, 500)", "ground_truth": box})
    assert at_centre["overall"] == pytest.approx(1.0), "Gaussian peaks at 1.0 on the centre"
    assert at_centre["accuracy"] == 1.0 and at_centre["format"] == 1.0
    one_sigma = gui_grounding.compute_score({"response": "(600, 500)", "ground_truth": box})
    assert one_sigma["overall"] == pytest.approx(math.exp(-0.5)), "sigma = 0.5 * 200 = 100 px"
    assert gui_grounding.compute_score({"response": "nope", "ground_truth": box})["overall"] == 0.0


def test_the_two_rewards_agree_on_the_hit_criterion():
    """`accuracy` must mean the same thing in both rewards, or the eval ladder is not comparable."""
    for px, py, box in PARITY_CASES[:200]:
        a = segui_point.compute_score({"response": _fmt_response(px, py), "ground_truth": list(box)})
        b = gui_grounding.compute_score({"response": _fmt_response(px, py), "ground_truth": list(box)})
        assert a["accuracy"] == b["accuracy"]
