"""Tests for the size-binning helper used by the size-binned pass@k eval.

Covers:
  - eval_screenspot_pro._row_size_bin(row, edges)  -- GT box area as a fraction of the
    [0,1000]^2 canvas -> one of _SIZE_LABELS via bisect_right over `edges`.
  - eval_screenspot_pro._SIZE_LABELS

_row_size_bin reads row["gt"], which may be a JSON string, a dict, or a bare list. When it
is (or decodes to) a dict, the box comes from the "bbox" key (falling back to "bbox_2d");
otherwise the value itself is treated as the [x1,y1,x2,y2] box. The area fraction is
    max(0, x2-x1) * max(0, y2-y1) / 1e6
and the label is _SIZE_LABELS[bisect.bisect_right(edges, area_frac)] (bisect_right, so a
fraction that lands exactly on an edge bins UP into the next-larger bucket).

All data is synthetic / in-memory. eval_screenspot_pro imports vLLM/transformers only inside
main(), so importing it here is CPU-safe.
"""
import bisect
import json

import numpy as np
import pytest

import eval_screenspot_pro as E


# The argparse default for --size_edges (see main()); _SIZE_LABELS has one more entry.
DEFAULT_EDGES = [0.0005, 0.002, 0.01]


# --------------------------------------------------------------------------- #
# helpers: the three GT representations _row_size_bin must accept
# --------------------------------------------------------------------------- #

def gt_json(bbox, key="bbox", wh=(1000, 1000)):
    """row whose gt is a JSON string of a dict {key: bbox, "wh": wh}."""
    return {"gt": json.dumps({key: list(bbox), "wh": list(wh)})}


def gt_dict(bbox, key="bbox"):
    """row whose gt is a dict (already decoded, not a string)."""
    return {"gt": {key: list(bbox)}}


def gt_list(bbox):
    """row whose gt is a bare [x1,y1,x2,y2] list."""
    return {"gt": list(bbox)}


def expected_label(bbox, edges=DEFAULT_EDGES):
    """Independent re-implementation of the size-bin, for cross-checking."""
    x1, y1, x2, y2 = [float(t) for t in bbox][:4]
    area_frac = max(0.0, x2 - x1) * max(0.0, y2 - y1) / 1.0e6
    return E._SIZE_LABELS[bisect.bisect_right(edges, area_frac)]


# --------------------------------------------------------------------------- #
# _SIZE_LABELS
# --------------------------------------------------------------------------- #

def test_size_labels_exact_content():
    assert E._SIZE_LABELS == ["tiny", "small", "medium", "large"]


def test_size_labels_length_matches_default_edges_plus_one():
    # The whole point of the bins: n edges -> n+1 buckets, so bisect can never index
    # past the end of _SIZE_LABELS when using the default edges.
    assert len(E._SIZE_LABELS) == len(DEFAULT_EDGES) + 1


def test_max_bisect_index_in_range_for_default_edges():
    # Largest possible bisect_right index with the default edges is len(edges) == 3,
    # which must be a valid index into the 4-entry label list.
    assert bisect.bisect_right(DEFAULT_EDGES, 1.0) == len(DEFAULT_EDGES)
    assert bisect.bisect_right(DEFAULT_EDGES, 1.0) < len(E._SIZE_LABELS)


# --------------------------------------------------------------------------- #
# anchor cases (exact values from the spec), via the JSON-string representation
# --------------------------------------------------------------------------- #

def test_anchor_tiny_10x10():
    # area 100 -> frac 1e-4 -> tiny
    assert E._row_size_bin(gt_json([495, 495, 505, 505]), DEFAULT_EDGES) == "tiny"


def test_anchor_small_30x30():
    # area 900 -> frac 9e-4 -> small
    assert E._row_size_bin(gt_json([485, 485, 515, 515]), DEFAULT_EDGES) == "small"


def test_anchor_medium_80x80():
    # area 6400 -> frac 6.4e-3 -> medium
    assert E._row_size_bin(gt_json([460, 460, 540, 540]), DEFAULT_EDGES) == "medium"


def test_anchor_large_300x300():
    # area 90000 -> frac 0.09 -> large
    assert E._row_size_bin(gt_json([350, 350, 650, 650]), DEFAULT_EDGES) == "large"


def test_all_four_anchors_together():
    cases = {
        "tiny": [495, 495, 505, 505],
        "small": [485, 485, 515, 515],
        "medium": [460, 460, 540, 540],
        "large": [350, 350, 650, 650],
    }
    for want, box in cases.items():
        assert E._row_size_bin(gt_json(box), DEFAULT_EDGES) == want


# --------------------------------------------------------------------------- #
# bisect_right boundary semantics (a fraction exactly on an edge bins UP)
# --------------------------------------------------------------------------- #

def test_boundary_frac_exactly_first_edge_is_small():
    # area 500 -> frac == 0.0005 == edges[0]; bisect_right sends it to bucket 1 == "small".
    assert 500 / 1.0e6 == 0.0005          # sanity: the fraction is bit-exact on the edge
    assert bisect.bisect_right(DEFAULT_EDGES, 0.0005) == 1
    assert E._row_size_bin(gt_list([0, 0, 25, 20]), DEFAULT_EDGES) == "small"


def test_boundary_frac_exactly_second_edge_is_medium():
    # area 2000 -> frac == 0.002 == edges[1] -> bucket 2 == "medium".
    assert 2000 / 1.0e6 == 0.002
    assert E._row_size_bin(gt_list([0, 0, 40, 50]), DEFAULT_EDGES) == "medium"


def test_boundary_frac_exactly_third_edge_is_large():
    # area 10000 -> frac == 0.01 == edges[2] -> bucket 3 == "large".
    assert 10000 / 1.0e6 == 0.01
    assert E._row_size_bin(gt_list([0, 0, 100, 100]), DEFAULT_EDGES) == "large"


def test_boundary_just_below_first_edge_is_tiny():
    # area 499 -> frac 4.99e-4 < 0.0005 -> stays tiny.
    assert E._row_size_bin(gt_list([0, 0, 499, 1]), DEFAULT_EDGES) == "tiny"


def test_boundary_just_above_first_edge_is_small():
    # area 501 -> frac 5.01e-4 > 0.0005 -> small.
    assert E._row_size_bin(gt_list([0, 0, 1, 501]), DEFAULT_EDGES) == "small"


# --------------------------------------------------------------------------- #
# GT representation variants (dict, bbox_2d key, bare list, json-list string)
# --------------------------------------------------------------------------- #

def test_gt_as_plain_dict():
    # gt is already a dict, not a JSON string.
    assert E._row_size_bin(gt_dict([495, 495, 505, 505]), DEFAULT_EDGES) == "tiny"
    assert E._row_size_bin(gt_dict([350, 350, 650, 650]), DEFAULT_EDGES) == "large"


def test_gt_dict_with_bbox_2d_key():
    # No "bbox" key -> falls back to "bbox_2d".
    assert E._row_size_bin(gt_dict([460, 460, 540, 540], key="bbox_2d"), DEFAULT_EDGES) == "medium"


def test_gt_json_string_with_bbox_2d_key():
    assert E._row_size_bin(gt_json([460, 460, 540, 540], key="bbox_2d"), DEFAULT_EDGES) == "medium"


def test_gt_as_bare_list():
    assert E._row_size_bin(gt_list([485, 485, 515, 515]), DEFAULT_EDGES) == "small"
    assert E._row_size_bin(gt_list([350, 350, 650, 650]), DEFAULT_EDGES) == "large"


def test_gt_as_json_string_of_bare_list():
    # json.loads decodes to a list (not a dict) -> value itself is the box.
    row = {"gt": json.dumps([350, 350, 650, 650])}
    assert E._row_size_bin(row, DEFAULT_EDGES) == "large"


def test_bbox_takes_precedence_over_bbox_2d():
    # v.get("bbox", v.get("bbox_2d")): when both keys exist, "bbox" wins.
    row = {"gt": {"bbox": [495, 495, 505, 505],       # tiny
                  "bbox_2d": [350, 350, 650, 650]}}    # large (ignored)
    assert E._row_size_bin(row, DEFAULT_EDGES) == "tiny"


def test_all_representations_agree_for_same_box():
    box = [460, 460, 540, 540]  # medium
    labels = {
        E._row_size_bin(gt_json(box), DEFAULT_EDGES),
        E._row_size_bin(gt_dict(box), DEFAULT_EDGES),
        E._row_size_bin(gt_list(box), DEFAULT_EDGES),
        E._row_size_bin({"gt": json.dumps(box)}, DEFAULT_EDGES),
    }
    assert labels == {"medium"}


# --------------------------------------------------------------------------- #
# coordinate parsing details
# --------------------------------------------------------------------------- #

def test_float_coordinates():
    # 100x100 box at fractional offsets -> area 10000 -> frac 0.01 -> large (edge -> up).
    assert E._row_size_bin(gt_list([100.5, 100.5, 200.5, 200.5]), DEFAULT_EDGES) == "large"


def test_string_coordinates_are_coerced_to_float():
    # each coord is float()-ed, so numeric strings work.
    row = {"gt": {"bbox": ["495", "495", "505", "505"]}}
    assert E._row_size_bin(row, DEFAULT_EDGES) == "tiny"


def test_extra_coordinates_beyond_four_are_sliced():
    # [x1,y1,x2,y2, extra] -> only the first four are unpacked.
    assert E._row_size_bin(gt_list([350, 350, 650, 650, 7]), DEFAULT_EDGES) == "large"


def test_wh_field_does_not_affect_binning():
    # The "wh" field in the JSON GT is ignored; only bbox drives the area fraction.
    a = E._row_size_bin(gt_json([460, 460, 540, 540], wh=(1000, 1000)), DEFAULT_EDGES)
    b = E._row_size_bin(gt_json([460, 460, 540, 540], wh=(7, 3)), DEFAULT_EDGES)
    assert a == b == "medium"


# --------------------------------------------------------------------------- #
# degenerate boxes -> area clamped to 0 -> tiny
# --------------------------------------------------------------------------- #

def test_zero_area_point_is_tiny():
    assert E._row_size_bin(gt_list([500, 500, 500, 500]), DEFAULT_EDGES) == "tiny"


def test_negative_width_clamped_to_tiny():
    # x2 < x1 -> max(0, x2-x1) == 0 -> area 0 -> tiny.
    assert E._row_size_bin(gt_list([600, 400, 500, 500]), DEFAULT_EDGES) == "tiny"


def test_negative_height_clamped_to_tiny():
    # y2 < y1 -> area 0 -> tiny.
    assert E._row_size_bin(gt_list([400, 600, 500, 500]), DEFAULT_EDGES) == "tiny"


# --------------------------------------------------------------------------- #
# custom edges
# --------------------------------------------------------------------------- #

def test_custom_single_edge_splits_tiny_small():
    # One edge -> two buckets ("tiny" below, "small" at/above).
    edges = [0.005]
    assert E._row_size_bin(gt_list([460, 460, 540, 540]), edges) == "small"  # frac 6.4e-3 >= 0.005
    assert E._row_size_bin(gt_list([485, 485, 515, 515]), edges) == "tiny"   # frac 9e-4  <  0.005


def test_custom_edges_shift_all_anchors_up():
    # Very small edges push every non-trivial box into the top buckets.
    edges = [1e-5, 5e-5, 1e-4]
    assert E._row_size_bin(gt_list([485, 485, 515, 515]), edges) == "large"   # frac 9e-4
    assert E._row_size_bin(gt_list([460, 460, 540, 540]), edges) == "large"   # frac 6.4e-3


def test_custom_edges_boundary_is_bisect_right():
    # frac exactly on a custom edge bins up.
    edges = [0.0009]           # == frac of the 30x30 box
    assert 900 / 1.0e6 == 0.0009
    assert E._row_size_bin(gt_list([485, 485, 515, 515]), edges) == "small"


# --------------------------------------------------------------------------- #
# property / cross-check tests
# --------------------------------------------------------------------------- #

def test_matches_independent_reimplementation_default_edges():
    rng = np.random.default_rng(1234)
    for _ in range(400):
        x1 = float(rng.integers(0, 900))
        y1 = float(rng.integers(0, 900))
        w = float(rng.integers(0, 1000 - int(x1) + 1))
        h = float(rng.integers(0, 1000 - int(y1) + 1))
        box = [x1, y1, x1 + w, y1 + h]
        got = E._row_size_bin(gt_json(box), DEFAULT_EDGES)
        assert got == expected_label(box, DEFAULT_EDGES), box


def test_matches_independent_reimplementation_random_edges():
    rng = np.random.default_rng(99)
    for _ in range(300):
        box = [0.0, 0.0, float(rng.integers(1, 1001)), float(rng.integers(1, 1001))]
        # 1..3 sorted edges keeps every bisect index within _SIZE_LABELS.
        k = int(rng.integers(1, 4))
        edges = sorted(float(e) for e in rng.uniform(1e-5, 0.2, size=k))
        got = E._row_size_bin(gt_list(box), edges)
        assert got == expected_label(box, edges), (box, edges)


def test_result_always_a_valid_label():
    rng = np.random.default_rng(2026)
    for _ in range(300):
        box = [0.0, 0.0, float(rng.integers(0, 1001)), float(rng.integers(0, 1001))]
        assert E._row_size_bin(gt_list(box), DEFAULT_EDGES) in E._SIZE_LABELS


def test_label_monotonic_non_decreasing_in_area():
    # Growing a square box (about the centre) can only move the label to a same-or-larger
    # bucket, never back down.
    order = {lab: i for i, lab in enumerate(E._SIZE_LABELS)}
    prev = -1
    for half in range(1, 400):
        box = [500 - half, 500 - half, 500 + half, 500 + half]
        idx = order[E._row_size_bin(gt_list(box), DEFAULT_EDGES)]
        assert idx >= prev
        prev = idx
    # A 800x800 box is unambiguously the largest bucket.
    assert prev == order["large"]
