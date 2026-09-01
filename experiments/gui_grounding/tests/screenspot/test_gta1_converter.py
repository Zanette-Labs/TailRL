"""Unit tests for scripts/convert_gta1_to_easyr1.py (FAITHFUL converter).

Pure helpers (bbox-convention detection, pixel & normalized_1000 -> [0,1000] scaling,
swap-normalize, degenerate/out-of-frame drop, empty-only instruction validity) + an
end-to-end schema check on a SYNTHETIC in-memory parquet (streamed directly via pyarrow;
no download / no `datasets` install needed).

FAITHFUL to GTA1: NO dedup, NO instruction filtering (short/symbolic instructions are kept);
only genuinely empty instructions and invalid image/bbox rows are dropped.
"""
import json
import os

import convert_gta1_to_easyr1 as gta1


# --------------------------------------------------------------------------- #
# _detect_convention (guards the pixels-vs-[0,1000] mirror-swap bug)           #
# --------------------------------------------------------------------------- #
def test_detect_convention_pixels():
    assert gta1._detect_convention([[33, 75, 534, 132], [30, 961, 186, 2862]]) == "pixels"


def test_detect_convention_normalized_1000():
    assert gta1._detect_convention([[457, 152, 542, 173], [38, 166, 961, 218]]) == "normalized_1000"


# --------------------------------------------------------------------------- #
# _box_1000: pixels convention -> [0,1000]                                     #
# --------------------------------------------------------------------------- #
def test_box_pixels_scales_to_1000():
    # 33*1000/1920=17.2->17, 75*1000/1080=69.4->69, 534*1000/1920=278.1->278, 132*1000/1080=122.2->122
    assert gta1._box_1000([33, 75, 534, 132], 1920, 1080, "pixels") == [17, 69, 278, 122]


def test_box_pixels_full_image_is_full_canvas():
    assert gta1._box_1000([0, 0, 1920, 1080], 1920, 1080, "pixels") == [0, 0, 1000, 1000]


def test_box_pixels_zero_area_is_none():
    assert gta1._box_1000([10, 10, 10, 50], 1920, 1080, "pixels") is None
    assert gta1._box_1000([10, 10, 50, 10], 1920, 1080, "pixels") is None


def test_box_pixels_out_of_frame_dropped_not_clipped():
    assert gta1._box_1000([0, 0, 5000, 100], 1920, 1080, "pixels") is None
    assert gta1._box_1000([-50, 0, 100, 100], 1920, 1080, "pixels") is None


def test_box_pixels_swapped_coords_normalized():
    assert gta1._box_1000([534, 132, 33, 75], 1920, 1080, "pixels") == [17, 69, 278, 122]


def test_box_pixels_bad_dims_none():
    assert gta1._box_1000([0, 0, 100, 100], 0, 1080, "pixels") is None
    assert gta1._box_1000([0, 0, 100, 100], 1920, 0, "pixels") is None


def test_box_non_numeric_none():
    assert gta1._box_1000(["a", "b", "c", "d"], 1920, 1080, "pixels") is None
    assert gta1._box_1000(None, 1920, 1080, "pixels") is None


# --------------------------------------------------------------------------- #
# _box_1000: normalized_1000 convention -> use as-is, do NOT divide by W,H     #
# --------------------------------------------------------------------------- #
def test_box_normalized_used_as_is():
    assert gta1._box_1000([457, 152, 542, 173], 1920, 1080, "normalized_1000") == [457, 152, 542, 173]


def test_box_normalized_out_of_range_dropped():
    assert gta1._box_1000([0, 0, 1500, 100], 1920, 1080, "normalized_1000") is None


# --------------------------------------------------------------------------- #
# _validity_instruction: FAITHFUL empty-only guard (no min-length/no-alpha)    #
# --------------------------------------------------------------------------- #
def test_instruction_collapses_whitespace():
    assert gta1._validity_instruction("  click   the  OK  ") == "click the OK"


def test_instruction_drops_only_empty_and_none():
    assert gta1._validity_instruction("") is None
    assert gta1._validity_instruction("   ") is None
    assert gta1._validity_instruction(None) is None


def test_instruction_keeps_short_and_symbolic():
    assert gta1._validity_instruction("ok") == "ok"                # short KEPT (no min-length)
    assert gta1._validity_instruction("12345 !!!") == "12345 !!!"  # no-alpha KEPT
    assert gta1._validity_instruction("a") == "a"


# --------------------------------------------------------------------------- #
# end-to-end schema on a synthetic in-memory parquet                           #
# --------------------------------------------------------------------------- #
def _write_gta1_parquet(path, rows, W, H):
    """Write a synthetic parquet with the REAL GTA1 schema: uuid(str),
    image(struct<bytes,path>), bbox(list<int64>[4] abs px), instruction(str)."""
    from io import BytesIO

    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (W, H), (10, 20, 30)).save(buf, format="PNG")
    png = buf.getvalue()
    imgs = [{"bytes": png, "path": None} for _ in rows]
    tbl = pa.table({
        "uuid": pa.array([r["uuid"] for r in rows], pa.string()),
        "image": pa.array(imgs, pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        "bbox": pa.array([r["bbox"] for r in rows], pa.list_(pa.int64(), 4)),
        "instruction": pa.array([r["instruction"] for r in rows], pa.string()),
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(tbl, path)


def test_convert_schema_and_values(tmp_path):
    W, H = 1920, 1080
    rows = [
        {"instruction": "click the file menu", "bbox": [33, 75, 534, 132], "uuid": "a1"},
        {"instruction": "press the run button", "bbox": [100, 100, 300, 300], "uuid": "b2"},
        {"instruction": "click the file menu", "bbox": [33, 75, 534, 132], "uuid": "dup"},  # (instr,box) dup -> KEPT (no dedup)
        {"instruction": "!!", "bbox": [10, 10, 50, 50], "uuid": "d4"},                       # symbolic instr -> KEPT (no filtering)
        {"instruction": "", "bbox": [10, 10, 50, 50], "uuid": "e5"},                         # EMPTY instr -> dropped (validity)
        {"instruction": "bad box", "bbox": [0, 0, 9999, 50], "uuid": "c3"},                  # OOF bbox -> dropped (validity)
    ]
    raw_dir = os.path.join(str(tmp_path), "raw")
    _write_gta1_parquet(os.path.join(raw_dir, "data", "train-00000.parquet"), rows, W, H)

    image_out_dir = os.path.join(str(tmp_path), "gta1_images")
    out_file = os.path.join(str(tmp_path), "gta1_train.parquet")
    # expected_rows=0 + max_drop_frac=1.0 disable the faithfulness abort/warn for this tiny synthetic set.
    counts = gta1.convert(raw_dir=raw_dir, image_out_dir=image_out_dir, out_file=out_file,
                          expected_rows=0, max_drop_frac=1.0)

    assert counts["kept"] == 4                     # a1, b2, dup, d4 (NO dedup, NO instruction filtering)
    assert "drop_dup" not in counts                # dedup removed for GTA1 fidelity
    assert counts["drop_bad_bbox"] == 1            # c3 (out of frame)
    assert counts["drop_empty_instruction"] == 1   # e5 (empty)

    import pandas as pd
    df = pd.read_parquet(out_file)
    assert list(df.columns) == ["images", "problem", "answer"]
    assert len(df) == 4

    first = df.iloc[0]
    assert first["problem"].startswith("<image>")
    assert list(first["images"]) == ["gta1_images/a1.jpg"]
    assert os.path.exists(os.path.join(image_out_dir, "a1.jpg"))

    ans = json.loads(first["answer"])
    assert ans["bbox"] == [17, 69, 278, 122]
    assert ans["wh"] == [W, H]
    assert all(0 <= v <= 1000 for v in ans["bbox"])
