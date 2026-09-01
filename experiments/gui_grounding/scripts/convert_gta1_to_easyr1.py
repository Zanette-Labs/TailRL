#!/usr/bin/env python3
"""GTA1 (Salesforce/grounding_dataset) -> EasyR1 GUI-grounding parquet.

Verified schema (HF Salesforce/grounding_dataset, ONE element/row):
  image        : Image (embedded bytes)
  bbox         : list[int] length 4, ABSOLUTE PIXEL [x1,y1,x2,y2]
  instruction  : str
  (+ description/function/combine/org_caption/uuid; only uuid used, for the filename)

Output (byte-identical to the reward/filter/eval expectations, same as convert_click100k):
  images  : list[str]  -- [ "<subdir>/<uuid>.png" ], resolved under data.image_dir
  problem : str        -- "<image>" + instruction
  answer  : str (JSON) -- {"bbox":[x1,y1,x2,y2] in [0,1000], "wh":[origW,origH]}

Usage:
  python3 scripts/convert_gta1_to_easyr1.py \
      --raw_dir $WORK_HDD/raw/gta1_raw --image_out_dir $IMG_ROOT/gta1_images \
      --out_file data/gui/gta1_train.parquet

FAITHFUL to GTA1: no dedup, no instruction filtering (drop only genuinely EMPTY instructions and
invalid image/bbox rows); bbox convention auto-detected; aborts if it drops >1% of rows.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

IMAGE_TOKEN = "<image>"


def _validity_instruction(instr):
    """FAITHFUL: GTA1 does NOT filter instructions (its only curation is an upstream OmniParser
    IoU>=0.1 bbox-alignment clean). Guard ONLY against a genuinely EMPTY string -- no min-length,
    no alpha requirement -- so short / symbolic instructions are KEPT."""
    if instr is None:
        return None
    s = " ".join(str(instr).split())
    return s if s else None


def _detect_convention(sample_bboxes):
    """Guard the mirror-swap / double-normalization bug. Salesforce/grounding_dataset stores bboxes
    in PIXELS (real screenshots are >1000px wide, so pixel coords routinely exceed 1000); GTA1's own
    HelloKKMe/grounding_dataset stores them in [0,1000]. Pixels iff any probed coordinate > 1000."""
    mx = 0.0
    for bb in sample_bboxes:
        try:
            mx = max(mx, max(float(v) for v in bb[:4]))
        except Exception:
            continue
    return "pixels" if mx > 1000.0 else "normalized_1000"


def _box_1000(bbox, W, H, convention):
    """[x1,y1,x2,y2] -> [0,1000] ints; None if degenerate/out-of-frame. `convention` is 'pixels'
    (normalize by W,H) or 'normalized_1000' (already [0,1000] -- use as-is, do NOT divide)."""
    try:
        x1, y1, x2, y2 = [float(t) for t in bbox][:4]
    except Exception:
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if convention == "pixels":
        if W <= 0 or H <= 0:
            return None
        if x1 < -1 or y1 < -1 or x2 > W + 1 or y2 > H + 1:   # wildly out of frame -> drop (not clip)
            return None
        b = [round(x1 * 1000.0 / W), round(y1 * 1000.0 / H),
             round(x2 * 1000.0 / W), round(y2 * 1000.0 / H)]
    else:  # normalized_1000 -- already [0,1000]; do NOT divide by W,H
        if x1 < -1 or y1 < -1 or x2 > 1001 or y2 > 1001:
            return None
        b = [round(x1), round(y1), round(x2), round(y2)]
    b = [max(0, min(1000, v)) for v in b]
    if b[2] <= b[0] or b[3] <= b[1]:                     # degenerate after rounding
        return None
    return b


def _probe_convention(raw_dir, n=200):
    """Read ONLY the bbox column of the first shard (cheap, columnar) to auto-detect the convention."""
    shards = sorted(glob.glob(os.path.join(raw_dir, "**", "*.parquet"), recursive=True))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {raw_dir}")
    bboxes = pq.read_table(shards[0], columns=["bbox"]).column("bbox").to_pylist()[:n]
    return _detect_convention([b for b in bboxes if b is not None])


def _iter_gta1_rows(raw_dir):
    """Stream rows from the downloaded GTA1 parquet shards DIRECTLY (no datasets.load_dataset, so
    no split-size verification and no ~37 GB Arrow cache -- and it works on a partial download).
    Yields {uuid, image(struct{bytes,path}), bbox(list[4] abs px), instruction}."""
    shards = sorted(glob.glob(os.path.join(raw_dir, "**", "*.parquet"), recursive=True))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {raw_dir}")
    for shard in shards:
        table = pq.read_table(shard, columns=["uuid", "image", "bbox", "instruction"])
        for batch in table.to_batches(max_chunksize=2000):
            d = batch.to_pydict()
            for i in range(len(d["instruction"])):
                yield {"uuid": d["uuid"][i], "image": d["image"][i],
                       "bbox": d["bbox"][i], "instruction": d["instruction"][i]}


def convert(raw_dir, image_out_dir, out_file, expected_rows=70688, max_drop_frac=0.01, limit=None):
    from io import BytesIO

    from PIL import Image
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_file)) or ".", exist_ok=True)
    subdir = os.path.basename(os.path.normpath(image_out_dir))
    convention = _probe_convention(raw_dir)
    print("[convert] detected bbox convention:", convention,
          "(pixels -> /W,H ; normalized_1000 -> as-is)")
    counts = defaultdict(int)
    problems, image_lists, answers = [], [], []
    writer = None

    def flush():
        nonlocal writer, problems, image_lists, answers
        if not problems:
            return
        tbl = pa.table({
            "images": pa.array(image_lists, type=pa.list_(pa.string())),
            "problem": pa.array(problems, type=pa.string()),
            "answer": pa.array(answers, type=pa.string()),
        })
        if writer is None:
            writer = pq.ParquetWriter(out_file, tbl.schema)
        writer.write_table(tbl)
        problems, image_lists, answers = [], [], []

    try:
        for i, row in enumerate(_iter_gta1_rows(raw_dir)):
            if limit is not None and counts["rows_in"] >= limit:
                break
            counts["rows_in"] += 1
            instr = _validity_instruction(row.get("instruction"))
            if instr is None:
                counts["drop_empty_instruction"] += 1
                continue
            img_field = row.get("image")
            data = img_field.get("bytes") if isinstance(img_field, dict) else img_field
            if not data:
                counts["drop_missing_image"] += 1
                continue
            try:
                img = Image.open(BytesIO(data))
                W, H = img.size                            # PIL image -> (W, H)
            except Exception:
                counts["drop_bad_image"] += 1
                continue
            box = _box_1000(row.get("bbox"), W, H, convention)
            if box is None:
                counts["drop_bad_bbox"] += 1
                continue
            # NO (instruction, box) dedup: two different screenshots that share an instruction and a
            # normalized box are two distinct supervision signals (GTA1 keeps both), and dedup ignoring
            # the image would collapse exactly the information grounding is about. Keep every VALID row
            # (bad-instruction / bad-bbox / bad-image drops above are the floor of "usable row", not
            # curation) -> gta1_train.parquet is the literal OmniParser-cleaned corpus.
            uid = str(row.get("uuid") or ("gta1_" + str(i)))
            # JPEG q90 (not PNG): UI screenshots are visually lossless at q90 for grounding, and only
            # pixel VALUES change (dims/coordinate-frame are preserved) -- but lossless PNG re-encoding
            # of decoded images is ~1.5 MB each (~108 GB for 70k), vs ~200 KB each (~14 GB) for JPEG.
            rel = os.path.join(subdir, uid + ".jpg")
            abspath = os.path.join(image_out_dir, uid + ".jpg")
            if not os.path.exists(abspath):
                img.convert("RGB").save(abspath, "JPEG", quality=90)
            problems.append(IMAGE_TOKEN + instr)
            image_lists.append([rel])
            answers.append(json.dumps({"bbox": box, "wh": [int(W), int(H)]}))
            counts["kept"] += 1
            if len(problems) >= 20000:
                flush()
    finally:
        flush()
        if writer is not None:
            writer.close()

    print("=== GTA1 conversion summary ===")
    for k in sorted(counts):
        print("  {:28s} {}".format(k, counts[k]))
    kept, rin = counts.get("kept", 0), counts.get("rows_in", 0)
    if kept:
        print("wrote {} rows -> {}".format(kept, out_file))
    # ---- faithfulness sanity: an already-cleaned corpus should lose ~nothing ----
    if limit is None:
        drop_frac = (rin - kept) / rin if rin else 1.0
        if drop_frac > max_drop_frac:
            raise SystemExit(
                "[FAITHFULNESS] dropped {:.2%} of rows (> {:.0%}). The released corpus is already "
                "clean, so large drops mean a converter bug (e.g. wrong bbox convention), not dirty "
                "data. Inspect the drop_* counts above.".format(drop_frac, max_drop_frac))
        if expected_rows and abs(kept - expected_rows) > max(50, int(max_drop_frac * expected_rows)):
            print("[warn] kept {} != expected ~{} (ok if a different mirror/version)."
                  .format(kept, expected_rows))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="local dir from `huggingface-cli download Salesforce/grounding_dataset`")
    ap.add_argument("--image_out_dir", required=True, help="e.g. $IMG_ROOT/gta1_images")
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--expected_rows", type=int, default=70688)
    ap.add_argument("--max_drop_frac", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    convert(a.raw_dir, a.image_out_dir, a.out_file, a.expected_rows, a.max_drop_frac, a.limit)


if __name__ == "__main__":
    main()
