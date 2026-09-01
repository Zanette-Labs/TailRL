#!/usr/bin/env python3
"""Convert ScreenSpot-Pro (and ScreenSpot-v2) -> EasyR1 parquet (GUI grounding eval set).

Output schema (RefCOCO-compatible + a `category` column the eval aggregates over):
  images   : list[str]  -- ["<subdir>/<img_filename>"], resolved under --image_root (a stable
             symlink <image_root>/<subdir> -> dataset images dir is created so paths don't carry
             the HF-cache hash). Eval does Image.open(os.path.join(image_dir, images[0])).
  problem  : str  -- "<image>" + instruction.
  answer   : str  -- JSON "[x1,y1,x2,y2]" of the GT box in NORMALIZED [0,1000] space
             (bbox_pixels / [W,H,W,H] * 1000, rounded, clipped; degenerate boxes dropped).
  category : str  -- ScreenSpot-Pro `group` (industry, e.g. "CAD", "Dev") for per-category eval.
  ui_type  : str  -- "text" / "icon" (extra split).

ScreenSpot-Pro source: annotations/*.json (list of records with img_filename, bbox[abs xyxy px],
instruction, img_size[W,H], group, ui_type, application, platform); images/<app>/<shot>.png.

Usage (inside the Apptainer container):
  python3 scripts/convert_screenspot_to_easyr1.py --dataset pro \
      --src <ScreenSpot-Pro snapshot dir> \
      --image_root ${GUI_IMAGE_DIR:-$PWD/gui_images} \
      --out_file data/gui/screenspot_pro.parquet
"""
import argparse
import glob
import json
import os

import pandas as pd

IMAGE_TOKEN = "<image>"


def _box_1000(bbox, W, H):
    """abs-pixel xyxy -> integer [0,1000] xyxy, clipped. None if unusable/degenerate."""
    if bbox is None or len(bbox) < 4 or not W or not H:
        return None
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    box = [int(round(x1 / W * 1000.0)), int(round(y1 / H * 1000.0)),
           int(round(x2 / W * 1000.0)), int(round(y2 / H * 1000.0))]
    box = [min(1000, max(0, c)) for c in box]
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _ensure_symlink(target_dir, link_path):
    if os.path.islink(link_path) or os.path.exists(link_path):
        return
    os.makedirs(os.path.dirname(link_path), exist_ok=True)
    os.symlink(os.path.abspath(target_dir), link_path)
    print(f"symlink {link_path} -> {target_dir}")


def convert_pro(src, image_root, subdir, out_file):
    ann_files = sorted(glob.glob(os.path.join(src, "annotations", "*.json")))
    images_dir = os.path.join(src, "images")
    if not ann_files:
        raise FileNotFoundError(f"no annotations under {src}/annotations")
    _ensure_symlink(images_dir, os.path.join(image_root, subdir))

    rows, skipped = [], 0
    for af in ann_files:
        for rec in json.load(open(af)):
            W, H = (rec.get("img_size") or [None, None])[:2]
            box = _box_1000(rec.get("bbox"), W, H)
            instr = (rec.get("instruction") or "").strip()
            if box is None or not instr:
                skipped += 1
                continue
            rows.append({
                "images": [f"{subdir}/{rec['img_filename']}"],
                "problem": IMAGE_TOKEN + instr,
                "answer": json.dumps({"bbox": box, "wh": [int(W), int(H)]}),
                "category": rec.get("group", "unknown"),
                "ui_type": rec.get("ui_type", "unknown"),
            })
    _write(rows, out_file, skipped)


def convert_v2(src, image_root, subdir, out_file):
    """ScreenSpot-v2 (os-atlas/ScreenSpot-v2): JSON annotation file(s) + images dir.
    Records carry img_filename, bbox, instruction, data_type, data_source. bbox is [x,y,w,h]
    in absolute pixels; image size is read from the file. (Schema confirmed at convert time.)"""
    from PIL import Image
    ann_files = sorted(glob.glob(os.path.join(src, "**", "*.json"), recursive=True))
    # locate the dir that actually holds the .png images (OS-Copilot/ScreenSpot-v2 unzips to
    # 'screenspotv2_image/'); pick the dir with the most PNGs.
    png_dirs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(src, "**", "*.png"), recursive=True)})
    images_dir = max(png_dirs, key=lambda d: len(glob.glob(os.path.join(d, "*.png")))) if png_dirs else src
    _ensure_symlink(images_dir, os.path.join(image_root, subdir))

    rows, skipped = [], 0
    for af in ann_files:
        try:
            data = json.load(open(af))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for rec in data:
            if not isinstance(rec, dict) or "bbox" not in rec:
                continue
            fn = rec.get("img_filename") or rec.get("image")
            instr = (rec.get("instruction") or "").strip()
            bbox = rec.get("bbox")
            if fn is None or not instr or bbox is None:
                skipped += 1
                continue
            try:
                with Image.open(os.path.join(images_dir, fn)) as im:
                    W, H = im.size
            except Exception:
                skipped += 1
                continue
            # ScreenSpot-v2 bbox is [x, y, w, h] -> xyxy
            xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            box = _box_1000(xyxy, W, H)
            if box is None:
                skipped += 1
                continue
            rows.append({
                "images": [f"{subdir}/{fn}"],
                "problem": IMAGE_TOKEN + instr,
                "answer": json.dumps({"bbox": box, "wh": [int(W), int(H)]}),
                "category": rec.get("data_source", "unknown"),
                "ui_type": rec.get("data_type", "unknown"),
            })
    _write(rows, out_file, skipped)


def _write(rows, out_file, skipped):
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out_file, index=False)
    print(f"wrote {len(df)} rows -> {out_file}  (skipped {skipped})")
    if len(df):
        print("categories:", df["category"].value_counts().to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["pro", "v2"], required=True)
    ap.add_argument("--src", required=True, help="dataset snapshot dir")
    ap.add_argument("--image_root",
                    default=os.environ.get("GUI_IMAGE_DIR", os.path.join(os.getcwd(), "gui_images")))
    ap.add_argument("--subdir", default=None, help="symlink subdir under image_root (default: screenspot_<dataset>)")
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()
    subdir = args.subdir or f"screenspot_{args.dataset}"
    if args.dataset == "pro":
        convert_pro(args.src, args.image_root, subdir, args.out_file)
    else:
        convert_v2(args.src, args.image_root, subdir, args.out_file)


if __name__ == "__main__":
    main()
