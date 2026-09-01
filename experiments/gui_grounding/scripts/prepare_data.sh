#!/usr/bin/env bash
# Download both datasets and convert them to the parquet layout the trainer reads.
#
#   bash scripts/prepare_data.sh
#
# Disk: ~38 GB of raw download, ~16 GB of converted JPEGs, ~5 MB of parquet. Export GUI_RAW_DIR /
# GUI_IMAGE_DIR / GUI_DATA_DIR to scratch storage BEFORE running, or all of it lands in the
# checkout. Both datasets are public; no HF token is required.
#
# Re-running is cheap: `hf download` is content-addressed and skips what it already has, and each
# conversion step is skipped if its parquet already exists (delete the parquet to force a rebuild).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
cd "$TAILRL_GUI_ROOT"

mkdir -p "$GUI_RAW_DIR" "$GUI_IMAGE_DIR" "$GUI_DATA_DIR"

hf_dl() {  # $1 = repo id, $2 = dest
  if command -v hf >/dev/null 2>&1; then
    hf download "$1" --repo-type dataset --local-dir "$2"
  else                                  # older CLI spelling
    huggingface-cli download "$1" --repo-type dataset --local-dir "$2"
  fi
}

# --- 1. GTA1 training corpus -------------------------------------------------------------------
# Salesforce/grounding_dataset: 75 parquet shards, 70,688 rows, ~34.7 GB.
if [ ! -f "$GUI_DATA_DIR/gta1_train.parquet" ]; then
  echo "==> downloading GTA1 (Salesforce/grounding_dataset, ~34.7 GB)"
  hf_dl Salesforce/grounding_dataset "$GUI_RAW_DIR/gta1_raw"
  echo "==> converting GTA1 -> parquet + JPEGs"
  # Streams the shards with pyarrow (no 37 GB Arrow cache), auto-detects the pixel-vs-[0,1000] bbox
  # convention, and writes JPEG q90. EXPECT "kept 70528" and a benign warning that 70,528 != the
  # nominal 70,688: 160 rows have unusable boxes and are dropped. That number is load-bearing --
  # 70,528 / 8 = 8,816 steps/epoch, and 3 epochs = the 26,448 steps the released runs trained for.
  "${RUN[@]}" python3 scripts/convert_gta1_to_easyr1.py \
      --raw_dir       "$GUI_RAW_DIR/gta1_raw" \
      --image_out_dir "$GUI_IMAGE_DIR/gta1_images" \
      --out_file      "$GUI_DATA_DIR/gta1_train.parquet"
  echo "    the raw GTA1 download can now be deleted: rm -rf $GUI_RAW_DIR/gta1_raw"
else
  echo "==> GTA1 parquet exists, skipping"
fi

# --- 2. ScreenSpot-Pro evaluation set -----------------------------------------------------------
# likaixin/ScreenSpot-Pro: 1,581 items, ~3.2 GB. Images are SYMLINKED, not copied, so this raw
# directory must be kept.
if [ ! -f "$GUI_DATA_DIR/screenspot_pro.parquet" ]; then
  echo "==> downloading ScreenSpot-Pro (likaixin/ScreenSpot-Pro, ~3.2 GB)"
  hf_dl likaixin/ScreenSpot-Pro "$GUI_RAW_DIR/ss_pro_raw"
  echo "==> converting ScreenSpot-Pro -> parquet"
  "${RUN[@]}" python3 scripts/convert_screenspot_to_easyr1.py \
      --dataset    pro \
      --src        "$GUI_RAW_DIR/ss_pro_raw" \
      --image_root "$GUI_IMAGE_DIR" \
      --out_file   "$GUI_DATA_DIR/screenspot_pro.parquet"
else
  echo "==> ScreenSpot-Pro parquet exists, skipping"
fi

echo
echo "done."
echo "  train : $GUI_DATA_DIR/gta1_train.parquet     (expect 70,528 rows)"
echo "  eval  : $GUI_DATA_DIR/screenspot_pro.parquet (expect 1,581 rows)"
echo "  images: $GUI_IMAGE_DIR/{gta1_images,screenspot_pro}"
