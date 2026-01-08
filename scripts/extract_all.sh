#!/usr/bin/env bash
set -euo pipefail

# Usage: ./extract_all.sh [DATA_DIR]
# Extracts all .tar files, combines .zip.part* files, and reorganizes into:
#   data/Training/image
#   data/Training/label
#   data/Validation/image
#   data/Validation/label

DATA_DIR=${1:-"$(pwd)/data"}

if [ ! -d "$DATA_DIR" ]; then
  echo "[ERROR] DATA_DIR not found: $DATA_DIR" >&2
  exit 1
fi

echo "[STEP 1/3] Extracting all .tar files..."
find "$DATA_DIR" -type f -name "*.tar" | while read tarfile; do
  target_dir="$(dirname "$tarfile")"
  echo "[INFO] tar: $tarfile"
  tar -xf "$tarfile" -C "$target_dir"
done

echo "[STEP 2/3] Combining and extracting .zip.part* files..."
declare -A combined
find "$DATA_DIR" -type f -name "*.zip.part*" -print0 | sort -z | while IFS= read -r -d '' partfile; do
  base="${partfile%\.part*}"
  if [[ -n "${combined[$base]:-}" ]]; then
    continue
  fi
  combined[$base]=1
  dir="$(dirname "$partfile")"
  
  # Only combine if .zip doesn't already exist
  if [ ! -f "$base" ]; then
    echo "[INFO] cat: combining $base.part* -> $base"
    cat "$base".part* > "$base"
  fi
  
  echo "[INFO] unzip: $base"
  unzip -o "$base" -d "$dir" > /dev/null 2>&1 || true
done

echo "[STEP 3/3] Organizing into Training/Validation image/label structure..."
SRC_ROOT="$DATA_DIR/086.주요질환_이미지_합성데이터(CT)/01-1.정식개방데이터"

DEST_TRAIN_IMG="$DATA_DIR/Training/image"
DEST_TRAIN_LABEL="$DATA_DIR/Training/label"
DEST_VAL_IMG="$DATA_DIR/Validation/image"
DEST_VAL_LABEL="$DATA_DIR/Validation/label"

mkdir -p "$DEST_TRAIN_IMG" "$DEST_TRAIN_LABEL" "$DEST_VAL_IMG" "$DEST_VAL_LABEL"

# Copy Training images (원천데이터 = source data)
if [ -d "$SRC_ROOT/Training/01.원천데이터" ]; then
  echo "[INFO] Copying Training images..."
  cp -nv "$SRC_ROOT/Training/01.원천데이터"/* "$DEST_TRAIN_IMG" 2>/dev/null || true
fi

# Copy Training labels (라벨링데이터 = labeling data)
if [ -d "$SRC_ROOT/Training/02.라벨링데이터" ]; then
  echo "[INFO] Copying Training labels..."
  cp -nv "$SRC_ROOT/Training/02.라벨링데이터"/* "$DEST_TRAIN_LABEL" 2>/dev/null || true
fi

# Copy Validation images
if [ -d "$SRC_ROOT/Validation/01.원천데이터" ]; then
  echo "[INFO] Copying Validation images..."
  cp -nv "$SRC_ROOT/Validation/01.원천데이터"/* "$DEST_VAL_IMG" 2>/dev/null || true
fi

# Copy Validation labels
if [ -d "$SRC_ROOT/Validation/02.라벨링데이터" ]; then
  echo "[INFO] Copying Validation labels..."
  cp -nv "$SRC_ROOT/Validation/02.라벨링데이터"/* "$DEST_VAL_LABEL" 2>/dev/null || true
fi

echo "[STEP 4/4] Cleaning up temporary files..."

# Remove zip and zip.part files from image/label folders
echo "[INFO] Removing .zip and .zip.part* files from image folders..."
rm -f "$DEST_TRAIN_IMG"/*.zip "$DEST_TRAIN_IMG"/*.zip.part* 2>/dev/null || true
rm -f "$DEST_VAL_IMG"/*.zip "$DEST_VAL_IMG"/*.zip.part* 2>/dev/null || true

echo "[INFO] Removing .zip and .zip.part* files from label folders..."
rm -f "$DEST_TRAIN_LABEL"/*.zip "$DEST_TRAIN_LABEL"/*.zip.part* 2>/dev/null || true
rm -f "$DEST_VAL_LABEL"/*.zip "$DEST_VAL_LABEL"/*.zip.part* 2>/dev/null || true

# Remove the original extracted directory structure
echo "[INFO] Removing original extracted directory: 086.주요질환..."
rm -rf "$DATA_DIR/086.주요질환_이미지_합성데이터(CT)" 2>/dev/null || true

echo "[DONE] Cleanup complete. Final structure:"
echo "  - $DEST_TRAIN_IMG"
echo "  - $DEST_TRAIN_LABEL"
echo "  - $DEST_VAL_IMG"
echo "  - $DEST_VAL_LABEL"
