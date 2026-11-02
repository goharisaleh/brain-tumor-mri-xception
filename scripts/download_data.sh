#!/usr/bin/env bash
# Download the Kaggle dataset into data/raw
# Needs a Kaggle API token in ~/.kaggle/kaggle.json (kaggle.com -> Settings -> API -> Create New Token)
set -euo pipefail
DATASET="omarradi1993/mri-scans-brain-neurological-classes"
OUT="data/raw"

if ! command -v kaggle >/dev/null 2>&1; then
  pip install -q kaggle
fi
if [ ! -f "$HOME/.kaggle/kaggle.json" ] && [ -z "${KAGGLE_USERNAME:-}" ]; then
  echo "ERROR: Kaggle token not found. Put kaggle.json in ~/.kaggle/ first." >&2
  exit 1
fi
chmod 600 "$HOME/.kaggle/kaggle.json" 2>/dev/null || true

mkdir -p "$OUT"
kaggle datasets download -d "$DATASET" -p "$OUT" --unzip
echo "Done. Images are in $OUT"
python -m src.data --force    # scan the folders and create the fixed train/val/test split
