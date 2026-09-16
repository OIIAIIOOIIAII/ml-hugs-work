#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../miniconda3/etc/profile.d/conda.sh"
conda activate human3r

HUMAN3R_DIR=""  # TODO: set path to Human3R repo on this server
cd "$HUMAN3R_DIR"

CUDA_VISIBLE_DEVICES=0 python demo.py \
  --model_path src/human3r_896L.pth \
  --size 1024 \
  --seq_path "$SCRIPT_DIR/../data/neuman/dataset/lab/images" \
  --output_dir output/lab_size1024_full_20260518 \
  --subsample 1 \
  --use_ttt3r \
  --vis_threshold 2 \
  --downsample_factor 1 \
  --reset_interval 100 \
  --save \
  --no_viewer
