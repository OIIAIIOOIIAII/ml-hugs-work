#!/usr/bin/env bash
set -eo pipefail

source /home/u202420081000003/anaconda3/etc/profile.d/conda.sh
conda activate human3r

cd /hdd/u202420081000003/Human3R

CUDA_VISIBLE_DEVICES=0 python demo.py \
  --model_path src/human3r_896L.pth \
  --size 1024 \
  --seq_path /hdd/u202420081000003/ml-hugs/data/neuman/dataset/lab/images \
  --output_dir output/lab_size1024_full_20260518 \
  --subsample 1 \
  --use_ttt3r \
  --vis_threshold 2 \
  --downsample_factor 1 \
  --reset_interval 100 \
  --save \
  --no_viewer
