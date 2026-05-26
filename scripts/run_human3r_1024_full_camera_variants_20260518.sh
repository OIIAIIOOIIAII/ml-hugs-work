#!/usr/bin/env bash
set -eo pipefail

source /home/u202420081000003/anaconda3/etc/profile.d/conda.sh
conda activate human3r

cd /hdd/u202420081000003/Human3R

COMMON_ARGS=(
  --model_path src/human3r_896L.pth
  --size 1024
  --seq_path /hdd/u202420081000003/ml-hugs/data/neuman/dataset/lab/images
  --subsample 1
  --vis_threshold 2
  --downsample_factor 1
  --save
  --no_viewer
)

CUDA_VISIBLE_DEVICES=0 python demo.py \
  "${COMMON_ARGS[@]}" \
  --output_dir output/lab_size1024_full_ttt_noreset_20260518 \
  --use_ttt3r \
  --reset_interval 1000

CUDA_VISIBLE_DEVICES=0 python demo.py \
  "${COMMON_ARGS[@]}" \
  --output_dir output/lab_size1024_full_nottt_noreset_20260518 \
  --reset_interval 1000
