#!/usr/bin/env bash
set -eo pipefail

source /home/u202420081000003/anaconda3/etc/profile.d/conda.sh
conda activate hugs

cd /hdd/u202420081000003/ml-hugs

CUDA_VISIBLE_DEVICES=0 python scripts/run_neuman_human_scene_noamass.py \
  --seq lab_size1024_full_sammask_lcc \
  --cfg-file cfg_files/release/human3r/hugs_human_scene.yaml \
  --quick \
  --quick-steps 1000 \
  --exp-name human3r_1024_full_sammask_lcc_1000_nohdens \
  dataset_path=data/human3r_hugs/lab_size1024_full_sammask_lcc_fit_smpl \
  human.canon_nframes=103 \
  train.save_progress_images=true \
  train.progress_save_interval=250 \
  human.densify_from_iter=3000 \
  human.densify_until_iter=3000 \
  scene.densify_from_iter=100 \
  scene.densify_until_iter=1000 \
  scene.densification_interval=100
