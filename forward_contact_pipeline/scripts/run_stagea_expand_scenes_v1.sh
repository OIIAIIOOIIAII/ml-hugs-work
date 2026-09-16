#!/usr/bin/env bash
# Extend Stage-A clean-oracle assets to independent PROX scenes.  Safe to resume:
# it never overwrites teacher labels or relation directories.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
py_bin="/workspace/nas_auto_backup/nas/yuzilang/miniconda3/envs/human3r/bin/python"
labels_root="$project_root/forward_contact_pipeline/labels/prox_gt_sdf_stagea_v1"
relation_root="$project_root/forward_contact_pipeline/learning/stagea_relation_clean_64x128_v3"
mkdir -p "$labels_root" "$relation_root"

# One motion clip per previously unseen scene.  These add scene coverage only;
# error-family coverage is introduced in a separate degradation builder.
sequences=(
  "MPH11_00034_01"
  "MPH16_00157_01"
  "MPH1Library_00034_01"
  "MPH8_00168_01"
  "N0SittingBooth_00162_01"
  "N3Library_00157_01"
  "N3OpenArea_00157_01"
  "Werkraum_03301_01"
)

for sequence in "${sequences[@]}"; do
  teacher="$labels_root/${sequence}.teacher.npz"
  if [[ ! -e "$teacher" ]]; then
    echo "[$(date -Is)] labels_start sequence=$sequence"
    "$py_bin" "$project_root/forward_contact_pipeline/scripts/build_prox_geometry_labels.py" \
      --sequence "$sequence" --output "$labels_root/${sequence}.npz" --device cuda --max-frames 60
    echo "[$(date -Is)] labels_done sequence=$sequence"
  fi
  destination="$relation_root/$sequence"
  if [[ -e "$destination" ]]; then
    echo "[$(date -Is)] relation_skip_existing sequence=$sequence"
    continue
  fi
  echo "[$(date -Is)] relation_start sequence=$sequence"
  "$py_bin" "$project_root/forward_contact_pipeline/scripts/build_stagea_relation_manifest.py" \
    --teacher "$teacher" --sequence "$sequence" --output-dir "$destination" \
    --device cuda --roi-vertices 64 --scene-points 128
  echo "[$(date -Is)] relation_done sequence=$sequence"
done
