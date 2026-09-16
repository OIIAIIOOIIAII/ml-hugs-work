#!/usr/bin/env bash
# Build the Stage-A clean-oracle relation contract without overwriting assets.
# This is a preparation run, not a real-front-end training experiment.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
py_bin="/workspace/nas_auto_backup/nas/yuzilang/miniconda3/envs/human3r/bin/python"
out_root="$project_root/forward_contact_pipeline/learning/stagea_relation_clean_64x128_v4"
log_root="$project_root/forward_contact_pipeline/experiments/stagea_relation_build_v4_20260909"
mkdir -p "$out_root" "$log_root"

sequences=(
  "BasementSittingBooth_00142_01"
  "MPH112_00034_01"
  "N0Sofa_00034_01"
  "N3Office_00034_01"
)

for sequence in "${sequences[@]}"; do
  teacher="$project_root/forward_contact_pipeline/labels/prox_gt_sdf_b/${sequence}.teacher.npz"
  destination="$out_root/$sequence"
  if [[ -e "$destination" ]]; then
    echo "[$(date -Is)] skip_existing sequence=$sequence destination=$destination"
    continue
  fi
  echo "[$(date -Is)] start sequence=$sequence"
  "$py_bin" "$project_root/forward_contact_pipeline/scripts/build_stagea_relation_manifest.py" \
    --teacher "$teacher" --sequence "$sequence" --output-dir "$destination" \
    --device cuda --roi-vertices 64 --scene-points 128
  echo "[$(date -Is)] done sequence=$sequence"
done

"$py_bin" - "$out_root" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
records = []
for manifest_path in sorted(root.glob('*/manifest.json')):
    manifest = json.loads(manifest_path.read_text())
    payload = np.load(manifest_path.parent / manifest['payload'], allow_pickle=False)
    records.append({
        'sequence': manifest['sequence'], 'scene': manifest['scene'], 'frames': manifest['frames'],
        'roi_shape': list(payload['roi_vertex_local'].shape),
        'patch_shape': list(payload['scene_point_local'].shape),
        'contact_rate': float(payload['vertex_contact'].mean()),
        'finite': bool(all(np.isfinite(payload[key]).all() for key in payload.files)),
    })
summary = {'schema': 'hugs.forward_contact.stage_a_relation.v1', 'kind': 'clean_oracle_data_contract', 'records': records}
(root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
PY
