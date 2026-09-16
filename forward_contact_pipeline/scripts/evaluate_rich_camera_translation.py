#!/usr/bin/env python3
"""Evaluate a fixed sensor-calibrated translation hypothesis without training."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from contact_streaming.camera_translation import reproject_translation
from contact_streaming.rich_assets import sha256,save_json
from contact_streaming.rich_supervision import pixel_transform
from evaluate_rich_geometry_experiments import metric


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['rich-root','cache','plan','output']:
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    manifest=json.loads((a.cache/'manifest.json').read_text())
    if manifest['data_sha256']!=sha256(a.cache/'data.npz'):raise ValueError('Cache changed')
    with np.load(a.cache/'data.npz',allow_pickle=False) as d: data={k:d[k] for k in d.files}
    plan=json.loads(a.plan.read_text());lookup={s['sample_id']:s for c in plan['clips'] for s in c['frames']}
    cameras=json.loads((a.rich_root/'processed/dataset_v1/cameras.json').read_text())['cameras']
    ids=np.sort(np.random.default_rng(42).choice(10475,512,replace=False))
    roi=np.array(json.loads((a.rich_root/'processed/supervision_v2/roi.json').read_text())['vertex_ids'])
    corrected=[];trans=[]
    for sample_id,v in zip(data['sample_id'],data['base']):
        sample=lookup[sample_id];t=pixel_transform(*sample['raw_size_wh'],512,896)
        # Multi-HMR initializes in a padded 896 square with released 60deg FOV.
        focal=896/(2*np.tan(np.pi/6));kmhmr=np.array([[focal,0,448],[0,focal,448],[0,0,1.]])
        virtual=np.linalg.inv(t['crop_to_pad'])@kmhmr
        sensor=np.asarray(t['raw_to_crop'])@np.asarray(cameras[sample['camera_key']]['Intrinsics'])
        delta=reproject_translation(v[ids],virtual,sensor)
        corrected.append(v+delta);trans.append(delta)
    corrected=np.array(corrected);rows=[]
    for ci in sorted(set(data['clip'].tolist())):
        index=np.flatnonzero(data['clip']==ci)
        rows.append(dict(sequence=str(data['sequence'][index[0]]),baseline=metric(data['base'][index],data['target'][index],roi),
            calibrated=metric(corrected[index],data['target'][index],roi),
            median_translation_m=np.median(np.array(trans)[index],axis=0).tolist()))
    keys=['raw_body_median_m','raw_sole_median_m','body_median_m','sole_median_m']
    summary={kind:{k:float(np.median([r[kind][k] for r in rows])) for k in keys} for kind in ['baseline','calibrated']}
    save_json(a.output,dict(kind='sensor_K_translation_hypothesis',body_or_contact_GT_input=False,sensor_K_required=True,
        clips=rows,summary=summary,plan_sha256=sha256(a.plan),cache_sha256=manifest['data_sha256'],
        training_ready=False,scope='Only solve translation against model-predicted pixels; no actual 2D observation/pose repair'))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
