#!/usr/bin/env python3
"""Numerical diagnostic of target coverage; never changes training/evaluation rows."""
import argparse
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from contact_streaming.rich_assets import sha256,save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['rich-root','plan','features','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    plan=json.loads(a.plan.read_text());manifest=json.loads((a.features/'manifest.json').read_text())
    if manifest['provenance']['plan_sha256']!=sha256(a.plan):raise ValueError('Plan mismatch')
    cameras=json.loads((a.rich_root/'processed/dataset_v1/cameras.json').read_text())['cameras']
    cache={};rows=[]
    for ci,(clip,pred) in enumerate(zip(plan['clips'],manifest['clips'])):
        for fi in [0,len(clip['frames'])//2,len(clip['frames'])-1]:
            s=clip['frames'][fi];f=pred['frames'][fi];ann=s['annotation'];key=ann['path']
            if key not in cache:
                if sha256(a.rich_root/key)!=ann['sha256']:raise ValueError('GT changed')
                with np.load(a.rich_root/key,allow_pickle=False) as d:
                    cache[key]=(d['vertices_multicam'],d['frame_ids'])
            vertices,ids=cache[key]
            if ids[ann['row']]!=s['frame_id'] or f['sample_id']!=s['sample_id']:raise ValueError('Frame mismatch')
            if sha256(a.features/f['file'])!=f['sha256']:raise ValueError('Feature changed')
            e=np.asarray(cameras[s['camera_key']]['CameraMatrix']);k=np.asarray(cameras[s['camera_key']]['Intrinsics'])
            camera=vertices[ann['row']]@e[:,:3].T+e[:,3]
            projected=camera@k.T;uv=projected[:,:2]/projected[:,2:3]
            transform=np.asarray(f['transform']['raw_to_pad'])
            uv=np.c_[uv,np.ones(len(uv))]@transform.T
            lo=uv[:,:2].min(0);hi=uv[:,:2].max(0)
            with np.load(a.features/f['file'],allow_pickle=False) as d:
                loc=d['smpl_loc'].reshape(-1,2) if 'smpl_loc' in d else np.empty((0,2))
            inside=((loc>=lo)&(loc<=hi)).all(1)
            rows.append(dict(clip=ci,sequence=clip['sequence'],frame_id=s['frame_id'],
                predicted_humans=int(f['humans']),heads_inside_gt_body_bbox=int(inside.sum()),
                positive_gt_depth=bool((camera[:,2]>0).all()),
                gt_bbox_in_padded_image=[lo.tolist(),hi.tolist()]))
    counts=Counter()
    for r in rows:
        n=r['predicted_humans'];inside=r['heads_inside_gt_body_bbox']
        counts['audited_frames']+=1
        counts['single_frames' if n==1 else 'non_single_frames']+=1
        if n==1:counts['single_head_inside_bbox' if inside==1 else 'single_head_outside_bbox']+=1
        if n>1:counts['multi_frames']+=1;counts['multi_exactly_one_head_inside_bbox']+=inside==1
        if n==0:counts['no_detection_frames']+=1
    result=dict(status='diagnostic_only',counts=dict(counts),frames=rows,
        plan_sha256=sha256(a.plan),features_manifest_sha256=sha256(a.features/'manifest.json'),
        note='First/middle/last frame per declared clip. GT body bbox is audit-only. A head inside it is a plausible association, not verified identity; no primary samples/metrics or deployment inputs are changed.')
    save_json(a.output,result);print(json.dumps(result['counts']))


if __name__=='__main__':main()
