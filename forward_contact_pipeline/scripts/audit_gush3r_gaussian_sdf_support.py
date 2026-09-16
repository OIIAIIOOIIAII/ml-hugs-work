#!/usr/bin/env python3
"""Audit whether exported GUSH3R Gaussian centers support PROX SDF surfaces.

Centers are evaluated only as a diagnostic: a Gaussian has volume, so neither
low SDF distance nor penetration is a contact label.  The goal is to decide
whether a per-frame Gaussian map may provide local evidence/reliability.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from contact_streaming.alignment import Sim3
from build_prox_geometry_labels import SDFGrid, scene_name

def summary(x):
    x=np.asarray(x,float); return {'mean_m':float(x.mean()),'median_m':float(np.median(x)),'p95_m':float(np.percentile(x,95)),'max_m':float(x.max())}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--evidence-dir',type=Path,required=True); p.add_argument('--alignment-report',type=Path,required=True); p.add_argument('--prox-root',type=Path,required=True); p.add_argument('--sequence',required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    evidence=json.loads((a.evidence_dir/'manifest.json').read_text()); alignment=json.loads(a.alignment_report.read_text()); sim=alignment['sim3']; transform=Sim3(float(sim['scale']),np.asarray(sim['rotation'],np.float32),np.asarray(sim['translation'],np.float32),float(sim['residual_m']),float(sim['confidence']),'gush_train_frame_sim3')
    held=set(alignment['heldout_frames']); sdf=SDFGrid(a.prox_root/'sdf'/f'{scene_name(a.sequence)}_sdf.npy',a.prox_root/'sdf'/f'{scene_name(a.sequence)}.json')
    values=[]; support=[]; records=[]
    for record in evidence['frames']:
        frame=record['frame']
        if frame not in held: continue
        data=np.load(a.evidence_dir/record['file'],allow_pickle=False)
        if 'scene_xyz' not in data: continue
        distance, valid=sdf.sample(transform.transform(data['scene_xyz']))
        distance=distance[valid]; values.append(distance)
        records.append({'frame':frame,'sampled_scene_gaussians':int(len(data['scene_xyz'])),'sdf_coverage':float(valid.mean()),'abs_sdf_median_m':float(np.median(np.abs(distance))) if len(distance) else None,'near_surface_2cm_fraction':float((np.abs(distance)<=.02).mean()) if len(distance) else None,'penetrating_5mm_fraction':float((distance<-.005).mean()) if len(distance) else None})
    if not values: raise RuntimeError('No valid held-out scene Gaussian/SDF samples')
    values=np.concatenate(values); out={'kind':'gush3r_gaussian_center_sdf_support_diagnostic','scope':'held-out Gaussian center diagnostic only; centers are not surfaces/contact labels','sequence':a.sequence,'heldout_frames':sorted(held),'alignment_source':'GUSH3R human-SMPL Sim(3) fitted on earlier frames; known to fail contact gate','sdf_coverage':float(np.mean([x['sdf_coverage'] for x in records])),'abs_sdf':summary(np.abs(values)),'near_surface_2cm_fraction':float((np.abs(values)<=.02).mean()),'penetrating_5mm_fraction':float((values<-.005).mean()),'frames':records,'verdict':'Use only as a degraded Gaussian evidence/reliability diagnostic. Do not threshold centers to infer contact or treat them as a surface.'}
    a.output.mkdir(parents=True,exist_ok=False); (a.output/'report.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
