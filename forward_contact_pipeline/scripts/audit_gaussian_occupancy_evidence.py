#!/usr/bin/env python3
"""Compare oracle local point distance with covariance-aware Gaussian evidence.

Uses the same PROX mesh/SDF relation assets for both scores.  This is a
representation diagnostic only: scene points are oracle mesh samples, so it
does not measure a forward reconstruction.  It asks whether treating each
local surface sample as an oriented anisotropic Gaussian yields a more useful
continuous proximity signal than nearest-point distance.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def rank(values):
    order=np.argsort(values,kind='mergesort'); out=np.empty(len(values),float); out[order]=np.arange(len(values)); return out

def spearman(a,b):
    a,b=rank(a),rank(b); return float(np.corrcoef(a,b)[0,1])

def auc(labels,scores):
    positives=int(labels.sum()); negatives=len(labels)-positives
    if positives==0 or negatives==0: return None
    # Mann-Whitney U using average rank is unnecessary here because scores are
    # effectively continuous; stable rank keeps the diagnostic deterministic.
    ranks=rank(scores)+1.0
    return float((ranks[labels].sum()-positives*(positives+1)/2)/(positives*negatives))

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--sigma-tangent-m',type=float,default=.03); p.add_argument('--sigma-normal-m',type=float,default=.005); a=p.parse_args()
    target=[]; point_score=[]; gaussian_score=[]; counts=[]
    for manifest_path in sorted(a.root.glob('*/manifest.json')):
        manifest=json.loads(manifest_path.read_text());
        with np.load(manifest_path.parent/manifest['payload'],allow_pickle=False) as d:
            points=np.asarray(d['scene_point_local'],np.float32); normals=np.asarray(d['scene_normal_local'],np.float32); proximity=np.asarray(d['vertex_proximity'],np.float32)
            valid=np.asarray(d['scene_point_valid'],np.float32) if 'scene_point_valid' in d else np.ones(points.shape[:-1],np.float32)
        length=np.linalg.norm(normals,axis=-1,keepdims=True); valid=valid*(length[...,0]>1e-6); normals=normals/np.maximum(length,1e-6)
        pn=np.sum(points*normals,axis=-1); pt2=np.maximum(np.sum(points*points,axis=-1)-pn*pn,0)
        energy=.5*((pn/a.sigma_normal_m)**2+(pt2/(a.sigma_tangent_m**2)))
        energy=np.where(valid>0,energy,np.inf)
        gaussian=-np.min(energy,axis=-1).reshape(-1)
        point=-np.min(np.where(valid>0,np.linalg.norm(points,axis=-1),np.inf),axis=-1).reshape(-1)
        target.append(-np.abs(proximity).reshape(-1)); point_score.append(point); gaussian_score.append(gaussian); counts.append({'sequence':manifest['sequence'],'vertices':len(point)})
    target=np.concatenate(target); point_score=np.concatenate(point_score); gaussian_score=np.concatenate(gaussian_score); labels=target>=-.02
    out={'kind':'oracle_gaussian_occupancy_representation_diagnostic','scope':'PROX oracle mesh/SDF relation assets; not a forward Gaussian/contact result','sigma_tangent_m':a.sigma_tangent_m,'sigma_normal_m':a.sigma_normal_m,'vertices':len(target),'contact_threshold_m':.02,'positive_fraction':float(labels.mean()),'nearest_point':{'spearman_with_negative_abs_sdf':spearman(point_score,target),'auc_contact':auc(labels,point_score)},'oriented_gaussian_occupancy':{'spearman_with_negative_abs_sdf':spearman(gaussian_score,target),'auc_contact':auc(labels,gaussian_score)},'sequences':counts,'verdict':'A positive Gaussian-vs-point gap supports using Gaussian covariance as learned local relation evidence, never as a hard contact threshold.'}
    a.output.mkdir(parents=True,exist_ok=False); (a.output/'report.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
