#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree


def read_colored(path, max_scene=500000, max_human=200000, seed=0):
    ply=PlyData.read(str(path), mmap=True); v=ply['vertex'].data
    xyz=np.stack([v['x'],v['y'],v['z']],1).astype(np.float32)
    part=v['part'].astype(np.uint8) if 'part' in v.dtype.names else np.zeros(len(xyz), dtype=np.uint8)
    scene=xyz[part==0]; human=xyz[part==1]
    rng=np.random.default_rng(seed)
    if max_scene and len(scene)>max_scene:
        scene=scene[rng.choice(len(scene), max_scene, replace=False)]
    if max_human and len(human)>max_human:
        human=human[rng.choice(len(human), max_human, replace=False)]
    return scene, human, int((part==0).sum()), int((part==1).sum())


def stats_for(path, prior, seed=0):
    scene, human, scene_total, human_total = read_colored(path, seed=seed)
    out={'path': str(path), 'scene_points_total': scene_total, 'human_points_total': human_total,
         'scene_points_sample': int(len(scene)), 'human_points_sample': int(len(human))}
    ptree=cKDTree(prior)
    d,_=ptree.query(scene,k=1,workers=-1)
    out['scene_to_prior_dist_p50']=float(np.percentile(d,50))
    out['scene_to_prior_dist_p90']=float(np.percentile(d,90))
    out['scene_to_prior_dist_p95']=float(np.percentile(d,95))
    out['scene_to_prior_dist_mean']=float(d.mean())
    for thr in [0.1,0.2,0.5,1.0,2.0]:
        out[f'scene_prior_frac_lt_{thr}']=float((d<thr).mean())
    if len(human)>0:
        htree=cKDTree(human)
        dh,_=htree.query(scene,k=1,workers=-1)
        out['scene_to_human_dist_p01']=float(np.percentile(dh,1))
        out['scene_to_human_dist_p05']=float(np.percentile(dh,5))
        out['scene_to_human_dist_p50']=float(np.percentile(dh,50))
        for thr in [0.05,0.1,0.2,0.5]:
            out[f'scene_human_frac_lt_{thr}']=float((dh<thr).mean())
    if len(scene)>10:
        stree=cKDTree(scene)
        nn,_=stree.query(scene,k=2,workers=-1)
        nn=nn[:,1]
        out['scene_nn_p50']=float(np.percentile(nn,50))
        out['scene_nn_p95']=float(np.percentile(nn,95))
        out['scene_nn_p99']=float(np.percentile(nn,99))
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prior-npz', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('plys', nargs='+')
    args=ap.parse_args()
    prior=np.load(args.prior_npz)['points'].astype(np.float32)
    rows=[stats_for(Path(p), prior, seed=i) for i,p in enumerate(args.plys)]
    Path(args.out_json).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out_json).write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(Path(r['path']).name)
        print(json.dumps({k:v for k,v in r.items() if k!='path'}, indent=2))

if __name__=='__main__': main()
