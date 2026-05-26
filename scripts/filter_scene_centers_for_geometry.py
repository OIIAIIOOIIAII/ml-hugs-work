#!/usr/bin/env python3
"""Filter a trained scene splat PLY into a cleaner contact-oriented point cloud."""

import argparse, json
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


def xyz_from_ply(path):
    ply=PlyData.read(str(path), mmap=True); v=ply['vertex'].data
    xyz=np.stack([v['x'],v['y'],v['z']],axis=1).astype(np.float32)
    return xyz, v


def read_human(path):
    ply=PlyData.read(str(path), mmap=True); v=ply['vertex'].data
    xyz=np.stack([v['x'],v['y'],v['z']],axis=1).astype(np.float32)
    if 'part' in v.dtype.names:
        xyz=xyz[v['part']==1]
    return xyz


def write_colored(path, scene_xyz, human_xyz, scene_color=(70,150,255), human_color=(255,80,40)):
    xyz=np.concatenate([scene_xyz,human_xyz],axis=0).astype(np.float32)
    n=len(scene_xyz)
    dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1'),('part','u1')]
    verts=np.empty(len(xyz),dtype=dtype)
    verts['x'],verts['y'],verts['z']=xyz[:,0],xyz[:,1],xyz[:,2]
    verts['red'][:n],verts['green'][:n],verts['blue'][:n],verts['part'][:n]=scene_color[0],scene_color[1],scene_color[2],0
    verts['red'][n:],verts['green'][n:],verts['blue'][n:],verts['part'][n:]=human_color[0],human_color[1],human_color[2],1
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    PlyData([PlyElement.describe(verts,'vertex')], text=False).write(str(path))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scene-splat-ply', required=True)
    ap.add_argument('--human-ply', required=True)
    ap.add_argument('--prior-npz', required=True)
    ap.add_argument('--out-ply', required=True)
    ap.add_argument('--max-prior-dist', type=float, default=2.0)
    ap.add_argument('--min-opacity', type=float, default=0.05, help='sigmoid opacity threshold')
    ap.add_argument('--human-exclusion-radius', type=float, default=0.12)
    ap.add_argument('--max-scene-points', type=int, default=300000)
    ap.add_argument('--seed', type=int, default=0)
    args=ap.parse_args()
    scene_xyz, v=xyz_from_ply(args.scene_splat_ply)
    keep=np.ones(len(scene_xyz), dtype=bool)
    report={'input_scene_points': int(len(scene_xyz))}
    if 'opacity' in v.dtype.names and args.min_opacity>0:
        op=np.asarray(v['opacity'])
        sig=1/(1+np.exp(-op))
        k=sig>=args.min_opacity
        report['removed_low_opacity']=int((~k).sum())
        keep &= k
    prior=np.load(args.prior_npz)['points'].astype(np.float32)
    if args.max_prior_dist>0:
        tree=cKDTree(prior)
        d,_=tree.query(scene_xyz, k=1, workers=-1)
        k=d<=args.max_prior_dist
        report['removed_far_from_prior']=int((~k).sum())
        report['prior_dist_kept_mean']=float(d[k].mean()) if k.any() else None
        keep &= k
    human=read_human(args.human_ply)
    if args.human_exclusion_radius>0 and len(human)>0:
        ht=cKDTree(human)
        d,_=ht.query(scene_xyz, k=1, workers=-1)
        k=d>args.human_exclusion_radius
        report['removed_near_human']=int((~k).sum())
        keep &= k
    filt=scene_xyz[keep]
    report['after_filters_scene_points']=int(len(filt))
    if args.max_scene_points>0 and len(filt)>args.max_scene_points:
        rng=np.random.default_rng(args.seed)
        idx=rng.choice(len(filt), args.max_scene_points, replace=False)
        filt=filt[idx]
        report['after_subsample_scene_points']=int(len(filt))
    write_colored(args.out_ply, filt, human)
    report['human_points']=int(len(human)); report['out_ply']=str(args.out_ply)
    Path(args.out_ply).with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__=='__main__': main()
