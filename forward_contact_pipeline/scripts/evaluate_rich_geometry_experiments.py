#!/usr/bin/env python3
"""Measure every declared geometry trial, retaining detection failures and per-clip metrics."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from contact_streaming.rich_assets import sha256,save_json
from contact_streaming.alignment import robust_sim3


def metric(pred,gt,roi):
 n=len(pred);split=3*n//4;ids=np.sort(np.random.default_rng(42).choice(10475,512,replace=False))
 fit=robust_sim3(pred[:split,ids].reshape(-1,3),gt[:split,ids].reshape(-1,3))
 error=np.linalg.norm(fit.transform(pred[split:].reshape(-1,3)).reshape(gt[split:].shape)-gt[split:],axis=-1)
 raw=np.linalg.norm(pred-gt,axis=-1)
 return dict(body_median_m=float(np.median(error)),body_p95_m=float(np.quantile(error,.95)),
  sole_median_m=float(np.median(error[:,roi])),sole_p95_m=float(np.quantile(error[:,roi],.95)),
  raw_body_median_m=float(np.median(raw)),raw_sole_median_m=float(np.median(raw[:,roi])),
  scale=fit.scale,passes_2cm=bool(np.median(error)<=.02 and np.median(error[:,roi])<=.02))


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--rich-root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
 p.add_argument('--experiments',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--original-pilot',type=Path);a=p.parse_args()
 plan=json.loads(a.plan.read_text());cameras=json.loads((a.rich_root/'processed/dataset_v1/cameras.json').read_text())['cameras']
 roi=np.array(json.loads((a.rich_root/'processed/supervision_v2/roi.json').read_text())['vertex_ids'])
 gt=[];cache={}
 for clip in plan['clips']:
  vals=[]
  for s in clip['frames']:
   ann=s['annotation'];key=ann['path']
   if key not in cache:
    if sha256(a.rich_root/key)!=ann['sha256']:raise ValueError('GT changed')
    with np.load(a.rich_root/key,allow_pickle=False) as d:cache[key]=(d['vertices_multicam'],d['frame_ids'])
   if cache[key][1][ann['row']]!=s['frame_id']:raise ValueError('GT wrong frame')
   e=np.array(cameras[s['camera_key']]['CameraMatrix']);v=cache[key][0][ann['row']]
   vals.append(v@e[:,:3].T+e[:,3])
  gt.append(np.array(vals))
 result={};reproduction=[]
 for path in sorted(a.experiments.glob('*/manifest.json')):
  data=json.loads(path.read_text());variant=data['variant']
  if data['provenance']['plan_sha256']!=sha256(a.plan):raise ValueError('Plan changed')
  if len(data['clips'])!=len(plan['clips']):raise ValueError('Missing clips')
  clips=[]
  for ci,(clip,ref) in enumerate(zip(data['clips'],plan['clips'])):
   cameras_pred=[];world=[];track=[];missing=[]
   if len(clip['frames'])!=len(ref['frames']):raise ValueError('Missing frames')
   for fi,(record,s) in enumerate(zip(clip['frames'],ref['frames'])):
    if record['sample_id']!=s['sample_id']:raise ValueError('Frame mismatch')
    if sha256(path.parent/record['file'])!=record['sha256']:raise ValueError('Trial data changed')
    with np.load(path.parent/record['file'],allow_pickle=False) as d:
     if record['humans']!=1:missing.append(s['frame_id']);continue
     cameras_pred.append(d['vertices_camera'][0]);world.append(d['vertices_world'][0]);track.append(d['smpl_id'].reshape(-1).tolist())
     if a.original_pilot and variant=='baseline':
      with np.load(a.original_pilot/f'clip_{ci:02}/{fi:04}.npz',allow_pickle=False) as old:
       reproduction.append(float(np.max(np.abs(old['smplx_vertices_world'][0]-d['vertices_world'][0]))))
   row=dict(sequence=clip['sequence'],camera_id=clip['camera_id'],split=clip['split'],frames=len(ref['frames']),missing=missing)
   if missing or any(t!=track[0] for t in track):
    row.update(status='detection_or_identity_failure',passes_2cm=False);clips.append(row);continue
   pred=np.array(cameras_pred);w=np.array(world)
   row.update(status='measured',world=metric(w,gt[ci],roi),camera_local=metric(pred,gt[ci],roi))
   # Causal mesh EMA is a diagnostic, not a valid articulated-body deployment.
   row['mesh_ema_diagnostic']={}
   for alpha in [.25,.5,.75]:
    smoothed=[pred[0]]
    for v in pred[1:]:smoothed.append(alpha*v+(1-alpha)*smoothed[-1])
    row['mesh_ema_diagnostic'][str(alpha)]=metric(np.array(smoothed),gt[ci],roi)
   clips.append(row)
  measured=[r for r in clips if r['status']=='measured']
  summary={}
  for frame in ['world','camera_local']:
   summary[frame]=dict(clips=len(clips),measured=len(measured),passes=sum(r[frame]['passes_2cm'] for r in measured),
    body_median_over_clips_m=float(np.median([r[frame]['body_median_m'] for r in measured])) if measured else None,
    sole_median_over_clips_m=float(np.median([r[frame]['sole_median_m'] for r in measured])) if measured else None,
    worst_sole_median_m=max([r[frame]['sole_median_m'] for r in measured],default=None))
  result[variant]=dict(summary=summary,clips=clips)
 # Preserve measurements even when the export reproduction check fails.
 # A failed check blocks use of ablation results until the discrepancy is resolved.
 out=dict(schema_version=1,plan_sha256=sha256(a.plan),results=result,
  baseline_reproduction_max_abs_m=max(reproduction) if reproduction else None,
  baseline_reproduction_passed=(max(reproduction)<=1e-4) if reproduction else None,
  scope='per-clip terminal holdout; all failed clips retained; no GT alignment applied at inference',training_ready=False)
 save_json(a.output,out)
 print(json.dumps({k:v['summary'] for k,v in result.items()},indent=2))
if __name__=='__main__':main()
