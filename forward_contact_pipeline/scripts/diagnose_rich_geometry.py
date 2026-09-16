#!/usr/bin/env python3
"""Separate coordinate drift, shape and pose error. Oracle rows are diagnostics only."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
import roma
import smplx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from contact_streaming.rich_assets import save_json
from contact_streaming.alignment import robust_sim3
from contact_streaming.rich_geometry import reconstruct_mesh


def metrics(pred,gt,roi):
 split=len(pred)*3//4;ids=np.sort(np.random.default_rng(42).choice(10475,512,replace=False))
 fit=robust_sim3(pred[:split,ids].reshape(-1,3),gt[:split,ids].reshape(-1,3))
 error=np.linalg.norm(fit.transform(pred[split:].reshape(-1,3)).reshape(gt[split:].shape)-gt[split:],axis=-1)
 return dict(body_median_m=float(np.median(error)),body_p95_m=float(np.quantile(error,.95)),
  sole_median_m=float(np.median(error[:,roi])),sole_p95_m=float(np.quantile(error[:,roi],.95)),scale=fit.scale)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rich-root',type=Path,required=True)
 p.add_argument('--frontend',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(2)
 plan=json.loads((a.rich_root/'processed/dataset_v1/pilot_clips.json').read_text())
 manifest=json.loads((a.frontend/'manifest.json').read_text())
 cameras=json.loads((a.rich_root/'processed/dataset_v1/cameras.json').read_text())['cameras']
 roi=np.array(json.loads((a.rich_root/'processed/supervision_v2/roi.json').read_text())['vertex_ids'])
 pred_models={g:smplx.create(str(a.models),model_type='smplx',gender=g,ext='npz',use_pca=False,flat_hand_mean=True).eval() for g in ['neutral','male','female']}
 gt_models={(g,flat):smplx.create(str(a.models),model_type='smplx',gender=g,ext='npz',use_pca=True,num_pca_comps=12,flat_hand_mean=flat).eval() for g in ['male','female'] for flat in [False,True]}
 results=[]
 for clip,ref in zip(manifest['clips'],plan['clips']):
  variations={k:[] for k in ['released_world','camera_local','oracle_shape_gender','oracle_body_pose','oracle_shape_and_body_pose','oracle_head_translation']}
  targets=[];regen={False:[],True:[]};export_error=[];local_shape=[];camera_drift=[]
  for record,s in zip(clip['frames'],ref['frames']):
   with np.load(a.frontend/record['file'],allow_pickle=False) as f: d={k:f[k] for k in f.files if k.startswith(('smpl','camera','human_param'))}
   with np.load(a.rich_root/s['annotation']['path'],allow_pickle=False) as f:
    row=s['annotation']['row'];gt=f['vertices_multicam'][row].astype(np.float64)
    params={k[5:]:torch.from_numpy(f[k][row:row+1]).float() for k in f.files if k.startswith('body_') and k!='body_pose_embedding'}
   e=np.array(cameras[s['camera_key']]['CameraMatrix']);target=gt@e[:,:3].T+e[:,3];targets.append(target)
   with torch.no_grad():
    for flat in [False,True]:
     g=gt_models[(s['gender'],flat)](**params);regen[flat].append(float(np.max(np.abs(g.vertices[0].numpy()-gt))))
    # Both flat conventions only differ in hands. Use standard RICH PCA model.
    g=gt_models[(s['gender'],False)](**params)
    headgt=g.joints[0,15].numpy()@e[:,:3].T+e[:,3]
    res={k:torch.from_numpy(d[k]) for k in ['smpl_rotmat','smpl_shape','smpl_transl','smpl_expression']}
    reconstructed,_=reconstruct_mesh(res,pred_models['neutral']);cam=reconstructed[0].numpy()
    c2w=d['camera_c2w'][0];actual_cam=(d['smplx_vertices_world'][0]-c2w[:3,3])@c2w[:3,:3]
    export_error.append(float(np.max(np.abs(cam-actual_cam))))
    # Compare world reconstruction in a fixed per-clip coordinate target below.
    variations['released_world'].append(d['smplx_vertices_world'][0])
    variations['camera_local'].append(cam)
    residual=(cam[roi]-cam[roi].mean(axis=1,keepdims=True))-(target[roi]-target[roi].mean(axis=1,keepdims=True))
    local_shape.append(np.linalg.norm(residual,axis=-1))
    camera_drift.append(c2w[:3,3].tolist())
    with_shape={k:v.clone() for k,v in res.items()};with_shape['smpl_shape']=params['betas'][None]
    vv,_=reconstruct_mesh(with_shape,pred_models[s['gender']]);variations['oracle_shape_gender'].append(vv[0].numpy())
    body=res['smpl_rotmat'].clone()
    body[:,:,0]=torch.from_numpy(e[:,:3]).float()@roma.rotvec_to_rotmat(params['global_orient'])
    body[:,:,1:22]=roma.rotvec_to_rotmat(params['body_pose'].reshape(1,21,3))
    with_pose={**res,'smpl_rotmat':body}
    vv,_=reconstruct_mesh(with_pose,pred_models['neutral']);variations['oracle_body_pose'].append(vv[0].numpy())
    vv,_=reconstruct_mesh({**with_pose,'smpl_shape':params['betas'][None]},pred_models[s['gender']])
    variations['oracle_shape_and_body_pose'].append(vv[0].numpy())
    variations['oracle_head_translation'].append(cam+headgt-res['smpl_transl'][0,0].numpy())
  target=np.array(targets)
  output={k:metrics(np.array(v),target,roi) for k,v in variations.items()}
  # Independent per-frame oracle similarity upper bound, never a deployment correction.
  errors=[]
  for pred,gt in zip(variations['camera_local'],target):
   fit=robust_sim3(pred,gt);errors.append(np.linalg.norm(fit.transform(pred)-gt,axis=-1))
  output['oracle_per_frame_alignment']=dict(body_median_m=float(np.median(errors)),sole_median_m=float(np.median(np.array(errors)[:,roi])))
  results.append(dict(sequence=clip['sequence'],variations=output,
   gt_parameter_regeneration_max_error_m={str(k):max(v) for k,v in regen.items()},
   independent_mesh_export_max_error_m=max(export_error),
   foot_centroid_removed_shape_error_m=float(np.median(local_shape)),
   predicted_camera_translation_range_m=np.ptp(camera_drift,axis=0).tolist()))
 save_json(a.output,dict(kind='error_decomposition',oracle_results_not_deployable=True,clips=results));print(json.dumps(results,indent=2))
if __name__=='__main__':main()
