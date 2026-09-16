#!/usr/bin/env python3
"""Resumable geometry coverage export, retaining frozen native896 predictions."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from contact_streaming.rich_assets import sha256,save_json
from contact_streaming.rich_supervision import atomic_npz,pixel_transform
from contact_streaming.rich_geometry import native_mhmr_image,reconstruct_humangs_geometry,calibrated_images

VARIANTS={
 'baseline':dict(size=512,native=False,reset=False),
 'native896':dict(size=512,native=True,reset=False),
 'encoder768_native896':dict(size=768,native=True,reset=False),
 'singleframe':dict(size=512,native=False,reset=True),
 'native896_singleframe':dict(size=512,native=True,reset=True),
 'calibrated_native896':dict(size=512,native=True,reset=False,calibrated=True),
 'native896_context32':dict(size=512,native=True,reset=False,context=32),
}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--repo',type=Path,required=True);p.add_argument('--rich-root',type=Path,required=True)
 p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--dino-repo',type=Path,required=True);p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
 a=p.parse_args();repo=a.repo.resolve();rich=a.rich_root.resolve();output=a.output.resolve();plan=a.plan.resolve()
 if not plan.is_file():raise FileNotFoundError(plan)
 output.mkdir(parents=True,exist_ok=True)
 os.environ['GUSH3R_DINO_HUB_LOCAL_REPO']=str(a.dino_repo.resolve())
 os.chdir(repo);sys.path[:0]=[str(repo),str(repo/'src')]
 import torch
 from infer import load_model,prepare_input
 from src.dust3r.inference import inference_recurrent_lighter
 from src.dust3r.utils.camera import pose_encoding_to_camera
 torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
 settings=SimpleNamespace(gs_conf_threshold=1.,bg_mask_threshold=.02,bg_mask_dilation=3,
  bg_voxel_size=.005,bg_gaussian_max=1,bg_cap_policy='random_legacy')
 model=load_model('cuda',settings);model.requires_grad_(False);model.eval()
 # Gaussian branches consume human predictions but never feed back into SMPL-X.
 # Disable them for geometry-only experiments. Compare baseline numerically to
 # the fully enabled original pilot before interpreting any ablation.
 model.gs_mode='geometry_only';model.skip_inference_render_outputs=True
 layer=model.human_gs_head.smplx_mesh.layer['neutral'];layer.eval();layer.requires_grad_(False)
 features={};original_head=model._downstream_head
 def wrapped_head(*args,**kwargs):
  if kwargs.get('smpl_token') is not None: features['smpl_query']=kwargs['smpl_token'].detach().cpu().numpy()
  return original_head(*args,**kwargs)
 model._downstream_head=wrapped_head
 clips=json.loads(plan.read_text())['clips'];start=time.time()
 context_index={}
 if any(VARIANTS[v].get('context') for v in a.variants):
  wanted={(c['sequence'],c['camera_id']) for c in clips}
  for split in ['train','val']:
   with (rich/f'processed/dataset_v1/samples_{split}.jsonl').open() as stream:
    for line in stream:
     sample=json.loads(line)
     if (sample['sequence'],sample['camera_id']) in wanted:
      context_index[(sample['sequence'],sample['camera_id'],sample['frame_id'])]=sample
 calibrations=json.loads((rich/'processed/dataset_v1/cameras.json').read_text())['cameras'] if any(VARIANTS[v].get('calibrated') for v in a.variants) else {}
 provenance=dict(frontend_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
  frontend_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff'])).hexdigest(),
  checkpoint_sha256=sha256(repo/'checkpoints/gush3r.pth'),runner_sha256=sha256(Path(__file__)),
  helper_sha256=sha256(ROOT/'contact_streaming/rich_geometry.py'),plan_sha256=sha256(plan),
  frozen=True,geometry_only=True,GT_body_or_contact_input=False,sensor_intrinsics_only_for_calibrated_variant=True,training_ready=False)
 for variant in a.variants:
  cfg=VARIANTS[variant];dest=output/variant;dest.mkdir(exist_ok=True)
  if (dest/'manifest.json').exists():
   old=json.loads((dest/'manifest.json').read_text())
   if old['provenance']!=provenance: raise ValueError('Existing experiment provenance changed')
   for c in old['clips']:
    for f in c['frames']:
     if sha256(dest/f['file'])!=f['sha256']:raise ValueError('Existing experiment file changed')
   print('Verified completed variant: '+variant,flush=True);continue
  contract=dict(config=cfg,provenance=provenance)
  if (dest/'provenance.json').exists():
   if json.loads((dest/'provenance.json').read_text())!=contract:raise ValueError('Partial-run provenance changed')
  else:save_json(dest/'provenance.json',contract)
  result=dict(variant=variant,config=cfg,provenance=provenance,clips=[])
  for ci,clip in enumerate(clips):
   completed=dest/f'clip_{ci:02}.json'
   if completed.exists():
    prior=json.loads(completed.read_text())
    if [f['sample_id'] for f in prior['frames']]!=[s['sample_id'] for s in clip['frames']]:raise ValueError('Partial clip identity changed')
    for f in prior['frames']:
     if sha256(dest/f['file'])!=f['sha256']:raise ValueError('Partial clip hash changed')
    result['clips'].append(prior);print(f'Verified completed clip {ci}',flush=True);continue
   torch.manual_seed(42+ci);np.random.seed(42+ci)
   context=[]
   for back in range(1,cfg.get('context',0)+1):
    key=(clip['sequence'],clip['camera_id'],clip['frames'][0]['frame_id']-back)
    if key not in context_index:break
    context.append(context_index[key])
   context=context[::-1];input_frames=context+clip['frames'];context_count=len(context)
   paths=[]
   for frame in input_frames:
    path=rich/frame['image']
    if sha256(path)!=frame['image_sha256']:raise ValueError('Input hash mismatch')
    paths.append(str(path))
   views=prepare_input(paths,size=cfg['size'],img_res=model.mhmr_img_res)
   actual_transforms=[]
   for path,view,sample in zip(paths,views,input_frames):
    if cfg.get('calibrated'):
     view['img'],view['img_mhmr'],actual=calibrated_images(path,calibrations[sample['camera_key']]['Intrinsics'],view['camera_intrinsics'].numpy(),tuple(view['img'].shape[-2:][::-1]),model.mhmr_img_res)
     actual_transforms.append(actual)
    else:
     actual_transforms.append(None)
     if cfg['native']:view['img_mhmr']=native_mhmr_image(path,cfg['size'],model.mhmr_img_res)
   records=[];current=0
   def callback(i,res,view):
    nonlocal current
    absolute=current if cfg['reset'] else i
    if absolute<context_count:features.clear();return
    idx=absolute-context_count;sample=clip['frames'][idx]
    arrays=dict(features);features.clear()
    for key in ['smpl_rotmat','smpl_shape','smpl_transl','smpl_expression','smpl_id','smpl_loc']:
     if torch.is_tensor(res.get(key)):arrays[key]=res[key].detach().cpu().numpy()
    c2w=pose_encoding_to_camera(res['camera_pose'])[0]
    arrays['camera_c2w']=c2w.detach().cpu().numpy()
    geometry=reconstruct_humangs_geometry(res,model.human_gs_head.smplx_mesh,c2w) if res.get('smpl_rotmat') is not None else None
    humans=0
    if geometry is not None:
     verts,joints,world=geometry;humans=verts.shape[0]
     arrays['vertices_camera']=verts.cpu().numpy();arrays['joints_camera']=joints.cpu().numpy()
     arrays['vertices_world']=world.cpu().numpy()
    for key,v in arrays.items():
     if not np.isfinite(v).all():raise ValueError('Nonfinite '+key)
    name=f'clip_{ci:02}/{idx:04}.npz';atomic_npz(dest/name,**arrays)
    records.append(dict(file=name,sha256=sha256(dest/name),sample_id=sample['sample_id'],frame_id=sample['frame_id'],
     humans=humans,transform=pixel_transform(*sample['raw_size_wh'],cfg['size'],model.mhmr_img_res),sensor_calibrated_raw_to_crop=actual_transforms[absolute]))
    save_json(output/'progress.json',dict(variant=variant,clip=ci,frame=idx+1,elapsed_s=time.time()-start))
   with torch.no_grad():
    if cfg['reset']:
     for current,view in enumerate(views):
      inference_recurrent_lighter([view],model,'cuda',verbose=False,output_callback=callback,keep_outputs=False)
    else:inference_recurrent_lighter(views,model,'cuda',verbose=False,output_callback=callback,keep_outputs=False)
   result['clips'].append({k:v for k,v in clip.items() if k!='frames'}|dict(frames=records,past_context_frame_ids=[c['frame_id'] for c in context]))
   save_json(dest/f'clip_{ci:02}.json',result['clips'][-1])
   print(json.dumps(dict(variant=variant,clip=ci,frames=len(records),humans=[r['humans'] for r in records],elapsed_s=time.time()-start)),flush=True)
   del views;gc.collect();torch.cuda.empty_cache()
  result.update(status='complete',elapsed_s=time.time()-start)
  save_json(dest/'manifest.json',result)
 save_json(output/'progress.json',dict(status='complete',elapsed_s=time.time()-start,variants=a.variants))
if __name__=='__main__':main()
