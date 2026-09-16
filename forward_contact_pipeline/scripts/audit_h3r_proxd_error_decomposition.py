#!/usr/bin/env python3
"""Numerically decompose Human3R-to-PROXD error without reading images.

This is an A3 feasibility diagnostic.  A global Sim(3), fitted on the stored
training correspondence frames only, maps Human3R SMPL-X vertices to the PROX
scene frame.  On every frame we report global, root/centroid and root-aligned
local geometry error.  It cannot turn a failed global alignment into training
labels; it tests whether a camera/root-local contact contract is plausible.
"""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from contact_streaming.alignment import Sim3
from contact_streaming.anchors import SMPLXAnchorExtractor

def summary(values):
    values=np.asarray(values,float)
    return {'mean_m':float(values.mean()),'median_m':float(np.median(values)),'p95_m':float(np.percentile(values,95)),'max_m':float(values.max())}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frameinput',type=Path,required=True); p.add_argument('--prox-root',type=Path,default=ROOT.parent/'PROX'); p.add_argument('--sequence',required=True)
    p.add_argument('--correspondence-json',type=Path,required=True); p.add_argument('--smplx-model-dir',type=Path,default=ROOT.parent/'Human3R'/'src'/'models'); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); manifest=json.loads((a.frameinput/'manifest.json').read_text()); calibration=json.loads(a.correspondence_json.read_text()); sim=calibration['sim3']; transform=Sim3(float(sim['scale']),np.asarray(sim['rotation'],np.float32),np.asarray(sim['translation'],np.float32),float(sim['residual']),float(sim['confidence']),'stored_heldout_calibration')
    scene=a.sequence.rsplit('_',2)[0]; c2w=np.asarray(json.loads((a.prox_root/'cam2world'/f'{scene}.json').read_text()),np.float32)
    source=SMPLXAnchorExtractor(a.smplx_model_dir,'cpu'); import smplx
    target=smplx.create(str(a.smplx_model_dir),model_type='smplx',gender='neutral',ext='npz',use_pca=True,num_pca_comps=12,batch_size=1).eval()
    result_dirs=sorted((a.prox_root/'PROXD'/a.sequence/'results').glob('*/000.pkl'))
    global_errors=[]; local_errors=[]; root_errors=[]; records=[]
    keys={'betas','body_pose','global_orient','transl','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression'}
    for frame, record in enumerate(manifest['frames']):
        data=np.load(a.frameinput/record['file']); pred=source(data['smpl_shape'][0],data['smpl_rotvec'][0],data['smpl_transl'][0],data['smpl_expression'][0])['vertices']
        with result_dirs[frame].open('rb') as handle: teacher=pickle.load(handle,encoding='latin1')
        kwargs={key:torch.from_numpy(value).float() for key,value in teacher.items() if key in keys}
        with torch.no_grad(): gt=target(return_verts=True,**kwargs).vertices[0].numpy()
        gt=(np.c_[gt,np.ones(len(gt))]@c2w.T)[:,:3]; pred=transform.transform(pred)
        pred_root=pred.mean(0); gt_root=gt.mean(0); root=float(np.linalg.norm(pred_root-gt_root))
        ge=np.linalg.norm(pred-gt,axis=1); le=np.linalg.norm((pred-pred_root)-(gt-gt_root),axis=1)
        global_errors.extend(ge); local_errors.extend(le); root_errors.append(root)
        records.append({'frame':frame,'global_vertex_median_m':float(np.median(ge)),'local_vertex_median_m':float(np.median(le)),'root_centroid_error_m':root})
    output={'kind':'human3r_proxd_error_decomposition','scope':'real Human3R RGB output vs PROXD; global Sim(3) fitted only on stored training frames; diagnostic only','sequence':a.sequence,'frames':len(records),'global_vertex_error':summary(global_errors),'root_centroid_error':summary(root_errors),'root_aligned_local_vertex_error':summary(local_errors),'frame_metrics':records,'gate':{'contact_scale_target_m':0.02,'passes_global_contact_scale':bool(np.median(global_errors)<=.02),'passes_root_aligned_local_contact_scale':bool(np.median(local_errors)<=.02)},'verdict':'camera/root-local residual remains diagnostic-only unless root-aligned local error reaches contact scale; no contact label is produced by this script'}
    a.output.parent.mkdir(parents=True,exist_ok=False); (a.output/'report.json').write_text(json.dumps(output,indent=2)); print(json.dumps({k:output[k] for k in ('global_vertex_error','root_centroid_error','root_aligned_local_vertex_error','gate')},indent=2))
if __name__=='__main__': main()
