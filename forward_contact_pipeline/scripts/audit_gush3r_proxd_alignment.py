#!/usr/bin/env python3
"""Fit/test GUSH3R coarse SMPL-X against PROXD; numerical A3 gate only."""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from contact_streaming.alignment import robust_sim3

def stat(x):
    x=np.asarray(x,float); return {'mean_m':float(x.mean()),'median_m':float(np.median(x)),'p95_m':float(np.percentile(x,95)),'max_m':float(x.max())}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-dir',type=Path,required=True); p.add_argument('--prox-root',type=Path,required=True); p.add_argument('--sequence',required=True); p.add_argument('--smplx-model-dir',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--vertex-sample',type=int,default=512); p.add_argument('--holdout-frames',type=int,default=2); p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(); import smplx
    manifest=json.loads((a.evidence_dir/'manifest.json').read_text()); frames=manifest['frames']; scene=a.sequence.rsplit('_',2)[0]
    c2w=np.asarray(json.loads((a.prox_root/'cam2world'/f'{scene}.json').read_text()),np.float32)
    # GUSH3R exports all 15 hand-joint rotation matrices, so this audit must
    # use the full-hand SMPL-X variant rather than the PCA-hand default.
    model_pred=smplx.create(str(a.smplx_model_dir),model_type='smplx',gender='neutral',ext='npz',use_pca=False,batch_size=1).eval()
    # Official PROXD pickles use its 12-component hand-PCA convention.
    model_teacher=smplx.create(str(a.smplx_model_dir),model_type='smplx',gender='neutral',ext='npz',use_pca=True,num_pca_comps=12,batch_size=1).eval()
    rng=np.random.default_rng(a.seed); vids=np.sort(rng.choice(10475,size=min(a.vertex_sample,10475),replace=False)); result_dirs=sorted((a.prox_root/'PROXD'/a.sequence/'results').glob('*/000.pkl'))
    pred,gt=[],[]; keys={'betas','body_pose','global_orient','transl','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression'}
    import roma
    for i, record in enumerate(frames):
        e=np.load(a.evidence_dir/record['file'],allow_pickle=False); rot=torch.from_numpy(e['smpl_rotmat'][0,0]).float(); rv=roma.rotmat_to_rotvec(rot)
        # The installed SMPL-X consumes axis-angle vectors robustly.  Convert
        # GUSH3R's 53 rotation matrices explicitly, rather than relying on a
        # version-dependent pose2rot=False hand-matrix path.
        kw={'betas':torch.from_numpy(e['smpl_shape'][0,0]).float().unsqueeze(0),'global_orient':rv[0].unsqueeze(0),'body_pose':rv[1:22].reshape(1,-1),'left_hand_pose':rv[22:37].reshape(1,-1),'right_hand_pose':rv[37:52].reshape(1,-1),'jaw_pose':rv[52].unsqueeze(0),'leye_pose':torch.zeros(1,3),'reye_pose':torch.zeros(1,3),'expression':torch.from_numpy(e['smpl_expression'][0,0]).float().unsqueeze(0),'transl':torch.from_numpy(e['smpl_transl'][0,0]).float().unsqueeze(0)}
        with torch.no_grad(): pv=model_pred(return_verts=True,**kw).vertices[0].numpy()
        with result_dirs[i].open('rb') as handle: teacher=pickle.load(handle,encoding='latin1')
        tk={key:torch.from_numpy(value).float() for key,value in teacher.items() if key in keys}
        with torch.no_grad(): tv=model_teacher(return_verts=True,**tk).vertices[0].numpy()
        tv=(np.c_[tv,np.ones(len(tv))]@c2w.T)[:,:3]; pred.append(pv[vids]); gt.append(tv[vids])
    pred,gt=np.asarray(pred),np.asarray(gt); split=len(pred)-a.holdout_frames
    if split < 3: raise ValueError('Need at least three fitting frames')
    fit=robust_sim3(pred[:split].reshape(-1,3),gt[:split].reshape(-1,3)); test=np.linalg.norm(fit.transform(pred[split:].reshape(-1,3))-gt[split:].reshape(-1,3),axis=1); train=np.linalg.norm(fit.transform(pred[:split].reshape(-1,3))-gt[:split].reshape(-1,3),axis=1)
    out={'kind':'gush3r_proxd_coarse_smplx_alignment_smoke','scope':'short-sequence feasibility gate; terminal held-out frames, insufficient for a formal multi-sequence result','sequence':a.sequence,'frames':len(pred),'fit_frames':list(range(split)),'heldout_frames':list(range(split,len(pred))),'sim3':{'scale':fit.scale,'rotation':fit.rotation.tolist(),'translation':fit.translation.tolist(),'residual_m':fit.residual,'confidence':fit.confidence},'train_vertex_error':stat(train),'heldout_vertex_error':stat(test),'contact_scale_m':.02,'passes_contact_gate':bool(np.median(test)<=.02),'verdict':'Only a pass permits expansion to multi-sequence real GUSH3R reliability assets.'}
    a.output.mkdir(parents=True,exist_ok=False); (a.output/'report.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
