#!/usr/bin/env python3
"""Fit/validate Human3R-SMPLX -> PROXD-SMPLX Sim(3) on held-out PROX frames."""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contact_streaming.alignment import robust_sim3
from contact_streaming.anchors import SMPLXAnchorExtractor

def err(x):
    return {"mean":float(x.mean()),"median":float(np.median(x)),"p95":float(np.percentile(x,95)),"max":float(x.max())}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frameinput',type=Path,required=True); p.add_argument('--prox-root',type=Path,default=ROOT.parent/'PROX')
    p.add_argument('--sequence',required=True); p.add_argument('--smplx-model-dir',type=Path,default=ROOT.parent/'Human3R'/'src'/'models')
    p.add_argument('--out',type=Path,required=True); p.add_argument('--vertex-sample',type=int,default=512); p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(); manifest=json.loads((a.frameinput/'manifest.json').read_text()); frames=manifest['frames']
    scene=a.sequence.rsplit('_',2)[0]; c2w=np.asarray(json.loads((a.prox_root/'cam2world'/f'{scene}.json').read_text()),np.float32)
    source=SMPLXAnchorExtractor(a.smplx_model_dir,'cpu'); import smplx
    target=smplx.create(str(a.smplx_model_dir),model_type='smplx',gender='neutral',ext='npz',use_pca=True,num_pca_comps=12,batch_size=1).eval()
    rng=np.random.default_rng(a.seed); vids=np.sort(rng.choice(10475,size=min(a.vertex_sample,10475),replace=False))
    src=[]; dst=[]; ids=[]
    for i,record in enumerate(frames):
        h=np.load(a.frameinput/record['file']); body=source(h['smpl_shape'][0],h['smpl_rotvec'][0],h['smpl_transl'][0],h['smpl_expression'][0])
        # export_frame_input preserves sorted Color-directory order; the
        # source basename is not yet stored in FrameInput v1, so PROXD is
        # matched by that independently audited order.
        name=sorted((a.prox_root/'PROXD'/a.sequence/'results').glob('*/000.pkl'))[i].parent.name
        with open(a.prox_root/'PROXD'/a.sequence/'results'/name/'000.pkl','rb') as f: q=pickle.load(f,encoding='latin1')
        keys={'betas','body_pose','global_orient','transl','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression'}
        kw={k:torch.from_numpy(v).float() for k,v in q.items() if k in keys}
        with torch.no_grad(): v=target(return_verts=True,**kw).vertices[0].numpy()
        v=(np.c_[v,np.ones(len(v))]@c2w.T)[:,:3]
        src.append(body['vertices'][vids]); dst.append(v[vids]); ids.append(i)
    ids=np.asarray(ids); hold=max(1,round(len(ids)*.25)); test=np.sort(rng.choice(ids,size=hold,replace=False)); train=np.asarray([i for i in ids if i not in set(test)])
    S=np.asarray(src); D=np.asarray(dst); fit=robust_sim3(S[train].reshape(-1,3),D[train].reshape(-1,3)); te=np.linalg.norm(fit.transform(S[test].reshape(-1,3))-D[test].reshape(-1,3),axis=1); tr=np.linalg.norm(fit.transform(S[train].reshape(-1,3))-D[train].reshape(-1,3),axis=1)
    a.out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.out,src=S[train].reshape(-1,3),dst=D[train].reshape(-1,3),train_frame_ids=train,holdout_frame_ids=test,vertex_ids=vids)
    report={'purpose':'PROX Human3R-to-PROXD calibration; held-out frames only','frames':len(ids),'train_frame_ids':train.tolist(),'holdout_frame_ids':test.tolist(),'sim3':{'scale':fit.scale,'rotation':fit.rotation.tolist(),'translation':fit.translation.tolist(),'residual':fit.residual,'confidence':fit.confidence},'train_vertex_error':err(tr),'holdout_vertex_error':err(te)}
    a.out.with_suffix('.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
