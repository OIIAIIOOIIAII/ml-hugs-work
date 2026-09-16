#!/usr/bin/env python3
"""Decompose existing Human3R/PROXD correspondence residuals without SMPL-X.

The stored correspondence asset contains 512 same-topology vertices for each
of 45 calibration frames.  This audit intentionally labels itself train-frame
diagnostic: it distinguishes global drift from root/centroid and local shape/
pose error, but cannot be used as an independent generalization metric.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from contact_streaming.alignment import Sim3

def summary(x):
    x=np.asarray(x,float); return {'mean_m':float(x.mean()),'median_m':float(np.median(x)),'p95_m':float(np.percentile(x,95)),'max_m':float(x.max())}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--correspondence-npz',type=Path,required=True); p.add_argument('--correspondence-json',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    c=json.loads(a.correspondence_json.read_text()); d=np.load(a.correspondence_npz,allow_pickle=False); src,dst=d['src'],d['dst']; frames=d['train_frame_ids']; count=len(frames)
    if len(src)%count: raise ValueError('correspondence points not divisible by train frames')
    per_frame=len(src)//count; sim=c['sim3']; transform=Sim3(float(sim['scale']),np.asarray(sim['rotation'],np.float32),np.asarray(sim['translation'],np.float32),float(sim['residual']),float(sim['confidence']),'stored_train_calibration')
    pred=transform.transform(src).reshape(count,per_frame,3); gt=dst.reshape(count,per_frame,3); pred_center=pred.mean(1); gt_center=gt.mean(1)
    global_error=np.linalg.norm(pred-gt,axis=-1); local_error=np.linalg.norm((pred-pred_center[:,None])-(gt-gt_center[:,None]),axis=-1); root_error=np.linalg.norm(pred_center-gt_center,axis=-1)
    records=[{'frame':int(frames[i]),'global_median_m':float(np.median(global_error[i])),'root_centroid_m':float(root_error[i]),'root_aligned_local_median_m':float(np.median(local_error[i]))} for i in range(count)]
    out={'kind':'human3r_proxd_stored_correspondence_decomposition','scope':'train-frame calibration diagnostic only; no RGB read, no held-out/contact result','frames':count,'vertices_per_frame':per_frame,'global_vertex_error':summary(global_error),'root_centroid_error':summary(root_error),'root_aligned_local_vertex_error':summary(local_error),'contact_scale_m':.02,'gate':{'root_aligned_local_contact_scale':bool(np.median(local_error)<=.02),'note':'Failure blocks use of this Human3R stream for contact-scale reliability labels.'},'frame_metrics':records}
    a.output.mkdir(parents=True,exist_ok=False); (a.output/'report.json').write_text(json.dumps(out,indent=2)); print(json.dumps({k:out[k] for k in ('global_vertex_error','root_centroid_error','root_aligned_local_vertex_error','gate')},indent=2))
if __name__=='__main__': main()
