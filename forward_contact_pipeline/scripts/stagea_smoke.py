#!/usr/bin/env python3
"""Numerical shape/gradient smoke for Stage-A; no images are read."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contact_streaming.stagea import LocalRelationEncoder
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 m=json.loads(a.manifest.read_text());d=np.load(a.manifest.parent/m['payload'],allow_pickle=False);dev=torch.device(a.device)
 x={k:torch.from_numpy(d[k][:2,:,:,]).float().to(dev) for k in ('roi_vertex_local','roi_vertex_normal_local','scene_point_local','scene_normal_local','vertex_contact','vertex_proximity')}; valid=torch.from_numpy(d['scene_point_valid'][:2]).float().to(dev) if 'scene_point_valid' in d else torch.ones(x['scene_point_local'].shape[:-1],device=dev)
 net=LocalRelationEncoder().to(dev);o=net(x['roi_vertex_local'],x['roi_vertex_normal_local'],x['scene_point_local'],x['scene_normal_local'],valid);loss=torch.nn.functional.binary_cross_entropy_with_logits(o['contact_logits'],x['vertex_contact'])+torch.nn.functional.smooth_l1_loss(o['proximity'],x['vertex_proximity']);loss.backward(); report={'shapes':{k:list(v.shape) for k,v in o.items()},'loss':float(loss.detach().cpu()),'finite':bool(all(torch.isfinite(v).all() for v in o.values()))};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
