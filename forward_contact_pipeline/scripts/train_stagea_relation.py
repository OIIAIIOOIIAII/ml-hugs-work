#!/usr/bin/env python3
"""Train/evaluate the Stage-A geometry-only relation ablation on a frozen split."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader,Dataset
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from contact_streaming.stagea import LocalRelationEncoder

class Relations(Dataset):
 def __init__(self, manifests):
  self.seqs=[];self.index=[]
  for mp in manifests:
   m=json.loads(Path(mp).read_text()); d=np.load(Path(mp).parent/m['payload'],allow_pickle=False)
   valid=d['scene_point_valid'] if 'scene_point_valid' in d else np.ones(d['scene_point_local'].shape[:-1],np.float32)
   arrays=tuple(np.asarray(x) for x in (d['roi_vertex_local'],d['roi_vertex_normal_local'],d['scene_point_local'],d['scene_normal_local'],valid,d['vertex_contact'],d['vertex_proximity']))
   si=len(self.seqs);self.seqs.append(arrays)
   for t in range(arrays[0].shape[0]):
    for f in range(2): self.index.append((si,t,f))
 def __len__(self): return len(self.index)
 def __getitem__(self,i):
  si,t,f=self.index[i]; return tuple(torch.from_numpy(x[t,f]).float() for x in self.seqs[si])
def metrics(net,loader,dev):
 net.eval(); logits=[];labels=[]; predp=[];targetp=[]
 with torch.no_grad():
  for b in loader:
   b=[x.to(dev) for x in b];o=net(*b[:5]);logits.append(o['contact_logits'].cpu());labels.append(b[5].cpu());predp.append(o['proximity'].cpu());targetp.append(b[6].cpu())
 l=torch.cat(logits).flatten(); y=torch.cat(labels).flatten(); p=(l.sigmoid()>=.5); tp=((p)&(y>.5)).sum().item(); fp=((p)&(y<=.5)).sum().item(); fn=((~p)&(y>.5)).sum().item(); return {'f1':2*tp/max(2*tp+fp+fn,1),'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'proximity_mae_m':float((torch.cat(predp).flatten()-torch.cat(targetp).flatten()).abs().mean())}
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch',type=int,default=32);p.add_argument('--seed',type=int,default=42);a=p.parse_args();torch.manual_seed(a.seed);np.random.seed(a.seed);s=json.loads(a.split.read_text());dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ds={k:Relations(v) for k,v in s['splits'].items()}; dl={k:DataLoader(v,batch_size=a.batch,shuffle=k=='train') for k,v in ds.items()};net=LocalRelationEncoder().to(dev);opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-4); pos=sum(x[5].sum().item() for x in ds['train']); total=sum(x[5].numel() for x in ds['train']);pw=torch.tensor([(total-pos)/max(pos,1)],device=dev);hist=[];best=None
 for ep in range(1,a.epochs+1):
  net.train(); losses=[]
  for b in dl['train']:
   b=[x.to(dev) for x in b];o=net(*b[:5]);loss=torch.nn.functional.binary_cross_entropy_with_logits(o['contact_logits'],b[5],pos_weight=pw)+0.5*torch.nn.functional.smooth_l1_loss(o['proximity'],b[6]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();losses.append(float(loss.detach()))
  val=metrics(net,dl['val'],dev);hist.append({'epoch':ep,'train_loss':float(np.mean(losses)),**val})
  if best is None or val['f1']>best['f1']: best={**val,'epoch':ep,'state':{k:v.detach().cpu() for k,v in net.state_dict().items()}}
 net.load_state_dict(best.pop('state'));out={'kind':'proxy_robustness_geometry_only','split':str(a.split),'train_samples':len(ds['train']),'val_samples':len(ds['val']),'test_samples':len(ds['test']),'best_val':best,'test':metrics(net,dl['test'],dev),'history':hist};a.output.mkdir(parents=True,exist_ok=False);torch.save(net.state_dict(),a.output/'best.pt');(a.output/'metrics.json').write_text(json.dumps(out,indent=2));print(json.dumps({'best_val':best,'test':out['test']}))
if __name__=='__main__':main()
