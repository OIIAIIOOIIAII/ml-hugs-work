#!/usr/bin/env python3
"""Detailed internal validation for a frozen full-RICH Stage-A checkpoint.

This is deliberately an internal ParkingLot2 evaluation, not the undistributed
official RICH validation/test set.  Missing or ambiguous associations are scored
as zero-contact predictions in all-candidate metrics.
"""
import argparse, hashlib, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from contact_streaming.rich_assets import sha256,save_json
from contact_streaming.rich_full import FullContactModel,flatten_cache


class Meter:
    def __init__(self,bins=1024,thresholds=None):
        self.bins=bins;self.thresholds=np.asarray(thresholds if thresholds is not None else np.arange(.05,1,.05))
        self.pos=np.zeros(bins,np.int64);self.neg=np.zeros(bins,np.int64)
        self.tp=np.zeros(len(self.thresholds),np.int64);self.fp=self.tp.copy();self.fn=self.tp.copy()
        self.count=self.positive=0;self.brier=0.;self.cal_count=np.zeros(10,np.int64);self.cal_pos=np.zeros(10,np.int64);self.cal_prob=np.zeros(10,np.float64)
    def update(self,p,y,valid=None):
        p=np.asarray(p,np.float64).reshape(-1);y=np.asarray(y,np.float64).reshape(-1)
        if valid is not None:
            take=np.asarray(valid).reshape(-1).astype(bool);p,y=p[take],y[take]
        if not len(p):return
        if not np.isfinite(p).all():raise ValueError('Non-finite probability')
        label=y>.5;idx=np.minimum((p*self.bins).astype(np.int64),self.bins-1)
        self.pos+=np.bincount(idx[label],minlength=self.bins);self.neg+=np.bincount(idx[~label],minlength=self.bins)
        predicted=p[:,None]>=self.thresholds[None]
        self.tp+=(predicted&label[:,None]).sum(0);self.fp+=(predicted&~label[:,None]).sum(0);self.fn+=((~predicted)&label[:,None]).sum(0)
        cidx=np.minimum((p*10).astype(np.int64),9)
        self.cal_count+=np.bincount(cidx,minlength=10);self.cal_pos+=np.bincount(cidx,weights=label.astype(float),minlength=10).astype(np.int64);self.cal_prob+=np.bincount(cidx,weights=p,minlength=10)
        self.count+=len(p);self.positive+=int(label.sum());self.brier+=float(((p-label)**2).sum())
    def compute(self):
        if not self.count:return {'valid_vertices':0}
        tp=self.pos[::-1].cumsum();fp=self.neg[::-1].cumsum();ap=float(((tp/np.maximum(tp+fp,1))*self.pos[::-1]).sum()/max(tp[-1],1))
        f1=2*self.tp/np.maximum(2*self.tp+self.fp+self.fn,1);best=int(np.argmax(f1));fixed=int(np.argmin(abs(self.thresholds-.5)))
        calibration=[]
        for i,n in enumerate(self.cal_count):
            if n:calibration.append({'lower':i/10,'upper':(i+1)/10,'count':int(n),'mean_probability':float(self.cal_prob[i]/n),'empirical_positive_rate':float(self.cal_pos[i]/n)})
        return {'valid_vertices':int(self.count),'positive_vertices':int(self.positive),'prevalence':self.positive/self.count,
                'average_precision_histogram':ap,'histogram_bins':self.bins,'brier':self.brier/self.count,
                'threshold_0_5':{'threshold':float(self.thresholds[fixed]),'precision':self.tp[fixed]/max(self.tp[fixed]+self.fp[fixed],1),'recall':self.tp[fixed]/max(self.tp[fixed]+self.fn[fixed],1),'f1':float(f1[fixed])},
                'best_f1_on_same_internal_validation':{'threshold':float(self.thresholds[best]),'precision':self.tp[best]/max(self.tp[best]+self.fp[best],1),'recall':self.tp[best]/max(self.tp[best]+self.fn[best],1),'f1':float(f1[best])},
                'threshold_curve':[{'threshold':float(t),'precision':self.tp[i]/max(self.tp[i]+self.fp[i],1),'recall':self.tp[i]/max(self.tp[i]+self.fn[i],1),'f1':float(f1[i])} for i,t in enumerate(self.thresholds)],'calibration_10bin':calibration}


def add_group(groups,key,p,y,valid=None):
    groups[key].update(p,y,valid)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();cfg=json.loads(a.config.read_text());index=json.loads((a.plan/'index.json').read_text());device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable')
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=True);model=FullContactModel(**cfg['model']).to(device);model.load_state_dict(ckpt['model']);model.eval()
    contract=json.loads((a.cache/'contract.json').read_text());expected=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    if ckpt['identity']!=expected:raise ValueError('Checkpoint/cache contract mismatch')
    overall=Meter();matched=Meter();regions=defaultdict(Meter);sequences=defaultdict(Meter);subjects=defaultdict(Meter);missing=Meter();counts=defaultdict(lambda:defaultdict(int))
    for ordinal,rid in enumerate(index['orders']['val']):
        record=index['records'][rid];data=a.cache/f'{rid:06}.npz';meta=a.cache/f'{rid:06}.json'
        m=json.loads(meta.read_text())
        if m['plan_sha256']!=record['sha256'] or m['data_sha256']!=sha256(data):raise ValueError('Cache shard changed: '+str(rid))
        with np.load(data,allow_pickle=False) as d:arrays={k:d[k] for k in d.files}
        rows=m['records'];ok=[x for x in rows if x['status']=='matched'];bad=[x for x in rows if x['status']!='matched']
        if len(ok)!=m['matched'] or len(rows)!=m['candidates']:raise ValueError('Bad metadata counts')
        counts[record['sequence']]['candidates']+=len(rows);counts[record['sequence']]['matched']+=len(ok)
        for x in rows:counts['subject:'+x['subject']]['candidates']+=1;counts['subject:'+x['subject']]['matched']+=int(x['status']=='matched')
        if ok:
            x,y=flatten_cache(arrays);out=[]
            for start in range(0,len(y['contact']),cfg['batch_size']):
                stop=start+cfg['batch_size'];inp={k:torch.from_numpy(v[start:stop]).to(device) for k,v in x.items()};out.append(model(inp)['contact_logits'].sigmoid().cpu().numpy())
            prob=np.concatenate(out);label=y['contact'];valid=y['contact_valid'];matched.update(prob,label,valid);overall.update(prob,label,valid)
            for i,row in enumerate(ok):
                sl=slice(2*i,2*i+2);add_group(sequences,record['sequence'],prob[sl],label[sl],valid[sl]);add_group(subjects,row['subject'],prob[sl],label[sl],valid[sl]);add_group(regions,'left',prob[2*i],label[2*i],valid[2*i]);add_group(regions,'right',prob[2*i+1],label[2*i+1],valid[2*i+1])
        if bad:
            label=arrays['failed_contact'].astype(np.float32).reshape(-1,64);valid=arrays['failed_contact_valid'].astype(np.float32).reshape(-1,64);zero=np.zeros_like(label)
            overall.update(zero,label,valid);missing.update(zero,label,valid)
            for i,row in enumerate(bad):
                sl=slice(2*i,2*i+2);add_group(sequences,record['sequence'],zero[sl],label[sl],valid[sl]);add_group(subjects,row['subject'],zero[sl],label[sl],valid[sl])
        if (ordinal+1)%200==0:print(f'validated {ordinal+1}/{len(index["orders"]["val"])} shards',flush=True)
    seq={k:{**v.compute(),**{'candidates':counts[k]['candidates'],'matched_samples':counts[k]['matched'],'coverage':counts[k]['matched']/max(counts[k]['candidates'],1)}} for k,v in sorted(sequences.items())}
    subj={k:{**v.compute(),**{'candidates':counts['subject:'+k]['candidates'],'matched_samples':counts['subject:'+k]['matched'],'coverage':counts['subject:'+k]['matched']/max(counts['subject:'+k]['candidates'],1)}} for k,v in sorted(subjects.items())}
    result={'kind':'rich_full_contact_internal_validation','warning':'ParkingLot2 split from RICH official train; this is not official validation/test. Threshold selection here is descriptive only.','checkpoint':str(a.checkpoint),'checkpoint_sha256':sha256(a.checkpoint),'config_sha256':sha256(a.config),'plan_sha256':sha256(a.plan/'index.json'),'cache_contract':contract,'split':'val','shards':len(index['orders']['val']),'all_candidates_missing_as_negative':overall.compute(),'matched_only':matched.compute(),'unmatched_or_ambiguous_as_zero_prediction':missing.compute(),'regions':{k:v.compute() for k,v in regions.items()},'sequences':seq,'subjects':subj}
    result['candidate_coverage']=sum(v['matched_samples'] for v in seq.values())/max(sum(v['candidates'] for v in seq.values()),1)
    result['all_positive_baseline']={'average_precision_histogram':overall.positive/max(overall.count,1),'contact_f1':2*(overall.positive/max(overall.count,1))/(1+overall.positive/max(overall.count,1))}
    a.output.parent.mkdir(parents=True,exist_ok=True);save_json(a.output,result);print(json.dumps({'output':str(a.output),'all':result['all_candidates_missing_as_negative'],'matched':result['matched_only'],'coverage':result['candidate_coverage']},indent=2))

if __name__=='__main__':main()
