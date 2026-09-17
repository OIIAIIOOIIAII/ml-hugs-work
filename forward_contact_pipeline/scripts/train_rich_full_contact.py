#!/usr/bin/env python3
"""Full RICH traversal: train during first frozen-cache pass, then reuse all shards."""
import argparse,fcntl,hashlib,json,os,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from contact_streaming.rich_assets import sha256,save_json
from contact_streaming.rich_supervision import atomic_npz
from contact_streaming.rich_full import FullContactModel,flatten_cache,INPUT_KEYS
from contact_streaming.training.components import ContactObjective,ContactMetrics
from contact_streaming.training.engine import atomic_checkpoint


def train_shard(model,optimizer,arrays,seed,batch_size,device):
    inputs,targets=flatten_cache(arrays);order=np.random.default_rng(seed).permutation(len(targets['contact']))
    objective=ContactObjective();model.train();loss_sum=steps=count=0
    for start in range(0,len(order),batch_size):
        ids=order[start:start+batch_size]
        x={k:torch.from_numpy(v[ids]).to(device) for k,v in inputs.items()}
        y={k:torch.from_numpy(v[ids]).to(device) for k,v in targets.items()}
        loss=objective(model(x),y)
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
        loss_sum+=float(loss.detach())*len(ids);count+=len(ids);steps+=1
    return loss_sum,count,steps


def meter_state(meter):return {k:v for k,v in vars(meter).items()}
def restore_meter(state):
    m=ContactMetrics()
    for k,v in state.items():setattr(m,k,v)
    return m


class Runner:
    def __init__(self,a):
        self.a=a;self.output=a.output.resolve();self.plan=a.plan.resolve();self.cache=a.cache.resolve();self.rich=a.rich_root.resolve();self.repo=a.repo.resolve()
        self.cfg=json.loads(a.config.read_text());self.index=json.loads((self.plan/'index.json').read_text())
        if not self.index.get('all_eligible_samples') or not self.index.get('training_authorized'):raise ValueError('Full eligible plan required')
        self.output.mkdir(parents=True,exist_ok=bool(a.resume));self.cache.mkdir(parents=True,exist_ok=True)
        self.lock=(self.output/'.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.front=None;self.executor=ThreadPoolExecutor(max_workers=1);self.prefetch=None;self.new_chunks=0;self.started=time.time()
        torch.set_num_threads(self.cfg['torch_threads']);torch.manual_seed(self.cfg['seed']);torch.backends.cuda.matmul.allow_tf32=True
        self.device=torch.device(a.device)
        if self.device.type!='cuda':raise ValueError('Full frozen extraction requires CUDA')
        files=[Path(__file__),ROOT/'contact_streaming/rich_full.py',ROOT/'contact_streaming/rich_full_frontend.py',ROOT/'contact_streaming/rich_geometry.py',ROOT/'contact_streaming/rich_supervision.py',ROOT/'contact_streaming/rich_sensor_intrinsics.py',ROOT/'contact_streaming/training/components.py']
        source={p.name:sha256(p) for p in files}
        self.contract=dict(plan_sha256=sha256(self.plan/'index.json'),config_sha256=sha256(a.config),source_sha256=source,
            frontend_source_sha256=sha256(self.repo/'src/dust3r/model.py'),checkpoint_sha256=sha256(self.repo/'checkpoints/gush3r.pth'))
        self.identity=hashlib.sha256(json.dumps(self.contract,sort_keys=True).encode()).hexdigest()
        self.model=FullContactModel(**self.cfg['model']).to(self.device)
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=self.cfg['lr'],weight_decay=self.cfg['weight_decay'])
        self.state=dict(epoch=1,phase='train',position=0,global_steps=0,loss_sum=0.,roi_samples=0,candidates=0,matched=0,
            best_ap=-1.,history=[],val_matched=None,val_all=None,val_regions=None,completed_epochs=0)
        contract_path=self.cache/'contract.json'
        if contract_path.exists():
            if json.loads(contract_path.read_text())!=self.contract:raise ValueError('Cache source/config changed')
        else:save_json(contract_path,self.contract)
        if a.resume:
            ckpt=torch.load(self.output/'last.pt',map_location='cpu',weights_only=True)
            if ckpt['identity']!=self.identity:raise ValueError('Resume source/config/data changed; use a new version')
            self.model.load_state_dict(ckpt['model']);self.optimizer.load_state_dict(ckpt['optimizer']);self.state=ckpt['state']
            torch.set_rng_state(ckpt['torch_rng']);torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
        else:
            (self.output/'sources').mkdir()
            for p in files:(self.output/'sources'/p.name).write_text(p.read_text())
            save_json(self.output/'contract.json',self.contract);save_json(self.output/'config.json',self.cfg)
            self.save()
        self.progress('running')

    def save(self,best=False):
        ckpt=dict(format_version=1,identity=self.identity,model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),state=self.state,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),training_authorized=True,deployment_ready=False)
        atomic_checkpoint(self.output/'last.pt',ckpt)
        if best:atomic_checkpoint(self.output/'best.pt',ckpt)

    def progress(self,status,**extra):
        s=self.state
        value=dict(status=status,epoch=s['epoch'],completed_epochs=s['completed_epochs'],phase=s['phase'],completed_chunks_in_phase=s['position'],
            total_chunks_in_phase=len(self.index['orders'][s['phase']]),global_steps=s['global_steps'],candidates_seen_in_phase=s['candidates'],
            matched_samples_in_phase=s['matched'],full_plan_samples=self.index['samples'],full_plan_unique_images=self.index['unique_images'],
            train_loss=s['loss_sum']/max(s['roi_samples'],1),updated=time.time(),session_elapsed_s=time.time()-self.started,
            training_authorized=True,deployment_ready=False,**extra)
        save_json(self.output/'progress.json',value);print(json.dumps(value),flush=True)

    def clip(self,record):
        file=self.plan/record['path']
        if sha256(file)!=record['sha256']:raise ValueError('Clip plan changed')
        return json.loads(file.read_text())

    def ensure_front(self):
        if self.front is None:
            from contact_streaming.rich_full_frontend import FrozenFullFrontend
            with torch.random.fork_rng(devices=[0]):self.front=FrozenFullFrontend(self.repo,self.rich,self.a.dino_repo,self.cfg['preprocess_workers'])

    def paths(self,record):return self.cache/f"{record['id']:06}.npz",self.cache/f"{record['id']:06}.json"

    def schedule(self,record):
        _,meta=self.paths(record)
        if meta.exists():return
        self.ensure_front()
        self.prefetch=(record['id'],self.executor.submit(self.front.prepare,self.clip(record)))

    def get(self,record,next_record):
        data,meta=self.paths(record)
        if meta.exists():
            m=json.loads(meta.read_text())
            if m['plan_sha256']!=record['sha256'] or m['data_sha256']!=sha256(data):raise ValueError('Cached shard changed')
            with np.load(data,allow_pickle=False) as d:arrays={k:d[k] for k in d.files}
            if self.prefetch is None and next_record is not None:self.schedule(next_record)
            return arrays,m
        self.ensure_front();clip=self.clip(record)
        if self.prefetch is not None:
            key,future=self.prefetch
            if key!=record['id']:raise ValueError('Prefetch order mismatch')
            views=future.result();self.prefetch=None
        else:views=self.front.prepare(clip)
        if next_record is not None:self.schedule(next_record)
        arrays,rows=self.front.extract(clip,views,42+record['id']);del views
        atomic_npz(data,**arrays)
        m=dict(id=record['id'],split=record['split'],sequence=record['sequence'],plan_sha256=record['sha256'],data_sha256=sha256(data),
            candidates=record['targets'],matched=sum(r['status']=='matched' for r in rows),records=rows,
            input_fields=list(INPUT_KEYS),label_fields=['contact','contact_valid'],GT_as_model_input=False,geometry_error_filter=False)
        if len(rows)!=record['targets']:raise ValueError('Lost planned target')
        save_json(meta,m);self.new_chunks+=1
        # First-pass training consumes the exact same quantized arrays as later epochs.
        return arrays,m

    @torch.no_grad()
    def validate_shard(self,arrays,meta):
        matched=restore_meter(self.state['val_matched']) if self.state['val_matched'] else ContactMetrics()
        all_meter=restore_meter(self.state['val_all']) if self.state['val_all'] else ContactMetrics()
        regions=[restore_meter(x) for x in self.state['val_regions']] if self.state['val_regions'] else [ContactMetrics(),ContactMetrics()]
        self.model.eval()
        if meta['matched']:
            x,y=flatten_cache(arrays)
            for start in range(0,len(y['contact']),self.cfg['batch_size']):
                stop=start+self.cfg['batch_size'];inputs={k:torch.from_numpy(v[start:stop]).to(self.device) for k,v in x.items()}
                targets={k:torch.from_numpy(v[start:stop]).to(self.device) for k,v in y.items()};pred=self.model(inputs)
                matched.update(pred,targets);all_meter.update(pred,targets)
                for region in [0,1]:
                    take=np.arange(start,min(stop,len(y['contact'])))%2==region
                    regions[region].update({k:v[take] for k,v in pred.items()},{k:v[take] for k,v in targets.items()})
        if len(arrays['failed_contact']):
            target={k:torch.from_numpy(arrays['failed_'+k].astype(np.float32).reshape(-1,64)) for k in ['contact','contact_valid']}
            all_meter.update({'contact_logits':torch.full_like(target['contact'],-30)},target)
        self.state['val_matched']=meter_state(matched);self.state['val_all']=meter_state(all_meter);self.state['val_regions']=[meter_state(m) for m in regions]

    def finish_validation(self):
        s=self.state;matched=restore_meter(s['val_matched']);all_meter=restore_meter(s['val_all'])
        positives=float(all_meter.pos.sum());prevalence=positives/max(all_meter.count,1)
        val=dict(epoch=s['epoch'],candidate_coverage=s['matched']/max(s['candidates'],1),matched_samples=s['matched'],candidates=s['candidates'],
            matched=matched.compute(),all_candidates_missing_as_negative=all_meter.compute(),per_region_matched={str(i):restore_meter(m).compute() for i,m in enumerate(s['val_regions'])},
            all_positive_baseline=dict(contact_f1=2*prevalence/(1+prevalence),average_precision_histogram=prevalence),
            training_authorized=True,deployment_ready=False)
        score=val['all_candidates_missing_as_negative']['average_precision_histogram'];improved=score>s['best_ap']
        s['best_ap']=max(score,s['best_ap']);s['history'].append(val);s['completed_epochs']=s['epoch'];s['epoch']+=1
        s.update(phase='train',position=0,loss_sum=0.,roi_samples=0,candidates=0,matched=0,val_matched=None,val_all=None,val_regions=None)
        save_json(self.output/'history.json',s['history']);self.save(best=improved);print(json.dumps(dict(stage='validation_complete',**val)),flush=True)
        if self.front is not None:self.front.close();self.front=None

    def run(self):
        try:
            while self.state['epoch']<=self.cfg['epochs']:
                s=self.state;phase=s['phase'];order=list(self.index['orders'][phase])
                if phase=='train' and s['epoch']>1:order=np.random.default_rng(self.cfg['seed']+s['epoch']).permutation(order).tolist()
                for position in range(s['position'],len(order)):
                    record=self.index['records'][order[position]];nxt=self.index['records'][order[position+1]] if position+1<len(order) else None
                    arrays,meta=self.get(record,nxt)
                    if phase=='train' and meta['matched']:
                        loss,count,steps=train_shard(self.model,self.optimizer,arrays,self.cfg['seed']+s['epoch']*100000+record['id'],self.cfg['batch_size'],self.device)
                        s['loss_sum']+=loss;s['roi_samples']+=count;s['global_steps']+=steps
                    elif phase=='val':self.validate_shard(arrays,meta)
                    s['position']=position+1;s['candidates']+=meta['candidates'];s['matched']+=meta['matched']
                    if s['position']%self.cfg['save_every_chunks']==0:self.save()
                    self.progress('running',last_clip=record['id'],last_clip_candidates=meta['candidates'],last_clip_matched=meta['matched'])
                    if self.a.max_chunks and s['position']>=self.a.max_chunks:
                        self.save();self.progress('paused_after_requested_chunks');return
                if phase=='train':
                    if not s['roi_samples']:raise ValueError('No matched training samples')
                    s['phase']='val';s['position']=0;s['candidates']=s['matched']=0;self.save()
                else:self.finish_validation()
            self.save();self.progress('complete')
        finally:
            self.executor.shutdown(wait=True)
            if self.front is not None:self.front.close()
            self.lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['repo','rich-root','plan','cache','output','config','dino-repo']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--max-chunks',type=int,default=0);p.add_argument('--device',default='cuda')
    a=p.parse_args();Runner(a).run()

if __name__=='__main__':main()
