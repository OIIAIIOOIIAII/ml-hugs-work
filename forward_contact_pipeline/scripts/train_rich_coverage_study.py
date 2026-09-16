#!/usr/bin/env python3
"""Fixed-update data coverage x kinematic architecture experiments."""
import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from contact_streaming.rich_assets import save_json,sha256
from contact_streaming.rich_refinement import ResidualMLP,DifferentiableHumanGS
from contact_streaming.rich_kinematic import KinematicResidualNet
from train_rich_neural_refiner import (load_cache,make_mesh,tensors,exact_vertices,audit,subset,gradient_audit,KEYS)


def camera_ids(data):
    return np.array([int(re.search(r'/cam_(\d+)/',str(s)).group(1)) for s in data['sample_id']])


def assemble(old,new):
    held=['003','020']
    mask=~np.isin(old['subject'],held)&(camera_ids(old)!=4)
    small=subset(old,np.flatnonzero(mask)); seen=set(small['sample_id'].tolist())
    new_fit=(~np.isin(new['subject'],held))&(camera_ids(new)!=4)
    excluded=[];kept=[]
    for ci in sorted(set(new['clip'][new_fit])):
        ids=np.flatnonzero((new['clip']==ci)&new_fit)
        if set(new['sample_id'][ids])&seen:excluded.append(int(ci))
        else:kept.extend(ids.tolist());seen.update(new['sample_id'][ids])
    extra=subset(new,np.array(kept,dtype=int));extra['clip']=extra['clip']+100000
    large={k:np.concatenate([small[k],extra[k]]) for k in small}
    tune=subset(new,np.flatnonzero(np.isin(new['subject'],held)&(camera_ids(new)==4)))
    if not len(small['pose']) or not len(tune['pose']) or len(large['pose'])<=len(small['pose']):
        raise ValueError('Invalid data budgets')
    if set(tune['subject'])&set(large['subject']) or 4 in camera_ids(large):
        raise ValueError('Subject/camera leakage')
    if len(set(large['sample_id']))!=len(large['sample_id']):raise ValueError('Duplicate samples')
    return dict(small=small,expanded=large),tune,excluded


def make_model(architecture,x,ids=None,hidden=128,dropout=.1,state=None,device='cuda'):
    if architecture not in ('mlp','kinematic'):raise ValueError('Unknown architecture')
    if state is None:
        values=x if ids is None else x[ids]
        mean=values.mean(0);std=values.std(0);std[std<.01]=1
    else:mean=state['mean'];std=state['std']
    klass=ResidualMLP if architecture=='mlp' else KinematicResidualNet
    model=klass(x.shape[1],mean,std,hidden,dropout).to(device)
    if state is not None:model.load_state_dict(state)
    return model


def features(data):return np.concatenate([data['pose'],data['token']],axis=1)


def scaled_score(result,base):return float(np.mean([result['mean'][k]/base['mean'][k] for k in KEYS]))


def check_plan_hash(manifest,expected):
    """Require the actual cache to descend from the predeclared sample plan."""
    plans=[value for key,value in manifest['provenance'].items() if key.endswith('_plan.json')]
    if plans != [expected]:raise ValueError('Cache does not match the declared plan')


def train(a):
    a.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(a.protocol.read_text())
    old,om=load_cache(a.small_cache);new,nm=load_cache(a.expanded_cache)
    check_plan_hash(nm,protocol['development_plan_sha256'])
    budgets,tune,duplicates=assemble(old,new)
    roi=np.array(json.loads((a.rich_root/'processed/supervision_v2/roi.json').read_text())['vertex_ids'])
    mesh=make_mesh(a.repo);body=DifferentiableHumanGS(mesh)
    checks=gradient_audit(mesh,new,roi);save_json(a.output/'geometry_checks.json',checks)
    tune_gpu=tensors(tune);xt=torch.from_numpy(features(tune)).float().cuda()
    baseline=audit(tune,tune['base'],roi)
    budgets_info={key:dict(samples=len(d['pose']),body_frames=len(set(zip(d['sequence'],d['frame_id']))),
        sequences=sorted(set(d['sequence'])),subjects=sorted(set(d['subject'])),cameras=sorted(set(camera_ids(d).tolist()))) for key,d in budgets.items()}
    save_json(a.output/'data_audit.json',dict(budgets=budgets_info,tune_samples=len(tune['pose']),tune_subjects=sorted(set(tune['subject'])),
        tune_sequences=sorted(set(tune['sequence'])),tune_camera=4,new_cache_failures=nm['failures'],duplicate_containing_clips_dropped=duplicates))
    print(json.dumps(dict(stage='data_audit',budgets=budgets_info,tune_samples=len(tune['pose']))),flush=True)
    vertex_ids=torch.tensor(np.sort(np.random.default_rng(42).choice(10475,512,replace=False)),device='cuda')
    sole_ids=torch.tensor(roi.reshape(-1),device='cuda')
    runs=[];start=time.time()
    for budget in protocol['budgets']:
        data=budgets[budget];gpu=tensors(data);x_np=features(data);x=torch.from_numpy(x_np).float().cuda()
        _,inverse,counts=np.unique(data['sequence'],return_inverse=True,return_counts=True)
        probability=(1/counts[inverse]);probability=probability/probability.sum()
        for architecture in protocol['architectures']:
            for seed in protocol['seeds']:
                torch.manual_seed(seed);rng=np.random.default_rng(seed)
                model=make_model(architecture,x_np,hidden=protocol['hidden'],dropout=protocol['dropout'])
                optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay'])
                best=None;state=None;trace=[];running_loss=0
                for step in range(1,protocol['steps']+1):
                    torch.backends.cuda.matmul.allow_tf32=False;model.train()
                    ids=rng.choice(len(x),size=protocol['batch_size'],replace=True,p=probability)
                    inputs={k:v[ids] for k,v in gpu.items()};residual=model(x[ids])
                    vertices,head=body(inputs,residual)
                    relative=vertices-head[:,None];truth=inputs['target']-inputs['head_gt'][:,None]
                    loss=F.smooth_l1_loss(relative[:,vertex_ids],truth[:,vertex_ids],beta=.02)
                    loss=loss+3*F.smooth_l1_loss(relative[:,sole_ids],truth[:,sole_ids],beta=.02)
                    loss=loss+F.smooth_l1_loss((head-inputs['head_gt'])/inputs['head'][:,2:3].clamp_min(.5),torch.zeros_like(head),beta=.02)
                    loss=loss+.01*F.smooth_l1_loss(residual/model.output_scale,inputs['y']/model.output_scale)
                    loss=loss+.0005*(residual/model.output_scale).square().mean()
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
                    running_loss+=float(loss.detach())
                    if step in protocol['checkpoints']:
                        torch.backends.cuda.matmul.allow_tf32=True;model.eval()
                        with torch.no_grad():prediction=model(xt);verts=exact_vertices(mesh,tune_gpu,prediction)
                        metrics=audit(tune,verts,roi);score=scaled_score(metrics,baseline)
                        row=dict(step=step,score=score,loss=running_loss/step,metrics=metrics);trace.append(row)
                        if best is None or score<best['score']:
                            best=row;state=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()})
                        print(json.dumps(dict(stage='train',budget=budget,architecture=architecture,seed=seed,step=step,score=score,median=metrics['median'],elapsed_s=time.time()-start)),flush=True)
                model.load_state_dict(state);model.eval();torch.backends.cuda.matmul.allow_tf32=True
                with torch.no_grad():prediction=model(xt)
                strengths=[]
                for strength in [0.,.25,.5,1.]:
                    metrics=baseline if strength==0 else audit(tune,exact_vertices(mesh,tune_gpu,prediction*strength),roi)
                    ratios=[metrics['mean'][k]/baseline['mean'][k] for k in KEYS]
                    strengths.append(dict(strength=strength,eligible=bool(max(ratios)<=1.01),score=float(np.mean(ratios)),metrics=metrics))
                guarded=min([s for s in strengths if s['eligible']],key=lambda s:s['score'])
                name=f'{budget}_{architecture}_seed{seed}';weight=a.output/(name+'.pt')
                torch.save(dict(state=state,architecture=architecture,hidden=protocol['hidden'],dropout=protocol['dropout'],dimension=x.shape[1]),weight)
                run=dict(name=name,budget=budget,architecture=architecture,seed=seed,best_step=best['step'],full_strength=best['metrics'],score=best['score'],
                    guarded=guarded,strength_search=strengths,trace=trace,weight=weight.name,weight_sha256=sha256(weight),parameters=sum(p.numel() for p in model.parameters()))
                save_json(a.output/(name+'.json'),run);runs.append(run)
                save_json(a.output/'progress.json',dict(completed=len(runs),total=8))
    chosen=min(runs,key=lambda r:r['guarded']['score'])
    result=dict(status='selection_locked',selected=chosen['name'],selected_strength=chosen['guarded']['strength'],runs=runs,
        tune_baseline=baseline,data_budgets=budgets_info,fit_subjects=budgets_info['expanded']['subjects'],
        small_cache_sha256=om['data_sha256'],expanded_cache_sha256=nm['data_sha256'],protocol_sha256=sha256(a.protocol),
        script_sha256=sha256(Path(__file__)),graph_sha256=sha256(ROOT/'contact_streaming/rich_kinematic.py'),
        confirmation_plan_sha256=protocol['confirmation_plan_sha256'],tune_subjects=sorted(set(tune['subject'])),
        training_ready=False,elapsed_s=time.time()-start)
    save_json(a.output/'selection.json',result)
    (a.output/'source.py').write_text(Path(__file__).read_text())
    for path in [a.protocol,ROOT/'contact_streaming/rich_kinematic.py',ROOT/'contact_streaming/rich_refinement.py',Path(__file__).with_name('train_rich_neural_refiner.py')]:
        (a.output/path.name).write_text(path.read_text())
    print(json.dumps(dict(stage='selection_locked',name=chosen['name'],strength=chosen['guarded']['strength'])),flush=True)


def evaluate(a):
    selection=json.loads((a.training_run/'selection.json').read_text())
    if selection['status']!='selection_locked':raise ValueError('Selection not locked')
    data,manifest=load_cache(a.cache)
    check_plan_hash(manifest,selection['confirmation_plan_sha256'])
    if set(data['subject'])&(set(selection['fit_subjects'])|set(selection['tune_subjects'])):raise ValueError('Subject leakage')
    a.output.mkdir(parents=True,exist_ok=False);mesh=make_mesh(a.repo);gpu=tensors(data)
    roi=np.array(json.loads((a.rich_root/'processed/supervision_v2/roi.json').read_text())['vertex_ids'])
    base=audit(data,data['base'],roi)
    zero=exact_vertices(mesh,gpu,torch.zeros((len(data['pose']),79),device='cuda'))
    error=float(np.max(np.abs(zero-data['base'])))
    if error>1e-4:raise ValueError('Zero reconstruction failed')
    x_np=features(data);x=torch.from_numpy(x_np).float().cuda();results=[]
    for run in selection['runs']:
        file=a.training_run/run['weight']
        if sha256(file)!=run['weight_sha256']:raise ValueError('Weight hash changed')
        payload=torch.load(file,map_location='cpu',weights_only=True)
        model=make_model(payload['architecture'],x_np,hidden=payload['hidden'],dropout=payload['dropout'],state=payload['state']);model.eval()
        with torch.no_grad():prediction=model(x)
        full=audit(data,exact_vertices(mesh,gpu,prediction),roi)
        strength=run['guarded']['strength']
        guarded=base if strength==0 else (full if strength==1 else audit(data,exact_vertices(mesh,gpu,prediction*strength),roi))
        results.append(dict(name=run['name'],budget=run['budget'],architecture=run['architecture'],seed=run['seed'],full=full,guarded=guarded,strength=strength))
        print(json.dumps(dict(stage='confirm',name=run['name'],strength=strength,full=full['median'],guarded=guarded['median'])),flush=True)
    save_json(a.output/'report.json',dict(status='complete',selected=selection['selected'],baseline=base,runs=results,
        declared_clips=manifest['declared_clips'],failures=manifest['failures'],samples=len(data['pose']),
        selection_sha256=sha256(a.training_run/'selection.json'),confirmation_cache_sha256=manifest['data_sha256'],zero_max_abs_m=error,training_ready=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['train','evaluate'])
    for key in ['repo','rich-root','output']:p.add_argument('--'+key,type=Path,required=True)
    for key in ['small-cache','expanded-cache','protocol','training-run','cache']:p.add_argument('--'+key,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    if a.stage=='train':train(a)
    else:evaluate(a)
if __name__=='__main__':main()
