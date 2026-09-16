#!/usr/bin/env python3
"""Train on frozen inputs; lock selection before separate new-holdout evaluation."""
from __future__ import annotations
import argparse
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import smplx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contact_streaming.rich_assets import sha256, save_json
from contact_streaming.rich_geometry import reconstruct_humangs_geometry
from contact_streaming.rich_refinement import ResidualMLP, DifferentiableHumanGS, apply_residual
from evaluate_rich_geometry_experiments import metric

INPUTS = ['rotation', 'shape', 'head', 'expression', 'c2w']
KEYS = ['raw_body_median_m', 'raw_sole_median_m', 'body_median_m', 'sole_median_m']
VARIANTS = ['pose_parameter', 'visual_parameter', 'visual_geometry']


def load_cache(path):
    manifest = json.loads((path/'manifest.json').read_text())
    if sha256(path/'data.npz') != manifest['data_sha256']:
        raise ValueError('Cache hash mismatch')
    with np.load(path/'data.npz', allow_pickle=False) as data:
        arrays = {k:data[k] for k in data.files}
    return arrays, manifest


def make_mesh(repo):
    sys.path.insert(0, str(repo.resolve()/'src'))
    from dust3r.heads.human_gs.smplx_mesh.smplx_mesh import SMPLX_Mesh
    kwargs = {f'create_{k}':False for k in ['transl','global_orient','body_pose','left_hand_pose',
        'right_hand_pose','jaw_pose','leye_pose','reye_pose','betas','expression']}
    layer = smplx.create(str(repo/'src/models'), model_type='smplx', gender='neutral',
                        ext='npz', use_pca=False, flat_hand_mean=True, **kwargs).cuda().eval()
    mesh = SMPLX_Mesh.__new__(SMPLX_Mesh)
    mesh.layer={'neutral':layer}; mesh.device=torch.device('cuda'); mesh.expr_param_dim=10
    return mesh


def feature_array(data, variant):
    return data['pose'] if variant == 'pose_parameter' else np.concatenate([data['pose'],data['token']],axis=1)


def tensors(data):
    return {k:torch.from_numpy(data[k]).float().cuda() for k in INPUTS+['target','head_gt','y']}


@torch.no_grad()
def exact_vertices(mesh, inputs, residual):
    result=[]
    for i in range(len(residual)):
        one={k:inputs[k][i:i+1] for k in INPUTS}
        rot,beta,head=apply_residual(one,residual[i:i+1])
        res=dict(smpl_rotmat=rot[None],smpl_shape=beta[None],smpl_transl=head[None],
                 smpl_expression=one['expression'][None])
        result.append(reconstruct_humangs_geometry(res,mesh,one['c2w'][0])[0][0].cpu().numpy())
    return np.array(result)


def audit(data, predicted, roi):
    rows=[]
    for ci in sorted(set(data['clip'].tolist())):
        ids=np.flatnonzero(data['clip']==ci)
        rows.append(dict(clip=int(ci),sequence=str(data['sequence'][ids[0]]),
                         **metric(predicted[ids],data['target'][ids],roi)))
    return dict(clips=rows, mean={k:float(np.mean([r[k] for r in rows])) for k in KEYS},
                median={k:float(np.median([r[k] for r in rows])) for k in KEYS},
                passes_2cm=sum(r['passes_2cm'] for r in rows))


def subset(data, ids):
    return {k:v[ids] for k,v in data.items()}


def gradient_audit(mesh, data, roi):
    chosen=np.linspace(0,len(data['pose'])-1,8,dtype=int)
    sample=subset(data,chosen); inputs=tensors(sample)
    residual=torch.zeros((8,79),device='cuda')
    exact=exact_vertices(mesh,inputs,residual)
    zero=float(np.max(np.abs(exact-sample['base'])))
    if zero > 1e-4:
        raise ValueError(f'Released zero reconstruction exceeds 0.1mm: {zero}')
    body=DifferentiableHumanGS(mesh)
    with torch.no_grad():
        batched,_=body(inputs,residual)
    difference=float(np.max(np.abs(batched.cpu().numpy()-exact)))
    # This is a training-kernel audit, not permission to relax the released export.
    if difference > .005:
        raise ValueError(f'Batched training geometry deviates by >5mm: {difference}')
    torch.backends.cuda.matmul.allow_tf32=False
    exact_fp32=exact_vertices(mesh,inputs,residual)
    with torch.no_grad():batched_fp32,_=body(inputs,residual)
    fp32_difference=float(np.max(np.abs(batched_fp32.cpu().numpy()-exact_fp32)))
    if fp32_difference>1e-4:
        raise ValueError(f'Batched/per-frame FP32 geometry differs by >0.1mm: {fp32_difference}')
    one={k:v[:1] for k,v in inputs.items()}
    delta=torch.full((1,79),.01,device='cuda',requires_grad=True)
    ids=torch.tensor(roi.reshape(-1),device='cuda')
    def scalar(d):
        vertices,_=body(one,d)
        return vertices[:,ids].square().mean()
    loss=scalar(delta); gradient=torch.autograd.grad(loss,delta)[0]
    checked=[]
    for dimension in [0,21,24,66,76,78]:
        step=.001
        plus=delta.detach().clone();minus=plus.clone()
        plus[0,dimension]+=step;minus[0,dimension]-=step
        with torch.no_grad(): numerical=float((scalar(plus)-scalar(minus))/(2*step))
        analytical=float(gradient[0,dimension]);error=abs(numerical-analytical)
        if error > .005 + .03*abs(analytical):
            raise ValueError(f'Gradient mismatch at {dimension}: {analytical}, {numerical}')
        checked.append(dict(dimension=dimension,autograd=analytical,finite_difference=numerical))
    # Starting at zero must be safe for all dimensions, including SO(3) updates.
    z=torch.zeros((1,79),device='cuda',requires_grad=True)
    g=torch.autograd.grad(scalar(z),z)[0]
    if not torch.isfinite(g).all():
        raise ValueError('Nonfinite zero-initialization gradient')
    torch.backends.cuda.matmul.allow_tf32=True
    return dict(released_zero_max_abs_m=zero,batched_training_max_abs_m=difference,
                batched_vs_per_frame_fp32_max_abs_m=fp32_difference,training_precision='FP32; evaluation matches released TF32',
                finite_difference=checked,zero_gradient_finite=True)


def train(a, data, manifest, mesh, roi):
    a.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(a.protocol.read_text())
    old=json.loads(a.inner_split.read_text()); tune_names=set(old['inner_tune_sequences'])
    fit_ids=np.flatnonzero(~np.isin(data['sequence'],list(tune_names)))
    tune_ids=np.flatnonzero(np.isin(data['sequence'],list(tune_names)))
    if not len(fit_ids) or not len(tune_ids):
        raise ValueError('Empty inner split')
    fit_data=subset(data,fit_ids); tune_data=subset(data,tune_ids)
    gpu=tensors(data); tune_gpu={k:v[tune_ids] for k,v in gpu.items()}
    baseline=audit(tune_data,tune_data['base'],roi)
    checks=gradient_audit(mesh,data,roi)
    save_json(a.output/'geometry_checks.json',checks)
    print(json.dumps(dict(stage='geometry_checks',**checks)),flush=True)
    body=DifferentiableHumanGS(mesh)
    vertex_ids=torch.tensor(np.sort(np.random.default_rng(42).choice(10475,512,replace=False)),device='cuda')
    sole_ids=torch.tensor(roi.reshape(-1),device='cuda')
    all_runs=[]; start=time.time()
    for variant in protocol['variants']:
        X=feature_array(data,variant).astype(np.float32)
        mean=X[fit_ids].mean(0); std=X[fit_ids].std(0);std[std<.01]=1
        x=torch.from_numpy(X).cuda()
        for seed in protocol['seeds']:
            torch.manual_seed(seed);np.random.seed(seed)
            model=ResidualMLP(X.shape[1],mean,std,protocol['hidden'],protocol['dropout']).cuda()
            optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay'])
            trace=[]; best=None; best_state=None
            for epoch in range(1,protocol['epochs']+1):
                torch.backends.cuda.matmul.allow_tf32=False
                model.train();loss_total=0.;count=0
                order=fit_ids[np.random.permutation(len(fit_ids))]
                for off in range(0,len(order),protocol['batch_size']):
                    index=order[off:off+protocol['batch_size']]
                    inputs={k:v[index] for k,v in gpu.items()};residual=model(x[index])
                    param=F.smooth_l1_loss(residual/model.output_scale,inputs['y']/model.output_scale)
                    if variant.endswith('geometry'):
                        vertices,head=body(inputs,residual)
                        relative=vertices-head[:,None];truth=inputs['target']-inputs['head_gt'][:,None]
                        loss=F.smooth_l1_loss(relative[:,vertex_ids],truth[:,vertex_ids],beta=.02)
                        loss=loss+3*F.smooth_l1_loss(relative[:,sole_ids],truth[:,sole_ids],beta=.02)
                        depth=inputs['head'][:,2:3].clamp_min(.5)
                        loss=loss+F.smooth_l1_loss((head-inputs['head_gt'])/depth,torch.zeros_like(head),beta=.02)+.01*param
                    else:
                        loss=param
                    loss=loss+.0005*(residual/model.output_scale).square().mean()
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    optimizer.zero_grad();loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step();loss_total+=float(loss.detach())*len(index);count+=len(index)
                if epoch%10==0 or epoch==1:
                    torch.backends.cuda.matmul.allow_tf32=True
                    model.eval()
                    with torch.no_grad():pred=exact_vertices(mesh,tune_gpu,model(x[tune_ids]))
                    results=audit(tune_data,pred,roi)
                    score=float(np.mean([results['mean'][k]/baseline['mean'][k] for k in KEYS]))
                    row=dict(epoch=epoch,loss=loss_total/count,score=score,inner_tune=results)
                    trace.append(row)
                    if best is None or score<best['score']:
                        best=row;best_state=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()})
                    print(json.dumps(dict(stage='train',variant=variant,seed=seed,epoch=epoch,loss=loss_total/count,score=score,metrics=results['median'],elapsed_s=time.time()-start)),flush=True)
            torch.backends.cuda.matmul.allow_tf32=True
            model.load_state_dict(best_state);model.eval()
            with torch.no_grad():fit_prediction=exact_vertices(mesh,{k:v[fit_ids] for k,v in gpu.items()},model(x[fit_ids]))
            fit_report=audit(fit_data,fit_prediction,roi)
            name=f'{variant}_seed{seed}';weight=a.output/(name+'.pt')
            torch.save(dict(state=best_state,variant=variant,dimension=X.shape[1],hidden=protocol['hidden'],dropout=protocol['dropout']),weight)
            run=dict(name=name,variant=variant,seed=seed,best_epoch=best['epoch'],score=best['score'],
                     inner_tune=best['inner_tune'],fit=fit_report,trace=trace,weight=weight.name,sha256=sha256(weight))
            save_json(a.output/(name+'.json'),run);all_runs.append(run)
            save_json(a.output/'progress.json',dict(completed_runs=len(all_runs),total_runs=len(protocol['variants'])*len(protocol['seeds'])))
    best=min(all_runs,key=lambda r:r['score'])
    selection=best['name'] if best['score']<1 else 'identity'
    report=dict(status='selection_locked',selected=selection,runs=all_runs,inner_baseline=baseline,
        fit_samples=len(fit_ids),tune_samples=len(tune_ids),fit_sequences=sorted(set(fit_data['sequence'])),
        tune_sequences=sorted(tune_names),train_cache_sha256=manifest['data_sha256'],train_subjects=manifest['subjects'],
        protocol_sha256=sha256(a.protocol),script_sha256=sha256(Path(__file__)),helper_sha256=sha256(ROOT/'contact_streaming/rich_refinement.py'),
        geometry_checks=checks,training_ready=False,elapsed_s=time.time()-start)
    save_json(a.output/'selection.json',report)
    (a.output/'source.py').write_text(Path(__file__).read_text())
    (a.output/'rich_refinement.py').write_text((ROOT/'contact_streaming/rich_refinement.py').read_text())
    print(json.dumps(dict(stage='selection_locked',selected=selection,sha256=sha256(a.output/'selection.json'))),flush=True)


def evaluate(a,data,manifest,mesh,roi):
    selected=json.loads((a.training_run/'selection.json').read_text())
    if selected['status']!='selection_locked' or set(manifest['subjects'])&set(selected['train_subjects']):
        raise ValueError('Selection unlocked or subject leakage')
    a.output.mkdir(parents=True,exist_ok=False)
    inputs=tensors(data); baseline=audit(data,data['base'],roi)
    zero=exact_vertices(mesh,inputs,torch.zeros((len(data['pose']),79),device='cuda'))
    error=float(np.max(np.abs(zero-data['base'])))
    if error>1e-4:raise ValueError(f'Holdout zero mismatch {error}')
    rows=[]
    for run in selected['runs']:
        path=a.training_run/run['weight']
        if sha256(path)!=run['sha256']:raise ValueError('Selected weights changed')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        state=payload['state'];model=ResidualMLP(payload['dimension'],state['mean'],state['std'],payload['hidden'],payload['dropout']).cuda()
        model.load_state_dict(state);model.eval();x=torch.from_numpy(feature_array(data,run['variant'])).float().cuda()
        with torch.no_grad():pred=exact_vertices(mesh,inputs,model(x))
        row=dict(name=run['name'],**audit(data,pred,roi));rows.append(row)
        print(json.dumps(dict(name=row['name'],holdout=row['median'],passes=row['passes_2cm'])),flush=True)
    result=dict(status='complete',selected=selected['selected'],baseline=baseline,runs=rows,
        selection_sha256=sha256(a.training_run/'selection.json'),holdout_cache_sha256=manifest['data_sha256'],
        declared_clips=manifest['declared_clips'],detection_failures=manifest['failures'],usable_samples=manifest['samples'],
        zero_reproduction_max_abs_m=error,training_ready=False,
        note='All predeclared trials reported; no selection using these new holdout errors')
    save_json(a.output/'report.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['check','train','evaluate'])
    for key in ['cache','repo','rich-root','output']:
        p.add_argument('--'+key,type=Path,required=True)
    for key in ['protocol','inner-split','training-run']:
        p.add_argument('--'+key,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    data,manifest=load_cache(a.cache);mesh=make_mesh(a.repo)
    roi=np.array(json.loads((a.rich_root/'processed/supervision_v2/roi.json').read_text())['vertex_ids'])
    if a.stage=='check':
        save_json(a.output,gradient_audit(mesh,data,roi));print(a.output.read_text())
    elif a.stage=='train':train(a,data,manifest,mesh,roi)
    else:evaluate(a,data,manifest,mesh,roi)

if __name__=='__main__':main()
