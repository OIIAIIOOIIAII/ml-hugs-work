#!/usr/bin/env python3
"""Small train-only SMPL parameter correction study with sequence-disjoint selection.

Frozen upstream tokens/parameters are inputs; GT is training/audit-only. This is
geometry calibration research, not permission to train the final contact model.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
import roma
import smplx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from contact_streaming.rich_assets import sha256,save_json
from contact_streaming.rich_geometry import reconstruct_humangs_geometry
from contact_streaming.rich_supervision import atomic_npz


def fit_ridge(x,y,alpha):
 xm=x.mean(0);xs=x.std(0);xs[xs<1e-4]=1
 ym=y.mean(0);ys=y.std(0);ys[ys<1e-4]=1
 z=(x-xm)/xs;target=(y-ym)/ys
 w=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@target)
 return dict(x_mean=xm,x_std=xs,y_mean=ym,y_std=ys,weight=w)

def infer_ridge(model,x):return ((x-model['x_mean'])/model['x_std']@model['weight'])*model['y_std']+model['y_mean']

def capped_vectors(x,maxnorm):
 return x*np.minimum(1,maxnorm/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-9))


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['rich-root','repo','train-plan','train-features','confirmation-plan','confirmation-features','output']:
  p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--component-study',action='store_true',help='Additional train-selected component/shrinkage and oracle diagnostics')
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
 # CROCO enables TF32 globally during upstream inference. Match its saved meshes.
 if not torch.cuda.is_available():raise RuntimeError('CUDA required to reproduce upstream TF32 geometry')
 device=torch.device('cuda');torch.backends.cuda.matmul.allow_tf32=True
 (a.output/'source.py').write_text(Path(__file__).read_text())
 sys.path.insert(0,str(a.repo.resolve()/'src'))
 from dust3r.heads.human_gs.smplx_mesh.smplx_mesh import SMPLX_Mesh
 models=a.repo.resolve()/'src/models'
 neutral=smplx.create(str(models),model_type='smplx',gender='neutral',ext='npz',use_pca=False,flat_hand_mean=True,
  create_transl=False,create_global_orient=False,create_body_pose=False,create_left_hand_pose=False,create_right_hand_pose=False,
  create_jaw_pose=False,create_leye_pose=False,create_reye_pose=False,create_betas=False,create_expression=False).to(device).eval()
 mesh=SMPLX_Mesh.__new__(SMPLX_Mesh);mesh.layer={'neutral':neutral};mesh.device=device;mesh.expr_param_dim=10
 teachers={g:smplx.create(str(models),model_type='smplx',gender=g,ext='npz',use_pca=True,num_pca_comps=12).eval() for g in ['male','female']}
 with np.load(models/'smplx/SMPLX_NEUTRAL.npz',allow_pickle=False) as d:
  template=d['v_template'];basis=d['shapedirs'][:,:,:10].reshape(-1,10);pinv=np.linalg.pinv(basis.astype(np.float64))
 gender_assets={}
 for gender in ['male','female']:
  with np.load(models/f'smplx/SMPLX_{gender.upper()}.npz',allow_pickle=False) as d:gender_assets[gender]=(d['v_template'],d['shapedirs'][:,:,:10])
 cameras=json.loads((a.rich_root/'processed/dataset_v1/cameras.json').read_text())['cameras']
 roi=np.array(json.loads((a.rich_root/'processed/supervision_v2/roi.json').read_text())['vertex_ids'])
 cache={}
 def collect(planpath,featuredir):
  plan=json.loads(planpath.read_text());manifest=json.loads((featuredir/'manifest.json').read_text())
  if manifest['provenance']['plan_sha256']!=sha256(planpath):raise ValueError('Plan hash mismatch')
  if len(manifest['clips'])!=len(plan['clips']):raise ValueError('Missing feature clips')
  samples=[];failures=[]
  for ci,(clip,ref) in enumerate(zip(manifest['clips'],plan['clips'])):
   if len(clip['frames'])!=len(ref['frames']):raise ValueError('Missing feature frames')
   if any(f['humans']!=1 for f in clip['frames']):
    failures.append(dict(sequence=clip['sequence'],camera_id=clip['camera_id'],reason='detection count',frames=len(clip['frames'])));continue
   clip_samples=[];tracks=[]
   for f,s in zip(clip['frames'],ref['frames']):
    if f['sample_id']!=s['sample_id'] or sha256(featuredir/f['file'])!=f['sha256']:raise ValueError('Feature identity/hash mismatch')
    with np.load(featuredir/f['file'],allow_pickle=False) as d:
     values={k:d[k] for k in ['smpl_rotmat','smpl_shape','smpl_transl','smpl_expression','smpl_id','smpl_query','camera_c2w','vertices_camera','vertices_world']}
    tracks.append(values['smpl_id'].reshape(-1).tolist());ann=s['annotation'];key=ann['path']
    if key not in cache:
     if sha256(a.rich_root/key)!=ann['sha256']:raise ValueError('GT changed')
     with np.load(a.rich_root/key,allow_pickle=False) as d:cache[key]={k:d[k] for k in d.files if k.startswith('body_') or k in ['frame_ids','vertices_multicam']}
    gt=cache[key];row=ann['row']
    if gt['frame_ids'][row]!=s['frame_id']:raise ValueError('GT frame mismatch')
    params={k[5:]:torch.from_numpy(v[row:row+1]).float() for k,v in gt.items() if k.startswith('body_') and k!='body_pose_embedding'}
    e=np.array(cameras[s['camera_key']]['CameraMatrix']);target=gt['vertices_multicam'][row]@e[:,:3].T+e[:,3]
    with torch.no_grad():body=teachers[s['gender']](**params)
    head=body.joints[0,15].numpy()@e[:,:3].T+e[:,3]
    rgt=np.concatenate([(torch.from_numpy(e[:,:3]).float()@roma.rotvec_to_rotmat(params['global_orient']))[None].numpy(),
                        roma.rotvec_to_rotmat(params['body_pose'].reshape(21,3)).numpy()[None]],axis=1)[0]
    rp=values['smpl_rotmat'][0,0,:22]
    delta=roma.rotmat_to_rotvec(torch.from_numpy(rgt@rp.transpose(0,2,1))).numpy().reshape(-1)
    vtemp,dirs=gender_assets[s['gender']]
    shaped=vtemp+np.einsum('vci,i->vc',dirs,gt['body_betas'][row])
    neutral_beta=pinv@(shaped-template).reshape(-1)
    predhead=values['smpl_transl'][0,0];z=max(float(predhead[2]),.5)
    y=np.r_[delta,neutral_beta-values['smpl_shape'][0,0],(head-predhead)/z]
    x=np.r_[rp.reshape(-1),values['smpl_shape'].reshape(-1),predhead]
    clip_samples.append(dict(sequence=s['sequence'],split=s['split'],clip=ci,frame_id=s['frame_id'],sample_id=s['sample_id'],
      x=x,token=values['smpl_query'].reshape(-1),y=y,values=values,target=target))
   if any(t!=tracks[0] for t in tracks):failures.append(dict(sequence=clip['sequence'],camera_id=clip['camera_id'],reason='identity switch',frames=len(clip['frames'])));continue
   samples.extend(clip_samples)
   print(json.dumps(dict(stage='collect',plan=str(planpath),clip=ci,usable_samples=len(samples))),flush=True)
  return samples,failures,len(plan['clips'])
 train,trainfail,ntrain=collect(a.train_plan,a.train_features)
 confirm,confirmfail,nconfirm=collect(a.confirmation_plan,a.confirmation_features)
 print(json.dumps(dict(stage='collected',train_frames=len(train),confirmation_frames=len(confirm),train_failures=trainfail,confirmation_failures=confirmfail)),flush=True)
 if {s['sequence'] for s in train}&{s['sequence'] for s in confirm}:raise ValueError('Sequence leakage')
 names=sorted({s['sequence'] for s in train});rng=np.random.default_rng(42);rng.shuffle(names)
 tune_names=set(names[-max(3,len(names)//4):]);fit=np.array([s['sequence'] not in tune_names for s in train]);tune=~fit
 X=np.stack([s['x'] for s in train]);Z=np.stack([s['token'] for s in train]);Y=np.stack([s['y'] for s in train])
 zmean=Z[fit].mean(0);_,_,vh=np.linalg.svd(Z[fit]-zmean,full_matrices=False);basis_token=vh[:32].T
 def features(samples,kind):
  x=np.stack([s['x'] for s in samples])
  if kind=='pose_tokens':x=np.c_[x,(np.stack([s['token'] for s in samples])-zmean)@basis_token]
  return x
 def corrected(sample,prediction,bounded=True):
  v=sample['values'];rot=v['smpl_rotmat'].copy();delta=prediction[:66].reshape(22,3)
  if bounded:delta=capped_vectors(delta,.35)
  rot[0,0,:22]=roma.rotvec_to_rotmat(torch.from_numpy(delta).float()).numpy()@rot[0,0,:22]
  beta_delta=prediction[66:76];trans_delta=prediction[76:79]
  if bounded:beta_delta=np.clip(beta_delta,-2,2);trans_delta=capped_vectors(trans_delta[None],.10)[0]
  beta=v['smpl_shape']+beta_delta[None,None]
  trans=v['smpl_transl']+trans_delta*max(float(v['smpl_transl'][0,0,2]),.5)
  res={k:torch.from_numpy(arr).float().to(device) for k,arr in dict(smpl_rotmat=rot,smpl_shape=beta,smpl_transl=trans,smpl_expression=v['smpl_expression']).items()}
  with torch.no_grad():cam,_,world=reconstruct_humangs_geometry(res,mesh,torch.from_numpy(v['camera_c2w']).float().to(device))
  return cam[0].cpu().numpy(),world[0].cpu().numpy()
 def raw_score(samples,predictions):
  byclip={}
  for s,y in zip(samples,predictions):
   cam,_=corrected(s,y);err=np.linalg.norm(cam-s['target'],axis=-1)
   byclip.setdefault(s['clip'],[]).append(err)
  medians=[(np.median(v),np.median(np.array(v)[:,roi])) for v in byclip.values()]
  return float(np.mean(medians)),dict(body_median_m=float(np.median(np.array(medians)[:,0])),sole_median_m=float(np.median(np.array(medians)[:,1])))
 zero_error=max(float(np.max(np.abs(corrected(s,np.zeros(79))[0]-s['values']['vertices_camera'][0]))) for s in train[::max(1,len(train)//8)])
 print(json.dumps(dict(stage='zero_correction_check',max_abs_m=zero_error)),flush=True)
 if zero_error>1e-4:raise ValueError(f'Zero correction exceeds 0.1mm: {zero_error} m')
 tune_samples=[s for s,t in zip(train,tune) if t];search=[]
 for kind in ['pose_only','pose_tokens']:
  x=features(train,kind)
  for alpha in [1.,10.,100.,1000.]:
   model=fit_ridge(x[fit],Y[fit],alpha);prediction=infer_ridge(model,x[tune])
   score,metrics=raw_score(tune_samples,prediction)
   search.append(dict(kind=kind,alpha=alpha,score=score,metrics=metrics))
   print(json.dumps(search[-1]),flush=True)
 best=min(search,key=lambda s:s['score']);x=features(train,best['kind']);model=fit_ridge(x,Y,best['alpha'])
 # PCA itself used only inner-fit labels-free tokens. Never refit on confirmation.
 atomic_npz(a.output/'calibrator.npz',**model,token_mean=zmean,token_basis=basis_token)
 prediction=infer_ridge(model,features(confirm,best['kind']));mean_prediction=np.broadcast_to(Y.mean(0),prediction.shape)
 from evaluate_rich_geometry_experiments import metric
 rows=[];groups={}
 for i,s in enumerate(confirm):groups.setdefault(s['clip'],[]).append(i)
 for ci,indices in groups.items():
  entries=[confirm[i] for i in indices];gt=np.array([s['target'] for s in entries]);base=np.array([s['values']['vertices_camera'][0] for s in entries])
  tuned=np.array([corrected(s,prediction[i])[0] for s,i in zip(entries,indices)])
  constant=np.array([corrected(s,mean_prediction[i])[0] for s,i in zip(entries,indices)])
  rows.append(dict(sequence=entries[0]['sequence'],split=entries[0]['split'],clip=ci,
   baseline=metric(base,gt,roi),train_mean_control=metric(constant,gt,roi),learned=metric(tuned,gt,roi)))
 # Fixed-budget learning curve: do not select hyperparameters on confirmation.
 # Refit all feature statistics using only each selected training subset.
 learning_curve=[]
 for seed in [7,19,42]:
  rng=np.random.default_rng(seed);by_capture={}
  for name in sorted(names):by_capture.setdefault(name.split('_')[0],[]).append(name)
  for group in by_capture.values():rng.shuffle(group)
  order=[]
  while any(by_capture.values()):
   for group in by_capture.values():
    if group:order.append(group.pop())
  for budget in sorted(set([min(4,len(names)),min(8,len(names)),len(names)])):
   chosen_names=set(order[:budget]);mask=np.array([sample['sequence'] in chosen_names for sample in train])
   subZ=Z[mask];meanZ=subZ.mean(0);_,_,svh=np.linalg.svd(subZ-meanZ,full_matrices=False);subbasis=svh[:32].T
   def subfeatures(samples):
    out=np.stack([sample['x'] for sample in samples])
    if best['kind']=='pose_tokens':out=np.c_[out,(np.stack([sample['token'] for sample in samples])-meanZ)@subbasis]
    return out
   subsamples=[sample for sample,keep in zip(train,mask) if keep]
   submodel=fit_ridge(subfeatures(subsamples),Y[mask],best['alpha'])
   subprediction=infer_ridge(submodel,subfeatures(confirm));per_split={}
   subrows=[]
   for ci,indices in groups.items():
    entries=[confirm[i] for i in indices];gt=np.array([s['target'] for s in entries])
    corrected_vertices=np.array([corrected(s,subprediction[i])[0] for s,i in zip(entries,indices)])
    subrows.append(dict(split=entries[0]['split'],sequence=entries[0]['sequence'],**metric(corrected_vertices,gt,roi)))
   for split in ['train','val']:
    indices=[i for i,sample in enumerate(confirm) if sample['split']==split]
    splitrows=[row for row in subrows if row['split']==split]
    per_split[split]={key:float(np.median([row[key] for row in splitrows])) for key in ['raw_body_median_m','raw_sole_median_m','body_median_m','sole_median_m']}
    per_split[split]['passes_2cm']=sum(row['passes_2cm'] for row in splitrows)
   learning_curve.append(dict(seed=seed,training_sequences=budget,training_frames=int(mask.sum()),sequence_names=sorted(chosen_names),confirmation=per_split,per_clip=subrows))
   print(json.dumps({k:v for k,v in learning_curve[-1].items() if k!='per_clip'}),flush=True)
 summary={}
 for split in ['train','val']:
  chosen=[r for r in rows if r['split']==split];summary[split]={}
  for method in ['baseline','train_mean_control','learned']:
   summary[split][method]={k:float(np.median([r[method][k] for r in chosen])) for k in ['raw_body_median_m','raw_sole_median_m','body_median_m','sole_median_m']}
   summary[split][method]['passes_2cm']=sum(r[method]['passes_2cm'] for r in chosen)
 # Optional second-stage diagnostic, declared before its measurements are read.
 components=None
 if a.component_study:
  keys=['raw_body_median_m','raw_sole_median_m','body_median_m','sole_median_m']
  def audit_predictions(samples,preds=None,bounded=True):
   grouped={}
   for i,sample in enumerate(samples):grouped.setdefault(sample['clip'],[]).append(i)
   result=[]
   for ci,indices in grouped.items():
    entries=[samples[i] for i in indices]
    cam=np.array([s['values']['vertices_camera'][0] if preds is None else corrected(s,preds[i],bounded)[0] for s,i in zip(entries,indices)])
    gt=np.array([s['target'] for s in entries])
    result.append(dict(sequence=entries[0]['sequence'],split=entries[0]['split'],**metric(cam,gt,roi)))
   aggregate={key:float(np.mean([r[key] for r in result])) for key in keys}
   return result,aggregate
  # Measure fit residual and unseen train sequences separately to expose overfit.
  fit_samples=[s for s,t in zip(train,fit) if t]
  inner=fit_ridge(x[fit],Y[fit],best['alpha'])
  inner_pred=infer_ridge(inner,x[tune]);inner_mean=np.broadcast_to(Y[fit].mean(0),inner_pred.shape)
  fit_rows,fit_metrics=audit_predictions(fit_samples,infer_ridge(inner,x[fit]))
  fit_base,fit_base_metrics=audit_predictions(fit_samples)
  tune_base,tune_base_metrics=audit_predictions(tune_samples)
  candidates=[dict(source='zero',scope='all',strength=0.,eligible=True,score=1.,metrics=tune_base_metrics)]
  masks={}
  for scope,start,stop in [('all',0,79),('rotation',0,66),('shape',66,76),('translation',76,79)]:
   mask=np.zeros(79);mask[start:stop]=1;masks[scope]=mask
  for source,preds in [('learned',inner_pred),('train_mean',inner_mean)]:
   for scope,mask in masks.items():
    for strength in [.25,.5,1.]:
     _,values=audit_predictions(tune_samples,preds*mask*strength)
     ratios=[values[key]/max(tune_base_metrics[key],1e-8) for key in keys]
     candidates.append(dict(source=source,scope=scope,strength=strength,eligible=bool(max(ratios)<=1.01),score=float(np.mean(ratios)),metrics=values))
     print(json.dumps(dict(stage='component_search',**candidates[-1])),flush=True)
  chosen=min([c for c in candidates if c['eligible']],key=lambda c:c['score'])
  if chosen['source']=='zero':component_pred=np.zeros_like(prediction)
  else:component_pred=(prediction if chosen['source']=='learned' else mean_prediction)*masks[chosen['scope']]*chosen['strength']
  component_rows,_=audit_predictions(confirm,component_pred)
  oracle=np.stack([s['y'] for s in confirm]);oracle_rows={}
  for scope,mask in masks.items():
   for bounded in [True,False]:
    label=scope+('_bounded' if bounded else '_unbounded')
    oracle_rows[label],_=audit_predictions(confirm,oracle*mask,bounded=bounded)
  components=dict(selection='Train-only: mean normalized raw+aligned body+sole, each <=1.01x baseline, zero fallback',
   selected=chosen,search=candidates,confirmation=component_rows,
   inner_fit=dict(baseline=fit_base_metrics,learned=fit_metrics,clips=len(fit_rows)),inner_tune_baseline=tune_base_metrics,
   oracle_not_deployable=oracle_rows)
 output=dict(status='complete',purpose='geometry correction experiment; final contact training remains gated',
  training_frames=len(train),training_sequences=names,inner_tune_sequences=sorted(tune_names),
  train_detection_failures=trainfail,confirmation_detection_failures=confirmfail,declared_train_clips=ntrain,declared_confirmation_clips=nconfirm,
  selected=best,search=search,component_study=components,confirmation_clips=rows,summary=summary,learning_curve=learning_curve,zero_correction_reproduction_max_abs_m=zero_error,
  execution=dict(device=str(device),torch_version=torch.__version__,tf32=torch.backends.cuda.matmul.allow_tf32,script_sha256=sha256(Path(__file__))),
  bounds=dict(rotation_rad=.35,beta_component=2,translation_fraction_of_depth=.10),
  training_plan_sha256=sha256(a.train_plan),confirmation_plan_sha256=sha256(a.confirmation_plan),
  train_manifest_sha256=sha256(a.train_features/'manifest.json'),confirmation_manifest_sha256=sha256(a.confirmation_features/'manifest.json'),
  model_sha256=sha256(a.output/'calibrator.npz'),training_ready=False,
  asset_sha256={str(path):sha256(path) for path in [a.rich_root/'processed/dataset_v1/cameras.json',a.rich_root/'processed/supervision_v2/roi.json',*[models/f'smplx/SMPLX_{g}.npz' for g in ['NEUTRAL','MALE','FEMALE']]]})
 save_json(a.output/'report.json',output);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
