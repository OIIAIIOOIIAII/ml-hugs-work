#!/usr/bin/env python3
"""Compare frozen physical-K predictions with their original planned controls."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from contact_streaming.rich_assets import sha256,save_json
from evaluate_rich_geometry_experiments import metric
KEYS=['raw_body_median_m','raw_sole_median_m','body_median_m','sole_median_m']


def predictions(directory,record,samples):
    if [f['sample_id'] for f in record['frames']]!=[s['sample_id'] for s in samples]:raise ValueError('Samples changed')
    if any(f['humans']!=1 for f in record['frames']):return None,'detection_count'
    vertices=[];ids=[]
    for f in record['frames']:
        if sha256(directory/f['file'])!=f['sha256']:raise ValueError('Prediction changed')
        with np.load(directory/f['file'],allow_pickle=False) as d:
            vertices.append(d['vertices_camera'][0]);ids.append(d['smpl_id'].reshape(-1).tolist())
    if any(i!=ids[0] for i in ids):return None,'identity_switch'
    return np.array(vertices),None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['rich-root','plan','features','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();plan=json.loads(a.plan.read_text());sensor=json.loads((a.features/'manifest.json').read_text())
    if sensor['provenance']['plan_sha256']!=sha256(a.plan) or len(sensor['clips'])!=len(plan['clips']):raise ValueError('Plan mismatch')
    cameras=json.loads((a.rich_root/'processed/dataset_v1/cameras.json').read_text())['cameras']
    roi=np.array(json.loads((a.rich_root/'processed/supervision_v2/roi.json').read_text())['vertex_ids'])
    cache={};manifests={};rows=[]
    fields=['smpl_rotmat','smpl_shape','smpl_transl','smpl_query','vertices_camera']
    intervention=dict(frames=0,changed_K_frames=0,shape_mismatches=0,max_abs_by_field={k:0. for k in fields})
    for ci,c in enumerate(plan['clips']):
        truth=[]
        for s in c['frames']:
            ann=s['annotation'];key=ann['path']
            if key not in cache:
                if sha256(a.rich_root/key)!=ann['sha256']:raise ValueError('GT changed')
                with np.load(a.rich_root/key,allow_pickle=False) as d:cache[key]=(d['vertices_multicam'],d['frame_ids'])
            v,ids=cache[key]
            if ids[ann['row']]!=s['frame_id']:raise ValueError('GT frame mismatch')
            e=np.asarray(cameras[s['camera_key']]['CameraMatrix']);truth.append(v[ann['row']]@e[:,:3].T+e[:,3])
        truth=np.array(truth);base_dir=Path(c['baseline_features'])
        if str(base_dir) not in manifests:manifests[str(base_dir)]=json.loads((base_dir/'manifest.json').read_text())
        baseline=manifests[str(base_dir)]['clips'][c['baseline_clip']]
        for before,after in zip(baseline['frames'],sensor['clips'][ci]['frames']):
            if before['sample_id']!=after['sample_id']:raise ValueError('Intervention frame mismatch')
            if sha256(base_dir/before['file'])!=before['sha256'] or sha256(a.features/after['file'])!=after['sha256']:raise ValueError('Intervention feature changed')
            with np.load(base_dir/before['file'],allow_pickle=False) as x,np.load(a.features/after['file'],allow_pickle=False) as y:
                k=y['sensor_K_mhmr'][0];size=after['transform']['padded_size_wh'][0]
                default=np.array([[size/(2*np.tan(np.pi/6)),0,size//2],[0,size/(2*np.tan(np.pi/6)),size//2],[0,0,1.]])
                intervention['frames']+=1;intervention['changed_K_frames']+=int(not np.allclose(k,default,atol=1e-4))
                for key in fields:
                    if x[key].shape!=y[key].shape:intervention['shape_mismatches']+=1;continue
                    intervention['max_abs_by_field'][key]=max(intervention['max_abs_by_field'][key],float(np.max(np.abs(x[key]-y[key]),initial=0)))
        row=dict(clip=ci,sequence=c['sequence'],role=c['study_role'])
        for name,directory,record in [('baseline',base_dir,baseline),('sensor',a.features,sensor['clips'][ci])]:
            pred,failure=predictions(directory,record,c['frames'])
            row[name]=dict(failure=failure,metrics=None if failure else metric(pred,truth,roi))
        rows.append(row)
    summaries={}
    for role in ['tune','confirm']:
        group=[r for r in rows if r['role']==role];summary=dict(declared=len(group))
        common=[r for r in group if not r['baseline']['failure'] and not r['sensor']['failure']]
        for name in ['baseline','sensor']:
            valid=[r[name]['metrics'] for r in group if not r[name]['failure']]
            summary[name]=dict(measured=len(valid),passes_2cm=sum(r['passes_2cm'] for r in valid),
                median={k:float(np.median([r[k] for r in valid])) if valid else None for k in KEYS})
        summary['paired_clips']=len(common)
        summary['paired_mean_delta_cm']={k:float(np.mean([r['sensor']['metrics'][k]-r['baseline']['metrics'][k] for r in common])*100) if common else None for k in KEYS}
        summaries[role]=summary
    save_json(a.output,dict(status='complete',summary=summaries,clips=rows,plan_sha256=sha256(a.plan),sensor_manifest_sha256=sha256(a.features/'manifest.json'),
        baseline_manifest_sha256={s:sha256(Path(s)/'manifest.json') for s in manifests},training_ready=False,
        intervention_audit=intervention,script_sha256=sha256(Path(__file__)),
        note='Supplemental frozen frontend intervention declared before primary confirmation; physical sensor K is extra deployment metadata, GT body/contact only used for evaluation.'))
    print(json.dumps(summaries))


if __name__=='__main__':main()
