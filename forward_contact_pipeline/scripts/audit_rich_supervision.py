#!/usr/bin/env python3
"""Independently read all RICH sole targets and verify full frame/view associations."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from contact_streaming.rich_assets import sha256, save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rich-root',type=Path,required=True); p.add_argument('--supervision',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    root=a.rich_root; base=a.supervision
    report=json.loads((base/'report.json').read_text())
    for name,digest in report['hashes'].items():
        if sha256(base/name)!=digest: raise ValueError('Published hash mismatch: '+name)
    roi=np.array(json.loads((base/'roi.json').read_text())['vertex_ids'])
    cameras=json.loads((root/'processed/dataset_v1/cameras.json').read_text())['cameras']
    transforms=json.loads((base/'image_transforms.json').read_text())
    candidates={}
    for split in ['train','val']:
        for line in (root/f'processed/dataset_v1/samples_{split}.jsonl').open():
            s=json.loads(line)
            if s['sample_id'] in candidates: raise ValueError('Duplicate input sample')
            candidates[s['sample_id']]=s
    actual={}
    for split in ['train','val']:
        for line in (base/f'samples_{split}.jsonl').open():
            s=json.loads(line)
            if s['sample_id'] in actual: raise ValueError('Duplicate eligible sample')
            if s['split']!=split: raise ValueError('Split mismatch')
            actual[s['sample_id']]=s
    excluded={json.loads(l)['sample_id']:json.loads(l)['reasons'] for l in (base/'excluded.jsonl').open()}
    if set(actual)&set(excluded) or set(actual)|set(excluded)!=set(candidates):
        raise ValueError('Incomplete or overlapping inclusion/exclusion')
    artifacts=json.loads((base/'artifacts.json').read_text())
    def check(record):
        data={}
        for key,path_key in [('source','source'),('target','target'),('frustum','frustum')]:
            path=(root if key=='source' else base)/record[path_key]
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=record[key+'_sha256']: raise ValueError('Shard hash mismatch')
            with np.load(io.BytesIO(raw),allow_pickle=False) as d:
                keys={'source':['frame_ids','contact_smplx','contact_smplx_valid','vertices_multicam','s2m_dist_id'],
                      'target':['frame_ids','vertex_ids','contact','contact_valid','unsigned_surface_distance'],
                      'frustum':['sample_ids','frame_ids','annotation_rows','sole_raw_frustum','sole_crop_frustum','contact_supervision_mask','body_raw_inbounds','body_crop_inbounds','sole_crop_uv']}[key]
                data[key]={k:d[k] for k in keys}
        src,tgt,view=data['source'],data['target'],data['frustum']
        np.testing.assert_array_equal(tgt['frame_ids'],src['frame_ids'])
        np.testing.assert_array_equal(tgt['vertex_ids'],roi)
        np.testing.assert_array_equal(tgt['contact'],src['contact_smplx'][:,roi])
        np.testing.assert_array_equal(tgt['contact_valid'],src['contact_smplx_valid'][:,roi])
        np.testing.assert_allclose(tgt['unsigned_surface_distance'],np.linalg.norm(src['s2m_dist_id'][:,roi],axis=-1),rtol=1e-6)
        stats=Counter(frames=len(tgt['frame_ids']),samples=len(view['sample_ids']),
            missing_label_frames=int((~tgt['contact_valid'].any(axis=(1,2))).sum()),
            label_valid=int(tgt['contact_valid'].sum()),label_positive=int((tgt['contact']*tgt['contact_valid']).sum()))
        projected={}
        for i,sid in enumerate(view['sample_ids']):
            s=candidates[str(sid)]; row=s['annotation']['row']; key=s['camera_key']
            if record['source']!=s['annotation']['path'] or s['annotation']['sha256']!=record['source_sha256']:
                raise ValueError('Source pointer mismatch')
            if int(view['frame_ids'][i])!=s['frame_id'] or int(view['annotation_rows'][i])!=row or int(src['frame_ids'][row])!=s['frame_id']:
                raise ValueError('Frame association mismatch')
            if key not in projected:
                cam=cameras[key]; tr=transforms[key]
                # Independent homogeneous projection with combined P=K[E|t].
                verts=src['vertices_multicam'].astype(np.float64)
                hom=np.concatenate([verts,np.ones(verts.shape[:-1]+(1,))],axis=-1)
                q=hom@(np.array(cam['Intrinsics'])@np.array(cam['CameraMatrix'])).T
                depth=(hom@np.array(cam['CameraMatrix']).T)[...,2]
                uv=q[...,:2]/q[...,2:]
                scale=np.array(tr['resized_size_wh'])/np.array(tr['raw_size_wh'])
                crop=(uv+.5)*scale-.5-np.array(tr['crop_xyxy'][:2])
                def mask(coords,wh):
                    return (depth>1e-6)&np.isfinite(coords).all(-1)&(coords>=0).all(-1)&(coords<np.array(wh)).all(-1)
                projected[key]=(mask(uv,tr['raw_size_wh']),mask(crop,tr['crop_size_wh']),crop)
            raw,crop,xy=projected[key]
            np.testing.assert_array_equal(view['sole_raw_frustum'][i],raw[row,roi])
            np.testing.assert_array_equal(view['sole_crop_frustum'][i],crop[row,roi])
            np.testing.assert_allclose(view['sole_crop_uv'][i],xy[row,roi],rtol=1e-6,atol=1e-4)
            if view['body_raw_inbounds'][i]!=raw[row].sum() or view['body_crop_inbounds'][i]!=crop[row].sum():
                raise ValueError('Whole-body frustum count mismatch')
            valid=tgt['contact_valid'][row]&crop[row,roi]
            np.testing.assert_array_equal(view['contact_supervision_mask'][i],valid)
            reasons=[]
            if not tgt['contact_valid'][row].any(): reasons.append('missing_smplx_labels')
            if not crop[row].any(): reasons.append('body_outside_crop')
            if not crop[row,roi].any(): reasons.append('soles_outside_crop')
            if reasons:
                if sorted(excluded.get(str(sid),[]))!=sorted(reasons): raise ValueError('Wrong exclusion reasons')
            else:
                item=actual[str(sid)]
                if item['target']!={'path':record['target'],'sha256':record['target_sha256'],'row':row}: raise ValueError('Target pointer mismatch')
                if item['frustum']!={'path':record['frustum'],'sha256':record['frustum_sha256'],'row':i}: raise ValueError('Frustum pointer mismatch')
                for key2,value in s.items():
                    if item[key2]!=value: raise ValueError('Source sample modified')
            stats['view_valid']+=int(valid.sum()); stats['view_positive']+=int((tgt['contact'][row]*valid).sum())
        return record['split'],stats
    counts={s:Counter() for s in ['train','val']}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for done,(split,stats) in enumerate(pool.map(check,artifacts),1):
            counts[split].update(stats)
            if done%100==0: print(json.dumps(dict(checked=done,total=len(artifacts))),flush=True)
    for split,c in counts.items():
        for key in ['frames','samples','label_valid','label_positive','view_valid','view_positive']:
            if c[key]!=report['counts'][split][key]: raise ValueError('Independent totals mismatch')
        if not 0<c['view_positive']<c['view_valid']: raise ValueError('Both target classes required')
    out=dict(status='passed',scope='full supervision/frustum/association audit; no frontend geometry assertion',
        counts=counts,eligible=len(actual),excluded=len(excluded),candidates=len(candidates),shards=len(artifacts),
        supervision_report_sha256=sha256(base/'report.json'),audit_implementation_sha256=sha256(Path(__file__)),
        training_ready=False)
    save_json(a.output,out); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
