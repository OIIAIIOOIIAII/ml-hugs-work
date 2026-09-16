#!/usr/bin/env python3
"""Post-selection paired comparisons; no model choice or correction fitting."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from contact_streaming.rich_assets import sha256,save_json

KEYS=['raw_body_median_m','raw_sole_median_m','body_median_m','sole_median_m']


def paired(before,after):
    a={r['clip']:r for r in before['clips']}
    b={r['clip']:r for r in after['clips']}
    if a.keys()!=b.keys():raise ValueError('Unpaired clips')
    return {k:dict(mean_delta_cm=float(np.mean([b[i][k]-a[i][k] for i in a])*100),
                   improved=sum(b[i][k]<a[i][k]-1e-6 for i in a),clips=len(a),
                   per_clip_delta_cm={str(i):float((b[i][k]-a[i][k])*100) for i in a}) for k in KEYS}


def cache_summary(path):
    manifest=json.loads((path/'manifest.json').read_text())
    if sha256(path/'data.npz')!=manifest['data_sha256']:raise ValueError('Cache changed')
    with np.load(path/'data.npz',allow_pickle=False) as d:
        z=d['head_gt'][:,2]-d['head'][:,2]
        seq=sorted(set(d['sequence']))
        rows=[dict(sequence=str(s),head_depth_correction_median_cm=float(np.median(z[d['sequence']==s])*100),
                   samples=int((d['sequence']==s).sum())) for s in seq]
        groups={};partitions=[('all',np.ones(len(z),dtype=bool))]
        if np.isin(d['subject'],['003','020']).any():
            partitions.extend([('fit',~np.isin(d['subject'],['003','020'])),('tune',np.isin(d['subject'],['003','020']))])
        for name,mask in partitions:
            if not mask.any():continue
            y=d['y'][mask]
            groups[name]=dict(samples=int(mask.sum()),depth_correction_median_cm=float(np.median(z[mask])*100),
                head_delta_exceeds_25pct_depth_fraction=float(np.mean(np.linalg.norm(y[:,76:79],axis=-1)>.25)),
                any_joint_delta_exceeds_1_2rad_fraction=float(np.mean((np.linalg.norm(y[:,:66].reshape(-1,22,3),axis=-1)>1.2).any(-1))),
                any_shape_delta_exceeds_3_fraction=float(np.mean((np.abs(y[:,66:76])>3).any(-1))))
    return dict(sequences=rows,frame_median_cm=float(np.median(z)*100),
                sequence_median_cm=float(np.median([r['head_depth_correction_median_cm'] for r in rows])),groups=groups)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--development-cache',type=Path,required=True)
    p.add_argument('--confirmation-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    selection=json.loads((a.study/'selection.json').read_text())
    report=json.loads((a.evaluation/'report.json').read_text())
    if report['selection_sha256']!=sha256(a.study/'selection.json'):raise ValueError('Selection changed')
    runs={(r['budget'],r['architecture'],r['seed']):r for r in report['runs']}
    comparisons=[]
    for mode in ['full','guarded']:
        for seed in [7,19]:
            for arch in ['mlp','kinematic']:
                comparisons.append(dict(effect='data',architecture=arch,seed=seed,mode=mode,
                    comparison=paired(runs['small',arch,seed][mode],runs['expanded',arch,seed][mode])))
            for budget in ['small','expanded']:
                comparisons.append(dict(effect='architecture',budget=budget,seed=seed,mode=mode,
                    comparison=paired(runs[budget,'mlp',seed][mode],runs[budget,'kinematic',seed][mode])))
    result=dict(selection=selection['selected'],selected_strength=selection['selected_strength'],
        declared_clips=report['declared_clips'],failures=report['failures'],
        versus_baseline={r['name']:{m:paired(report['baseline'],r[m]) for m in ['full','guarded']} for r in report['runs']},
        comparisons=comparisons,development_depth=cache_summary(a.development_cache),confirmation_depth=cache_summary(a.confirmation_cache),
        source_sha256=sha256(Path(__file__)),selection_sha256=report['selection_sha256'],
        note='Post-selection diagnostics only. Four declared sequences cannot establish broad generalization; no significance claim.')
    save_json(a.output,result)
    print(json.dumps(dict(selected=result['selection'],strength=result['selected_strength'],declared=result['declared_clips'],failures=result['failures'])))


if __name__=='__main__':main()
