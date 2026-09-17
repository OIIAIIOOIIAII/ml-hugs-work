#!/usr/bin/env python3
"""Enumerate every eligible RICH sample once, with scene/subject-disjoint splits."""
import argparse,json,sys
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from contact_streaming.rich_assets import sha256,save_json


def build(rich,output,chunk=32,seed=42):
    output.mkdir(parents=True,exist_ok=False)
    groups=defaultdict(dict);seen=set();owners={k:{} for k in ['sequence','scene','subject']};counts=Counter()
    provenance={}
    for split in ['train','val']:
        file=rich/f'processed/supervision_v2/samples_{split}.jsonl';provenance[split]=sha256(file)
        with file.open() as stream:
            for line in stream:
                s=json.loads(line)
                if s['sample_id'] in seen or s['split']!=split:raise ValueError('Duplicate sample or wrong split')
                seen.add(s['sample_id']);counts[split]+=1
                for key,mapping in owners.items():
                    value=s[key]
                    if value in mapping and mapping[value]!=split:raise ValueError('Split leakage: '+key)
                    mapping[value]=split
                key=(split,s['sequence'],s['camera_id']);frames=groups[key]
                f=frames.setdefault(s['frame_id'],{k:s[k] for k in ['image','image_sha256','frame_id','camera_key','raw_size_wh']})
                f.setdefault('targets',[]).append({k:s[k] for k in ['sample_id','subject','gender','annotation','target']})
    clips=[]
    for (split,seq,cam),frames in sorted(groups.items()):
        part=[]
        def emit(part):
            if part:clips.append(dict(split=split,sequence=seq,camera_id=cam,frames=part))
        for frame in [frames[i] for i in sorted(frames)]:
            if part and (len(part)==chunk or frame['frame_id']!=part[-1]['frame_id']+1):emit(part);part=[]
            part.append(frame)
        emit(part)
    def write(pair):
        i,c=pair;name=f'clips/{i:06}.json';path=output/name
        save_json(path,c)
        return dict(id=i,split=c['split'],sequence=c['sequence'],camera_id=c['camera_id'],path=name,
                    sha256=sha256(path),frames=len(c['frames']),targets=sum(len(f['targets']) for f in c['frames']))
    (output/'clips').mkdir()
    with ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(write,enumerate(clips)))
    rng=np.random.default_rng(seed)
    orders={s:rng.permutation([r['id'] for r in records if r['split']==s]).tolist() for s in ['train','val']}
    manifest=dict(schema_version=1,kind='full_rich_contact_research',all_eligible_samples=True,
        source_sha256=provenance,samples=dict(counts),unique_images={s:sum(r['frames'] for r in records if r['split']==s) for s in ['train','val']},
        splits={s:{k:sorted(v for v,owner in mapping.items() if owner==s) for k,mapping in owners.items()} for s in ['train','val']},
        chunk_frames=chunk,seed=seed,records=records,orders=orders,training_authorized=True,deployment_ready=False,
        note='Full eligible official-train assets; ParkingLot2 is internal validation. No official val/test assets. GT is matching/supervision only; geometry error never filters training rows.')
    save_json(output/'index.json',manifest)
    print(json.dumps({k:manifest[k] for k in ['samples','unique_images','chunk_frames']}),flush=True)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rich-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--chunk',type=int,default=32);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.chunk<1:raise ValueError('Positive chunk size required')
    build(a.rich_root,a.output,a.chunk,a.seed)
