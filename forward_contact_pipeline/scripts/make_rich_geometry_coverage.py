#!/usr/bin/env python3
"""Freeze expanded coverage and a subject/camera-disjoint geometry study."""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from contact_streaming.rich_assets import save_json,sha256
TUNE_SUBJECTS=['003','020']
CONFIRM=['ParkingLot2_008_eating1','ParkingLot2_014_overfence3',
         'ParkingLot2_015_pushup1','ParkingLot2_016_burpeejump2']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rich-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    catalog=json.loads((a.rich_root/'processed/annotations_v1/catalog.json').read_text())
    count=Counter(t['sequence'] for t in catalog['tracks'])
    tracks={t['sequence']:t for t in catalog['tracks'] if count[t['sequence']]==1 and t['capture']!='ParkingLot2'}
    groups=defaultdict(list)
    for split in ['train','val']:
        with (a.rich_root/f'processed/supervision_v2/samples_{split}.jsonl').open() as stream:
            for line in stream:
                row=json.loads(line)
                if row['sequence'] in tracks or row['sequence'] in CONFIRM:
                    groups[(row['sequence'],row['camera_id'])].append(row)
    def clip(name,cam,fraction,role):
        rows=sorted(groups[(name,cam)],key=lambda s:s['frame_id'])
        for i in range(int(len(rows)*fraction),len(rows)-15):
            frames=rows[i:i+16]
            if [s['frame_id'] for s in frames]==list(range(frames[0]['frame_id'],frames[0]['frame_id']+16)):
                return dict(sequence=name,subject=frames[0]['subject'],camera_id=cam,length=16,
                    split=frames[0]['split'],study_role=role,selection_fraction=fraction,frames=frames)
        raise ValueError('No fixed consecutive run: '+name+f' cam{cam}')
    clips=[]
    for name,t in sorted(tracks.items()):
        role='tune' if t['subject'] in TUNE_SUBJECTS else 'fit'
        available=sorted(cam for seq,cam in groups if seq==name and cam!=4)
        for cam in ([4] if role=='tune' else available[:2]):
            for fraction in [.2,.7]:clips.append(clip(name,cam,fraction,role))
    plan=dict(schema_version=1,kind='geometry_coverage_expansion',
        selection='All single-annotation internal-train sequences; subjects003/020 only camera4 tune; other subjects first two available cameras excluding4 fit; 16 frames after20/70 percent',
        tune_subjects=TUNE_SUBJECTS,tune_camera=4,clips=clips)
    ids=[s['sample_id'] for c in clips for s in c['frames']]
    if len(ids)!=len(set(ids)):raise ValueError('Repeated frame-view')
    save_json(a.output/'development_plan.json',plan)
    previous=set()
    for old in ['geometry_experiments_v1/confirmation_plan.json','geometry_experiments_v4/holdout_plan.json']:
        previous.update(c['sequence'] for c in json.loads((a.rich_root/'processed'/old).read_text())['clips'])
    if set(CONFIRM)&previous:raise ValueError('Confirmation sequence reused')
    save_json(a.output/'confirmation_plan.json',dict(schema_version=1,kind='new_geometry_confirmation',
        selection='Four unused sequences, one per heldout participant; camera0,16 frames after midpoint; fixed before errors',
        clips=[clip(n,0,.5,'confirm') for n in CONFIRM]))
    summary=dict(development_clips=len(clips),development_frames=len(ids),
        fit_subjects=sorted({c['subject'] for c in clips if c['study_role']=='fit'}),
        tune_subjects=TUNE_SUBJECTS,confirmation_frames=64,
        development_sha256=sha256(a.output/'development_plan.json'),confirmation_sha256=sha256(a.output/'confirmation_plan.json'))
    save_json(a.output/'plan_summary.json',summary);print(json.dumps(summary))
if __name__=='__main__':main()
