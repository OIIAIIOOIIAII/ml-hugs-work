#!/usr/bin/env python3
"""Freeze train-only regression experiments disjoint from confirmation sequences."""
import argparse,json,sys
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from contact_streaming.rich_assets import save_json,sha256

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rich-root',type=Path,required=True)
 p.add_argument('--confirmation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('Plan already frozen')
 conf=json.loads(a.confirmation.read_text());excluded={c['sequence'] for c in conf['clips']}
 excluded.update(['ParkingLot1_002_stretching1','ParkingLot2_016_stretching1'])
 catalog=json.loads((a.rich_root/'processed/annotations_v1/catalog.json').read_text())
 multiplicity=defaultdict(int)
 for t in catalog['tracks']:multiplicity[t['sequence']]+=1
 available=defaultdict(list)
 for t in catalog['tracks']:
  if t['capture']!='ParkingLot2' and t['sequence'] not in excluded and multiplicity[t['sequence']]==1:
   available[t['capture']].append(t)
 limits={'BBQ':1,'LectureHall':2,'ParkingLot1':6,'Pavallion':8}
 selected=[]
 for capture,tracks in sorted(available.items()):
  selected.extend(sorted(tracks,key=lambda t:t['sequence'])[:limits[capture]])
 names={t['sequence'] for t in selected};groups=defaultdict(list)
 with (a.rich_root/'processed/supervision_v2/samples_train.jsonl').open() as stream:
  for line in stream:
   s=json.loads(line)
   if s['sequence'] in names:groups[(s['sequence'],s['camera_id'])].append(s)
 clips=[]
 for t in selected:
  cams=sorted(k[1] for k in groups if k[0]==t['sequence'])
  for cam in [cams[0],cams[len(cams)//2]]:
   rows=sorted(groups[(t['sequence'],cam)],key=lambda s:s['frame_id'])
   for quantile in [.25,.70]:
    for start in range(int(len(rows)*quantile),len(rows)-15):
     candidate=rows[start:start+16]
     if [s['frame_id'] for s in candidate]==list(range(candidate[0]['frame_id'],candidate[0]['frame_id']+16)):break
    else:raise ValueError('No predeclared consecutive run: '+t['sequence'])
    clips.append(dict(split='train',sequence=t['sequence'],subject=t['subject'],camera_id=cam,length=16,
     selection_quantile=quantile,frames=candidate))
 ids=[s['sample_id'] for c in clips for s in c['frames']]
 if len(ids)!=len(set(ids)):raise ValueError('Duplicate training frame-view')
 result=dict(schema_version=1,kind='train_only_geometry_residual_study',confirmation_plan_sha256=sha256(a.confirmation),
  selection='Named capture limits; sorted sequences; first/middle available camera; first valid run after 25% and 70%; no frontend error selection',
  sequences=sorted(names),confirmation_sequences_disjoint=not bool(names&excluded),clips=clips)
 save_json(a.output,result);print(json.dumps(dict(clips=len(clips),frames=len(ids),sequences=len(names),sha256=sha256(a.output))))
if __name__=='__main__':main()
