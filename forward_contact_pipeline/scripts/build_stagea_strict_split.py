#!/usr/bin/env python3
"""Freeze a deliberately strict Stage-A scene/sequence/error-family split."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def load(root: Path):
    return [json.loads(p.read_text()) | {"manifest": str(p)} for p in sorted(root.glob("*/manifest.json"))]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--clean-root',type=Path,required=True); ap.add_argument('--normal-root',type=Path,required=True); ap.add_argument('--dropout-root',type=Path,required=True); ap.add_argument('--distance-root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    roots={'clean':a.clean_root,'normal18':a.normal_root,'dropout20':a.dropout_root,'distance':a.distance_root}
    data={k:load(v) for k,v in roots.items()}
    by={k:{x['scene']:x for x in v} for k,v in data.items()}
    common=set.intersection(*(set(v) for v in by.values()))
    if len(common)!=12: raise ValueError(f'expected 12 common scenes, got {len(common)}')
    scenes=sorted(common); groups={'train':scenes[:8],'val':scenes[8:10],'test':scenes[10:]}
    families={'train':['clean','normal18'],'val':['dropout20'],'test':['distance']}
    splits={g:[by[f][s]['manifest'] for s in groups[g] for f in families[g]] for g in groups}
    out={'schema':'hugs.forward_contact.stage_a.strict_split.v1','kind':'strict_scene_sequence_error_family','groups':groups,'families':families,'splits':splits,'note':'No scene or error family appears across splits; this is a robustness split, not the same-distribution mechanism split.'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps({k:len(v) for k,v in splits.items()}))
if __name__=='__main__': main()
