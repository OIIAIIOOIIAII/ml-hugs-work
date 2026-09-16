#!/usr/bin/env python3
"""Make a scene-disjoint Stage-A split with every available proxy corruption.

Unlike the strict error-family OOD split, this is an augmentation effectiveness
protocol: train/val/test scenes are disjoint, while clean, normal18,
dropout20, and distance occur in every split.  It answers whether a reliability
head can work *within its declared corruption coverage*, not whether it detects
an entirely unseen error family.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

def read(root):
    return {m['scene']: str(p) for p in sorted(root.glob('*/manifest.json')) for m in [json.loads(p.read_text())]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clean-root',type=Path,required=True); p.add_argument('--normal-root',type=Path,required=True)
    p.add_argument('--dropout-root',type=Path,required=True); p.add_argument('--distance-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); roots=[a.clean_root,a.normal_root,a.dropout_root,a.distance_root]; tables=[read(root) for root in roots]
    common=set.intersection(*(set(table) for table in tables))
    if len(common)!=12: raise ValueError(f'expected 12 common scenes, got {len(common)}')
    scenes=sorted(common); groups={'train':scenes[:8],'val':scenes[8:10],'test':scenes[10:]}
    out={'schema':'hugs.forward_contact.stage_a.augmented_scene_split.v1','kind':'scene_disjoint_corruption_covered','groups':groups,'families':['clean','normal18','dropout20','distance'],'splits':{name:[table[scene] for scene in group_scenes for table in tables] for name,group_scenes in groups.items()},'note':'Scene-disjoint; every declared proxy error family appears in every split. This evaluates augmented in-distribution reliability, not unseen-error OOD detection.'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)); print(json.dumps({k:len(v) for k,v in out['splits'].items()}))
if __name__=='__main__': main()
