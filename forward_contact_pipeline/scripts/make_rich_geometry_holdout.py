#!/usr/bin/env python3
"""Freeze new sequence holdouts from metadata before running any predictions."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contact_streaming.rich_assets import save_json, sha256

NAMES = [
    'ParkingLot2_008_pushup2', 'ParkingLot2_008_phonetalk1',
    'ParkingLot2_014_burpeejump2', 'ParkingLot2_014_takingphotos2',
    'ParkingLot2_015_overfence1', 'ParkingLot2_016_pushup2',
]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rich-root', type=Path, required=True)
    p.add_argument('--previous-plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--camera-zero-of', type=Path, help='Coverage supplement, same sequence/frame IDs at camera0')
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Holdout plan already frozen')
    if a.camera_zero_of:
        original = json.loads(a.camera_zero_of.read_text())
        wanted = {(c['sequence'], s['frame_id']) for c in original['clips'] for s in c['frames']}
        index = {}
        with (a.rich_root/'processed/supervision_v2/samples_val.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['camera_id'] == 0 and (row['sequence'], row['frame_id']) in wanted:
                    index[(row['sequence'], row['frame_id'])] = row
        clips = []
        for clip in original['clips']:
            frames = [index[(clip['sequence'], s['frame_id'])] for s in clip['frames']]
            clips.append({k:v for k,v in clip.items() if k != 'frames'} | dict(camera_id=0, frames=frames))
        save_json(a.output, dict(schema_version=1, kind='coverage_supplement',
            original_plan_sha256=sha256(a.camera_zero_of),
            selection='Same six sequences and exact frame IDs at camera0, all included; added after detection coverage only, before training or geometry error evaluation',
            clips=clips))
        print(sha256(a.output))
        return
    old = json.loads(a.previous_plan.read_text())
    if set(NAMES) & {c['sequence'] for c in old['clips']}:
        raise ValueError('Reused confirmation sequence')
    groups = {}
    with (a.rich_root / 'processed/supervision_v2/samples_val.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['sequence'] in NAMES:
                groups.setdefault((row['sequence'], row['camera_id']), []).append(row)
    clips = []
    for name in NAMES:
        cams = sorted(c for s, c in groups if s == name)
        camera = cams[len(cams) // 2]
        rows = sorted(groups[(name, camera)], key=lambda s: s['frame_id'])
        for i in range(len(rows) // 2, len(rows) - 31):
            frames = rows[i:i + 32]
            if [s['frame_id'] for s in frames] == list(range(frames[0]['frame_id'], frames[0]['frame_id'] + 32)):
                break
        else:
            raise ValueError('No consecutive window: ' + name)
        clips.append(dict(sequence=name, camera_id=camera, subject=frames[0]['subject'],
                          split='val', length=32, frames=frames))
    save_json(a.output, dict(schema_version=1, kind='new_geometry_holdout',
        selection='Six named unused sequences; middle valid camera; first 32 consecutive frames after midpoint; no error selection',
        previous_plan_sha256=sha256(a.previous_plan), training_subjects_disjoint=True,
        note='Subjects occur in older inspected validation sequences; sequences are new; not official test', clips=clips))
    print(json.dumps(dict(sha256=sha256(a.output), clips=[{k:v for k,v in c.items() if k!='frames'} for c in clips])))

if __name__ == '__main__':
    main()
