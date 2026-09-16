#!/usr/bin/env python3
"""Prepare hashed frozen inputs and separate RICH supervision for refinement."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import roma
import smplx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contact_streaming.rich_assets import sha256, save_json
from contact_streaming.rich_supervision import atomic_npz


def prepare(rich, models, planpath, features, output):
    provenance = {str(p): sha256(p) for p in [planpath, features/'manifest.json',
        rich/'processed/dataset_v1/cameras.json', rich/'processed/supervision_v2/roi.json',
        *[models/f'smplx/SMPLX_{g}.npz' for g in ['NEUTRAL', 'MALE', 'FEMALE']]]}
    if (output/'manifest.json').exists():
        old = json.loads((output/'manifest.json').read_text())
        if old['provenance'] != provenance or old['data_sha256'] != sha256(output/'data.npz'):
            raise ValueError('Cache provenance/hash changed')
        return old
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(planpath.read_text())
    manifest = json.loads((features/'manifest.json').read_text())
    if manifest['provenance']['plan_sha256'] != sha256(planpath) or len(manifest['clips']) != len(plan['clips']):
        raise ValueError('Plan mismatch or incomplete features')
    teachers = {g: smplx.create(str(models), model_type='smplx', gender=g, ext='npz',
                    use_pca=True, num_pca_comps=12).eval() for g in ['male', 'female']}
    with np.load(models/'smplx/SMPLX_NEUTRAL.npz') as d:
        template = d['v_template']; pinv = np.linalg.pinv(d['shapedirs'][:, :, :10].reshape(-1, 10).astype(np.float64))
    assets = {}
    for g in teachers:
        with np.load(models/f'smplx/SMPLX_{g.upper()}.npz') as d:
            assets[g] = (d['v_template'], d['shapedirs'][:, :, :10])
    cameras = json.loads((rich/'processed/dataset_v1/cameras.json').read_text())['cameras']
    anncache = {}; rows = []; failed = []
    for ci, (clip, ref) in enumerate(zip(manifest['clips'], plan['clips'])):
        if len(clip['frames']) != len(ref['frames']):
            raise ValueError('Missing feature frame')
        if any(f['humans'] != 1 for f in clip['frames']):
            failed.append(dict(clip=ci, sequence=clip['sequence'], reason='detection_count'))
            continue
        local = []; tracks = []
        for f, sample in zip(clip['frames'], ref['frames']):
            if f['sample_id'] != sample['sample_id'] or sha256(features/f['file']) != f['sha256']:
                raise ValueError('Feature mismatch')
            with np.load(features/f['file']) as d:
                values = {k: d[k] for k in d.files}
            tracks.append(values['smpl_id'].reshape(-1).tolist())
            ann = sample['annotation']; key = ann['path']
            if key not in anncache:
                if sha256(rich/key) != ann['sha256']:
                    raise ValueError('GT hash mismatch')
                with np.load(rich/key) as d:
                    anncache[key] = {k:d[k] for k in d.files if k.startswith('body_') or k in ['frame_ids','vertices_multicam']}
            gt = anncache[key]; i = ann['row']
            if gt['frame_ids'][i] != sample['frame_id']:
                raise ValueError('GT frame mismatch')
            params = {k[5:]:torch.from_numpy(v[i:i+1]).float() for k,v in gt.items()
                      if k.startswith('body_') and k != 'body_pose_embedding'}
            e = np.array(cameras[sample['camera_key']]['CameraMatrix'])
            target = gt['vertices_multicam'][i] @ e[:, :3].T + e[:, 3]
            with torch.no_grad():
                head_gt = teachers[sample['gender']](**params).joints[0, 15].numpy() @ e[:, :3].T + e[:, 3]
            rotation = values['smpl_rotmat'][0, 0]
            rgt = torch.cat([(torch.tensor(e[:, :3]).float() @ roma.rotvec_to_rotmat(params['global_orient'])).reshape(1, 3, 3),
                              roma.rotvec_to_rotmat(params['body_pose'].reshape(21, 3))])
            delta = roma.rotmat_to_rotvec(rgt @ torch.from_numpy(rotation[:22]).transpose(-1, -2)).numpy().reshape(-1)
            vt, dirs = assets[sample['gender']]
            neutral_beta = pinv @ (vt + np.einsum('vci,i->vc', dirs, gt['body_betas'][i]) - template).reshape(-1)
            shape = values['smpl_shape'][0, 0]; head = values['smpl_transl'][0, 0]
            pose = np.r_[rotation[:22].reshape(-1), shape, head]
            y = np.r_[delta, neutral_beta-shape, (head_gt-head)/max(float(head[2]), .5)]
            local.append(dict(pose=pose, token=values['smpl_query'].reshape(-1), rotation=rotation, shape=shape,
                head=head, expression=values['smpl_expression'][0, 0], c2w=values['camera_c2w'],
                target=target.astype(np.float32), base=values['vertices_camera'][0], y=y, head_gt=head_gt,
                sequence=sample['sequence'], subject=sample['subject'], sample_id=sample['sample_id'],
                frame_id=sample['frame_id'], clip=ci))
        if any(t != tracks[0] for t in tracks):
            failed.append(dict(clip=ci, sequence=clip['sequence'], reason='identity_switch'))
            continue
        rows.extend(local)
        print(json.dumps(dict(stage='prepare',clip=ci,samples=len(rows))), flush=True)
    if not rows:
        raise ValueError('No usable samples')
    arrays = {key:np.array([row[key] for row in rows]) for key in rows[0]}
    for key, value in arrays.items():
        if value.dtype.kind == 'f':
            arrays[key] = value.astype(np.float32)
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite '+key)
    atomic_npz(output/'data.npz', **arrays)
    result = dict(status='complete', provenance=provenance, data_sha256=sha256(output/'data.npz'),
        samples=len(rows), unique_body_frames=len({(r['sequence'],r['frame_id']) for r in rows}),
        subjects=sorted({r['subject'] for r in rows}), sequences=sorted({r['sequence'] for r in rows}),
        declared_clips=len(plan['clips']), failures=failed, GT_input=False,
        input_keys=['pose','token','rotation','shape','head','expression','c2w'],
        label_keys=['target','head_gt','y'])
    save_json(output/'manifest.json', result)
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['rich-root','models','plan','features','output']:
        p.add_argument('--'+key, type=Path, required=True)
    a=p.parse_args(); torch.set_num_threads(2)
    print(json.dumps(prepare(a.rich_root,a.models,a.plan,a.features,a.output)))
