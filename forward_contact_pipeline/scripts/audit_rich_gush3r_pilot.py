#!/usr/bin/env python3
"""Held-out RICH geometry measurement from head-corrected frozen frontend outputs.

Fits one diagnostic Sim(3) on the first 75% of each clip. Never writes aligned
geometry back into model inputs. A short-clip pass is necessary, not sufficient,
for full real frontend preparation.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from contact_streaming.alignment import robust_sim3
from contact_streaming.rich_assets import sha256,save_json


def stats(values):
    x=np.asarray(values)
    return dict(mean_m=float(x.mean()),median_m=float(np.median(x)),p95_m=float(np.quantile(x,.95)),max_m=float(x.max()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rich-root',type=Path,required=True); p.add_argument('--frontend',type=Path,required=True)
    p.add_argument('--plan',type=Path); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    root=a.rich_root; front=a.frontend
    plan_path=a.plan or root/'processed/dataset_v1/pilot_clips.json'
    plan=json.loads(plan_path.read_text())
    manifest=json.loads((front/'manifest.json').read_text())
    if manifest['plan_sha256']!=sha256(plan_path): raise ValueError('Plan changed')
    roi=np.array(json.loads((root/'processed/supervision_v2/roi.json').read_text())['vertex_ids'])
    cameras=json.loads((root/'processed/dataset_v1/cameras.json').read_text())['cameras']
    fixed_ids=np.sort(np.random.default_rng(42).choice(10475,512,replace=False))
    results=[]
    for clip,expected in zip(manifest['clips'],plan['clips']):
        if len(clip['frames'])!=len(expected['frames']): raise ValueError('Incomplete clip')
        predicted=[]; teacher=[]; raw_camera_errors=[]; projection_errors=[]; track_ids=[]
        arrays_by_shard={}; matched=[]; unavailable=[]
        for i,(record,sample) in enumerate(zip(clip['frames'],expected['frames'])):
            if record['sample_id']!=sample['sample_id'] or record['frame_id']!=sample['frame_id']:
                raise ValueError('Frontend frame association mismatch')
            if sha256(front/record['file'])!=record['sha256']: raise ValueError('Frontend hash mismatch')
            with np.load(front/record['file'],allow_pickle=False) as d:
                if record['humans']!=1:
                    unavailable.append(dict(frame_id=record['frame_id'],reason='not_exactly_one_predicted_person',humans=record['humans'])); continue
                pred=d['smplx_vertices_world'][0].astype(np.float64)
                c2w=d['camera_c2w'][0].astype(np.float64)
                camera_pred=(pred-c2w[:3,3])@c2w[:3,:3]
                track_ids.append(d['smpl_id'].reshape(-1).tolist())
                if not np.isfinite(pred).all() or pred.shape!=(10475,3): raise ValueError('Invalid predicted mesh')
                # Exported head-position convention must be honored exactly.
                np.testing.assert_allclose(d['head_world'],d['human_param_trans'],atol=1e-6)
            ann=sample['annotation']; path=ann['path']
            if path not in arrays_by_shard:
                if sha256(root/path)!=ann['sha256']: raise ValueError('GT hash mismatch')
                with np.load(root/path,allow_pickle=False) as d:
                    arrays_by_shard[path]={k:d[k] for k in ['frame_ids','vertices_multicam']}
            gt=arrays_by_shard[path]
            if int(gt['frame_ids'][ann['row']])!=sample['frame_id']: raise ValueError('GT frame mismatch')
            target=gt['vertices_multicam'][ann['row']].astype(np.float64)
            e=np.array(cameras[sample['camera_key']]['CameraMatrix'])
            camera_gt=target@e[:,:3].T+e[:,3]
            raw_camera_errors.append(np.linalg.norm(camera_pred-camera_gt,axis=-1))
            transform=record['transform']; k=np.array(transform['raw_to_crop'])@np.array(cameras[sample['camera_key']]['Intrinsics'])
            q_gt=camera_gt@k.T
            # Predicted projection uses the released prior K, not GT K.
            q_pred=camera_pred@np.asarray(record['K_model']).reshape(3,3).T
            uvgt=q_gt[:,:2]/q_gt[:,2:]; uvpred=q_pred[:,:2]/q_pred[:,2:]
            projection_errors.append(np.linalg.norm(uvgt-uvpred,axis=-1))
            predicted.append(pred); teacher.append(target); matched.append(sample['frame_id'])
        if unavailable or len(predicted)<8 or any(ids!=track_ids[0] for ids in track_ids):
            results.append(dict(sequence=clip['sequence'],status='identity_or_detection_failure',unavailable=unavailable,
                track_ids=track_ids,passes_contact_gate=False)); continue
        pred,gt=np.array(predicted),np.array(teacher); split=len(pred)*3//4
        fit=robust_sim3(pred[:split,fixed_ids].reshape(-1,3),gt[:split,fixed_ids].reshape(-1,3))
        aligned=fit.transform(pred.reshape(-1,3)).reshape(pred.shape)
        errors=np.linalg.norm(aligned-gt,axis=-1)
        heldout=stats(errors[split:]); soles=stats(errors[split:][:,roi])
        result=dict(sequence=clip['sequence'],split=clip['split'],camera_id=clip['camera_id'],
            status='measured',frames=len(pred),fit_frame_ids=matched[:split],heldout_frame_ids=matched[split:],
            track_ids=track_ids,fit_vertex_sample=512,
            sim3=dict(scale=fit.scale,rotation=fit.rotation.tolist(),translation=fit.translation.tolist(),fit_residual_m=fit.residual),
            fit_whole_body_error=stats(errors[:split]),heldout_whole_body_error=heldout,heldout_sole_error=soles,
            camera_space_without_alignment=stats(raw_camera_errors),
            projection_pixel_error=dict(median_px=float(np.median(projection_errors)),p95_px=float(np.quantile(projection_errors,.95))),
            threshold_m=.02,passes_contact_gate=heldout['median_m']<=.02 and soles['median_m']<=.02)
        results.append(result)
    passed=len(results)==len(plan['clips']) and all(c['passes_contact_gate'] for c in results)
    out=dict(schema_version=1,status='measured',kind='RICH_real_frontend_heldout_geometry_pilot',
        passes_contact_gate=passed,training_ready=False,clips=results,
        frontier='expand geometry audit' if passed else 'stop full feature export; retain diagnostics and complete GT preparation',
        scope='two short clips; no claim of dataset-wide accuracy; frustum is not occlusion visibility',
        frontend_manifest_sha256=sha256(front/'manifest.json'),plan_sha256=sha256(plan_path),
        auditor_sha256=sha256(Path(__file__)),inputs_modified_by_GT_alignment=False)
    save_json(a.output,out); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
