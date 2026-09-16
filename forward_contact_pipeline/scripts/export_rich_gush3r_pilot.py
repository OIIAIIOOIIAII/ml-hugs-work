#!/usr/bin/env python3
"""Frozen causal GUSH3R numerical pilot; never calls infer.main or any renderer.

No RICH body/contact/calibration is read during model inference. Ground truth
matching and geometry measurements belong to the separate audit command.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from contact_streaming.rich_assets import save_json, sha256
from contact_streaming.rich_supervision import atomic_npz, pixel_transform


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--rich-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--dino-repo',type=Path,required=True)
    a=parser.parse_args()
    repo=a.repo.resolve(); rich=a.rich_root.resolve(); output=a.output.resolve()
    plan=(a.plan or rich/'processed/dataset_v1/pilot_clips.json').resolve()
    output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists():
        raise ValueError('Pilot already published; use a new output directory')
    os.environ['GUSH3R_DINO_HUB_LOCAL_REPO']=str(a.dino_repo.resolve())
    os.chdir(repo); sys.path.insert(0,str(repo)); sys.path.insert(0,str(repo/'src'))
    import torch
    from infer import load_model, prepare_input
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.utils.camera import pose_encoding_to_camera
    torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    settings=SimpleNamespace(gs_conf_threshold=1.,bg_mask_threshold=.02,bg_mask_dilation=3,
        bg_voxel_size=.005,bg_gaussian_max=2000000,bg_cap_policy='random_legacy')
    started=time.time(); model=load_model('cuda',settings)
    model.requires_grad_(False); model.eval(); model.skip_inference_render_outputs=True
    mesh=model.human_gs_head.smplx_mesh
    for layer in mesh.layer.values(): layer.requires_grad_(False); layer.eval()
    roi=json.loads((rich/'processed/supervision_v2/roi.json').read_text())
    with np.load(rich/'processed/annotations_v1/smplx_topology.npz',allow_pickle=False) as topology:
        if not np.array_equal(mesh.faces.cpu().numpy(),topology['faces']):
            raise ValueError('Frontend SMPL-X topology mismatch')
    # Fixed topology only is read above. No frame-specific GT reaches the frontend.
    atomic_npz(output/'binding.npz',faces=mesh.faces.cpu().numpy(),
        sample_vertex_ids=mesh.sample_indices.cpu().numpy(),
        lbs_weights=mesh.layer['neutral'].lbs_weights.cpu().numpy(),
        sole_vertex_ids=np.asarray(roi['vertex_ids']))
    stash={}
    def cpu(value): return value.detach().float().cpu().numpy()
    def head_hook(module,args,kwargs,result):
        if kwargs.get('render') is not False: raise RuntimeError('Rendering must remain disabled')
        p=kwargs['smpl_params']; layer=module.smplx_mesh.layer['neutral']
        with torch.no_grad():
            out=layer(betas=p['betas'],expression=p['expr'],global_orient=p['root_pose'],
                body_pose=p['body_pose'],left_hand_pose=p['lhand_pose'],right_hand_pose=p['rhand_pose'],
                jaw_pose=p['jaw_pose'],leye_pose=p['leye_pose'],reye_pose=p['reye_pose'])
            # Released GUSH3R translation is the head position, not SMPL-X transl.
            corrected=out.vertices+(p['trans']-p['head_posed'])[:,None]
        stash.update(smplx_vertices_world=cpu(corrected),head_world=cpu(p['trans']),
            posed_head_before_translation=cpu(p['head_posed']),
            decoder_image_tokens=cpu(kwargs['image_tokens']),smpl_query=cpu(kwargs['smpl_query']),
            human_c2w=cpu(kwargs['extrinsics']))
        for key,value in p.items(): stash['human_param_'+key]=cpu(value)
    def gs_hook(module,args,kwargs):
        stash['human_lbs_query_world']=cpu(kwargs['pts'])
    hooks=[model.human_gs_head.register_forward_hook(head_hook,with_kwargs=True),
           model.human_gs_head.gs_layer.register_forward_pre_hook(gs_hook,with_kwargs=True)]
    manifest=dict(schema_version=1,kind='real_frontend_pilot_diagnostic',training_ready=False,
        checkpoint_sha256=sha256(repo/'checkpoints/gush3r.pth'),
        frontend_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        frontend_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff'],cwd=repo)).hexdigest(),
        exporter_sha256=sha256(Path(__file__)),plan_sha256=sha256(plan),settings=vars(settings),
        frozen=True,causality='current and past only; recurrent state resets at each clip',
        intrinsics='released default FOV prior; RICH calibration is audit-only',
        translation='predicted head position; SMPL-X verts + trans - posed_head',
        scene_snapshot='current cumulative scene; at most 8192 highest-opacity Gaussians for diagnostics',
        note='No training cache or geometry pass implied by this export',clips=[])
    clips=json.loads(plan.read_text())['clips']
    try:
        for ci,clip in enumerate(clips):
            if len(clip['frames'])>32: raise ValueError('Pilot windows must fit in 32 frames')
            paths=[]
            for frame in clip['frames']:
                path=rich/frame['image']
                if sha256(path)!=frame['image_sha256']: raise ValueError('Input image hash mismatch')
                paths.append(str(path))
            views=prepare_input(paths,size=512,img_res=model.mhmr_img_res)
            records=[]
            for frame,view in zip(clip['frames'],views):
                transform=pixel_transform(*frame['raw_size_wh'],size=512,pad_to=model.mhmr_img_res)
                if list(view['img'].shape[-2:])!=transform['crop_size_wh'][::-1]:
                    raise ValueError('Actual frontend crop disagrees with numeric transform')
                records.append(dict(sample_id=frame['sample_id'],frame_id=frame['frame_id'],
                    image=frame['image'],image_sha256=frame['image_sha256'],transform=transform,
                    K_model=cpu(view['camera_intrinsics']).tolist(),K_mhmr_model=cpu(view['K_mhmr']).tolist()))
            def callback(i,res,view):
                if 'human_rgb' in res or 'human_mask' in res: raise RuntimeError('Unexpected rendered output')
                arrays=dict(stash); stash.clear()
                for key in ['smpl_rotmat','smpl_shape','smpl_transl','smpl_expression','smpl_id','smpl_loc',
                            'pts3d_in_self_view','conf_self','conf','msk']:
                    if torch.is_tensor(res.get(key)): arrays[key]=cpu(res[key])
                arrays['camera_c2w']=cpu(pose_encoding_to_camera(res['camera_pose']))
                scene=res.get('gaussians')
                if scene is not None:
                    rank=scene.opacities[0].reshape(-1).topk(min(8192,scene.means.shape[1])).indices
                    for field in ['means','covariances','harmonics','opacities','scales','rotations']:
                        arrays['scene_'+field]=cpu(getattr(scene,field)[:,rank])
                    records[i]['scene_total_count']=scene.means.shape[1]
                human=res.get('human_gaussians')
                if human is not None:
                    for field in ['xyz','scaling','rotation','opacity','shs']:
                        arrays['human_gaussian_'+field]=cpu(getattr(human,field))
                for key,value in arrays.items():
                    if not np.isfinite(value).all(): raise ValueError('Nonfinite frontend output: '+key)
                name=f'clip_{ci:02d}/{i:04d}.npz'; atomic_npz(output/name,**arrays)
                records[i].update(file=name,sha256=sha256(output/name),
                    humans=int(arrays.get('smplx_vertices_world',np.empty((0,))).shape[0]),
                    fields={k:list(v.shape) for k,v in arrays.items()})
                save_json(output/'progress.json',dict(status='inference',clip=ci,frame=i+1,clips=len(clips),elapsed_s=time.time()-started))
                print(json.dumps(dict(clip=ci,frame=i+1,humans=records[i]['humans'],elapsed_s=time.time()-started)),flush=True)
            with torch.no_grad():
                inference_recurrent_lighter(views,model,'cuda',verbose=False,use_ttt3r=False,
                    save_all_bg_gaussians=True,output_callback=callback,keep_outputs=False)
            manifest['clips'].append({k:v for k,v in clip.items() if k!='frames'}|dict(frames=records))
            save_json(output/f'clip_{ci:02d}.json',manifest['clips'][-1])
            del views; gc.collect(); torch.cuda.empty_cache()
        manifest.update(status='complete',elapsed_s=time.time()-started,
            max_gpu_memory_bytes=torch.cuda.max_memory_allocated())
        save_json(output/'manifest.json',manifest); save_json(output/'progress.json',dict(status='complete',elapsed_s=time.time()-started))
        print(json.dumps(dict(status='complete',frames=sum(len(c['frames']) for c in manifest['clips']),elapsed_s=manifest['elapsed_s'])),flush=True)
    finally:
        for hook in hooks: hook.remove()
if __name__=='__main__': main()
