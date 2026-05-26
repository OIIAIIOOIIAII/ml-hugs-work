# Export ExpE-style semantic anchor overlays and compare them with binding-center anchors.

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from contact_eval_lib import (
    anchor_world_from_bindings,
    forward_frame,
    get_rgb_mask_bbox,
    make_trainer,
    project_points,
    render_frame,
)
from hugs.renderer.gs_renderer import render_human_scene
from hugs.utils.rotations import rotation_6d_to_axis_angle

CONTACT_ANCHORS = ['left_sole', 'right_sole', 'left_toe', 'right_toe', 'left_heel', 'right_heel']


def human_forward_optimized(trainer, data, frame_idx):
    human_gs = trainer.human_gs
    has_optimized_pose = hasattr(human_gs, 'global_orient') and hasattr(human_gs, 'body_pose') and hasattr(human_gs, 'transl')
    if has_optimized_pose:
        return human_gs.forward(
            smpl_scale=data['smpl_scale'][None] if data['smpl_scale'].dim() == 0 else data['smpl_scale'],
            dataset_idx=int(frame_idx),
            is_train=False,
            ext_tfs=None,
        )
    return human_gs.forward(
        global_orient=data['global_orient'],
        body_pose=data['body_pose'],
        betas=data['betas'],
        transl=data['transl'],
        smpl_scale=data['smpl_scale'][None],
        dataset_idx=-1,
        is_train=False,
        ext_tfs=None,
    )


@torch.no_grad()
def posed_semantic_anchor_world(trainer, data, frame_idx, use_optimized_pose):
    hg = trainer.human_gs
    if use_optimized_pose and hasattr(hg, 'global_orient'):
        global_orient = rotation_6d_to_axis_angle(hg.global_orient[int(frame_idx)].reshape(-1, 6)).reshape(3)
    else:
        global_orient = data['global_orient']
    if use_optimized_pose and hasattr(hg, 'body_pose'):
        body_pose = rotation_6d_to_axis_angle(hg.body_pose[int(frame_idx)].reshape(-1, 6)).reshape(23 * 3)
    else:
        body_pose = data['body_pose']
    betas = hg.betas if use_optimized_pose and hasattr(hg, 'betas') else data['betas']
    transl = hg.transl[int(frame_idx)] if use_optimized_pose and hasattr(hg, 'transl') else data['transl']

    smpl_out = hg.smpl_template(
        betas=betas.unsqueeze(0),
        body_pose=body_pose.unsqueeze(0),
        global_orient=global_orient.unsqueeze(0),
        disable_posedirs=False,
    )
    verts = smpl_out.vertices[0]
    scale = data['smpl_scale']
    verts = verts * (scale.reshape(-1)[0] if scale.dim() > 0 else scale)
    verts = verts + transl.reshape(1, 3)

    names = list(trainer.anchor_data['anchor_names'])
    vertex_ids = trainer.anchor_data['anchors']['vertex_ids']
    pts = []
    for a in range(len(names)):
        ids = torch.as_tensor(vertex_ids[a], dtype=torch.long, device=verts.device)
        pts.append(verts[ids].mean(dim=0))
    return torch.stack(pts, dim=0)


def to_np_img(tensor):
    return (tensor.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)


def draw_compare(gt, render, sem_xy, bind_xy, names, out_path, title):
    gt_np = to_np_img(gt)
    rd_np = to_np_img(render)
    canvas = np.concatenate([gt_np, rd_np], axis=1)
    h, w = gt_np.shape[:2]
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1] - 1, 760), 28), (255, 255, 255), -1)
    cv2.putText(canvas, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'green=ExpE semantic anchor, magenta=binding-center anchor', (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, 'green=ExpE semantic anchor, magenta=binding-center anchor', (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

    for side_offset in [0, w]:
        for i, name in enumerate(names):
            sx, sy = float(sem_xy[i, 0]), float(sem_xy[i, 1])
            bx, by = float(bind_xy[i, 0]), float(bind_xy[i, 1])
            if 0 <= sx < w and 0 <= sy < h and 0 <= bx < w and 0 <= by < h:
                cv2.line(canvas, (int(round(sx)) + side_offset, int(round(sy))), (int(round(bx)) + side_offset, int(round(by))), (255, 255, 255), 2, cv2.LINE_AA)
            if 0 <= sx < w and 0 <= sy < h:
                cv2.circle(canvas, (int(round(sx)) + side_offset, int(round(sy))), 7, (40, 230, 80), -1, cv2.LINE_AA)
                cv2.circle(canvas, (int(round(sx)) + side_offset, int(round(sy))), 7, (0, 40, 0), 2, cv2.LINE_AA)
            if 0 <= bx < w and 0 <= by < h:
                cv2.circle(canvas, (int(round(bx)) + side_offset, int(round(by))), 8, (230, 50, 230), 2, cv2.LINE_AA)
            lx, ly = int(round(sx)) + side_offset + 8, int(round(sy)) - 6
            if 0 <= sx < w and 0 <= sy < h:
                cv2.putText(canvas, name.replace('_', ' '), (lx, max(12, ly)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(canvas, name.replace('_', ' '), (lx, max(12, ly)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (40, 230, 80), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-dir', required=True)
    parser.add_argument('--out-dir', default='')
    parser.add_argument('--frames', default='0')
    parser.add_argument('--anchors', default=','.join(CONTACT_ANCHORS))
    parser.add_argument('--render-pose', choices=['dataset', 'optimized'], default='optimized')
    parser.add_argument('--semantic-pose', choices=['dataset', 'optimized'], default='optimized')
    args, extras = parser.parse_known_args()

    trainer, _ = make_trainer(args.output_dir, extras=extras, enable_anchors=True, enable_attention=False)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.output_dir) / 'anchor_overlay_compare' / f'expe_compare_{args.render_pose}_render_{args.semantic_pose}_semantic'
    out_dir.mkdir(parents=True, exist_ok=True)

    names_all = list(trainer.anchor_data['anchor_names'])
    wanted = [x.strip() for x in args.anchors.split(',') if x.strip()]
    anchor_ids = [names_all.index(x) for x in wanted]
    anchor_names = [names_all[i] for i in anchor_ids]
    frames = [int(x) for x in args.frames.split(',') if x.strip()]

    rows = []
    for ds_idx in frames:
        data = trainer.all_dataset[ds_idx]
        frame_idx = int(data['frame_idx'].detach().cpu())
        if args.render_pose == 'optimized':
            human_out = human_forward_optimized(trainer, data, frame_idx)
            scene_out = trainer.scene_gs.forward() if trainer.scene_gs else None
        else:
            human_out, scene_out, _ = forward_frame(trainer, data, apply_attention=False)
        render = render_human_scene(data=data, human_gs_out=human_out, scene_gs_out=scene_out, bg_color=trainer.bg_color, render_mode=trainer.cfg.mode)['render'].clamp(0, 1)
        gt, _, _ = get_rgb_mask_bbox(trainer.all_dataset, data)

        bind_world = anchor_world_from_bindings(
            human_out,
            trainer.human_gs.anchor_ids,
            trainer.human_gs.anchor_weights,
            len(names_all),
            top_m=int(getattr(trainer.cfg.anchor_attention, 'top_m', 2)),
        )
        sem_world = posed_semantic_anchor_world(trainer, data, frame_idx, use_optimized_pose=(args.semantic_pose == 'optimized'))
        bind_xy, _ = project_points(data, bind_world[anchor_ids])
        sem_xy, _ = project_points(data, sem_world[anchor_ids])

        draw_compare(
            gt,
            render,
            sem_xy.detach().cpu().numpy(),
            bind_xy.detach().cpu().numpy(),
            anchor_names,
            out_dir / 'overlays' / f'{frame_idx:05d}.png',
            f'frame {frame_idx:05d} | render={args.render_pose} | semantic={args.semantic_pose}',
        )
        for j, aid in enumerate(anchor_ids):
            delta = bind_xy[j] - sem_xy[j]
            rows.append({
                'frame_idx': frame_idx,
                'anchor_id': aid,
                'anchor_name': names_all[aid],
                'semantic_x': float(sem_xy[j, 0].detach().cpu()),
                'semantic_y': float(sem_xy[j, 1].detach().cpu()),
                'binding_x': float(bind_xy[j, 0].detach().cpu()),
                'binding_y': float(bind_xy[j, 1].detach().cpu()),
                'pixel_delta_x_binding_minus_semantic': float(delta[0].detach().cpu()),
                'pixel_delta_y_binding_minus_semantic': float(delta[1].detach().cpu()),
                'world_delta_m': float(torch.linalg.norm(bind_world[aid] - sem_world[aid]).detach().cpu()),
            })

    csv_path = out_dir / 'anchor_overlay_compare.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'OVERLAY_DIR={out_dir / "overlays"}')
    print(f'CSV={csv_path}')


if __name__ == '__main__':
    main()
