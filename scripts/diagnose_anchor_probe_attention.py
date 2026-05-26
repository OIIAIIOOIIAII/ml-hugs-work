# Diagnose anchor scene probe and attention for HUGS outputs.

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from contact_eval_lib import (
    anchor_world_from_bindings,
    draw_anchor_overlay,
    dump_json,
    forward_frame,
    get_rgb_mask_bbox,
    make_trainer,
    project_points,
    render_frame,
    save_image_tensor,
    summarize_values,
    write_ply,
)


def main():
    parser = argparse.ArgumentParser(description='Diagnose anchor scene query / attention for one initialized HUGS output.')
    parser.add_argument('-o', '--output-dir', required=True)
    parser.add_argument('--out-dir', default='')
    parser.add_argument('--tag', default='probe')
    parser.add_argument('--anchors', default='')
    parser.add_argument('--max-frames', type=int, default=-1)
    parser.add_argument('--frame-stride', type=int, default=5)
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--scene-topk', type=int, default=-1)
    parser.add_argument('--scene-opacity-threshold', type=float, default=-1.0)
    parser.add_argument('--anchor-attention-ckpt', default='')
    args, extras = parser.parse_known_args()

    more = list(extras)
    if args.scene_topk > 0:
        more.append(f'anchor_attention.scene_topk={args.scene_topk}')
    if args.scene_opacity_threshold >= 0:
        more.append(f'anchor_attention.scene_opacity_threshold={args.scene_opacity_threshold}')

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.output_dir) / 'anchor_probe_diagnostics' / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer, ckpts = make_trainer(
        args.output_dir,
        extras=more,
        enable_anchors=True,
        enable_attention=True,
        anchor_ckpt=args.anchor_attention_ckpt or None,
    )
    if trainer.anchor_data is None or trainer.anchor_attention is None:
        raise RuntimeError('Anchor attention was not initialized')
    names = list(trainer.anchor_data['anchor_names'])
    if args.anchors:
        wanted = [a.strip() for a in args.anchors.split(',') if a.strip()]
        anchor_ids = [names.index(a) for a in wanted if a in names]
    else:
        anchor_ids = list(range(len(names)))
    anchor_names = [names[i] for i in anchor_ids]

    rows = []
    ply_points = []
    ply_colors = []
    frame_indices = list(range(0, len(trainer.all_dataset), max(1, args.frame_stride)))
    if args.max_frames > 0:
        frame_indices = frame_indices[:args.max_frames]

    for local_i, ds_idx in enumerate(tqdm(frame_indices, desc='Probe diagnosis')):
        data = trainer.all_dataset[ds_idx]
        frame_idx = int(data['frame_idx'].detach().cpu())
        human_out, scene_out, stats = forward_frame(trainer, data, apply_attention=True)
        render = render_frame(trainer, data, human_out, scene_out)
        gt, _, _ = get_rgb_mask_bbox(trainer.all_dataset, data)
        if 'anchor_world' in stats:
            anchor_world = stats['anchor_world'].to(render.device)
        else:
            anchor_world = anchor_world_from_bindings(
                human_out,
                trainer.human_gs.anchor_ids,
                trainer.human_gs.anchor_weights,
                len(names),
                top_m=int(getattr(trainer.cfg.anchor_attention, 'top_m', 2)),
            )
        xy, depth = project_points(data, anchor_world[anchor_ids])
        knn_idx = stats.get('scene_knn_idx')
        knn_dist = stats.get('scene_knn_dist')
        attn = stats.get('attention_weights')
        if knn_idx is None or knn_dist is None:
            raise RuntimeError('Anchor attention did not return scene_knn_idx/scene_knn_dist')
        knn_idx = knn_idx.detach().cpu()
        knn_dist = knn_dist.detach().cpu()
        attn = attn.detach().cpu() if attn is not None else None
        scene_opacity = scene_out['opacity'].detach().reshape(-1).cpu()
        scene_scale = scene_out['scales'].detach().cpu()
        scene_xyz = scene_out['xyz'].detach().cpu().numpy()

        overlay_xy, overlay_names = [], []
        for local_anchor_i, aid in enumerate(anchor_ids):
            d = knn_dist[aid].numpy()
            idx = knn_idx[aid].numpy().astype(np.int64)
            op = scene_opacity[idx].numpy()
            sc = scene_scale[idx].numpy()
            if attn is not None:
                w = attn[aid].numpy()
                entropy = float(-(np.clip(w, 1e-8, 1.0) * np.log(np.clip(w, 1e-8, 1.0))).sum())
                max_attn = float(w.max())
                weighted_dist = float((w * d).sum())
            else:
                entropy = None
                max_attn = None
                weighted_dist = None
            px, py = float(xy[local_anchor_i, 0].detach().cpu()), float(xy[local_anchor_i, 1].detach().cpu())
            row = {
                'frame_idx': frame_idx,
                'anchor_id': aid,
                'anchor_name': names[aid],
                'anchor_x': float(anchor_world[aid, 0].detach().cpu()),
                'anchor_y': float(anchor_world[aid, 1].detach().cpu()),
                'anchor_z': float(anchor_world[aid, 2].detach().cpu()),
                'proj_x': px,
                'proj_y': py,
                'proj_depth': float(depth[local_anchor_i].detach().cpu()),
                'nearest_scene_distance': float(d.min()),
                'median_scene_distance': float(np.median(d)),
                'trimmed_scene_distance': float(np.mean(np.sort(d)[:max(1, min(len(d), max(2, len(d)//4)))])),
                'attention_entropy': '' if entropy is None else entropy,
                'max_attention': '' if max_attn is None else max_attn,
                'attention_weighted_distance': '' if weighted_dist is None else weighted_dist,
                'mean_scene_opacity': float(op.mean()),
                'max_scene_opacity': float(op.max()),
                'mean_scene_scale': float(sc.mean()),
                'max_scene_scale': float(sc.max()),
                'scene_candidate_count': int(len(idx)),
            }
            rows.append(row)
            overlay_xy.append((px, py))
            overlay_names.append(names[aid])
            if args.save_every > 0 and local_i % args.save_every == 0:
                ply_points.append(anchor_world[aid].detach().cpu().numpy())
                ply_colors.append([255, 0, 0])
                take = idx[:min(16, len(idx))]
                ply_points.extend(scene_xyz[take])
                ply_colors.extend([[60, 160, 255]] * len(take))
        if args.save_every > 0 and local_i % args.save_every == 0:
            draw_anchor_overlay(gt, render, overlay_xy, overlay_names, out_dir / 'overlays' / f'{frame_idx:05d}.png')
            save_image_tensor(out_dir / 'renders' / f'{frame_idx:05d}_render.png', render)
            save_image_tensor(out_dir / 'renders' / f'{frame_idx:05d}_gt.png', gt)

    csv_path = out_dir / 'anchor_probe_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    summary = {'output_dir': args.output_dir, 'tag': args.tag, 'checkpoints': ckpts, 'anchors': anchor_names, 'num_rows': len(rows)}
    for key in [
        'nearest_scene_distance', 'median_scene_distance', 'trimmed_scene_distance',
        'attention_entropy', 'max_attention', 'attention_weighted_distance',
        'mean_scene_opacity', 'max_scene_opacity', 'mean_scene_scale', 'max_scene_scale'
    ]:
        vals = [r[key] for r in rows if r.get(key) != '']
        summary[key] = summarize_values(vals)
    dump_json(out_dir / 'anchor_probe_summary.json', summary)
    if ply_points:
        write_ply(out_dir / 'anchor_probe_scene_samples.ply', np.asarray(ply_points, dtype=np.float32), np.asarray(ply_colors, dtype=np.uint8))
    print(f'PROBE_SUMMARY={out_dir / "anchor_probe_summary.json"}')
    print(f'PROBE_CSV={csv_path}')


if __name__ == '__main__':
    main()
