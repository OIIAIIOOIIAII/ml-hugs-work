# Contact proxy evaluation for HUGS outputs, with optional DECO contact anchors.

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from contact_eval_lib import (
    aggregate_deco_contact_to_anchors,
    anchor_world_from_bindings,
    crop_tensor,
    deco_contact_obj_path,
    deco_contact_png_path,
    draw_anchor_overlay,
    dump_json,
    forward_frame,
    get_rgb_mask_bbox,
    make_trainer,
    project_points,
    posed_semantic_anchor_world,
    psnr_value,
    read_deco_pred_obj,
    render_frame,
    save_image_tensor,
    ssim_value,
    summarize_values,
    write_ply,
)
from hugs.utils.anchor_utils import generate_semantic_anchor_vertices, load_smpl_template

DEFAULT_FOOT_ANCHORS = ['left_sole', 'right_sole', 'left_toe', 'right_toe', 'left_heel', 'right_heel']


def load_proxy(path, device, max_points):
    data = np.load(path)
    pts = data['points'].astype(np.float32)
    cols = data['colors'].astype(np.uint8) if 'colors' in data else np.full((pts.shape[0], 3), 180, np.uint8)
    if max_points > 0 and pts.shape[0] > max_points:
        rng = np.random.default_rng(1)
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    return torch.from_numpy(pts).to(device), pts, cols


def knn_stats(point, proxy, k):
    d = torch.linalg.norm(proxy - point[None], dim=-1)
    k = min(int(k), d.numel())
    vals, idx = torch.topk(d, k=k, largest=False)
    return vals, idx


def parse_list(text):
    return [a.strip() for a in text.split(',') if a.strip()]


def build_deco_anchor_vertex_ids(anchor_names, model_path='data/smpl', count_per_anchor=64):
    template = load_smpl_template(model_path=model_path, canonical_pose='t_pose')
    anchors = generate_semantic_anchor_vertices(template, count_per_anchor=count_per_anchor)
    return [anchors[name] for name in anchor_names]


def choose_overlay_anchor_ids(rows_for_frame, anchor_ids, source, top_k):
    if source == 'deco-anchor':
        contact = [r['anchor_id'] for r in rows_for_frame if int(r.get('is_deco_contact', 0)) == 1]
        if contact:
            return contact
        if top_k > 0:
            ordered = sorted(rows_for_frame, key=lambda r: float(r.get('deco_contact_prob', 0.0)), reverse=True)
            return [r['anchor_id'] for r in ordered[:top_k]]
        return []
    return anchor_ids


def main():
    parser = argparse.ArgumentParser(description='Evaluate contact proxy metrics and save contact visualizations.')
    parser.add_argument('-o', '--output-dir', required=True)
    parser.add_argument('--proxy-npz', required=True)
    parser.add_argument('--out-dir', default='')
    parser.add_argument('--tag', default='exp0_full')
    parser.add_argument('--contact-source', choices=['feet', 'deco-anchor'], default='feet')
    parser.add_argument('--anchors', default=','.join(DEFAULT_FOOT_ANCHORS))
    parser.add_argument('--deco-dir', default='')
    parser.add_argument('--deco-contact-threshold', type=float, default=0.01)
    parser.add_argument('--deco-top-anchors', type=int, default=6)
    parser.add_argument('--deco-anchor-verts', type=int, default=64)
    parser.add_argument('--overlay-anchor-position', choices=['semantic', 'binding'], default='semantic')
    parser.add_argument('--smpl-model-path', default='data/smpl')
    parser.add_argument('--max-frames', type=int, default=-1)
    parser.add_argument('--frame-stride', type=int, default=1)
    parser.add_argument('--proxy-max-points', type=int, default=200000)
    parser.add_argument('--knn', type=int, default=32)
    parser.add_argument('--roi-size', type=int, default=96)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--contact-thresholds', default='0.02,0.05,0.10')
    args, extras = parser.parse_known_args()

    if args.contact_source == 'deco-anchor' and not args.deco_dir:
        raise ValueError('--deco-dir is required when --contact-source deco-anchor')

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.output_dir) / 'contact_eval' / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer, ckpts = make_trainer(args.output_dir, extras=extras, enable_anchors=True, enable_attention=False)
    if trainer.anchor_data is None:
        raise RuntimeError('Anchor data was not initialized. Check cfg.anchor_attention.use_anchors.')
    names = list(trainer.anchor_data['anchor_names'])

    if args.contact_source == 'deco-anchor':
        anchor_names = names
        anchor_ids = list(range(len(names)))
        deco_anchor_vertex_ids = build_deco_anchor_vertex_ids(
            anchor_names,
            model_path=args.smpl_model_path,
            count_per_anchor=args.deco_anchor_verts,
        )
    else:
        wanted = parse_list(args.anchors)
        anchor_ids = [names.index(a) for a in wanted if a in names]
        anchor_names = [names[i] for i in anchor_ids]
        deco_anchor_vertex_ids = None
        if not anchor_ids:
            raise ValueError(f'No requested anchors found. Available: {names}')

    device = trainer.bg_color.device
    proxy_t, proxy_np, proxy_cols = load_proxy(args.proxy_npz, device, args.proxy_max_points)
    thresholds = [float(v) for v in args.contact_thresholds.split(',') if v.strip()]

    rows = []
    prev_anchor_world_all = None
    frame_indices = list(range(0, len(trainer.all_dataset), max(1, args.frame_stride)))
    if args.max_frames > 0:
        frame_indices = frame_indices[:args.max_frames]

    missing_deco_frames = []
    deco_vertex_counts = []
    for local_i, ds_idx in enumerate(tqdm(frame_indices, desc='Contact eval')):
        data = trainer.all_dataset[ds_idx]
        frame_idx = int(data['frame_idx'].detach().cpu())
        human_out, scene_out, _ = forward_frame(trainer, data, apply_attention=False)
        render = render_frame(trainer, data, human_out, scene_out)
        gt, mask, _ = get_rgb_mask_bbox(trainer.all_dataset, data)
        aw_all = anchor_world_from_bindings(
            human_out,
            trainer.human_gs.anchor_ids,
            trainer.human_gs.anchor_weights,
            len(names),
            top_m=int(getattr(trainer.cfg.anchor_attention, 'top_m', 2)),
        )
        overlay_aw_all = aw_all
        if args.overlay_anchor_position == 'semantic':
            overlay_aw_all = posed_semantic_anchor_world(trainer, data, frame_idx, use_optimized_pose=False)

        deco_probs = {name: 0.0 for name in names}
        deco_counts = {name: 0 for name in names}
        deco_contact_vertices = 0
        if args.contact_source == 'deco-anchor':
            obj_path = deco_contact_obj_path(args.deco_dir, frame_idx)
            verts, _, contact = read_deco_pred_obj(obj_path)
            if contact.size == 0:
                missing_deco_frames.append(frame_idx)
            else:
                deco_vertex_counts.append(int(contact.shape[0]))
                deco_contact_vertices = int(contact.sum())
                deco_probs, deco_counts = aggregate_deco_contact_to_anchors(contact, deco_anchor_vertex_ids, anchor_names)
            png_path = deco_contact_png_path(args.deco_dir, frame_idx)
            if png_path.exists() and args.save_every > 0 and local_i % args.save_every == 0:
                target = out_dir / 'deco_preds' / f'{frame_idx:05d}.png'
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(png_path, target)

        selected = aw_all[anchor_ids]
        overlay_selected = overlay_aw_all[anchor_ids]
        xy, depth = project_points(data, overlay_selected)
        velocities = [None] * len(anchor_ids)
        if prev_anchor_world_all is not None:
            diff = selected - prev_anchor_world_all[anchor_ids]
            velocities = torch.linalg.norm(diff, dim=-1).detach().cpu().tolist()
        prev_anchor_world_all = aw_all.detach().clone()

        frame_rows = []
        for j, (aid, aname) in enumerate(zip(anchor_ids, anchor_names)):
            vals, _ = knn_stats(selected[j], proxy_t, args.knn)
            vals_cpu = vals.detach().cpu().numpy()
            nearest = float(vals_cpu[0])
            median = float(np.median(vals_cpu))
            trimmed = float(np.mean(vals_cpu[:max(1, min(len(vals_cpu), max(2, len(vals_cpu)//4)))]))
            px, py = float(xy[j, 0].detach().cpu()), float(xy[j, 1].detach().cpu())
            in_view = 0 <= px < int(data['image_width']) and 0 <= py < int(data['image_height'])
            crop_r, crop_box = crop_tensor(render, xy[j], args.roi_size)
            crop_g, _ = crop_tensor(gt, xy[j], args.roi_size)
            roi_psnr = None
            roi_ssim = None
            deco_prob = float(deco_probs.get(aname, 0.0))
            is_deco_contact = int(args.contact_source == 'deco-anchor' and deco_prob >= args.deco_contact_threshold)
            save_roi = args.contact_source != 'deco-anchor' or is_deco_contact or (args.deco_top_anchors > 0 and deco_prob > 0)
            if crop_r is not None and crop_g is not None and crop_r.shape == crop_g.shape:
                roi_psnr = psnr_value(crop_r, crop_g)
                roi_ssim = ssim_value(crop_r, crop_g)
                if save_roi and args.save_every > 0 and local_i % args.save_every == 0:
                    roi_dir = out_dir / 'contact_rois' / f'{aname}'
                    save_image_tensor(roi_dir / f'{frame_idx:05d}_render.png', crop_r)
                    save_image_tensor(roi_dir / f'{frame_idx:05d}_gt.png', crop_g)
                    save_image_tensor(roi_dir / f'{frame_idx:05d}_absdiff.png', (crop_r - crop_g).abs())
            row = {
                'frame_idx': frame_idx,
                'contact_source': args.contact_source,
                'anchor_id': aid,
                'anchor_name': aname,
                'anchor_x': float(selected[j, 0].detach().cpu()),
                'anchor_y': float(selected[j, 1].detach().cpu()),
                'anchor_z': float(selected[j, 2].detach().cpu()),
                'overlay_anchor_x': float(overlay_selected[j, 0].detach().cpu()),
                'overlay_anchor_y': float(overlay_selected[j, 1].detach().cpu()),
                'overlay_anchor_z': float(overlay_selected[j, 2].detach().cpu()),
                'proj_x': px,
                'proj_y': py,
                'proj_depth': float(depth[j].detach().cpu()),
                'in_view': int(in_view),
                'nearest_proxy_dist': nearest,
                'median_proxy_dist': median,
                'trimmed_proxy_dist': trimmed,
                'velocity': '' if velocities[j] is None else float(velocities[j]),
                'roi_psnr': '' if roi_psnr is None else roi_psnr,
                'roi_ssim': '' if roi_ssim is None else roi_ssim,
                'deco_contact_prob': deco_prob,
                'is_deco_contact': is_deco_contact,
                'deco_anchor_vertex_count': int(deco_counts.get(aname, 0)),
                'deco_contact_vertices_frame': deco_contact_vertices,
            }
            for tau in thresholds:
                row[f'within_{tau:.3f}m'] = int(nearest <= tau)
            rows.append(row)
            frame_rows.append(row)

        if args.save_every > 0 and local_i % args.save_every == 0:
            overlay_ids = choose_overlay_anchor_ids(frame_rows, anchor_ids, args.contact_source, args.deco_top_anchors)
            overlay_xy = []
            overlay_names = []
            by_id = {r['anchor_id']: r for r in frame_rows}
            for oid in overlay_ids:
                if oid not in by_id:
                    continue
                r = by_id[oid]
                overlay_xy.append((r['proj_x'], r['proj_y']))
                if args.contact_source == 'deco-anchor':
                    overlay_names.append(f"{r['anchor_name']}:{float(r['deco_contact_prob']):.2f}")
                else:
                    overlay_names.append(r['anchor_name'])
            draw_anchor_overlay(gt, render, overlay_xy, overlay_names, out_dir / 'overlays' / f'{frame_idx:05d}.png')
            save_image_tensor(out_dir / 'renders' / f'{frame_idx:05d}_render.png', render)
            save_image_tensor(out_dir / 'renders' / f'{frame_idx:05d}_gt.png', gt)

    csv_path = out_dir / 'per_frame_contact_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    contact_rows = [r for r in rows if args.contact_source != 'deco-anchor' or int(r.get('is_deco_contact', 0)) == 1]
    summary = {
        'output_dir': args.output_dir,
        'proxy_npz': args.proxy_npz,
        'checkpoints': ckpts,
        'contact_source': args.contact_source,
        'anchors': anchor_names,
        'num_rows': len(rows),
        'num_contact_rows': len(contact_rows),
        'deco_dir': args.deco_dir,
        'deco_contact_threshold': args.deco_contact_threshold,
        'overlay_anchor_position': args.overlay_anchor_position,
        'missing_deco_frames': missing_deco_frames,
        'deco_vertex_counts': summarize_values(deco_vertex_counts),
    }
    for prefix, subset in [('all', rows), ('contact', contact_rows)]:
        for key in ['nearest_proxy_dist', 'median_proxy_dist', 'trimmed_proxy_dist', 'roi_psnr', 'roi_ssim', 'deco_contact_prob']:
            vals = [r[key] for r in subset if r.get(key) != '']
            summary[f'{prefix}_{key}'] = summarize_values(vals)
        for tau in thresholds:
            key = f'within_{tau:.3f}m'
            vals = [float(r[key]) for r in subset]
            summary[f'{prefix}_contact_recall_proxy_{tau:.3f}m'] = float(np.mean(vals)) if vals else None
    dump_json(out_dir / 'contact_eval_summary.json', summary)

    pts = np.array([[r['anchor_x'], r['anchor_y'], r['anchor_z']] for r in rows], dtype=np.float32)
    cols = []
    for r in rows:
        if args.contact_source == 'deco-anchor' and int(r.get('is_deco_contact', 0)) == 1:
            cols.append([0, 255, 0])
        elif r['nearest_proxy_dist'] <= 0.05:
            cols.append([255, 60, 60])
        else:
            cols.append([255, 180, 40])
    if pts.shape[0] > 0:
        write_ply(out_dir / 'contact_anchor_trajectory.ply', pts, np.asarray(cols, dtype=np.uint8))
    proxy_take = min(proxy_np.shape[0], 50000)
    write_ply(out_dir / 'scene_proxy_sample.ply', proxy_np[:proxy_take], proxy_cols[:proxy_take])
    print(f'CONTACT_SUMMARY={out_dir / "contact_eval_summary.json"}')
    print(f'CONTACT_CSV={csv_path}')


if __name__ == '__main__':
    main()
