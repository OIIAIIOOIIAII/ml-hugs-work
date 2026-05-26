#
# Unified evaluation for HUGS / anchor-attention Gaussian outputs.
#

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.append('.')

import hugs.trainer.gs_trainer as gst
from hugs.cfg.config import cfg as default_cfg
from hugs.losses.utils import ssim
from hugs.renderer.gs_renderer import render_human_scene
from hugs.trainer import GaussianTrainer
from hugs.utils.general import safe_state, create_video
from hugs.utils.image import psnr


def _natural_ckpt_key(path):
    name = Path(path).stem
    if name.endswith('final'):
        return (10**12, name)
    digits = ''.join(ch for ch in name if ch.isdigit())
    return (int(digits) if digits else -1, name)


def _find_ckpt(ckpt_dir, pattern):
    files = glob.glob(str(Path(ckpt_dir) / pattern))
    files += glob.glob(str(Path(ckpt_dir) / 'ckpt' / pattern))
    files = sorted(set(files), key=_natural_ckpt_key)
    return files[-1] if files else None




def _rgb_mask_bbox_from_dataset(dataset, data):
    if 'rgb' in data:
        rgb = data['rgb'].clamp(0, 1)
        mask = data.get('mask', None)
        bbox = data.get('bbox', None)
        return rgb, mask, bbox

    frame_idx = int(data.get('frame_idx', torch.tensor(0)).detach().cpu())
    rgb = None
    mask = None
    bbox = None

    if hasattr(dataset, 'scene') and hasattr(dataset.scene, 'captures'):
        img = dataset.scene.captures[frame_idx].captured_image.image
        img = (img[..., :3] / 255.0).astype(np.float32).transpose(2, 0, 1)
        rgb = torch.from_numpy(img).float().to(data['world_view_transform'].device)

    if hasattr(dataset, 'msk_lists') and frame_idx < len(dataset.msk_lists):
        msk = cv2.imread(dataset.msk_lists[frame_idx], cv2.IMREAD_GRAYSCALE)
        if msk is not None:
            msk = (msk.astype(np.float32) / 255.0)
            mask = torch.from_numpy(msk).float().to(data['world_view_transform'].device)
            rows = np.any(msk > 0.5, axis=0)
            cols = np.any(msk > 0.5, axis=1)
            if rows.any() and cols.any():
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                bbox = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32, device=mask.device)

    if rgb is None:
        raise KeyError('Could not find rgb in datum or reconstruct it from dataset')
    return rgb, mask, bbox

def _masked_psnr(pred, gt, mask, eps=1e-8):
    mask = mask.float()
    denom = mask.sum() * pred.shape[0]
    if denom.item() <= 0:
        return None
    mse = ((pred - gt).pow(2) * mask).sum() / denom.clamp_min(eps)
    return float((-10.0 * torch.log10(mse.clamp_min(eps))).detach().cpu())


def _masked_mae_255(pred, gt, mask):
    mask = mask.float()
    denom = mask.sum() * pred.shape[0]
    if denom.item() <= 0:
        return None
    mae = (torch.abs(pred - gt) * mask).sum() / denom.clamp_min(1e-8)
    return float((mae * 255.0).detach().cpu())


def _append_metric(store, key, value):
    if value is None:
        return
    if torch.is_tensor(value):
        value = float(value.detach().double().mean().cpu())
    store.setdefault(key, []).append(float(value))


def _mean_metrics(store):
    return {k: float(sum(v) / max(len(v), 1)) for k, v in store.items() if len(v) > 0}


def configure_from_output(output_dir, extras):
    output_dir = Path(output_dir)
    cfg_path = output_dir / 'config_train.yaml'
    if not cfg_path.exists():
        raise FileNotFoundError(f'Missing config_train.yaml under {output_dir}')

    cfg_file = OmegaConf.load(cfg_path)
    cfg = OmegaConf.merge(default_cfg, cfg_file, OmegaConf.from_cli(extras))
    cfg.eval = True
    cfg.logdir = str(output_dir)
    cfg.logdir_ckpt = str(output_dir / 'ckpt')
    cfg.train.anim_interval = -1

    human_ckpt = _find_ckpt(output_dir / 'ckpt', 'human_*.pth')
    scene_ckpt = _find_ckpt(output_dir / 'ckpt', 'scene_*.pth')
    anchor_ckpt = _find_ckpt(output_dir / 'ckpt', 'anchor_attention_*.pth')

    if cfg.mode in ['human', 'human_scene']:
        if human_ckpt is None:
            raise FileNotFoundError(f'Missing human checkpoint under {output_dir}/ckpt')
        cfg.human.ckpt = human_ckpt
    if cfg.mode in ['scene', 'human_scene']:
        if scene_ckpt is None:
            raise FileNotFoundError(f'Missing scene checkpoint under {output_dir}/ckpt')
        cfg.scene.ckpt = scene_ckpt
    if hasattr(cfg, 'anchor_attention') and anchor_ckpt is not None:
        cfg.anchor_attention.ckpt = anchor_ckpt

    return cfg, {'human': human_ckpt, 'scene': scene_ckpt, 'anchor_attention': anchor_ckpt}


@torch.no_grad()
def evaluate_all_frames(trainer, iter_label='final', keep_images=False, max_frames=-1, fps=20):
    if trainer.all_dataset is None:
        logger.warning('No all split dataset found; skipping full-sequence evaluation')
        return {}

    if trainer.human_gs:
        trainer.human_gs.eval()
    if getattr(trainer, 'anchor_attention', None) is not None:
        trainer.anchor_attention.eval()

    output_dir = Path(trainer.cfg.logdir)
    frame_dir = output_dir / f'eval_render_all_{iter_label}'
    frame_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}
    per_frame = []
    bg_color = trainer.bg_color
    eval_iter = trainer.cfg.train.num_steps

    for idx, data in enumerate(tqdm(trainer.all_dataset, desc='Evaluate all frames')):
        if max_frames > 0 and idx >= max_frames:
            break

        human_gs_out, scene_gs_out = None, None
        render_mode = trainer.cfg.mode

        if trainer.human_gs:
            human_gs_out = trainer.human_gs.forward(
                global_orient=data['global_orient'],
                body_pose=data['body_pose'],
                betas=data['betas'],
                transl=data['transl'],
                smpl_scale=data['smpl_scale'][None],
                dataset_idx=-1,
                is_train=False,
                ext_tfs=None,
            )
        if trainer.scene_gs:
            scene_gs_out = trainer.scene_gs.forward()

        human_gs_out, _ = trainer.maybe_apply_anchor_attention(
            human_gs_out,
            scene_gs_out,
            render_mode,
            eval_iter,
        )

        render_pkg = render_human_scene(
            data=data,
            human_gs_out=human_gs_out,
            scene_gs_out=scene_gs_out,
            bg_color=bg_color,
            render_mode=render_mode,
        )

        image = render_pkg['render'].clamp(0, 1)
        gt, data_mask, data_bbox = _rgb_mask_bbox_from_dataset(trainer.all_dataset, data)
        frame_metrics = {'frame': idx}

        frame_metrics['psnr'] = float(psnr(image, gt).mean().double().cpu())
        frame_metrics['ssim'] = float(ssim(image, gt).mean().double().cpu())
        _append_metric(metrics, 'all_psnr', frame_metrics['psnr'])
        _append_metric(metrics, 'all_ssim', frame_metrics['ssim'])

        if data_bbox is not None:
            bbox = data_bbox.to(int)
            cropped_gt = gt[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
            cropped_img = image[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
            if cropped_img.numel() > 0:
                frame_metrics['human_crop_psnr'] = float(psnr(cropped_img, cropped_gt).mean().double().cpu())
                frame_metrics['human_crop_ssim'] = float(ssim(cropped_img, cropped_gt).mean().double().cpu())
                _append_metric(metrics, 'human_crop_psnr', frame_metrics['human_crop_psnr'])
                _append_metric(metrics, 'human_crop_ssim', frame_metrics['human_crop_ssim'])

        if data_mask is not None:
            mask = data_mask
            if mask.dim() == 2:
                mask = mask[None]
            human_mask = (mask > 0.5).float()
            bg_mask = 1.0 - human_mask
            frame_metrics['human_mask_psnr'] = _masked_psnr(image, gt, human_mask)
            frame_metrics['bg_psnr'] = _masked_psnr(image, gt, bg_mask)
            frame_metrics['human_mae_255'] = _masked_mae_255(image, gt, human_mask)
            frame_metrics['bg_mae_255'] = _masked_mae_255(image, gt, bg_mask)
            _append_metric(metrics, 'human_mask_psnr', frame_metrics['human_mask_psnr'])
            _append_metric(metrics, 'bg_psnr', frame_metrics['bg_psnr'])
            _append_metric(metrics, 'human_mae_255', frame_metrics['human_mae_255'])
            _append_metric(metrics, 'bg_mae_255', frame_metrics['bg_mae_255'])

        per_frame.append(frame_metrics)
        if keep_images:
            torchvision.utils.save_image(image, frame_dir / f'{idx:05d}.png')

    mean_metrics = _mean_metrics(metrics)
    with (output_dir / f'eval_all_frames_{iter_label}.json').open('w', encoding='utf-8') as f:
        json.dump({'mean': mean_metrics, 'per_frame': per_frame}, f, indent=2)

    if keep_images:
        create_video(str(frame_dir), str(output_dir / f'eval_render_all_{iter_label}.mp4'), fps=fps)

    return mean_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained HUGS Gaussian outputs with a unified metric set.')
    parser.add_argument('-o', '--output-dir', required=True, help='Training output directory containing config_train.yaml and ckpt/')
    parser.add_argument('--skip-val', action='store_true', help='Skip trainer.validate().')
    parser.add_argument('--skip-all', action='store_true', help='Skip full-sequence all-split metrics.')
    parser.add_argument('--keep-images', action='store_true', help='Save full-sequence evaluated renders and mp4.')
    parser.add_argument('--max-frames', type=int, default=-1, help='Limit all-split evaluation frames for smoke tests.')
    parser.add_argument('--tag', default='final', help='Metric output suffix label.')
    parser.add_argument('--fps', type=int, default=20)
    args, extras = parser.parse_known_args()

    os.environ.setdefault('TORCH_HOME', '/hdd/u202420081000003/torch_cache')
    os.environ.setdefault('TMPDIR', '/hdd/u202420081000003/tmp')
    gst.get_anim_dataset = lambda cfg: None

    cfg, ckpts = configure_from_output(args.output_dir, extras)
    safe_state(seed=cfg.seed)
    logger.add(str(Path(cfg.logdir) / f'eval_unified_{args.tag}.log'), level='INFO')
    logger.info(f'Evaluating {args.output_dir}')
    logger.info(f'Checkpoints: {ckpts}')
    logger.info(OmegaConf.to_yaml(cfg))

    trainer = GaussianTrainer(cfg)
    out = {'checkpoints': ckpts}

    if not args.skip_val:
        validate_iter = None if args.tag == 'final' else int(args.tag) if str(args.tag).isdigit() else None
        trainer.validate(validate_iter)
        val_key = 'final' if validate_iter is None else f'{validate_iter:06d}'
        out['val'] = trainer.eval_metrics.get(val_key, {})

    if not args.skip_all:
        out['all_frames'] = evaluate_all_frames(
            trainer,
            iter_label=args.tag,
            keep_images=args.keep_images,
            max_frames=args.max_frames,
            fps=args.fps,
        )

    result_path = Path(cfg.logdir) / f'eval_unified_{args.tag}.json'
    with result_path.open('w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)
    print(f'EVAL_JSON={result_path}', flush=True)


if __name__ == '__main__':
    main()
