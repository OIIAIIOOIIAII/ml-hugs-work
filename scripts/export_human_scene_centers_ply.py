#!/usr/bin/env python3
"""Export posed human + scene Gaussian centers as a colored point-cloud PLY.

This is intentionally a point-cloud export, not a full 3DGS splat export. The
human centers are produced through the same HUGS forward path used for rendering,
so they are in the posed/world coordinates of the selected dataset frame.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf
from plyfile import PlyData, PlyElement

from hugs.datasets import NeumanDataset, Human3RDataset
from hugs.models import SceneGS
from hugs.models.hugs_trimlp import HUGS_TRIMLP
from hugs.models.hugs_wo_trimlp import HUGS_WO_TRIMLP


def parse_color(text):
    vals = [int(x) for x in text.split(',')]
    if len(vals) != 3 or any(v < 0 or v > 255 for v in vals):
        raise argparse.ArgumentTypeError('color must be R,G,B with values in [0,255]')
    return vals


def load_dataset(cfg, split):
    if cfg.dataset.name == 'neuman':
        return NeumanDataset(
            cfg.dataset.seq,
            split,
            cfg.mode,
            init_pcd_path=getattr(cfg.scene, 'init_pcd_path', None),
        )
    if cfg.dataset.name == 'human3r':
        return Human3RDataset(cfg.dataset_path, split, render_mode=cfg.mode)
    raise ValueError(f'Unsupported dataset: {cfg.dataset.name}')


def build_human_model(cfg, betas):
    kwargs = dict(
        sh_degree=cfg.human.sh_degree,
        n_subdivision=cfg.human.n_subdivision,
        use_surface=cfg.human.use_surface,
        init_2d=cfg.human.init_2d,
        rotate_sh=cfg.human.rotate_sh,
        isotropic=cfg.human.isotropic,
        init_scale_multiplier=cfg.human.init_scale_multiplier,
    )
    if cfg.human.name == 'hugs_trimlp':
        model = HUGS_TRIMLP(
            **kwargs,
            n_features=32,
            use_deformer=cfg.human.use_deformer,
            disable_posedirs=cfg.human.disable_posedirs,
            triplane_res=cfg.human.triplane_res,
            betas=betas,
        )
    elif cfg.human.name == 'hugs_wo_trimlp':
        model = HUGS_WO_TRIMLP(**kwargs)
        model.create_betas(betas, False)
        model.initialize()
    else:
        raise ValueError(f'Unsupported human model: {cfg.human.name}')
    return model


def write_colored_ply(out_path, scene_xyz, human_xyz, scene_color, human_color):
    scene_xyz = np.asarray(scene_xyz, dtype=np.float32)
    human_xyz = np.asarray(human_xyz, dtype=np.float32)
    n_scene = scene_xyz.shape[0]
    n_human = human_xyz.shape[0]
    xyz = np.concatenate([scene_xyz, human_xyz], axis=0)

    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('part', 'u1'),
    ]
    verts = np.empty(xyz.shape[0], dtype=dtype)
    verts['x'] = xyz[:, 0]
    verts['y'] = xyz[:, 1]
    verts['z'] = xyz[:, 2]
    verts['red'][:n_scene] = scene_color[0]
    verts['green'][:n_scene] = scene_color[1]
    verts['blue'][:n_scene] = scene_color[2]
    verts['part'][:n_scene] = 0
    verts['red'][n_scene:] = human_color[0]
    verts['green'][n_scene:] = human_color[1]
    verts['blue'][n_scene:] = human_color[2]
    verts['part'][n_scene:] = 1

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(verts, 'vertex')], text=False).write(str(out_path))
    return n_scene, n_human


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True, help='Experiment logdir containing config_train.yaml and ckpt/*.pth')
    parser.add_argument('--out', required=True, help='Output colored point-cloud PLY path')
    parser.add_argument('--frame-index', type=int, default=0, help='Dataset frame used to pose the human centers')
    parser.add_argument('--split', default='all', choices=['train', 'val', 'all'], help='Dataset split for frame-index')
    parser.add_argument('--scene-color', type=parse_color, default=parse_color('70,150,255'))
    parser.add_argument('--human-color', type=parse_color, default=parse_color('255,80,40'))
    parser.add_argument('--suffix', default='final', help='Checkpoint suffix, e.g. final or 015000')
    args = parser.parse_args()

    logdir = Path(args.logdir)
    cfg = OmegaConf.load(logdir / 'config_train.yaml')
    dataset = load_dataset(cfg, args.split)
    data = dataset[args.frame_index]

    human_ckpt = torch.load(logdir / 'ckpt' / f'human_{args.suffix}.pth', map_location='cuda', weights_only=False)
    scene_ckpt = torch.load(logdir / 'ckpt' / f'scene_{args.suffix}.pth', map_location='cuda', weights_only=False)

    human = build_human_model(cfg, data['betas'])
    human.load_state_dict(human_ckpt, cfg.human.lr)
    scene = SceneGS(sh_degree=cfg.scene.sh_degree)
    scene.restore(scene_ckpt, cfg.scene.lr)

    human_out = human.forward(
        global_orient=data['global_orient'],
        body_pose=data['body_pose'],
        betas=data['betas'],
        transl=data['transl'],
        smpl_scale=data['smpl_scale'][None],
        dataset_idx=-1,
        is_train=False,
        ext_tfs=None,
    )
    scene_out = scene.forward()

    n_scene, n_human = write_colored_ply(
        args.out,
        scene_out['xyz'].detach().cpu().numpy(),
        human_out['xyz'].detach().cpu().numpy(),
        args.scene_color,
        args.human_color,
    )
    print(f'Wrote {args.out}')
    print(f'scene_points={n_scene} human_points={n_human} frame_index={args.frame_index} split={args.split}')
    print(f'scene_color={args.scene_color} human_color={args.human_color} part: 0=scene, 1=human')


if __name__ == '__main__':
    main()
