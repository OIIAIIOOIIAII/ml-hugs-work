#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import os
import glob
import shutil
from pathlib import Path
import torch
import itertools
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
from lpips import LPIPS
from loguru import logger

from hugs.datasets.utils import (
    get_rotating_camera,
    get_smpl_canon_params,
    get_smpl_static_params, 
    get_static_camera
)
from hugs.losses.utils import ssim
from hugs.datasets import Human3RDataset, NeumanDataset
from hugs.losses.loss import HumanSceneLoss
from hugs.models.hugs_trimlp import HUGS_TRIMLP
from hugs.models.hugs_wo_trimlp import HUGS_WO_TRIMLP
from hugs.models import SceneGS
from hugs.models.anchor_attention import AnchorSceneAttentionBaseline, save_anchor_attention_debug
from hugs.utils.anchor_io import (
    load_anchor_vertices,
    save_anchor_binding_summary_csv,
    save_anchor_vertices,
    save_anchors_csv,
    save_anchors_pt,
)
from hugs.utils.anchor_utils import (
    bind_gaussians_to_anchors,
    compute_anchors,
    generate_semantic_anchor_vertices,
    make_template_from_vertices,
)
from hugs.utils.init_opt import optimize_init
from hugs.renderer.gs_renderer import render_human_scene
from hugs.utils.vis import save_ply
from hugs.utils.image import psnr, save_image
from hugs.utils.general import RandomIndexIterator, load_human_ckpt, save_images, create_video
from hugs.utils.scene_sugar_regularizer import (
    compute_scene_sugar_loss,
    prune_scene_gaussians_by_sugar,
    save_scene_sugar_debug_ply,
    scene_sugar_enabled,
    scene_sugar_stats,
    should_prune_scene_sugar,
)
from hugs.utils.scene_global_aware_adc import (
    SceneMaxGradientAccumulator,
    compute_weighted_adc_loss,
    scene_global_aware_adc_active,
    scene_global_aware_adc_enabled,
    scene_global_aware_adc_should_densify,
    voxel_based_scene_densify_and_prune,
)


def get_train_dataset(cfg):
    if cfg.dataset.name == 'neuman':
        logger.info(f'Loading NeuMan dataset {cfg.dataset.seq}-train')
        dataset = NeumanDataset(
            cfg.dataset.seq, 'train', 
            render_mode=cfg.mode,
            add_bg_points=cfg.scene.add_bg_points,
            num_bg_points=cfg.scene.num_bg_points,
            bg_sphere_dist=cfg.scene.bg_sphere_dist,
            clean_pcd=cfg.scene.clean_pcd,
            init_pcd_path=getattr(cfg.scene, 'init_pcd_path', None),
            depth_prior_dir=getattr(cfg.scene, 'depth_prior_dir', None),
        )
    elif cfg.dataset.name == 'human3r':
        logger.info(f'Loading Human3R converted dataset {cfg.dataset_path}-train')
        dataset = Human3RDataset(cfg.dataset_path, 'train', render_mode=cfg.mode)
    
    return dataset


def get_val_dataset(cfg):
    if cfg.dataset.name == 'neuman':
        logger.info(f'Loading NeuMan dataset {cfg.dataset.seq}-val')
        dataset = NeumanDataset(
            cfg.dataset.seq, 'val', cfg.mode,
            init_pcd_path=getattr(cfg.scene, 'init_pcd_path', None),
        )
    elif cfg.dataset.name == 'human3r':
        logger.info(f'Loading Human3R converted dataset {cfg.dataset_path}-val')
        dataset = Human3RDataset(cfg.dataset_path, 'val', render_mode=cfg.mode)
   
    return dataset


def get_anim_dataset(cfg):
    if cfg.train.anim_interval < 0:
        logger.info('Animation disabled; skipping animation dataset')
        return None

    if cfg.dataset.name == 'neuman':
        logger.info(f'Loading NeuMan dataset {cfg.dataset.seq}-anim')
        dataset = NeumanDataset(cfg.dataset.seq, 'anim', cfg.mode)
    elif cfg.dataset.name == 'human3r':
        logger.info('Human3R converted dataset does not provide AMASS animation data')
        dataset = None
    elif cfg.dataset.name == 'zju':
        dataset = None
        
    return dataset


def get_all_dataset(cfg):
    if cfg.dataset.name == 'neuman':
        logger.info(f'Loading NeuMan dataset {cfg.dataset.seq}-all')
        dataset = NeumanDataset(
            cfg.dataset.seq, 'all', cfg.mode,
            init_pcd_path=getattr(cfg.scene, 'init_pcd_path', None),
        )
    elif cfg.dataset.name == 'human3r':
        logger.info(f'Loading Human3R converted dataset {cfg.dataset_path}-all')
        dataset = Human3RDataset(cfg.dataset_path, 'all', render_mode=cfg.mode)
    else:
        dataset = None

    return dataset


class GaussianTrainer():
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        
        # get dataset
        if not cfg.eval:
            self.train_dataset = get_train_dataset(cfg)
        self.val_dataset = get_val_dataset(cfg)
        self.anim_dataset = get_anim_dataset(cfg)
        self.all_dataset = get_all_dataset(cfg)
        
        self.eval_metrics = {}
        self.lpips = LPIPS(net="alex", pretrained=True).to('cuda')
        # get models
        self.human_gs, self.scene_gs = None, None
        
        if cfg.mode in ['human', 'human_scene']:
            if cfg.human.name == 'hugs_wo_trimlp':
                self.human_gs = HUGS_WO_TRIMLP(
                    sh_degree=cfg.human.sh_degree, 
                    n_subdivision=cfg.human.n_subdivision,  
                    use_surface=cfg.human.use_surface,
                    init_2d=cfg.human.init_2d,
                    rotate_sh=cfg.human.rotate_sh,
                    isotropic=cfg.human.isotropic,
                    init_scale_multiplier=cfg.human.init_scale_multiplier,
                )
                init_betas = torch.stack([x['betas'] for x in self.train_dataset.cached_data], dim=0)
                self.human_gs.create_betas(init_betas[0], cfg.human.optim_betas)
                self.human_gs.initialize()
            elif cfg.human.name == 'hugs_trimlp':
                init_betas = torch.stack([x['betas'] for x in self.val_dataset.cached_data], dim=0)
                self.human_gs = HUGS_TRIMLP(
                    sh_degree=cfg.human.sh_degree, 
                    n_subdivision=cfg.human.n_subdivision,  
                    use_surface=cfg.human.use_surface,
                    init_2d=cfg.human.init_2d,
                    rotate_sh=cfg.human.rotate_sh,
                    isotropic=cfg.human.isotropic,
                    init_scale_multiplier=cfg.human.init_scale_multiplier,
                    n_features=32,
                    use_deformer=cfg.human.use_deformer,
                    disable_posedirs=cfg.human.disable_posedirs,
                    triplane_res=cfg.human.triplane_res,
                    betas=init_betas[0]
                )
                self.human_gs.create_betas(init_betas[0], cfg.human.optim_betas)
                if not cfg.eval:
                    self.human_gs.initialize()
                    if getattr(cfg.human, 'skip_init_opt', False):
                        logger.info('Skipping human optimize_init because cfg.human.skip_init_opt=True')
                    else:
                        self.human_gs = optimize_init(self.human_gs, num_steps=7000)
        
        if cfg.mode in ['scene', 'human_scene']:
            self.scene_gs = SceneGS(
                sh_degree=cfg.scene.sh_degree,
            )
            
        # setup the optimizers
        if self.human_gs:
            self.human_gs.setup_optimizer(cfg=cfg.human.lr)
            logger.info(self.human_gs)
            if cfg.human.ckpt:
                # load_human_ckpt(self.human_gs, cfg.human.ckpt)
                self.human_gs.load_state_dict(torch.load(cfg.human.ckpt))
                logger.info(f'Loaded human model from {cfg.human.ckpt}')
            else:
                ckpt_files = sorted(glob.glob(f'{cfg.logdir_ckpt}/*human*.pth'))
                if len(ckpt_files) > 0:
                    ckpt = torch.load(ckpt_files[-1])
                    self.human_gs.load_state_dict(ckpt)
                    logger.info(f'Loaded human model from {ckpt_files[-1]}')

            if not cfg.eval:
                init_smpl_global_orient = torch.stack([x['global_orient'] for x in self.train_dataset.cached_data])
                init_smpl_body_pose = torch.stack([x['body_pose'] for x in self.train_dataset.cached_data])
                init_smpl_trans = torch.stack([x['transl'] for x in self.train_dataset.cached_data], dim=0)
                init_betas = torch.stack([x['betas'] for x in self.train_dataset.cached_data], dim=0)
                init_eps_offsets = torch.zeros((len(self.train_dataset), self.human_gs.n_gs, 3), 
                                            dtype=torch.float32, device="cuda")

                self.human_gs.create_betas(init_betas[0], cfg.human.optim_betas)
                
                self.human_gs.create_body_pose(init_smpl_body_pose, cfg.human.optim_pose)
                self.human_gs.create_global_orient(init_smpl_global_orient, cfg.human.optim_pose)
                self.human_gs.create_transl(init_smpl_trans, cfg.human.optim_trans)
                
                self.human_gs.setup_optimizer(cfg=cfg.human.lr)
                    
        if self.scene_gs:
            logger.info(self.scene_gs)
            if cfg.scene.ckpt:
                ckpt = torch.load(cfg.scene.ckpt)
                self.scene_gs.restore(ckpt, cfg.scene.lr)
                logger.info(f'Loaded scene model from {cfg.scene.ckpt}')
            else:
                ckpt_files = sorted(glob.glob(f'{cfg.logdir_ckpt}/*scene*.pth'))
                if len(ckpt_files) > 0:
                    ckpt = torch.load(ckpt_files[-1])
                    self.scene_gs.restore(ckpt, cfg.scene.lr)
                    logger.info(f'Loaded scene model from {cfg.scene.ckpt}')
                else:
                    pcd = self.train_dataset.init_pcd
                    spatial_lr_scale = self.train_dataset.radius
                    self.scene_gs.create_from_pcd(pcd, spatial_lr_scale)
                
            self.scene_gs.setup_optimizer(cfg=cfg.scene.lr)
        
        bg_color = cfg.bg_color
        if bg_color == 'white':
            self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        elif bg_color == 'black':
            self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        else:
            raise ValueError(f"Unknown background color {bg_color}")

        self.scene_anchor_points = None
        anchor_pcd_path = getattr(cfg.scene, 'anchor_pcd_path', None)
        if anchor_pcd_path:
            anchor_npz = np.load(anchor_pcd_path)
            anchor_points = anchor_npz['points'].astype(np.float32)
            self.scene_anchor_points = torch.from_numpy(anchor_points).float().to('cuda')
            logger.info(f'Loaded scene geometry anchor points: {anchor_pcd_path} ({self.scene_anchor_points.shape[0]} points)')
        
        if cfg.mode in ['human', 'human_scene']:
            l = cfg.human.loss

            self.loss_fn = HumanSceneLoss(
                l_ssim_w=l.ssim_w,
                l_l1_w=l.l1_w,
                l_lpips_w=l.lpips_w,
                l_lbs_w=l.lbs_w,
                l_humansep_w=l.humansep_w,
                num_patches=l.num_patches,
                patch_size=l.patch_size,
                use_patches=l.use_patches,
                bg_color=self.bg_color,
            )
        else:
            self.cfg.train.optim_scene = True
            l = cfg.scene.loss
            self.loss_fn = HumanSceneLoss(
                l_ssim_w=l.ssim_w,
                l_l1_w=l.l1_w,
                bg_color=self.bg_color,
            )
                
        if cfg.mode in ['human', 'human_scene']:
            self.canon_camera_params = get_rotating_camera(
                dist=5.0, img_size=512, 
                nframes=cfg.human.canon_nframes, device='cuda',
                angle_limit=2*torch.pi,
            )
            betas = self.human_gs.betas.detach() if hasattr(self.human_gs, 'betas') else self.train_dataset.betas[0]
            self.static_smpl_params = get_smpl_static_params(
                betas=betas,
                pose_type=self.cfg.human.canon_pose_type
            )

        self.anchor_attention = None
        self.anchor_attention_optimizer = None
        self.anchor_data = None
        self.setup_anchor_attention_baseline()

        self.scene_sugar_debug_dir = None
        self.setup_scene_sugar()

        self.scene_global_adc_accumulator = None
        self.setup_scene_global_aware_adc()

    def setup_scene_global_aware_adc(self):
        g_cfg = getattr(self.cfg, 'scene_global_aware_adc', None)
        if g_cfg is None or not scene_global_aware_adc_enabled(g_cfg):
            return
        if self.scene_gs is None:
            logger.warning('Global-aware voxel ADC requested but no scene Gaussian model is active')
            return
        if not bool(getattr(g_cfg, 'apply_to_scene_only', True)):
            raise ValueError('Global-aware voxel ADC integration requires apply_to_scene_only=true')
        if bool(getattr(g_cfg, 'disable_for_human', True)) and self.cfg.mode == 'human':
            raise ValueError('Global-aware voxel ADC is a scene-only background method and cannot run in human mode')
        s_cfg = getattr(self.cfg, 'scene_sugar', None)
        if (
            s_cfg is not None
            and scene_sugar_enabled(s_cfg)
            and bool(getattr(g_cfg, 'disable_sugar_losses', True))
        ):
            raise ValueError('Global-aware voxel ADC and Scene-SuGaR are separate background branches; disable scene_sugar or set disable_sugar_losses=false explicitly')

        self.scene_global_adc_accumulator = SceneMaxGradientAccumulator(
            self.scene_gs.get_xyz.shape[0],
            self.scene_gs.get_xyz.device,
        )
        logger.info(
            'Global-aware voxel ADC enabled for scene Gaussians only: '
            f'start_iter={getattr(g_cfg, "start_iter", 0)}, '
            f'densify_until_iter={getattr(g_cfg, "densify_until_iter", 15000)}, '
            f'interval={getattr(g_cfg, "densification_interval", 100)}, '
            f'voxel_size={getattr(g_cfg, "voxel_size", 0.005)}, '
            f'tau_pos={getattr(g_cfg, "tau_pos", 0.0002)}'
        )

    def setup_scene_sugar(self):
        s_cfg = getattr(self.cfg, 'scene_sugar', None)
        if s_cfg is None or not scene_sugar_enabled(s_cfg):
            return
        if self.scene_gs is None:
            logger.warning('Scene-SuGaR requested but no scene Gaussian model is active')
            return
        if not bool(getattr(s_cfg, 'apply_to_scene_only', True)):
            raise ValueError('Scene-SuGaR integration requires apply_to_scene_only=true')
        if not bool(getattr(s_cfg, 'disable_mesh_extraction', True)):
            logger.warning('Scene-SuGaR mesh extraction is not implemented in HUGS; continuing with scene-only regularization')

        debug_dir = Path(getattr(s_cfg, 'debug_dir', '') or f'{self.cfg.logdir}/scene_sugar_debug')
        debug_dir.mkdir(parents=True, exist_ok=True)
        self.scene_sugar_debug_dir = debug_dir
        logger.info(
            'Scene-SuGaR enabled for background Gaussians only: '
            f'start_iter={getattr(s_cfg, "start_iter", 0)}, '
            f'prune_iter={getattr(s_cfg, "prune_iter", -1)}, '
            f'debug_dir={debug_dir}'
        )

    def maybe_add_scene_sugar_loss(self, loss, loss_dict, iteration):
        s_cfg = getattr(self.cfg, 'scene_sugar', None)
        if self.scene_gs is None or s_cfg is None or not scene_sugar_enabled(s_cfg):
            return loss

        sugar_loss, sugar_logs = compute_scene_sugar_loss(self.scene_gs, s_cfg, iteration)
        if sugar_logs:
            loss = loss + sugar_loss
            loss_dict.update(sugar_logs)

        log_interval = int(getattr(s_cfg, 'log_interval', 0) or 0)
        if log_interval > 0 and iteration % log_interval == 0:
            stats = scene_sugar_stats(
                self.scene_gs,
                low_opacity_threshold=float(getattr(s_cfg, 'opacity_prune_threshold', 0.03) or 0.03),
            )
            for key, value in stats.items():
                loss_dict[f'scene_sugar_{key}'] = value.detach()
        return loss

    def maybe_scene_sugar_post_step(self, iteration):
        s_cfg = getattr(self.cfg, 'scene_sugar', None)
        if self.scene_gs is None or s_cfg is None or not scene_sugar_enabled(s_cfg):
            return

        debug_dir = self.scene_sugar_debug_dir
        if debug_dir is not None and should_prune_scene_sugar(s_cfg, iteration):
            try:
                save_scene_sugar_debug_ply(
                    self.scene_gs,
                    str(debug_dir / f'scene_gaussians_iter{iteration:06d}_before_prune.ply'),
                    color_by='opacity',
                )
            except Exception as exc:
                logger.warning(f'Failed to save Scene-SuGaR pre-prune PLY at {iteration:06d}: {exc}')

        prune_stats = prune_scene_gaussians_by_sugar(self.scene_gs, s_cfg, iteration)
        if prune_stats.get('pruned_count', 0.0) > 0 and debug_dir is not None:
            try:
                save_scene_sugar_debug_ply(
                    self.scene_gs,
                    str(debug_dir / f'scene_gaussians_iter{iteration:06d}_after_prune.ply'),
                    color_by='opacity',
                )
            except Exception as exc:
                logger.warning(f'Failed to save Scene-SuGaR post-prune PLY at {iteration:06d}: {exc}')

        ply_interval = int(getattr(s_cfg, 'ply_interval', 0) or 0)
        if debug_dir is not None and ply_interval > 0 and iteration % ply_interval == 0:
            try:
                save_scene_sugar_debug_ply(
                    self.scene_gs,
                    str(debug_dir / f'scene_gaussians_iter{iteration:06d}.ply'),
                    color_by='scale_volume',
                )
            except Exception as exc:
                logger.warning(f'Failed to save Scene-SuGaR debug PLY at {iteration:06d}: {exc}')

    def setup_anchor_attention_baseline(self):
        a_cfg = getattr(self.cfg, 'anchor_attention', None)
        if a_cfg is None or not bool(getattr(a_cfg, 'use_anchors', False)):
            return
        if self.cfg.mode != 'human_scene' or self.human_gs is None or self.scene_gs is None:
            logger.warning('Anchor-attention baseline requires mode=human_scene with both human and scene models')
            return

        debug_dir = Path(getattr(a_cfg, 'debug_dir', '') or f'{self.cfg.logdir}/anchor_attention_debug')
        debug_dir.mkdir(parents=True, exist_ok=True)

        template = make_template_from_vertices(
            self.human_gs.get_vitruvian_verts_template(),
            self.human_gs.smpl_template.faces,
            self.human_gs.smpl_template.lbs_weights,
            canonical_pose='hugs_vitruvian',
        )
        anchor_vertices_path = str(getattr(a_cfg, 'anchor_vertices_path', '') or '')
        if anchor_vertices_path:
            anchor_vertices = load_anchor_vertices(anchor_vertices_path)
        else:
            anchor_vertices = generate_semantic_anchor_vertices(
                template,
                count_per_anchor=int(getattr(a_cfg, 'count_per_anchor', 64)),
            )
        anchors = compute_anchors(template, anchor_vertices)

        mu_canon = self.human_gs.get_xyz.detach().cpu()
        has_valid_bindings = (
            hasattr(self.human_gs, 'anchor_ids')
            and self.human_gs.anchor_ids is not None
            and self.human_gs.anchor_ids.shape[0] == mu_canon.shape[0]
        )
        if has_valid_bindings:
            anchor_ids = self.human_gs.anchor_ids.detach().cpu()
            anchor_weights = self.human_gs.anchor_weights.detach().cpu()
            anchor_pos = anchors['pos_canon']
            top1 = anchor_ids[:, 0]
            bindings = {
                'gaussian_anchor_ids': anchor_ids,
                'gaussian_anchor_weights': anchor_weights,
                'top1_anchor_id': top1,
                'top1_anchor_dist': torch.linalg.norm(mu_canon - anchor_pos[top1], dim=1),
            }
        else:
            gaussian_lbs = self.human_gs.smpl_template.lbs_weights.detach().cpu()
            if gaussian_lbs.shape[0] != mu_canon.shape[0]:
                gaussian_lbs = torch.zeros(mu_canon.shape[0], anchors['lbs'].shape[1], dtype=mu_canon.dtype)
            bindings = bind_gaussians_to_anchors(
                mu_canon,
                gaussian_lbs,
                anchors,
                top_m=int(getattr(a_cfg, 'top_m', 2)),
                lambda_lbs=float(getattr(a_cfg, 'lambda_lbs', 0.35)),
                sigma_scale=float(getattr(a_cfg, 'sigma_scale', 1.8)),
                min_sigma=float(getattr(a_cfg, 'min_sigma', 0.035)),
            )
            anchor_ids = bindings['gaussian_anchor_ids']
            anchor_weights = bindings['gaussian_anchor_weights']
            self.human_gs.create_anchor_bindings(anchor_ids, anchor_weights)

        save_anchor_vertices(anchor_vertices, debug_dir / 'anchor_vertices.json')
        save_anchors_pt(anchors, debug_dir / 'anchors.pt')
        save_anchors_csv(anchors, debug_dir / 'anchors.csv')
        save_anchor_binding_summary_csv(bindings, anchors, debug_dir / 'anchor_binding_summary_init.csv')

        self.anchor_data = {
            'anchors': anchors,
            'anchor_names': anchors['names'],
            'debug_dir': debug_dir,
        }

        should_create = any(bool(getattr(a_cfg, name, False)) for name in [
            'use_anchor_token_encoder',
            'use_scene_query',
            'use_cross_attention',
            'use_interaction_correction',
        ])
        if not should_create:
            logger.info(f'Anchor bindings initialized only; debug files written to {debug_dir}')
            return

        self.anchor_attention = AnchorSceneAttentionBaseline(
            a_cfg,
            num_anchors=len(anchors['names']),
            top_m=int(getattr(a_cfg, 'top_m', 2)),
        ).to('cuda')
        ckpt = str(getattr(a_cfg, 'ckpt', '') or '')
        if ckpt:
            self.anchor_attention.load_state_dict(torch.load(ckpt))
            logger.info(f'Loaded anchor-attention module from {ckpt}')
        self.anchor_attention_optimizer = torch.optim.Adam(
            self.anchor_attention.parameters(),
            lr=float(getattr(a_cfg, 'lr', 1e-4)),
        )
        logger.info(f'Anchor-attention baseline initialized with {len(anchors["names"])} anchors; debug files in {debug_dir}')

    def maybe_apply_anchor_attention(self, human_gs_out, scene_gs_out, render_mode, iteration):
        if self.anchor_attention is None or human_gs_out is None or scene_gs_out is None or render_mode != 'human_scene':
            return human_gs_out, {}
        if (not hasattr(self.human_gs, 'anchor_ids')) or self.human_gs.anchor_ids is None:
            return human_gs_out, {}
        return self.anchor_attention(
            human_gs_out,
            scene_gs_out,
            self.human_gs.anchor_ids,
            self.human_gs.anchor_weights,
            iteration=0 if iteration is None else iteration,
        )

    def scene_geometry_regularization(self, data, scene_gs_out, human_gs_out):
        if scene_gs_out is None:
            return None, {}

        geo_loss = None
        geo_dict = {}
        scene_xyz = scene_gs_out['xyz']

        anchor_w = float(getattr(self.cfg.scene, 'anchor_loss_w', 0.0) or 0.0)
        if anchor_w > 0.0 and self.scene_anchor_points is not None and self.scene_anchor_points.shape[0] > 0:
            n_scene = min(int(getattr(self.cfg.scene, 'anchor_loss_sample_scene', 2048)), scene_xyz.shape[0])
            n_prior = min(int(getattr(self.cfg.scene, 'anchor_loss_sample_prior', 8192)), self.scene_anchor_points.shape[0])
            scene_idx = torch.randint(scene_xyz.shape[0], (n_scene,), device=scene_xyz.device)
            prior_idx = torch.randint(self.scene_anchor_points.shape[0], (n_prior,), device=scene_xyz.device)
            scene_sample = scene_xyz[scene_idx]
            prior_sample = self.scene_anchor_points[prior_idx]
            dist2 = torch.cdist(scene_sample, prior_sample).pow(2).min(dim=1).values
            trunc = float(getattr(self.cfg.scene, 'anchor_loss_trunc', 0.5) or 0.5)
            trunc2 = max(trunc * trunc, 1e-8)
            anchor_loss = torch.clamp(dist2, max=trunc2).mean() / trunc2
            geo_loss = anchor_w * anchor_loss if geo_loss is None else geo_loss + anchor_w * anchor_loss
            geo_dict['scene_anchor'] = anchor_loss.detach()

        exclusion_w = float(getattr(self.cfg.scene, 'human_exclusion_w', 0.0) or 0.0)
        if exclusion_w > 0.0 and human_gs_out is not None and 'xyz' in human_gs_out:
            human_xyz = human_gs_out['xyz'].detach()
            if human_xyz.shape[0] > 0:
                n_scene = min(int(getattr(self.cfg.scene, 'human_exclusion_sample_scene', 2048)), scene_xyz.shape[0])
                n_human = min(int(getattr(self.cfg.scene, 'human_exclusion_sample_human', 8192)), human_xyz.shape[0])
                scene_idx = torch.randint(scene_xyz.shape[0], (n_scene,), device=scene_xyz.device)
                human_idx = torch.randint(human_xyz.shape[0], (n_human,), device=scene_xyz.device)
                scene_sample = scene_xyz[scene_idx]
                human_sample = human_xyz[human_idx]
                min_dist = torch.sqrt(torch.cdist(scene_sample, human_sample).pow(2).min(dim=1).values + 1e-8)
                radius = float(getattr(self.cfg.scene, 'human_exclusion_radius', 0.12) or 0.12)
                exclusion_loss = torch.relu(radius - min_dist).pow(2).mean() / max(radius * radius, 1e-8)
                geo_loss = exclusion_w * exclusion_loss if geo_loss is None else geo_loss + exclusion_w * exclusion_loss
                geo_dict['scene_human_exclusion'] = exclusion_loss.detach()

        depth_w = float(getattr(self.cfg.scene, 'depth_prior_w', 0.0) or 0.0)
        if depth_w > 0.0 and data is not None and 'depth_prior' in data:
            depth_prior = data['depth_prior']
            if depth_prior.dim() == 3:
                depth_prior = depth_prior.squeeze(0)
            valid_depth = torch.isfinite(depth_prior) & (depth_prior > 0)
            if bool(getattr(self.cfg.scene, 'depth_prior_mask_human', True)) and 'mask' in data:
                valid_depth = valid_depth & (data['mask'] < 0.5)
            if valid_depth.any():
                depth_prior_eff = torch.where(valid_depth, depth_prior, torch.zeros_like(depth_prior))
                scene_xyz_d = scene_xyz.detach()
                n_scene = min(int(getattr(self.cfg.scene, 'depth_prior_sample_scene', 4096)), scene_xyz.shape[0])
                # Prefer visible points when available; fall back to all scene points.
                if scene_gs_out is not None and 'visibility_filter' in scene_gs_out:
                    candidate_idx = torch.where(scene_gs_out['visibility_filter'])[0]
                else:
                    candidate_idx = torch.arange(scene_xyz.shape[0], device=scene_xyz.device)
                if candidate_idx.numel() == 0:
                    candidate_idx = torch.arange(scene_xyz.shape[0], device=scene_xyz.device)
                pick = candidate_idx[torch.randint(candidate_idx.numel(), (min(n_scene, candidate_idx.numel()),), device=scene_xyz.device)]
                xyz = scene_xyz[pick]
                ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
                homog = torch.cat([xyz, ones], dim=1)
                cam = homog @ data['world_view_transform']
                z = cam[:, 2]
                uvw = cam @ data['full_proj_transform']
                # Use pinhole intrinsics directly for stable pixel lookup.
                K = data['cam_intrinsics'].to(xyz.device)
                u = torch.round(K[0, 0] * (cam[:, 0] / torch.clamp(z, min=1e-6)) + K[0, 2]).long()
                v = torch.round(K[1, 1] * (cam[:, 1] / torch.clamp(z, min=1e-6)) + K[1, 2]).long()
                h, w = depth_prior.shape[-2], depth_prior.shape[-1]
                inside = (z > 1e-4) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
                if inside.any():
                    u_in = u[inside]
                    v_in = v[inside]
                    z_in = z[inside]
                    dp = depth_prior_eff[v_in, u_in].to(z_in.device)
                    vd = torch.isfinite(dp) & (dp > 0)
                    if vd.any():
                        residual = torch.abs(z_in[vd] - dp[vd])
                        max_res = float(getattr(self.cfg.scene, 'depth_prior_max_residual', 2.0) or 2.0)
                        tol = float(getattr(self.cfg.scene, 'depth_prior_tolerance', 0.25) or 0.25)
                        residual = torch.clamp(residual, max=max_res)
                        depth_loss = torch.relu(residual - tol).mean() / max(max_res - tol, 1e-6)
                        geo_loss = depth_w * depth_loss if geo_loss is None else geo_loss + depth_w * depth_loss
                        geo_dict['scene_depth_prior'] = depth_loss.detach()

        return geo_loss, geo_dict

    def maybe_accumulate_scene_global_aware_adc(
        self,
        data,
        render_pkg,
        loss_extras,
        render_mode,
        scene_gs_out,
        human_gs_out,
        iteration,
    ):
        g_cfg = getattr(self.cfg, 'scene_global_aware_adc', None)
        if (
            self.scene_gs is None
            or scene_gs_out is None
            or self.scene_global_adc_accumulator is None
            or g_cfg is None
            or not scene_global_aware_adc_active(g_cfg, iteration)
            or render_mode not in ['scene', 'human_scene']
        ):
            return {}

        accumulate_interval = int(getattr(g_cfg, 'accumulate_interval', 1) or 1)
        if accumulate_interval > 1 and iteration % accumulate_interval != 0:
            return {}

        pred_img = loss_extras.get('pred_img', render_pkg.get('render', None))
        gt_img = loss_extras.get('gt_img', data.get('rgb', None))
        viewspace_points = render_pkg.get('viewspace_points', None)
        if pred_img is None or gt_img is None or viewspace_points is None:
            return {}

        adc_loss, adc_logs = compute_weighted_adc_loss(pred_img, gt_img, g_cfg)
        full_grad = torch.autograd.grad(
            adc_loss,
            viewspace_points,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]
        if full_grad is None:
            return adc_logs

        if render_mode == 'human_scene':
            human_n = int(human_gs_out['xyz'].shape[0]) if human_gs_out is not None else 0
            scene_n = int(scene_gs_out['xyz'].shape[0])
            scene_grad = full_grad[human_n:human_n + scene_n]
        else:
            scene_grad = full_grad[:scene_gs_out['xyz'].shape[0]]

        visibility_filter = render_pkg.get('scene_visibility_filter', None)
        self.scene_global_adc_accumulator.update(scene_grad, visibility_filter=visibility_filter)
        if bool(getattr(g_cfg, 'log_blur_stats', True)):
            return adc_logs
        return {'scene_adc_loss': adc_logs['scene_adc_loss']}

    def train(self):
        if self.human_gs:
            self.human_gs.train()

        pbar = tqdm(range(self.cfg.train.num_steps+1), desc="Training")
        
        rand_idx_iter = RandomIndexIterator(len(self.train_dataset))
        sgrad_means, sgrad_stds = [], []
        for t_iter in range(self.cfg.train.num_steps+1):
            render_mode = self.cfg.mode
            
            if self.scene_gs and self.cfg.train.optim_scene:
                self.scene_gs.update_learning_rate(t_iter)
            
            if hasattr(self.human_gs, 'update_learning_rate'):
                self.human_gs.update_learning_rate(t_iter)
        
            rnd_idx = next(rand_idx_iter)
            data = self.train_dataset[rnd_idx]
            
            human_gs_out, scene_gs_out = None, None
            
            if self.human_gs:
                human_gs_out = self.human_gs.forward(
                    smpl_scale=data['smpl_scale'][None],
                    dataset_idx=rnd_idx,
                    is_train=True,
                    ext_tfs=None,
                )
            
            if self.scene_gs:
                if t_iter >= self.cfg.scene.opt_start_iter:
                    scene_gs_out = self.scene_gs.forward()
                else:
                    render_mode = 'human'

            anchor_stats = {}
            human_gs_out, anchor_stats = self.maybe_apply_anchor_attention(
                human_gs_out,
                scene_gs_out,
                render_mode,
                t_iter,
            )
            
            bg_color = torch.rand(3, dtype=torch.float32, device="cuda")
            
            
            if self.cfg.human.loss.humansep_w > 0.0 and render_mode == 'human_scene':
                render_human_separate = True
                human_bg_color = torch.rand(3, dtype=torch.float32, device="cuda")
            else:
                human_bg_color = None
                render_human_separate = False
            
            render_pkg = render_human_scene(
                data=data, 
                human_gs_out=human_gs_out, 
                scene_gs_out=scene_gs_out, 
                bg_color=bg_color,
                human_bg_color=human_bg_color,
                render_mode=render_mode,
                render_human_separate=render_human_separate,
            )
            
            if self.human_gs:
                self.human_gs.init_values['edges'] = self.human_gs.edges
                        
            loss, loss_dict, loss_extras = self.loss_fn(
                data,
                render_pkg,
                human_gs_out,
                render_mode=render_mode,
                human_gs_init_values=self.human_gs.init_values if self.human_gs else None,
                bg_color=bg_color,
                human_bg_color=human_bg_color,
            )

            geo_loss, geo_loss_dict = self.scene_geometry_regularization(data, scene_gs_out, human_gs_out)
            if geo_loss is not None:
                loss = loss + geo_loss
                loss_dict.update(geo_loss_dict)

            if scene_gs_out is not None and self.cfg.train.optim_scene:
                loss = self.maybe_add_scene_sugar_loss(loss, loss_dict, t_iter)

            if anchor_stats:
                delta_loss = anchor_stats.get('delta_loss', None)
                delta_w = float(getattr(self.cfg.anchor_attention, 'delta_loss_w', 0.0) or 0.0)
                if delta_loss is not None and delta_w > 0.0:
                    loss = loss + delta_w * delta_loss
                    loss_dict['anchor_delta'] = delta_loss.detach()
                debug_interval = int(getattr(self.cfg.anchor_attention, 'debug_interval', 0) or 0)
                if debug_interval > 0 and t_iter % debug_interval == 0:
                    save_anchor_attention_debug(
                        self.anchor_data['debug_dir'],
                        t_iter,
                        anchor_stats,
                        anchor_names=self.anchor_data['anchor_names'],
                    )
            
            adc_logs = self.maybe_accumulate_scene_global_aware_adc(
                data,
                render_pkg,
                loss_extras,
                render_mode,
                scene_gs_out,
                human_gs_out,
                (t_iter - self.cfg.scene.opt_start_iter) + 1 if t_iter >= self.cfg.scene.opt_start_iter else t_iter + 1,
            )
            if adc_logs:
                loss_dict.update(adc_logs)

            loss.backward()
            
            loss_dict['loss'] = loss
            
            if t_iter % 10 == 0:
                postfix_dict = {
                    "#hp": f"{self.human_gs.n_gs/1000 if self.human_gs else 0:.1f}K",
                    "#sp": f"{self.scene_gs.get_xyz.shape[0]/1000 if self.scene_gs else 0:.1f}K",
                    'h_sh_d': self.human_gs.active_sh_degree if self.human_gs else 0,
                    's_sh_d': self.scene_gs.active_sh_degree if self.scene_gs else 0,
                }
                for k, v in loss_dict.items():
                    postfix_dict["l_"+k] = f"{v.item():.4f}"
                        
                pbar.set_postfix(postfix_dict)
                pbar.update(10)
                
            if t_iter == self.cfg.train.num_steps:
                pbar.close()

            if t_iter % 1000 == 0:
                with torch.no_grad():
                    pred_img = loss_extras['pred_img']
                    gt_img = loss_extras['gt_img']
                    log_pred_img = (pred_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    log_gt_img = (gt_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    log_img = np.concatenate([log_gt_img, log_pred_img], axis=1)
                    save_images(log_img, f'{self.cfg.logdir}/train/{t_iter:06d}.png')
            
            if t_iter >= self.cfg.scene.opt_start_iter:
                if (t_iter - self.cfg.scene.opt_start_iter) < self.cfg.scene.densify_until_iter and self.cfg.mode in ['scene', 'human_scene']:
                    render_pkg['scene_viewspace_points'] = render_pkg['viewspace_points']
                    render_pkg['scene_viewspace_points'].grad = render_pkg['viewspace_points'].grad
                        
                    sgrad_mean, sgrad_std = render_pkg['scene_viewspace_points'].grad.mean(), render_pkg['scene_viewspace_points'].grad.std()
                    sgrad_means.append(sgrad_mean.item())
                    sgrad_stds.append(sgrad_std.item())
                    with torch.no_grad():
                        self.scene_densification(
                            visibility_filter=render_pkg['scene_visibility_filter'],
                            radii=render_pkg['scene_radii'],
                            viewspace_point_tensor=render_pkg['scene_viewspace_points'],
                            iteration=(t_iter - self.cfg.scene.opt_start_iter) + 1,
                            data=data,
                        )
                        
            if t_iter < self.cfg.human.densify_until_iter and self.cfg.mode in ['human', 'human_scene']:
                render_pkg['human_viewspace_points'] = render_pkg['viewspace_points'][:human_gs_out['xyz'].shape[0]]
                render_pkg['human_viewspace_points'].grad = render_pkg['viewspace_points'].grad[:human_gs_out['xyz'].shape[0]]
                with torch.no_grad():
                    self.human_densification(
                        human_gs_out=human_gs_out,
                        visibility_filter=render_pkg['human_visibility_filter'],
                        radii=render_pkg['human_radii'],
                        viewspace_point_tensor=render_pkg['human_viewspace_points'],
                        iteration=t_iter+1,
                    )
            
            if self.human_gs:
                self.human_gs.optimizer.step()
                self.human_gs.optimizer.zero_grad(set_to_none=True)
                
            if self.scene_gs and self.cfg.train.optim_scene:
                if t_iter >= self.cfg.scene.opt_start_iter:
                    self.scene_gs.optimizer.step()
                    self.scene_gs.optimizer.zero_grad(set_to_none=True)

            if self.anchor_attention_optimizer is not None:
                self.anchor_attention_optimizer.step()
                self.anchor_attention_optimizer.zero_grad(set_to_none=True)

            if self.scene_gs and self.cfg.train.optim_scene and t_iter >= self.cfg.scene.opt_start_iter:
                self.maybe_scene_sugar_post_step(t_iter)
                
            # save checkpoint
            if (t_iter % self.cfg.train.save_ckpt_interval == 0 and t_iter > 0) or \
                (t_iter == self.cfg.train.num_steps and t_iter > 0):
                self.save_ckpt(t_iter)

            # run validation
            if t_iter % self.cfg.train.val_interval == 0 and t_iter > 0:
                self.validate(t_iter)
            
            if t_iter == 0:
                if self.scene_gs:
                    try:
                        self.scene_gs.save_ply(f'{self.cfg.logdir}/meshes/scene_{t_iter:06d}_splat.ply')
                    except Exception as exc:
                        logger.warning(f'Failed to save scene mesh {t_iter:06d}: {exc}')
                if self.human_gs:
                    try:
                        save_ply(human_gs_out, f'{self.cfg.logdir}/meshes/human_{t_iter:06d}_splat.ply')
                    except Exception as exc:
                        logger.warning(f'Failed to save human mesh {t_iter:06d}: {exc}')

                if self.cfg.mode in ['human', 'human_scene']:
                    self.render_canonical(t_iter, nframes=self.cfg.human.canon_nframes)
                
            if t_iter % self.cfg.train.anim_interval == 0 and t_iter > 0 and self.cfg.train.anim_interval > 0:
                if self.human_gs:
                    save_ply(human_gs_out, f'{self.cfg.logdir}/meshes/human_{t_iter:06d}_splat.ply')
                if self.anim_dataset is not None:
                    self.animate(t_iter)
                    
                if self.cfg.mode in ['human', 'human_scene']:
                    self.render_canonical(t_iter, nframes=self.cfg.human.canon_nframes)
            
            if t_iter % 1000 == 0 and t_iter > 0:
                if self.human_gs: self.human_gs.oneupSHdegree()
                if self.scene_gs: self.scene_gs.oneupSHdegree()
                
            if self.cfg.train.save_progress_images and t_iter % self.cfg.train.progress_save_interval == 0 and self.cfg.mode in ['human', 'human_scene']:
                self.render_canonical(t_iter, nframes=2, is_train_progress=True)
        
        # train progress images
        if self.cfg.train.save_progress_images:
            video_fname = f'{self.cfg.logdir}/train_{self.cfg.dataset.name}_{self.cfg.dataset.seq}.mp4'
            create_video(f'{self.cfg.logdir}/train_progress/', video_fname, fps=10)
            shutil.rmtree(f'{self.cfg.logdir}/train_progress/')
            
    def save_ckpt(self, iter=None):
        
        iter_s = 'final' if iter is None else f'{iter:06d}'
        
        if self.human_gs:
            try:
                torch.save(self.human_gs.state_dict(), f'{self.cfg.logdir_ckpt}/human_{iter_s}.pth')
            except Exception as exc:
                logger.warning(f'Failed to save human checkpoint {iter_s}: {exc}')
            
        if self.scene_gs and bool(getattr(self.cfg.train, 'save_scene_ckpt', True)):
            try:
                torch.save(self.scene_gs.state_dict(), f'{self.cfg.logdir_ckpt}/scene_{iter_s}.pth')
            except Exception as exc:
                logger.warning(f'Failed to save scene checkpoint {iter_s}: {exc}')
            try:
                self.scene_gs.save_ply(f'{self.cfg.logdir}/meshes/scene_{iter_s}_splat.ply')
            except Exception as exc:
                logger.warning(f'Failed to save scene mesh {iter_s}: {exc}')

        if self.anchor_attention is not None:
            try:
                torch.save(self.anchor_attention.state_dict(), f'{self.cfg.logdir_ckpt}/anchor_attention_{iter_s}.pth')
            except Exception as exc:
                logger.warning(f'Failed to save anchor-attention checkpoint {iter_s}: {exc}')
            
        logger.info(f'Saved checkpoint {iter_s}')
                
    def scene_densification(self, visibility_filter, radii, viewspace_point_tensor, iteration, data=None):
        self.scene_gs.max_radii2D[visibility_filter] = torch.max(
            self.scene_gs.max_radii2D[visibility_filter],
            radii[visibility_filter]
        )

        g_cfg = getattr(self.cfg, 'scene_global_aware_adc', None)
        use_global_adc = (
            g_cfg is not None
            and scene_global_aware_adc_enabled(g_cfg)
            and self.scene_global_adc_accumulator is not None
        )

        if use_global_adc:
            self.scene_global_adc_accumulator.sync(self.scene_gs.get_xyz.shape[0], self.scene_gs.get_xyz.device)
        else:
            self.scene_gs.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if use_global_adc and scene_global_aware_adc_should_densify(g_cfg, iteration):
            size_threshold = 20 if iteration > self.cfg.scene.opacity_reset_interval else None
            max_n_gs = int(getattr(g_cfg, 'max_scene_gaussians', -1) or -1)
            if max_n_gs <= 0:
                max_n_gs = self.cfg.scene.max_n_gaussians
            stats = voxel_based_scene_densify_and_prune(
                self.scene_gs,
                self.scene_global_adc_accumulator.max_grad,
                g_cfg,
                extent=self.train_dataset.radius,
                min_opacity=self.cfg.scene.prune_min_opacity,
                max_screen_size=size_threshold,
                max_n_gs=max_n_gs,
            )
            self.scene_global_adc_accumulator.reset(self.scene_gs.get_xyz.shape[0], self.scene_gs.get_xyz.device)
            if bool(getattr(g_cfg, 'log_voxel_stats', True)):
                logger.info(
                    f"[{iteration:06d}] Global-aware voxel ADC stats: "
                    f"selected={stats.get('selected_gaussians', 0.0):.0f}, "
                    f"voxels={stats.get('num_voxels', 0.0):.0f}, "
                    f"scene={stats.get('num_before', 0.0):.0f}->{stats.get('num_after', 0.0):.0f}"
                )
        elif (not use_global_adc) and iteration > self.cfg.scene.densify_from_iter and iteration % self.cfg.scene.densification_interval == 0:
            size_threshold = 20 if iteration > self.cfg.scene.opacity_reset_interval else None
            self.scene_gs.densify_and_prune(
                self.cfg.scene.densify_grad_threshold,
                min_opacity=self.cfg.scene.prune_min_opacity,
                extent=self.train_dataset.radius,
                max_screen_size=size_threshold,
                max_n_gs=self.cfg.scene.max_n_gaussians,
            )

        is_white = self.bg_color.sum().item() == 3.
        reset_interval = self.cfg.scene.opacity_reset_interval
        if use_global_adc:
            reset_interval = int(getattr(g_cfg, 'reset_opacity_interval', reset_interval) or reset_interval)

        if iteration % reset_interval == 0 or (is_white and iteration == self.cfg.scene.densify_from_iter):
            logger.info(f"[{iteration:06d}] Resetting opacity!!!")
            self.scene_gs.reset_opacity()

        prune_interval = int(getattr(self.cfg.scene, 'depth_prune_interval', 0) or 0)
        if prune_interval > 0 and data is not None:
            prune_from = int(getattr(self.cfg.scene, 'depth_prune_from_iter', 0) or 0)
            prune_until = int(getattr(self.cfg.scene, 'depth_prune_until_iter', 1_000_000) or 1_000_000)
            if prune_from <= iteration <= prune_until and iteration % prune_interval == 0:
                self.scene_depth_guided_pruning(data, iteration)
                if use_global_adc:
                    self.scene_global_adc_accumulator.reset(self.scene_gs.get_xyz.shape[0], self.scene_gs.get_xyz.device)

    @torch.no_grad()
    def scene_depth_guided_pruning(self, data, iteration):
        if self.scene_gs is None or data is None or 'depth_prior' not in data:
            return

        depth_prior = data['depth_prior']
        if depth_prior.dim() == 3:
            depth_prior = depth_prior.squeeze(0)
        valid_depth = torch.isfinite(depth_prior) & (depth_prior > 0)
        if bool(getattr(self.cfg.scene, 'depth_prune_mask_human', True)) and 'mask' in data:
            valid_depth = valid_depth & (data['mask'] < 0.5)
        if not valid_depth.any():
            return

        xyz = self.scene_gs.get_xyz
        n_points = xyz.shape[0]
        if n_points == 0:
            return

        ones = torch.ones((n_points, 1), dtype=xyz.dtype, device=xyz.device)
        homog = torch.cat([xyz, ones], dim=1)
        cam = homog @ data['world_view_transform']
        z = cam[:, 2]
        k = data['cam_intrinsics'].to(xyz.device)
        u = torch.round(k[0, 0] * (cam[:, 0] / torch.clamp(z, min=1e-6)) + k[0, 2]).long()
        v = torch.round(k[1, 1] * (cam[:, 1] / torch.clamp(z, min=1e-6)) + k[1, 2]).long()
        h, w = depth_prior.shape[-2], depth_prior.shape[-1]
        inside = (z > 1e-4) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not inside.any():
            return

        point_idx = torch.where(inside)[0]
        u_in = u[inside]
        v_in = v[inside]
        bg_valid = valid_depth[v_in, u_in]
        if not bg_valid.any():
            return

        point_idx = point_idx[bg_valid]
        z_in = z[inside][bg_valid]
        depth_in = depth_prior[v_in[bg_valid], u_in[bg_valid]].to(z_in.device)
        residual_abs = torch.abs(z_in - depth_in)
        opacity = self.scene_gs.get_opacity.squeeze(-1)[point_idx]

        abs_thr = float(getattr(self.cfg.scene, 'depth_prune_abs_threshold', 2.0) or 2.0)
        hard_thr = float(getattr(self.cfg.scene, 'depth_prune_abs_hard_threshold', 8.0) or 8.0)
        opacity_thr = float(getattr(self.cfg.scene, 'depth_prune_opacity_threshold', 0.03) or 0.03)
        candidate = ((residual_abs > abs_thr) & (opacity < opacity_thr)) | (residual_abs > hard_thr)
        if not candidate.any():
            return

        candidate_idx = point_idx[candidate]
        candidate_residual = residual_abs[candidate]
        max_frac = float(getattr(self.cfg.scene, 'depth_prune_max_frac', 0.05) or 0.05)
        max_prune = max(1, int(n_points * max_frac))
        if candidate_idx.numel() > max_prune:
            keep = torch.topk(candidate_residual, k=max_prune, largest=True).indices
            candidate_idx = candidate_idx[keep]

        if n_points - candidate_idx.numel() < 1000:
            return

        prune_mask = torch.zeros(n_points, dtype=torch.bool, device=xyz.device)
        prune_mask[candidate_idx] = True
        self.scene_gs.prune_points(prune_mask)
        logger.info(
            f"[{iteration:06d}] Depth-pruned {candidate_idx.numel()}/{n_points} scene gaussians "
            f"(abs_thr={abs_thr}, hard_thr={hard_thr}, opacity_thr={opacity_thr})"
        )
    
    def human_densification(self, human_gs_out, visibility_filter, radii, viewspace_point_tensor, iteration):
        self.human_gs.max_radii2D[visibility_filter] = torch.max(
            self.human_gs.max_radii2D[visibility_filter], 
            radii[visibility_filter]
        )
        
        self.human_gs.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if iteration > self.cfg.human.densify_from_iter and iteration % self.cfg.human.densification_interval == 0:
            size_threshold = 20
            self.human_gs.densify_and_prune(
                human_gs_out,
                self.cfg.human.densify_grad_threshold, 
                min_opacity=self.cfg.human.prune_min_opacity, 
                extent=self.cfg.human.densify_extent, 
                max_screen_size=size_threshold,
                max_n_gs=self.cfg.human.max_n_gaussians,
            )
    
    @torch.no_grad()
    def validate(self, iter=None):
        
        iter_s = 'final' if iter is None else f'{iter:06d}'
        
        bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
        
        if self.human_gs:
            self.human_gs.eval()
                
        methods = ['hugs', 'hugs_human']
        metrics = ['lpips', 'psnr', 'ssim']
        metrics = dict.fromkeys(['_'.join(x) for x in itertools.product(methods, metrics)])
        metrics = {k: [] for k in metrics}
        
        for idx, data in enumerate(tqdm(self.val_dataset, desc="Validation")):
            human_gs_out, scene_gs_out = None, None
            render_mode = self.cfg.mode
            
            if self.human_gs:
                human_gs_out = self.human_gs.forward(
                    global_orient=data['global_orient'], 
                    body_pose=data['body_pose'], 
                    betas=data['betas'], 
                    transl=data['transl'], 
                    smpl_scale=data['smpl_scale'][None],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=None,
                )
                
            if self.scene_gs:
                if iter is not None:
                    if iter >= self.cfg.scene.opt_start_iter:
                        scene_gs_out = self.scene_gs.forward()
                    else:
                        render_mode = 'human'
                else:
                    scene_gs_out = self.scene_gs.forward()

            eval_iter = self.cfg.train.num_steps if iter is None else iter
            human_gs_out, _ = self.maybe_apply_anchor_attention(
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
            
            gt_image = data['rgb']
            
            image = render_pkg["render"]
            if self.cfg.dataset.name == 'zju':
                image = image * data['mask']
                gt_image = gt_image * data['mask']
            
            metrics['hugs_psnr'].append(psnr(image, gt_image).mean().double())
            metrics['hugs_ssim'].append(ssim(image, gt_image).mean().double())
            metrics['hugs_lpips'].append(self.lpips(image.clip(max=1), gt_image).mean().double())
            
            log_img = torchvision.utils.make_grid([gt_image, image], nrow=2, pad_value=1)
            imf = f'{self.cfg.logdir}/val/full_{iter_s}_{idx:03d}.png'
            os.makedirs(os.path.dirname(imf), exist_ok=True)
            torchvision.utils.save_image(log_img, imf)
            
            log_img = []
            if self.cfg.mode in ['human', 'human_scene']:
                bbox = data['bbox'].to(int)
                cropped_gt_image = gt_image[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
                cropped_image = image[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
                log_img += [cropped_gt_image, cropped_image]
                
                metrics['hugs_human_psnr'].append(psnr(cropped_image, cropped_gt_image).mean().double())
                metrics['hugs_human_ssim'].append(ssim(cropped_image, cropped_gt_image).mean().double())
                metrics['hugs_human_lpips'].append(self.lpips(cropped_image.clip(max=1), cropped_gt_image).mean().double())
            
            if len(log_img) > 0:
                log_img = torchvision.utils.make_grid(log_img, nrow=len(log_img), pad_value=1)
                torchvision.utils.save_image(log_img, f'{self.cfg.logdir}/val/human_{iter_s}_{idx:03d}.png')
        
        
        self.eval_metrics[iter_s] = {}
        
        for k, v in metrics.items():
            if v == []:
                continue
            
            logger.info(f"{iter_s} - {k.upper()}: {torch.stack(v).mean().item():.4f}")
            self.eval_metrics[iter_s][k] = torch.stack(v).mean().item()
        
        torch.save(metrics, f'{self.cfg.logdir}/val/eval_{iter_s}.pth')

    @torch.no_grad()
    def render_full_sequence(self, iter=None, keep_images=False, fps=20):
        if self.all_dataset is None:
            logger.info("No full-sequence dataset found")
            return 0

        iter_s = 'final' if iter is None else f'{iter:06d}'

        if self.human_gs:
            self.human_gs.eval()

        frame_dir = f'{self.cfg.logdir}/render_all/'
        os.makedirs(frame_dir, exist_ok=True)

        for idx, data in enumerate(tqdm(self.all_dataset, desc="Render full sequence")):
            human_gs_out, scene_gs_out = None, None

            if self.human_gs:
                human_gs_out = self.human_gs.forward(
                    global_orient=data['global_orient'],
                    body_pose=data['body_pose'],
                    betas=data['betas'],
                    transl=data['transl'],
                    smpl_scale=data['smpl_scale'][None],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=None,
                )

            if self.scene_gs:
                scene_gs_out = self.scene_gs.forward()

            eval_iter = self.cfg.train.num_steps if iter is None else iter
            human_gs_out, _ = self.maybe_apply_anchor_attention(
                human_gs_out,
                scene_gs_out,
                self.cfg.mode,
                eval_iter,
            )

            render_pkg = render_human_scene(
                data=data,
                human_gs_out=human_gs_out,
                scene_gs_out=scene_gs_out,
                bg_color=self.bg_color,
                render_mode=self.cfg.mode,
            )

            image = render_pkg["render"]
            torchvision.utils.save_image(image, f'{frame_dir}/{idx:05d}.png')

        video_fname = f'{self.cfg.logdir}/render_all_{self.cfg.dataset.name}_{self.cfg.dataset.seq}_{iter_s}.mp4'
        create_video(frame_dir, video_fname, fps=fps)

        if not keep_images:
            shutil.rmtree(frame_dir)
            os.makedirs(frame_dir)

        return video_fname
    
    @torch.no_grad()
    def animate(self, iter=None, keep_images=False):
        if self.anim_dataset is None:
            logger.info("No animation dataset found")
            return 0
        
        iter_s = 'final' if iter is None else f'{iter:06d}'
        if self.human_gs:
            self.human_gs.eval()
        
        os.makedirs(f'{self.cfg.logdir}/anim/', exist_ok=True)
        
        for idx, data in enumerate(tqdm(self.anim_dataset, desc="Animation")):
            human_gs_out, scene_gs_out = None, None
            
            if self.human_gs:
                ext_tfs = (data['manual_trans'], data['manual_rotmat'], data['manual_scale'])
                human_gs_out = self.human_gs.forward(
                    global_orient=data['global_orient'],
                    body_pose=data['body_pose'],
                    betas=data['betas'],
                    transl=data['transl'],
                    smpl_scale=data['smpl_scale'][None],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=ext_tfs,
                )
            
            if self.scene_gs:
                scene_gs_out = self.scene_gs.forward()
                    
            render_pkg = render_human_scene(
                data=data, 
                human_gs_out=human_gs_out, 
                scene_gs_out=scene_gs_out, 
                bg_color=self.bg_color,
                render_mode=self.cfg.mode,
            )
            
            image = render_pkg["render"]
            
            torchvision.utils.save_image(image, f'{self.cfg.logdir}/anim/{idx:05d}.png')
            
        video_fname = f'{self.cfg.logdir}/anim_{self.cfg.dataset.name}_{self.cfg.dataset.seq}_{iter_s}.mp4'
        create_video(f'{self.cfg.logdir}/anim/', video_fname, fps=20)
        if not keep_images:
            shutil.rmtree(f'{self.cfg.logdir}/anim/')
            os.makedirs(f'{self.cfg.logdir}/anim/')
    
    @torch.no_grad()
    def render_canonical(self, iter=None, nframes=100, is_train_progress=False, pose_type=None):
        iter_s = 'final' if iter is None else f'{iter:06d}'
        iter_s += f'_{pose_type}' if pose_type is not None else ''
        
        if self.human_gs:
            self.human_gs.eval()
        
        os.makedirs(f'{self.cfg.logdir}/canon/', exist_ok=True)
        
        camera_params = get_rotating_camera(
            dist=5.0, img_size=256 if is_train_progress else 512, 
            nframes=nframes, device='cuda',
            angle_limit=torch.pi if is_train_progress else 2*torch.pi,
        )
        
        betas = self.human_gs.betas.detach() if hasattr(self.human_gs, 'betas') else self.train_dataset.betas[0]
        
        static_smpl_params = get_smpl_static_params(
            betas=betas,
            pose_type=self.cfg.human.canon_pose_type if pose_type is None else pose_type,
        )
        
        if is_train_progress:
            progress_imgs = []
        
        pbar = range(nframes) if is_train_progress else tqdm(range(nframes), desc="Canonical:")
        
        for idx in pbar:
            human_gs_out, scene_gs_out = None, None
            
            cam_p = camera_params[idx]
            data = dict(static_smpl_params, **cam_p)

            if self.human_gs:
                human_gs_out = self.human_gs.forward(
                    global_orient=data['global_orient'],
                    body_pose=data['body_pose'],
                    betas=data['betas'],
                    transl=data['transl'],
                    smpl_scale=data['smpl_scale'],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=None,
                )
                
            if is_train_progress:
                scale_mod = 0.5
                render_pkg = render_human_scene(
                    data=data, 
                    human_gs_out=human_gs_out, 
                    scene_gs_out=scene_gs_out, 
                    bg_color=self.bg_color,
                    render_mode='human',
                    scaling_modifier=scale_mod,
                )
                
                image = render_pkg["render"]
                
                progress_imgs.append(image)
                
                render_pkg = render_human_scene(
                    data=data, 
                    human_gs_out=human_gs_out, 
                    scene_gs_out=scene_gs_out, 
                    bg_color=self.bg_color,
                    render_mode='human',
                )
                
                image = render_pkg["render"]
                
                progress_imgs.append(image)
                
            else:
                render_pkg = render_human_scene(
                    data=data, 
                    human_gs_out=human_gs_out, 
                    scene_gs_out=scene_gs_out, 
                    bg_color=self.bg_color,
                    render_mode='human',
                )
                
                image = render_pkg["render"]
                
                torchvision.utils.save_image(image, f'{self.cfg.logdir}/canon/{idx:05d}.png')
        
        if is_train_progress:
            os.makedirs(f'{self.cfg.logdir}/train_progress/', exist_ok=True)
            log_img = torchvision.utils.make_grid(progress_imgs, nrow=4, pad_value=0)
            save_image(log_img, f'{self.cfg.logdir}/train_progress/{iter:06d}.png', 
                       text_labels=f"{iter:06d}, n_gs={self.human_gs.n_gs}")
            return
        
        video_fname = f'{self.cfg.logdir}/canon_{self.cfg.dataset.name}_{self.cfg.dataset.seq}_{iter_s}.mp4'
        create_video(f'{self.cfg.logdir}/canon/', video_fname, fps=10)
        shutil.rmtree(f'{self.cfg.logdir}/canon/')
        os.makedirs(f'{self.cfg.logdir}/canon/')
        
    def render_poses(self, camera_params, smpl_params, pose_type='a_pose', bg_color='white'):
    
        if self.human_gs:
            self.human_gs.eval()
        
        betas = self.human_gs.betas.detach() if hasattr(self.human_gs, 'betas') else self.val_dataset.betas[0]
        
        nframes = len(camera_params)
        
        canon_forward_out = None
        if hasattr(self.human_gs, 'canon_forward'):
            canon_forward_out = self.human_gs.canon_forward()
        
        pbar = tqdm(range(nframes), desc="Canonical:")
        if bg_color == 'white':
            bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        elif bg_color == 'black':
            bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
            
            
        imgs = []
        for idx in pbar:
            human_gs_out, scene_gs_out = None, None
            
            cam_p = camera_params[idx]
            data = dict(smpl_params, **cam_p)

            if self.human_gs:
                if canon_forward_out is not None:
                    human_gs_out = self.human_gs.forward_test(
                        canon_forward_out,
                        global_orient=data['global_orient'],
                        body_pose=data['body_pose'],
                        betas=data['betas'],
                        transl=data['transl'],
                        smpl_scale=data['smpl_scale'],
                        dataset_idx=-1,
                        is_train=False,
                        ext_tfs=None,
                    )
                else:
                    human_gs_out = self.human_gs.forward(
                        global_orient=data['global_orient'],
                        body_pose=data['body_pose'],
                        betas=data['betas'],
                        transl=data['transl'],
                        smpl_scale=data['smpl_scale'],
                        dataset_idx=-1,
                        is_train=False,
                        ext_tfs=None,
                    )

            render_pkg = render_human_scene(
                data=data, 
                human_gs_out=human_gs_out, 
                scene_gs_out=scene_gs_out, 
                bg_color=self.bg_color,
                render_mode='human',
            )
            image = render_pkg["render"]
            imgs.append(image)
        return imgs
