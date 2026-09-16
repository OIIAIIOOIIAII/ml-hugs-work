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
from hugs.losses.utils import ssim, l1_loss as _l1_loss
from hugs.losses.floor_grounding import estimate_floor_plane, estimate_floor_from_feet, get_foot_joints_world, floor_grounding_loss
from hugs.datasets import Human3RDataset, NeumanDataset
from hugs.losses.loss import HumanSceneLoss
from hugs.models.hugs_trimlp import HUGS_TRIMLP
from hugs.models.hugs_wo_trimlp import HUGS_WO_TRIMLP
from hugs.models import SceneGS
from hugs.models.anchor_attention import AnchorSceneAttentionBaseline, save_anchor_attention_debug
from hugs.models.fusion_mlp import HumanSceneFuseDecoder
from hugs.models.temporal_contact_attention import TemporalAnchorAttention
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
from hugs.renderer.gs_renderer import render_human_scene, render_depth_map
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
            mono_depth_dir=getattr(cfg.dataset, 'mono_depth_dir', None),
            vitpose_kp_dir=getattr(cfg.dataset, 'vitpose_kp_dir', None),
            max_frames=getattr(cfg.dataset, 'max_frames', None),
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
            max_frames=getattr(cfg.dataset, 'max_frames', None),
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
        self._best_human_psnr = -1.0
        self.gt_transl = None
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
                if getattr(cfg.human, 'zero_transl_init', False):
                    logger.info("zero_transl_init=True: overriding GT transl with zeros")
                    init_smpl_trans = torch.zeros_like(init_smpl_trans)
                init_betas = torch.stack([x['betas'] for x in self.train_dataset.cached_data], dim=0)
                init_eps_offsets = torch.zeros((len(self.train_dataset), self.human_gs.n_gs, 3), 
                                            dtype=torch.float32, device="cuda")

                self.human_gs.create_betas(init_betas[0], cfg.human.optim_betas)
                
                self.human_gs.create_body_pose(init_smpl_body_pose, cfg.human.optim_pose)
                self.human_gs.create_global_orient(init_smpl_global_orient, cfg.human.optim_pose)
                self.human_gs.create_transl(init_smpl_trans, cfg.human.optim_trans)

                # Load GT transl for alignment tracking (train-split only)
                self.gt_transl = None
                if hasattr(self.train_dataset, 'dataset_path'):
                    gt_npz = os.path.join(self.train_dataset.dataset_path,
                                          '4d_humans', 'smpl_optimized_aligned_scale_gt.npz')
                    if os.path.exists(gt_npz):
                        _gt = np.load(gt_npz)
                        _gt_all = torch.tensor(_gt['transl'], dtype=torch.float32)
                        # filter to train split to match human_gs.transl shape
                        if hasattr(self.train_dataset, 'train_split'):
                            _idx = list(self.train_dataset.train_split)
                            self.gt_transl = _gt_all[_idx]
                        else:
                            self.gt_transl = _gt_all
                        if self.gt_transl.shape[0] == self.human_gs.transl.shape[0]:
                            _err = torch.norm(self.human_gs.transl.detach().cpu() - self.gt_transl, dim=1)
                            logger.info(f"[transl vs GT] init (VIMO): mean_L2={_err.mean():.4f}, max_L2={_err.max():.4f}")
                        else:
                            logger.warning(f"[transl vs GT] shape mismatch: transl={self.human_gs.transl.shape}, gt={self.gt_transl.shape}, skipping")

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
                l_depth_w=float(getattr(l, 'depth_w', 0.0) or 0.0),
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
            if hasattr(self.human_gs, 'betas'):
                betas = self.human_gs.betas.detach()
            elif hasattr(self, 'train_dataset'):
                betas = self.train_dataset.betas[0]
            else:
                betas = torch.stack([x['betas'] for x in self.val_dataset.cached_data], dim=0)[0]
            self.static_smpl_params = get_smpl_static_params(
                betas=betas,
                pose_type=self.cfg.human.canon_pose_type
            )

        self.anchor_attention = None
        self.anchor_attention_optimizer = None
        self.anchor_data = None
        self.setup_anchor_attention_baseline()

        self.frame_trans_offsets = None
        self.frame_trans_optimizer = None
        self.setup_frame_trans_offsets()

        self.ground_y = None
        self.scale_corrections = None
        self.scale_corr_optimizer = None
        self.setup_coarse_align()

        self.scene_sugar_debug_dir = None
        self.setup_scene_sugar()

        self.scene_global_adc_accumulator = None
        self.setup_scene_global_aware_adc()

        self.floor_normal = None
        self.floor_point = None
        self.floor_update_counter = 0
        self.setup_floor_grounding()

        # Cache for temporal smoothness constraint on correction outputs (Solution B)
        self._delta_correction_cache = {}

        self.fusion_mlp = None
        self.setup_fusion_mlp()

    def setup_fusion_mlp(self):
        cfg = getattr(self.cfg, 'fusion_mlp', None)
        if cfg is None or not bool(getattr(cfg, 'enabled', False)):
            return
        if self.scene_gs is None or self.cfg.mode != 'human_scene':
            return
        hidden_dim = int(getattr(cfg, 'hidden_dim', 64))
        lr_init = float(getattr(cfg, 'lr_init', 1e-3))
        lr_final = float(getattr(cfg, 'lr_final', 1e-5))
        lr_delay_mult = float(getattr(cfg, 'lr_delay_mult', 0.01))
        max_steps = int(getattr(self.cfg.train, 'num_steps', 20000))
        property_dims = {
            'shs': self.scene_gs.get_features.shape[1] * self.scene_gs.get_features.shape[2],
            'xyz': self.scene_gs.get_xyz.shape[1],
            'opacity': self.scene_gs.get_opacity.shape[1],
            'scales': self.scene_gs.get_scaling.shape[1],
            'rotq': self.scene_gs.get_rotation.shape[1],
        }
        self.fusion_mlp = HumanSceneFuseDecoder(property_dims, hidden_dim=hidden_dim).to('cuda')
        self.fusion_mlp.setup_optimizer(
            lr_init=lr_init,
            lr_final=lr_final,
            lr_delay_mult=lr_delay_mult,
            max_steps=max_steps,
        )
        from loguru import logger
        logger.info(f"FusionMLP enabled: property_dims={property_dims}, hidden_dim={hidden_dim}, lr_init={lr_init}, lr_final={lr_final}")

    def setup_floor_grounding(self):
        fg_cfg = getattr(self.cfg, 'floor_grounding', None)
        if fg_cfg is None or not bool(getattr(fg_cfg, 'enabled', False)):
            return
        if self.scene_gs is None or self.human_gs is None:
            logger.warning('Floor grounding requires both human and scene models')
            return
        if not hasattr(self.human_gs, 'transl'):
            logger.warning('Floor grounding requires human model with transl parameter')
            return
        logger.info(
            f'Floor grounding enabled: loss_w={getattr(fg_cfg, "loss_w", 0.1)}, '
            f'contact_zone={getattr(fg_cfg, "contact_zone", 0.10)}, '
            f'margin={getattr(fg_cfg, "margin", 0.005)}, '
            f'start_iter={getattr(fg_cfg, "start_iter", 0)}'
        )

    def _update_floor_plane(self, scene_gs_out):
        """Estimate floor plane from SMPL foot positions (primary) or scene GS (fallback)."""
        fg_cfg = getattr(self.cfg, 'floor_grounding', None)
        up_axis = int(getattr(fg_cfg, 'up_axis', 1))
        bottom_frac = float(getattr(fg_cfg, 'bottom_frac', 0.05))

        normal, centroid = None, None

        # Primary: foot-based estimation — anchors floor at actual foot contact level
        if self.human_gs is not None and hasattr(self.train_dataset, 'smpl_params'):
            smpl_scales = self.train_dataset.smpl_params['scale']
            if hasattr(self.train_dataset, 'train_split'):
                smpl_scales = smpl_scales[self.train_dataset.train_split]
            try:
                normal, centroid = estimate_floor_from_feet(
                    self.human_gs, smpl_scales, up_axis=up_axis, bottom_frac=bottom_frac
                )
            except Exception as exc:
                logger.warning(f'estimate_floor_from_feet failed: {exc}')

        # Fallback: scene GS if foot estimation failed
        if normal is None and scene_gs_out is not None:
            opacity_thresh = float(getattr(fg_cfg, 'opacity_thresh', 0.1))
            scene_xyz = scene_gs_out['xyz'].detach()
            scene_opacity = scene_gs_out.get('opacity', None)
            normal, centroid = estimate_floor_plane(
                scene_xyz, scene_opacity, opacity_thresh, bottom_frac, up_axis
            )

        if normal is not None:
            first_time = self.floor_normal is None
            self.floor_normal = normal
            self.floor_point = centroid
            if first_time:
                logger.info(
                    f'Floor plane estimated: normal=[{normal[0]:.3f},{normal[1]:.3f},{normal[2]:.3f}] '
                    f'centroid=[{centroid[0]:.3f},{centroid[1]:.3f},{centroid[2]:.3f}]'
                )

    def maybe_add_floor_grounding_loss(self, loss, loss_dict, data, scene_gs_out, anchor_stats, rnd_idx, t_iter):
        """Add floor grounding loss to pull foot joints toward the scene floor plane."""
        fg_cfg = getattr(self.cfg, 'floor_grounding', None)
        if fg_cfg is None or not bool(getattr(fg_cfg, 'enabled', False)):
            return loss
        if self.human_gs is None or scene_gs_out is None:
            return loss
        start_iter = int(getattr(fg_cfg, 'start_iter', 0))
        if t_iter < start_iter:
            return loss
        loss_w = float(getattr(fg_cfg, 'loss_w', 0.1))
        if loss_w <= 0.0:
            return loss

        # Update cached floor plane periodically
        update_interval = int(getattr(fg_cfg, 'floor_update_interval', 50))
        if self.floor_normal is None or self.floor_update_counter % update_interval == 0:
            self._update_floor_plane(scene_gs_out)
        self.floor_update_counter += 1

        if self.floor_normal is None:
            return loss

        # Get delta_transl from anchor attention if available
        delta_transl = None
        if anchor_stats and 'delta_transl' in anchor_stats:
            dt = anchor_stats['delta_transl']
            if dt is not None:
                delta_transl = dt

        smpl_scale = data['smpl_scale']
        contact_zone = float(getattr(fg_cfg, 'contact_zone', 0.10))
        margin = float(getattr(fg_cfg, 'margin', 0.005))

        try:
            foot_world = get_foot_joints_world(
                self.human_gs, rnd_idx, smpl_scale, delta_transl
            )
            fg_loss, fg_stats = floor_grounding_loss(
                foot_world, self.floor_normal, self.floor_point, contact_zone, margin
            )
            loss = loss + loss_w * fg_loss
            loss_dict['floor_grounding'] = fg_loss.detach()
            loss_dict['floor_contact'] = fg_stats['floor_contact']
            loss_dict['floor_pen'] = fg_stats['floor_penetration']
            debug_interval = int(getattr(fg_cfg, 'debug_interval', 200))
            if t_iter % debug_interval == 0:
                logger.info(
                    f'[fg iter={t_iter}] loss={fg_loss.item():.5f} '
                    f'contact={fg_stats["floor_contact"].item():.5f} '
                    f'pen={fg_stats["floor_penetration"].item():.5f} '
                    f'foot_dist_min={fg_stats["foot_dist_min"].item():.4f} '
                    f'foot_dist_mean={fg_stats["foot_dist_mean"].item():.4f}'
                )
        except Exception as exc:
            logger.warning(f'Floor grounding loss failed at iter {t_iter}: {exc}')

        return loss

    def maybe_add_delta_correction_smooth_loss(self, loss, loss_dict, anchor_stats, frame_idx):
        """Problem-1 Solution-B: Temporal smoothness constraint on correction outputs.

        Caches delta_transl and per-anchor delta_xyz mean from each frame, then for
        adjacent frames (|diff|<=3) adds an MSE loss to discourage abrupt changes.
        Weight is controlled by cfg.anchor_attention.delta_correction_smooth_w (default 0).
        """
        if not anchor_stats:
            return loss
        smooth_w = float(getattr(self.cfg.anchor_attention, 'delta_correction_smooth_w', 0.0) or 0.0)
        if smooth_w <= 0.0:
            return loss

        cur_dt = anchor_stats.get('delta_transl', None)
        cur_dmu = anchor_stats.get('delta_mu', None)

        # Build current frame snapshot (detached — cache must not accumulate grads)
        cur_snap = {}
        if cur_dt is not None:
            cur_snap['delta_transl'] = cur_dt.detach().float()
        if cur_dmu is not None:
            # Reduce per-Gaussian delta_mu to per-frame scalar vector to keep cache small
            cur_snap['delta_mu_mean'] = cur_dmu.detach().float().mean(dim=0)

        if not cur_snap:
            return loss

        # Search for an adjacent cached frame
        smooth_loss_accum = None
        n_terms = 0
        for adj_idx in (frame_idx - 1, frame_idx + 1, frame_idx - 2, frame_idx + 2, frame_idx - 3, frame_idx + 3):
            if adj_idx not in self._delta_correction_cache:
                continue
            prev_snap = self._delta_correction_cache[adj_idx]
            for key in ('delta_transl', 'delta_mu_mean'):
                if key in cur_snap and key in prev_snap:
                    diff = (cur_snap[key] - prev_snap[key]).pow(2).mean()
                    smooth_loss_accum = diff if smooth_loss_accum is None else smooth_loss_accum + diff
                    n_terms += 1
            break  # use only the nearest adjacent frame

        if smooth_loss_accum is not None and n_terms > 0:
            smooth_loss = smooth_loss_accum / n_terms
            loss = loss + smooth_w * smooth_loss
            loss_dict['delta_smooth'] = smooth_loss.detach()

        # Update cache
        self._delta_correction_cache[frame_idx] = cur_snap
        # Keep cache bounded: evict oldest entries beyond 30 frames
        if len(self._delta_correction_cache) > 30:
            oldest = min(self._delta_correction_cache.keys())
            del self._delta_correction_cache[oldest]

        return loss

    def setup_frame_trans_offsets(self):
        """Per-frame learnable translation offsets to correct coarse alignment errors."""
        cfg = getattr(self.cfg, 'frame_trans_offset', None)
        if cfg is None or not bool(getattr(cfg, 'enabled', False)):
            return
        if not hasattr(self, 'train_dataset') or self.train_dataset is None:
            return
        n = len(self.train_dataset)
        self.frame_trans_offsets = torch.nn.Parameter(
            torch.zeros(n, 3, device='cuda'), requires_grad=True
        )
        lr = float(getattr(cfg, 'lr', 0.002))
        self.frame_trans_optimizer = torch.optim.Adam([self.frame_trans_offsets], lr=lr)
        logger.info(f'Per-frame trans offsets initialized: {n} frames, lr={lr}')

    def maybe_apply_frame_trans_offset(self, human_gs_out, frame_idx, iteration=0):
        if self.frame_trans_offsets is None or human_gs_out is None:
            return human_gs_out
        cfg = getattr(self.cfg, 'frame_trans_offset', None)
        offset_start = int(getattr(cfg, 'offset_start_iter', 0)) if cfg else 0
        if iteration < offset_start:
            return human_gs_out
        offset = self.frame_trans_offsets[int(frame_idx) % len(self.frame_trans_offsets)]
        out = dict(human_gs_out)
        out['xyz'] = human_gs_out['xyz'] + offset.unsqueeze(0)
        return out

    def maybe_add_mask_reproj_loss(self, loss, loss_dict, data, human_gs_out, iteration=0):
        """L2 loss between projected human GS centroid and GT mask centroid."""
        cfg = getattr(self.cfg, 'frame_trans_offset', None)
        if cfg is None or human_gs_out is None:
            return loss, loss_dict
        offset_start = int(getattr(cfg, 'offset_start_iter', 0))
        if iteration < offset_start:
            return loss, loss_dict
        w = float(getattr(cfg, 'mask_reproj_w', 0.0) or 0.0)
        if w <= 0.0:
            return loss, loss_dict
        mask = data.get('mask', None)
        if mask is None:
            return loss, loss_dict
        device = human_gs_out['xyz'].device
        mask = mask.to(device)
        ys, xs = torch.where(mask > 0.5)
        if xs.numel() == 0:
            return loss, loss_dict
        mask_centroid = torch.stack([xs.float().mean(), ys.float().mean()])

        xyz = human_gs_out['xyz']
        with torch.no_grad():
            opacity_w = human_gs_out['opacity'].squeeze(-1)
            opacity_w = opacity_w / (opacity_w.sum() + 1e-6)
        xyz_weighted = (xyz * opacity_w[:, None]).sum(0)  # [3]

        W2C = data['world_view_transform'].T.float().to(device)
        xyz_homo = torch.cat([xyz_weighted, xyz_weighted.new_ones(1)])
        xyz_cam = (W2C @ xyz_homo)[:3]
        if xyz_cam[2].item() < 1e-3:
            return loss, loss_dict
        K = data['cam_intrinsics'].float().to(device)
        xyz_proj = K @ xyz_cam
        uv_human = xyz_proj[:2] / xyz_proj[2]

        reproj_loss = (uv_human - mask_centroid).pow(2).mean()
        loss = loss + w * reproj_loss
        loss_dict['mask_reproj'] = reproj_loss.detach()
        return loss, loss_dict

    def maybe_add_frame_trans_smooth_loss(self, loss, loss_dict):
        """Temporal smoothness regularization on per-frame trans offsets."""
        cfg = getattr(self.cfg, 'frame_trans_offset', None)
        if cfg is None or self.frame_trans_offsets is None:
            return loss, loss_dict
        w = float(getattr(cfg, 'smooth_w', 0.0) or 0.0)
        if w <= 0.0:
            return loss, loss_dict
        offsets = self.frame_trans_offsets
        smooth = (offsets[1:] - offsets[:-1]).pow(2).mean()
        loss = loss + w * smooth
        loss_dict['frame_trans_smooth'] = smooth.detach()
        return loss, loss_dict

    def setup_coarse_align(self):
        """Estimate ground plane and setup learnable scale correction for coarse align phase."""
        cfg = getattr(self.cfg, 'coarse_align', None)
        if cfg is None or not bool(getattr(cfg, 'enabled', False)):
            return
        if not hasattr(self, 'train_dataset') or self.train_dataset is None:
            return
        try:
            # Ground level ≈ body_centre_Y – scale × canonical_foot_offset
            # Using SMPL smpl_params so the estimate is anchored to the coarse alignment,
            # not to arbitrary scene point cloud percentiles.
            smpl_params = self.train_dataset.smpl_params
            transl_y = float(smpl_params['transl'][:, 1].mean())
            scale_mean = float(smpl_params['scale'].mean())
            canonical_foot_h = float(getattr(cfg, 'canonical_foot_h', 0.9))
            self.ground_y = transl_y - scale_mean * canonical_foot_h
            logger.info(
                f'Coarse align: ground_y={self.ground_y:.3f} '
                f'(transl_Y={transl_y:.3f}, scale={scale_mean:.3f}, foot_h={canonical_foot_h})'
            )
        except Exception as exc:
            logger.warning(f'Coarse align: ground_y estimation failed: {exc}')
            return
        self.scale_corrections = torch.nn.Parameter(
            torch.ones(1, device='cuda'), requires_grad=True
        )
        lr = float(getattr(cfg, 'scale_lr', 0.001))
        self.scale_corr_optimizer = torch.optim.Adam([self.scale_corrections], lr=lr)
        logger.info(f'Coarse align: scale_corrections initialized, scale_lr={lr}')

    def maybe_add_foot_contact_loss(self, loss, loss_dict, human_gs_out, iteration):
        """Foot-ground contact loss active during coarse align phase."""
        cfg = getattr(self.cfg, 'coarse_align', None)
        if cfg is None or human_gs_out is None or self.ground_y is None:
            return loss, loss_dict
        until = int(getattr(cfg, 'until_iter', 2000))
        if iteration >= until:
            return loss, loss_dict
        w = float(getattr(cfg, 'foot_contact_w', 0.0))
        if w <= 0.0:
            return loss, loss_dict
        xyz = human_gs_out['xyz']
        ratio = float(getattr(cfg, 'foot_ratio', 0.05))
        k = max(1, int(len(xyz) * ratio))
        foot_ys = xyz[:, 1].topk(k, largest=False).values
        ground_y = torch.tensor(self.ground_y, device=xyz.device, dtype=xyz.dtype)
        # penalise if feet penetrate ground
        penetration = torch.relu(ground_y - foot_ys).pow(2).mean()
        # penalise if feet float too high above ground
        float_thr = float(getattr(cfg, 'float_threshold_colmap', 0.5))
        floating = torch.relu(foot_ys - (ground_y + float_thr)).pow(2).mean()
        contact_loss = penetration + floating
        loss = loss + w * contact_loss
        loss_dict['foot_contact'] = contact_loss.detach()
        return loss, loss_dict

    def maybe_add_vitpose_kp_loss(self, loss, loss_dict, data, human_gs_out):
        """2D keypoint alignment loss: project SMPL body joints and match ViTPose detections."""
        w = float(getattr(self.cfg.human.loss, 'vitpose_kp_w', 0.0))
        if w <= 0.0 or human_gs_out is None:
            return loss, loss_dict
        vitpose_kp = data.get('vitpose_kp', None)
        smpl_joints = human_gs_out.get('smpl_joints_world', None)
        if vitpose_kp is None or smpl_joints is None:
            return loss, loss_dict

        device = smpl_joints.device
        vitpose_kp = vitpose_kp.to(device)  # (17, 3) [x, y, conf]

        # SMPL joints → COCO body keypoints (skip face joints 0-4)
        smpl_body_idx = torch.tensor([16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8], device=device)
        coco_body_idx = torch.tensor([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], device=device)

        smpl_body = smpl_joints[smpl_body_idx]  # (12, 3)
        vp_body = vitpose_kp[coco_body_idx]     # (12, 3)

        conf = vp_body[:, 2]
        W2C = data['world_view_transform'].T.float().to(device)  # 4x4
        K = data['cam_intrinsics'].float().to(device)            # 3x3

        ones = smpl_body.new_ones(smpl_body.shape[0], 1)
        joints_cam = (W2C @ torch.cat([smpl_body, ones], dim=-1).T).T[:, :3]  # (12, 3)

        valid = (conf > 0.3) & (joints_cam[:, 2] > 1e-3)
        if valid.sum() == 0:
            return loss, loss_dict

        proj = (K @ joints_cam[valid].T).T  # (N, 3)
        uv_smpl = proj[:, :2] / proj[:, 2:3]
        uv_vp = vp_body[valid, :2]

        kp_loss = ((uv_smpl - uv_vp) ** 2).mean()
        loss = loss + w * kp_loss
        loss_dict['vitpose_kp'] = kp_loss.detach()
        return loss, loss_dict

    def maybe_add_coarse_reproj_loss(self, loss, loss_dict, data, human_gs_out, iteration):
        """Mask reprojection loss for coarse align phase (active from step 0, gradient→SMPL transl+scale)."""
        cfg = getattr(self.cfg, 'coarse_align', None)
        if cfg is None or human_gs_out is None:
            return loss, loss_dict
        until = int(getattr(cfg, 'until_iter', 2000))
        if iteration >= until:
            return loss, loss_dict
        w = float(getattr(cfg, 'mask_reproj_w', 0.0))
        if w <= 0.0:
            return loss, loss_dict
        mask = data.get('mask', None)
        if mask is None:
            return loss, loss_dict
        device = human_gs_out['xyz'].device
        mask = mask.to(device)
        ys, xs = torch.where(mask > 0.5)
        if xs.numel() == 0:
            return loss, loss_dict
        mask_centroid = torch.stack([xs.float().mean(), ys.float().mean()])
        xyz = human_gs_out['xyz']
        with torch.no_grad():
            opacity_w = human_gs_out['opacity'].squeeze(-1)
            opacity_w = opacity_w / (opacity_w.sum() + 1e-6)
        xyz_weighted = (xyz * opacity_w[:, None]).sum(0)
        W2C = data['world_view_transform'].T.float().to(device)
        xyz_homo = torch.cat([xyz_weighted, xyz_weighted.new_ones(1)])
        xyz_cam = (W2C @ xyz_homo)[:3]
        if xyz_cam[2].item() < 1e-3:
            return loss, loss_dict
        K = data['cam_intrinsics'].float().to(device)
        xyz_proj = K @ xyz_cam
        uv_human = xyz_proj[:2] / xyz_proj[2]
        reproj_loss = (uv_human - mask_centroid).pow(2).mean()
        loss = loss + w * reproj_loss
        loss_dict['coarse_reproj'] = reproj_loss.detach()
        return loss, loss_dict

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

    # ------------------------------------------------------------------ #
    # Mask-aware scene training: suppress floaters in human mask region  #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _project_scene_to_image(self, data):
        """Project scene GS centers to pixel coords for the current camera frame."""
        xyz = self.scene_gs.get_xyz           # (N_scene, 3)
        H = int(data['image_height'])
        W = int(data['image_width'])
        full_proj = data['full_proj_transform'].to(xyz.device)   # (4, 4)
        ones = torch.ones(xyz.shape[0], 1, device=xyz.device, dtype=xyz.dtype)
        xyz_h = torch.cat([xyz, ones], dim=-1)                   # (N, 4)
        proj = xyz_h @ full_proj                                  # (N, 4)
        w = proj[:, 3]
        valid = w > 1e-6
        w_safe = w.clamp(min=1e-8)
        ndc_x = proj[:, 0] / w_safe
        ndc_y = proj[:, 1] / w_safe
        px = ((ndc_x + 1.0) * 0.5 * W).long()
        py = ((1.0 - ndc_y) * 0.5 * H).long()
        valid = valid & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        return px, py, valid, H, W

    def maybe_add_scene_mask_aware_loss(self, loss, loss_dict, data, scene_gs_out, bg_color):
        """Scene-only background loss: scene GS trained only on non-human pixels."""
        s_cfg = self.cfg.scene
        if not bool(getattr(s_cfg, 'mask_aware_enabled', False)):
            return loss
        w = float(getattr(s_cfg, 'mask_aware_loss_w', 0.0) or 0.0)
        if w <= 0.0 or scene_gs_out is None:
            return loss

        scene_pkg = render_human_scene(
            data=data,
            human_gs_out=None,
            scene_gs_out=scene_gs_out,
            bg_color=bg_color,
            render_mode='scene',
        )
        bg_mask = (1.0 - data['mask']).unsqueeze(0)          # (1, H, W)  1=background
        scene_bg_l1 = _l1_loss(scene_pkg['render'] * bg_mask, data['rgb'] * bg_mask)
        loss = loss + w * scene_bg_l1
        loss_dict['scene_mask_aware'] = scene_bg_l1.detach()
        return loss

    def maybe_add_scene_opacity_reg_loss(self, loss, loss_dict, scene_gs_out):
        """Push scene Gaussian opacities toward 1 (fully opaque): loss = mean((1 - opacity)^2)."""
        if scene_gs_out is None:
            return loss, loss_dict
        w = float(getattr(getattr(self.cfg.scene, 'loss', None), 'opacity_reg_w', 0.0) or 0.0)
        if w <= 0.0:
            return loss, loss_dict
        opacity = scene_gs_out['opacity']  # already sigmoid-activated, shape [N, 1]
        reg = (1.0 - opacity).pow(2).mean()
        loss = loss + w * reg
        loss_dict['scene_opacity_reg'] = reg.detach()
        return loss, loss_dict

    def maybe_add_scene_human_opacity_suppress_loss(self, loss, loss_dict, scene_gs_out, human_gs_out, iteration):
        """Suppress scene GS opacity within human bounding sphere.

        Corrupted scene GS in round-trip-trajectory regions occlude human GS during rendering.
        Penalizing their opacity toward 0 lets human GS render through them without moving them.
        """
        if scene_gs_out is None or human_gs_out is None:
            return loss, loss_dict
        loss_cfg = getattr(self.cfg.scene, 'loss', None)
        w = float(getattr(loss_cfg, 'human_region_opacity_suppress_w', 0.0) or 0.0)
        if w <= 0.0:
            return loss, loss_dict
        until_iter = int(getattr(loss_cfg, 'human_region_opacity_suppress_until', 1_000_000) or 1_000_000)
        if iteration > until_iter:
            return loss, loss_dict

        scene_xyz = scene_gs_out.get('xyz', None)
        human_xyz = human_gs_out.get('xyz', None)
        if scene_xyz is None or human_xyz is None:
            return loss, loss_dict
        if scene_xyz.shape[0] == 0 or human_xyz.shape[0] == 0:
            return loss, loss_dict

        with torch.no_grad():
            human_center = human_xyz.detach().mean(0)
            dists_h = (human_xyz.detach() - human_center).norm(dim=-1)
            human_radius = dists_h.quantile(0.95) * 1.3
            scene_dist = (scene_xyz.detach() - human_center).norm(dim=-1)
            in_human = scene_dist < human_radius

        if not in_human.any():
            return loss, loss_dict

        scene_opacity = scene_gs_out['opacity']  # [N, 1], sigmoid-activated
        suppress_loss = scene_opacity[in_human].mean()
        loss = loss + w * suppress_loss
        loss_dict['scene_human_opacity_suppress'] = suppress_loss.detach()
        return loss, loss_dict

    def maybe_add_scene_behind_human_depth_loss(self, loss, loss_dict, data, scene_gs_out, human_gs_out):
        """Per-pixel hinge loss: in human mask pixels, scene GS must be deeper than human GS.

        Renders scene-only and human-only depth maps independently, then penalises
        pixels in the human mask where the scene depth is shallower (closer to camera)
        than the human depth.  This directly counteracts scene GS invasion without
        relying on the global Pearson depth loss whose gradient is too weak in the
        human region.
        """
        if scene_gs_out is None or human_gs_out is None:
            return loss, loss_dict

        loss_cfg = getattr(self.cfg.scene, 'loss', None)
        w = float(getattr(loss_cfg, 'scene_behind_human_depth_w', 0.0) or 0.0)
        if w <= 0.0:
            return loss, loss_dict

        if 'mask' not in data:
            return loss, loss_dict

        margin = float(getattr(loss_cfg, 'scene_behind_human_depth_margin', 0.05) or 0.05)

        scene_depth = render_depth_map(
            scene_gs_out['xyz'], scene_gs_out['opacity'],
            scene_gs_out['scales'], scene_gs_out['rotq'], data,
        )  # (1, H, W), camera-space z; larger = further from camera
        human_depth = render_depth_map(
            human_gs_out['xyz'], human_gs_out['opacity'],
            human_gs_out['scales'], human_gs_out['rotq'], data,
        )  # (1, H, W)

        human_mask = data['mask'].unsqueeze(0)              # (1, H, W)
        # Only consider pixels where the human is present AND some scene GS exists there
        valid = (human_mask > 0.5) & (scene_depth > 1e-3)
        if not valid.any():
            return loss, loss_dict

        # Penalise when scene is closer than human: relu(human_depth - scene_depth + margin)
        # Detach human_depth so gradients only flow to scene GS, not human GS.
        violation = torch.relu(human_depth.detach() - scene_depth + margin)
        n_valid = valid.float().sum().clamp(min=1.0)
        depth_loss = (violation * valid.float()).sum() / n_valid

        loss = loss + w * depth_loss
        loss_dict['scene_behind_human_depth'] = depth_loss.detach()
        return loss, loss_dict

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

        t_cfg = getattr(a_cfg, 'temporal', None)
        use_temporal = t_cfg is not None and bool(getattr(t_cfg, 'enabled', False))
        if use_temporal:
            self.anchor_attention = TemporalAnchorAttention(
                a_cfg,
                t_cfg,
                num_anchors=len(anchors['names']),
                top_m=int(getattr(a_cfg, 'top_m', 2)),
            ).to('cuda')
            logger.info(
                f'Temporal anchor-attention enabled: '
                f'memory_size={getattr(t_cfg, "memory_size", 100)}, '
                f'window={getattr(t_cfg, "temporal_window", 5)}, '
                f'smooth_loss_w={getattr(t_cfg, "smooth_loss_w", 0.01)}'
            )
        else:
            _n_frames = len(self.train_dataset) if hasattr(self, 'train_dataset') else len(self.val_dataset)
            # In eval mode, infer num_frames from checkpoint to avoid frame_embed size mismatch
            _attn_ckpt = str(getattr(a_cfg, 'ckpt', '') or '')
            if _attn_ckpt and self.cfg.eval:
                try:
                    _ckpt_sd = torch.load(_attn_ckpt, map_location='cpu', weights_only=False)
                    if 'frame_embed.weight' in _ckpt_sd:
                        _n_frames = _ckpt_sd['frame_embed.weight'].shape[0]
                        logger.info(f'Eval mode: inferred num_frames={_n_frames} from anchor_attention ckpt')
                except Exception:
                    pass
            self.anchor_attention = AnchorSceneAttentionBaseline(
                a_cfg,
                num_anchors=len(anchors['names']),
                top_m=int(getattr(a_cfg, 'top_m', 2)),
                num_frames=_n_frames,
            ).to('cuda')
        ckpt = str(getattr(a_cfg, 'ckpt', '') or '')
        if ckpt:
            # strict=False in eval mode (frame_embed size may differ) or temporal compatibility
            _strict = (not use_temporal) and (not self.cfg.eval)
            missing, unexpected = self.anchor_attention.load_state_dict(
                torch.load(ckpt, weights_only=False), strict=_strict
            )
            if missing:
                logger.info(f'Anchor-attention load: {len(missing)} missing params (expected in eval/temporal mode)')
            logger.info(f'Loaded anchor-attention module from {ckpt}')
        self.anchor_attention_optimizer = torch.optim.Adam(
            self.anchor_attention.parameters(),
            lr=float(getattr(a_cfg, 'lr', 1e-4)),
        )
        logger.info(f'Anchor-attention baseline initialized with {len(anchors["names"])} anchors; debug files in {debug_dir}')

    def maybe_apply_anchor_attention(self, human_gs_out, scene_gs_out, render_mode, iteration,
                                     frame_idx=None):
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
            frame_idx=frame_idx,
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

            if self.fusion_mlp is not None:
                self.fusion_mlp.update_learning_rate(t_iter)

            rnd_idx = next(rand_idx_iter)
            data = self.train_dataset[rnd_idx]
            
            human_gs_out, scene_gs_out = None, None
            
            if self.human_gs:
                _ca_cfg = getattr(self.cfg, 'coarse_align', None)
                _ca_until = int(getattr(_ca_cfg, 'until_iter', 2000)) if _ca_cfg else 0
                if self.scale_corrections is not None and t_iter < _ca_until:
                    _eff_scale = data['smpl_scale'][None] * self.scale_corrections
                else:
                    _eff_scale = data['smpl_scale'][None]
                human_gs_out = self.human_gs.forward(
                    smpl_scale=_eff_scale,
                    dataset_idx=rnd_idx,
                    is_train=True,
                    ext_tfs=None,
                )
            
            if self.scene_gs:
                if t_iter >= self.cfg.scene.opt_start_iter:
                    scene_gs_out = self.scene_gs.forward()
                else:
                    render_mode = 'human'

            # per-frame coarse alignment correction (before anchor attention)
            human_gs_out = self.maybe_apply_frame_trans_offset(human_gs_out, rnd_idx, iteration=t_iter)

            anchor_stats = {}
            human_gs_out, anchor_stats = self.maybe_apply_anchor_attention(
                human_gs_out,
                scene_gs_out,
                render_mode,
                t_iter,
                frame_idx=rnd_idx,
            )
            
            bg_color = torch.rand(3, dtype=torch.float32, device="cuda")


            if self.cfg.human.loss.humansep_w > 0.0 and render_mode == 'human_scene':
                render_human_separate = True
                human_bg_color = torch.rand(3, dtype=torch.float32, device="cuda")
            else:
                human_bg_color = None
                render_human_separate = False

            fused_gs_out = None
            if self.fusion_mlp is not None and render_mode == 'human_scene' and scene_gs_out is not None:
                human_props = {
                    'shs': human_gs_out['shs'].reshape(-1, 48),
                    'xyz': human_gs_out['xyz'],
                    'opacity': human_gs_out['opacity'],
                    'scales': human_gs_out['scales'],
                    'rotq': human_gs_out['rotq'],
                }
                scene_props = {
                    'shs': scene_gs_out['shs'].reshape(-1, 48),
                    'xyz': scene_gs_out['xyz'],
                    'opacity': scene_gs_out['opacity'],
                    'scales': scene_gs_out['scales'],
                    'rotq': scene_gs_out['rotq'],
                }
                fused_gs_out = self.fusion_mlp(human_props, scene_props)

            render_pkg = render_human_scene(
                data=data,
                human_gs_out=human_gs_out,
                scene_gs_out=scene_gs_out,
                bg_color=bg_color,
                human_bg_color=human_bg_color,
                render_mode=render_mode,
                render_human_separate=render_human_separate,
                fused_gs_out=fused_gs_out,
            )

            # Depth supervision: render depth map when loss is enabled and mono depth is available.
            if self.loss_fn.l_depth_w > 0.0 and 'mono_depth' in data:
                if render_mode == 'human_scene':
                    depth_means = torch.cat([human_gs_out['xyz'], scene_gs_out['xyz']], dim=0)
                    depth_opacity = torch.cat([human_gs_out['opacity'], scene_gs_out['opacity']], dim=0)
                    depth_scales = torch.cat([human_gs_out['scales'], scene_gs_out['scales']], dim=0)
                    depth_rotations = torch.cat([human_gs_out['rotq'], scene_gs_out['rotq']], dim=0)
                elif render_mode == 'human':
                    depth_means = human_gs_out['xyz']
                    depth_opacity = human_gs_out['opacity']
                    depth_scales = human_gs_out['scales']
                    depth_rotations = human_gs_out['rotq']
                else:
                    depth_means = scene_gs_out['xyz']
                    depth_opacity = scene_gs_out['opacity']
                    depth_scales = scene_gs_out['scales']
                    depth_rotations = scene_gs_out['rotq']
                render_pkg['depth'] = render_depth_map(
                    depth_means, depth_opacity, depth_scales, depth_rotations, data
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
                loss = self.maybe_add_scene_mask_aware_loss(loss, loss_dict, data, scene_gs_out, bg_color)
                loss, loss_dict = self.maybe_add_scene_opacity_reg_loss(loss, loss_dict, scene_gs_out)
                loss, loss_dict = self.maybe_add_scene_human_opacity_suppress_loss(loss, loss_dict, scene_gs_out, human_gs_out, t_iter)
                loss, loss_dict = self.maybe_add_scene_behind_human_depth_loss(loss, loss_dict, data, scene_gs_out, human_gs_out)

            loss, loss_dict = self.maybe_add_mask_reproj_loss(loss, loss_dict, data, human_gs_out, iteration=t_iter)
            loss, loss_dict = self.maybe_add_frame_trans_smooth_loss(loss, loss_dict)
            loss, loss_dict = self.maybe_add_coarse_reproj_loss(loss, loss_dict, data, human_gs_out, iteration=t_iter)
            loss, loss_dict = self.maybe_add_foot_contact_loss(loss, loss_dict, human_gs_out, iteration=t_iter)
            loss, loss_dict = self.maybe_add_vitpose_kp_loss(loss, loss_dict, data, human_gs_out)

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

            loss = self.maybe_add_floor_grounding_loss(
                loss, loss_dict, data, scene_gs_out, anchor_stats, rnd_idx, t_iter
            )

            loss = self.maybe_add_delta_correction_smooth_loss(
                loss, loss_dict, anchor_stats, rnd_idx
            )

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
                    if render_mode == 'human_scene' and human_gs_out is not None:
                        # viewspace_points is [human|scene]; pass only the scene slice to
                        # scene densification so grad stats are computed on the correct points.
                        human_n_gs = human_gs_out['xyz'].shape[0]
                        scene_n_gs = render_pkg['scene_visibility_filter'].shape[0]
                        scene_vp = torch.zeros(scene_n_gs, 3, device='cuda')
                        scene_vp.grad = render_pkg['viewspace_points'].grad[human_n_gs:].clone()
                        render_pkg['scene_viewspace_points'] = scene_vp
                    else:
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

            if self.fusion_mlp is not None:
                self.fusion_mlp.optimizer.step()
                self.fusion_mlp.optimizer.zero_grad(set_to_none=True)

            if self.frame_trans_optimizer is not None:
                _ft_cfg = getattr(self.cfg, 'frame_trans_offset', None)
                _ft_start = int(getattr(_ft_cfg, 'offset_start_iter', 0)) if _ft_cfg else 0
                if t_iter >= _ft_start:
                    self.frame_trans_optimizer.step()
                self.frame_trans_optimizer.zero_grad(set_to_none=True)

            if self.scale_corr_optimizer is not None:
                _ca_cfg = getattr(self.cfg, 'coarse_align', None)
                _ca_until = int(getattr(_ca_cfg, 'until_iter', 2000)) if _ca_cfg else 0
                if t_iter < _ca_until:
                    self.scale_corr_optimizer.step()
                self.scale_corr_optimizer.zero_grad(set_to_none=True)

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

            _ply_interval = int(getattr(self.cfg.train, 'debug_ply_interval', 0) or 0)
            if _ply_interval > 0 and t_iter % _ply_interval == 0:
                self._save_debug_ply(t_iter)

            if self.cfg.train.save_progress_images and t_iter % self.cfg.train.progress_save_interval == 0 and self.cfg.mode in ['human', 'human_scene']:
                self.render_canonical(t_iter, nframes=2, is_train_progress=True)
        
        # train progress images
        if self.cfg.train.save_progress_images:
            video_fname = f'{self.cfg.logdir}/train_{self.cfg.dataset.name}_{self.cfg.dataset.seq}.mp4'
            create_video(f'{self.cfg.logdir}/train_progress/', video_fname, fps=10)
            shutil.rmtree(f'{self.cfg.logdir}/train_progress/')
            
    def _save_debug_ply(self, iteration):
        """每 debug_ply_interval 步保存一次彩色点云快照（二进制 PLY）。
        gray  = scene GS 远离人体部分（下采样 50K）
        orange= scene GS 人体周围 1.5 COLMAP 半径内（全量保留）
        red   = human GS（世界坐标，mid-val 帧）
        white = COLMAP sparse 原始点云
        """
        ply_dir = os.path.join(self.cfg.logdir, 'debug_ply')
        os.makedirs(ply_dir, exist_ok=True)

        parts_xyz, parts_rgb = [], []

        # ── 获取人体中心（用于空间感知下采样）────────────────────────────────
        human_center = None
        mid_data = None
        if self.val_dataset is not None and len(self.val_dataset) > 0:
            mid_idx = len(self.val_dataset) // 2
            mid_data = self.val_dataset[mid_idx]
            human_center = mid_data['transl'].cpu().numpy()  # (3,) world coords

        # ── Scene GS（空间感知下采样：人体附近全保留，远处采 50K）────────────
        if self.scene_gs is not None:
            s_xyz_all = self.scene_gs.get_xyz.detach().cpu().numpy()
            if human_center is not None:
                dist = np.linalg.norm(s_xyz_all - human_center, axis=1)
                near_mask = dist < 1.5  # 1.5 COLMAP 单位 ≈ 约 1 体高半径
                near_xyz = s_xyz_all[near_mask]
                far_xyz  = s_xyz_all[~near_mask]
                if len(far_xyz) > 50_000:
                    idx = np.random.choice(len(far_xyz), 50_000, replace=False)
                    far_xyz = far_xyz[idx]
                # 近处橙色（侵入区），远处灰色
                near_rgb = np.full((len(near_xyz), 3), [255, 140, 0], dtype=np.uint8)
                far_rgb  = np.full((len(far_xyz),  3), [140, 140, 140], dtype=np.uint8)
                parts_xyz.extend([near_xyz, far_xyz])
                parts_rgb.extend([near_rgb,  far_rgb])
                logger.info(f"{iteration:06d} - debug PLY: scene GS near={len(near_xyz):,} far={len(far_xyz):,}")
            else:
                s_xyz = s_xyz_all
                if len(s_xyz) > 200_000:
                    idx = np.random.choice(len(s_xyz), 200_000, replace=False)
                    s_xyz = s_xyz[idx]
                parts_xyz.append(s_xyz)
                parts_rgb.append(np.full((len(s_xyz), 3), [150, 150, 150], dtype=np.uint8))

        # ── Human GS（匹配 training loop 的 forward 调用，不做 unsqueeze）─────
        if self.human_gs is not None and mid_data is not None:
            try:
                with torch.no_grad():
                    h_out = self.human_gs.forward(
                        global_orient=mid_data['global_orient'].to('cuda'),
                        body_pose=mid_data['body_pose'].to('cuda'),
                        betas=mid_data['betas'].to('cuda'),
                        transl=mid_data['transl'].to('cuda'),
                        smpl_scale=mid_data['smpl_scale'][None].to('cuda'),
                        dataset_idx=-1,
                        is_train=False,
                        ext_tfs=None,
                    )
                h_xyz = h_out['xyz'].detach().cpu().numpy()
                h_rgb = np.full((len(h_xyz), 3), [220, 30, 30], dtype=np.uint8)
                parts_xyz.append(h_xyz)
                parts_rgb.append(h_rgb)
                logger.info(f"{iteration:06d} - debug PLY: human GS {len(h_xyz):,} pts")
            except Exception as exc:
                logger.warning(f"debug PLY human forward failed: {exc}")

        # ── COLMAP sparse（缓存加载）─────────────────────────────────────────
        if not hasattr(self, '_debug_colmap_xyz'):
            from hugs.cfg.constants import NEUMAN_PATH
            seq = getattr(self.cfg.dataset, 'seq', '')
            base_seq = seq
            for suf in ['_vimo_v4', '_vimo_v3', '_vimo', '_gt']:
                if base_seq.endswith(suf):
                    base_seq = base_seq[:-len(suf)]
                    break
            self._debug_colmap_xyz = None
            for try_seq in [base_seq, seq]:
                txt = os.path.join(NEUMAN_PATH, try_seq, 'sparse', 'points3D.txt')
                if os.path.exists(txt):
                    xyz_l, rgb_l = [], []
                    with open(txt) as f:
                        for line in f:
                            if line.startswith('#') or not line.strip():
                                continue
                            p = line.split()
                            xyz_l.append([float(p[1]), float(p[2]), float(p[3])])
                            rgb_l.append([int(p[4]), int(p[5]), int(p[6])])
                    self._debug_colmap_xyz = np.array(xyz_l, dtype=np.float32)
                    self._debug_colmap_rgb = np.array(rgb_l, dtype=np.uint8)
                    logger.info(f"debug PLY: loaded {len(self._debug_colmap_xyz)} COLMAP pts from {txt}")
                    break

        if self._debug_colmap_xyz is not None:
            parts_xyz.append(self._debug_colmap_xyz)
            parts_rgb.append(self._debug_colmap_rgb)

        if not parts_xyz:
            return

        all_xyz = np.concatenate(parts_xyz, axis=0).astype(np.float32)
        all_rgb = np.concatenate(parts_rgb, axis=0).astype(np.uint8)
        n = len(all_xyz)

        ply_path = os.path.join(ply_dir, f'gs_snapshot_{iteration:06d}.ply')
        header = (
            f"ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            f"property float x\nproperty float y\nproperty float z\n"
            f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"end_header\n"
        ).encode('ascii')

        dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                       ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        rec = np.empty(n, dtype=dt)
        rec['x'] = all_xyz[:, 0]; rec['y'] = all_xyz[:, 1]; rec['z'] = all_xyz[:, 2]
        rec['red'] = all_rgb[:, 0]; rec['green'] = all_rgb[:, 1]; rec['blue'] = all_rgb[:, 2]

        with open(ply_path, 'wb') as f:
            f.write(header)
            rec.tofile(f)

        logger.info(f"{iteration:06d} - debug PLY saved ({n:,} pts) → {ply_path}")

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

        if self.frame_trans_offsets is not None:
            try:
                torch.save(self.frame_trans_offsets.data, f'{self.cfg.logdir_ckpt}/frame_trans_offsets_{iter_s}.pth')
            except Exception as exc:
                logger.warning(f'Failed to save frame_trans_offsets checkpoint {iter_s}: {exc}')

        logger.info(f'Saved checkpoint {iter_s}')
                
    def scene_densification(self, visibility_filter, radii, viewspace_point_tensor, iteration, data=None):
        # mask-aware densify: exclude scene GS projecting into human mask from grad stats
        if (bool(getattr(self.cfg.scene, 'mask_aware_enabled', False))
                and bool(getattr(self.cfg.scene, 'mask_aware_densify', True))
                and data is not None and 'mask' in data):
            with torch.no_grad():
                px, py, valid_proj, H, W = self._project_scene_to_image(data)
                human_mask_2d = (data['mask'] > 0.5).to(visibility_filter.device)
                in_human_mask = valid_proj & human_mask_2d[py.clamp(0, H - 1), px.clamp(0, W - 1)]
                visibility_filter = visibility_filter & ~in_human_mask

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

            # mask-aware prune: after densify_and_prune, remove scene GS whose centers
            # project into the human mask.  Prevents re-invasion after opacity reset.
            if (bool(getattr(self.cfg.scene, 'mask_aware_enabled', False))
                    and bool(getattr(self.cfg.scene, 'mask_aware_prune', False))
                    and data is not None and 'mask' in data):
                with torch.no_grad():
                    px_p, py_p, vld_p, H_p, W_p = self._project_scene_to_image(data)
                    hm2d_p = (data['mask'] > 0.5).to(self.scene_gs.get_xyz.device)
                    in_hm_p = vld_p & hm2d_p[py_p.clamp(0, H_p - 1), px_p.clamp(0, W_p - 1)]
                    if in_hm_p.any():
                        logger.info(
                            f"[{iteration:06d}] mask_aware_prune: removing "
                            f"{int(in_hm_p.sum().item())} scene GS from human mask "
                            f"(scene total: {self.scene_gs.get_xyz.shape[0]})"
                        )
                        self.scene_gs.prune_points(in_hm_p)

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
        invasion_counts = []
        invasion_opacities = []
        
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
                frame_idx=idx,
            )

            fused_gs_out = None
            if self.fusion_mlp is not None and render_mode == 'human_scene' and scene_gs_out is not None:
                human_props = {
                    'shs': human_gs_out['shs'].reshape(-1, 48),
                    'xyz': human_gs_out['xyz'],
                    'opacity': human_gs_out['opacity'],
                    'scales': human_gs_out['scales'],
                    'rotq': human_gs_out['rotq'],
                }
                scene_props = {
                    'shs': scene_gs_out['shs'].reshape(-1, 48),
                    'xyz': scene_gs_out['xyz'],
                    'opacity': scene_gs_out['opacity'],
                    'scales': scene_gs_out['scales'],
                    'rotq': scene_gs_out['rotq'],
                }
                fused_gs_out = self.fusion_mlp(human_props, scene_props)

            render_pkg = render_human_scene(
                data=data,
                human_gs_out=human_gs_out,
                scene_gs_out=scene_gs_out,
                bg_color=bg_color,
                render_mode=render_mode,
                fused_gs_out=fused_gs_out,
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

            # Scene GS invasion into human bbox metric
            if human_gs_out is not None and scene_gs_out is not None:
                with torch.no_grad():
                    h_xyz = human_gs_out['xyz'].detach()         # [N_human, 3]
                    s_xyz = scene_gs_out['xyz'].detach()         # [N_scene, 3]
                    s_opacity = torch.sigmoid(scene_gs_out['opacity'].detach().squeeze(-1))  # [N_scene]
                    margin = 0.1
                    bbox_min = h_xyz.min(0).values - margin
                    bbox_max = h_xyz.max(0).values + margin
                    in_bbox = ((s_xyz >= bbox_min) & (s_xyz <= bbox_max)).all(-1)
                    invasion_counts.append(in_bbox.sum().item())
                    invasion_opacities.append((s_opacity * in_bbox.float()).sum().item())
        
        
        self.eval_metrics[iter_s] = {}
        
        for k, v in metrics.items():
            if v == []:
                continue
            
            logger.info(f"{iter_s} - {k.upper()}: {torch.stack(v).mean().item():.4f}")
            self.eval_metrics[iter_s][k] = torch.stack(v).mean().item()
        
        torch.save(metrics, f'{self.cfg.logdir}/val/eval_{iter_s}.pth')

        # Log scene GS invasion into human bbox
        if invasion_counts:
            avg_count = sum(invasion_counts) / len(invasion_counts)
            avg_opacity = sum(invasion_opacities) / len(invasion_opacities)
            logger.info(f"{iter_s} - SCENE_INVASION_COUNT: {avg_count:.1f}")
            logger.info(f"{iter_s} - SCENE_INVASION_OPACITY: {avg_opacity:.2f}")
            self.eval_metrics[iter_s]['scene_invasion_count'] = avg_count
            self.eval_metrics[iter_s]['scene_invasion_opacity'] = avg_opacity

        # Log transl alignment vs GT
        if getattr(self, 'gt_transl', None) is not None and self.human_gs and hasattr(self.human_gs, 'transl'):
            _err = torch.norm(self.human_gs.transl.detach().cpu() - self.gt_transl, dim=1)
            logger.info(f"{iter_s} - TRANSL_L2_VS_GT: mean={_err.mean():.4f}, max={_err.max():.4f}")
            self.eval_metrics[iter_s]['transl_l2_vs_gt_mean'] = _err.mean().item()
            self.eval_metrics[iter_s]['transl_l2_vs_gt_max'] = _err.max().item()

        # save best checkpoint based on human PSNR
        human_psnr = self.eval_metrics[iter_s].get('hugs_human_psnr', -1.0)
        if human_psnr > self._best_human_psnr:
            self._best_human_psnr = human_psnr
            self.save_ckpt(iter)
            logger.info(f'New best human PSNR={human_psnr:.4f} at iter {iter_s}, saved best checkpoint')

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
                frame_idx=idx,
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
        
        if hasattr(self.human_gs, 'betas'):
            betas = self.human_gs.betas.detach()
        elif hasattr(self, 'train_dataset'):
            betas = self.train_dataset.betas[0]
        else:
            betas = torch.stack([x['betas'] for x in self.val_dataset.cached_data], dim=0)[0]

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
