"""
批量生成完整验证所需的训练配置文件。

每个场景生成三类配置：
  1. OUR  — HUGS + VIMO + anchor attention (18k steps)
  2. STM  — StM + VIMO (20k steps)
  3. HUGS — HUGS + VIMO baseline, 无 attention (15k steps)

用法：
  cd ml-hugs-work
  python scripts/gen_validation_configs.py
"""

import os
import yaml

SCENES = ['lab', 'bike', 'citron', 'jogging', 'parkinglot', 'seattle']
DATE = '20260607'

# ─── OUR pipeline 模板（基于 Exp G） ─────────────────────────────────────────
OUR_TEMPLATE = {
    'seed': 0,
    'mode': 'human_scene',
    'output_path': 'output',
    'dataset_path': '',
    'detect_anomaly': False,
    'debug': False,
    'wandb': False,
    'eval': False,
    'bg_color': 'white',
    'dataset': {
        'name': 'neuman',
        'seq': None,  # 填充
        'mono_depth_dir': None,  # 填充
    },
    'train': {
        'batch_size': 1,
        'num_workers': 0,
        'num_steps': 17998,
        'save_ckpt_interval': 6000,
        'val_interval': 1000,
        'anim_interval': -1,
        'optim_scene': True,
        'save_progress_images': True,
        'progress_save_interval': 1000,
    },
    'human': {
        'name': 'hugs_trimlp',
        'ckpt': None,
        'sh_degree': 0,
        'n_subdivision': 2,
        'only_rgb': False,
        'use_surface': False,
        'use_deformer': True,
        'init_2d': False,
        'disable_posedirs': True,
        'res_offset': False,
        'rotate_sh': False,
        'isotropic': False,
        'init_scale_multiplier': 0.5,
        'run_init': False,
        'estimate_delta': True,
        'triplane_res': 256,
        'optim_pose': True,
        'optim_betas': False,
        'optim_trans': True,
        'optim_eps_offsets': False,
        'activation': 'relu',
        'canon_nframes': 60,
        'canon_pose_type': 'da_pose',
        'knn_n_hops': 3,
        'lr': {
            'wd': 0.0,
            'position': 0.00016,
            'position_init': 0.00016,
            'position_final': 1.6e-06,
            'position_delay_mult': 0.01,
            'position_max_steps': 30000,
            'opacity': 0.05,
            'scaling': 0.005,
            'rotation': 0.001,
            'feature': 0.0025,
            'smpl_spatial': 2.0,
            'smpl_pose': 0.0001,
            'smpl_betas': 0.0001,
            'smpl_trans': 0.0005,
            'smpl_eps_offset': 0.0001,
            'lbs_weights': 0.0,
            'posedirs': 0.0,
            'percent_dense': 0.01,
            'appearance': 0.001,
            'geometry': 0.001,
            'vembed': 0.001,
            'deformation': 0.0001,
            'scale_lr_w_npoints': False,
        },
        'loss': {
            'ssim_w': 0.2,
            'l1_w': 0.8,
            'lpips_w': 1.0,
            'lbs_w': 1000.0,
            'humansep_w': 1.0,
            'depth_w': 0.05,
            'num_patches': 4,
            'patch_size': 128,
            'use_patches': 1,
        },
        'densification_interval': 600,
        'opacity_reset_interval': 3000,
        'densify_from_iter': 3000,
        'densify_until_iter': 15000,
        'densify_grad_threshold': 0.0002,
        'prune_min_opacity': 0.005,
        'densify_extent': 1.0,
        'max_n_gaussians': 524288,
        'percent_dense': 0.01,
    },
    'scene': {
        'name': 'scene_gs',
        'ckpt': None,
        'sh_degree': 3,
        'add_bg_points': False,
        'num_bg_points': 204800,
        'bg_sphere_dist': 5.0,
        'clean_pcd': False,
        'init_pcd_path': None,
        'opt_start_iter': -1,
        'lr': {
            'percent_dense': 0.01,
            'spatial_scale': 1.0,
            'position_init': 0.00016,
            'position_final': 1.6e-06,
            'position_delay_mult': 0.01,
            'position_max_steps': 30000,
            'opacity': 0.05,
            'scaling': 0.005,
            'rotation': 0.001,
            'feature': 0.0025,
        },
        'percent_dense': 0.01,
        'densification_interval': 100,
        'opacity_reset_interval': 20000,
        'densify_from_iter': 500,
        'densify_until_iter': 15000,
        'densify_grad_threshold': 0.0002,
        'prune_min_opacity': 0.005,
        'max_n_gaussians': 2097152,
        'loss': {'ssim_w': 0.2, 'l1_w': 0.8},
    },
    'scene_sugar': {'enabled': False},
    'scene_global_aware_adc': {'enabled': False},
    'anchor_attention': {
        'use_anchors': True,
        'anchor_vertices_path': '',
        'count_per_anchor': 64,
        'top_m': 2,
        'lambda_lbs': 0.35,
        'sigma_scale': 1.8,
        'min_sigma': 0.035,
        'use_anchor_token_encoder': True,
        'use_scene_query': True,
        'use_cross_attention': True,
        'use_interaction_correction': True,
        'correct_xyz': True,
        'correct_opacity': False,
        'correct_scale': False,
        'correct_feature_dc': False,
        'correct_transl': True,
        'gamma_mu': 0.005,
        'module_start_iter': 2000,
        'correction_start_iter': 2000,
        'correction_warmup_iters': 500,
        'gamma_transl_init': 2.0,
        'gamma_transl_final': 0.05,
        'gamma_transl_decay_start': 2000,
        'gamma_transl_decay_end': 15000,
        'transl_delta_clamp': 0.5,
        'hidden_dim': 128,
        'human_pooling': 'attention',
        'scene_topk': 32,
        'scene_opacity_threshold': 0.01,
        'max_scene_candidates': 200000,
        'zero_init_delta': True,
        'gamma_opacity': 0.05,
        'delta_loss_w': 0.001,
        'delta_correction_smooth_w': 0.0,
        'lr': 0.0001,
        'debug_interval': 1000,
        'debug_dir': '',
        'ckpt': '',
    },
}

# ─── STM 模板 ─────────────────────────────────────────────────────────────────
STM_TEMPLATE = {
    'seed': 10086,
    'mode': 'human_scene',
    'output_path': 'output_stm',
    'wandb': False,
    'eval': False,
    'bg_color': 'white',
    'use_hugs': False,
    'dataset': {'name': 'neuman', 'seq': None},
    'train': {
        'batch_size': 1,
        'num_workers': 0,
        'num_steps': 20000,
        'save_ckpt_interval': 5000,
        'val_interval': 1000,
        'anim_interval': 20000,
        'optim_scene': True,
        'save_progress_images': False,
        'progress_save_interval': 10,
    },
    'human': {
        'name': 'hugs_trimlp',
        'sh_degree': 0,
        'n_subdivision': 2,
        'use_deformer': True,
        'disable_posedirs': True,
        'init_scale_multiplier': 0.5,
        'optim_pose': True,
        'optim_trans': True,
        'densification_interval': 600,
        'densify_from_iter': 3000,
        'densify_until_iter': 15000,
        'densify_extent': 1.0,
        'max_n_gaussians': 524288,
        'loss': {
            'ssim_w': 0.2, 'l1_w': 0.8, 'lpips_w': 1.0,
            'lbs_w': 100.0, 'depth_w': 0.03,
            'humansep_w': 1.0, 'num_patches': 4,
            'patch_size': 128, 'use_patches': 1,
        },
    },
    'scene': {
        'opacity_reset_interval': 20000,
        'densify_until_iter': 15000,
    },
}

# ─── HUGS baseline 模板 ───────────────────────────────────────────────────────
HUGS_BASE_TEMPLATE = {
    'seed': 0,
    'mode': 'human_scene',
    'output_path': 'output',
    'dataset_path': '',
    'detect_anomaly': False,
    'debug': False,
    'wandb': False,
    'eval': False,
    'bg_color': 'white',
    'dataset': {'name': 'neuman', 'seq': None, 'mono_depth_dir': None},
    'train': {
        'batch_size': 1,
        'num_workers': 0,
        'num_steps': 15000,
        'save_ckpt_interval': 5000,
        'val_interval': 1000,
        'anim_interval': -1,
        'optim_scene': True,
        'save_progress_images': True,
        'progress_save_interval': 1000,
    },
    'human': {
        'name': 'hugs_trimlp',
        'ckpt': None,
        'sh_degree': 0,
        'n_subdivision': 2,
        'only_rgb': False,
        'use_surface': False,
        'use_deformer': True,
        'init_2d': False,
        'disable_posedirs': True,
        'res_offset': False,
        'rotate_sh': False,
        'isotropic': False,
        'init_scale_multiplier': 0.5,
        'run_init': False,
        'estimate_delta': True,
        'triplane_res': 256,
        'optim_pose': True,
        'optim_betas': False,
        'optim_trans': True,
        'optim_eps_offsets': False,
        'activation': 'relu',
        'canon_nframes': 60,
        'canon_pose_type': 'da_pose',
        'knn_n_hops': 3,
        'lr': {
            'wd': 0.0, 'position': 0.00016, 'position_init': 0.00016,
            'position_final': 1.6e-06, 'position_delay_mult': 0.01,
            'position_max_steps': 30000, 'opacity': 0.05, 'scaling': 0.005,
            'rotation': 0.001, 'feature': 0.0025, 'smpl_spatial': 2.0,
            'smpl_pose': 0.0001, 'smpl_betas': 0.0001, 'smpl_trans': 0.0005,
            'smpl_eps_offset': 0.0001, 'lbs_weights': 0.0, 'posedirs': 0.0,
            'percent_dense': 0.01, 'appearance': 0.001, 'geometry': 0.001,
            'vembed': 0.001, 'deformation': 0.0001, 'scale_lr_w_npoints': False,
        },
        'loss': {
            'ssim_w': 0.2, 'l1_w': 0.8, 'lpips_w': 1.0,
            'lbs_w': 1000.0, 'humansep_w': 1.0, 'depth_w': 0.0,
            'num_patches': 4, 'patch_size': 128, 'use_patches': 1,
        },
        'densification_interval': 600,
        'opacity_reset_interval': 3000,
        'densify_from_iter': 3000,
        'densify_until_iter': 15000,
        'densify_grad_threshold': 0.0002,
        'prune_min_opacity': 0.005,
        'densify_extent': 1.0,
        'max_n_gaussians': 524288,
        'percent_dense': 0.01,
    },
    'scene': {
        'name': 'scene_gs',
        'ckpt': None,
        'sh_degree': 3,
        'add_bg_points': False,
        'num_bg_points': 204800,
        'bg_sphere_dist': 5.0,
        'clean_pcd': False,
        'init_pcd_path': None,
        'opt_start_iter': -1,
        'lr': {
            'percent_dense': 0.01, 'spatial_scale': 1.0,
            'position_init': 0.00016, 'position_final': 1.6e-06,
            'position_delay_mult': 0.01, 'position_max_steps': 30000,
            'opacity': 0.05, 'scaling': 0.005, 'rotation': 0.001, 'feature': 0.0025,
        },
        'percent_dense': 0.01,
        'densification_interval': 100,
        'opacity_reset_interval': 20000,
        'densify_from_iter': 500,
        'densify_until_iter': 15000,
        'densify_grad_threshold': 0.0002,
        'prune_min_opacity': 0.005,
        'max_n_gaussians': 2097152,
        'loss': {'ssim_w': 0.2, 'l1_w': 0.8},
    },
    'scene_sugar': {'enabled': False},
    'scene_global_aware_adc': {'enabled': False},
    'anchor_attention': {'use_anchors': False},
}


def write_yaml(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f'  written: {path}')


def main():
    import copy

    our_dir  = 'cfg_files/release/neuman'
    stm_dir  = 'Dynamic-Avatar-Scene-Rendering-from-Human-centric-Context/cfg_files'
    hugs_dir = 'cfg_files/release/neuman'

    for seq in SCENES:
        seq_vimo = f'{seq}_vimo'
        mono_depth = f'data/neuman/dataset/{seq}_vimo/mono_depth'

        print(f'\n=== {seq} ===')

        # ── OUR ──────────────────────────────────────────────────────────────
        if seq == 'lab':
            print(f'  [OUR] lab: 已有 Exp G，跳过')
        else:
            cfg = copy.deepcopy(OUR_TEMPLATE)
            cfg['exp_name'] = f'vimo_inline_attn_18k_{seq}_{DATE}'
            cfg['dataset']['seq'] = seq_vimo
            cfg['dataset']['mono_depth_dir'] = mono_depth
            out = f'{our_dir}/hugs_vimo_{seq}_inline_attn_18k.yaml'
            write_yaml(out, cfg)

        # ── STM ──────────────────────────────────────────────────────────────
        if seq == 'lab':
            print(f'  [STM] lab: 已有 Exp F2，跳过')
        else:
            cfg = copy.deepcopy(STM_TEMPLATE)
            cfg['exp_name'] = f'stm_vimo_{seq}_20k_{DATE}'
            cfg['dataset']['seq'] = seq_vimo
            out = f'{stm_dir}/stm_vimo_{seq}_20k.yaml'
            write_yaml(out, cfg)

        # ── HUGS baseline ────────────────────────────────────────────────────
        if seq == 'lab':
            print(f'  [HUGS] lab: 已有 vimo_baseline (16.7751)，跳过')
        else:
            cfg = copy.deepcopy(HUGS_BASE_TEMPLATE)
            cfg['exp_name'] = f'vimo_hugs_baseline_{seq}_{DATE}'
            cfg['dataset']['seq'] = seq_vimo
            cfg['dataset']['mono_depth_dir'] = mono_depth
            out = f'{hugs_dir}/hugs_vimo_{seq}_baseline_15k.yaml'
            write_yaml(out, cfg)

    print('\n全部配置文件生成完成')


if __name__ == '__main__':
    main()
