#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from omegaconf import OmegaConf

# general configuration
cfg = OmegaConf.create()
cfg.seed = 0
cfg.mode = 'human' # 'human_scene' or 'scene'
cfg.output_path = 'output'
cfg.cfg_file = ''
cfg.exp_name = 'test'
cfg.dataset_path = ''
cfg.detect_anomaly = False
cfg.debug = False
cfg.wandb = False
cfg.logdir = ''
cfg.logdir_ckpt = ''
cfg.eval = False
cfg.bg_color = 'white'

# human dataset configuration
cfg.dataset = OmegaConf.create()
cfg.dataset.name = 'neuman' # 'zju', 'colmap', 'people_snapshot', 'itw'
cfg.dataset.seq = 'citron'

# training configuration
cfg.train = OmegaConf.create()
cfg.train.batch_size = 1
cfg.train.num_workers = 0
cfg.train.num_steps = 30_000
cfg.train.save_ckpt_interval = 4000
cfg.train.val_interval = 2000
cfg.train.anim_interval = 4000
cfg.train.optim_scene = True
cfg.train.save_progress_images = False
cfg.train.progress_save_interval = 10

# human model configuration
cfg.human = OmegaConf.create()
cfg.human.name = 'hugs'
cfg.human.ckpt = None
cfg.human.sh_degree = 3
cfg.human.n_subdivision = 0
cfg.human.only_rgb = False
cfg.human.use_surface = False
cfg.human.use_deformer = False
cfg.human.init_2d = False
cfg.human.disable_posedirs = False

cfg.human.res_offset = False
cfg.human.rotate_sh = False
cfg.human.isotropic = False
cfg.human.init_scale_multiplier = 1.0
cfg.human.run_init = False
cfg.human.skip_init_opt = False
cfg.human.estimate_delta = True
cfg.human.triplane_res = 256

cfg.human.optim_pose = False
cfg.human.optim_betas = False
cfg.human.optim_trans = False
cfg.human.optim_eps_offsets = False
cfg.human.activation = 'relu'

cfg.human.canon_nframes = 60
cfg.human.canon_pose_type = 'da_pose'
cfg.human.knn_n_hops = 3

# human model learning rate configuration
cfg.human.lr = OmegaConf.create()
cfg.human.lr.wd = 0.0
cfg.human.lr.position = 0.00016
cfg.human.lr.position_init = 0.00016
cfg.human.lr.position_final = 0.0000016
cfg.human.lr.position_delay_mult = 0.01
cfg.human.lr.position_max_steps = 30_000
cfg.human.lr.opacity = 0.05
cfg.human.lr.scaling = 0.005
cfg.human.lr.rotation = 0.001
cfg.human.lr.feature = 0.0025
cfg.human.lr.smpl_spatial = 2.0
cfg.human.lr.smpl_pose = 0.0001
cfg.human.lr.smpl_betas = 0.0001
cfg.human.lr.smpl_trans = 0.0001
cfg.human.lr.smpl_eps_offset = 0.0001
cfg.human.lr.lbs_weights = 0.0
cfg.human.lr.posedirs = 0.0
cfg.human.lr.percent_dense = 0.01

cfg.human.lr.appearance = 1e-3
cfg.human.lr.geometry = 1e-3
cfg.human.lr.vembed = 1e-3
cfg.human.lr.deformation = 1e-4
# scale
cfg.human.lr.scale_lr_w_npoints = False

# human model loss coefficients
cfg.human.loss = OmegaConf.create()
cfg.human.loss.ssim_w = 0.2
cfg.human.loss.l1_w = 0.8
cfg.human.loss.lpips_w = 1.0
cfg.human.loss.lbs_w = 0.0
cfg.human.loss.humansep_w = 0.0
cfg.human.loss.num_patches = 4
cfg.human.loss.patch_size = 128
cfg.human.loss.use_patches = 1

# human model densification configuration
cfg.human.densification_interval = 100
cfg.human.opacity_reset_interval = 3000
cfg.human.densify_from_iter = 500
cfg.human.densify_until_iter = 15_000
cfg.human.densify_grad_threshold = 0.0002
cfg.human.prune_min_opacity = 0.005
cfg.human.densify_extent = 2.0
cfg.human.max_n_gaussians = 2e5

# scene model configuration
cfg.scene = OmegaConf.create()
cfg.scene.name = 'scene_gs'
cfg.scene.ckpt = None
cfg.scene.sh_degree = 3
cfg.scene.add_bg_points = False
cfg.scene.num_bg_points = 204_800
cfg.scene.bg_sphere_dist = 5.0
cfg.scene.clean_pcd = False
cfg.scene.opt_start_iter = -1
cfg.scene.lr = OmegaConf.create()
cfg.scene.lr.percent_dense = 0.01
cfg.scene.lr.spatial_scale = 1.0
cfg.scene.lr.position_init = 0.00016
cfg.scene.lr.position_final = 0.0000016
cfg.scene.lr.position_delay_mult = 0.01
cfg.scene.lr.position_max_steps = 30_000
cfg.scene.lr.opacity = 0.05
cfg.scene.lr.scaling = 0.005
cfg.scene.lr.rotation = 0.001
cfg.scene.lr.feature = 0.0025

# scene model densification configuration
cfg.scene.percent_dense = 0.01
cfg.scene.densification_interval = 100
cfg.scene.opacity_reset_interval = 3000
cfg.scene.densify_from_iter = 500
cfg.scene.densify_until_iter = 15_000
cfg.scene.densify_grad_threshold = 0.0002
cfg.scene.prune_min_opacity = 0.005
cfg.scene.max_n_gaussians = 2e6

# scene geometry regularization for contact-aware experiments
cfg.scene.anchor_pcd_path = None
cfg.scene.anchor_loss_w = 0.0
cfg.scene.anchor_loss_sample_scene = 2048
cfg.scene.anchor_loss_sample_prior = 8192
cfg.scene.anchor_loss_trunc = 0.5
cfg.scene.human_exclusion_w = 0.0
cfg.scene.human_exclusion_radius = 0.12
cfg.scene.human_exclusion_sample_scene = 2048
cfg.scene.human_exclusion_sample_human = 8192
cfg.scene.depth_prior_dir = None
cfg.scene.depth_prior_w = 0.0
cfg.scene.depth_prior_sample_scene = 4096
cfg.scene.depth_prior_tolerance = 0.25
cfg.scene.depth_prior_max_residual = 2.0
cfg.scene.depth_prior_mask_human = True
cfg.scene.depth_prune_interval = 0
cfg.scene.depth_prune_from_iter = 0
cfg.scene.depth_prune_until_iter = 1_000_000
cfg.scene.depth_prune_abs_threshold = 2.0
cfg.scene.depth_prune_abs_hard_threshold = 8.0
cfg.scene.depth_prune_opacity_threshold = 0.03
cfg.scene.depth_prune_max_frac = 0.05
cfg.scene.depth_prune_mask_human = True

# scene model loss coefficients
cfg.scene.loss = OmegaConf.create()
cfg.scene.loss.ssim_w = 0.2
cfg.scene.loss.l1_w = 0.8

# scene-only SuGaR-style background regularization
cfg.scene_sugar = OmegaConf.create()
cfg.scene_sugar.enabled = False
cfg.scene_sugar.start_iter = 7000
cfg.scene_sugar.opacity_entropy_start_iter = 7000
cfg.scene_sugar.surface_reg_start_iter = 9000
cfg.scene_sugar.prune_iter = 9000
cfg.scene_sugar.prune_interval = 3000
cfg.scene_sugar.stop_after_iter = -1
cfg.scene_sugar.lambda_opacity_entropy = 1e-4
cfg.scene_sugar.opacity_prune_threshold = 0.03
cfg.scene_sugar.lambda_scale_volume = 1e-5
cfg.scene_sugar.lambda_flatten = 1e-4
cfg.scene_sugar.lambda_surface_alignment = 0.0
cfg.scene_sugar.max_prune_frac = 0.25
cfg.scene_sugar.min_remaining = 1000
cfg.scene_sugar.eps = 1e-6
cfg.scene_sugar.log_interval = 1000
cfg.scene_sugar.ply_interval = 3000
cfg.scene_sugar.debug_dir = ''
cfg.scene_sugar.apply_to_scene_only = True
cfg.scene_sugar.detach_human_branch = True
cfg.scene_sugar.disable_mesh_extraction = True

# scene-only Global-aware Voxelized ADC background densification
cfg.scene_global_aware_adc = OmegaConf.create()
cfg.scene_global_aware_adc.enabled = False
cfg.scene_global_aware_adc.apply_to_scene_only = True
cfg.scene_global_aware_adc.scene_mode = 'global_aware_voxel_adc_scene'
cfg.scene_global_aware_adc.disable_for_human = True
cfg.scene_global_aware_adc.disable_sugar_losses = True
cfg.scene_global_aware_adc.start_iter = 500
cfg.scene_global_aware_adc.densify_until_iter = 15_000
cfg.scene_global_aware_adc.densification_interval = 100
cfg.scene_global_aware_adc.reset_opacity_interval = 3000
cfg.scene_global_aware_adc.tau_pos = 0.0002
cfg.scene_global_aware_adc.threshold_policy = 'fixed'
cfg.scene_global_aware_adc.relative_blur_backend = 'gaussian_highpass_approx'
cfg.scene_global_aware_adc.blur_window_size = 11
cfg.scene_global_aware_adc.highpass_cutoff_ratio = 0.5
cfg.scene_global_aware_adc.blur_percentile = 95.0
cfg.scene_global_aware_adc.weight_lower_bound = 0.5
cfg.scene_global_aware_adc.weight_upper_bound = 1.0
cfg.scene_global_aware_adc.use_grayscale = True
cfg.scene_global_aware_adc.use_max_viewspace_gradient = True
cfg.scene_global_aware_adc.gradient_accumulation = 'max'
cfg.scene_global_aware_adc.accumulate_interval = 1
cfg.scene_global_aware_adc.voxel_size = 0.005
cfg.scene_global_aware_adc.one_densify_per_voxel = True
cfg.scene_global_aware_adc.voxel_score = 'mean'
cfg.scene_global_aware_adc.selection_inside_voxel = 'importance_sampling'
cfg.scene_global_aware_adc.clone_scale_threshold = 'same_as_hugs'
cfg.scene_global_aware_adc.split_scale_threshold = 'same_as_hugs'
cfg.scene_global_aware_adc.child_opacity_policy = 'half_parent_opacity'
cfg.scene_global_aware_adc.split_children = 2
cfg.scene_global_aware_adc.min_scene_gaussians = 1000
cfg.scene_global_aware_adc.max_scene_gaussians = -1
cfg.scene_global_aware_adc.max_densify_per_call = -1
cfg.scene_global_aware_adc.skip_empty_voxels = True
cfg.scene_global_aware_adc.preserve_optimizer_state = True
cfg.scene_global_aware_adc.log_blur_stats = True
cfg.scene_global_aware_adc.log_voxel_stats = True
cfg.scene_global_aware_adc.save_debug_maps = False
cfg.scene_global_aware_adc.debug_map_interval = 1000
cfg.scene_global_aware_adc.eps = 1e-6

# anchor-attention baseline configuration
cfg.anchor_attention = OmegaConf.create()
cfg.anchor_attention.use_anchors = False
cfg.anchor_attention.anchor_vertices_path = ''
cfg.anchor_attention.count_per_anchor = 64
cfg.anchor_attention.top_m = 2
cfg.anchor_attention.lambda_lbs = 0.35
cfg.anchor_attention.sigma_scale = 1.8
cfg.anchor_attention.min_sigma = 0.035
cfg.anchor_attention.use_anchor_token_encoder = False
cfg.anchor_attention.use_scene_query = False
cfg.anchor_attention.use_cross_attention = False
cfg.anchor_attention.use_interaction_correction = False
cfg.anchor_attention.correct_xyz = True
cfg.anchor_attention.correct_transl = False
cfg.anchor_attention.correct_opacity = False
cfg.anchor_attention.correct_scale = False
cfg.anchor_attention.correct_feature_dc = False
cfg.anchor_attention.gamma_transl = 0.005
cfg.anchor_attention.transl_delta_clamp = 0.05
cfg.anchor_attention.gamma_scale = 0.01
cfg.anchor_attention.gamma_feature_dc = 0.01
cfg.anchor_attention.opacity_delta_clamp = 0.0
cfg.anchor_attention.scale_delta_clamp = 0.05
cfg.anchor_attention.feature_delta_clamp = 0.05
cfg.anchor_attention.scale_max = 1.0
cfg.anchor_attention.hidden_dim = 128
cfg.anchor_attention.human_pooling = 'attention' # 'mean' or 'attention'
cfg.anchor_attention.scene_topk = 32
cfg.anchor_attention.scene_opacity_threshold = 0.01
cfg.anchor_attention.max_scene_candidates = 200000
cfg.anchor_attention.module_start_iter = 3000
cfg.anchor_attention.correction_start_iter = 3000
cfg.anchor_attention.correction_warmup_iters = 0
cfg.anchor_attention.zero_init_delta = True
cfg.anchor_attention.gamma_mu = 0.02
cfg.anchor_attention.gamma_opacity = 0.05
cfg.anchor_attention.delta_loss_w = 0.001
cfg.anchor_attention.lr = 0.0001
cfg.anchor_attention.debug_interval = 1000
cfg.anchor_attention.debug_dir = ''
cfg.anchor_attention.ckpt = ''
