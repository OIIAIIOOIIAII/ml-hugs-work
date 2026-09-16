# Code adapted from 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/gaussian_renderer/__init__.py
# License from 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import math
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings, 
    GaussianRasterizer
)

from hugs.utils.spherical_harmonics import SH2RGB
from hugs.utils.rotations import quaternion_to_matrix


def render_human_scene(
    data,
    human_gs_out,
    scene_gs_out,
    bg_color,
    human_bg_color=None,
    scaling_modifier=1.0,
    render_mode='human_scene',
    render_human_separate=False,
    fused_gs_out=None,
):

    feats = None
    if render_mode == 'human_scene':
        if fused_gs_out is not None:
            # STM-style: fused_gs_out contains human+scene concatenated (N_human+N_scene points).
            # Use fused output for all GS, matching STM's HumanSceneFuseDecoder design exactly.
            feats = fused_gs_out['shs'].reshape(-1, 16, 3)
            means3D = fused_gs_out['xyz']
            opacity = fused_gs_out['opacity']
            scales = fused_gs_out['scales']
            rotations = fused_gs_out['rotq']
        else:
            feats = torch.cat([human_gs_out['shs'], scene_gs_out['shs']], dim=0)
            means3D = torch.cat([human_gs_out['xyz'], scene_gs_out['xyz']], dim=0)
            opacity = torch.cat([human_gs_out['opacity'], scene_gs_out['opacity']], dim=0)
            scales = torch.cat([human_gs_out['scales'], scene_gs_out['scales']], dim=0)
            rotations = torch.cat([human_gs_out['rotq'], scene_gs_out['rotq']], dim=0)
        active_sh_degree = human_gs_out['active_sh_degree']
    elif render_mode == 'human':
        feats = human_gs_out['shs']
        means3D = human_gs_out['xyz']
        opacity = human_gs_out['opacity']
        scales = human_gs_out['scales']
        rotations = human_gs_out['rotq']
        active_sh_degree = human_gs_out['active_sh_degree']
    elif render_mode == 'scene':
        feats = scene_gs_out['shs']
        means3D = scene_gs_out['xyz']
        opacity = scene_gs_out['opacity']
        scales = scene_gs_out['scales']
        rotations = scene_gs_out['rotq']
        active_sh_degree = scene_gs_out['active_sh_degree']
    else:
        raise ValueError(f'Unknown render mode: {render_mode}')
    
    render_pkg = render(
        means3D=means3D,
        feats=feats,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
        data=data,
        scaling_modifier=scaling_modifier,
        bg_color=bg_color,
        active_sh_degree=active_sh_degree,
    )
        
    if render_human_separate and render_mode == 'human_scene':
        render_human_pkg = render(
            means3D=human_gs_out['xyz'],
            feats=human_gs_out['shs'],
            opacity=human_gs_out['opacity'],
            scales=human_gs_out['scales'],
            rotations=human_gs_out['rotq'],
            data=data,
            scaling_modifier=scaling_modifier,
            bg_color=human_bg_color if human_bg_color is not None else bg_color,
            active_sh_degree=human_gs_out['active_sh_degree'],
        )
        render_pkg['human_img'] = render_human_pkg['render']
        render_pkg['human_visibility_filter'] = render_human_pkg['visibility_filter']
        render_pkg['human_radii'] = render_human_pkg['radii']
        
    if render_mode == 'human':
        render_pkg['human_visibility_filter'] = render_pkg['visibility_filter']
        render_pkg['human_radii'] = render_pkg['radii']
    elif render_mode == 'human_scene':
        human_n_gs = human_gs_out['xyz'].shape[0]
        scene_n_gs = scene_gs_out['xyz'].shape[0]
        render_pkg['scene_visibility_filter'] = render_pkg['visibility_filter'][human_n_gs:]
        render_pkg['scene_radii'] = render_pkg['radii'][human_n_gs:]
        if not 'human_visibility_filter' in render_pkg.keys():
            render_pkg['human_visibility_filter'] = render_pkg['visibility_filter'][:-scene_n_gs]
            render_pkg['human_radii'] = render_pkg['radii'][:-scene_n_gs]
            
    elif render_mode == 'scene':
        render_pkg['scene_visibility_filter'] = render_pkg['visibility_filter']
        render_pkg['scene_radii'] = render_pkg['radii']
        
    return render_pkg
    
    
def render_depth_map(means3D, opacity, scales, rotations, data):
    """Render depth by alpha-compositing camera-space z values as precomputed colors.

    Returns a (1, H, W) tensor of rendered depth (camera-space z).
    Gradients flow back to means3D through both the projection (alpha weights)
    and the precomputed depth values used as color channels.
    """
    # Camera-space z: p_cam = p_world @ world_view_transform (row-vector convention)
    # world_view_transform = W2C.T, so column j of result = row j of world_view_transform
    wvt = data['world_view_transform']
    depth_vals = means3D @ wvt[:3, 2] + wvt[3, 2]  # (N,)
    depth_3ch = depth_vals.unsqueeze(-1).expand(-1, 3).contiguous()  # (N, 3)

    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype,
                                          requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    bg_color_zero = torch.zeros(3, dtype=torch.float32, device="cuda")
    tanfovx = math.tan(data['fovx'] * 0.5)
    tanfovy = math.tan(data['fovy'] * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(data['image_height']),
        image_width=int(data['image_width']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color_zero,
        scale_modifier=1.0,
        viewmatrix=data['world_view_transform'],
        projmatrix=data['full_proj_transform'],
        sh_degree=0,
        campos=data['camera_center'],
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    raster_out = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=depth_3ch,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
    )
    if isinstance(raster_out, tuple):
        rendered_depth = raster_out[0]
    else:
        rendered_depth = raster_out
    return rendered_depth[0:1]  # (1, H, W)


def render(means3D, feats, opacity, scales, rotations, data, scaling_modifier=1.0, bg_color=None, active_sh_degree=0):
    if bg_color is None:
        bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
        
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception as e:
        pass

    means2D = screenspace_points

    # Set up rasterization configuration
    tanfovx = math.tan(data['fovx'] * 0.5)
    tanfovy = math.tan(data['fovy'] * 0.5)

    shs, rgb = None, None
    if len(feats.shape) == 2:
        rgb = feats
    else:
        shs = feats

    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(data['image_height']),
        image_width=int(data['image_width']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=data['world_view_transform'],
        projmatrix=data['full_proj_transform'],
        sh_degree=active_sh_degree,
        campos=data['camera_center'],
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    raster_out = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        colors_precomp=rgb,
    )
    if isinstance(raster_out, tuple):
        rendered_image = None
        radii = None
        for out in raster_out:
            if not torch.is_tensor(out):
                continue
            if rendered_image is None and out.dim() == 3 and out.shape[0] in (3, 4):
                rendered_image = out
            if radii is None and out.dim() == 1 and out.shape[0] == means3D.shape[0]:
                radii = out
        if rendered_image is None:
            rendered_image = raster_out[0]
        if radii is None:
            radii = torch.ones(means3D.shape[0], dtype=torch.bool, device=means3D.device)
    else:
        rendered_image = raster_out
        radii = torch.ones(means3D.shape[0], dtype=torch.bool, device=means3D.device)
    rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
    }
