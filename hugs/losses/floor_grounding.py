"""Floor grounding loss: pull SMPL foot joints toward the scene floor plane."""

import torch

# SMPL joint indices for feet
FOOT_JOINT_INDICES = [7, 8, 10, 11]  # L_Ankle, R_Ankle, L_Foot, R_Foot


def estimate_floor_from_feet(human_gs, smpl_scales, up_axis=1, bottom_frac=0.05):
    """Estimate floor plane from SMPL foot positions across all training frames.

    This is more reliable than scene-GS-based estimation because it directly
    uses where the feet touch the ground, avoiding scene GS from walls/ceiling.

    Args:
        human_gs: HUGS_TRIMLP with transl, global_orient, body_pose, betas, smpl
        smpl_scales: (N,) tensor of per-frame SMPL scale values (world = smpl * scale + transl)
        up_axis: world up axis index (1=Y)
        bottom_frac: fraction of lowest foot positions used to fit the floor plane

    Returns:
        (normal, centroid): detached floor plane params, or (None, None) on failure
    """
    from hugs.utils.rotations import rotation_6d_to_axis_angle

    n_frames = human_gs.transl.shape[0]
    if n_frames == 0:
        return None, None

    device = human_gs.transl.device
    scales = torch.as_tensor(smpl_scales, dtype=torch.float32, device=device)

    with torch.no_grad():
        # Batch 6D-rotation → axis-angle for all frames
        go_aa = rotation_6d_to_axis_angle(
            human_gs.global_orient.detach().reshape(n_frames, 1, 6)
        ).reshape(n_frames, 3)

        bp_aa = rotation_6d_to_axis_angle(
            human_gs.body_pose.detach().reshape(n_frames * 23, 1, 6)
        ).reshape(n_frames, 23 * 3)

        betas_batch = human_gs.betas.detach().unsqueeze(0).expand(n_frames, -1)

        smpl_out = human_gs.smpl(
            betas=betas_batch,
            body_pose=bp_aa,
            global_orient=go_aa,
            disable_posedirs=False,
            return_full_pose=True,
        )
        foot_smpl = smpl_out.joints[:, FOOT_JOINT_INDICES]  # (N, 4, 3)
        transl = human_gs.transl.detach()  # (N, 3)
        # foot_world: broadcast scale over 4 joints
        foot_world = foot_smpl * scales.reshape(n_frames, 1, 1) + transl.unsqueeze(1)  # (N, 4, 3)
        foot_world = foot_world.reshape(-1, 3)  # (N*4, 3)

    n_bottom = max(4, int(foot_world.shape[0] * bottom_frac))
    _, idx = torch.topk(foot_world[:, up_axis], n_bottom, largest=False)
    floor_pts = foot_world[idx]

    centroid = floor_pts.mean(0)
    try:
        _, _, Vt = torch.linalg.svd(floor_pts - centroid, full_matrices=False)
        normal = Vt[-1]
        if normal[up_axis] < 0:
            normal = -normal
    except Exception:
        normal = torch.zeros(3, device=device)
        normal[up_axis] = 1.0

    return normal.detach(), centroid.detach()


def estimate_floor_plane(scene_xyz, scene_opacity=None, opacity_thresh=0.1, bottom_frac=0.05, up_axis=1):
    """Estimate floor plane from scene GS using SVD on the lowest points.

    Args:
        scene_xyz: (N, 3) scene GS world positions
        scene_opacity: (N,) or (N, 1) opacity values, optional
        opacity_thresh: minimum opacity to include a point
        bottom_frac: fraction of lowest points to use for plane fitting
        up_axis: 0=X, 1=Y, 2=Z is the approximate world-up direction

    Returns:
        (normal, centroid): unit normal pointing up-ward and a point on the plane,
                            both detached (no grad). Returns (None, None) on failure.
    """
    pts = scene_xyz.float().detach()

    if scene_opacity is not None:
        op = scene_opacity.float().detach().squeeze(-1)
        pts = pts[op > opacity_thresh]

    if pts.shape[0] < 50:
        return None, None

    n_bottom = max(50, int(pts.shape[0] * bottom_frac))
    up_vals = pts[:, up_axis]
    _, idx = torch.topk(up_vals, n_bottom, largest=False)
    floor_pts = pts[idx]

    centroid = floor_pts.mean(0)
    centered = floor_pts - centroid

    try:
        _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
        normal = Vt[-1]  # smallest singular value direction = plane normal
    except Exception:
        normal = torch.zeros(3, device=scene_xyz.device)
        normal[up_axis] = 1.0

    if normal[up_axis] < 0:
        normal = -normal

    return normal.detach(), centroid.detach()


def get_foot_joints_world(human_gs, dataset_idx, smpl_scale, delta_transl=None):
    """Compute foot joint world positions, with grad flowing through transl.

    Args:
        human_gs: HUGS_TRIMLP model
        dataset_idx: current frame index
        smpl_scale: scalar scale factor (world = smpl * scale + transl)
        delta_transl: (3,) anchor-attention transl correction, optional

    Returns:
        foot_joints_world: (4, 3) world-space foot positions (leaf=transl has grad)
    """
    from hugs.utils.rotations import rotation_6d_to_axis_angle

    with torch.no_grad():
        global_orient_aa = rotation_6d_to_axis_angle(
            human_gs.global_orient[dataset_idx].detach().reshape(-1, 6)
        ).reshape(3)
        body_pose_aa = rotation_6d_to_axis_angle(
            human_gs.body_pose[dataset_idx].detach().reshape(-1, 6)
        ).reshape(23 * 3)
        betas = human_gs.betas.detach()

        smpl_out = human_gs.smpl(
            betas=betas.unsqueeze(0),
            body_pose=body_pose_aa.unsqueeze(0),
            global_orient=global_orient_aa.unsqueeze(0),
            disable_posedirs=False,
            return_full_pose=True,
        )
        foot_smpl = smpl_out.joints[0, FOOT_JOINT_INDICES].detach()  # (4, 3)

    transl = human_gs.transl[dataset_idx]  # (3,) learnable parameter, grad flows
    effective_transl = transl + delta_transl if delta_transl is not None else transl
    foot_world = foot_smpl * smpl_scale + effective_transl  # (4, 3)
    return foot_world


def floor_grounding_loss(foot_joints_world, floor_normal, floor_point, contact_zone=0.10, margin=0.005):
    """Compute floor grounding loss.

    For feet within `contact_zone` of the floor, pull them toward `margin` above floor.
    Always penalize penetration (signed_dist < margin).

    Args:
        foot_joints_world: (4, 3) foot positions in world space (grad via transl)
        floor_normal: (3,) unit normal pointing upward (no grad)
        floor_point: (3,) a point on the floor plane (no grad)
        contact_zone: height above floor (m) within which contact force activates
        margin: target resting height above floor (m)

    Returns:
        loss: scalar tensor with grad
        stats: dict of detached diagnostic scalars
    """
    # Signed distance: positive = above floor, negative = below
    signed_dist = ((foot_joints_world - floor_point) * floor_normal).sum(-1)  # (4,)

    # Contact pull: feet in [margin, contact_zone] get pulled toward margin
    in_contact = (signed_dist < contact_zone)
    contact_err = torch.clamp(signed_dist - margin, min=0.0)  # only pull downward
    contact_loss = (in_contact.float() * contact_err).pow(2).mean()

    # Penetration penalty: feet below margin
    penetration_loss = torch.relu(margin - signed_dist).pow(2).mean()

    total = contact_loss + penetration_loss
    stats = {
        'floor_contact': contact_loss.detach(),
        'floor_penetration': penetration_loss.detach(),
        'foot_dist_mean': signed_dist.mean().detach(),
        'foot_dist_min': signed_dist.min().detach(),
    }
    return total, stats
