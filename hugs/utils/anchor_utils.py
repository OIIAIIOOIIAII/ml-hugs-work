#
# Utilities for semantic anchor debug experiments.
#

from pathlib import Path

import numpy as np
import torch
import trimesh
from smplx.body_models import SMPL


DEFAULT_ANCHOR_NAMES = [
    "left_sole",
    "right_sole",
    "left_toe",
    "right_toe",
    "left_heel",
    "right_heel",
    "left_palm",
    "right_palm",
    "left_fingers",
    "right_fingers",
    "buttocks",
    "back",
    "left_knee",
    "right_knee",
    "left_elbow",
    "right_elbow",
]


SMPL_JOINTS = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_knee": 4,
    "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "spine3": 9,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "left_collar": 13,
    "right_collar": 14,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
    "left_hand": 22,
    "right_hand": 23,
}


def resolve_smpl_model_file(model_path):
    model_path = Path(model_path)
    if model_path.is_file():
        return model_path
    candidate = model_path / "SMPL_NEUTRAL.pkl"
    if candidate.exists():
        return candidate
    candidate = model_path / "smpl" / "SMPL_NEUTRAL.pkl"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find SMPL_NEUTRAL.pkl under {model_path}")


def _make_body_pose(canonical_pose):
    body_pose = torch.zeros(1, 69, dtype=torch.float32)
    if canonical_pose in ("t_pose", "template"):
        return body_pose
    if canonical_pose in ("hugs_vitruvian", "vitruvian"):
        body_pose[:, 2] = 1.0
        body_pose[:, 5] = -1.0
        return body_pose
    raise ValueError(f"Unknown canonical_pose: {canonical_pose}")


def load_smpl_template(model_path="data/smpl", canonical_pose="t_pose"):
    smpl_file = resolve_smpl_model_file(model_path)
    smpl = SMPL(str(smpl_file), gender="neutral", batch_size=1)
    if canonical_pose in ("t_pose", "template"):
        vertices_t = smpl.v_template.detach().cpu()
    else:
        body_pose = _make_body_pose(canonical_pose)
        with torch.no_grad():
            out = smpl(
                betas=torch.zeros(1, 10, dtype=torch.float32),
                global_orient=torch.zeros(1, 3, dtype=torch.float32),
                body_pose=body_pose,
                transl=torch.zeros(1, 3, dtype=torch.float32),
            )
        vertices_t = out.vertices[0].detach().cpu()

    vertices = vertices_t.numpy().astype(np.float32)
    faces = np.asarray(smpl.faces, dtype=np.int32)
    lbs_weights = smpl.lbs_weights.detach().cpu().numpy().astype(np.float32)
    joints = (smpl.J_regressor.detach().cpu().numpy().astype(np.float32) @ vertices)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    normals = normalize_np(normals)

    return {
        "model_file": str(smpl_file),
        "canonical_pose": canonical_pose,
        "vertices": vertices,
        "faces": faces,
        "normals": normals,
        "lbs_weights": lbs_weights,
        "joints": joints.astype(np.float32),
    }


def make_template_from_vertices(vertices, faces, lbs_weights, canonical_pose="hugs_vitruvian", joints=None):
    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy()
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()
    if torch.is_tensor(lbs_weights):
        lbs_weights = lbs_weights.detach().cpu().numpy()

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    lbs_weights = np.asarray(lbs_weights, dtype=np.float32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    normals = normalize_np(np.asarray(mesh.vertex_normals, dtype=np.float32))
    if joints is None:
        joints = np.zeros((24, 3), dtype=np.float32)
    elif torch.is_tensor(joints):
        joints = joints.detach().cpu().numpy()

    return {
        "model_file": "hugs_model",
        "canonical_pose": canonical_pose,
        "vertices": vertices,
        "faces": faces,
        "normals": normals,
        "lbs_weights": lbs_weights,
        "joints": np.asarray(joints, dtype=np.float32),
    }


def normalize_np(x, eps=1e-8):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def _side_mask(vertices, side):
    if side == "left":
        return vertices[:, 0] > 0.0
    if side == "right":
        return vertices[:, 0] < 0.0
    return np.ones(vertices.shape[0], dtype=bool)


def _lbs_region(vertices, lbs, joint_ids, side=None, min_score=0.18, percentile=72.0):
    score = lbs[:, joint_ids].sum(axis=1)
    threshold = max(float(np.percentile(score, percentile)), min_score)
    mask = score >= threshold
    mask &= _side_mask(vertices, side)
    return mask, score


def _select_extreme(vertices, mask, axis, largest, count, prefer_normal=None, normals=None):
    candidates = np.where(mask)[0]
    if candidates.size == 0:
        raise ValueError("Cannot select anchor vertices from an empty mask")

    values = vertices[candidates, axis]
    order = np.argsort(values)
    if largest:
        order = order[::-1]
    candidates = candidates[order]

    if prefer_normal is not None and normals is not None:
        normal_values = normals[candidates] @ np.asarray(prefer_normal, dtype=np.float32)
        keep = normal_values > 0.05
        if keep.sum() >= max(8, count // 4):
            candidates = candidates[keep]

    return candidates[: min(count, candidates.size)].astype(int).tolist()


def _select_nearest(vertices, mask, center, count):
    candidates = np.where(mask)[0]
    if candidates.size == 0:
        candidates = np.arange(vertices.shape[0])
    dists = np.linalg.norm(vertices[candidates] - center[None], axis=1)
    order = np.argsort(dists)
    return candidates[order[: min(count, candidates.size)]].astype(int).tolist()


def _select_quantile_box(vertices, mask, count, z_low=True, y_range=None, x_abs_max=None, normals=None):
    work = mask.copy()
    if y_range is not None:
        work &= (vertices[:, 1] >= y_range[0]) & (vertices[:, 1] <= y_range[1])
    if x_abs_max is not None:
        work &= np.abs(vertices[:, 0]) <= x_abs_max
    candidates = np.where(work)[0]
    if candidates.size == 0:
        candidates = np.where(mask)[0]
    values = vertices[candidates, 2]
    order = np.argsort(values)
    if not z_low:
        order = order[::-1]
    candidates = candidates[order]
    if normals is not None and z_low:
        keep = normals[candidates, 2] < 0.25
        if keep.sum() >= max(8, count // 4):
            candidates = candidates[keep]
    return candidates[: min(count, candidates.size)].astype(int).tolist()


def generate_semantic_anchor_vertices(template, count_per_anchor=64):
    vertices = template["vertices"]
    normals = template["normals"]
    lbs = template["lbs_weights"]
    joints = template["joints"]

    anchors = {}

    left_foot, _ = _lbs_region(
        vertices,
        lbs,
        [SMPL_JOINTS["left_ankle"], SMPL_JOINTS["left_foot"]],
        side="left",
        min_score=0.20,
        percentile=67.0,
    )
    right_foot, _ = _lbs_region(
        vertices,
        lbs,
        [SMPL_JOINTS["right_ankle"], SMPL_JOINTS["right_foot"]],
        side="right",
        min_score=0.20,
        percentile=67.0,
    )
    low_left_foot = left_foot & (vertices[:, 1] <= np.percentile(vertices[left_foot, 1], 38))
    low_right_foot = right_foot & (vertices[:, 1] <= np.percentile(vertices[right_foot, 1], 38))

    anchors["left_sole"] = _select_extreme(
        vertices, low_left_foot, axis=1, largest=False, count=count_per_anchor, prefer_normal=(0, -1, 0), normals=normals
    )
    anchors["right_sole"] = _select_extreme(
        vertices, low_right_foot, axis=1, largest=False, count=count_per_anchor, prefer_normal=(0, -1, 0), normals=normals
    )
    anchors["left_toe"] = _select_extreme(vertices, left_foot, axis=2, largest=True, count=count_per_anchor)
    anchors["right_toe"] = _select_extreme(vertices, right_foot, axis=2, largest=True, count=count_per_anchor)
    anchors["left_heel"] = _select_extreme(vertices, left_foot, axis=2, largest=False, count=count_per_anchor)
    anchors["right_heel"] = _select_extreme(vertices, right_foot, axis=2, largest=False, count=count_per_anchor)

    left_hand, _ = _lbs_region(
        vertices,
        lbs,
        [SMPL_JOINTS["left_wrist"], SMPL_JOINTS["left_hand"]],
        side="left",
        min_score=0.12,
        percentile=78.0,
    )
    right_hand, _ = _lbs_region(
        vertices,
        lbs,
        [SMPL_JOINTS["right_wrist"], SMPL_JOINTS["right_hand"]],
        side="right",
        min_score=0.12,
        percentile=78.0,
    )
    anchors["left_fingers"] = _select_extreme(vertices, left_hand, axis=0, largest=True, count=count_per_anchor)
    anchors["right_fingers"] = _select_extreme(vertices, right_hand, axis=0, largest=False, count=count_per_anchor)
    anchors["left_palm"] = _select_nearest(vertices, left_hand, joints[SMPL_JOINTS["left_hand"]], count_per_anchor)
    anchors["right_palm"] = _select_nearest(vertices, right_hand, joints[SMPL_JOINTS["right_hand"]], count_per_anchor)

    torso, _ = _lbs_region(
        vertices,
        lbs,
        [
            SMPL_JOINTS["pelvis"],
            SMPL_JOINTS["spine1"],
            SMPL_JOINTS["spine2"],
            SMPL_JOINTS["spine3"],
            SMPL_JOINTS["neck"],
        ],
        min_score=0.20,
        percentile=55.0,
    )
    anchors["back"] = _select_quantile_box(
        vertices, torso, count_per_anchor, z_low=True, y_range=(-0.12, 0.34), x_abs_max=0.32, normals=normals
    )
    anchors["buttocks"] = _select_quantile_box(
        vertices, torso, count_per_anchor, z_low=True, y_range=(-0.54, -0.20), x_abs_max=0.48, normals=normals
    )

    left_leg = _side_mask(vertices, "left")
    right_leg = _side_mask(vertices, "right")
    anchors["left_knee"] = _select_nearest(vertices, left_leg, joints[SMPL_JOINTS["left_knee"]], count_per_anchor)
    anchors["right_knee"] = _select_nearest(vertices, right_leg, joints[SMPL_JOINTS["right_knee"]], count_per_anchor)
    anchors["left_elbow"] = _select_nearest(vertices, _side_mask(vertices, "left"), joints[SMPL_JOINTS["left_elbow"]], count_per_anchor)
    anchors["right_elbow"] = _select_nearest(vertices, _side_mask(vertices, "right"), joints[SMPL_JOINTS["right_elbow"]], count_per_anchor)

    return {name: anchors[name] for name in DEFAULT_ANCHOR_NAMES}


def compute_anchors(template, anchor_vertices):
    vertices = template["vertices"]
    normals = template["normals"]
    lbs = template["lbs_weights"]

    names = list(anchor_vertices.keys())
    pos = []
    normal = []
    anchor_lbs = []
    radii = []
    vertex_ids_out = []

    for name in names:
        ids = np.asarray(anchor_vertices[name], dtype=np.int64)
        if ids.size == 0:
            raise ValueError(f"Anchor {name} has no vertices")
        if ids.min() < 0 or ids.max() >= vertices.shape[0]:
            raise ValueError(f"Anchor {name} has vertex ids outside [0, {vertices.shape[0]})")
        pts = vertices[ids]
        pos_a = pts.mean(axis=0)
        normal_a = normalize_np(normals[ids].mean(axis=0, keepdims=True))[0]
        lbs_a = lbs[ids].mean(axis=0)
        lbs_a = lbs_a / max(float(lbs_a.sum()), 1e-8)
        radius_a = np.linalg.norm(pts - pos_a[None], axis=1).max()

        pos.append(pos_a)
        normal.append(normal_a)
        anchor_lbs.append(lbs_a)
        radii.append(radius_a)
        vertex_ids_out.append([int(v) for v in ids.tolist()])

    return {
        "names": names,
        "pos_canon": torch.from_numpy(np.asarray(pos, dtype=np.float32)),
        "normal_canon": torch.from_numpy(np.asarray(normal, dtype=np.float32)),
        "lbs": torch.from_numpy(np.asarray(anchor_lbs, dtype=np.float32)),
        "radius": torch.from_numpy(np.asarray(radii, dtype=np.float32)),
        "part_id": torch.arange(len(names), dtype=torch.long),
        "vertex_ids": vertex_ids_out,
    }


def summarize_anchor_quality(anchors):
    rows = []
    pos = anchors["pos_canon"].detach().cpu().numpy()
    normals = anchors["normal_canon"].detach().cpu().numpy()
    radii = anchors["radius"].detach().cpu().numpy()
    for idx, name in enumerate(anchors["names"]):
        rows.append(
            {
                "name": name,
                "num_vertices": len(anchors["vertex_ids"][idx]),
                "pos": pos[idx].tolist(),
                "normal": normals[idx].tolist(),
                "radius": float(radii[idx]),
            }
        )
    return rows


def bind_gaussians_to_anchors(
    mu_canon,
    gaussian_lbs,
    anchors,
    top_m=2,
    lambda_lbs=0.35,
    sigma_scale=1.8,
    min_sigma=0.035,
):
    if not torch.is_tensor(mu_canon):
        mu_canon = torch.as_tensor(mu_canon, dtype=torch.float32)
    if not torch.is_tensor(gaussian_lbs):
        gaussian_lbs = torch.as_tensor(gaussian_lbs, dtype=torch.float32)

    anchor_pos = anchors["pos_canon"].to(mu_canon.device, dtype=mu_canon.dtype)
    anchor_lbs = anchors["lbs"].to(mu_canon.device, dtype=mu_canon.dtype)
    anchor_radius = anchors["radius"].to(mu_canon.device, dtype=mu_canon.dtype)
    sigma = torch.clamp(anchor_radius * sigma_scale, min=min_sigma)

    dist2 = torch.cdist(mu_canon, anchor_pos, p=2) ** 2
    lbs_dist2 = torch.cdist(gaussian_lbs, anchor_lbs, p=2) ** 2
    scores = -(dist2 / (sigma[None] ** 2)) - lambda_lbs * lbs_dist2
    weights, ids = torch.topk(scores, k=min(top_m, scores.shape[1]), dim=1)
    soft_weights = torch.softmax(weights, dim=1)

    top1_ids = ids[:, 0]
    top1_dist = torch.linalg.norm(mu_canon - anchor_pos[top1_ids], dim=1)
    return {
        "gaussian_anchor_ids": ids.cpu(),
        "gaussian_anchor_weights": soft_weights.cpu(),
        "gaussian_anchor_scores": weights.cpu(),
        "top1_anchor_id": top1_ids.cpu(),
        "top1_anchor_dist": top1_dist.cpu(),
        "score_matrix": scores.cpu(),
        "sigma": sigma.cpu(),
    }


def compute_binding_local_mask(bindings, vis_radius_factor=1.35):
    top_ids = bindings["top1_anchor_id"]
    top_dist = bindings["top1_anchor_dist"]
    sigma = bindings["sigma"][top_ids]
    return top_dist <= (sigma * vis_radius_factor)


def build_initial_human_gaussians(template):
    vertices = np.asarray(template["vertices"], dtype=np.float32)
    faces = np.asarray(template["faces"], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    edge_lengths = np.linalg.norm(vertices[mesh.edges_unique[:, 0]] - vertices[mesh.edges_unique[:, 1]], axis=1)
    vertex_scale = np.full(vertices.shape[0], float(np.median(edge_lengths)), dtype=np.float32)
    for vid in range(vertices.shape[0]):
        edge_mask = np.any(mesh.edges_unique == vid, axis=1)
        if edge_mask.any():
            vertex_scale[vid] = float(edge_lengths[edge_mask].mean())
    scales = torch.from_numpy(vertex_scale[:, None].repeat(3, axis=1)).float() * 0.5
    return {
        "mu_canon": torch.from_numpy(vertices).float(),
        "lbs_weights": torch.from_numpy(template["lbs_weights"]).float(),
        "opacity": torch.full((template["vertices"].shape[0], 1), 0.1, dtype=torch.float32),
        "scale": scales.clamp(min=1e-4),
    }


def anchor_frame_from_normal(normal):
    normal = normal / torch.clamp(torch.linalg.norm(normal), min=1e-8)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=normal.dtype, device=normal.device)
    if torch.abs(torch.dot(normal, up)) > 0.92:
        up = torch.tensor([1.0, 0.0, 0.0], dtype=normal.dtype, device=normal.device)
    tangent = torch.cross(up, normal, dim=0)
    tangent = tangent / torch.clamp(torch.linalg.norm(tangent), min=1e-8)
    bitangent = torch.cross(normal, tangent, dim=0)
    bitangent = bitangent / torch.clamp(torch.linalg.norm(bitangent), min=1e-8)
    return torch.stack([tangent, bitangent, normal], dim=1)


def lbs_entropy(lbs_weights):
    lbs = lbs_weights.clamp(min=1e-8)
    return -(lbs * torch.log(lbs)).sum(dim=-1)


def compute_anchor_local_features(human, anchors, bindings, local_mask=None):
    mu = human["mu_canon"].float()
    scale = human["scale"].float().clamp(min=1e-8)
    opacity = human["opacity"].float().reshape(-1, 1)
    lbs = human["lbs_weights"].float()
    top_ids = bindings["top1_anchor_id"].long()
    top_weights = bindings["gaussian_anchor_weights"][:, 0].float().reshape(-1, 1)
    if local_mask is None:
        local_mask = torch.ones(mu.shape[0], dtype=torch.bool)
    else:
        local_mask = local_mask.bool()

    features_by_anchor = {}
    stats_rows = []
    pooled_vectors = []
    pooled_names = []
    entropy = lbs_entropy(lbs)

    for anchor_id, name in enumerate(anchors["names"]):
        mask = (top_ids == anchor_id) & local_mask
        all_mask = top_ids == anchor_id
        anchor_pos = anchors["pos_canon"][anchor_id].float()
        anchor_normal = anchors["normal_canon"][anchor_id].float()
        frame = anchor_frame_from_normal(anchor_normal)

        if mask.any():
            rel = mu[mask] - anchor_pos[None]
            rel_local = rel @ frame
            dist = torch.linalg.norm(rel, dim=1, keepdim=True)
            normal_dist = rel @ anchor_normal.reshape(3, 1)
            log_scale = torch.log(scale[mask])
            cov_diag = scale[mask] ** 2
            cov_local_diag = cov_diag
            extent_normal = torch.sqrt(torch.clamp((anchor_normal[None] ** 2 * cov_diag).sum(dim=1), min=1e-12)).reshape(-1, 1)
            weight = top_weights[mask]
            feat = torch.cat([
                rel_local,
                dist,
                normal_dist,
                log_scale,
                cov_local_diag,
                opacity[mask],
                entropy[mask].reshape(-1, 1),
                weight,
            ], dim=1)
            weighted_mean_dist = (dist * weight).sum() / torch.clamp(weight.sum(), min=1e-8)
            row = {
                "anchor_name": name,
                "num_gaussians": int(all_mask.sum()),
                "num_local_gaussians": int(mask.sum()),
                "mean_opacity": float(opacity[mask].mean()),
                "max_opacity": float(opacity[mask].max()),
                "mean_log_scale_x": float(log_scale[:, 0].mean()),
                "mean_log_scale_y": float(log_scale[:, 1].mean()),
                "mean_log_scale_z": float(log_scale[:, 2].mean()),
                "mean_cov_x": float(cov_local_diag[:, 0].mean()),
                "mean_cov_y": float(cov_local_diag[:, 1].mean()),
                "mean_cov_z": float(cov_local_diag[:, 2].mean()),
                "mean_extent_along_anchor_normal": float(extent_normal.mean()),
                "mean_distance_to_anchor": float(dist.mean()),
                "weighted_mean_distance_to_anchor": float(weighted_mean_dist),
                "mean_lbs_entropy": float(entropy[mask].mean()),
                "mean_assignment_weight": float(weight.mean()),
            }
            pooled = torch.cat([
                torch.tensor([float(mask.sum()), float(all_mask.sum())]),
                feat.mean(dim=0).detach().cpu(),
                feat.std(dim=0, unbiased=False).detach().cpu(),
            ])
        else:
            feat = torch.empty((0, 14), dtype=torch.float32)
            row = {
                "anchor_name": name,
                "num_gaussians": int(all_mask.sum()),
                "num_local_gaussians": 0,
                "mean_opacity": float("nan"),
                "max_opacity": float("nan"),
                "mean_log_scale_x": float("nan"),
                "mean_log_scale_y": float("nan"),
                "mean_log_scale_z": float("nan"),
                "mean_cov_x": float("nan"),
                "mean_cov_y": float("nan"),
                "mean_cov_z": float("nan"),
                "mean_extent_along_anchor_normal": float("nan"),
                "mean_distance_to_anchor": float("nan"),
                "weighted_mean_distance_to_anchor": float("nan"),
                "mean_lbs_entropy": float("nan"),
                "mean_assignment_weight": float("nan"),
            }
            pooled = torch.full((30,), float("nan"))
            pooled[:2] = torch.tensor([0.0, float(all_mask.sum())])
        features_by_anchor[name] = feat
        stats_rows.append(row)
        pooled_vectors.append(pooled.numpy())
        pooled_names.append(name)

    return {
        "features_by_anchor": features_by_anchor,
        "stats_rows": stats_rows,
        "pooled_vectors": np.stack(pooled_vectors, axis=0).astype(np.float32),
        "pooled_names": pooled_names,
        "feature_names": [
            "rel_x", "rel_y", "rel_z", "distance", "normal_distance",
            "log_scale_x", "log_scale_y", "log_scale_z",
            "cov_x", "cov_y", "cov_z", "opacity", "lbs_entropy", "assignment_weight",
        ],
    }
