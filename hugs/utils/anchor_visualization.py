#
# PLY visualization helpers for anchor debug experiments.
#

from pathlib import Path

import numpy as np
import trimesh
from plyfile import PlyData, PlyElement


ANCHOR_COLORS = {
    "left_sole": (38, 139, 210),
    "right_sole": (220, 50, 47),
    "left_toe": (42, 161, 152),
    "right_toe": (203, 75, 22),
    "left_heel": (108, 113, 196),
    "right_heel": (211, 54, 130),
    "left_palm": (133, 153, 0),
    "right_palm": (181, 137, 0),
    "left_fingers": (88, 110, 117),
    "right_fingers": (101, 123, 131),
    "buttocks": (147, 161, 161),
    "back": (181, 137, 165),
    "left_knee": (0, 128, 128),
    "right_knee": (170, 80, 40),
    "left_elbow": (70, 120, 210),
    "right_elbow": (210, 90, 70),
}


def get_anchor_color(name, index=0):
    if name in ANCHOR_COLORS:
        return np.array(ANCHOR_COLORS[name], dtype=np.uint8)
    rng = np.random.default_rng(index + 12345)
    return rng.integers(40, 230, size=3, dtype=np.uint8)


def _vertex_dtype(extra_fields=None):
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    if extra_fields:
        dtype.extend(extra_fields)
    return dtype


def write_point_ply(path, points, colors, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    extra = extra or {}

    dtype = _vertex_dtype([(k, v[1]) for k, v in extra.items()])
    elements = np.empty(points.shape[0], dtype=dtype)
    elements["x"] = points[:, 0]
    elements["y"] = points[:, 1]
    elements["z"] = points[:, 2]
    elements["red"] = colors[:, 0]
    elements["green"] = colors[:, 1]
    elements["blue"] = colors[:, 2]
    for key, (values, _) in extra.items():
        elements[key] = values

    PlyData([PlyElement.describe(elements, "vertex")], text=False).write(path)


def write_mesh_ply(path, vertices, faces, colors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    colors = np.asarray(colors, dtype=np.uint8)

    vertex_elements = np.empty(vertices.shape[0], dtype=_vertex_dtype())
    vertex_elements["x"] = vertices[:, 0]
    vertex_elements["y"] = vertices[:, 1]
    vertex_elements["z"] = vertices[:, 2]
    vertex_elements["red"] = colors[:, 0]
    vertex_elements["green"] = colors[:, 1]
    vertex_elements["blue"] = colors[:, 2]

    face_elements = np.empty(faces.shape[0], dtype=[("vertex_indices", "i4", (3,))])
    face_elements["vertex_indices"] = faces

    PlyData([
        PlyElement.describe(vertex_elements, "vertex"),
        PlyElement.describe(face_elements, "face"),
    ], text=False).write(path)


def fibonacci_sphere(center, radius, color, n_points=96):
    center = np.asarray(center, dtype=np.float32)
    indices = np.arange(n_points, dtype=np.float32) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / n_points)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * indices
    points = np.stack([
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi),
    ], axis=1)
    points = center[None] + radius * points
    colors = np.repeat(np.asarray(color, dtype=np.uint8)[None], n_points, axis=0)
    return points.astype(np.float32), colors


def line_points(start, direction, length, color, n_points=24):
    start = np.asarray(start, dtype=np.float32)
    direction = np.asarray(direction, dtype=np.float32)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-8)
    ts = np.linspace(0.0, length, n_points, dtype=np.float32)
    points = start[None] + ts[:, None] * direction[None]
    colors = np.repeat(np.asarray(color, dtype=np.uint8)[None], n_points, axis=0)
    return points.astype(np.float32), colors


def export_template_mesh_with_anchor_regions(path, vertices, faces, anchors):
    colors = np.full((vertices.shape[0], 3), 188, dtype=np.uint8)
    for idx, name in enumerate(anchors["names"]):
        color = get_anchor_color(name, idx)
        vertex_ids = np.asarray(anchors["vertex_ids"][idx], dtype=np.int64)
        colors[vertex_ids] = color
    write_mesh_ply(path, vertices, faces, colors)


def export_template_mesh_with_anchor_spheres(path, vertices, faces, anchors, sphere_radius=0.025):
    mesh_vertices = [np.asarray(vertices, dtype=np.float32)]
    mesh_faces = [np.asarray(faces, dtype=np.int32)]
    mesh_colors = [np.full((vertices.shape[0], 3), 188, dtype=np.uint8)]
    vertex_offset = vertices.shape[0]

    unit_sphere = trimesh.creation.icosphere(subdivisions=2, radius=sphere_radius)
    sphere_vertices = np.asarray(unit_sphere.vertices, dtype=np.float32)
    sphere_faces = np.asarray(unit_sphere.faces, dtype=np.int32)
    pos = anchors["pos_canon"].detach().cpu().numpy()

    for idx, name in enumerate(anchors["names"]):
        color = get_anchor_color(name, idx)
        verts = sphere_vertices + pos[idx][None]
        faces_i = sphere_faces + vertex_offset
        colors = np.repeat(color[None], verts.shape[0], axis=0)
        mesh_vertices.append(verts)
        mesh_faces.append(faces_i)
        mesh_colors.append(colors)
        vertex_offset += verts.shape[0]

    write_mesh_ply(
        path,
        np.concatenate(mesh_vertices, axis=0),
        np.concatenate(mesh_faces, axis=0),
        np.concatenate(mesh_colors, axis=0),
    )


def _rotation_from_z_axis(direction):
    direction = np.asarray(direction, dtype=np.float32)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-8)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    cross = np.cross(z_axis, direction)
    dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < 1e-8:
        if dot > 0.0:
            return np.eye(3, dtype=np.float32)
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float32,
        )

    axis = cross / cross_norm
    kx = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float32,
    )
    angle = np.arccos(dot)
    return (np.eye(3, dtype=np.float32) + np.sin(angle) * kx + (1.0 - np.cos(angle)) * (kx @ kx)).astype(np.float32)


def _append_oriented_mesh(mesh_vertices, mesh_faces, mesh_colors, primitive, center, direction, color, vertex_offset):
    rot = _rotation_from_z_axis(direction)
    verts = np.asarray(primitive.vertices, dtype=np.float32)
    verts = (rot[None] @ verts[..., None])[..., 0] + np.asarray(center, dtype=np.float32)[None]
    faces = np.asarray(primitive.faces, dtype=np.int32) + vertex_offset
    colors = np.repeat(np.asarray(color, dtype=np.uint8)[None], verts.shape[0], axis=0)
    mesh_vertices.append(verts)
    mesh_faces.append(faces)
    mesh_colors.append(colors)
    return vertex_offset + verts.shape[0]


def export_template_mesh_with_anchor_spheres_normals(
    path,
    vertices,
    faces,
    anchors,
    sphere_radius=0.025,
    normal_length=0.12,
    normal_radius=0.004,
):
    mesh_vertices = [np.asarray(vertices, dtype=np.float32)]
    mesh_faces = [np.asarray(faces, dtype=np.int32)]
    mesh_colors = [np.full((vertices.shape[0], 3), 188, dtype=np.uint8)]
    vertex_offset = vertices.shape[0]

    unit_sphere = trimesh.creation.icosphere(subdivisions=2, radius=sphere_radius)
    shaft = trimesh.creation.cylinder(radius=normal_radius, height=normal_length * 0.78, sections=12)
    tip = trimesh.creation.cone(radius=normal_radius * 2.8, height=normal_length * 0.22, sections=16)

    sphere_vertices = np.asarray(unit_sphere.vertices, dtype=np.float32)
    sphere_faces = np.asarray(unit_sphere.faces, dtype=np.int32)
    pos = anchors["pos_canon"].detach().cpu().numpy()
    normals = anchors["normal_canon"].detach().cpu().numpy()

    for idx, name in enumerate(anchors["names"]):
        color = get_anchor_color(name, idx)
        verts = sphere_vertices + pos[idx][None]
        faces_i = sphere_faces + vertex_offset
        colors = np.repeat(color[None], verts.shape[0], axis=0)
        mesh_vertices.append(verts)
        mesh_faces.append(faces_i)
        mesh_colors.append(colors)
        vertex_offset += verts.shape[0]

        normal = normals[idx] / max(float(np.linalg.norm(normals[idx])), 1e-8)
        shaft_center = pos[idx] + normal * (sphere_radius + normal_length * 0.39)
        tip_center = pos[idx] + normal * (sphere_radius + normal_length * 0.89)
        vertex_offset = _append_oriented_mesh(
            mesh_vertices,
            mesh_faces,
            mesh_colors,
            shaft,
            shaft_center,
            normal,
            (255, 255, 255),
            vertex_offset,
        )
        vertex_offset = _append_oriented_mesh(
            mesh_vertices,
            mesh_faces,
            mesh_colors,
            tip,
            tip_center,
            normal,
            (255, 214, 64),
            vertex_offset,
        )

    write_mesh_ply(
        path,
        np.concatenate(mesh_vertices, axis=0),
        np.concatenate(mesh_faces, axis=0),
        np.concatenate(mesh_colors, axis=0),
    )


def export_anchor_region_points(path, vertices, anchors):
    all_points = []
    all_colors = []
    all_anchor_ids = []
    all_is_center = []

    pos = anchors["pos_canon"].detach().cpu().numpy()
    for idx, name in enumerate(anchors["names"]):
        color = get_anchor_color(name, idx)
        vertex_ids = np.asarray(anchors["vertex_ids"][idx], dtype=np.int64)
        pts = vertices[vertex_ids]
        all_points.append(pts)
        all_colors.append(np.repeat(color[None], len(vertex_ids), axis=0))
        all_anchor_ids.append(np.full(len(vertex_ids), idx, dtype=np.int32))
        all_is_center.append(np.zeros(len(vertex_ids), dtype=np.uint8))

        all_points.append(pos[idx][None])
        all_colors.append(np.array([[255, 255, 255]], dtype=np.uint8))
        all_anchor_ids.append(np.array([idx], dtype=np.int32))
        all_is_center.append(np.array([1], dtype=np.uint8))

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    extra = {
        "anchor_id": (np.concatenate(all_anchor_ids, axis=0), "i4"),
        "is_center": (np.concatenate(all_is_center, axis=0), "u1"),
    }
    write_point_ply(path, points, colors, extra=extra)


def export_anchor_spheres_with_normals(path, anchors, sphere_radius=0.018, normal_length=0.11):
    all_points = []
    all_colors = []
    all_anchor_ids = []
    all_point_types = []

    pos = anchors["pos_canon"].detach().cpu().numpy()
    normals = anchors["normal_canon"].detach().cpu().numpy()
    for idx, name in enumerate(anchors["names"]):
        color = get_anchor_color(name, idx)
        sp, sp_c = fibonacci_sphere(pos[idx], sphere_radius, color)
        ln, ln_c = line_points(pos[idx], normals[idx], normal_length, (255, 255, 255))

        all_points.extend([sp, ln])
        all_colors.extend([sp_c, ln_c])
        all_anchor_ids.extend([
            np.full(sp.shape[0], idx, dtype=np.int32),
            np.full(ln.shape[0], idx, dtype=np.int32),
        ])
        all_point_types.extend([
            np.zeros(sp.shape[0], dtype=np.uint8),
            np.ones(ln.shape[0], dtype=np.uint8),
        ])

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    extra = {
        "anchor_id": (np.concatenate(all_anchor_ids, axis=0), "i4"),
        "point_type": (np.concatenate(all_point_types, axis=0), "u1"),
    }
    write_point_ply(path, points, colors, extra=extra)


def export_anchor_labels_legend(path, anchors):
    rows = []
    colors = []
    spacing = 0.08
    for idx, name in enumerate(anchors["names"]):
        y = -idx * spacing
        for j in range(12):
            rows.append([j * 0.012, y, 0.0])
            colors.append(get_anchor_color(name, idx))
    extra = {
        "anchor_id": (np.repeat(np.arange(len(anchors["names"]), dtype=np.int32), 12), "i4"),
    }
    write_point_ply(path, np.asarray(rows, dtype=np.float32), np.asarray(colors, dtype=np.uint8), extra=extra)


def colors_for_anchor_ids(anchor_ids, anchors, local_mask=None, gray=(205, 205, 205)):
    anchor_ids = np.asarray(anchor_ids, dtype=np.int64)
    colors = np.zeros((anchor_ids.shape[0], 3), dtype=np.uint8)
    for idx, name in enumerate(anchors["names"]):
        colors[anchor_ids == idx] = get_anchor_color(name, idx)
    if local_mask is not None:
        local_mask = np.asarray(local_mask, dtype=bool)
        colors[~local_mask] = np.asarray(gray, dtype=np.uint8)
    return colors


def export_human_gaussian_anchor_points(path, mu_canon, bindings, anchors, local_mask=None):
    mu = np.asarray(mu_canon, dtype=np.float32)
    top_ids = bindings["top1_anchor_id"].detach().cpu().numpy().astype(np.int32)
    top_dist = bindings["top1_anchor_dist"].detach().cpu().numpy().astype(np.float32)
    top_weight = bindings["gaussian_anchor_weights"][:, 0].detach().cpu().numpy().astype(np.float32)
    local_mask_np = np.ones(mu.shape[0], dtype=np.uint8)
    if local_mask is not None:
        local_mask_np = local_mask.detach().cpu().numpy().astype(np.uint8)
    colors = colors_for_anchor_ids(top_ids, anchors, local_mask=local_mask_np.astype(bool))
    extra = {
        "anchor_id": (top_ids, "i4"),
        "top1_weight": (top_weight, "f4"),
        "dist_to_anchor": (top_dist, "f4"),
        "is_local": (local_mask_np, "u1"),
        "point_type": (np.zeros(mu.shape[0], dtype=np.uint8), "u1"),
    }
    write_point_ply(path, mu, colors, extra=extra)


def export_human_gaussian_anchor_points_with_anchors(path, mu_canon, bindings, anchors, sphere_radius=0.018, normal_length=0.10, local_mask=None):
    mu = np.asarray(mu_canon, dtype=np.float32)
    top_ids = bindings["top1_anchor_id"].detach().cpu().numpy().astype(np.int32)
    top_dist = bindings["top1_anchor_dist"].detach().cpu().numpy().astype(np.float32)
    top_weight = bindings["gaussian_anchor_weights"][:, 0].detach().cpu().numpy().astype(np.float32)
    local_mask_np = np.ones(mu.shape[0], dtype=np.uint8)
    if local_mask is not None:
        local_mask_np = local_mask.detach().cpu().numpy().astype(np.uint8)

    points = [mu]
    colors = [colors_for_anchor_ids(top_ids, anchors, local_mask=local_mask_np.astype(bool))]
    anchor_ids = [top_ids]
    point_types = [np.zeros(mu.shape[0], dtype=np.uint8)]
    dists = [top_dist]
    weights = [top_weight]
    is_local = [local_mask_np]

    pos = anchors["pos_canon"].detach().cpu().numpy()
    normals = anchors["normal_canon"].detach().cpu().numpy()
    for idx, name in enumerate(anchors["names"]):
        color = get_anchor_color(name, idx)
        sp, sp_c = fibonacci_sphere(pos[idx], sphere_radius, color, n_points=80)
        ln, ln_c = line_points(pos[idx], normals[idx], normal_length, (255, 255, 255), n_points=20)
        points.extend([sp, ln])
        colors.extend([sp_c, ln_c])
        anchor_ids.extend([
            np.full(sp.shape[0], idx, dtype=np.int32),
            np.full(ln.shape[0], idx, dtype=np.int32),
        ])
        point_types.extend([
            np.ones(sp.shape[0], dtype=np.uint8),
            np.full(ln.shape[0], 2, dtype=np.uint8),
        ])
        dists.extend([
            np.zeros(sp.shape[0], dtype=np.float32),
            np.zeros(ln.shape[0], dtype=np.float32),
        ])
        weights.extend([
            np.ones(sp.shape[0], dtype=np.float32),
            np.ones(ln.shape[0], dtype=np.float32),
        ])
        is_local.extend([
            np.ones(sp.shape[0], dtype=np.uint8),
            np.ones(ln.shape[0], dtype=np.uint8),
        ])

    extra = {
        "anchor_id": (np.concatenate(anchor_ids, axis=0), "i4"),
        "top1_weight": (np.concatenate(weights, axis=0), "f4"),
        "dist_to_anchor": (np.concatenate(dists, axis=0), "f4"),
        "is_local": (np.concatenate(is_local, axis=0), "u1"),
        "point_type": (np.concatenate(point_types, axis=0), "u1"),
    }
    write_point_ply(
        path,
        np.concatenate(points, axis=0),
        np.concatenate(colors, axis=0),
        extra=extra,
    )


def export_binding_on_template_mesh(path, vertices, faces, bindings, anchors, local_mask=None):
    top_ids = bindings["top1_anchor_id"].detach().cpu().numpy().astype(np.int64)
    local_mask_np = None
    if local_mask is not None:
        local_mask_np = local_mask.detach().cpu().numpy().astype(bool)
    colors = colors_for_anchor_ids(top_ids, anchors, local_mask=local_mask_np)
    write_mesh_ply(path, vertices, faces, colors)


def render_binding_pngs(out_dir, mu_canon, bindings, anchors, image_size=1500, local_mask=None, suffix=""):
    from PIL import Image, ImageDraw, ImageFont

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mu = np.asarray(mu_canon, dtype=np.float32)
    top_ids = bindings["top1_anchor_id"].detach().cpu().numpy().astype(np.int64)
    local_mask_np = None
    if local_mask is not None:
        local_mask_np = local_mask.detach().cpu().numpy().astype(bool)
    colors = colors_for_anchor_ids(top_ids, anchors, local_mask=local_mask_np).astype(np.float32) / 255.0
    pos = anchors["pos_canon"].detach().cpu().numpy()
    normals = anchors["normal_canon"].detach().cpu().numpy()

    mins = mu.min(axis=0)
    maxs = mu.max(axis=0)
    center = (mins + maxs) * 0.5
    span = float(np.max(maxs - mins))

    views = {
        "front": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
        "back": (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        "left": (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        "right": (np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0])),
        "top": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
    }
    font = ImageFont.load_default()

    def project(points, right, up):
        rel = points - center[None]
        uv = np.stack([rel @ right, rel @ up], axis=1)
        scale = image_size * 0.78 / max(span, 1e-6)
        px = image_size * 0.5 + uv[:, 0] * scale
        py = image_size * 0.5 - uv[:, 1] * scale
        return np.stack([px, py], axis=1), scale

    written = []
    for name, (right, up, depth_axis) in views.items():
        image = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        draw = ImageDraw.Draw(image, "RGBA")
        pix, scale = project(mu, right, up)
        depth = (mu - center[None]) @ depth_axis
        order = np.argsort(depth)

        point_radius = max(1, int(round(image_size / 900)))
        for idx in order:
            x, y = pix[idx]
            color = tuple((colors[idx] * 255).astype(np.uint8).tolist()) + (215,)
            draw.ellipse(
                (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
                fill=color,
            )

        anchor_pix, _ = project(pos, right, up)
        for idx, anchor_name in enumerate(anchors["names"]):
            x, y = anchor_pix[idx]
            color = tuple(get_anchor_color(anchor_name, idx).tolist()) + (255,)
            r = max(7, int(round(image_size / 95)))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(0, 0, 0, 255), width=2)

            normal_end = pos[idx] + normals[idx] * 0.10
            end_pix, _ = project(normal_end[None], right, up)
            ex, ey = end_pix[0]
            draw.line((x, y, ex, ey), fill=(0, 0, 0, 255), width=max(2, int(round(image_size / 500))))
            draw.ellipse((ex - 3, ey - 3, ex + 3, ey + 3), fill=(255, 214, 64, 255))

            text = anchor_name.replace("_", " ")
            draw.text((x + r + 3, y - r), text, fill=(0, 0, 0, 230), font=font)

        title = f"Exp B initial human Gaussian -> anchor binding{suffix} ({name})"
        draw.rectangle((0, 0, image_size, 34), fill=(255, 255, 255, 210))
        draw.text((10, 10), title, fill=(0, 0, 0, 255), font=font)
        out_path = out_dir / f"human_gaussians_anchor_init{suffix}_{name}.png"
        image.save(out_path)
        written.append(out_path)
    return written
