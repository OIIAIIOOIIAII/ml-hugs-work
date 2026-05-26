#
# Anchor debug IO helpers.
#

import csv
import json
import os
from pathlib import Path

import torch


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_anchor_vertices(anchor_vertices, path):
    path = Path(path)
    ensure_dir(path.parent)
    serializable = {
        name: [int(v) for v in vertex_ids]
        for name, vertex_ids in anchor_vertices.items()
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, sort_keys=True)


def load_anchor_vertices(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {name: [int(v) for v in vertex_ids] for name, vertex_ids in data.items()}


def save_anchors_pt(anchors, path):
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(anchors, path)


def save_anchors_csv(anchors, path):
    path = Path(path)
    ensure_dir(path.parent)
    names = anchors["names"]
    pos = anchors["pos_canon"].detach().cpu().numpy()
    normals = anchors["normal_canon"].detach().cpu().numpy()
    radii = anchors["radius"].detach().cpu().numpy()
    vertex_ids = anchors["vertex_ids"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "anchor_name",
            "num_vertices",
            "pos_x",
            "pos_y",
            "pos_z",
            "normal_x",
            "normal_y",
            "normal_z",
            "radius",
            "vertex_ids",
        ])
        for idx, name in enumerate(names):
            writer.writerow([
                name,
                len(vertex_ids[idx]),
                float(pos[idx, 0]),
                float(pos[idx, 1]),
                float(pos[idx, 2]),
                float(normals[idx, 0]),
                float(normals[idx, 1]),
                float(normals[idx, 2]),
                float(radii[idx]),
                " ".join(str(int(v)) for v in vertex_ids[idx]),
            ])


def save_anchor_binding_csv(bindings, anchors, path):
    path = Path(path)
    ensure_dir(path.parent)
    top_ids = bindings["top1_anchor_id"].detach().cpu()
    top_dist = bindings["top1_anchor_dist"].detach().cpu()
    ids = bindings["gaussian_anchor_ids"].detach().cpu()
    weights = bindings["gaussian_anchor_weights"].detach().cpu()
    names = anchors["names"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "gaussian_id",
            "top1_anchor_id",
            "top1_anchor_name",
            "top1_distance",
            "topm_anchor_ids",
            "topm_anchor_names",
            "topm_weights",
        ])
        for idx in range(top_ids.shape[0]):
            topm_ids = [int(v) for v in ids[idx].tolist()]
            writer.writerow([
                idx,
                int(top_ids[idx]),
                names[int(top_ids[idx])],
                float(top_dist[idx]),
                " ".join(str(v) for v in topm_ids),
                " ".join(names[v] for v in topm_ids),
                " ".join(f"{float(v):.6f}" for v in weights[idx].tolist()),
            ])


def save_anchor_binding_summary_csv(bindings, anchors, path):
    path = Path(path)
    ensure_dir(path.parent)
    top_ids = bindings["top1_anchor_id"].detach().cpu()
    top_dist = bindings["top1_anchor_dist"].detach().cpu()
    top_weights = bindings["gaussian_anchor_weights"][:, 0].detach().cpu()

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "anchor_id",
            "anchor_name",
            "num_gaussians",
            "mean_top1_distance",
            "max_top1_distance",
            "mean_top1_weight",
        ])
        for anchor_id, name in enumerate(anchors["names"]):
            mask = top_ids == anchor_id
            if mask.any():
                writer.writerow([
                    anchor_id,
                    name,
                    int(mask.sum()),
                    float(top_dist[mask].mean()),
                    float(top_dist[mask].max()),
                    float(top_weights[mask].mean()),
                ])
            else:
                writer.writerow([anchor_id, name, 0, "", "", ""])


def make_anchor_debug_dir(output_dir, subdir="expA_template"):
    output_dir = Path(output_dir)
    if output_dir.name == "anchor_debug":
        return ensure_dir(output_dir / subdir)
    return ensure_dir(output_dir / "anchor_debug" / subdir)


def relative_to_cwd(path):
    try:
        return os.path.relpath(path, Path.cwd())
    except ValueError:
        return str(path)
