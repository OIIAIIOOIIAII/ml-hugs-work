#
# Fit HUGS-compatible SMPL parameters to Human3R SMPL-X predictions.
#
# Human3R writes SMPL-X in a head-centered convention:
#   transl == the selected person center (head) in camera coordinates.
# HUGS expects SMPL vertices to be transformed as:
#   world_vertex = smpl_vertex * smpl_scale + transl
#
# This script reconstructs Human3R's SMPL-X vertices in world coordinates,
# maps them to SMPL topology, and optimizes SMPL global_orient/body_pose,
# a shared shape, scale, and translation for direct use by HUGS.
#

import argparse
import importlib.util
import json
import pickle
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import torch


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def sorted_stems(folder, suffix):
    return [p.stem for p in sorted(Path(folder).glob(f"*{suffix}"))]


def copy_dataset(src, dst):
    src = Path(src)
    dst = Path(dst)
    if src.resolve() == dst.resolve():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_hugs_smpl_class(repo_root):
    repo_root = Path(repo_root)
    hugs_pkg = sys.modules.setdefault("hugs", types.ModuleType("hugs"))
    hugs_pkg.__path__ = [str(repo_root / "hugs")]
    models_pkg = sys.modules.setdefault("hugs.models", types.ModuleType("hugs.models"))
    models_pkg.__path__ = [str(repo_root / "hugs" / "models")]
    modules_pkg = sys.modules.setdefault("hugs.models.modules", types.ModuleType("hugs.models.modules"))
    modules_pkg.__path__ = [str(repo_root / "hugs" / "models" / "modules")]

    load_module_from_file(
        "hugs.models.modules.lbs",
        repo_root / "hugs" / "models" / "modules" / "lbs.py",
    )
    smpl_layer = load_module_from_file(
        "hugs.models.modules.smpl_layer",
        repo_root / "hugs" / "models" / "modules" / "smpl_layer.py",
    )
    return smpl_layer.SMPL


def batch_rodrigues(rotvec):
    angle = torch.linalg.norm(rotvec + 1e-8, dim=-1, keepdim=True)
    direction = rotvec / angle
    ca = torch.cos(angle).unsqueeze(-1)
    sa = torch.sin(angle).unsqueeze(-1)
    x, y, z = direction.unbind(-1)
    zeros = torch.zeros_like(x)
    k = torch.stack(
        [
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)
    ident = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).unsqueeze(0)
    return ident + sa * k + (1.0 - ca) * (k @ k)


def robust_l1(x, eps=1e-4):
    return torch.sqrt(x * x + eps * eps).mean()


def pick_person(smpl_npz, person_index):
    n_people = int(smpl_npz["shape"].shape[0])
    if n_people == 0:
        raise ValueError("Human3R SMPL file contains zero people")
    if person_index >= n_people:
        raise IndexError(f"person_index={person_index} but only {n_people} people found")
    return person_index


def transform_points(points, c2w):
    rot = c2w[:, :3, :3]
    trans = c2w[:, :3, 3]
    return torch.einsum("bij,bnj->bni", rot, points) + trans[:, None, :]


def world_to_camera(points_world, c2w):
    rot = c2w[:, :3, :3]
    trans = c2w[:, :3, 3]
    return torch.einsum("bji,bnj->bni", rot, points_world - trans[:, None, :])


def project_bbox(points_world, c2w, intrinsics, image_sizes):
    points_cam = world_to_camera(points_world, c2w)
    z = points_cam[..., 2].clamp_min(1e-4)
    u = intrinsics[:, 0, 0, None] * points_cam[..., 0] / z + intrinsics[:, 0, 2, None]
    v = intrinsics[:, 1, 1, None] * points_cam[..., 1] / z + intrinsics[:, 1, 2, None]
    h = image_sizes[:, 0, None].float()
    w = image_sizes[:, 1, None].float()
    u = u.clamp(-w, 2.0 * w)
    v = v.clamp(-h, 2.0 * h)
    return torch.stack([v.min(1).values, u.min(1).values, v.max(1).values, u.max(1).values], dim=-1)


def load_human3r_sequence(result_dir, person_index, max_frames):
    result_dir = Path(result_dir)
    stems = sorted_stems(result_dir / "color", ".png")
    if max_frames > 0:
        stems = stems[:max_frames]
    if not stems:
        raise RuntimeError(f"No Human3R frames found under {result_dir / 'color'}")

    shapes, rotvecs, transls, exprs, c2ws, intrinsics = [], [], [], [], [], []
    for stem in stems:
        smpl_npz = np.load(result_dir / "smpl" / f"{stem}.npz", allow_pickle=True)
        idx = pick_person(smpl_npz, person_index)
        shapes.append(smpl_npz["shape"][idx].astype(np.float32))
        rotvecs.append(smpl_npz["rotvec"][idx].astype(np.float32))
        transls.append(smpl_npz["transl"][idx].astype(np.float32))
        if "expression" in smpl_npz.files and smpl_npz["expression"] is not None:
            expr = smpl_npz["expression"]
            if not (getattr(expr, "dtype", None) == object and expr.shape == () and expr.item() is None):
                exprs.append(expr[idx].astype(np.float32))
            else:
                exprs.append(None)
        else:
            exprs.append(None)

        cam = np.load(result_dir / "camera" / f"{stem}.npz")
        c2ws.append(cam["pose"].astype(np.float32))
        intrinsics.append(cam["intrinsics"].astype(np.float32))

    expression = None
    if all(x is not None for x in exprs):
        expression = np.stack(exprs).astype(np.float32)

    return {
        "stems": stems,
        "shape": np.stack(shapes).astype(np.float32),
        "rotvec": np.stack(rotvecs).astype(np.float32),
        "transl": np.stack(transls).astype(np.float32),
        "expression": expression,
        "c2w": np.stack(c2ws).astype(np.float32),
        "intrinsics": np.stack(intrinsics).astype(np.float32),
    }


def create_smplx_layer(model_dir, num_betas, device):
    import smplx

    model_dir = Path(model_dir)
    if model_dir.name.lower() == "smplx":
        model_dir = model_dir.parent

    kwargs = dict(
        model_path=str(model_dir),
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=num_betas,
    )
    try:
        return smplx.create(**kwargs).to(device)
    except (AttributeError, KeyError, ValueError):
        kwargs["ext"] = "pkl"
        return smplx.create(**kwargs).to(device)


def human3r_smplx_world_targets(seq, smplx_model_dir, smplx2smpl_path, params_frame, device, chunk_size):
    from smplx.joint_names import JOINT_NAMES

    num_frames = seq["rotvec"].shape[0]
    num_betas = seq["shape"].shape[1]
    smplx_layer = create_smplx_layer(smplx_model_dir, num_betas, device)
    head_idx = JOINT_NAMES[:127].index("head")

    matrix = None
    valid_vertices = None
    mapping = None
    if smplx2smpl_path:
        with open(smplx2smpl_path, "rb") as f:
            smplx2smpl = pickle.load(f, encoding="latin1")
        if "matrix" in smplx2smpl:
            matrix = torch.as_tensor(smplx2smpl["matrix"], dtype=torch.float32, device=device)
        else:
            valid_vertices = torch.as_tensor(smplx2smpl["valid_vertices"], dtype=torch.long, device=device)
            mapping = torch.as_tensor(smplx2smpl["mapping"], dtype=torch.long, device=device)

    target_smpl_vertices, target_joints = [], []
    with torch.no_grad():
        for start in range(0, num_frames, chunk_size):
            end = min(num_frames, start + chunk_size)
            rotvec = torch.as_tensor(seq["rotvec"][start:end], dtype=torch.float32, device=device)
            shape = torch.as_tensor(seq["shape"][start:end], dtype=torch.float32, device=device)
            transl = torch.as_tensor(seq["transl"][start:end], dtype=torch.float32, device=device)

            kwargs = {
                "betas": shape,
                "global_orient": torch.zeros((end - start, 3), dtype=torch.float32, device=device),
                "body_pose": rotvec[:, 1:22].reshape(end - start, -1),
                "left_hand_pose": rotvec[:, 22:37].reshape(end - start, -1),
                "right_hand_pose": rotvec[:, 37:52].reshape(end - start, -1),
                "jaw_pose": rotvec[:, 52:53].reshape(end - start, -1),
                "leye_pose": smplx_layer.leye_pose.repeat(end - start, 1),
                "reye_pose": smplx_layer.reye_pose.repeat(end - start, 1),
            }
            if seq["expression"] is not None:
                kwargs["expression"] = torch.as_tensor(seq["expression"][start:end], dtype=torch.float32, device=device)
            else:
                kwargs["expression"] = smplx_layer.expression.repeat(end - start, 1)

            out = smplx_layer(**kwargs)
            smplx_verts = out.vertices
            smplx_joints = out.joints

            root_rot = batch_rodrigues(rotvec[:, 0])
            pelvis = smplx_joints[:, [0]]
            smplx_verts = torch.einsum("bij,bnj->bni", root_rot, smplx_verts - pelvis)
            smplx_joints = torch.einsum("bij,bnj->bni", root_rot, smplx_joints - pelvis)
            person_center = smplx_joints[:, [head_idx]]
            smplx_verts = smplx_verts - person_center + transl[:, None, :]
            smplx_joints = smplx_joints - person_center + transl[:, None, :]

            if params_frame == "camera":
                c2w = torch.as_tensor(seq["c2w"][start:end], dtype=torch.float32, device=device)
                smplx_verts = transform_points(smplx_verts, c2w)
                smplx_joints = transform_points(smplx_joints, c2w)
            elif params_frame != "world":
                raise ValueError(f"Unknown params_frame: {params_frame}")

            if matrix is not None:
                smpl_verts = torch.matmul(matrix, smplx_verts)
            elif valid_vertices is not None:
                smpl_verts = torch.zeros((end - start, 6890, 3), dtype=torch.float32, device=device)
                smpl_verts[:, valid_vertices] = smplx_verts[:, mapping]
            else:
                raise ValueError("A smplx2smpl mapping file is required")

            target_smpl_vertices.append(smpl_verts.cpu())
            target_joints.append(smplx_joints[:, :22].cpu())

    return torch.cat(target_smpl_vertices, dim=0), torch.cat(target_joints, dim=0)


def prepare_initial_params(hugs_dir, num_frames, device):
    smpl_path = Path(hugs_dir) / "4d_humans" / "smpl_optimized_aligned_scale.npz"
    if smpl_path.is_file():
        data = np.load(smpl_path)
        global_orient = torch.as_tensor(data["global_orient"][:num_frames], dtype=torch.float32, device=device)
        body_pose = torch.as_tensor(data["body_pose"][:num_frames], dtype=torch.float32, device=device)
        betas = torch.as_tensor(data["betas"][:num_frames].mean(axis=0), dtype=torch.float32, device=device)
        transl = torch.as_tensor(data["transl"][:num_frames], dtype=torch.float32, device=device)
        scale = torch.as_tensor(data["scale"][:num_frames], dtype=torch.float32, device=device).clamp_min(1e-4)
        bbox = data["bbox"][:num_frames].astype(np.float32) if "bbox" in data.files else None
    else:
        global_orient = torch.zeros((num_frames, 3), dtype=torch.float32, device=device)
        body_pose = torch.zeros((num_frames, 69), dtype=torch.float32, device=device)
        betas = torch.zeros(10, dtype=torch.float32, device=device)
        transl = torch.zeros((num_frames, 3), dtype=torch.float32, device=device)
        scale = torch.ones(num_frames, dtype=torch.float32, device=device)
        bbox = None
    return global_orient, body_pose, betas, transl, scale, bbox


def fit(args):
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir)
    copy_dataset(args.hugs_dir, out_dir)
    ensure_dir(out_dir / "4d_humans")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    device = torch.device(device)
    print(f"Using device: {device}")

    seq = load_human3r_sequence(args.human3r_result_dir, args.person_index, args.max_frames)
    num_frames = len(seq["stems"])

    image_sizes = None
    bbox_np = None
    cams_path = Path(args.hugs_dir) / "cameras.npz"
    if cams_path.is_file():
        cams = np.load(cams_path)
        if "image_sizes" in cams.files:
            image_sizes = torch.as_tensor(cams["image_sizes"][:num_frames], dtype=torch.float32, device=device)

    smplx_model_dir = Path(args.smplx_model_dir)
    smplx2smpl_path = args.smplx2smpl_path
    if smplx2smpl_path is None:
        if smplx_model_dir.name.lower() == "smplx":
            smplx2smpl_path = str(smplx_model_dir / "smplx2smpl.pkl")
        else:
            smplx2smpl_path = str(smplx_model_dir / "smplx" / "smplx2smpl.pkl")

    target_vertices, target_joints = human3r_smplx_world_targets(
        seq,
        smplx_model_dir,
        smplx2smpl_path,
        args.smpl_params_frame,
        device,
        args.target_chunk_size,
    )
    target_vertices = target_vertices.to(device)
    target_joints = target_joints.to(device)

    SMPL = load_hugs_smpl_class(repo_root)
    smpl = SMPL(args.smpl_model_dir).to(device)

    init_global, init_body, init_betas, init_transl, init_scale, bbox_np = prepare_initial_params(
        args.hugs_dir, num_frames, device
    )
    target_center = target_vertices.mean(dim=1)
    init_smpl = smpl(
        betas=init_betas[None].repeat(num_frames, 1),
        body_pose=init_body,
        global_orient=init_global,
    )
    init_center = init_smpl.vertices.mean(dim=1)
    init_transl = target_center - init_center

    global_orient = torch.nn.Parameter(init_global.clone())
    body_pose = torch.nn.Parameter(init_body.clone())
    betas = torch.nn.Parameter(init_betas.clone())
    transl = torch.nn.Parameter(init_transl.clone())
    if args.per_frame_scale:
        log_scale = torch.nn.Parameter(init_scale.log().clone())
    else:
        log_scale = torch.nn.Parameter(init_scale.mean().log().reshape(1).clone())

    params = [
        {"params": [global_orient, body_pose], "lr": args.lr_pose},
        {"params": [transl], "lr": args.lr_transl},
        {"params": [betas], "lr": args.lr_betas},
        {"params": [log_scale], "lr": args.lr_scale},
    ]
    optim = torch.optim.Adam(params)

    if args.vertex_sample > 0 and args.vertex_sample < target_vertices.shape[1]:
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed)
        vertex_ids = torch.randperm(target_vertices.shape[1], generator=gen, device=device)[: args.vertex_sample]
    else:
        vertex_ids = torch.arange(target_vertices.shape[1], device=device)

    frame_ids_all = torch.arange(num_frames, device=device)
    c2w_t = torch.as_tensor(seq["c2w"], dtype=torch.float32, device=device)
    intr_t = torch.as_tensor(seq["intrinsics"], dtype=torch.float32, device=device)
    bbox_t = None
    if bbox_np is not None:
        bbox_t = torch.as_tensor(bbox_np[:, :4], dtype=torch.float32, device=device)

    last = {}
    for it in range(1, args.iters + 1):
        if args.batch_size > 0 and args.batch_size < num_frames:
            batch_ids = frame_ids_all[torch.randperm(num_frames, device=device)[: args.batch_size]]
        else:
            batch_ids = frame_ids_all

        scale = log_scale.exp()
        scale_b = scale[batch_ids] if args.per_frame_scale else scale.expand(len(batch_ids))
        smpl_out = smpl(
            betas=betas[None].repeat(len(batch_ids), 1),
            body_pose=body_pose[batch_ids],
            global_orient=global_orient[batch_ids],
        )
        pred_vertices = smpl_out.vertices * scale_b[:, None, None] + transl[batch_ids, None, :]
        pred_joints = smpl_out.joints[:, :22] * scale_b[:, None, None] + transl[batch_ids, None, :]

        loss_vertex = robust_l1(pred_vertices[:, vertex_ids] - target_vertices[batch_ids][:, vertex_ids])
        loss_joints = robust_l1(pred_joints - target_joints[batch_ids])
        loss_pose_prior = (body_pose[batch_ids] ** 2).mean()
        loss_betas_prior = (betas ** 2).mean()

        loss_temporal = torch.zeros((), dtype=torch.float32, device=device)
        if num_frames > 1 and args.lambda_temporal > 0:
            loss_temporal = (
                (body_pose[1:] - body_pose[:-1]).pow(2).mean()
                + 0.1 * (global_orient[1:] - global_orient[:-1]).pow(2).mean()
                + 0.1 * (transl[1:] - transl[:-1]).pow(2).mean()
            )
            if args.per_frame_scale:
                loss_temporal = loss_temporal + 0.1 * (log_scale[1:] - log_scale[:-1]).pow(2).mean()

        loss_bbox = torch.zeros((), dtype=torch.float32, device=device)
        if args.lambda_bbox > 0 and bbox_t is not None and image_sizes is not None:
            pred_bbox = project_bbox(pred_vertices, c2w_t[batch_ids], intr_t[batch_ids], image_sizes[batch_ids])
            norm = image_sizes[batch_ids].repeat(1, 2).clamp_min(1.0)
            loss_bbox = robust_l1((pred_bbox - bbox_t[batch_ids]) / norm)

        loss = (
            args.lambda_vertex * loss_vertex
            + args.lambda_joints * loss_joints
            + args.lambda_bbox * loss_bbox
            + args.lambda_pose_prior * loss_pose_prior
            + args.lambda_betas_prior * loss_betas_prior
            + args.lambda_temporal * loss_temporal
        )

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if it == 1 or it % args.log_interval == 0 or it == args.iters:
            last = {
                "iter": it,
                "loss": float(loss.detach().cpu()),
                "vertex": float(loss_vertex.detach().cpu()),
                "joints": float(loss_joints.detach().cpu()),
                "bbox": float(loss_bbox.detach().cpu()),
                "pose_prior": float(loss_pose_prior.detach().cpu()),
                "betas_prior": float(loss_betas_prior.detach().cpu()),
                "temporal": float(loss_temporal.detach().cpu()),
                "scale_mean": float(scale.detach().mean().cpu()),
            }
            print(json.dumps(last, indent=None))

    with torch.no_grad():
        final_scale = log_scale.exp()
        all_scale = final_scale if args.per_frame_scale else final_scale.expand(num_frames)
        smpl_out = smpl(
            betas=betas[None].repeat(num_frames, 1),
            body_pose=body_pose,
            global_orient=global_orient,
        )
        pred_vertices = smpl_out.vertices * all_scale[:, None, None] + transl[:, None, :]
        pred_joints = smpl_out.joints[:, :22] * all_scale[:, None, None] + transl[:, None, :]
        full_vertex_l1 = robust_l1(pred_vertices - target_vertices).item()
        full_joint_l1 = robust_l1(pred_joints - target_joints).item()

        bbox_l1_px = None
        if bbox_t is not None and image_sizes is not None:
            pred_bbox = project_bbox(pred_vertices, c2w_t, intr_t, image_sizes)
            bbox_l1_px = torch.abs(pred_bbox - bbox_t).mean().item()

    smpl_src = Path(args.hugs_dir) / "4d_humans" / "smpl_optimized_aligned_scale.npz"
    src_npz = np.load(smpl_src) if smpl_src.is_file() else None
    if src_npz is not None and "bbox" in src_npz.files:
        bbox_out = src_npz["bbox"][:num_frames].astype(np.float32)
    else:
        bbox_out = np.zeros((num_frames, 5), dtype=np.float32)
        bbox_out[:, 4] = 1.0

    out_smpl_path = out_dir / "4d_humans" / "smpl_optimized_aligned_scale.npz"
    np.savez(
        out_smpl_path,
        global_orient=global_orient.detach().cpu().numpy().astype(np.float32),
        body_pose=body_pose.detach().cpu().numpy().astype(np.float32),
        betas=betas.detach().cpu().numpy()[None].repeat(num_frames, axis=0).astype(np.float32),
        transl=transl.detach().cpu().numpy().astype(np.float32),
        scale=all_scale.detach().cpu().numpy().astype(np.float32),
        bbox=bbox_out,
    )

    report = {
        "source_human3r_result_dir": str(Path(args.human3r_result_dir).resolve()),
        "source_hugs_dir": str(Path(args.hugs_dir).resolve()),
        "out_dir": str(out_dir.resolve()),
        "num_frames": num_frames,
        "smpl_params_frame": args.smpl_params_frame,
        "person_index": args.person_index,
        "smplx_model_dir": str(smplx_model_dir.resolve()),
        "smplx2smpl_path": str(Path(smplx2smpl_path).resolve()),
        "smpl_model_dir": str(Path(args.smpl_model_dir).resolve()),
        "last_log": last,
        "final_vertex_l1_m": full_vertex_l1,
        "final_joint_l1_m": full_joint_l1,
        "final_bbox_l1_px": bbox_l1_px,
        "per_frame_scale": args.per_frame_scale,
    }
    with open(out_dir / "4d_humans" / "fit_smpl_to_human3r_smplx_report.json", "w") as f:
        json.dump(report, f, indent=2)

    metadata_path = out_dir / "metadata.json"
    if metadata_path.is_file():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}
    metadata["smplx_to_smpl_mapping"] = "fitted: Human3R head-centered SMPL-X world vertices -> SMPL topology -> optimized HUGS SMPL scale/transl/pose/shape"
    metadata["smpl_fit_report"] = "4d_humans/fit_smpl_to_human3r_smplx_report.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote fitted SMPL parameters to {out_smpl_path}")
    print(f"Final vertex L1: {full_vertex_l1:.6f} m")
    print(f"Final joint L1: {full_joint_l1:.6f} m")
    if bbox_l1_px is not None:
        print(f"Final bbox L1: {bbox_l1_px:.3f} px")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human3r-result-dir", required=True, help="Human3R saved output directory with color/depth/conf/camera/smpl")
    parser.add_argument("--hugs-dir", required=True, help="Existing converted HUGS Human3R dataset directory")
    parser.add_argument("--out-dir", required=True, help="Output HUGS dataset directory with fitted SMPL params")
    parser.add_argument("--repo-root", default=".", help="ml-hugs repository root")
    parser.add_argument("--smpl-model-dir", default="data/smpl")
    parser.add_argument("--smplx-model-dir", default="/hdd/u202420081000003/Human3R/src/models")
    parser.add_argument("--smplx2smpl-path", default=None)
    parser.add_argument("--person-index", type=int, default=0)
    parser.add_argument("--smpl-params-frame", choices=("camera", "world"), default="camera")
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--target-chunk-size", type=int, default=8)
    parser.add_argument("--vertex-sample", type=int, default=4096)
    parser.add_argument("--per-frame-scale", action="store_true")
    parser.add_argument("--lr-pose", type=float, default=2e-3)
    parser.add_argument("--lr-transl", type=float, default=5e-3)
    parser.add_argument("--lr-betas", type=float, default=1e-3)
    parser.add_argument("--lr-scale", type=float, default=1e-3)
    parser.add_argument("--lambda-vertex", type=float, default=1.0)
    parser.add_argument("--lambda-joints", type=float, default=0.5)
    parser.add_argument("--lambda-bbox", type=float, default=0.01)
    parser.add_argument("--lambda-pose-prior", type=float, default=1e-4)
    parser.add_argument("--lambda-betas-prior", type=float, default=1e-3)
    parser.add_argument("--lambda-temporal", type=float, default=1e-2)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    fit(parse_args())
