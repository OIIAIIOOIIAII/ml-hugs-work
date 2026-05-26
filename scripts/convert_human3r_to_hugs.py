#
# Convert saved Human3R inference outputs into a HUGS-readable dataset.
#

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def sorted_stems(folder, suffix):
    return [p.stem for p in sorted(Path(folder).glob(f"*{suffix}"))]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_rgb(path, img):
    ensure_dir(Path(path).parent)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def pick_person(smpl_npz, person_index):
    n_people = int(smpl_npz["shape"].shape[0])
    if n_people == 0:
        raise ValueError("Human3R SMPL file contains zero people")
    if person_index >= n_people:
        raise IndexError(f"person_index={person_index} but only {n_people} people found")
    return person_index


def transform_axis_angle(axis_angle, rotmat):
    aa = axis_angle.astype(np.float32)
    local_rot, _ = cv2.Rodrigues(aa)
    world_rot = rotmat.astype(np.float32) @ local_rot.astype(np.float32)
    world_aa, _ = cv2.Rodrigues(world_rot)
    return world_aa.reshape(3).astype(np.float32)


def smplx_to_smpl_fields(smpl_npz, person_index, c2w=None, params_frame="camera"):
    idx = pick_person(smpl_npz, person_index)
    rotvec = smpl_npz["rotvec"][idx].astype(np.float32)
    if rotvec.shape[0] < 24:
        raise ValueError(f"Expected at least 24 SMPL-X rotations, got {rotvec.shape}")

    global_orient = rotvec[0].astype(np.float32)
    transl = smpl_npz["transl"][idx].astype(np.float32)

    if params_frame == "camera":
        if c2w is None:
            raise ValueError("c2w is required when params_frame='camera'")
        global_orient = transform_axis_angle(global_orient, c2w[:3, :3])
        transl = (c2w[:3, :3] @ transl + c2w[:3, 3]).astype(np.float32)
    elif params_frame != "world":
        raise ValueError(f"Unknown params_frame: {params_frame}")

    return {
        "global_orient": global_orient,
        "body_pose": rotvec[1:24].reshape(69).astype(np.float32),
        "betas": smpl_npz["shape"][idx, :10].astype(np.float32),
        "transl": transl,
        "scale": np.float32(1.0),
    }


def human_mask_from_smpl(smpl_npz, height, width, score_threshold, mask_threshold):
    mask = None
    if "msk" in smpl_npz.files and smpl_npz["msk"] is not None:
        msk = smpl_npz["msk"]
        if not (msk.dtype == object and msk.shape == () and msk.item() is None):
            if msk.ndim == 3:
                msk = msk[0]
            mask = msk.astype(np.float32) >= mask_threshold

    if mask is None and "scores" in smpl_npz.files:
        scores = smpl_npz["scores"].astype(np.float32)
        if scores.ndim == 3:
            scores = scores.max(axis=0)
        mask = scores >= score_threshold

    if mask is None:
        mask = np.zeros((height, width), dtype=bool)

    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def resize_crop_like_human3r(img, target_height, target_width, interpolation):
    """Resize long edge to Human3R output width/height and center-crop to target size."""
    in_height, in_width = img.shape[:2]
    long_edge = max(target_height, target_width)
    scale = long_edge / max(in_height, in_width)
    resized_width = int(round(in_width * scale))
    resized_height = int(round(in_height * scale))
    resized = cv2.resize(img, (resized_width, resized_height), interpolation=interpolation)

    y0 = max((resized_height - target_height) // 2, 0)
    x0 = max((resized_width - target_width) // 2, 0)
    cropped = resized[y0 : y0 + target_height, x0 : x0 + target_width]
    if cropped.shape[:2] != (target_height, target_width):
        cropped = cv2.resize(cropped, (target_width, target_height), interpolation=interpolation)
    return cropped


def load_external_mask(mask_dir, frame_index, height, width, pattern):
    if not mask_dir:
        return None
    mask_path = Path(mask_dir) / pattern.format(idx=frame_index)
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    mask = resize_crop_like_human3r(mask, height, width, cv2.INTER_NEAREST)
    return mask > 127


def keep_largest_connected_component(mask):
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return mask
    keep_label = 1 + int(np.argmax(areas))
    return labels == keep_label


def bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        h, w = mask.shape
        return np.array([0, 0, h - 1, w - 1, 0], dtype=np.float32)
    xmin, xmax = ys.min(), ys.max()
    ymin, ymax = xs.min(), xs.max()
    return np.array([xmin, ymin, xmax, ymax, 1], dtype=np.float32)


def backproject_points(depth, conf, rgb, intrinsics, c2w, valid_mask, stride):
    h, w = depth.shape
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[yy, xx]
    conf_s = conf[yy, xx]
    valid = valid_mask[yy, xx] & np.isfinite(z) & np.isfinite(conf_s) & (z > 0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    x = xx[valid].astype(np.float32)
    y = yy[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x_cam = (x - cx) * z / fx
    y_cam = (y - cy) * z / fy
    p_cam = np.stack([x_cam, y_cam, z], axis=-1)
    p_world = p_cam @ c2w[:3, :3].T + c2w[:3, 3]

    colors = rgb[yy[valid], xx[valid]].astype(np.float32) / 255.0
    return p_world.astype(np.float32), colors.astype(np.float32)


def write_ascii_ply(path, points, colors):
    ensure_dir(Path(path).parent)
    colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors_u8):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def convert(args):
    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir)

    required = ["color", "depth", "conf", "camera", "smpl"]
    for name in required:
        if not (result_dir / name).is_dir():
            raise FileNotFoundError(f"Missing Human3R output subdir: {result_dir / name}")

    stems = sorted_stems(result_dir / "color", ".png")
    if args.max_frames > 0:
        stems = stems[: args.max_frames]
    if len(stems) == 0:
        raise RuntimeError(f"No color PNG frames found under {result_dir / 'color'}")

    ensure_dir(out_dir / "images")
    ensure_dir(out_dir / "masks")
    ensure_dir(out_dir / "4d_humans")
    ensure_dir(out_dir / "scene")

    global_orients, body_poses, betas, transls, scales, bboxes = [], [], [], [], [], []
    c2ws, intrinsics, image_sizes = [], [], []
    all_points, all_colors = [], []
    frame_names = []

    for out_idx, stem in enumerate(stems):
        color_path = result_dir / "color" / f"{stem}.png"
        depth_path = result_dir / "depth" / f"{stem}.npy"
        conf_path = result_dir / "conf" / f"{stem}.npy"
        camera_path = result_dir / "camera" / f"{stem}.npz"
        smpl_path = result_dir / "smpl" / f"{stem}.npz"

        rgb = load_rgb(color_path)
        depth = np.load(depth_path).astype(np.float32)
        conf = np.load(conf_path).astype(np.float32)
        camera = np.load(camera_path)
        smpl_npz = np.load(smpl_path, allow_pickle=True)

        h, w = rgb.shape[:2]
        c2w = camera["pose"].astype(np.float32)
        k = camera["intrinsics"].astype(np.float32)

        fields = smplx_to_smpl_fields(
            smpl_npz,
            args.person_index,
            c2w=c2w,
            params_frame=args.smpl_params_frame,
        )
        external_mask = load_external_mask(args.external_mask_dir, out_idx, h, w, args.external_mask_pattern)
        if external_mask is not None:
            human_mask = external_mask
        else:
            human_mask = human_mask_from_smpl(
                smpl_npz,
                height=h,
                width=w,
                score_threshold=args.score_threshold,
                mask_threshold=args.mask_threshold,
            )

        if args.mask_largest_cc:
            human_mask = keep_largest_connected_component(human_mask)

        if args.dilate_mask > 0:
            kernel = np.ones((args.dilate_mask, args.dilate_mask), dtype=np.uint8)
            human_mask_for_scene = cv2.dilate(human_mask.astype(np.uint8), kernel, iterations=1) > 0
        else:
            human_mask_for_scene = human_mask

        scene_valid = np.isfinite(depth) & np.isfinite(conf) & (depth > 0) & (conf >= args.conf_threshold)
        if args.exclude_human_from_scene:
            scene_valid = scene_valid & (~human_mask_for_scene)

        pts, cols = backproject_points(depth, conf, rgb, k, c2w, scene_valid, args.point_stride)
        if pts.shape[0] > 0:
            all_points.append(pts)
            all_colors.append(cols)

        frame_name = f"{out_idx:06d}.png"
        write_rgb(out_dir / "images" / frame_name, rgb)
        cv2.imwrite(str(out_dir / "masks" / frame_name), (human_mask.astype(np.uint8) * 255))

        global_orients.append(fields["global_orient"])
        body_poses.append(fields["body_pose"])
        betas.append(fields["betas"])
        transls.append(fields["transl"])
        scales.append(fields["scale"])
        bboxes.append(bbox_from_mask(human_mask))
        c2ws.append(c2w)
        intrinsics.append(k)
        image_sizes.append(np.array([h, w], dtype=np.int32))
        frame_names.append(frame_name)

    if all_points:
        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
    else:
        points = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.float32)

    if args.max_scene_points > 0 and points.shape[0] > args.max_scene_points:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(points.shape[0], size=args.max_scene_points, replace=False)
        points = points[keep]
        colors = colors[keep]

    np.savez_compressed(
        out_dir / "scene" / "points.npz",
        points=points.astype(np.float32),
        colors=colors.astype(np.float32),
    )
    write_ascii_ply(out_dir / "scene" / "points.ply", points, colors)

    np.savez(
        out_dir / "cameras.npz",
        c2w=np.stack(c2ws).astype(np.float32),
        intrinsics=np.stack(intrinsics).astype(np.float32),
        image_sizes=np.stack(image_sizes).astype(np.int32),
        frame_names=np.array(frame_names),
    )
    np.savez(
        out_dir / "4d_humans" / "smpl_optimized_aligned_scale.npz",
        global_orient=np.stack(global_orients).astype(np.float32),
        body_pose=np.stack(body_poses).astype(np.float32),
        betas=np.stack(betas).astype(np.float32),
        transl=np.stack(transls).astype(np.float32),
        scale=np.array(scales, dtype=np.float32),
        bbox=np.stack(bboxes).astype(np.float32),
    )

    metadata = {
        "source_result_dir": str(result_dir),
        "num_frames": len(stems),
        "num_scene_points": int(points.shape[0]),
        "person_index": args.person_index,
        "conf_threshold": args.conf_threshold,
        "point_stride": args.point_stride,
        "exclude_human_from_scene": args.exclude_human_from_scene,
        "external_mask_dir": args.external_mask_dir,
        "external_mask_pattern": args.external_mask_pattern if args.external_mask_dir else None,
        "mask_largest_cc": args.mask_largest_cc,
        "smpl_params_frame": args.smpl_params_frame,
        "smplx_to_smpl_mapping": "approximate: body_pose=rotvec[1:24], betas=shape[:10], scale=1; if smpl_params_frame=camera, global_orient and transl are transformed by camera c2w",
        "warning": "Validate SMPL projection before full training; Human3R transl/rotvec conventions may require an additional alignment.",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote HUGS Human3R dataset to {out_dir}")
    print(f"Frames: {len(stems)}")
    print(f"Scene points: {points.shape[0]}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, help="Human3R --save output directory")
    parser.add_argument("--out-dir", required=True, help="Output HUGS-compatible dataset directory")
    parser.add_argument("--person-index", type=int, default=0)
    parser.add_argument("--conf-threshold", type=float, default=1.5)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--dilate-mask", type=int, default=15)
    parser.add_argument("--mask-largest-cc", action="store_true", help="Keep only the largest connected component in the human mask before bbox/scene exclusion.")
    parser.add_argument("--point-stride", type=int, default=2)
    parser.add_argument("--max-scene-points", type=int, default=500000)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-human-in-scene", action="store_true")
    parser.add_argument(
        "--external-mask-dir",
        default=None,
        help="Optional mask directory to replace Human3R masks, resized/cropped to Human3R output resolution.",
    )
    parser.add_argument(
        "--external-mask-pattern",
        default="mask_{idx:04d}.png",
        help="Python format pattern for external masks. Available field: idx.",
    )
    parser.add_argument(
        "--smpl-params-frame",
        choices=("camera", "world"),
        default="camera",
        help="Coordinate frame of Human3R smpl/*.npz global_orient/transl. Human3R saved transl is usually camera-frame.",
    )
    args = parser.parse_args()
    args.exclude_human_from_scene = not args.include_human_in_scene
    return args


if __name__ == "__main__":
    convert(parse_args())
