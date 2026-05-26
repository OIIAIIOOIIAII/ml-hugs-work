import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(".")
from hugs.datasets.neuman import NeumanDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="lab")
    parser.add_argument("--init-pcd", default=None)
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--mask-human", action="store_true")
    args = parser.parse_args()

    dataset = NeumanDataset(
        args.seq,
        "train",
        render_mode="human_scene",
        init_pcd_path=args.init_pcd,
        depth_prior_dir=args.depth_dir,
    )
    points = np.asarray(dataset.init_pcd.points, dtype=np.float32)
    if points.shape[0] > args.max_points:
        rng = np.random.default_rng(0)
        points = points[rng.choice(points.shape[0], args.max_points, replace=False)]

    xyz = torch.from_numpy(points).float().cuda()
    ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
    homog = torch.cat([xyz, ones], dim=1)

    ratios = []
    offsets = []
    rows = []
    for i in range(min(args.max_frames, len(dataset))):
        data = dataset[i]
        frame_idx = int(data["frame_idx"].item())
        depth = data["depth_prior"]
        valid = torch.isfinite(depth) & (depth > 0)
        if args.mask_human and "mask" in data:
            valid = valid & (data["mask"] < 0.5)

        cam = homog @ data["world_view_transform"]
        z = cam[:, 2]
        k = data["cam_intrinsics"]
        u = torch.round(k[0, 0] * (cam[:, 0] / torch.clamp(z, min=1e-6)) + k[0, 2]).long()
        v = torch.round(k[1, 1] * (cam[:, 1] / torch.clamp(z, min=1e-6)) + k[1, 2]).long()
        h, w = depth.shape[-2], depth.shape[-1]
        inside = (z > 1e-4) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not inside.any():
            continue
        u_in, v_in, z_in = u[inside], v[inside], z[inside]
        d_in = depth[v_in, u_in]
        vd = torch.isfinite(d_in) & (d_in > 0) & valid[v_in, u_in]
        if vd.sum().item() < 100:
            continue
        z_np = z_in[vd].detach().cpu().numpy()
        d_np = d_in[vd].detach().cpu().numpy()
        ratio = float(np.median(z_np / d_np))
        offset = float(np.median(z_np - d_np))
        ratios.append(ratio)
        offsets.append(offset)
        rows.append((frame_idx, int(vd.sum().item()), float(np.median(z_np)), float(np.median(d_np)), ratio, offset))

    print("frame matched median_z median_depth median_z_over_depth median_z_minus_depth")
    for r in rows:
        print("%05d %d %.6f %.6f %.6f %.6f" % r)
    if ratios:
        print("global_ratio_median %.8f" % float(np.median(ratios)))
        print("global_offset_median %.8f" % float(np.median(offsets)))


if __name__ == "__main__":
    main()
