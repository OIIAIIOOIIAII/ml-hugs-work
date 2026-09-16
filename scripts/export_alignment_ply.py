"""
导出各场景 inline_attn_18k 最终结果的人体+场景点云 PLY，用于可视化对齐质量。

输出目录: output/alignment_viz/
每个场景生成一个 <scene>_alignment.ply，颜色编码：
  - 蓝色 (0, 100, 220)  : Scene GS（按 opacity 采样前 200K）
  - 红色 (220, 60, 60)   : Human GS（取 10 帧等间隔，每帧 5K 点，经 SMPL 变换到 world space）
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from scipy.spatial.transform import Rotation


# ─── 场景配置 ──────────────────────────────────────────────────────────────
SCENES = {
    "lab":        "output/human_scene/neuman/lab_vimo/hugs_trimlp/vimo_inline_attn_18k_lab_20260604/2026-06-04_10-40-34/ckpt",
    "bike":       "output/human_scene/neuman/bike_vimo/hugs_trimlp/vimo_inline_attn_18k_bike_f60_20260607/2026-06-08_09-26-55/ckpt",
    "citron":     "output/human_scene/neuman/citron_vimo/hugs_trimlp/vimo_inline_attn_18k_citron_20260607/2026-06-07_06-05-27/ckpt",
    "jogging":    "output/human_scene/neuman/jogging_vimo/hugs_trimlp/vimo_inline_attn_18k_jogging_20260607/2026-06-07_10-47-18/ckpt",
    "parkinglot": "output/human_scene/neuman/parkinglot_vimo/hugs_trimlp/vimo_inline_attn_18k_parkinglot_20260607/2026-06-07_15-16-01/ckpt",
    "seattle":    "output/human_scene/neuman/seattle_vimo/hugs_trimlp/vimo_inline_attn_18k_seattle_20260607/2026-06-07_23-29-42/ckpt",
}

SEQ_MAP = {
    "lab":        "lab_vimo",
    "bike":       "bike_vimo",
    "citron":     "citron_vimo",
    "jogging":    "jogging_vimo",
    "parkinglot": "parkinglot_vimo",
    "seattle":    "seattle_vimo",
}

N_SCENE_POINTS = 200_000   # 从 scene GS 按 opacity 采样的点数
N_HUMAN_FRAMES = 15        # 均匀采样的帧数
N_HUMAN_PER_FRAME = 5_000  # 每帧采样的 human GS 点数


# ─── PLY 写入 ──────────────────────────────────────────────────────────────
def write_ply(path, xyz, rgb):
    """xyz: (N,3) float32, rgb: (N,3) uint8"""
    N = xyz.shape[0]
    with open(path, 'wb') as f:
        header = (
            f"ply\nformat binary_little_endian 1.0\n"
            f"element vertex {N}\n"
            f"property float x\nproperty float y\nproperty float z\n"
            f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"end_header\n"
        ).encode()
        f.write(header)
        data = np.concatenate([xyz.astype(np.float32), rgb.astype(np.uint8)], axis=1)
        # interleave: 3×float + 3×uint8 = 15 bytes per vertex
        buf = np.zeros(N, dtype=[('x','f4'),('y','f4'),('z','f4'),
                                  ('r','u1'),('g','u1'),('b','u1')])
        buf['x'] = xyz[:, 0]; buf['y'] = xyz[:, 1]; buf['z'] = xyz[:, 2]
        buf['r'] = rgb[:, 0]; buf['g'] = rgb[:, 1]; buf['b'] = rgb[:, 2]
        f.write(buf.tobytes())
    print(f"  -> 写入 {N:,} 点: {path}")


# ─── axis-angle -> rotation matrix ────────────────────────────────────────
def aa_to_rotmat(aa):
    """aa: (3,) numpy -> (3,3)"""
    return Rotation.from_rotvec(aa).as_matrix()


# ─── canonical xyz -> world xyz（近似，无 LBS joint，只用 global_orient + transl + scale）──
def canonical_to_world(xyz_canon, global_orient, transl, smpl_scale):
    """
    xyz_canon: (N,3) numpy
    global_orient: (3,) numpy (axis-angle)
    transl: (3,) numpy
    smpl_scale: scalar
    """
    R = aa_to_rotmat(global_orient)
    # HUGS 中 canonical 坐标系 z 朝上，已归一化到 ~1m 身高
    xyz_w = smpl_scale * (R @ xyz_canon.T).T + transl
    return xyz_w


# ─── 主流程 ────────────────────────────────────────────────────────────────
def process_scene(scene_name, ckpt_dir, seq):
    print(f"\n{'='*60}")
    print(f"处理场景: {scene_name}  (seq={seq})")

    # 1. 加载 scene GS
    scene_path = os.path.join(ckpt_dir, "scene_final.pth")
    scene_ckpt = torch.load(scene_path, map_location='cpu', weights_only=False)
    s_xyz = scene_ckpt['xyz'].detach().numpy()           # (N, 3)
    s_opacity = torch.sigmoid(scene_ckpt['opacity'].detach()).squeeze().numpy()  # (N,)

    # 用 SfM 初始点云范围过滤爆炸 floater GS（margin=3x）
    from hugs.datasets.neuman import NeumanDataset as _DS
    _ds_tmp = _DS(seq=seq, split='train')
    pcd_pts = np.array(_ds_tmp.init_pcd.points)
    pcd_min = pcd_pts.min(0)
    pcd_max = pcd_pts.max(0)
    margin = np.maximum((pcd_max - pcd_min) * 0.5, 10.0)  # 50% 余量，至少 10m
    valid_mask = np.all((s_xyz >= pcd_min - margin) & (s_xyz <= pcd_max + margin), axis=1)
    n_raw = len(s_xyz)
    s_xyz = s_xyz[valid_mask]
    s_opacity = s_opacity[valid_mask]
    print(f"  SfM bbox: X={pcd_min[0]:.1f}~{pcd_max[0]:.1f}  Y={pcd_min[1]:.1f}~{pcd_max[1]:.1f}  Z={pcd_min[2]:.1f}~{pcd_max[2]:.1f}")
    print(f"  Scene GS: {n_raw:,} -> 过滤后 {len(s_xyz):,} (去掉 {n_raw-len(s_xyz):,} 个 floater)")

    # 按 opacity 降序采样
    n_scene = min(N_SCENE_POINTS, len(s_xyz))
    top_idx = np.argsort(s_opacity)[::-1][:n_scene]
    s_xyz_sampled = s_xyz[top_idx]
    print(f"  采样 {n_scene:,}  bbox X={s_xyz_sampled[:,0].min():.1f}~{s_xyz_sampled[:,0].max():.1f}  "
          f"Y={s_xyz_sampled[:,1].min():.1f}~{s_xyz_sampled[:,1].max():.1f}  "
          f"Z={s_xyz_sampled[:,2].min():.1f}~{s_xyz_sampled[:,2].max():.1f}")

    # 2. 加载 human GS canonical xyz
    human_path = os.path.join(ckpt_dir, "human_final.pth")
    human_ckpt = torch.load(human_path, map_location='cpu', weights_only=False)
    h_xyz_canon = human_ckpt['xyz'].detach().numpy()     # (M, 3) canonical space
    n_human_total = len(h_xyz_canon)
    print(f"  Human GS canonical: {n_human_total:,}")

    # 3. 加载 dataset，获取每帧 SMPL 参数
    from hugs.datasets.neuman import NeumanDataset
    ds = NeumanDataset(seq=seq, split='train')
    n_frames = len(ds)
    frame_indices = np.linspace(0, n_frames - 1, N_HUMAN_FRAMES, dtype=int)
    print(f"  Dataset 帧数: {n_frames}, 采样帧: {frame_indices.tolist()}")

    # 4. 对每个采样帧，把 human GS 变换到 world space
    rng = np.random.default_rng(42)
    per_frame_sample = rng.choice(n_human_total, size=min(N_HUMAN_PER_FRAME, n_human_total), replace=False)
    h_xyz_canon_sub = h_xyz_canon[per_frame_sample]  # (K, 3)

    all_h_xyz = []
    transl_trajectory = []

    for fi in frame_indices:
        sample = ds[int(fi)]
        global_orient = sample['global_orient'].cpu().numpy()  # (3,)
        transl = sample['transl'].cpu().numpy()                # (3,)
        smpl_scale = float(sample['smpl_scale'].cpu())

        h_xyz_w = canonical_to_world(h_xyz_canon_sub, global_orient, transl, smpl_scale)
        all_h_xyz.append(h_xyz_w)
        transl_trajectory.append(transl)

    h_xyz_all = np.concatenate(all_h_xyz, axis=0)   # (K*N_frames, 3)
    traj = np.array(transl_trajectory)               # (N_frames, 3)
    print(f"  Human world 坐标范围: X={h_xyz_all[:,0].min():.2f}~{h_xyz_all[:,0].max():.2f}  "
          f"Y={h_xyz_all[:,1].min():.2f}~{h_xyz_all[:,1].max():.2f}  "
          f"Z={h_xyz_all[:,2].min():.2f}~{h_xyz_all[:,2].max():.2f}")
    print(f"  Human transl 轨迹范围: X={traj[:,0].min():.2f}~{traj[:,0].max():.2f}  "
          f"Y={traj[:,1].min():.2f}~{traj[:,1].max():.2f}  "
          f"Z={traj[:,2].min():.2f}~{traj[:,2].max():.2f}")

    # 5. 合并并染色
    n_s = len(s_xyz_sampled)
    n_h = len(h_xyz_all)
    xyz_all = np.concatenate([s_xyz_sampled, h_xyz_all], axis=0)
    rgb_all = np.zeros((n_s + n_h, 3), dtype=np.uint8)
    rgb_all[:n_s] = [0, 100, 220]      # 场景: 蓝色
    rgb_all[n_s:] = [220, 60, 60]      # 人体: 红色

    # 6. 输出 PLY
    out_dir = "output/alignment_viz"
    os.makedirs(out_dir, exist_ok=True)
    ply_path = os.path.join(out_dir, f"{scene_name}_alignment.ply")
    write_ply(ply_path, xyz_all, rgb_all)

    # 也单独输出轨迹 PLY（每个轨迹点用黄色小球代替，用大量重复点模拟）
    # 用轨迹点重复 200 次让它更显眼
    traj_rep = np.repeat(traj, 200, axis=0)
    traj_rgb = np.full((len(traj_rep), 3), [255, 220, 0], dtype=np.uint8)  # 黄色
    xyz_with_traj = np.concatenate([s_xyz_sampled, h_xyz_all, traj_rep], axis=0)
    rgb_with_traj = np.concatenate([rgb_all, traj_rgb], axis=0)
    ply_traj_path = os.path.join(out_dir, f"{scene_name}_alignment_traj.ply")
    write_ply(ply_traj_path, xyz_with_traj, rgb_with_traj)

    return {
        "scene": scene_name,
        "n_scene_gs": len(s_xyz),
        "n_human_gs": n_human_total,
        "n_frames": n_frames,
        "scene_bbox": (s_xyz.min(0).tolist(), s_xyz.max(0).tolist()),
        "human_world_center": h_xyz_all.mean(0).tolist(),
        "transl_range": (traj.min(0).tolist(), traj.max(0).tolist()),
    }


def main():
    results = []
    for scene_name, ckpt_dir in SCENES.items():
        if not os.path.exists(os.path.join(ckpt_dir, "scene_final.pth")):
            print(f"跳过 {scene_name}: checkpoint 不存在 ({ckpt_dir})")
            continue
        try:
            info = process_scene(scene_name, ckpt_dir, SEQ_MAP[scene_name])
            results.append(info)
        except Exception as e:
            print(f"  !! {scene_name} 失败: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*60}")
    print("汇总:")
    for r in results:
        print(f"  {r['scene']:12s}: scene={r['n_scene_gs']:>8,}  human={r['n_human_gs']:>7,}  frames={r['n_frames']}")
        print(f"              human中心世界坐标: {[f'{v:.2f}' for v in r['human_world_center']]}")
    print(f"\nPLY 文件输出到: output/alignment_viz/")
    print("蓝色=Scene GS, 红色=Human GS(多帧叠加), 黄色=人体重心轨迹")
    print("推荐用 MeshLab 或 CloudCompare 打开（可调整点大小）")


if __name__ == '__main__':
    main()
