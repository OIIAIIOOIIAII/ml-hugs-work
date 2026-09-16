"""
方案 A：离线逐帧预对齐
对每帧用 2D mask 质心对齐 SMPL 顶点均值投影，修正 XY 方向系统偏差。

【关键设计】：
  - 对齐目标：SMPL 所有顶点均值在图像中的投影 → mask 质心
  - 不是 transl（pelvis 位置），pelvis 与 mask 质心有 ~257px 固有解剖学偏差
  - 顶点均值世界坐标 = verts_local_mean * scale + transl
  - 修正量 delta_world 直接加到 transl（平移整体人体）

输出：修正后的 smpl_optimized_aligned_scale.npz（保存为 _prealigned.npz）

用法：
    cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work
    conda run -n hugs python scripts/offline_mask_prealign.py \
        --seq lab_vimo [--n_iters 3] [--output_suffix _prealigned]
"""
import argparse
import sys
import os
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from hugs.datasets.neuman_utils import neuman_helper
from hugs.models.modules.smpl_layer import SMPL


def compute_mask_centroid(mask_path):
    """从 mask 图像计算 2D 质心（像素坐标）"""
    msk = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if msk is None:
        return None
    binary = (msk > 128).astype(np.uint8)
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())  # (u, v)


def get_body_center_world(smpl_model, global_orient_i, body_pose_i, betas_i, transl_i, scale_i):
    """
    计算第 i 帧人体顶点均值在世界坐标系中的位置。
    世界坐标 = verts_local * scale + transl
    """
    with torch.no_grad():
        output = smpl_model(
            global_orient=torch.tensor(global_orient_i[None], dtype=torch.float32),
            body_pose=torch.tensor(body_pose_i[None], dtype=torch.float32),
            betas=torch.tensor(betas_i[None], dtype=torch.float32),
            transl=torch.zeros(1, 3),  # 不在 SMPL 内加 transl，后面手动加
        )
    verts_local = output.vertices[0].numpy()  # (6890, 3)，SMPL local space
    verts_mean_local = verts_local.mean(axis=0)  # (3,)
    body_center_world = verts_mean_local * scale_i + transl_i
    return body_center_world


def project_point(K, w2c, point_world):
    """将世界坐标点投影到图像平面，返回 (u, v, depth)"""
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    p_cam = R @ point_world + t
    depth = p_cam[2]
    if depth <= 0:
        return None, None, depth
    p_2d = K @ p_cam
    u = p_2d[0] / p_2d[2]
    v = p_2d[1] / p_2d[2]
    return u, v, depth


def compute_correction(K, w2c, ref_point_world, cx_mask, cy_mask):
    """
    计算让 proj(ref_point + delta_world) = (cx_mask, cy_mask) 的 3D 修正量。
    只修正 XY（camera 坐标系横纵方向），不改变深度。
    delta_world 直接加到 transl（人体整体平移）。
    """
    u_proj, v_proj, depth = project_point(K, w2c, ref_point_world)
    if u_proj is None:
        return np.zeros(3)

    fx, fy = K[0, 0], K[1, 1]
    delta_u = cx_mask - u_proj
    delta_v = cy_mask - v_proj

    # 修正量在 camera 坐标系中（只修正 XY，Z=0）
    delta_cam = np.array([delta_u * depth / fx, delta_v * depth / fy, 0.0])

    # 转回世界坐标系
    R = w2c[:3, :3]
    delta_world = R.T @ delta_cam
    return delta_world, (u_proj, v_proj)


def prealign(seq, n_iters=3, output_suffix='_prealigned', dataset_root='data/neuman/dataset',
             smpl_model_path='data/smpl'):
    dataset_path = f'{dataset_root}/{seq}'
    smpl_path = f'{dataset_path}/4d_humans/smpl_optimized_aligned_scale.npz'
    mask_dir = Path(f'{dataset_path}/segmentations')
    output_path = smpl_path.replace('.npz', f'{output_suffix}.npz')

    print(f'[prealign] seq={seq}, n_iters={n_iters}')
    print(f'  smpl_path: {smpl_path}')
    print(f'  mask_dir:  {mask_dir}')
    print(f'  output:    {output_path}')

    # 加载 SMPL 模型（用于计算顶点均值）
    print(f'[prealign] loading SMPL model from {smpl_model_path}...')
    smpl_model = SMPL(smpl_model_path)
    smpl_model.eval()

    # 加载场景相机参数
    print('[prealign] loading scene cameras...')
    scene = neuman_helper.NeuManReader.read_scene(
        dataset_path, tgt_size=None, normalize=False, smpl_type='optimized'
    )
    captures = scene.captures
    n_frames = len(captures)
    print(f'[prealign] {n_frames} frames')

    # 加载 SMPL params
    smpl_data = np.load(smpl_path)
    smpl_dict = {k: smpl_data[k].copy() for k in smpl_data.files}
    transl = smpl_dict['transl'].copy()  # (N, 3)
    global_orient = smpl_dict['global_orient']  # (N, 3)
    body_pose = smpl_dict['body_pose']  # (N, 69)
    betas = smpl_dict['betas']  # (N, 10)
    scale = smpl_dict['scale']  # (N,)

    # 找 mask 文件列表（支持两种命名方式）
    mask_files_seg = sorted(mask_dir.glob('*.png'))
    mask_files_sam = sorted(Path(f'{dataset_path}/4d_humans/sam_segmentations').glob('mask_*.png'))
    if len(mask_files_seg) == n_frames:
        mask_files = mask_files_seg
        print(f'[prealign] using segmentations/ ({len(mask_files)} files)')
    elif len(mask_files_sam) == n_frames:
        mask_files = mask_files_sam
        print(f'[prealign] using sam_segmentations/ ({len(mask_files)} files)')
    else:
        print(f'[prealign] WARNING: mask count mismatch: seg={len(mask_files_seg)}, sam={len(mask_files_sam)}, frames={n_frames}')
        mask_files = mask_files_seg

    # 诊断：帧0的初始对齐状态
    print('\n[prealign] === 初始对齐诊断（帧0）===')
    cap0 = captures[0]
    K0 = cap0.intrinsic_matrix
    w2c0 = cap0.cam_pose.world_to_camera
    centroid0 = compute_mask_centroid(mask_files[0])
    if centroid0 is not None:
        cx0, cy0 = centroid0
        # 顶点均值投影
        body_center0 = get_body_center_world(
            smpl_model, global_orient[0], body_pose[0], betas[0], transl[0], scale[0])
        u_body, v_body, d_body = project_point(K0, w2c0, body_center0)
        # pelvis（transl）投影
        u_pelvis, v_pelvis, d_pelvis = project_point(K0, w2c0, transl[0])
        print(f'  mask centroid:     ({cx0:.1f}, {cy0:.1f})')
        print(f'  pelvis proj:       ({u_pelvis:.1f}, {v_pelvis:.1f})  offset={np.sqrt((cx0-u_pelvis)**2+(cy0-v_pelvis)**2):.1f}px')
        print(f'  body_center proj:  ({u_body:.1f}, {v_body:.1f})  offset={np.sqrt((cx0-u_body)**2+(cy0-v_body)**2):.1f}px')
        print(f'  body_center depth: {d_body:.3f} COLMAP units')
    print()

    # 迭代修正
    total_errors_before = []
    total_errors_after = []

    for it in range(n_iters):
        errors_px = []
        corrections = []

        for i, cap in enumerate(tqdm(captures, desc=f'iter {it+1}/{n_iters}')):
            K = cap.intrinsic_matrix
            w2c = cap.cam_pose.world_to_camera

            # mask 质心
            centroid = compute_mask_centroid(mask_files[i])
            if centroid is None:
                corrections.append(np.zeros(3))
                continue
            cx_mask, cy_mask = centroid

            # 当前人体中心（顶点均值）投影
            body_center = get_body_center_world(
                smpl_model, global_orient[i], body_pose[i], betas[i], transl[i], scale[i])
            u_proj, v_proj, depth = project_point(K, w2c, body_center)
            if u_proj is not None:
                err_px = np.sqrt((cx_mask - u_proj)**2 + (cy_mask - v_proj)**2)
                errors_px.append(err_px)

            # 计算修正量（delta_world 加到 transl）
            result = compute_correction(K, w2c, body_center, cx_mask, cy_mask)
            if isinstance(result, tuple):
                delta, _ = result
            else:
                delta = result
            corrections.append(delta)

        if it == 0:
            total_errors_before = errors_px.copy()
            print(f'  [before] mean proj error: {np.mean(errors_px):.1f} px, '
                  f'max: {np.max(errors_px):.1f} px')

        # 应用修正
        for i, delta in enumerate(corrections):
            transl[i] += delta

    # 计算修正后投影误差
    for i, cap in enumerate(captures):
        K = cap.intrinsic_matrix
        w2c = cap.cam_pose.world_to_camera
        centroid = compute_mask_centroid(mask_files[i])
        if centroid is None:
            continue
        cx_mask, cy_mask = centroid
        body_center = get_body_center_world(
            smpl_model, global_orient[i], body_pose[i], betas[i], transl[i], scale[i])
        u_proj, v_proj, _ = project_point(K, w2c, body_center)
        if u_proj is not None:
            total_errors_after.append(np.sqrt((cx_mask - u_proj)**2 + (cy_mask - v_proj)**2))

    print(f'  [after]  mean proj error: {np.mean(total_errors_after):.2f} px, '
          f'max: {np.max(total_errors_after):.2f} px')
    print(f'  improvement: {np.mean(total_errors_before):.1f} → {np.mean(total_errors_after):.2f} px')

    # 修正量统计
    orig_transl = smpl_data['transl']
    delta_transl = transl - orig_transl
    print(f'  delta_transl mean: {delta_transl.mean(axis=0)}')
    print(f'  delta_transl std:  {delta_transl.std(axis=0)}')
    print(f'  delta_transl norm mean: {np.linalg.norm(delta_transl, axis=1).mean():.4f} COLMAP units')

    # 保存
    smpl_dict['transl'] = transl
    np.savez(output_path, **smpl_dict)
    print(f'[prealign] saved to {output_path}')
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', default='lab_vimo')
    parser.add_argument('--n_iters', type=int, default=3)
    parser.add_argument('--output_suffix', default='_prealigned')
    parser.add_argument('--dataset_root', default='data/neuman/dataset')
    parser.add_argument('--smpl_model_path', default='data/smpl')
    args = parser.parse_args()

    os.chdir(str(Path(__file__).parent.parent))
    prealign(args.seq, args.n_iters, args.output_suffix, args.dataset_root, args.smpl_model_path)
