"""
计算 VIMO 粗对齐质量量化指标，逐场景输出汇总表。

指标：
  Mask_IoU     : SMPL 投影 silhouette 与 GT segmentation mask 的 IoU（越高越好）
  KP_err_px    : SMPL 关节 2D 重投影误差（MPJPE-2D，像素，越低越好）
  KP_err_norm  : 上述误差除以人体 bbox 对角线归一化（越低越好）
  Transl_jitter: 相邻帧 transl 差的均值（m/frame，越低越好，反映轨迹平滑度）
  Scale_std    : smpl_scale 跨帧标准差（越低越好）
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# COCO-17 关节到 SMPL-45 关节的映射（SMPL output joints 索引）
# SMPL joints: 0=pelvis,1=L_Hip,2=R_Hip,3=Spine1,4=L_Knee,5=R_Knee,
# 6=Spine2,7=L_Ankle,8=R_Ankle,9=Spine3,10=L_Foot,11=R_Foot,
# 12=Neck,13=L_Collar,14=R_Collar,15=Head,16=L_Shoulder,17=R_Shoulder,
# 18=L_Elbow,19=R_Elbow,20=L_Wrist,21=R_Wrist,22=L_Hand,23=R_Hand
# COCO-17: 0=nose,1=L_eye,2=R_eye,3=L_ear,4=R_ear,
# 5=L_shldr,6=R_shldr,7=L_elbow,8=R_elbow,9=L_wrist,10=R_wrist,
# 11=L_hip,12=R_hip,13=L_knee,14=R_knee,15=L_ankle,16=R_ankle
COCO_TO_SMPL = {
    5:  16,   # L_shoulder
    6:  17,   # R_shoulder
    7:  18,   # L_elbow
    8:  19,   # R_elbow
    9:  20,   # L_wrist
    10: 21,   # R_wrist
    11: 1,    # L_hip
    12: 2,    # R_hip
    13: 4,    # L_knee
    14: 5,    # R_knee
    15: 7,    # L_ankle
    16: 8,    # R_ankle
}
COCO_IDXS  = list(COCO_TO_SMPL.keys())    # 12 个有对应关节的 COCO 点
SMPL_IDXS  = [COCO_TO_SMPL[k] for k in COCO_IDXS]

SEQS = {
    "lab":        "lab_vimo",
    "bike":       "bike_vimo",
    "citron":     "citron_vimo",
    "jogging":    "jogging_vimo",
    "parkinglot": "parkinglot_vimo",
    "seattle":    "seattle_vimo",
}


def project_points(pts_world, K, w2c):
    """
    pts_world : (N, 3) numpy, world space
    K         : (3, 3) intrinsics
    w2c       : (4, 4) world-to-camera (extrinsic)
    returns   : (N, 2) pixel coords
    """
    ones = np.ones((pts_world.shape[0], 1))
    pts_h = np.concatenate([pts_world, ones], axis=1)   # (N, 4)
    pts_cam = (w2c @ pts_h.T).T[:, :3]                  # (N, 3)
    pts_img = (K @ pts_cam.T).T                          # (N, 3)
    pts_img = pts_img[:, :2] / pts_img[:, 2:3]          # (N, 2) pixel
    return pts_img


def run_smpl_forward(smpl_model, betas, global_orient, body_pose, transl, scale):
    """返回 world-space joint positions (N_joints, 3)"""
    with torch.no_grad():
        out = smpl_model(
            betas=betas.unsqueeze(0),
            global_orient=global_orient.unsqueeze(0),
            body_pose=body_pose.unsqueeze(0),
            transl=torch.zeros(1, 3, device=betas.device),  # 先不加 transl
        )
    # joints 在 SMPL local space，需加 transl 和 scale
    joints = out.joints[0].cpu().numpy()   # (J, 3)
    # 旋转和 transl 已经在 global_orient 里，但 transl 从参数单独给
    joints_w = joints * scale.cpu().item() + transl.cpu().numpy()
    return joints_w  # (J, 3) world space


def get_smpl_silhouette(smpl_model, sample, img_h, img_w):
    """投影 SMPL vertices 到图像，用凸包填充返回 binary mask"""
    betas     = sample['betas'].cuda()
    g_orient  = sample['global_orient'].cuda()
    body_pose = sample['body_pose'].cuda()
    transl    = sample['transl'].cuda()
    scale     = sample['smpl_scale'].cuda()
    K         = sample['cam_intrinsics'].cpu().numpy()
    w2c       = sample['world_view_transform'].cpu().numpy().T   # inv(c2w)

    with torch.no_grad():
        out = smpl_model(
            betas=betas.unsqueeze(0),
            global_orient=g_orient.unsqueeze(0),
            body_pose=body_pose.unsqueeze(0),
            transl=torch.zeros(1, 3, device=betas.device),
        )
    verts   = out.vertices[0].cpu().numpy()     # (6890, 3)
    verts_w = verts * scale.cpu().item() + transl.cpu().numpy()

    pts2d   = project_points(verts_w, K, w2c)  # (6890, 2)
    pts_int = np.round(pts2d).astype(np.int32)
    valid   = (pts_int[:, 0] >= 0) & (pts_int[:, 0] < img_w) & \
              (pts_int[:, 1] >= 0) & (pts_int[:, 1] < img_h)
    pts_valid = pts_int[valid]

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if len(pts_valid) < 3:
        return mask
    hull = cv2.convexHull(pts_valid)
    cv2.fillPoly(mask, [hull], 1)
    return mask


def compute_scene_metrics(scene_name, seq):
    print(f"\n  [{scene_name}] seq={seq}", flush=True)

    from hugs.datasets.neuman import NeumanDataset
    from hugs.models.modules.smpl_layer import SMPL
    from hugs.cfg.constants import SMPL_PATH

    ds = NeumanDataset(seq=seq, split='train')
    smpl = SMPL(SMPL_PATH).cuda()

    seg_dir = f"data/neuman/dataset/{seq}/segmentations"
    kp_dir  = f"data/neuman/dataset/{seq}/keypoints"
    img_dir = f"data/neuman/dataset/{seq}/images"

    n_frames = len(ds)
    ious, kp_errs, kp_norms = [], [], []
    transls = []

    for fi in range(n_frames):
        sample = ds[fi]
        img_h  = int(sample['image_height'])
        img_w  = int(sample['image_width'])
        K      = sample['cam_intrinsics'].cpu().numpy()
        w2c_gl = sample['world_view_transform'].cpu().numpy()
        w2c    = w2c_gl.T

        # ── GT segmentation mask ──────────────────────────────────────
        seg_files = sorted(f for f in os.listdir(seg_dir) if f.endswith('.png'))
        seg_path  = os.path.join(seg_dir, seg_files[fi % len(seg_files)])
        seg_img   = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        if seg_img is None:
            continue
        gt_mask = (seg_img > 128).astype(np.uint8)

        # 跳过几乎没人的帧
        if gt_mask.sum() < 500:
            continue

        # ── SMPL vertices → silhouette IoU ───────────────────────────
        sil_mask = get_smpl_silhouette(smpl, sample, img_h, img_w)
        inter = (sil_mask & gt_mask).sum()
        union = (sil_mask | gt_mask).sum()
        iou   = inter / (union + 1e-6)
        ious.append(iou)

        # ── SMPL joints → 2D keypoint reprojection error ─────────────
        kp_files = sorted(f for f in os.listdir(kp_dir) if f.endswith('.npy'))
        kp_path  = os.path.join(kp_dir, kp_files[fi % len(kp_files)])
        kp_det   = np.load(kp_path, allow_pickle=True)   # (17, 3): x, y, conf
        conf_thr = 0.3

        betas     = sample['betas'].cuda()
        g_orient  = sample['global_orient'].cuda()
        body_pose = sample['body_pose'].cuda()
        transl_t  = sample['transl'].cuda()
        scale_t   = sample['smpl_scale'].cuda()

        with torch.no_grad():
            out = smpl(
                betas=betas.unsqueeze(0),
                global_orient=g_orient.unsqueeze(0),
                body_pose=body_pose.unsqueeze(0),
                transl=torch.zeros(1, 3, device=betas.device),
            )
        joints_w = out.joints[0].cpu().numpy() * scale_t.cpu().item() \
                   + transl_t.cpu().numpy()   # (J, 3)

        errs = []
        for ci, si in zip(COCO_IDXS, SMPL_IDXS):
            if si >= joints_w.shape[0]:
                continue
            conf = kp_det[ci, 2]
            if conf < conf_thr:
                continue
            proj = project_points(joints_w[[si]], K, w2c)[0]  # (2,)
            gt   = kp_det[ci, :2]
            errs.append(np.linalg.norm(proj - gt))

        if errs:
            # 归一化：用 GT mask bbox 对角线
            ys, xs = np.where(gt_mask)
            bbox_diag = np.sqrt((xs.max()-xs.min())**2 + (ys.max()-ys.min())**2) + 1e-6
            mean_err = np.mean(errs)
            kp_errs.append(mean_err)
            kp_norms.append(mean_err / bbox_diag)

        transls.append(sample['transl'].cpu().numpy())

    # transl jitter
    transls = np.array(transls)
    jitter = np.linalg.norm(np.diff(transls, axis=0), axis=1).mean() if len(transls) > 1 else 0.0

    # scale std
    scales = np.array([ds[fi]['smpl_scale'].item() for fi in range(n_frames)])

    result = {
        "scene":         scene_name,
        "n_frames":      n_frames,
        "mask_iou":      np.mean(ious)    if ious    else float('nan'),
        "kp_err_px":     np.mean(kp_errs) if kp_errs else float('nan'),
        "kp_err_norm":   np.mean(kp_norms)if kp_norms else float('nan'),
        "transl_jitter": jitter,
        "scale_std":     scales.std(),
        "scale_mean":    scales.mean(),
    }
    print(f"    IoU={result['mask_iou']:.3f}  KP={result['kp_err_px']:.1f}px  "
          f"KP_norm={result['kp_err_norm']:.3f}  jitter={result['transl_jitter']:.3f}m/f  "
          f"scale={result['scale_mean']:.2f}±{result['scale_std']:.3f}", flush=True)
    return result


def main():
    results = []
    for scene_name, seq in SEQS.items():
        try:
            r = compute_scene_metrics(scene_name, seq)
            results.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  !! {scene_name} 失败: {e}")

    # 打印汇总表
    print("\n" + "="*90)
    print(f"{'场景':12s} {'帧数':>5} {'Mask_IoU':>9} {'KP_err(px)':>11} {'KP_norm':>8} "
          f"{'Transl_jitter(m/f)':>19} {'Scale(mean±std)':>16}")
    print("-"*90)
    for r in results:
        print(f"{r['scene']:12s} {r['n_frames']:>5d} "
              f"{r['mask_iou']:>9.3f} {r['kp_err_px']:>11.1f} {r['kp_err_norm']:>8.3f} "
              f"{r['transl_jitter']:>19.3f} "
              f"{r['scale_mean']:>7.2f}±{r['scale_std']:.3f}")
    print("="*90)
    print("\n指标说明:")
    print("  Mask_IoU     : SMPL投影silhouette与GT分割mask的IoU（越高=对齐越好）")
    print("  KP_err(px)   : SMPL关节2D重投影误差 vs VitPose检测点，均值（越低越好）")
    print("  KP_norm      : KP_err 除以人体bbox对角线，归一化（越低越好）")
    print("  Transl_jitter: 相邻帧 SMPL transl 变化均值（越低=轨迹越平滑）")
    print("  Scale        : smpl_scale 跨帧均值±std（std越低=尺度越稳定）")


if __name__ == '__main__':
    main()
