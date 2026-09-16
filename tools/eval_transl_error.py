"""
从各训练步数的 checkpoint 提取 transl，与 NeuMan GT 对比，观察对齐误差是否随训练下降。

用法:
  python tools/eval_transl_error.py --exp vimo_optim_transl_1k_seattle_20260608
  python tools/eval_transl_error.py --exp vimo_inline_attn_18k_seattle_20260607  # (旧格式，transl 未保存，只打印 VIMO 基线)
"""
import argparse
import os
import glob
import numpy as np
import torch


def extract_transl_from_ckpt(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "transl" in state:
        t = state["transl"]
        return t.numpy() if isinstance(t, torch.Tensor) else np.array(t)
    # 旧格式不含 transl
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="vimo_optim_transl_1k_seattle_20260608")
    parser.add_argument("--scene", default="seattle_vimo")
    parser.add_argument("--base_output",
                        default="/workspace/nas_auto_backup/yuzilang/ml-hugs-work/output/human_scene/neuman")
    parser.add_argument("--base_data",
                        default="/workspace/nas_auto_backup/yuzilang/ml-hugs-work/data/neuman/dataset")
    args = parser.parse_args()

    exp_root = os.path.join(args.base_output, args.scene, "hugs_trimlp", args.exp)
    runs = sorted(glob.glob(os.path.join(exp_root, "*")))
    if not runs:
        print(f"[ERROR] 找不到实验目录: {exp_root}")
        return
    ckpt_dir = os.path.join(runs[-1], "ckpt")

    gt_npz  = os.path.join(args.base_data, args.scene, "4d_humans", "smpl_optimized_aligned_scale_gt.npz")
    vimo_npz = os.path.join(args.base_data, args.scene, "4d_humans", "smpl_optimized_aligned_scale.npz")

    print(f"ckpt_dir : {ckpt_dir}")
    print(f"GT  npz  : {gt_npz}")
    print(f"VIMO npz : {vimo_npz}")
    print()

    gt_transl   = np.load(gt_npz)["transl"]    # (N, 3)
    vimo_transl = np.load(vimo_npz)["transl"]   # (N, 3)
    n_frames = gt_transl.shape[0]

    vimo_err = np.linalg.norm(vimo_transl - gt_transl, axis=1)
    print(f"{'[VIMO baseline]':>12}  mean_L2={vimo_err.mean():.4f}  max_L2={vimo_err.max():.4f}  (delta_vs_vimo=0)")
    print()

    # 推断训练集帧索引（与 NeumanDataset.train_split 保持一致）
    scene_length = n_frames   # GT 帧总数
    num_val = scene_length // 5
    length  = int(1 / num_val * scene_length)
    offset  = length // 2
    val_list   = list(range(scene_length))[offset::length]
    train_list = sorted(set(range(scene_length)) - set(val_list))
    print(f"train frames: {len(train_list)} / {scene_length}  (val={len(val_list)})")
    print(f"train indices: {train_list}")
    print()

    gt_transl_train   = gt_transl[train_list]    # (33, 3)
    vimo_transl_train = vimo_transl[train_list]  # (33, 3)

    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "human_*.pth")))
    if not ckpt_files:
        print(f"[WARNING] 没有找到 checkpoint: {ckpt_dir}")
        return

    found_any = False
    rows = []
    for ckpt_path in ckpt_files:
        step_str = os.path.basename(ckpt_path).replace("human_", "").replace(".pth", "")
        transl = extract_transl_from_ckpt(ckpt_path)
        if transl is None:
            print(f"  step {step_str:>8}: [transl 未保存，旧格式 checkpoint]")
            continue
        if transl.ndim == 3:
            transl = transl[0]
        # 支持 train-only (33,3) 或 full (41,3) 两种格式
        if transl.shape[0] == len(train_list):
            gt_ref   = gt_transl_train
            vimo_ref = vimo_transl_train
        elif transl.shape[0] == n_frames:
            gt_ref   = gt_transl
            vimo_ref = vimo_transl
        else:
            print(f"  step {step_str:>8}: [shape 不匹配 {transl.shape}]")
            continue

        err   = np.linalg.norm(transl - gt_ref, axis=1)
        delta = np.linalg.norm(transl - vimo_ref, axis=1)
        rows.append((step_str, err.mean(), err.max(), delta.mean()))
        found_any = True

    if not found_any:
        print("[结论] 所有 checkpoint 均为旧格式，transl 未保存。")
        print("       当前运行的实验（vimo_optim_transl_*）保存了 transl，等训练完成后再运行此脚本。")
        return

    print(f"{'Step':>10}  {'mean_L2_vs_GT':>15}  {'max_L2_vs_GT':>14}  {'moved_from_VIMO':>16}")
    print("-" * 62)
    vimo_train_err = np.linalg.norm(vimo_transl_train - gt_transl_train, axis=1)
    print(f"{'VIMO_init':>10}  {vimo_train_err.mean():>15.4f}  {vimo_train_err.max():>14.4f}  {'0.000000':>16}")
    for step_str, mean_err, max_err, delta in rows:
        improve = vimo_err.mean() - mean_err
        flag = "↓ better" if improve > 0.01 else ("↑ worse" if improve < -0.01 else "≈ same")
        print(f"{step_str:>10}  {mean_err:>15.4f}  {max_err:>14.4f}  {delta:>16.6f}  {flag}")

    print()
    print("=== 说明 ===")
    print("mean_L2_vs_GT: 当前 transl 与 NeuMan GT transl 的逐帧 L2 均值（越小越好）")
    print("moved_from_VIMO: 相对初始 VIMO 值移动了多少（体现 optimizer 是否在更新 transl）")


if __name__ == "__main__":
    main()
