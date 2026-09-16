"""
方案五 PSNR 评估：推理后 savgol 低通滤波对 PSNR 的影响。

做法：
1. 加载已由 analyze_delta_mu_temporal.py 保存的 delta_mu_f16.npy
2. 用 savgol_filter 在时间轴（axis=0）滤波
3. 逐帧推理时，将原始 delta_mu 替换为滤波后的值，重渲染
4. 与 GT 计算 PSNR，对比 baseline（无滤波）

关键：anchor_attn.forward() 返回 stats["delta_mu"] 和 stats["gamma_mu_eff"]，
通过 corrected["xyz"] += gamma_mu_eff * (filtered_dm - original_dm) 完成替换。

用法：
  cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work
  python scripts/eval_method5_psnr.py --seq bike_vimo_v4 --windows 7 11
  python scripts/eval_method5_psnr.py --all_seqs --windows 7 11 21
"""

import argparse
import sys
import json
import numpy as np
import torch
from pathlib import Path
from scipy.signal import savgol_filter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

SEQ_CKPT = {
    'bike_vimo_v4':       'output/human_scene/neuman/bike_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/2026-06-14_12-52-08/ckpt',
    'jogging_vimo_v4':    'output/human_scene/neuman/jogging_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/2026-06-14_21-51-55/ckpt',
    'seattle_vimo_v4':    'output/human_scene/neuman/seattle_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/2026-06-14_17-28-48/ckpt',
    'lab_vimo_v4':        'output/human_scene/neuman/lab_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/2026-06-15_01-16-10/ckpt',
    'parkinglot_vimo_v4': 'output/human_scene/neuman/parkinglot_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/2026-06-15_03-29-29/ckpt',
    'citron_vimo_v4':     'output/human_scene/neuman/citron_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/2026-06-15_05-32-10/ckpt',
}

DM_BASE = Path('output/delta_mu_analysis')
OUT_FILE = Path('output/delta_mu_analysis/method5_psnr_results.json')
POLYORDER = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seq', default='bike_vimo_v4')
    p.add_argument('--ckpt_dir', default='')
    p.add_argument('--windows', nargs='+', type=int, default=[7, 11, 21])
    p.add_argument('--device', default='cuda')
    p.add_argument('--all_seqs', action='store_true')
    return p.parse_args()


def load_models(ckpt_dir, device):
    from hugs.models.hugs_trimlp import HUGS_TRIMLP
    from hugs.models.scene import SceneGS
    from hugs.models.anchor_attention import AnchorSceneAttentionBaseline
    from omegaconf import OmegaConf

    ckpt_dir = Path(ckpt_dir)
    cfg = OmegaConf.load(ckpt_dir.parent / 'config_train.yaml')

    print('[load] human_gs...', flush=True)
    human_ckpt = torch.load(ckpt_dir / 'human_final.pth', map_location=device, weights_only=False)
    init_betas = human_ckpt.get('betas', torch.zeros(10, device=device))
    if hasattr(init_betas, 'data'):
        init_betas = init_betas.data
    human_gs = HUGS_TRIMLP(
        sh_degree=cfg.human.sh_degree,
        n_subdivision=cfg.human.n_subdivision,
        use_surface=cfg.human.use_surface,
        init_2d=cfg.human.init_2d,
        rotate_sh=cfg.human.rotate_sh,
        isotropic=cfg.human.isotropic,
        init_scale_multiplier=cfg.human.init_scale_multiplier,
        n_features=32,
        use_deformer=cfg.human.use_deformer,
        disable_posedirs=cfg.human.disable_posedirs,
        triplane_res=cfg.human.triplane_res,
        betas=init_betas,
    )
    human_gs.load_state_dict(human_ckpt)

    print('[load] scene_gs...', flush=True)
    scene_gs = SceneGS(sh_degree=cfg.scene.sh_degree)
    scene_gs.restore(
        torch.load(ckpt_dir / 'scene_final.pth', map_location='cpu', weights_only=False),
        cfg.scene.lr)
    scene_out = scene_gs.forward()
    scene_out = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in scene_out.items()}
    print(f'  scene GS: {scene_out["xyz"].shape[0]:,} pts', flush=True)

    print('[load] anchor_attention...', flush=True)
    num_anchors = int(human_gs.anchor_ids.max().item()) + 1
    aa_sd = torch.load(ckpt_dir / 'anchor_attention_final.pth', map_location='cpu', weights_only=False)
    _n_frames = 0
    if 'frame_embed.weight' in aa_sd:
        _n_frames = aa_sd['frame_embed.weight'].shape[0]
    anchor_attn = AnchorSceneAttentionBaseline(
        cfg.anchor_attention,
        num_anchors=num_anchors,
        top_m=int(getattr(cfg.anchor_attention, 'top_m', 2)),
        num_frames=_n_frames,
    ).to(device)
    anchor_attn.load_state_dict(aa_sd)
    anchor_attn.eval()

    return human_gs, scene_out, anchor_attn, cfg


def get_dataset(cfg, device):
    from hugs.datasets.neuman import NeumanDataset
    # split='val' 包含 rgb/bbox，frame_idx 字段是全局帧号，用于索引 delta_mu_f16.npy
    ds = NeumanDataset(
        seq=cfg.dataset.seq,
        split='val',
        mono_depth_dir=getattr(cfg.dataset, 'mono_depth_dir', None),
    )
    print(f'  val dataset: {len(ds)} frames', flush=True)
    return ds


def psnr_torch(pred, gt):
    mse = ((pred.float() - gt.float()) ** 2).mean()
    if mse == 0:
        return float('inf')
    return float(20.0 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse)))


@torch.no_grad()
def eval_seq(seq, ckpt_dir, windows, device):
    from hugs.renderer.gs_renderer import render_human_scene

    human_gs, scene_out, anchor_attn, cfg = load_models(ckpt_dir, device)
    dataset = get_dataset(cfg, device)

    # 加载 delta_mu_f16.npy 并滤波
    dm_path = DM_BASE / seq.replace('_vimo_v4', '') / 'delta_mu_f16.npy'
    print(f'[{seq}] 加载 {dm_path} ...', flush=True)
    dm_f32 = np.load(dm_path).astype(np.float32)   # (T_all, N, 3)，T_all = 全部帧数
    T_all, N, _ = dm_f32.shape

    filtered_dms = {}
    for w in windows:
        if w < T_all:
            filtered_dms[w] = savgol_filter(dm_f32, window_length=w, polyorder=POLYORDER, axis=0)
            print(f'  savgol window={w} 完成', flush=True)

    anchor_ids = human_gs.anchor_ids.to(device).long()
    anchor_weights = human_gs.anchor_weights.to(device)
    iteration = int(cfg.train.num_steps)
    bg_color = torch.zeros(3, dtype=torch.float32, device=device)

    baseline_psnrs = []
    filtered_psnrs = {w: [] for w in filtered_dms}

    for i, data in enumerate(tqdm(dataset, desc=f'[{seq}] 渲染')):
        data_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in data.items()}

        human_out = human_gs.forward(
            global_orient=data_gpu['global_orient'],
            body_pose=data_gpu['body_pose'],
            betas=data_gpu['betas'],
            transl=data_gpu['transl'],
            smpl_scale=data_gpu['smpl_scale'][None],
            dataset_idx=-1,
            is_train=False,
            ext_tfs=None,
        )

        corrected, stats = anchor_attn(
            human_out, scene_out, anchor_ids, anchor_weights,
            iteration=iteration, frame_idx=i,
        )

        # 全局帧号，用于索引 delta_mu_f16.npy（split='val' 返回的是 val 帧子集的全局 idx）
        global_frame_idx = int(data_gpu['frame_idx'].item())

        # Baseline 渲染
        pkg_base = render_human_scene(
            data=data_gpu,
            human_gs_out=corrected,
            scene_gs_out=scene_out,
            bg_color=bg_color,
            render_mode='human_scene',
        )
        gt = data_gpu['rgb'].clamp(0, 1)
        baseline_psnrs.append(psnr_torch(pkg_base['render'].clamp(0, 1), gt))

        # 滤波替换渲染
        if stats and 'delta_mu' in stats and 'gamma_mu_eff' in stats:
            gamma_mu_eff = float(stats['gamma_mu_eff'])
            orig_dm = stats['delta_mu']  # (N, 3) on GPU, detached

            for w, fdm_np in filtered_dms.items():
                fdm = torch.from_numpy(fdm_np[global_frame_idx]).to(device)   # (N, 3)
                corrected_f = dict(corrected)
                # swap delta_mu contribution: xyz已经包含 gamma*orig_dm，改为 gamma*filtered_dm
                corrected_f['xyz'] = corrected['xyz'] + gamma_mu_eff * (fdm - orig_dm)

                pkg_f = render_human_scene(
                    data=data_gpu,
                    human_gs_out=corrected_f,
                    scene_gs_out=scene_out,
                    bg_color=bg_color,
                    render_mode='human_scene',
                )
                filtered_psnrs[w].append(psnr_torch(pkg_f['render'].clamp(0, 1), gt))
        else:
            # anchor_attn 未产生 delta_mu（理论上不会发生，iteration=18000）
            for w in filtered_dms:
                filtered_psnrs[w].append(baseline_psnrs[-1])

    mean_base = float(np.mean(baseline_psnrs))
    result = {
        'n_frames': len(dataset),
        'baseline_psnr': mean_base,
        'windows': {},
    }
    for w in filtered_dms:
        mean_f = float(np.mean(filtered_psnrs[w]))
        result['windows'][int(w)] = {
            'filtered_psnr': mean_f,
            'delta_psnr': mean_f - mean_base,
        }
    return result


def main():
    args = parse_args()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if args.all_seqs:
        todo = list(SEQ_CKPT.items())
    else:
        ckpt = args.ckpt_dir or SEQ_CKPT.get(args.seq, '')
        todo = [(args.seq, ckpt)]

    all_results = {}
    # 尝试读取已有结果，支持断点续跑
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            all_results = json.load(f)

    header = f"{'场景':20s} {'baseline':>10s} " + ' '.join(f'W={w:>2d}Δ' for w in args.windows)
    print(header, flush=True)
    print('-' * 80, flush=True)

    for seq, ckpt_dir in todo:
        short = seq.replace('_vimo_v4', '')
        if short in all_results:
            print(f'[{short}] 已有结果，跳过', flush=True)
            continue
        result = eval_seq(seq, ckpt_dir, args.windows, args.device)
        all_results[short] = result

        with open(OUT_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)

        line = f"{short:20s} {result['baseline_psnr']:10.4f}"
        for w in args.windows:
            wr = result['windows'].get(int(w))
            if wr:
                line += f"  {wr['delta_psnr']:+.4f}"
            else:
                line += f"  {'skip':>7s}"
        print(line, flush=True)

    # 最终汇总表
    print(f'\n{"场景":20s} {"baseline":>10s}', end='')
    for w in args.windows:
        print(f'  {"W="+str(w)+" PSNR":>12s}  {"ΔdB":>7s}', end='')
    print()
    print('-' * 100)
    for short, res in all_results.items():
        print(f'{short:20s} {res["baseline_psnr"]:10.4f}', end='')
        for w in args.windows:
            wr = res['windows'].get(int(w))
            if wr:
                print(f'  {wr["filtered_psnr"]:12.4f}  {wr["delta_psnr"]:+7.4f}', end='')
            else:
                print(f'  {"—":>12s}  {"—":>7s}', end='')
        print()
    print(f'\n结果保存到 {OUT_FILE}', flush=True)


if __name__ == '__main__':
    main()
