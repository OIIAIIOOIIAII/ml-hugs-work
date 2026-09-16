"""
分析 AnchorAttention delta_mu 的帧间时序相关性。
对已训练好的 v4_correct_inline_attn_18k 模型，逐帧跑推理，提取每帧 delta_mu，
计算帧间差分分布，估算差分编码后的压缩比。

用法：
  cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work
  python scripts/analyze_delta_mu_temporal.py --seq bike_vimo_v4
  python scripts/analyze_delta_mu_temporal.py --all_seqs
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seq', default='bike_vimo_v4')
    p.add_argument('--ckpt_dir', default='')
    p.add_argument('--out_dir', default='')
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

    print('[load] human_gs...')
    # HUGS_TRIMLP 内部 self.device='cuda'，betas 必须在 cuda 上
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
    # HUGS_TRIMLP 不继承 nn.Module，device 已经内部固定为 cuda

    print('[load] scene_gs...')
    scene_gs = SceneGS(sh_degree=cfg.scene.sh_degree)
    scene_gs.restore(
        torch.load(ckpt_dir / 'scene_final.pth', map_location='cpu', weights_only=False),
        cfg.scene.lr)
    scene_out = scene_gs.forward()
    scene_out = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in scene_out.items()}
    print(f'  scene GS: {scene_out["xyz"].shape[0]:,} pts')

    print('[load] anchor_attention...')
    num_anchors = int(human_gs.anchor_ids.max().item()) + 1
    aa_sd = torch.load(ckpt_dir / 'anchor_attention_final.pth', map_location='cpu', weights_only=False)
    # 从 checkpoint 推断 num_frames（frame_embed 可能存在）
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
    ds = NeumanDataset(
        seq=cfg.dataset.seq,
        split='all',
        mono_depth_dir=getattr(cfg.dataset, 'mono_depth_dir', None),
    )
    print(f'  dataset: {len(ds)} frames')
    return ds


@torch.no_grad()
def extract_all_delta_mu(human_gs, scene_out, anchor_attn, dataset, cfg, device):
    N_gs = human_gs.anchor_ids.shape[0]
    anchor_ids = human_gs.anchor_ids.to(device).long()
    anchor_weights = human_gs.anchor_weights.to(device)
    iteration = int(cfg.train.num_steps)

    all_delta_mu = []
    for i, data in enumerate(tqdm(dataset, desc='forward all frames')):
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
        _, stats = anchor_attn(
            human_out, scene_out, anchor_ids, anchor_weights,
            iteration=iteration, frame_idx=i,
        )
        dm = stats.get('delta_mu', torch.zeros(N_gs, 3))
        all_delta_mu.append(dm.cpu().float())

    return torch.stack(all_delta_mu, dim=0)  # (F, N, 3)


def analyze(delta_mu, seq_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dm = delta_mu.numpy() if isinstance(delta_mu, torch.Tensor) else delta_mu
    F, N, C = dm.shape
    print(f'\n{"="*60}')
    print(f'[{seq_name}]  {F} frames × {N:,} GS pts × {C} dims')
    print(f'{"="*60}')

    flat = dm.reshape(-1)
    diff = dm[1:] - dm[:-1]          # (F-1, N, 3)
    flat_diff = diff.reshape(-1)

    # ── 原始分布
    print(f'\n● 原始 delta_mu')
    print(f'  std={flat.std():.5f}  '
          f'|v| P50={np.percentile(np.abs(flat),50):.5f}  '
          f'P90={np.percentile(np.abs(flat),90):.5f}  '
          f'P99={np.percentile(np.abs(flat),99):.5f}')
    fp32_per_frame_kb = N * C * 4 / 1024
    print(f'  单帧 fp32: {fp32_per_frame_kb:.0f} KB    fp16: {fp32_per_frame_kb/2:.0f} KB')

    # ── 帧间差分分布
    print(f'\n● 帧间差分 (delta_mu_t - delta_mu_{{t-1}})')
    print(f'  std={flat_diff.std():.6f}  '
          f'|d| P50={np.percentile(np.abs(flat_diff),50):.6f}  '
          f'P90={np.percentile(np.abs(flat_diff),90):.6f}  '
          f'P99={np.percentile(np.abs(flat_diff),99):.6f}')
    ratio = flat_diff.std() / (flat.std() + 1e-10)
    print(f'  diff_std / raw_std = {ratio:.4f}  (越小 → 时序冗余越高)')

    # ── 稀疏性
    raw_p99 = np.percentile(np.abs(flat), 99)
    print(f'\n● 差分稀疏性（相对原始P99={raw_p99:.5f}）')
    for pct in [0.01, 0.05, 0.10, 0.20]:
        thresh = raw_p99 * pct
        frac = (np.abs(flat_diff) < thresh).mean()
        print(f'  |diff| < {thresh:.5f} ({pct*100:.0f}% of raw P99): {frac*100:.1f}% 的值')

    # ── 量化 & 熵估算
    sigma_diff = flat_diff.std()
    # 理论熵（高斯假设）
    entropy_bits = max(0.0, 0.5 * np.log2(2 * np.pi * np.e * sigma_diff**2 + 1e-40))
    entropy_kb = N * C * entropy_bits / 8 / 1024
    print(f'\n● 量化 / 熵编码估算（高斯假设，sigma_diff={sigma_diff:.6f}）')
    print(f'  理论熵: {entropy_bits:.3f} bits/value → {entropy_kb:.1f} KB/帧 (理论下界)')
    int8_kb = N * C * 1 / 1024
    int16_kb = N * C * 2 / 1024
    print(f'  int8 差分帧 (无熵编码): {int8_kb:.0f} KB/帧')
    print(f'  int16 差分帧 (无熵编码): {int16_kb:.0f} KB/帧')
    print(f'  fp32 原始: {fp32_per_frame_kb:.0f} KB/帧')
    ratio_entropy = fp32_per_frame_kb / (entropy_kb + 1e-6)
    print(f'  理论压缩比 (熵编码 vs fp32): {ratio_entropy:.1f}x')

    # ── 逐帧跳变
    per_frame_rms = np.sqrt((diff**2).mean(axis=(1, 2)))  # (F-1,)
    print(f'\n● 逐帧差分 RMS')
    print(f'  mean={per_frame_rms.mean():.6f}  std={per_frame_rms.std():.6f}  '
          f'max={per_frame_rms.max():.6f} @frame {per_frame_rms.argmax()+1}')
    top5_idx = np.argsort(per_frame_rms)[-5:][::-1]
    print(f'  最大跳变5帧: {[(int(i+1), round(float(per_frame_rms[i]),6)) for i in top5_idx]}')

    # ── 保存
    np.save(f'{out_dir}/delta_mu_f16.npy', dm.astype(np.float16))
    np.save(f'{out_dir}/per_frame_diff_rms.npy', per_frame_rms)
    stats = dict(
        seq=seq_name, n_frames=F, n_gs=N,
        raw_std=float(flat.std()),
        diff_std=float(sigma_diff),
        diff_to_raw_ratio=float(ratio),
        diff_p50=float(np.percentile(np.abs(flat_diff), 50)),
        diff_p90=float(np.percentile(np.abs(flat_diff), 90)),
        diff_p99=float(np.percentile(np.abs(flat_diff), 99)),
        entropy_bits=float(entropy_bits),
        entropy_kb_per_frame=float(entropy_kb),
        fp32_kb_per_frame=float(fp32_per_frame_kb),
        int8_kb_per_frame=float(int8_kb),
        compression_ratio_entropy=float(ratio_entropy),
        max_jump_rms=float(per_frame_rms.max()),
        mean_jump_rms=float(per_frame_rms.mean()),
    )
    with open(f'{out_dir}/stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'\n  → 已保存 {out_dir}/')
    return stats


def main():
    args = parse_args()

    if args.all_seqs:
        todo = [(seq, SEQ_CKPT[seq],
                 f'output/delta_mu_analysis/{seq.replace("_vimo_v4","")}')
                for seq in SEQ_CKPT]
    else:
        ckpt = args.ckpt_dir or SEQ_CKPT.get(args.seq, '')
        out  = args.out_dir  or f'output/delta_mu_analysis/{args.seq.replace("_vimo_v4","")}'
        todo = [(args.seq, ckpt, out)]

    all_stats = []
    for seq, ckpt_dir, out_dir in todo:
        print(f'\n{"#"*60}\n# {seq}\n{"#"*60}')
        human_gs, scene_out, anchor_attn, cfg = load_models(ckpt_dir, args.device)
        dataset = get_dataset(cfg, args.device)
        delta_mu = extract_all_delta_mu(human_gs, scene_out, anchor_attn, dataset, cfg, args.device)
        st = analyze(delta_mu, seq, out_dir)
        all_stats.append(st)
        del human_gs, scene_out, anchor_attn
        torch.cuda.empty_cache()

    if len(all_stats) > 1:
        print(f'\n{"="*70}')
        print(f'{"场景":<16} {"帧数":>5} {"ratio":>7} {"熵bits":>8} '
              f'{"熵KB/帧":>9} {"fp32KB":>7} {"压缩比":>7}')
        print(f'{"-"*70}')
        for s in all_stats:
            print(f'{s["seq"]:<16} {s["n_frames"]:>5} {s["diff_to_raw_ratio"]:>7.4f} '
                  f'{s["entropy_bits"]:>8.3f} {s["entropy_kb_per_frame"]:>9.1f} '
                  f'{s["fp32_kb_per_frame"]:>7.0f} {s["compression_ratio_entropy"]:>7.1f}x')


if __name__ == '__main__':
    main()
