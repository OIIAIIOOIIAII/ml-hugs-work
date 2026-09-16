"""
方案五：推理后低通滤波（savgol_filter）评估。
不需要 GPU，直接对已保存的 delta_mu_f16.npy 进行滤波，
评估：ratio 降低幅度（各 window_length）+ 滤波后数据量估算。

用法：
  cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work
  python scripts/eval_method5_savgol.py
"""

import sys
import numpy as np
import json
from pathlib import Path
from scipy.signal import savgol_filter

SEQS = ['bike', 'jogging', 'seattle', 'lab', 'parkinglot', 'citron']
WINDOWS = [5, 7, 11, 15, 21]
POLYORDER = 2
BASE_DIR = Path('output/delta_mu_analysis')
OUT_FILE  = Path('output/delta_mu_analysis/method5_savgol_results.json')


def compute_ratio(delta_mu_f32):
    """ratio = diff_std / raw_std，与 analyze_delta_mu_temporal.py 保持一致"""
    raw_std  = delta_mu_f32.std()
    if raw_std < 1e-9:
        return float('nan')
    diff = delta_mu_f32[1:] - delta_mu_f32[:-1]   # (T-1, N, 3)
    diff_std = diff.std()
    return float(diff_std / raw_std)


def estimate_entropy_bytes(diff_arr):
    """对差分展平后做直方图熵估计，返回理论下界字节数"""
    flat = diff_arr.flatten()
    counts, _ = np.histogram(flat, bins=256)
    counts = counts[counts > 0]
    probs = counts / counts.sum()
    bits_per_elem = float(np.sum(-probs * np.log2(probs)))  # 香农熵 bits
    total_bits = bits_per_elem * flat.size
    return total_bits / 8  # 字节数


def analyze_seq(seq, windows):
    npy_path = BASE_DIR / seq / 'delta_mu_f16.npy'
    print(f'[{seq}] 加载 {npy_path} ...', flush=True)
    raw_f16 = np.load(npy_path)           # (T, N, 3) float16
    raw_f32 = raw_f16.astype(np.float32)
    del raw_f16
    T, N, _ = raw_f32.shape
    print(f'[{seq}] shape=({T},{N},3), 开始计算...', flush=True)

    baseline_ratio = compute_ratio(raw_f32)
    baseline_bytes_fp32 = raw_f32.nbytes          # T帧 fp32

    # 差分 int8（baseline无滤波）
    diff_raw = raw_f32[1:] - raw_f32[:-1]
    diff_max = np.abs(diff_raw).max()
    int8_scale = diff_max / 127.0 if diff_max > 0 else 1.0
    baseline_int8_bytes = (T - 1) * N * 3 * 1   # int8 = 1 byte/elem

    results = {
        'n_frames': int(T),
        'n_pts': int(N),
        'baseline_ratio': baseline_ratio,
        'baseline_fp32_MB_per_frame': raw_f32[0].nbytes / 1e6,
        'baseline_int8_diff_MB_per_frame': (N * 3) / 1e6,
        'windows': {}
    }

    for w in windows:
        if w >= T:
            results['windows'][w] = {'skip': 'window >= T'}
            continue

        print(f'[{seq}] savgol window={w} ...', flush=True)
        # savgol 沿时间轴（axis=0）滤波
        filtered = savgol_filter(raw_f32, window_length=w, polyorder=POLYORDER, axis=0)

        ratio_after = compute_ratio(filtered)

        # 差分（I帧+P帧差分传输）
        diff_filtered = filtered[1:] - filtered[:-1]

        # int8 量化差分
        diff_max_f = np.abs(diff_filtered).max()
        scale_f = diff_max_f / 127.0 if diff_max_f > 0 else 1.0
        int8_diff = np.clip(np.round(diff_filtered / scale_f), -128, 127).astype(np.int8)
        recon_diff = int8_diff.astype(np.float32) * scale_f
        int8_quant_error = float(np.abs(recon_diff - diff_filtered).mean())

        # 熵编码理论下界
        entropy_bytes = estimate_entropy_bytes(diff_filtered)
        entropy_kb_per_frame = entropy_bytes / (T - 1) / 1024

        # int8 差分字节数（不含熵编码）
        int8_bytes_per_frame = N * 3   # int8

        results['windows'][int(w)] = {
            'ratio_before': baseline_ratio,
            'ratio_after':  ratio_after,
            'ratio_reduction_pct': (1 - ratio_after / baseline_ratio) * 100 if baseline_ratio > 0 else 0,
            'int8_diff_KB_per_frame': int8_bytes_per_frame / 1024,
            'entropy_lb_KB_per_frame': entropy_kb_per_frame,
            'fp32_MB_per_frame': raw_f32[0].nbytes / 1e6,
            'int8_quant_error_mean': int8_quant_error,
        }

    return results


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_results = {}

    header = f"{'场景':12s} {'W':>4s} {'ratio前':>8s} {'ratio后':>8s} {'降幅%':>7s} {'int8 KB/帧':>12s} {'熵LB KB/帧':>12s}"
    print(header, flush=True)
    print('-' * 75, flush=True)

    for seq in SEQS:
        res = analyze_seq(seq, WINDOWS)
        all_results[seq] = res

        # 每个场景结束后立即持久化
        with open(OUT_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)

        for w in WINDOWS:
            wr = res['windows'].get(int(w), {})
            if 'skip' in wr:
                continue
            print(f"{seq:12s} {w:4d} "
                  f"{wr['ratio_before']:8.3f} "
                  f"{wr['ratio_after']:8.3f} "
                  f"{wr['ratio_reduction_pct']:7.1f}% "
                  f"{wr['int8_diff_KB_per_frame']:12.1f} "
                  f"{wr['entropy_lb_KB_per_frame']:12.1f}", flush=True)
        print(flush=True)

    print(f"\n全部完成，结果保存到 {OUT_FILE}", flush=True)


if __name__ == '__main__':
    main()
