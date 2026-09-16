"""
粗对齐质量分析脚本：分析 bike/jogging/seattle（OUR 不如 STM 的场景）vs citron/parkinglot（OUR 超过 STM 的场景）
分析维度：
1. 3D 轨迹往返性
2. 2D keypoint 检测置信度
3. 帧间轨迹跳变
4. 轨迹方向反转次数（往返计数）
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCENES = ['citron', 'parkinglot', 'bike', 'jogging', 'seattle']
COLORS = {
    'citron':     '#2196F3',   # 蓝 - OUR 好
    'parkinglot': '#4CAF50',   # 绿 - OUR 好
    'bike':       '#F44336',   # 红 - OUR 差
    'jogging':    '#FF9800',   # 橙 - OUR 差
    'seattle':    '#9C27B0',   # 紫 - OUR 差
}
STATUS = {
    'citron': 'GOOD(+0.52)', 'parkinglot': 'GOOD(+0.83)',
    'bike': 'BAD(-0.52)', 'jogging': 'BAD(-0.39)', 'seattle': 'BAD(-1.42)'
}
OUR_PSNR = {'citron': 17.4008, 'parkinglot': 15.1924, 'bike': 18.0921, 'jogging': 16.9871, 'seattle': 15.3117}
STM_PSNR = {'citron': 16.8763, 'parkinglot': 14.3660, 'bike': 18.6133, 'jogging': 17.3776, 'seattle': 16.7278}


def load_trajectory(scene):
    """从 alignments.npy 提取世界坐标系下的人体 root translation"""
    path = os.path.join(ROOT, f'data/neuman/dataset/{scene}_vimo/alignments.npy')
    raw = np.load(path, allow_pickle=True).item()
    frames = sorted(raw.keys())
    transl = np.array([raw[f][3] for f in frames])  # (N, 3) 最后一行是 translation
    return transl, frames


def load_keypoints(scene, frames):
    """加载 vitpose 关键点置信度"""
    kp_dir = os.path.join(ROOT, f'data/neuman/dataset/{scene}_vimo/keypoints')
    confs = []
    for fname in frames:
        kp_path = os.path.join(kp_dir, fname + '.npy')
        if os.path.isfile(kp_path):
            kp = np.load(kp_path)   # (17, 3) → x, y, conf
            confs.append(kp[:, 2].mean())
        else:
            confs.append(np.nan)
    return np.array(confs)


def round_trip_score(transl):
    """往返程度：前半轨迹 vs 后半轨迹翻转后的距离（越小 = 越往返）"""
    N = len(transl)
    half = N // 2
    first = transl[:half]
    second = transl[half:][::-1]
    n = min(len(first), len(second))
    return np.mean(np.linalg.norm(first[:n] - second[:n], axis=1))


def direction_reversals(transl):
    """沿主运动轴的方向反转次数"""
    # 找主运动轴
    span = transl.max(0) - transl.min(0)
    ax = span.argmax()
    proj = transl[:, ax]
    delta = np.diff(proj)
    signs = np.sign(delta[delta != 0])
    reversals = int(np.sum(np.diff(signs) != 0))
    return reversals, ax


def analyze_all():
    stats = {}
    trajs = {}
    for sc in SCENES:
        transl, frames = load_trajectory(sc)
        confs = load_keypoints(sc, frames)
        delta = np.diff(transl, axis=0)
        delta_norm = np.linalg.norm(delta, axis=1)
        rt = round_trip_score(transl)
        rev, ax = direction_reversals(transl)
        span = transl.max(0) - transl.min(0)
        stats[sc] = {
            'N': len(frames),
            'transl': transl,
            'delta_norm': delta_norm,
            'rt_dist': rt,
            'reversals': rev,
            'main_axis': ax,
            'span': span,
            'conf_mean': np.nanmean(confs),
            'conf_min': np.nanmin(confs),
            'confs': confs,
        }
        trajs[sc] = transl
    return stats


def print_summary(stats):
    print("\n" + "="*110)
    print(f"{'场景':12s} | {'帧数':>5} | {'往返距离':>10} | {'方向反转':>8} | {'最大位移范围':>14} | {'帧间Δ均值':>10} | {'帧间Δ最大':>10} | {'关键点置信':>10} | 状态")
    print("-"*110)
    for sc in SCENES:
        s = stats[sc]
        span_str = f"[{s['span'][0]:.2f},{s['span'][1]:.2f},{s['span'][2]:.2f}]"
        print(f"{sc:12s} | {s['N']:5d} | {s['rt_dist']:10.3f} | {s['reversals']:8d} | {span_str:>14s} | {s['delta_norm'].mean():10.3f} | {s['delta_norm'].max():10.3f} | {s['conf_mean']:10.3f} | {STATUS[sc]}")
    print("="*110)
    print("\n说明：往返距离 越小 = 往返轨迹越强；方向反转次数 越多 = 往返越明显")


def plot_analysis(stats, outdir):
    os.makedirs(outdir, exist_ok=True)

    # ── 图1：3D 轨迹俯视图（XZ 平面） ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    fig.suptitle('VIMO 粗对齐 3D 轨迹俯视图（XZ 平面）', fontsize=14, fontproperties='DejaVu Sans')
    for i, sc in enumerate(SCENES):
        ax = axes[i]
        tr = stats[sc]['transl']
        N = len(tr)
        sc_color = COLORS[sc]
        # 画轨迹，颜色渐变显示时间
        for j in range(N - 1):
            alpha = 0.4 + 0.6 * j / N
            ax.plot(tr[j:j+2, 0], tr[j:j+2, 2], color=sc_color, alpha=alpha, linewidth=1.5)
        ax.scatter(tr[0, 0], tr[0, 2], c='green', s=80, zorder=5, label='start')
        ax.scatter(tr[-1, 0], tr[-1, 2], c='red', marker='X', s=100, zorder=5, label='end')
        # 标注往返分界点
        half = N // 2
        ax.scatter(tr[half, 0], tr[half, 2], c='orange', marker='D', s=60, zorder=5, label='mid')
        ax.set_title(f"{sc}  {STATUS[sc]}\nOUR={OUR_PSNR[sc]:.4f} STM={STM_PSNR[sc]:.4f}\n往返={stats[sc]['rt_dist']:.2f} 反转={stats[sc]['reversals']}次", fontsize=9)
        ax.set_xlabel('X (world)')
        ax.set_ylabel('Z (world)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='datalim')
    axes[-1].axis('off')
    plt.tight_layout()
    out1 = os.path.join(outdir, 'vimo_trajectory_xz.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"轨迹俯视图已保存: {out1}")

    # ── 图2：逐帧分析（X 坐标随时间变化 + 帧间跳变） ─────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    fig.suptitle('VIMO 逐帧分析：X 坐标时序 & 帧间位移', fontsize=13)
    for i, sc in enumerate(SCENES):
        s = stats[sc]
        tr = s['transl']
        N = len(tr)
        ax_top = axes[0, i]
        ax_bot = axes[1, i]
        sc_color = COLORS[sc]
        # 上图：X坐标随帧变化
        ax_top.plot(tr[:, 0], color=sc_color, linewidth=1.5)
        ax_top.axvline(N // 2, color='gray', linestyle='--', alpha=0.6, label='mid')
        ax_top.set_title(f"{sc}\n往返={s['rt_dist']:.2f}", fontsize=8)
        ax_top.set_ylabel('X (world)')
        ax_top.set_xlabel('Frame')
        ax_top.grid(True, alpha=0.3)
        # 下图：帧间位移 + 关键点置信度
        ax_bot.bar(range(len(s['delta_norm'])), s['delta_norm'], color=sc_color, alpha=0.6, label='Δ位移')
        ax_bot2 = ax_bot.twinx()
        ax_bot2.plot(s['confs'], color='navy', alpha=0.7, linewidth=1, label='KP conf')
        ax_bot2.set_ylim(0, 1.1)
        ax_bot2.set_ylabel('KP conf', fontsize=7, color='navy')
        ax_bot.set_ylabel('帧间Δ', fontsize=7)
        ax_bot.set_xlabel('Frame')
        ax_bot.set_title(f"Δ均={s['delta_norm'].mean():.3f} 反转={s['reversals']}次 KPconf={s['conf_mean']:.3f}", fontsize=7)
        ax_bot.grid(True, alpha=0.2)
    plt.tight_layout()
    out2 = os.path.join(outdir, 'vimo_frame_analysis.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"逐帧分析图已保存: {out2}")

    # ── 图3：往返距离 vs OUR-STM HUMAN PSNR 散点图 ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('往返距离 vs 方法性能差', fontsize=13)
    rt_dists = [stats[sc]['rt_dist'] for sc in SCENES]
    reversals = [stats[sc]['reversals'] for sc in SCENES]
    our_stm_diff = [OUR_PSNR[sc] - STM_PSNR[sc] for sc in SCENES]
    for i, sc in enumerate(SCENES):
        axes[0].scatter(rt_dists[i], our_stm_diff[i], color=COLORS[sc], s=120, zorder=5)
        axes[0].annotate(sc, (rt_dists[i], our_stm_diff[i]), textcoords='offset points', xytext=(5, 3), fontsize=9)
    axes[0].axhline(0, color='gray', linestyle='--')
    axes[0].set_xlabel('往返距离（越小=越往返）')
    axes[0].set_ylabel('OUR - STM HUMAN PSNR (↑ better)')
    axes[0].set_title('往返距离 vs 性能差')
    axes[0].grid(True, alpha=0.3)

    for i, sc in enumerate(SCENES):
        axes[1].scatter(reversals[i], our_stm_diff[i], color=COLORS[sc], s=120, zorder=5)
        axes[1].annotate(sc, (reversals[i], our_stm_diff[i]), textcoords='offset points', xytext=(5, 3), fontsize=9)
    axes[1].axhline(0, color='gray', linestyle='--')
    axes[1].set_xlabel('方向反转次数（越多=越往返）')
    axes[1].set_ylabel('OUR - STM HUMAN PSNR (↑ better)')
    axes[1].set_title('反转次数 vs 性能差')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    out3 = os.path.join(outdir, 'vimo_roundtrip_vs_perf.png')
    plt.savefig(out3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"相关性散点图已保存: {out3}")


if __name__ == '__main__':
    os.chdir(ROOT)
    stats = analyze_all()
    print_summary(stats)
    outdir = os.path.join(ROOT, 'run_logs', 'vimo_alignment_analysis')
    plot_analysis(stats, outdir)
    print(f"\n所有图表保存至: {outdir}")
