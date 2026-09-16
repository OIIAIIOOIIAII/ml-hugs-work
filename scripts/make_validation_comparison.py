"""
验证结果汇总脚本：收集 OUR / STM / HUGS 的 val 图像，生成逐帧对比图。

用法：
    python scripts/make_validation_comparison.py \
        --tag vimo_inline_attn_20260606 \
        --scenes lab bike citron jogging parkinglot seattle \
        --our   output/human_scene/neuman/{seq}/hugs_trimlp/EXP_NAME/TIMESTAMP \
        --stm   Dynamic-.../output_stm/human_scene/neuman/{seq}/hugs_trimlp/EXP_NAME/TIMESTAMP \
        --hugs  output/human_scene/neuman/{seq}/hugs_trimlp/EXP_NAME/TIMESTAMP

    路径中用 {seq} 作为占位符，脚本会对每个场景替换。
    若某个角色某场景暂无结果，用 none 代替对应路径。

输出目录：output/validation_{tag}/
    ├── our/{seq}/         -> 符号链接到原始 val 目录
    ├── stm/{seq}/
    ├── hugs/{seq}/
    ├── comparisons/{seq}/
    │   ├── full_compare_{idx:03d}.png   # GT | OUR | STM | HUGS（全图）
    │   └── human_compare_{idx:03d}.png  # GT | OUR | STM | HUGS（人体 crop）
    └── summary.md
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROLES = ['our', 'stm', 'hugs']
ROLE_COLORS = {
    'our':  (255, 100, 100),   # 红
    'stm':  (100, 180, 255),   # 蓝
    'hugs': (100, 220, 100),   # 绿
}


def find_val_dir(exp_root: str) -> Path | None:
    """在实验根目录下找 val/ 目录（可能隔一层 timestamp）。"""
    p = Path(exp_root)
    if not p.exists():
        return None
    val = p / 'val'
    if val.exists():
        return val
    # 找最新的 timestamp 子目录
    subdirs = sorted([d for d in p.iterdir() if d.is_dir() and not d.name.startswith('.')])
    for sub in reversed(subdirs):
        val = sub / 'val'
        if val.exists():
            return val
    return None


def get_final_frames(val_dir: Path, prefix: str) -> dict[int, Path]:
    """返回 {idx: path} 的 final 帧字典，prefix 为 'full' 或 'human'。"""
    frames = {}
    for f in val_dir.glob(f'{prefix}_final_*.png'):
        try:
            idx = int(f.stem.split('_')[-1])
            frames[idx] = f
        except ValueError:
            pass
    return frames


def add_label(img: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    """在图像左上角叠加文字标签。"""
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.text((6, 4), text, fill=color)
    return np.array(pil)


def split_half(img_path: Path, side: str = 'right') -> np.ndarray:
    """从 GT|Render 拼接图中切出左（GT）或右（Render）半边。"""
    img = np.array(Image.open(img_path).convert('RGB'))
    w = img.shape[1]
    mid = w // 2
    return img[:, :mid] if side == 'left' else img[:, mid:]


def make_compare_row(gt: np.ndarray, renders: dict[str, np.ndarray | None],
                     label_height: int = 24) -> np.ndarray:
    """
    横向拼接：GT | OUR | STM | HUGS
    空缺的位置用灰色占位。
    """
    h, w = gt.shape[:2]
    placeholder = np.full((h, w, 3), 128, dtype=np.uint8)

    panels = [add_label(gt, 'GT', color=(220, 220, 220))]
    for role in ROLES:
        panel = renders.get(role)
        if panel is None:
            panel = placeholder.copy()
        panels.append(add_label(panel, role.upper(), color=ROLE_COLORS[role]))

    # 如果尺寸不完全一致，统一 resize 到 gt 尺寸
    resized = []
    for p in panels:
        if p.shape[:2] != (h, w):
            p = np.array(Image.fromarray(p).resize((w, h), Image.LANCZOS))
        resized.append(p)

    row = np.concatenate(resized, axis=1)

    # 顶部加场景标题条
    label_bar = np.zeros((label_height, row.shape[1], 3), dtype=np.uint8)
    pil_bar = Image.fromarray(label_bar)
    draw = ImageDraw.Draw(pil_bar)
    panel_w = row.shape[1] // 4
    titles = ['GT', 'OUR', 'STM', 'HUGS']
    colors = [(220, 220, 220), ROLE_COLORS['our'], ROLE_COLORS['stm'], ROLE_COLORS['hugs']]
    for i, (title, color) in enumerate(zip(titles, colors)):
        x = i * panel_w + panel_w // 2 - 15
        draw.text((x, 4), title, fill=color)
    label_bar = np.array(pil_bar)

    return np.concatenate([label_bar, row], axis=0)


def process_scene(scene: str, role_dirs: dict[str, str | None],
                  out_root: Path, prefix: str):
    """处理单个场景的单个前缀（full / human）。"""
    out_dir = out_root / 'comparisons' / scene
    out_dir.mkdir(parents=True, exist_ok=True)

    # 找到各角色的 val 目录
    val_dirs: dict[str, Path | None] = {}
    for role, tmpl in role_dirs.items():
        if tmpl and tmpl.lower() != 'none':
            exp_path = tmpl.replace('{seq}', scene)
            vd = find_val_dir(exp_path)
            val_dirs[role] = vd
            if vd is None:
                print(f'  [{role}] val dir not found under: {exp_path}')
        else:
            val_dirs[role] = None

    # 找公共帧集合（以有 val 目录的角色为准，取并集）
    all_frames: set[int] = set()
    frame_maps: dict[str, dict[int, Path]] = {}
    for role, vd in val_dirs.items():
        if vd is not None:
            fm = get_final_frames(vd, prefix)
            frame_maps[role] = fm
            all_frames |= set(fm.keys())
        else:
            frame_maps[role] = {}

    if not all_frames:
        print(f'  [{scene}] no {prefix}_final frames found for any role, skipping')
        return 0

    n = 0
    for idx in sorted(all_frames):
        renders: dict[str, np.ndarray | None] = {}
        gt_img = None

        for role in ROLES:
            fp = frame_maps[role].get(idx)
            if fp is None:
                renders[role] = None
                continue
            render = split_half(fp, 'right')
            renders[role] = render
            if gt_img is None:
                gt_img = split_half(fp, 'left')

        if gt_img is None:
            continue

        row = make_compare_row(gt_img, renders)
        out_path = out_dir / f'{prefix}_compare_{idx:03d}.png'
        Image.fromarray(row).save(out_path)
        n += 1

    return n


def collect_symlinks(scene: str, role_dirs: dict[str, str | None], out_root: Path):
    """为每个角色创建符号链接到原始 val 目录。"""
    for role, tmpl in role_dirs.items():
        if not tmpl or tmpl.lower() == 'none':
            continue
        exp_path = tmpl.replace('{seq}', scene)
        vd = find_val_dir(exp_path)
        if vd is None:
            continue
        link = out_root / role / scene
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(vd.resolve())


def write_summary(out_root: Path, tag: str, scenes: list[str],
                  role_dirs: dict[str, str | None]):
    """生成 summary.md，记录本次验证的路径配置。"""
    lines = [
        f'# Validation: {tag}\n',
        '\n## Source directories\n',
    ]
    for role in ROLES:
        tmpl = role_dirs.get(role) or 'none'
        lines.append(f'- **{role.upper()}**: `{tmpl}`\n')
    lines += [
        '\n## Scenes\n',
        ', '.join(scenes) + '\n',
        '\n## Metrics\n',
        '> Fill in after all experiments complete.\n',
        '\n| Scene | OUR HUGS | OUR HUMAN | STM HUGS | STM HUMAN | HUGS HUGS | HUGS HUMAN |\n',
        '|---|---:|---:|---:|---:|---:|---:|\n',
    ]
    for s in scenes:
        lines.append(f'| {s} | | | | | | |\n')
    lines.append('| **avg** | | | | | | |\n')

    (out_root / 'summary.md').write_text(''.join(lines))


def main():
    parser = argparse.ArgumentParser(description='生成验证对比图')
    parser.add_argument('--tag', required=True, help='验证标识，如 vimo_inline_attn_20260606')
    parser.add_argument('--scenes', nargs='+',
                        default=['lab', 'bike', 'citron', 'jogging', 'parkinglot', 'seattle'])
    parser.add_argument('--our',  default='none',
                        help='OUR pipeline 实验根目录模板（用 {seq} 代替场景名）')
    parser.add_argument('--stm',  default='none',
                        help='STM pipeline 实验根目录模板')
    parser.add_argument('--hugs', default='none',
                        help='HUGS pipeline 实验根目录模板')
    parser.add_argument('--output-root', default='output',
                        help='输出根目录（默认 output/）')
    args = parser.parse_args()

    out_root = Path(args.output_root) / f'validation_{args.tag}'
    out_root.mkdir(parents=True, exist_ok=True)
    print(f'Output: {out_root}')

    role_dirs = {'our': args.our, 'stm': args.stm, 'hugs': args.hugs}

    for scene in args.scenes:
        print(f'\n=== {scene} ===')
        collect_symlinks(scene, role_dirs, out_root)
        for prefix in ['full', 'human']:
            n = process_scene(scene, role_dirs, out_root, prefix)
            if n:
                print(f'  {prefix}: {n} comparison frames saved')

    write_summary(out_root, args.tag, args.scenes, role_dirs)
    print(f'\nDone. Summary: {out_root}/summary.md')


if __name__ == '__main__':
    main()
