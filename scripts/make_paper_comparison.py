"""
论文对比图生成脚本

生成 4行×3列 的方法对比图，4行分别为 GT / HUGS / STM / Ours，
3列为三个场景。支持可选的脚部放大框。

使用示例（粗对齐 coarse-aligned）:
  python scripts/make_paper_comparison.py \
      --mode coarse \
      --scenes bike citron parkinglot \
      --frames 3 5 2 \
      --foot-zoom \
      --out paper_figures/coarse_comparison.png

使用示例（精准对齐 GT-aligned）:
  python scripts/make_paper_comparison.py \
      --mode gt \
      --scenes bike citron jogging \
      --frames 4 2 6 \
      --out paper_figures/gt_comparison.png

关于 --frames:
  每个场景一个整数（0-based），对应 val/ 目录下的第 N 帧。
  val 图像格式：full_final_NNN.png（左半=GT，右半=渲染结果）
  STM 无 final 时使用最后一个 checkpoint 的图像。

关于 --foot-zoom:
  脚部放大框坐标通过 --foot-box 指定，格式：
    x1 y1 x2 y2（相对于单帧图像的像素坐标，从左上角计）
  若未指定，脚部框默认为图像下方中心区域。
  不同场景可以在 SCENE_FOOT_BOXES 中分别配置。
"""

import os
import sys
import glob
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


# ─────────────────────────────────────────────────────────────
# PSNR / 自动选帧
# ─────────────────────────────────────────────────────────────

def compute_psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """计算两张 uint8 RGB 图像的 PSNR（单位 dB）。"""
    mse = np.mean((img_a.astype(np.float32) - img_b.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def _load_gt_and_render(val_dir: Path, frame_idx: int):
    """
    从 val 目录加载某帧的 GT（左半）和渲染（右半）。
    返回 (gt_arr, render_arr) 或 (None, None)。
    """
    if val_dir is None or not val_dir.exists():
        return None, None

    # 优先 full_final，否则取最后一个含该帧号的 checkpoint
    final_path = val_dir / f"full_final_{frame_idx:03d}.png"
    if final_path.exists():
        candidates = [final_path]
    else:
        candidates = sorted(val_dir.glob(f"full_*_{frame_idx:03d}.png"))

    if not candidates:
        return None, None

    img = np.array(Image.open(candidates[-1]).convert("RGB"))
    w = img.shape[1] // 2
    return img[:, :w], img[:, w:]


def find_best_frame(ours_val_dir: Path, stm_val_dir: Path) -> tuple[int, list[dict]]:
    """
    遍历 Ours val 目录的所有帧，计算 PSNR(Ours)-PSNR(STM)，
    返回差值最大的帧索引以及每帧的统计列表。
    """
    # 从 Ours 目录枚举所有可用帧号
    if ours_val_dir is None or not ours_val_dir.exists():
        return 0, []

    pngs = sorted(ours_val_dir.glob("full_final_*.png"))
    if not pngs:
        pngs = sorted(ours_val_dir.glob("full_*_*.png"))

    frame_indices = []
    for p in pngs:
        try:
            idx = int(p.stem.split("_")[-1])
            if idx not in frame_indices:
                frame_indices.append(idx)
        except ValueError:
            pass
    frame_indices.sort()

    if not frame_indices:
        return 0, []

    stats = []
    for fi in frame_indices:
        gt_ours, render_ours = _load_gt_and_render(ours_val_dir, fi)
        gt_stm,  render_stm  = _load_gt_and_render(stm_val_dir,  fi)

        psnr_ours = compute_psnr(gt_ours, render_ours) if (gt_ours is not None and render_ours is not None) else float("nan")
        psnr_stm  = compute_psnr(gt_stm,  render_stm)  if (gt_stm  is not None and render_stm  is not None) else float("nan")
        diff = psnr_ours - psnr_stm if (not np.isnan(psnr_ours) and not np.isnan(psnr_stm)) else float("nan")
        stats.append({"frame": fi, "ours": psnr_ours, "stm": psnr_stm, "diff": diff})

    # 选差值最大（nan 视为 -inf）
    best = max(stats, key=lambda s: s["diff"] if not np.isnan(s["diff"]) else -float("inf"))
    return best["frame"], stats

# ─────────────────────────────────────────────────────────────
# 默认路径配置
# ─────────────────────────────────────────────────────────────

ROOT = Path("/workspace/nas_auto_backup/yuzilang/ml-hugs-work")
STM_ROOT = ROOT / "Dynamic-Avatar-Scene-Rendering-from-Human-centric-Context" / "output_stm"
OUR_ROOT = ROOT / "output"
DATA_ROOT = ROOT / "data" / "neuman" / "dataset"

# 每种 mode 下每个场景的实验路径配置
# val_dir 可以是完整路径，也可以是 glob 模式（取第一个匹配）
# last_ckpt: 当没有 full_final_*.png 时，使用最后一个 checkpoint（如 020000）

GT_MODE_PATHS = {
    # key: scene_name
    # val: dict with keys: hugs, stm, ours (各自指向 val/ 目录的 glob 模式)
    "bike": {
        "hugs": str(OUR_ROOT / "human_scene/neuman/bike/hugs_trimlp/exp0_hugs_original_15000_bike_20260601/*/val"),
        "stm":  str(STM_ROOT / "human_scene/neuman/bike/hugs_trimlp/stm_gt_bike_20k/*/val"),
        "ours": str(OUR_ROOT / "human_scene/neuman/bike/hugs_trimlp/depth_sup12k_transl_xyz_attn_6000_bike_20260529/*/val"),
    },
    "citron": {
        "hugs": str(OUR_ROOT / "human_scene/neuman/citron/hugs_trimlp/exp0_hugs_original_15000_citron_20260601/*/val"),
        "stm":  str(STM_ROOT / "human_scene/neuman/citron/hugs_trimlp/stm_gt_citron_20k/*/val"),
        "ours": str(OUR_ROOT / "human_scene/neuman/citron/hugs_trimlp/depth_sup12k_transl_xyz_attn_6000_citron_20260601/*/val"),
    },
    "parkinglot": {
        "hugs": str(OUR_ROOT / "human_scene/neuman/parkinglot/hugs_trimlp/exp0_hugs_original_15000_parkinglot_20260601/*/val"),
        "stm":  str(STM_ROOT / "human_scene/neuman/parkinglot/hugs_trimlp/stm_gt_parkinglot_20k/*/val"),
        "ours": str(OUR_ROOT / "human_scene/neuman/parkinglot/hugs_trimlp/depth_sup12k_transl_xyz_attn_6000_parkinglot_20260529/*/val"),
    },
    "jogging": {
        "hugs": str(OUR_ROOT / "human_scene/neuman/jogging/hugs_trimlp/exp0_hugs_original_15000_jogging_20260601/*/val"),
        "stm":  str(STM_ROOT / "human_scene/neuman/jogging/hugs_trimlp/stm_gt_jogging_20k/*/val"),
        "ours": str(OUR_ROOT / "human_scene/neuman/jogging/hugs_trimlp/depth_sup12k_transl_xyz_attn_6000_jogging_20260601/*/val"),
    },
    "seattle": {
        "hugs": str(OUR_ROOT / "human_scene/neuman/seattle/hugs_trimlp/exp0_hugs_original_15000_seattle_*/*/val"),
        "stm":  str(STM_ROOT / "human_scene/neuman/seattle/hugs_trimlp/stm_gt_seattle_20k/*/val"),
        "ours": str(OUR_ROOT / "human_scene/neuman/seattle/hugs_trimlp/depth_sup12k_transl_xyz_attn_6000_seattle_20260529/*/val"),
    },
    "lab": {
        "hugs": str(OUR_ROOT / "human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260527/*/val"),
        "stm":  str(STM_ROOT / "human_scene/neuman/lab/hugs_trimlp/stm_expE2_20260604/*/val"),
        "ours": str(OUR_ROOT / "human_scene/neuman/lab/hugs_trimlp/depth_sup12k_transl_xyz_attn_6000_lab_20260528/*/val"),
    },
}

COARSE_MODE_PATHS = {
    "bike": {
        "hugs": str(OUR_ROOT / "human_scene/neuman/bike_vimo/hugs_trimlp/vimo_hugs_baseline_bike_20260607/*/val"),
        "stm":  str(STM_ROOT / "human_scene/neuman/bike_vimo/hugs_trimlp/stm_vimo_bike_20k_20260607/*/val"),
        "ours": str(OUR_ROOT / "human_scene/neuman/bike_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/*/val"),
    },
    "citron": {
        "hugs": str(OUR_ROOT / "human_scene/neuman/citron_vimo/hugs_trimlp/vimo_hugs_baseline_citron_20260607/*/val"),
        "stm":  str(STM_ROOT / "human_scene/neuman/citron_vimo/hugs_trimlp/stm_vimo_citron_20k_20260607/*/val"),
        "ours": str(OUR_ROOT / "human_scene/neuman/citron_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/*/val"),
    },
    "parkinglot": {
        "hugs": str(OUR_ROOT / "human_scene/neuman/parkinglot_vimo/hugs_trimlp/vimo_hugs_baseline_parkinglot_20260607/*/val"),
        "stm":  str(STM_ROOT / "human_scene/neuman/parkinglot_vimo/hugs_trimlp/stm_vimo_parkinglot_20k_20260607/*/val"),
        "ours": str(OUR_ROOT / "human_scene/neuman/parkinglot_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/*/val"),
    },
    "jogging": {
        "hugs": str(OUR_ROOT / "human_scene/neuman/jogging_vimo/hugs_trimlp/vimo_hugs_baseline_jogging_20260607/*/val"),
        "stm":  str(STM_ROOT / "human_scene/neuman/jogging_vimo/hugs_trimlp/stm_vimo_jogging_20k_20260607/*/val"),
        "ours": str(OUR_ROOT / "human_scene/neuman/jogging_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/*/val"),
    },
    "seattle": {
        "hugs": str(OUR_ROOT / "human_scene/neuman/seattle_vimo/hugs_trimlp/vimo_hugs_baseline_seattle_20260607/*/val"),
        "stm":  str(STM_ROOT / "human_scene/neuman/seattle_vimo/hugs_trimlp/stm_vimo_seattle_20k_20260607/*/val"),
        "ours": str(OUR_ROOT / "human_scene/neuman/seattle_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/*/val"),
    },
    "lab": {
        "hugs": str(OUR_ROOT / "human_scene/neuman/lab_vimo/hugs_trimlp/vimo_hugs_baseline_lab_20260605/*/val"),
        "stm":  str(STM_ROOT / "human_scene/neuman/lab_vimo/hugs_trimlp/stm_expF2_20260604/*/val"),
        "ours": str(OUR_ROOT / "human_scene/neuman/lab_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/*/val"),
    },
}

# 各场景的 GT 图像目录（原始视频帧）
GT_IMAGE_DIRS = {
    seq: str(DATA_ROOT / seq / "images")
    for seq in ["bike", "citron", "parkinglot", "jogging", "seattle", "lab"]
}

# 脚部放大框默认配置（x1, y1, x2, y2，相对于单帧完整图像像素坐标）
# None 表示自动计算为图像下方中心 1/6 区域
SCENE_FOOT_BOXES = {
    "bike":       None,
    "citron":     None,
    "parkinglot": None,
    "jogging":    None,
    "seattle":    None,
    "lab":        None,
}

# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def resolve_val_dir(pattern: str) -> Path | None:
    """Glob 展开 val 目录路径，返回第一个存在的匹配。"""
    if pattern is None:
        return None
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return Path(matches[0])


def get_render_image(val_dir: Path, frame_idx: int) -> Image.Image | None:
    """
    从 val 目录加载渲染结果（full_final 图像的右半）。

    策略：
    1. 优先使用 full_final_{frame_idx:03d}.png
    2. 若无 final，使用最后一个可用 checkpoint 的图像
    3. 从图像中提取右半（渲染结果）
    """
    if val_dir is None or not val_dir.exists():
        return None

    # 1. 尝试 final
    final_path = val_dir / f"full_final_{frame_idx:03d}.png"
    if final_path.exists():
        img = Image.open(final_path).convert("RGB")
        w, h = img.size
        return img.crop((w // 2, 0, w, h))  # 右半 = 渲染结果

    # 2. 找最后一个 checkpoint
    all_pngs = sorted(val_dir.glob(f"full_*_{frame_idx:03d}.png"))
    if not all_pngs:
        # frame_idx 可能超出范围，尝试最近的
        all_any = sorted(val_dir.glob("full_*_*.png"))
        frame_files = [p for p in all_any if p.stem.split("_")[-1] == f"{frame_idx:03d}"]
        if not frame_files:
            all_any = sorted(val_dir.glob("full_*_000.png"))
            frame_files = all_any
        all_pngs = sorted(frame_files)

    if not all_pngs:
        return None

    img = Image.open(all_pngs[-1]).convert("RGB")
    w, h = img.size
    return img.crop((w // 2, 0, w, h))


def get_gt_image(val_dir: Path, frame_idx: int, gt_image_dir: str) -> Image.Image | None:
    """
    获取 GT 图像：
    1. 优先从 val/full_final_{frame_idx:03d}.png 取左半
    2. 若不存在，从原始数据目录按 frame_idx 匹配（需要与 val 帧对应）
    """
    if val_dir is not None and val_dir.exists():
        final_path = val_dir / f"full_final_{frame_idx:03d}.png"
        if final_path.exists():
            img = Image.open(final_path).convert("RGB")
            w, h = img.size
            return img.crop((0, 0, w // 2, h))  # 左半 = GT

        # 从其他 checkpoint 图中取 GT（GT 每帧相同）
        all_pngs = sorted(val_dir.glob(f"full_*_{frame_idx:03d}.png"))
        if all_pngs:
            img = Image.open(all_pngs[0]).convert("RGB")
            w, h = img.size
            return img.crop((0, 0, w // 2, h))

    # 从原始图像目录获取（需要知道 val_frame_idx 对应的实际文件名）
    # 由于 val 帧是从数据集均匀采样的，这里直接按序号索引
    if gt_image_dir and os.path.isdir(gt_image_dir):
        imgs = sorted(glob.glob(os.path.join(gt_image_dir, "*.png")) +
                      glob.glob(os.path.join(gt_image_dir, "*.jpg")))
        if imgs:
            step = max(1, len(imgs) // 10)
            candidates = imgs[::step][:10]
            if frame_idx < len(candidates):
                return Image.open(candidates[frame_idx]).convert("RGB")
            return Image.open(imgs[min(frame_idx * step, len(imgs) - 1)]).convert("RGB")
    return None


def add_zoom_box(
    img: Image.Image,
    box: tuple[int, int, int, int],
    zoom_size: tuple[int, int] = (120, 120),
    border_color: str = "yellow",
    border_width: int = 2,
    zoom_pos: str = "bottom-right",
) -> Image.Image:
    """
    在图像上绘制放大框。
    box: (x1, y1, x2, y2) 要放大的区域（相对于 img）
    zoom_size: 放大后的显示尺寸
    zoom_pos: 放大图贴在哪个角 (bottom-right / bottom-left / top-right / top-left)
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = box
    w, h = img.size

    # 在原图上画矩形框
    for i in range(border_width):
        draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=border_color)

    # 裁剪并放大
    region = img.crop((x1, y1, x2, y2))
    zw, zh = zoom_size
    zoomed = region.resize((zw, zh), Image.LANCZOS)

    # 贴到指定角落
    pad = 4
    if zoom_pos == "bottom-right":
        paste_x, paste_y = w - zw - pad, h - zh - pad
    elif zoom_pos == "bottom-left":
        paste_x, paste_y = pad, h - zh - pad
    elif zoom_pos == "top-right":
        paste_x, paste_y = w - zw - pad, pad
    else:  # top-left
        paste_x, paste_y = pad, pad

    # 放大图边框
    zoomed_draw = ImageDraw.Draw(zoomed)
    for i in range(border_width):
        zoomed_draw.rectangle([i, i, zw - 1 - i, zh - 1 - i], outline=border_color)

    img.paste(zoomed, (paste_x, paste_y))

    # 画连接线（从原框右下角到放大图左上角）
    if zoom_pos == "bottom-right":
        draw.line([(x2, y2), (paste_x, paste_y)], fill=border_color, width=1)
    elif zoom_pos == "bottom-left":
        draw.line([(x1, y2), (paste_x + zw, paste_y)], fill=border_color, width=1)

    return img


def auto_foot_box(img_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """自动计算脚部区域（图像下方中央 1/5 宽 × 1/6 高）。"""
    w, h = img_size
    x1 = int(w * 0.3)
    y1 = int(h * 0.75)
    x2 = int(w * 0.7)
    y2 = h - 2
    return (x1, y1, x2, y2)


def add_label(img: Image.Image, text: str, position: str = "left",
              font_size: int = 18, bg_color=(0, 0, 0, 160),
              text_color=(255, 255, 255)) -> Image.Image:
    """在图像左侧或顶部添加文字标签。"""
    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default(size=font_size)

    if position == "left":
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = 6
        y = (img.height - th) // 2
        draw.rectangle([x - 2, y - 2, x + tw + 4, y + th + 4], fill=bg_color)
        draw.text((x, y), text, fill=text_color, font=font)
    elif position == "top":
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = (img.width - tw) // 2
        y = 6
        draw.rectangle([x - 4, y - 2, x + tw + 4, y + th + 4], fill=bg_color)
        draw.text((x, y), text, fill=text_color, font=font)

    result = Image.alpha_composite(img, overlay)
    return result.convert("RGB")


# ─────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────

def build_comparison(args):
    mode = args.mode
    scenes = args.scenes
    mode_paths = GT_MODE_PATHS if mode == "gt" else COARSE_MODE_PATHS

    # 应用 override（先于自动选帧，以使 override 路径也参与计算）
    for method, scene, val_dir in getattr(args, "override", []):
        if scene not in mode_paths:
            mode_paths[scene] = {}
        mode_paths[scene][method] = val_dir

    if args.auto_frames:
        print("\n[自动选帧] 按 PSNR(Ours) - PSNR(STM) 最大化选帧...\n")
        frames = []
        for scene in scenes:
            scene_paths = mode_paths.get(scene, {})
            ours_val = resolve_val_dir(scene_paths.get("ours", ""))
            stm_val  = resolve_val_dir(scene_paths.get("stm",  ""))
            best_frame, stats = find_best_frame(ours_val, stm_val)
            frames.append(best_frame)
            print(f"  {scene}:")
            for s in stats:
                marker = " ◀ BEST" if s["frame"] == best_frame else ""
                print(f"    frame {s['frame']:3d}: Ours={s['ours']:.3f}  STM={s['stm']:.3f}  Δ={s['diff']:+.3f}{marker}")
        print(f"\n  选定帧: {dict(zip(scenes, frames))}\n")
    else:
        frames = args.frames if args.frames else [0] * len(scenes)
        assert len(frames) == len(scenes), \
            f"--frames 数量({len(frames)})必须与 --scenes 数量({len(scenes)})相同"
    row_labels = ["GT", "HUGS", "STM", "Ours"]
    methods = ["gt", "hugs", "stm", "ours"]

    # ── 收集所有格图像 ──────────────────────────────────────────
    # grid[row][col] = PIL Image
    grid = []

    target_h = None  # 统一行高

    for row_idx, method in enumerate(methods):
        row_imgs = []
        for col_idx, (scene, frame_idx) in enumerate(zip(scenes, frames)):
            scene_paths = mode_paths.get(scene, {})

            if method == "gt":
                # GT 从任意方法的 val 目录中取左半，或从原始图像目录
                ref_pattern = scene_paths.get("ours") or scene_paths.get("hugs")
                ref_val = resolve_val_dir(ref_pattern) if ref_pattern else None
                img = get_gt_image(ref_val, frame_idx, GT_IMAGE_DIRS.get(scene, ""))
            else:
                pattern = scene_paths.get(method)
                val_dir = resolve_val_dir(pattern) if pattern else None
                img = get_render_image(val_dir, frame_idx)

            if img is None:
                print(f"  [WARN] {method}/{scene}/frame{frame_idx}: 图像未找到，使用占位图")
                ph_w, ph_h = (640, 480)
                img = Image.new("RGB", (ph_w, ph_h), (80, 80, 80))
                draw = ImageDraw.Draw(img)
                draw.text((ph_w // 4, ph_h // 2), f"MISSING\n{method}/{scene}", fill=(200, 200, 0))
            else:
                print(f"  [OK]   {method}/{scene}/frame{frame_idx}: {img.size}")

            row_imgs.append(img)

        grid.append(row_imgs)

    # ── 统一尺寸（以第一个非空图为基准缩放） ───────────────────
    # 找所有图的最小宽度
    all_imgs = [img for row in grid for img in row]
    min_w = min(img.width for img in all_imgs)
    min_h = min(img.height for img in all_imgs)

    # 统一缩放：保持原始宽高比，以 min_h 为基准
    cell_h = min_h
    resized_grid = []
    for row in grid:
        resized_row = []
        for img in row:
            ratio = cell_h / img.height
            new_w = int(img.width * ratio)
            resized_row.append(img.resize((new_w, cell_h), Image.LANCZOS))
        resized_grid.append(resized_row)

    # 每列统一宽度（同一列内取最小宽）
    n_cols = len(scenes)
    n_rows = len(methods)
    col_widths = []
    for col in range(n_cols):
        col_widths.append(min(resized_grid[row][col].width for row in range(n_rows)))

    for row in range(n_rows):
        for col in range(n_cols):
            img = resized_grid[row][col]
            if img.width != col_widths[col]:
                # 居中裁剪到 col_widths[col]
                dx = (img.width - col_widths[col]) // 2
                resized_grid[row][col] = img.crop((dx, 0, dx + col_widths[col], cell_h))

    # ── 添加脚部放大框 ──────────────────────────────────────────
    if args.foot_zoom:
        for row in range(n_rows):
            for col, scene in enumerate(scenes):
                img = resized_grid[row][col]
                # 自定义 box 或自动
                if args.foot_box:
                    box = tuple(args.foot_box)
                elif SCENE_FOOT_BOXES.get(scene):
                    box = SCENE_FOOT_BOXES[scene]
                else:
                    box = auto_foot_box(img.size)
                # clamp box 到图像范围
                x1, y1, x2, y2 = box
                x1 = max(0, min(x1, img.width - 2))
                x2 = max(x1 + 1, min(x2, img.width))
                y1 = max(0, min(y1, img.height - 2))
                y2 = max(y1 + 1, min(y2, img.height))

                zoom_w = max(80, (x2 - x1) * 3)
                zoom_h = max(80, (y2 - y1) * 3)
                zoom_w = min(zoom_w, img.width // 2)
                zoom_h = min(zoom_h, img.height // 2)

                resized_grid[row][col] = add_zoom_box(
                    img, (x1, y1, x2, y2),
                    zoom_size=(zoom_w, zoom_h),
                    border_color=args.zoom_color,
                    border_width=2,
                    zoom_pos="bottom-right",
                )

    # ── 拼接图像 ───────────────────────────────────────────────
    gap = args.gap           # 格间距（像素）
    label_w = 160            # 行标签宽度（旋转后需稍宽，字体放大后增加）
    col_label_h = 140        # 底部场景名标签高度

    # 场景名放底部：图像区在上，场景名条在下
    total_w = label_w + sum(col_widths) + gap * (n_cols - 1)
    total_h = n_rows * cell_h + gap * (n_rows - 1) + col_label_h

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))

    # 行标签 + 图像格（行标签逆时针旋转90°）
    for row, label in enumerate(row_labels):
        y = row * (cell_h + gap)

        # 先在横向画布上写文字，再旋转90°逆时针
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 80)
        except Exception:
            font = ImageFont.load_default(size=80)
        txt_img = Image.new("RGB", (cell_h, label_w), (240, 240, 240))
        draw = ImageDraw.Draw(txt_img)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((cell_h - tw) // 2, (label_w - th) // 2), label,
                  fill=(30, 30, 30), font=font)
        label_img = txt_img.rotate(90, expand=True)   # 逆时针90°
        canvas.paste(label_img, (0, y))

        for col in range(n_cols):
            x = label_w + sum(col_widths[:col]) + gap * col
            canvas.paste(resized_grid[row][col], (x, y))

    # 列标签（场景名）— 放在最底部
    grid_h = n_rows * cell_h + gap * (n_rows - 1)
    for col, scene in enumerate(scenes):
        x = label_w + sum(col_widths[:col]) + gap * col
        label_img = Image.new("RGB", (col_widths[col], col_label_h), (240, 240, 240))
        draw = ImageDraw.Draw(label_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 90)
        except Exception:
            font = ImageFont.load_default(size=90)
        bbox = draw.textbbox((0, 0), scene.capitalize(), font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(((col_widths[col] - tw) // 2, (col_label_h - th) // 2),
                  scene.capitalize(), fill=(30, 30, 30), font=font)
        canvas.paste(label_img, (x, grid_h))

    # ── 保存 ───────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path), dpi=(300, 300))
    print(f"\n已保存: {out_path}  ({canvas.width}×{canvas.height} px)")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="生成论文对比图（GT / HUGS / STM / Ours）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["gt", "coarse"], default="coarse",
                        help="对比模式：gt=精准对齐(expG) / coarse=粗对齐(vimo v4)")
    parser.add_argument("--scenes", nargs="+",
                        default=["bike", "citron", "parkinglot"],
                        help="要展示的场景列表（默认3个）")
    parser.add_argument("--frames", nargs="+", type=int, default=None,
                        help="每个场景对应的帧索引（0-based，对应 val/ 目录中的第N帧）")
    parser.add_argument("--foot-zoom", action="store_true",
                        help="在每张图上添加脚部放大框")
    parser.add_argument("--foot-box", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"),
                        default=None,
                        help="全局脚部框坐标（像素），覆盖默认自动计算")
    parser.add_argument("--zoom-color", default="yellow",
                        help="放大框颜色（默认：yellow）")
    parser.add_argument("--gap", type=int, default=4,
                        help="格间距像素（默认：4）")
    parser.add_argument("--out", default="paper_figures/comparison.png",
                        help="输出图像路径（默认：paper_figures/comparison.png）")

    # 允许直接覆盖某个方法/场景的 val 目录
    parser.add_argument("--override", nargs=3, action="append",
                        metavar=("METHOD", "SCENE", "VAL_DIR"),
                        default=[],
                        help="覆盖某个格的 val 目录路径，可多次使用。"
                             "例：--override ours bike /path/to/val")

    parser.add_argument("--auto-frames", action="store_true",
                        help="自动按每帧 PSNR(Ours)-PSNR(STM) 最大化选帧，忽略 --frames 参数")

    args = parser.parse_args()

    print(f"模式: {args.mode}  场景: {args.scenes}  帧: {'auto' if args.auto_frames else args.frames}")
    build_comparison(args)


if __name__ == "__main__":
    main()
