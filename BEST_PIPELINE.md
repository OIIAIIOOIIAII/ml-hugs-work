# 最优 Pipeline 记录

> **维护规则**：本文件只记录当前最优结果。有更好的结果时直接覆盖对应 section，不保留历史。
> 上次更新：2026-07-15

---

## 零、核心实验矩阵（论文主表口径）

当前论文主线只保留两类我们的核心实验：

1. **GT 对齐 Ours**：`depth_sup12k + AnchorAttention_6k` 两阶段 pipeline。
2. **粗对齐 Ours**：`v4_correct_inline_attn_18k` 单阶段 pipeline。

对应对比实验为每种对齐方式下的 HUGS baseline 和 STM baseline。按 `results_train.json` final 指标核对，以下 6 组实验均已跑齐 NeuMan 6 场景。

| 对齐方式 | 方法 | 实验角色 | HUMAN_PSNR avg | 状态 | 备注 |
|---|---|---|---:|---|---|
| GT 对齐 | HUGS | 原版 HUGS baseline | 18.8704 | 完成 6/6 | 原始 NeuMan COLMAP/GT 对齐 |
| GT 对齐 | STM | STM baseline | 19.4725 | 完成 6/6 | lab 使用 `stm_expE2_20260604`，其余为 `stm_gt_*_20k` |
| GT 对齐 | Ours | 我们的核心实验 | **19.6174** | 完成 6/6 | `depth_sup12k_transl_xyz_attn_6000_*` |
| 粗对齐 | HUGS | 粗对齐 HUGS baseline | 15.8039 | 完成 6/6 | `vimo_hugs_baseline_*` |
| 粗对齐 | STM | 粗对齐 STM baseline | 16.9020 | 完成 6/6 | `stm_vimo_*_20k`，lab 使用 `stm_expF2_20260604` |
| 粗对齐 | Ours | 我们的核心实验 | **17.2944** | 完成 6/6 | `v4_correct_inline_attn_18k` |

### 口径说明

- **GT 对齐主表**：HUGS / STM / Ours 均使用原始 NeuMan 场景坐标下的 GT 对齐口径。
- **粗对齐主表**：当前论文脚本 `scripts/make_paper_comparison.py` 采用的 baseline 路径为早期 `*_vimo`，Ours 为当前最优 `*_vimo_v4`。这是现有主表/对比图口径。
- 如果后续要求“所有方法完全同一 VIMO v4 对齐”，目前还没有 6 场景齐全的 HUGS-v4 与 STM-v4 baseline；需要另行补跑。

### 核心结果路径

| 对齐 | 方法 | 路径模式 |
|---|---|---|
| GT | HUGS | `output/human_scene/neuman/{seq}/hugs_trimlp/exp0_hugs_original_15000_*/*/results_train.json` |
| GT | STM | `Dynamic-Avatar-Scene-Rendering-from-Human-centric-Context/output_stm/human_scene/neuman/{seq}/hugs_trimlp/stm_gt_*_20k/*/results_train.json`（lab: `stm_expE2_20260604`） |
| GT | Ours | `output/human_scene/neuman/{seq}/hugs_trimlp/depth_sup12k_transl_xyz_attn_6000_*/*/results_train.json` |
| 粗对齐 | HUGS | `output/human_scene/neuman/{seq}_vimo/hugs_trimlp/vimo_hugs_baseline_*/*/results_train.json` |
| 粗对齐 | STM | `Dynamic-Avatar-Scene-Rendering-from-Human-centric-Context/output_stm/human_scene/neuman/{seq}_vimo/hugs_trimlp/stm_vimo_*_20k_20260607/*/results_train.json`（lab: `stm_expF2_20260604`） |
| 粗对齐 | Ours | `output/human_scene/neuman/{seq}_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/*/results_train.json` |

---

## 一、GT 对齐下最优（expG）

### Pipeline 名称

`depth_sup12k + AnchorAttention_6k`（两阶段）

### 对齐方式

GT 对齐（原始 NeuMan COLMAP，无 VIMO）

### 训练设置

| 参数 | Stage 1（12k） | Stage 2（6k） |
|------|:--------------|:-------------|
| 步数 | 11998 | 5998 |
| depth supervision | `depth_w: 0.05`（mono depth） | 继承 S1 |
| AnchorAttention | 关闭 | 开启 |
| module_start_iter | — | 0（S2 起始即生效） |
| correction_warmup_iters | — | 1000 |
| human_pooling | — | attention |
| scene_topk | — | 32 |
| lbs_w | 1000.0 | 1000.0 |
| S2 从 S1 ckpt 热启动 | — | ✓ |

### 关键配置文件

- S1：`cfg_files/debug/expg_repro_bike_s1_12k.yaml`（以 bike 为例，其他场景同结构）
- S2：`cfg_files/debug/expg_repro_bike_s2_6k.yaml`
- Lab 参考（单 YAML 15k inline 版）：`cfg_files/release/neuman/hugs_coarsealign_lab_expG.yaml`

### 各场景结果（NeuMan 6 场景）

| 场景 | HUMAN_PSNR | HUGS_PSNR | HUMAN_SSIM | HUMAN_LPIPS |
|------|:----------:|:---------:|:----------:|:-----------:|
| lab | 19.7831 | 26.3816 | — | — |
| bike | **20.4297** | 26.3024 | 0.6958 | 0.1332 |
| seattle | 19.6412 | 26.3995 | 0.6884 | 0.1275 |
| parkinglot | 19.8686 | 27.1149 | 0.7342 | 0.1369 |
| jogging | 17.8027 | 23.5966 | 0.5901 | 0.2214 |
| citron | **20.1794** | 26.2592 | 0.7156 | 0.1152 |

### 代表性输出目录（bike）

- bike S1：`output/human_scene/neuman/bike/hugs_trimlp/hugs_depth_sup_12000_bike_20260529/2026-05-29_00-57-34/`
- bike S2：`output/human_scene/neuman/bike/hugs_trimlp/depth_sup12k_transl_xyz_attn_6000_bike_20260529/2026-05-29_01-38-26/`
- jogging S1：`output/human_scene/neuman/jogging/hugs_trimlp/hugs_depth_sup_12000_jogging_20260601/2026-06-01_07-02-15/`
- jogging S2：`output/human_scene/neuman/jogging/hugs_trimlp/depth_sup12k_transl_xyz_attn_6000_jogging_20260601/2026-06-01_07-43-10/`
- citron S1：`output/human_scene/neuman/citron/hugs_trimlp/hugs_depth_sup_12000_citron_20260601/2026-06-01_05-21-56/`
- citron S2：`output/human_scene/neuman/citron/hugs_trimlp/depth_sup12k_transl_xyz_attn_6000_citron_20260601/2026-06-01_06-01-03/`

---

## 二、粗对齐（VIMO v4）下最优

### Pipeline 名称

`v4_correct_inline_attn_18k`（单阶段）

### 对齐方式

VIMO v4 粗对齐（无 GT，自动估算 scale/translation）

### 训练设置

| 参数 | 值 |
|------|:--|
| 步数 | 18000（单阶段） |
| depth supervision | `depth_w: 0.05`（mono depth，Pearson 相关系数 loss） |
| AnchorAttention | 开启，inline（与主训练同一阶段） |
| module_start_iter | 2000 |
| correction_start_iter | 2000 |
| correction_warmup_iters | 500 |
| human_pooling | attention |
| scene_topk | 32 |
| lbs_w | 1000.0 |
| gs_trainer.py | 原版 HUGS（full viewspace_points 传法，已回滚） |

### 关键配置文件

- `cfg_files/debug/hugs_bike_v4_correct_inline_attn_18k.yaml`（或各场景对应文件）
- `cfg_files/debug/hugs_jogging_v4_correct_inline_attn_18k.yaml`
- `cfg_files/debug/hugs_seattle_v4_correct_inline_attn_18k.yaml`

### 各场景结果

| 场景 | 峰值 HUMAN_PSNR（best step） | final HUMAN_PSNR（step 18k） | STM 基准 | vs STM（final） | 状态 |
|------|:---------------------------:|:---------------------------:|:--------:|:--------------:|:----:|
| bike | **18.9631**（step 16k） | 18.9626 | 18.6133 | **+0.35** ✓ | ✅ 完成 |
| jogging | **17.6535**（step 17k） | 17.5463 | 17.3776 | **+0.17** ✓ | ✅ 完成 |
| seattle | **16.9382**（step 2k） | 16.7521 | 16.7278 | **+0.02** ✓ | ✅ 完成 |
| lab | **18.0743**（step 5k） | 17.7588 | 17.4512 | **+0.31** ✓ | ✅ 完成 |
| parkinglot | **16.7302**（step 11k） | 15.2645 | 14.3660 | **+0.90** ✓ | ✅ 完成 ⚠️ |
| citron | **17.6701**（step 3k） | 17.4823 | 16.8763 | **+0.61** ✓ | ✅ 完成 |
| **avg（全6）** | — | **17.2944** | **16.9020** | **+0.39** | |

> ⚠️ parkinglot：step 11k 达峰值 16.73，但 step 14k 后出现晚期崩坏，final 降至 15.26。final 仍高于 STM(14.37) +0.90。
> 所有场景 STM 基准均使用原始 VIMO 对齐。lab 使用 `stm_expF2_20260604`（lab_vimo，HUM_PSNR=17.45）。

### 完整指标对比（全图渲染 + Human，final step）

> 全6场景均已完成，数据来自各场景 `results_train.json` final key。

**全图渲染（FULL）— 全帧合成质量**

| 场景 | OUR PSNR↑ | OUR SSIM↑ | OUR LPIPS↓ | STM PSNR↑ | STM SSIM↑ | STM LPIPS↓ | Δ PSNR |
|------|:---------:|:---------:|:----------:|:---------:|:---------:|:----------:|:------:|
| bike | **25.1030** | **0.8534** | **0.0949** | 24.1732 | 0.8290 | 0.1284 | **+0.93** |
| jogging | 23.0749 | 0.7761 | **0.1810** | **23.6131** | **0.7739** | 0.1988 | -0.54 |
| seattle | 25.1602 | 0.8554 | **0.0950** | **25.2448** | **0.8788** | 0.0983 | -0.08 |
| lab | 24.9539 | 0.9004 | 0.0828 | **25.0124** | **0.9130** | **0.0747** | -0.06 |
| parkinglot | **23.3722** | 0.8346 | 0.1530 | 22.7068 | **0.8522** | **0.1425** | **+0.67** |
| citron | 24.4869 | 0.8522 | **0.0945** | **24.5515** | **0.8538** | 0.1012 | -0.06 |
| **avg** | **24.3585** | **0.8454** | **0.1169** | **24.2170** | **0.8501** | **0.1240** | **+0.14** |

**Human 区域（HUM）— 人体渲染质量（核心指标）**

| 场景 | OUR PSNR↑ | OUR SSIM↑ | OUR LPIPS↓ | STM PSNR↑ | STM SSIM↑ | STM LPIPS↓ | Δ PSNR |
|------|:---------:|:---------:|:----------:|:---------:|:---------:|:----------:|:------:|
| bike | **18.9626** | **0.6553** | **0.1648** | 18.6133 | 0.6418 | 0.1700 | **+0.35** |
| jogging | **17.5463** | 0.5812 | **0.2354** | 17.3776 | **0.5818** | 0.2577 | **+0.17** |
| seattle | **16.7521** | 0.5983 | **0.1783** | 16.7278 | **0.6254** | 0.2261 | **+0.02** |
| lab | **17.7588** | **0.7287** | **0.1745** | 17.4512 | 0.7213 | 0.1820 | **+0.31** |
| parkinglot | **15.2645** | **0.6325** | **0.2291** | 14.3660 | 0.6253 | 0.2544 | **+0.90** |
| citron | **17.4823** | **0.6305** | **0.1496** | 16.8763 | 0.6074 | 0.1997 | **+0.61** |
| **avg** | **17.2944** | **0.6378** | **0.1886** | **16.9020** | **0.6338** | **0.2150** | **+0.39** |

### 代表性输出目录

- bike：`output/human_scene/neuman/bike_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/2026-06-14_12-52-08/`
- jogging：`output/human_scene/neuman/jogging_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/2026-06-14_17-39-01/`
- seattle：`output/human_scene/neuman/seattle_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/2026-06-14_17-28-48/`
- lab：`output/human_scene/neuman/lab_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/2026-06-15_01-16-10/`
- parkinglot：`output/human_scene/neuman/parkinglot_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/2026-06-15_03-29-29/`
- citron：`output/human_scene/neuman/citron_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/2026-06-15_05-32-10/`

---

## 三、面向传输的压缩 Pipeline

> 上次更新：2026-06-21
> 基于 v4_correct_inline_attn_18k（粗对齐最优 pipeline）训练的模型做推理侧压缩，不需要重训练。

### 传输场景设定

HUGS 传输分两层：

| 层级 | 内容 | 频率 |
|------|------|------|
| 一次性下载 | 场景 GS + canonical 人体 GS + LBS weights | 进入场景时 |
| 实时逐帧 | delta_mu correction + SMPL pose | 每帧 |

---

### 一次性传输大小（v4_correct baseline）

| 内容 | 大小 | 说明 |
|------|:----:|------|
| 场景 GS（SH order 3，6场景均值） | **~486 MB** | 约 215 万点，SH rest 占 76% |
| Canonical 人体 GS | **5.9 MB** | 110K 初始点，14 floats/点 |
| LBS skinning weights | **10.1 MB** | 110K × 24 joints × fp32 |
| **合计** | **~502 MB** | |

> 场景 GS 是大头（486 MB）。SH order 降阶（3→1）可将场景 GS 压缩约 4x → ~120 MB，但 PSNR 损失待评估。

---

### 每帧 correction 大小对比

Human GS 点数（训练后 densification 后，6场景实测）：

| 场景 | GS 点数 N |
|------|:---------:|
| bike | 515,804 |
| jogging | 457,757 |
| seattle | 539,986 |
| lab | 501,392 |
| parkinglot | 491,338 |
| citron | 593,754 |
| **平均** | **~516K** |

delta_mu shape 为 `[N, 3]`，各压缩层级对比：

| 压缩方案 | 大小/帧（均值） | 说明 |
|---------|:--------------:|------|
| fp32 无压缩 | **~6 MB** | N×3×4B，传输基线 |
| fp16 无压缩 | **~3 MB** | N×3×2B，简单量化 |
| int8 差分（无滤波，无熵编码） | **~1.5 MB** | N×3×1B，4x |
| int8 差分 + 熵编码（无滤波，实测理论下界） | **0~210 KB** | 场景间差异大（bike≈0, parkinglot≈210 KB） |
| **[推荐] savgol(w=11) + int8 差分 + 熵编码** | **估算 ~50~75 KB** | 高 ratio 场景显著改善 |

---

### 方案五：推理后 Savgol 低通滤波（最优压缩方案）

**原理**：对 delta_mu 序列 `[T, N, 3]` 沿时间轴做 Savitzky-Golay 平滑，减小帧间差分幅度，从而提升差分编码压缩率。

**推荐参数**：`window_length=11, polyorder=2`（沿 axis=0 滤波）

**Ratio 降低效果（diff_std / raw_std，越小越好）**：

| 场景 | 滤波前 ratio | 滤波后 ratio（w=11） | 降幅 |
|------|:-----------:|:-------------------:|:----:|
| bike | 0.287 | 0.127 | -56% |
| jogging | 0.307 | 0.108 | -65% |
| seattle | 0.482 | 0.254 | -47% |
| lab | 0.525 | 0.192 | -64% |
| parkinglot | 0.505 | 0.136 | -73% |
| citron | 0.329 | 0.143 | -57% |

**PSNR 影响（GPU 真实渲染对比 GT，val 帧）**：

| 场景 | baseline PSNR | 滤波后 PSNR | ΔdB |
|------|:------------:|:-----------:|:---:|
| bike | 24.929 | 24.926 | -0.003 |
| jogging | 23.257 | 23.260 | +0.004 |
| seattle | 25.120 | 25.122 | +0.002 |
| lab | 24.894 | 24.901 | +0.007 |
| parkinglot | 23.350 | 23.356 | +0.006 |
| citron | 24.453 | 24.448 | -0.004 |

**结论：PSNR 变化 ≤ ±0.01 dB，视为零损失。**

---

### 关键脚本

| 脚本 | 功能 |
|------|------|
| `scripts/analyze_delta_mu_temporal.py` | 从 checkpoint 提取 delta_mu 序列，计算 ratio 和 int8 熵下界 |
| `scripts/eval_method5_savgol.py` | 对已保存的 delta_mu_f16.npy 做 savgol 滤波，评估各 window 的 ratio 降低效果 |
| `scripts/eval_method5_psnr.py` | GPU 渲染对比：用滤波后 delta_mu 替换原始，计算 PSNR 差值 |

结果存放目录：`output/delta_mu_analysis/`
- `{seq}/delta_mu_f16.npy`：各场景完整 delta_mu 序列（fp16，约 1.5 GB/场景）
- `method5_savgol_results.json`：ratio 分析结果（所有场景 × 所有 window）
- `method5_psnr_results.json`：PSNR 评估结果（所有场景 × window=7,11,21）

---

### 工程注意事项

1. **Savgol 是非因果滤波器**：计算第 t 帧需要前后各 5 帧（w=11），不能用于实时流式推理。实时场景替换为因果版：EMA（指数移动平均）或单向滑动窗口。

2. **SMPL pose 可忽略不计**：每帧 300 B（75 个 float），与 delta_mu 相比完全可忽略。

3. **场景 GS 为主要瓶颈**：一次性传输的 ~502 MB 中，场景 GS 占 486 MB。若需进一步压缩，优先考虑 SH 阶数降阶或 GS 点数剪枝。

---

## STM 基准参考（各场景，完整指标）

> 各场景 STM 20k 步结果（final step）。lab 使用 v4 对齐，其余使用原始 VIMO 对齐。

| 场景 | FULL PSNR | FULL SSIM | FULL LPIPS | HUM PSNR | HUM SSIM | HUM LPIPS | 实验名 |
|------|:---------:|:---------:|:----------:|:--------:|:--------:|:---------:|--------|
| bike | 24.1732 | 0.8290 | 0.1284 | 18.6133 | 0.6418 | 0.1700 | stm_vimo_bike_20k |
| jogging | 23.6131 | 0.7739 | 0.1988 | 17.3776 | 0.5818 | 0.2577 | stm_vimo_jogging_20k |
| seattle | 25.2448 | 0.8788 | 0.0983 | 16.7278 | 0.6254 | 0.2261 | stm_vimo_seattle_20k |
| lab | 25.0124 | 0.9130 | 0.0747 | 17.4512 | 0.7213 | 0.1820 | stm_expF2_20260604 |
| parkinglot | 22.7068 | 0.8522 | 0.1425 | 14.3660 | 0.6253 | 0.2544 | stm_vimo_parkinglot_20k |
| citron | 24.5515 | 0.8538 | 0.1012 | 16.8763 | 0.6074 | 0.1997 | stm_vimo_citron_20k |
| **avg** | **24.2170** | **0.8501** | **0.1240** | **16.9020** | **0.6338** | **0.2150** | |
