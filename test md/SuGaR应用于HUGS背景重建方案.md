# SuGaR 应用于 HUGS 背景重建的实现方案

本文档面向 coding agent，用于说明如何将 SuGaR 的几何约束思想引入 HUGS 的背景/场景 Gaussian 重建分支。

重点约束：

- 本方案 **只作用于 HUGS 的背景/场景 Gaussians**。
- 本方案 **不作用于人体 Gaussians**，不改变 human canonical Gaussians、LBS weights、SMPL/SMPL-X pose deformation、人体 anchor binding 等人体分支逻辑。
- 目标不是完整复现 SuGaR 的 mesh pipeline，而是优先借鉴 SuGaR 中能让 3DGS 高斯更贴近表面、更少 floaters、更适合后续 human-scene attention 的部分。
- 项目中已 clone SuGaR 代码仓库，实现时可以参考其代码，但第一版应尽量保持对 HUGS 主流程的轻量侵入。

---

## 1. 背景与目标

当前 HUGS-style 重建中，人体和背景分别建模为 Gaussians。人体侧有 SMPL/LBS 等强先验，因此结构相对稳定；背景侧主要依赖 COLMAP/SfM 点云和 photometric loss，容易出现：

- scene Gaussians 空间分布松散；
- 接触区域附近有 floaters；
- 低 opacity / 半透明高斯残留；
- 背景高斯缺乏明确表面几何；
- 后续 human-scene attention 查询 scene Gaussians 时被噪声点干扰。

SuGaR 的核心贡献是让普通 3DGS 中无组织的 3D Gaussians 更贴近真实表面。我们希望把这个思想用于 HUGS 背景重建，使背景 Gaussians 更干净、更像 surface elements，从而提高后续接触建模与 attention 的可靠性。

第一版目标：

```text
HUGS scene Gaussian branch
  + SuGaR-style opacity regularization
  + SuGaR-style low-opacity pruning
  + SuGaR-style surface alignment / flatten regularization
  -> cleaner background Gaussians
  -> better scene candidates for human-scene attention
```

明确不做：

- 不替换人体 Gaussian 表示；
- 不对人体 Gaussian 加 SuGaR regularization；
- 不强制实现 SuGaR 的 mesh extraction；
- 不强制把背景 Gaussians bind 到 mesh；
- 不把 SuGaR loss 用作 contact loss；
- 不让该模块直接修改 SMPL pose / human root transform。

---

## 2. SuGaR 方法中需要借鉴的部分

SuGaR 原始 pipeline 大致为：

```text
1. 训练普通 3DGS
2. 对 opacity 加 entropy regularization，使 opacity 趋向 0 或 1
3. prune 低 opacity Gaussians
4. 加 surface alignment regularization，使 Gaussians 更贴近表面
5. 从 aligned Gaussians 中采样 level-set points
6. Poisson reconstruction 提取 mesh
7. 可选：把新 Gaussians 绑定到 mesh 上继续 refine
```

本项目第一版只借鉴步骤 2-4：

```text
HUGS background scene Gaussians
  -> opacity binary regularization
  -> low-opacity / low-confidence prune
  -> surface/flatten alignment regularization
```

暂不实现：

```text
level-set sampling
Poisson reconstruction
mesh binding
mesh-Gaussian joint refinement
```

如果后续需要显式 scene mesh 或 SDF，再考虑接入 SuGaR mesh extraction。

---

## 3. 集成位置

需要在 HUGS 训练代码中找到 scene/background Gaussian 分支。通常 HUGS 有两类 Gaussian：

```text
human_gaussians:
  canonical human space
  driven by LBS / pose deformation
  do not apply this module

scene_gaussians:
  static world/camera scene space
  background representation
  apply SuGaR-style module here only
```

coding agent 需要先定位以下对象或等价变量：

```text
scene_xyz          # [Ns, 3], background Gaussian centers
scene_scaling      # [Ns, 3], background Gaussian scales
scene_rotation     # [Ns, 4] or [Ns, 3, 3], background Gaussian rotations
scene_opacity      # [Ns, 1], background Gaussian opacity logits or activated opacity
scene_features     # SH/color, not directly touched by SuGaR regularization
```

如果 HUGS 代码中 scene/human 共用一个 GaussianModel，需要确认是否有 mask 或 type flag 区分：

```text
is_scene_gaussian: BoolTensor[N]
is_human_gaussian: BoolTensor[N]
```

所有新增 SuGaR-style loss 和 pruning 必须用 `is_scene_gaussian` 过滤。禁止对 human Gaussian 执行。

---

## 4. 新增配置项

建议新增配置，例如：

```yaml
scene_sugar:
  enabled: true

  # schedule
  start_iter: 7000
  opacity_entropy_start_iter: 7000
  surface_reg_start_iter: 9000
  prune_iter: 9000
  prune_interval: 3000
  stop_after_iter: -1

  # opacity
  lambda_opacity_entropy: 0.001
  opacity_prune_threshold: 0.05

  # scale / flatten
  lambda_scale_volume: 0.0001
  lambda_flatten: 0.001
  max_scale_ratio: 20.0

  # surface alignment
  lambda_surface_alignment: 0.001
  knn_k: 16
  knn_update_interval: 500
  sample_points_per_gaussian: 1

  # safety
  apply_to_scene_only: true
  detach_human_branch: true
  disable_mesh_extraction: true
```

推荐第一版默认只开启：

```yaml
lambda_opacity_entropy: 0.001
lambda_scale_volume: 0.0001
lambda_flatten: 0.001
lambda_surface_alignment: 0.0
```

也就是先实现最稳的轻量版；确认训练正常后，再打开 `lambda_surface_alignment`。

---

## 5. 训练 schedule

不要从训练一开始就施加 SuGaR-style 约束。背景 Gaussians 需要先通过 photometric loss 找到基本位置。

推荐 schedule：

```text
Stage 0: 原始 HUGS warmup
iter 0 - 7000
  使用原始 HUGS loss
  不加 SuGaR-style loss
  不额外 prune

Stage 1: opacity binary regularization
iter 7000 - 9000
  开启 scene opacity entropy
  目标：让背景 opacity 从半透明状态向 0/1 分化

Stage 2: prune low-opacity scene Gaussians
iter 9000
  只 prune scene Gaussians
  不 prune human Gaussians
  删除 opacity 过低且可见性/贡献低的背景高斯

Stage 3: surface-aware refinement
iter 9000 - end
  开启 scale/flatten regularization
  可选开启 surface alignment regularization
  目标：让背景高斯更扁、更贴近表面

Stage 4: periodic cleanup
iter 12000, 15000, ...
  周期性 prune scene floaters
  同步更新 scene spatial index / KNN cache
```

如果当前 HUGS 已有 densification/pruning schedule，需要把 SuGaR prune 插入到 scene densification 之后，并确保不破坏 optimizer state。

---

## 6. Loss 设计

### 6.1 总 loss

新增项只加到 scene/background 分支：

```text
L_total =
  L_hugs_original
  + lambda_opacity_entropy * L_scene_opacity_entropy
  + lambda_scale_volume * L_scene_scale_volume
  + lambda_flatten * L_scene_flatten
  + lambda_surface_alignment * L_scene_surface_alignment
```

其中：

```text
L_hugs_original:
  原始 HUGS photometric / SSIM / LPIPS / mask / human regularization 等

L_scene_*:
  只对 scene Gaussians 计算
```

禁止把 `L_scene_*` 计算到 human Gaussians 上。

### 6.2 Opacity entropy regularization

目的：减少大量半透明背景高斯，让 opacity 趋向 0 或 1，方便后续 prune。

如果 `alpha = sigmoid(opacity_logits)`，则：

```text
L_scene_opacity_entropy =
  mean[- alpha * log(alpha + eps) - (1 - alpha) * log(1 - alpha + eps)]
```

实现要点：

```python
alpha = scene_gaussians.get_opacity()  # activated opacity in [0, 1]
eps = 1e-6
entropy = -alpha * torch.log(alpha + eps) - (1.0 - alpha) * torch.log(1.0 - alpha + eps)
loss_opacity_entropy = entropy.mean()
```

注意：

- 只对 scene Gaussians 计算；
- 不要对 human opacity 施加该 loss；
- 该 loss 不宜过大，否则可能牺牲渲染质量；
- 第一版 `lambda_opacity_entropy` 可从 `1e-4` 到 `1e-3` 搜索。

### 6.3 Low-opacity pruning

目的：清除低 opacity 背景 floaters 和无贡献高斯。

基础规则：

```text
prune scene Gaussian if:
  opacity < opacity_prune_threshold
```

更稳的规则：

```text
prune scene Gaussian if:
  opacity < opacity_prune_threshold
  and visibility_count < min_visibility
```

如果当前代码没有 visibility_count，第一版只用 opacity threshold。

实现要点：

```python
scene_alpha = scene_gaussians.get_opacity().squeeze(-1)
prune_mask_scene = scene_alpha < opacity_prune_threshold

# 如果 scene/human 在同一 GaussianModel 中：
prune_mask_all = torch.zeros_like(all_alpha, dtype=torch.bool)
prune_mask_all[is_scene_gaussian] = prune_mask_scene
gaussians.prune_points(prune_mask_all)

# 如果 scene/human 是两个 model：
scene_gaussians.prune_points(prune_mask_scene)
```

禁止：

```python
human_gaussians.prune_points(prune_mask_human_from_sugar)
```

### 6.4 Scale volume regularization

目的：抑制背景中用于“糊颜色”的大体积高斯。

```text
L_scene_scale_volume = mean(sx * sy * sz)
```

实现：

```python
scale = scene_gaussians.get_scaling()  # [Ns, 3], positive
loss_scale_volume = torch.prod(scale, dim=-1).mean()
```

注意：

- 权重要小；
- 它只抑制过大高斯，不能单独保证贴表面；
- 不要对 human scales 使用该项。

### 6.5 Flatten regularization

目的：鼓励背景高斯呈现 surface-like anisotropy，即一个轴较薄，另外两个轴沿切平面展开。

简单实现：

```text
s_sorted = sort(sx, sy, sz)
L_scene_flatten = mean(s_min / (s_mid + eps))
```

或：

```text
L_scene_flatten = mean(s_min)
```

推荐第一版使用：

```python
scale = scene_gaussians.get_scaling()
s_sorted, _ = torch.sort(scale, dim=-1)
s_min = s_sorted[:, 0]
s_mid = s_sorted[:, 1]
loss_flatten = (s_min / (s_mid + 1e-6)).mean()
```

该项含义：

- 当最小尺度比中间尺度小，高斯更扁；
- 过强会伤害渲染质量；
- 第一版权重建议很小，例如 `1e-4` 到 `1e-3`。

### 6.6 Surface alignment regularization

SuGaR 更完整的 surface alignment 会基于 Gaussian density / SDF approximation，让高斯分布更符合局部表面。该项实现复杂度高于上面几项，建议第二步实现。

可参考 SuGaR 代码中与 regularization 相关的实现，重点查找：

```text
regularization
density
sdf
nearest neighbors
surface alignment
```

概念上，该项做的是：

```text
1. 对每个 scene Gaussian 找 KNN 邻域
2. 在 Gaussian 附近采样点 p
3. 计算真实 Gaussian density d(p)
4. 根据“理想 flat Gaussian surface”构造目标 density / SDF
5. 最小化二者差异
```

第一版可以使用轻量替代项：

```text
L_scene_surface_alignment =
  L_scene_flatten
  + L_scene_scale_volume
```

也就是说，先不实现完整 SuGaR SDF density loss。等轻量版稳定后，再参考 SuGaR 源码补全。

---

## 7. 与 HUGS 数据流的关系

### 7.1 原始 HUGS 渲染保持不变

渲染仍然是：

```text
deformed human Gaussians + static scene Gaussians
  -> concatenate / merge
  -> Gaussian rasterizer
  -> rendered RGB / depth / alpha
  -> original HUGS losses
```

SuGaR-style 模块只在训练 loss 和 scene pruning 阶段额外作用于 `scene_gaussians`。

### 7.2 不改变 human branch

以下模块不应被本方案修改：

```text
human canonical Gaussian initialization
human Gaussian LBS weights
human pose deformation
SMPL / SMPL-X pose optimization
human anchor binding
human Gaussian densification inheritance
human-specific mask / LBS regularization
```

### 7.3 与后续 contact-aware attention 的关系

该模块输出的是更干净的 scene Gaussians。后续 attention 可以继续按原方案查询：

```text
human contact anchor world position
  -> query nearby scene Gaussians
  -> scene token encoding
  -> human-scene attention
```

但建议在 attention 查询前增加 scene confidence filter：

```text
scene opacity > threshold
scene scale volume < threshold
scene visible in at least one/multiple views
```

这样可以避免 attention 被剩余 floaters 干扰。

---

## 8. 代码实现任务拆分

### Task 1: 定位 scene Gaussian 对象

需要找到 HUGS 中背景高斯的 model/class/变量。

输出：

```text
scene_gaussians.get_xyz
scene_gaussians.get_scaling
scene_gaussians.get_rotation
scene_gaussians.get_opacity
scene_gaussians.prune_points(mask)
```

如果 scene/human 共用一个 model，则新增或确认：

```text
is_scene_gaussian
is_human_gaussian
```

### Task 2: 新增 scene_sugar 配置

新增配置入口，例如：

```text
configs/*.yaml
arguments / parser
train options
```

需要支持：

```text
--scene_sugar_enabled
--scene_sugar_start_iter
--scene_sugar_lambda_opacity_entropy
--scene_sugar_lambda_scale_volume
--scene_sugar_lambda_flatten
--scene_sugar_prune_iter
--scene_sugar_prune_interval
--scene_sugar_opacity_prune_threshold
```

### Task 3: 实现 loss 函数

建议新建文件：

```text
utils/scene_sugar_regularizer.py
```

或放在现有 loss utils 中。

函数建议：

```python
def scene_opacity_entropy_loss(scene_gaussians, eps=1e-6):
    ...

def scene_scale_volume_loss(scene_gaussians):
    ...

def scene_flatten_loss(scene_gaussians, eps=1e-6):
    ...

def compute_scene_sugar_loss(scene_gaussians, cfg, iteration):
    ...
```

返回：

```python
loss_dict = {
    "scene_sugar/opacity_entropy": loss_opacity_entropy,
    "scene_sugar/scale_volume": loss_scale_volume,
    "scene_sugar/flatten": loss_flatten,
    "scene_sugar/total": loss_total,
}
```

### Task 4: 接入 training loop

在 HUGS training loop 中：

```python
loss = original_hugs_loss

if cfg.scene_sugar.enabled and iteration >= cfg.scene_sugar.start_iter:
    scene_sugar_loss, scene_sugar_logs = compute_scene_sugar_loss(
        scene_gaussians=scene_gaussians,
        cfg=cfg.scene_sugar,
        iteration=iteration,
    )
    loss = loss + scene_sugar_loss
```

注意：

- `scene_gaussians` 必须只包含背景；
- 如果传入的是 all_gaussians，必须用 scene mask 过滤；
- logging 中明确显示 scene-only 数量和 loss。

### Task 5: 实现 scene-only prune

建议函数：

```python
def prune_scene_gaussians_by_sugar(scene_gaussians, cfg, iteration):
    if iteration < cfg.prune_iter:
        return
    if cfg.prune_interval > 0 and (iteration - cfg.prune_iter) % cfg.prune_interval != 0:
        return

    alpha = scene_gaussians.get_opacity().squeeze(-1)
    prune_mask = alpha < cfg.opacity_prune_threshold
    scene_gaussians.prune_points(prune_mask)
```

如果 scene/human 共用 model：

```python
prune_mask_all = torch.zeros([N_all], dtype=torch.bool, device=device)
prune_mask_all[scene_indices] = prune_mask_scene
all_gaussians.prune_points(prune_mask_all)
```

必须保证：

```text
human Gaussian count 不因 scene_sugar prune 改变
human anchor binding 不因 scene_sugar prune 被重排或破坏
```

### Task 6: optimizer state 同步

如果 HUGS 的 `prune_points` 已经处理 optimizer state，直接复用。

如果需要手动同步，应确保：

```text
xyz optimizer state
scaling optimizer state
rotation optimizer state
opacity optimizer state
features optimizer state
```

都按 prune mask 删除对应 scene 参数。

---

## 9. 参考 SuGaR 代码的方式

由于项目中已 clone SuGaR 仓库，coding agent 可在实现时参考其源码。建议将 SuGaR repo 路径作为环境变量或配置：

```text
SUGAR_REPO=/path/to/SuGaR
```

优先参考：

```text
1. SuGaR 中 opacity regularization / entropy 的实现
2. SuGaR 中 pruning 低 opacity Gaussians 的时机和阈值
3. SuGaR 中 regularization stage 的训练 schedule
4. SuGaR 中 scale / covariance / normal 的处理方式
5. SuGaR 中 KNN / nearest neighbor cache 的实现
```

不要直接照搬：

```text
1. mesh extraction 主流程
2. Poisson reconstruction 依赖
3. mesh-bound Gaussian refinement
4. 完整 scene export pipeline
```

除非后续明确需要 mesh。

---

## 10. Debug 与可视化要求

为了验证该模块确实改善背景几何，需要新增以下日志和导出。

### 10.1 Scalar logs

每隔固定 iteration 记录：

```text
scene_sugar/num_scene_gaussians
scene_sugar/mean_opacity
scene_sugar/median_opacity
scene_sugar/low_opacity_ratio
scene_sugar/mean_scale_volume
scene_sugar/mean_flatten_ratio
scene_sugar/loss_opacity_entropy
scene_sugar/loss_scale_volume
scene_sugar/loss_flatten
scene_sugar/pruned_count
```

### 10.2 PLY exports

导出背景高斯中心点 PLY：

```text
outputs/scene_sugar_debug/scene_gaussians_iter0000.ply
outputs/scene_sugar_debug/scene_gaussians_iter7000.ply
outputs/scene_sugar_debug/scene_gaussians_iter9000_before_prune.ply
outputs/scene_sugar_debug/scene_gaussians_iter9000_after_prune.ply
outputs/scene_sugar_debug/scene_gaussians_iter15000.ply
```

PLY 上色建议：

```text
opacity 高 -> 亮色
opacity 低 -> 暗色
scale volume 大 -> 红色
scale volume 小 -> 蓝/绿
```

不要导出或修改 human Gaussians，除非只是作为独立只读参考显示。

### 10.3 Contact area inspection

重点查看：

```text
脚底附近 scene Gaussians 是否从空气中减少
地面附近是否更集中
人体 silhouette 周围是否减少 background ghost
anchor -> scene neighbor query 是否更稳定
```

---

## 11. 验收标准

第一版实现完成后，应满足：

1. 训练能正常跑通。
2. 开启 `scene_sugar.enabled=false` 时，结果与原 HUGS 基本一致。
3. 开启 `scene_sugar.enabled=true` 时，只影响 scene/background Gaussians。
4. human Gaussian 数量、LBS weights、anchor binding 不被 scene_sugar prune 改动。
5. scene low-opacity Gaussians 数量下降。
6. scene Gaussian scale volume 平均值不过度膨胀。
7. 接触区域附近 scene neighbor query 更少命中明显 floaters。
8. 渲染质量不能明显崩坏；PSNR/SSIM 可小幅波动，但不能出现大面积背景缺失。

如果出现背景渲染质量明显下降，优先调低：

```text
lambda_opacity_entropy
lambda_scale_volume
lambda_flatten
opacity_prune_threshold
```

---

## 12. 推荐第一版最小实现

如果只想最快落地，先实现以下最小版本：

```text
1. scene-only opacity entropy loss
2. scene-only opacity threshold prune
3. scene-only scale volume loss
4. scene-only flatten loss
5. debug logs + PLY export
```

最小版本不实现：

```text
KNN surface alignment
SDF density regularization
mesh extraction
mesh-bound refinement
```

推荐默认 schedule：

```text
iter < 7000:
  no scene_sugar

7000 <= iter < 9000:
  opacity entropy only

iter == 9000:
  prune scene opacity < 0.05

iter >= 9000:
  opacity entropy + scale volume + flatten

every 3000 iters after 9000:
  prune scene opacity < 0.03 or 0.05
```

推荐初始权重：

```text
lambda_opacity_entropy = 1e-4
lambda_scale_volume = 1e-5
lambda_flatten = 1e-4
opacity_prune_threshold = 0.03
```

如果效果太弱，再逐步提高到：

```text
lambda_opacity_entropy = 1e-3
lambda_scale_volume = 1e-4
lambda_flatten = 1e-3
opacity_prune_threshold = 0.05
```

---

## 13. 后续增强版本

最小版本稳定后，可以考虑接入更接近 SuGaR 原文的 surface alignment：

```text
1. 为 scene Gaussians 建 KNN cache
2. 每隔 knn_update_interval 更新 nearest neighbors
3. 在每个 scene Gaussian 附近采样点
4. 计算局部 Gaussian density
5. 构造 SuGaR-style SDF/density regularization
6. 只对 scene Gaussians 加该 loss
```

增强版本仍然不应触碰 human Gaussians。

如果后续需要更稳定的 scene surface proxy 给 attention 使用，可以额外导出：

```text
filtered_scene_gaussians.ply
scene_surface_candidates.npy
scene_attention_candidates.pt
```

过滤规则：

```text
opacity > 0.1
scale_volume < percentile_90
flatten_ratio < threshold
visible_count >= min_visible_count
```

---

## 14. 一句话实现原则

把 SuGaR 当作 **背景高斯几何清理器**，而不是人体建模方法：

```text
SuGaR-style regularization cleans scene Gaussians.
HUGS human branch remains untouched.
Contact-aware attention consumes cleaner scene candidates.
```

---

## 15. Lab 质量实验记录：SuGaR 背景与 Anchor-Attention 续训

本次记录固定在 NeuMan `lab` 场景，评估口径使用 HUGS 训练脚本输出的 validation metrics。指标方向为：PSNR/SSIM 越高越好，LPIPS 越低越好。表中粗体表示相对对照或上一阶段提升，斜体表示下降。

### 15.1 15000-step 背景重建对照

| Method | Steps | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUGS_HUMAN_PSNR | HUGS_HUMAN_SSIM | HUGS_HUMAN_LPIPS | Output |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| HUGS original | 15000 | 25.9582 | 0.9147 | 0.0710 | 18.8005 | 0.7580 | 0.1532 | `output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54` |
| SuGaR-HUGS background | 15000 | **26.1260** | *0.9143* | **0.0707** | **18.9424** | **0.7606** | **0.1494** | `output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_15000_lab_20260521/2026-05-21_21-21-45` |

### 15.2 不同背景重建下的 attention 阶段指标

| Background / Attention pipeline | Stage | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUGS_HUMAN_PSNR | HUGS_HUMAN_SSIM | HUGS_HUMAN_LPIPS | Output |
|---|---|---:|---:|---:|---:|---:|---:|---|
| HUGS background + attention | 12000 before attention | 25.9258 | 0.9127 | 0.0719 | 18.7633 | 0.7488 | 0.1537 | `output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_correction_15000_20260521/2026-05-21_02-45-33` |
| HUGS background + attention | 12000 + 3000 attention | 26.0527 | 0.9149 | 0.0705 | 18.9251 | 0.7609 | 0.1491 | same as above |
| HUGS background + attention | 12000 + 3000 + 3000 attention | 26.4029 | 0.9210 | 0.0640 | 19.2466 | 0.7701 | 0.1408 | `output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_extend_from15000_plus3000_20260521/2026-05-21_19-49-01` |
| SuGaR-HUGS background + attention | 12000 before attention | **25.9340** | **0.9130** | *0.0721* | **18.8669** | **0.7588** | **0.1512** | `output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_hugs_12000_lab_20260521_rerun/2026-05-21_22-55-25` |
| SuGaR-HUGS background + attention | 12000 + 3000 attention | **26.2388** | **0.9191** | **0.0659** | **19.1240** | **0.7689** | **0.1422** | `output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_hugs12000_anchor_attention_plus3000_stage1_20260521/2026-05-21_23-34-40` |
| SuGaR-HUGS background + attention | 12000 + 3000 + 3000 attention | *26.2989* | *0.9201* | *0.0648* | *19.1753* | *0.7697* | *0.1414* | `output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_hugs12000_anchor_attention_plus3000_stage2_20260522/2026-05-22_00-05-47` |

### 15.3 Attention 阶段增长幅度

| Background | Attention interval | Delta PSNR | Delta SSIM | Delta LPIPS | Delta HUMAN_PSNR | Delta HUMAN_SSIM | Delta HUMAN_LPIPS |
|---|---|---:|---:|---:|---:|---:|---:|
| HUGS background | 12000 -> 15000 | **+0.1269** | **+0.0022** | **-0.0014** | **+0.1618** | **+0.0121** | **-0.0046** |
| HUGS background | 15000 -> 18000 | **+0.3502** | **+0.0061** | **-0.0065** | **+0.3215** | **+0.0092** | **-0.0083** |
| HUGS background | 12000 -> 18000 | **+0.4771** | **+0.0083** | **-0.0079** | **+0.4833** | **+0.0213** | **-0.0129** |
| SuGaR-HUGS background | 12000 -> 15000 | **+0.3049** | **+0.0061** | **-0.0061** | **+0.2572** | **+0.0101** | **-0.0090** |
| SuGaR-HUGS background | 15000 -> 18000 | **+0.0600** | **+0.0010** | **-0.0011** | **+0.0512** | **+0.0009** | **-0.0009** |
| SuGaR-HUGS background | 12000 -> 18000 | **+0.3649** | **+0.0071** | **-0.0072** | **+0.3084** | **+0.0110** | **-0.0099** |

### 15.4 结论

在 15000-step 背景重建本身，SuGaR-HUGS 相比原版 HUGS 的全图 PSNR、LPIPS 和人体区域指标更好，SSIM 略低。将 attention 接在 12000-step 背景之后时，SuGaR-HUGS 背景的第一个 3000-step attention 增益更明显，说明更干净的背景高斯能帮助 attention/correction 更快获得收益。第二个 3000-step attention 继续带来小幅 LPIPS 改善，但 PSNR/SSIM 基本进入平台期；而原 HUGS 背景在第二个 3000-step extension 中最终指标更高。

