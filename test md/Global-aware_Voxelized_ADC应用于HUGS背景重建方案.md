# Global-aware Voxelized ADC 应用于 HUGS 背景重建的实现方案

本文档面向 coding agent，用于说明如何在当前 Contact-aware / HSI 4D Gaussian Splatting 项目中，新增第三种静态背景重建方案：基于论文 **Global-aware Voxelized Adaptive Density Control for Efficient and Compact 3D Gaussian Splatting** 的背景 Gaussian 致密化控制。

该方案与现有两种背景重建方式平行存在：

```text
background_reconstruction:
  1. hugs_original_scene
     使用 HUGS 原版 scene/background Gaussian 重建流程

  2. sugar_scene
     使用 SuGaR-style opacity / pruning / flatten / surface alignment 约束

  3. global_aware_voxel_adc_scene
     使用 Global-aware Gradient Rescaling + Voxel-based Densification
```

最重要的约束：

- 本方案 **只作用于 HUGS 的静态背景/场景 Gaussians**。
- 本方案 **不作用于前景人体 Gaussians**。
- 本方案 **不修改 human canonical Gaussians、LBS weights、SMPL/SMPL-X pose deformation、人体 anchor binding、contact-aware attention 主模块**。
- 本方案的目标是改进 scene/background Gaussian 的致密化策略，让背景 Gaussian 分布更合理、更紧凑、更少漂浮物，从而为后续 human-scene attention 提供更干净的场景候选。
- 本方案不是 SuGaR 的替代，也不是 mesh extraction 方案；它是和 HUGS 原版背景、SuGaR 背景并列的第三条背景重建路线。

---

## 1. 方案定位

当前项目的主线目标是 **人体-场景交互场景下的单目动态 4D Gaussian Splatting 重建**。HUGS-style 方法通常将人体和静态场景分别表示成 Gaussian：

```text
human_gaussians:
  canonical human space
  通过 SMPL/SMPL-X pose、LBS、deformation 映射到每帧

scene_gaussians:
  static world space
  表示背景、地面、墙面、家具、环境结构等
```

目前我们关心的一个问题是：背景 Gaussian 的几何质量不稳定，点分布杂乱，接触区域附近有 floaters。这会进一步影响后续 contact-aware 或 human-scene attention 模块，因为人体局部 anchor 查询到的 scene candidates 可能不是可靠表面，而是噪声点、半透明残留或错误致密化产生的漂浮 Gaussian。

已有的 SuGaR 背景方案主要从 **表面约束、opacity 二值化、低 opacity pruning、flatten regularization** 角度让 scene Gaussians 更像 surface elements。

本方案从另一个角度入手：**改造 scene Gaussian 的 adaptive density control，让系统只在真正缺几何/缺细节的静态背景区域补 Gaussian，并抑制噪声区域、弱纹理区域、漂浮伪影附近的错误增殖。**

因此它适合作为第三个背景重建实验分支：

```text
HUGS baseline
  -> 保持 human branch 不变
  -> 只替换 scene branch 的 densification decision
  -> 得到更合理、更紧凑的 scene Gaussians
  -> 后续 contact-aware attention 查询更稳定
```

---

## 2. 论文方法解决的问题

原始 3DGS / HUGS scene branch 的 densification 通常依赖每个 Gaussian 的 view-space position gradient。梯度大，就触发 clone 或 split。这种策略有三个问题。

### 2.1 Local Enclosure Bias

原始 position gradient 是局部信号，缺少全局参考。

同样大小的梯度，在不同区域含义可能完全不同：

- 在细结构区域，可能表示真实几何缺失，需要补点。
- 在弱纹理背景区域，可能只是颜色/opacity 没优化好。
- 在边缘 aliasing 或噪声区域，可能是伪高频，不应该继续加点。
- 在已有 floaters 附近，高梯度可能会诱导更多错误 Gaussian。

如果所有区域使用同一个 gradient threshold，就容易误判。

### 2.2 Density-dependent Bias

原始 ADC 是 per-Gaussian 触发。

某个空间区域初始 Gaussian 越多，就有越多 Gaussian 可能触发 densification；某个空间区域初始 Gaussian 越少，即使真实缺几何，也可能因为候选 Gaussian 少而增长慢。

这会导致：

```text
已经密的地方继续变密
真正稀疏但需要补几何的地方补不起来
```

这对我们项目中的背景重建尤其危险，因为 Human3R / COLMAP / SfM 初始化的 scene points 本来就可能不均匀。

### 2.3 Global Density Misalignment

上面两个偏差会共同导致全局 Gaussian 分布和真实场景复杂度不匹配：

- 背景弱纹理区域出现冗余 Gaussians。
- 接触区域附近出现漂浮点。
- 真实边缘、支撑面、细结构没有被充分补点。
- Gaussian 数量增加，但 scene geometry 不一定变好。
- human-scene attention 查询时被错误 scene candidates 干扰。

---

## 3. 本方案核心思想

本方案引入论文中的两个核心模块，并限定它们只服务于静态背景 scene Gaussians：

```text
Global-aware Gradient Rescaling
  -> 用图像高频缺失程度重标定 densification gradient
  -> 判断哪里是真的缺几何/缺细节

Voxel-based Densification
  -> 从 per-Gaussian 触发改为 per-voxel 触发
  -> 每个空间区域公平竞争 densification slot
  -> 避免局部 Gaussian 越长越乱
```

完整训练路径：

```text
HUGS training iteration
  -> render human + scene image
  -> compute original photometric loss for normal optimization
  -> for scene densification only:
       compute relative blur map from GT/rendered image
       compute global-aware pixel weights
       compute scene-only global-aware positional gradients
       accumulate max gradient per scene Gaussian
  -> at densification interval:
       group scene Gaussians into voxels
       compute voxel-level demand
       select voxels above threshold
       one densification operation per selected voxel
       clone/split selected scene Gaussian
       half-opacity initialization for child Gaussians
  -> continue normal HUGS optimization
```

---

## 4. 和现有两个背景版本的关系

### 4.1 与 HUGS 原版背景的关系

HUGS 原版背景作为 baseline，不改。

```text
scene_mode = "hugs_original"
```

该模式继续使用项目已有的 scene Gaussian densification/pruning 逻辑。

### 4.2 与 SuGaR 背景版本的关系

SuGaR 背景版本作为另一个独立实验分支，不被本方案覆盖。

```text
scene_mode = "sugar_scene"
```

SuGaR 分支重点是 regularization 和 pruning：

```text
opacity entropy
low-opacity pruning
scale / flatten regularization
surface alignment
```

### 4.3 新增 Global-aware Voxel ADC 背景版本

新增第三个分支：

```text
scene_mode = "global_aware_voxel_adc_scene"
```

该分支重点是 densification control：

```text
relative blur map
global-aware gradient rescaling
max view-space gradient accumulation
voxel-level densification decision
importance sampling inside voxel
half opacity child initialization
```

第一版建议不要和 SuGaR regularization 混用，先做三组平行对比：

```text
A. HUGS original scene
B. HUGS + SuGaR-style scene regularization
C. HUGS + Global-aware Voxel ADC scene densification
```

等第三组稳定后，再考虑第四组组合实验：

```text
D. HUGS + Global-aware Voxel ADC + light SuGaR pruning
```

但组合实验不是本文档第一版目标。

---

## 5. 需要定位的代码对象

coding agent 需要先在 HUGS 代码中定位 scene/background Gaussian 分支。

理想情况下，项目中有显式对象：

```text
human_gaussians
scene_gaussians
```

或等价变量：

```text
scene_xyz          # [Ns, 3]
scene_scaling      # [Ns, 3]
scene_rotation     # [Ns, 4] or [Ns, 3, 3]
scene_opacity      # [Ns, 1], logits or activated opacity
scene_features     # SH/color
```

如果 human 和 scene 共用一个 GaussianModel，则必须有 mask 或 type flag：

```text
is_scene_gaussian: BoolTensor[N]
is_human_gaussian: BoolTensor[N]
```

所有新增逻辑必须使用 `is_scene_gaussian` 过滤。

严禁发生：

```text
global-aware ADC 作用于 human Gaussians
voxel densification 修改 human Gaussians
half-opacity child initialization 修改 human branch
scene voxel grouping 混入 human Gaussians
```

---

## 6. 新增配置项

建议新增配置命名空间：

```yaml
scene_global_aware_adc:
  enabled: true
  apply_to_scene_only: true

  # mode control
  scene_mode: global_aware_voxel_adc_scene
  disable_for_human: true
  disable_sugar_losses: true

  # schedule
  start_iter: 500
  densify_until_iter: 15000
  densification_interval: 100
  reset_opacity_interval: 3000

  # gradient threshold
  tau_pos: 0.00025
  threshold_policy: fixed

  # relative blur map
  blur_window_size: 11
  highpass_cutoff_ratio: 0.5
  blur_percentile: 95
  weight_lower_bound: 0.5
  weight_upper_bound: 1.0
  use_grayscale: true

  # max view-space gradient
  use_max_viewspace_gradient: true
  gradient_accumulation: max

  # voxel densification
  voxel_size: 0.005
  one_densify_per_voxel: true
  voxel_score: mean
  selection_inside_voxel: importance_sampling

  # split / clone
  clone_scale_threshold: same_as_hugs
  split_scale_threshold: same_as_hugs
  child_opacity_policy: half_parent_opacity

  # safety
  min_scene_gaussians: 1000
  max_scene_gaussians: -1
  skip_empty_voxels: true
  preserve_optimizer_state: true

  # debug / logging
  log_blur_stats: true
  log_voxel_stats: true
  save_debug_maps: false
  debug_map_interval: 1000
```

`tau_pos` 需要根据当前 HUGS scene branch 的原始 densification threshold 调整。第一版可以沿用原 HUGS/3DGS 的 scene gradient threshold；如果训练过稀或过密，再做 sweep。

---

## 7. 训练 Schedule

建议第一版遵循论文设置，但适配 HUGS：

```text
Stage 0: HUGS warmup
iter 0 - start_iter
  正常优化 human + scene
  不启用 global-aware ADC

Stage 1: Global-aware scene gradient accumulation
iter start_iter - densify_until_iter
  正常 photometric loss 优化所有可训练参数
  额外为 scene densification 计算 global-aware gradient
  只累计 scene Gaussians 的 max view-space gradient

Stage 2: Voxel-based scene densification
every densification_interval
  只对 scene Gaussians 建 voxel
  计算 voxel demand
  选中超过 tau_pos 的 voxel
  每个 voxel 最多 densify 一个 scene Gaussian
  clone/split 后更新 scene optimizer state

Stage 3: Stop densification
after densify_until_iter
  关闭 scene densification
  继续正常优化 human + scene attributes
  可保留原有 scene pruning，但不能 prune human unless 原 HUGS 已经这么做
```

注意：如果项目已有 contact-aware attention 模块，建议先不要在 global-aware ADC 的早期同时开启 contact-aware 修正。推荐顺序：

```text
先训练稳定的 HUGS + global-aware scene branch
再开启 contact-aware attention / human correction
```

原因是 early stage 的 scene Gaussians 尚未稳定，过早让 human anchors 查询 scene candidates 容易学到错误接触关系。

---

## 8. 模块一：Pixel-wise Relative Blur Map

### 8.1 输入输出

输入：

```text
rendered_rgb: [H, W, 3]
gt_rgb:       [H, W, 3]
```

输出：

```text
B_relative:  [H, W]
```

### 8.2 计算逻辑

将 rendered image 和 GT image 转成灰度：

```text
I_render_gray
I_gt_gray
```

对每个像素附近的局部窗口计算高频能量：

```text
E_render(p) = local_high_frequency_energy(I_render_gray around p)
E_gt(p)     = local_high_frequency_energy(I_gt_gray around p)
```

然后计算相对模糊图：

```text
B_relative(p) = max(0, E_gt(p) - E_render(p))
```

含义：

- `B_relative` 高：GT 有高频结构，但 render 没有，可能是真实缺几何或缺细节。
- `B_relative` 低：render 已经有足够高频，或 GT 本身是弱纹理区域，不鼓励继续 densify。

### 8.3 工程实现建议

论文使用局部 FFT。第一版可以先实现严格版本：

```text
window_size = 11
highpass_cutoff_ratio = 0.5
```

实现位置建议：

```text
scene_global_aware_adc/relative_blur.py
```

建议提供函数：

```python
def compute_relative_blur_map(
    rendered_rgb,
    gt_rgb,
    window_size: int = 11,
    highpass_cutoff_ratio: float = 0.5,
    eps: float = 1e-6,
):
    """
    Return:
        relative_blur: Tensor[H, W]
    """
```

实现注意事项：

- 保持 tensor 在 GPU 上，避免每轮 CPU/GPU 往返。
- 第一版优先保证正确性；如果局部 FFT 太慢，再做优化。
- 如果当前训练 resolution 很高，可以先只在 densification interval 或低频率 iteration 计算。
- relative blur map 只用于 densification gradient，不改变主 photometric loss。

### 8.4 可选近似实现

如果局部 FFT 实现成本太高，可以先用高通滤波近似：

```text
high_freq(I) = abs(I - gaussian_blur(I))
B_relative = relu(high_freq(gt) - high_freq(render))
```

但如果使用近似实现，必须在配置和日志中标记：

```yaml
relative_blur_backend: gaussian_highpass_approx
```

论文复现版本应使用：

```yaml
relative_blur_backend: local_fft
```

---

## 9. 模块二：Pixel-wise Global-aware Loss

### 9.1 目标

relative blur map 仍然是局部图像信号，需要变成全局可比较的 densification 权重。

计算全图 95% 分位数：

```text
q95 = percentile(B_relative, 95)
```

归一化：

```text
W_init = B_relative / (q95 + eps)
```

截断到 `[0.5, 1.0]`：

```text
W_final = clamp(W_init, min=0.5, max=1.0)
```

构造用于 densification gradient 的 weighted pixel loss：

```text
L_global_aware_pixel = W_final * L_pixel
```

### 9.2 关键约束

`L_global_aware_pixel` 只用于计算 scene ADC 所需的位置梯度。

正常参数优化仍然使用 HUGS 原始 loss：

```text
L_total_train = L_hugs_original + optional_contact_losses
```

不要用 `L_global_aware_pixel` 替代主训练 loss，否则会改变颜色、opacity、scale、human pose 等优化目标。

### 9.3 代码落地建议

实现位置建议：

```text
scene_global_aware_adc/global_aware_loss.py
```

函数：

```python
def compute_global_aware_weights(
    relative_blur,
    percentile: float = 95.0,
    lower: float = 0.5,
    upper: float = 1.0,
    eps: float = 1e-6,
):
    """
    Return:
        weights: Tensor[H, W]
    """
```

训练循环中需要保留 per-pixel loss：

```python
pixel_loss = compute_pixel_reconstruction_loss(rendered_rgb, gt_rgb, reduction="none")
relative_blur = compute_relative_blur_map(rendered_rgb.detach_or_not, gt_rgb)
weights = compute_global_aware_weights(relative_blur)
global_aware_loss_for_adc = (weights * pixel_loss).mean()
```

关于 detach：

- `gt_rgb` 不需要梯度。
- `relative_blur` 本身不需要参与反向传播，建议 detach。
- `rendered_rgb` 用于 `relative_blur` 时建议 detach，避免 FFT/highpass 路径引入额外梯度。
- `pixel_loss` 必须保留对 rendered image 和 scene Gaussian projected position 的梯度，用于获得 densification gradient。

---

## 10. 模块三：Scene-only Gaussian-wise Max Gradient

### 10.1 原始问题

原始 3DGS 通常在 densification interval 内累计/平均每个 Gaussian 的 view-space positional gradient。论文认为平均会抹掉某些视角下暴露出来的真实结构缺失。

本方案改为：

```text
g_max_i = max gradient magnitude of scene Gaussian i within current densification interval
```

### 10.2 只累计 scene Gaussians

训练时 rasterizer 通常会返回 view-space points 或 screen-space means，并通过它们的 `.grad` 得到 densification signal。

需要确保：

```python
scene_grad = full_grad[is_scene_gaussian]
human_grad = full_grad[is_human_gaussian]  # ignore for this module
```

只更新：

```python
scene_max_grad_accumulator
```

不要更新 human densification 统计。

### 10.3 代码落地建议

实现位置建议：

```text
scene_global_aware_adc/gradient_accumulator.py
```

类：

```python
class SceneMaxGradientAccumulator:
    def __init__(self, num_scene_gaussians, device):
        self.max_grad = torch.zeros(num_scene_gaussians, device=device)

    @torch.no_grad()
    def update(self, scene_viewspace_grad):
        grad_norm = torch.linalg.norm(scene_viewspace_grad[..., :2], dim=-1)
        self.max_grad = torch.maximum(self.max_grad, grad_norm)

    @torch.no_grad()
    def reset(self, num_scene_gaussians=None):
        ...
```

如果 densification 后 scene Gaussian 数量变化，必须同步扩展/裁剪 accumulator。

---

## 11. 模块四：Voxel-based Densification

### 11.1 目标

把 densification decision 从 per-Gaussian 改成 per-voxel。

对于 scene Gaussians：

```text
scene_xyz: [Ns, 3]
scene_gmax: [Ns]
```

按 voxel size `epsilon` 建立空间分组：

```text
voxel_id_i = floor(scene_xyz_i / epsilon)
```

对每个 voxel 计算平均需求：

```text
g_vox = mean(gmax of Gaussians in voxel)
```

如果：

```text
g_vox >= tau_pos
```

则该 voxel 获得一个 densification slot。

### 11.2 每个 voxel 只 densify 一个 Gaussian

在被选中的 voxel 内，按 `gmax` 做 importance sampling：

```text
P(G_i) = gmax_i / sum(gmax_j in same voxel)
```

选出的 Gaussian 执行 HUGS/3DGS 原有 clone 或 split 操作。

clone/split 判断仍沿用原逻辑：

```text
if scale < threshold:
    clone
else:
    split
```

### 11.3 Half-opacity child initialization

原始 3DGS 里 child Gaussian 可能继承 parent opacity。论文建议两个 child 各继承 parent opacity 的一半，以减少 densification 对局部 radiance field 的突变。

如果 opacity 存的是 activated opacity：

```python
child_opacity = parent_opacity * 0.5
```

如果 opacity 存的是 logit，需要先确认项目中的 opacity activation：

```python
parent_alpha = opacity_activation(parent_opacity_logit)
child_alpha = parent_alpha * 0.5
child_opacity_logit = inverse_opacity_activation(child_alpha)
```

不要直接把 logit 乘 0.5，当项目使用 sigmoid activation 时这不是等价操作。

### 11.4 代码落地建议

实现位置建议：

```text
scene_global_aware_adc/voxel_densifier.py
```

核心函数：

```python
def voxel_based_scene_densify(
    scene_model,
    scene_xyz,
    scene_gmax,
    voxel_size: float,
    tau_pos: float,
    scale_threshold,
    optimizer,
    child_opacity_policy: str = "half_parent_opacity",
):
    """
    Select voxels by mean scene_gmax.
    Select one scene Gaussian in each selected voxel by importance sampling.
    Apply clone/split using existing scene_model helpers.
    Preserve optimizer state.
    Return densification stats.
    """
```

如果原 HUGS 代码已有：

```python
densify_and_clone(...)
densify_and_split(...)
prune_points(...)
cat_tensors_to_optimizer(...)
```

优先复用原有 helper，避免重写 optimizer state 管理。

---

## 12. 训练循环集成位置

伪代码如下：

```python
for iteration in range(num_iters):
    camera = sample_camera()
    render_pkg = render_hugs(camera, human_gaussians, scene_gaussians)

    rendered_rgb = render_pkg["render"]
    gt_rgb = camera.original_image

    # 1. normal HUGS optimization
    loss_main = compute_hugs_loss(render_pkg, camera)
    loss_main.backward(retain_graph=use_global_aware_adc_this_iter)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # 2. scene-only global-aware ADC gradient
    if cfg.scene_global_aware_adc.enabled and should_accumulate_adc(iteration):
        pixel_loss = compute_pixel_loss(rendered_rgb, gt_rgb, reduction="none")
        relative_blur = compute_relative_blur_map(
            rendered_rgb.detach(),
            gt_rgb,
            window_size=cfg.blur_window_size,
            highpass_cutoff_ratio=cfg.highpass_cutoff_ratio,
        )
        weights = compute_global_aware_weights(
            relative_blur,
            percentile=cfg.blur_percentile,
            lower=cfg.weight_lower_bound,
            upper=cfg.weight_upper_bound,
        )
        adc_loss = (weights * pixel_loss).mean()

        # This backward is only for densification stats.
        adc_loss.backward()

        scene_viewspace_grad = extract_scene_viewspace_grad(render_pkg)
        scene_grad_accumulator.update(scene_viewspace_grad)

        clear_adc_only_gradients()

    # 3. voxel densification
    if should_densify(iteration):
        stats = voxel_based_scene_densify(
            scene_model=scene_gaussians,
            scene_xyz=scene_gaussians.get_xyz,
            scene_gmax=scene_grad_accumulator.max_grad,
            voxel_size=cfg.voxel_size,
            tau_pos=cfg.tau_pos,
            scale_threshold=cfg.clone_scale_threshold,
            optimizer=scene_optimizer,
            child_opacity_policy=cfg.child_opacity_policy,
        )
        scene_grad_accumulator.reset(num_scene_gaussians=scene_gaussians.num_points)
```

实际实现时要根据 HUGS 当前训练代码调整 backward 顺序。更推荐的方式是：

- 主 loss backward 用于正常 optimizer update。
- global-aware adc loss backward 只用于读取 viewspace position gradient。
- 读取完 ADC gradient 后清理相关梯度，避免污染主优化。

如果当前 rasterizer 只能在一次 backward 中拿到 viewspace point grad，则可以把主 loss 和 adc loss 的计算拆到两次 render。第一版优先保证逻辑正确，之后再优化速度。

---

## 13. 与 Human Branch 的隔离规则

这是本方案最关键的工程安全规则。

### 13.1 不允许修改的对象

以下对象不能被本模块创建、删除、clone、split、prune 或 regularize：

```text
human canonical Gaussians
human Gaussian opacity / scale / rotation
SMPL / SMPL-X pose parameters
LBS weights
human anchor binding
contact-aware attention network
human correction module
```

### 13.2 渲染可以合并，更新必须分离

渲染时仍然可以 human + scene 一起 render：

```text
render(human_gaussians + scene_gaussians)
```

但是 densification update 必须只对 scene 执行：

```text
global-aware gradient: scene only
voxel grouping: scene only
clone/split: scene only
half opacity: scene children only
```

### 13.3 推荐添加 assert

coding agent 应在关键函数中添加安全检查：

```python
assert cfg.scene_global_aware_adc.apply_to_scene_only
assert not selected_indices_include_human
assert scene_xyz.shape[0] == scene_gmax.shape[0]
```

如果项目里没有明确的 scene/human mask，应先重构出明确接口，再实现本模块。

---

## 14. 日志与可视化

为了判断方法是否真的解决背景问题，建议记录以下日志。

### 14.1 数值日志

每个 densification interval 记录：

```text
scene_num_gaussians
human_num_gaussians
num_scene_voxels
num_selected_voxels
num_scene_clone
num_scene_split
mean_scene_gmax
max_scene_gmax
mean_voxel_score
max_voxel_score
relative_blur_mean
relative_blur_p95
global_weight_mean
global_weight_min
global_weight_max
```

必须确认：

```text
human_num_gaussians 不因本模块改变
```

### 14.2 图像可视化

可选保存：

```text
rendered_rgb
gt_rgb
relative_blur_map
global_aware_weight_map
selected_voxel_projection_debug
scene_floaters_before_after
```

保存频率建议：

```yaml
debug_map_interval: 1000
```

### 14.3 3D 可视化

建议导出或记录：

```text
scene Gaussian centers before/after
low-opacity scene Gaussians
selected densification voxels
contact-region scene candidates
```

重点看：

- 地面附近是否更干净。
- 人脚/手/臀/背接触区域附近是否减少漂浮点。
- 弱纹理背景是否减少无意义 Gaussian 堆积。
- 真实边缘和细结构是否没有被过度抑制。

---

## 15. 实验目标

本实验不是为了证明 contact-aware attention 本身更强，而是为了证明：

```text
更合理的静态背景 Gaussian densification
  -> 更干净的 scene geometry
  -> 更少 background floaters
  -> 更稳定的 human-scene query candidates
  -> 为后续接触建模提供更可靠输入
```

第一阶段目标：

- 对比 HUGS 原版背景，验证本方案是否减少 scene Gaussian 冗余和 floaters。
- 对比 SuGaR 背景版本，验证本方案是否在不引入 surface regularization 的情况下改善背景致密化。
- 保持 human branch 不变，排除人体分支改动带来的干扰。

第二阶段目标：

- 将本方案产生的 scene Gaussians 输入 human-scene attention。
- 检查接触区域 query candidates 是否更稳定。
- 检查人体漂浮/穿模是否间接受益。

---

## 16. 实验设置

建议至少跑以下三组：

```text
Exp A: HUGS original
  scene_mode = hugs_original

Exp B: HUGS + SuGaR scene
  scene_mode = sugar_scene

Exp C: HUGS + Global-aware Voxel ADC scene
  scene_mode = global_aware_voxel_adc_scene
```

如果资源允许，再跑：

```text
Exp D: HUGS + Global-aware Voxel ADC scene + contact-aware attention
Exp E: HUGS + SuGaR scene + contact-aware attention
```

注意 Exp C 第一版不要混入 SuGaR losses，否则无法判断收益来自哪一部分。

---

## 17. 评价指标

### 17.1 渲染质量

常规指标：

```text
PSNR
SSIM
LPIPS
```

这些指标用于保证背景 densification 改造没有破坏外观质量。

### 17.2 紧凑性与效率

重点记录：

```text
scene Gaussian count
total Gaussian count
model storage
training time
rendering FPS
peak GPU memory
```

论文结果表明，该类方法的优势通常不只是 PSNR，而是更好的 quality-storage trade-off。我们也应该重点观察 scene Gaussian 数量和渲染速度。

### 17.3 背景几何质量

需要设计项目内指标：

```text
background floater count
low-opacity residual count
contact-region scene candidate noise ratio
scene Gaussian distance to estimated support surface
scene Gaussian opacity distribution
scale anisotropy distribution
```

如果暂时没有自动指标，至少做固定视角和 3D viewer 的定性对比。

### 17.4 接触建模辅助指标

当接入 contact-aware attention 后，额外记录：

```text
human anchor 到 nearest scene Gaussian 的距离稳定性
接触区域 query 到的 scene candidates 数量
接触区域 query candidates 的 opacity 均值
接触区域 query candidates 的空间方差
脚底/手掌/臀部/背部穿模率
人体局部漂浮距离
```

本方案本身不直接修正 human pose，但它应该让 scene candidates 更可靠。

---

## 18. Ablation 设计

为了确认每个组件的贡献，建议在 Exp C 稳定后做以下消融：

```text
C0: HUGS original scene
C1: only Global-aware Gradient Rescaling, still per-Gaussian densification
C2: only Voxel-based Densification, use original gradient
C3: Global-aware Gradient Rescaling + Voxel-based Densification
C4: C3 + half-opacity child initialization
```

预期：

- C1 能减少噪声/弱纹理区域错误 densification。
- C2 能缓解局部密度反馈造成的过度增殖。
- C3 应该同时提升质量和紧凑性。
- C4 应该让 densification 后的训练更稳定。

---

## 19. 风险与处理

### 19.1 训练变慢

原因：

```text
relative blur map 的局部 FFT 有额外开销
```

处理：

- 只在 densification 相关 iteration 计算。
- 使用低分辨率 debug 版本确认逻辑。
- 后续改用 unfold + batched FFT 或高通近似。

### 19.2 背景补点不足

现象：

```text
scene Gaussian 过少
细节变糊
PSNR/LPIPS 下降
```

处理：

- 降低 `tau_pos`。
- 降低 `weight_lower_bound` 的实验要谨慎，太低会抑制非关键区域。
- 检查 `q95` 是否被异常 outlier 拉高。
- 检查 relative blur map 是否响应真实细节区域。

### 19.3 背景仍然漂浮

可能原因：

- 漂浮来自初始化/深度尺度错误，而不是 densification 错误。
- scene/human mask 泄漏，前景残影被 scene branch 吸收。
- photometric loss 本身允许半透明解释。

处理：

- 结合已有 SuGaR-style low-opacity pruning 做后续组合实验。
- 加强 scene mask 或 dynamic foreground filtering。
- 对 contact region 做 visibility / opacity / multi-view consistency pruning。

### 19.4 误伤人体分支

这是不可接受错误。

处理：

- 所有新增函数都只接收 scene_model 或 scene mask 后的数据。
- 添加 assert 和日志。
- 每次 densification 后记录 human Gaussian count，必须不变。

---

## 20. 推荐实施步骤

### Step 1: 新增配置和 mode

新增：

```text
scene_mode = global_aware_voxel_adc_scene
```

确保训练入口可以在三种背景模式间切换。

### Step 2: 定位 scene Gaussian 接口

明确获取：

```text
scene xyz / scale / rotation / opacity / features
scene optimizer state
scene densification helper
scene prune helper
```

如果没有独立 scene_model，先实现 scene/human mask 安全过滤。

### Step 3: 实现 relative blur map

先实现可运行版本，并保存 debug map 验证：

```text
真实缺细节区域响应高
天空/墙面/弱纹理区域响应低
明显噪声区域不要过度响应
```

### Step 4: 实现 global-aware weights

检查：

```text
weights in [0.5, 1.0]
q95 非 0
weight map 和 relative blur map 趋势一致
```

### Step 5: 实现 scene max gradient accumulator

检查：

```text
只累计 scene Gaussians
densification interval 后 reset
scene Gaussian 数量变化后 accumulator 同步变化
```

### Step 6: 实现 voxel densifier

检查：

```text
只用 scene xyz 建 voxel
每个 selected voxel 最多产生一次 clone/split
selected Gaussian 来自该 voxel 内部
optimizer state 正确扩展
```

### Step 7: 接入训练循环

先跑短训练：

```text
1000-3000 iterations
```

确认不 crash、不 NaN、human count 不变。

### Step 8: 完整训练和对比实验

跑完整：

```text
HUGS original
SuGaR scene
Global-aware Voxel ADC scene
```

记录统一指标。

---

## 21. 验收标准

第一版实现完成的最低标准：

```text
1. 可以通过配置启用 global_aware_voxel_adc_scene
2. human branch 完全不被 clone/split/prune
3. scene branch 使用 global-aware gradient 而不是原始平均 gradient 做 densification
4. scene densification 从 per-Gaussian 变成 per-voxel
5. 每个 selected voxel 最多 densify 一个 scene Gaussian
6. child opacity 使用 half-parent activated opacity
7. 训练可以完整跑完
8. 输出日志包含 scene Gaussian count、selected voxel count、blur/weight stats
9. 至少完成 HUGS original vs Global-aware Voxel ADC 的定量和定性对比
```

理想结果：

```text
PSNR/SSIM/LPIPS 不下降，最好提升
scene Gaussian count 下降或更合理
model storage 下降
rendering FPS 提升
背景 floaters 减少
接触区域 scene candidates 更干净
```

如果 Tanks/复杂场景中 scene Gaussian count 没有下降但质量提升，也可以接受。该方法不是强行压缩 Gaussian，而是让 Gaussian 分布更符合真实几何需求。

---

## 22. 与论文设置的对应关系

本文档中的模块和论文对应如下：

```text
Pixel-wise Relative Blur Map
  -> 论文 Section 4.1 的频域高频差异图

Pixel-wise Global-aware Loss
  -> 论文 Section 4.1 的 global-aware loss

Gaussian-wise Max Gradient
  -> 论文 Section 4.1 的 maximum view-space gradient

Voxel-based Densification
  -> 论文 Section 4.2 的 voxel-level decision and execution

Half-opacity child initialization
  -> 论文 Section 4.2 的 opacity calibration
```

默认超参对应：

```text
blur_window_size = 11
highpass_cutoff_ratio = 0.5
blur_percentile = 95
weight_lower_bound = 0.5
voxel_size = 0.005
densify_until_iter = 15000
```

`tau_pos` 需要结合当前 HUGS scene branch 重新选择。

---

## 23. 本方案对项目的预期价值

对当前项目来说，这个第三背景分支的价值不只是提升渲染指标，而是改善 contact-aware pipeline 的输入质量。

我们希望最终形成如下链路：

```text
Global-aware Voxel ADC
  -> scene Gaussians 更少错误增殖
  -> 背景几何更干净
  -> 接触区域附近 floaters 更少
  -> human anchors 查询到的 scene candidates 更可信
  -> human-scene attention 更容易学习真实接触关系
  -> 人体漂浮/穿模问题间接受益
```

因此，该方案在论文层面的定位可以写成：

```text
我们不是直接提出新的 contact loss，
而是先提升静态场景 Gaussian 的几何可靠性，
为后续人体-场景交互建模提供更可信的场景表示。
```

这和 SuGaR 背景方案形成互补：

```text
SuGaR scene:
  偏 surface regularization / pruning / flatten

Global-aware Voxel ADC scene:
  偏 densification signal / density control / compact distribution
```

第一版先平行比较，后续可以探索二者组合。


---

## 24. 已实现版本与实验记录（2026-05-25）

本方案已经以平行于原版 HUGS scene densification 和 SuGaR scene regularization 的方式接入当前 HUGS 代码，不改变默认训练路径。只有配置项 `scene_global_aware_adc.enabled=true` 时，scene branch 才会使用 Global-aware Voxelized ADC；默认 `false`，因此原版 HUGS、SuGaR-HUGS、anchor-attention correction 的既有路径保持兼容。

### 24.1 代码接入位置

```text
核心工具:
  hugs/utils/scene_global_aware_adc.py

配置入口:
  hugs/cfg/config.py
  cfg.scene_global_aware_adc

Trainer 接入:
  hugs/trainer/gs_trainer.py
  setup_scene_global_aware_adc()
  maybe_accumulate_scene_global_aware_adc()
  scene_densification()

实验配置:
  cfg_files/release/neuman/hugs_global_aware_voxel_adc_15000_lab.yaml
  cfg_files/release/neuman/hugs_global_adc_hugs12000_xyz_attention_plus3000_lab.yaml
```

实现要点：

```text
1. relative blur map:
   使用 gaussian high-pass approximation 构造 per-pixel blur/edge 权重。

2. global-aware gradient weighting:
   对 scene view-space gradient 施加图像全局感知权重，降低高频/不可靠区域的错误增殖倾向。

3. max view-space gradient accumulation:
   支持对同一 Gaussian 在多视角/多步中的最大梯度信号进行累计。

4. voxelized densification:
   以 voxel 为单位选择 densify 候选，避免局部区域无限增殖。

5. half-opacity child initialization:
   split/clone 产生的 child Gaussian 使用半 opacity 初始化，控制 density expansion。
```

注意：attention correction 阶段会显式关闭 `scene_global_aware_adc.enabled=false`，只把 ADC-HUGS checkpoint 作为起点，后续只做人体 Gaussian 的 xyz-only correction。

### 24.2 lab 背景重建实验结果

实验条件：

```text
场景: lab
训练目标: 只验证 ADC 背景重建方案 + HUGS 联合重建效果
attention correction: disabled
GPU: gpu-l40-1
起始点云: Neuman lab scene point cloud, 9013 points
```

| 场景 | 实验设置 | 步数 | Scene GS | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUMAN_PSNR | HUMAN_SSIM | HUMAN_LPIPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lab | Global-aware Voxelized ADC + HUGS | 12000 | 1,138,038 | 25.8364 | 0.9082 | 0.0794 | 19.2437 | 0.7594 | 0.1472 |
| lab | Global-aware Voxelized ADC + HUGS | 15000 | 1,236,705 | 25.7032 | 0.9071 | 0.0809 | 19.0630 | 0.7549 | 0.1530 |

对照参考（同场景已有实验）：

| 场景 | 实验设置 | 步数 | Scene GS | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUMAN_PSNR | HUMAN_SSIM | HUMAN_LPIPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lab | 原版 HUGS | 15000 | 2,136,961 | 25.9582 | 0.9147 | 0.0710 | 18.8005 | 0.7580 | 0.1532 |
| lab | SuGaR-HUGS | 15000 | 2,156,810 | 26.1260 | 0.9143 | 0.0707 | 18.9424 | 0.7606 | 0.1494 |

输出目录：

```text
gpu-l40-1:/hdd/u202420081000003/ml-hugs/output/human_scene/neuman/lab/hugs_trimlp/global_aware_voxel_adc_12000_lab_20260525/2026-05-25_08-31-58
gpu-l40-1:/hdd/u202420081000003/ml-hugs/output/human_scene/neuman/lab/hugs_trimlp/global_aware_voxel_adc_15000_lab_20260525/2026-05-25_09-18-26
```

训练日志：

```text
gpu-l40-1:/hdd/u202420081000003/ml-hugs/run_logs/global_aware_voxel_adc_12000_15000_lab_20260525.log
```

### 24.3 背景-only 阶段结论

1. ADC 方案能稳定完成 lab 场景 12000/15000 步训练，未破坏原版 HUGS/SuGaR/attention 入口。
2. ADC12000 明显比 ADC15000 更适合作为当前后续 attention 起点：full 和 human 指标均更好，scene GS 也更少。
3. ADC 的 scene GS 数量约为 1.14M，显著低于原版 HUGS15000/SuGaR15000 的约 2.14M/2.16M，背景表示更紧凑。
4. 单独背景重建阶段，ADC12000 的 full-image 指标低于 HUGS15000/SuGaR15000，但 human 区域指标更接近后续 attention correction 的需求。
5. 后续实际接入 xyz-only attention 后，ADC12000 起点被验证有效：6000 步 attention final 达到 HUGS_PSNR 26.3999、HUGS_SSIM 0.9199、HUGS_LPIPS 0.0656，超过单独 ADC12000，也超过当前 lab 上已有 HUGS/SuGaR 15000 full 指标。

当前建议：后续 ADC 分支默认使用 `ADC-HUGS12000` 作为 attention correction 起点，不优先使用 ADC15000。
