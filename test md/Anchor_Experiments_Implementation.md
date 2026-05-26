# Anchor 验证实验实现说明

本文档用于指导服务器端 Codex 在 HUGS / StM 代码基础上实现一组 anchor 相关的验证实验。当前阶段的目标不是实现完整方法，也不是固定最终技术路线，而是先建立一套可靠的 debug / visualization / analysis 工具，帮助我们判断“稳定的人体语义锚点”是否适合作为后续人体高斯与场景高斯信息交流的基础。

这些实验应尽量不侵入原训练流程，不修改主训练 loss，也不依赖某个尚未确定的 contact / attention 设计。实现后应能先在 HUGS 上运行，并且后续可以迁移到 StM 或其他 HUGS-like 代码中。

---

## 1. 背景与目标

当前项目关注人体-场景 HSI 感知的 4DGS 单目重建。我们希望在 HUGS 类框架中，让人体高斯和场景高斯能够在接触或邻近区域进行更合理的信息交流。但具体的交互模块形式仍可能继续调整，例如使用 StM-style shared mapping、anchor-conditioned attention、Gaussian volume-aware features，或它们的简化组合。

在进入完整方法实现之前，我们先验证一个更底层的问题：能否在人体上建立稳定、可解释、可视化友好的人体语义 anchor，并用这些 anchor 组织 human Gaussians 和查询 nearby scene Gaussians。

本阶段只要求实现 anchor 相关的可视化和统计验证。Coding agent 不需要实现最终 contact attention，也不需要实现完整接触优化。

本阶段的验证目标是回答以下问题：

1. 我们定义的 anchors 是否落在正确的人体语义区域？
2. HUGS 初始化出的 human Gaussians 能否稳定绑定到这些 anchors？
3. HUGS 的 clone / split / prune 之后，anchor binding 是否仍然稳定、连续、可解释？
4. 每帧 anchor world position 是否能查询到合理的 scene Gaussian 邻域，例如脚底附近能否查询到地面区域？
5. Anchor 局部 Gaussian 统计是否具有基本区分度，足以支持后续交互模块设计？

---
## 2. 实验总览

需要实现 5 个实验或 debug 功能。

```text
Exp A: Anchor template visualization
Exp B: Initial human Gaussian -> anchor binding visualization
Exp C: Densification inheritance visualization
Exp D: Anchor-local Gaussian statistics export
Exp E: Anchor -> scene Gaussian neighbor query visualization
```

这些实验都不需要实现 contact attention，不需要新增 contact loss，也不需要改变主渲染逻辑。

---

## 3. 推荐新增文件 / 模块

服务器端 Codex 可根据实际 HUGS / StM 仓库结构调整文件路径。建议新增以下模块：

```text
utils/anchor_utils.py
utils/anchor_io.py
utils/anchor_visualization.py
scripts/debug_anchors.py
```

如果仓库已有类似 `utils` / `tools` / `scripts` 目录，请遵循原有组织方式。

### 3.1 `anchor_utils.py`

负责：

- 加载 / 创建 anchors；
- 计算 anchor canonical position / normal / LBS weights；
- human Gaussian 到 anchor 的绑定；
- clone / split / prune 后同步 anchor binding；
- anchor 随 pose 变换到 world space；
- anchor-local Gaussian statistics。

### 3.2 `anchor_io.py`

负责：

- 保存 / 加载 `anchor_vertices.json`；
- 保存 / 加载 `anchor_bindings.pt`；
- 导出 anchor statistics CSV；
- 管理 debug 输出目录。

### 3.3 `anchor_visualization.py`

负责：

- 导出 template anchors PLY；
- 导出按 anchor 上色的 human Gaussians PLY；
- 导出 anchor -> scene neighbors PLY；
- 提供颜色表。

### 3.4 `debug_anchors.py`

提供命令行入口，至少支持以下模式：

```bash
python scripts/debug_anchors.py --mode template
python scripts/debug_anchors.py --mode bind_init --checkpoint <path>
python scripts/debug_anchors.py --mode export_iter --checkpoint <path> --iter 6000
python scripts/debug_anchors.py --mode scene_neighbors --checkpoint <path> --frame 0 --anchor right_sole
python scripts/debug_anchors.py --mode stats --checkpoint <path> --iter 6000
```

具体参数名可以根据仓库风格调整。

---

## 4. Anchor 定义

### 4.1 Anchor 列表

第一版推荐定义 15-20 个 anchors。为了最小验证，可以先从以下列表开始：

```text
left_sole
right_sole
left_toe
right_toe
left_heel
right_heel
left_palm
right_palm
left_fingers
right_fingers
buttocks
back
left_knee
right_knee
left_elbow
right_elbow
```

如果当前 HUGS 使用 SMPL 而非 SMPL-X，手指区域可能没有细节，可将 `left_fingers/right_fingers` 临时合并到 `left_palm/right_palm` 或使用 hand/wrist region 替代。

### 4.2 `anchor_vertices.json`

每个 anchor 对应一组 SMPL / SMPL-X template vertex indices。

文件格式：

```json
{
  "left_sole": [0, 1, 2],
  "right_sole": [10, 11, 12],
  "left_toe": [20, 21],
  "right_toe": [30, 31]
}
```

实际顶点索引需要根据项目使用的 SMPL / SMPL-X template 决定。

如果项目已有 SMPL part segmentation，可基于 segmentation 半自动生成。如果没有，可以先手工标注少量关键区域。

### 4.3 半自动生成规则，可选

若没有现成 `anchor_vertices.json`，可以临时实现半自动规则：

- foot sole：在 left/right foot body part vertices 中选取竖直方向最低的一批点；
- toe：在 foot region 中沿 foot forward direction 最前的一批点；
- heel：在 foot region 中沿 foot forward direction 最后的一批点；
- palm：若使用 SMPL-X，使用 hand / palm vertex labels；若使用 SMPL，使用 hand/wrist region 表面点；
- back：torso/back body part 中 normal 朝后的顶点；
- buttocks：pelvis/hip region 后下方顶点；
- knee/elbow：对应 limb region 中靠近关节且位于表面的顶点。

半自动规则仅用于 debug 初版，最终最好保存固定 `anchor_vertices.json`，保证实验可复现。

---

## 5. Anchor 数据结构

加载 template vertices、vertex normals 和 LBS weights 后，计算每个 anchor 的 canonical 属性。

```python
anchors = {
    "names": List[str],             # len A
    "pos_canon": Tensor[A, 3],      # canonical anchor position
    "normal_canon": Tensor[A, 3],   # canonical anchor normal
    "lbs": Tensor[A, J],            # averaged LBS weights
    "part_id": Tensor[A],           # semantic body part id
    "vertex_ids": List[List[int]],  # original template vertex ids
}
```

计算方式：

```python
pos_canon[a] = template_vertices[ids].mean(dim=0)
normal_canon[a] = normalize(template_normals[ids].mean(dim=0))
lbs[a] = normalize(template_lbs_weights[ids].mean(dim=0))
```

注意：

- `normal_canon` 必须归一化；
- `lbs[a]` 的和应为 1；
- 如果某个 anchor 的 vertex list 为空，应报错；
- 输出时保存一份 `anchors.pt` 或在每次运行时重新计算均可。

---

## 6. Exp A: Anchor Template Visualization

### 6.1 目标

验证 anchor 的 canonical 位置和法向是否合理。

### 6.2 输入

- SMPL / SMPL-X template vertices；
- template faces，可选，用于导出 mesh；
- vertex normals；
- template LBS weights；
- `anchor_vertices.json`。

### 6.3 输出

```text
outputs/anchor_debug/template_anchors.ply
outputs/anchor_debug/template_anchors_with_normals.ply
outputs/anchor_debug/anchors.csv
```

### 6.4 可视化要求

`template_anchors.ply` 至少包含：

- template mesh，可选；
- anchor positions，以小 sphere 或 points 表示；
- 每个 anchor 使用固定颜色；
- anchor normal 可用短线段表示。

如果实现 sphere 较麻烦，第一版可只导出 anchor points + normal line segments。

### 6.5 验收标准

人工检查：

- `left_sole/right_sole` 是否位于脚底；
- `toe/heel` 是否区分清楚；
- `palm/fingers` 是否在手部区域；
- `back/buttocks` 是否位于正确表面；
- normal 方向是否大致朝外。

---

## 7. Human Gaussian 到 Anchor 的初始绑定

### 7.1 绑定时机

在 HUGS 初始化 human Gaussians 后立即绑定。

HUGS 初始 human Gaussians 通常来自 SMPL template vertices 或附近点，因此最适合在 canonical space 中绑定。

### 7.2 输入

每个 human Gaussian 需要：

```python
mu_canon: Tensor[Nh, 3]
lbs_w: Tensor[Nh, J]
opacity: Tensor[Nh, 1]      # optional
scale: Tensor[Nh, 3]        # optional
```

Anchor 需要：

```python
anchor_pos_canon: Tensor[A, 3]
anchor_lbs: Tensor[A, J]
anchor_part_id: Tensor[A]
```

若有 Gaussian 对应 SMPL vertex id 或 body part id，也可以作为额外先验。

### 7.3 绑定分数

建议使用以下 score：

```text
score(g, a) =
    - ||mu_g_canon - anchor_pos_a||^2 / sigma_a^2
    - lambda_lbs * ||lbs_g - anchor_lbs_a||^2
    + body_part_bonus(g, a)
```

其中：

- `sigma_a` 可设为每个 anchor region 的半径，或统一常数；
- `lambda_lbs` 初始可设为 0.1-1.0；
- `body_part_bonus` 如果没有 body part label，可先设为 0。

### 7.4 Top-m 绑定

支持两种模式：

#### Hard assignment

```python
gaussian_anchor_id: Tensor[Nh]
```

每个 Gaussian 只绑定一个 anchor。

#### Sparse soft assignment，推荐

```python
gaussian_anchor_ids: Tensor[Nh, m]
gaussian_anchor_weights: Tensor[Nh, m]
```

第一版推荐：

```text
m = 2
```

weights 使用 top-m score softmax：

```python
weights = softmax(score_topm, dim=-1)
```

### 7.5 保存

保存为：

```text
outputs/anchor_debug/anchor_bindings_init.pt
```

内容包括：

```python
{
    "gaussian_anchor_ids": Tensor[Nh, m],
    "gaussian_anchor_weights": Tensor[Nh, m],
    "anchor_names": List[str],
    "mu_canon": Tensor[Nh, 3],
}
```

---

## 8. Exp B: Initial Binding Visualization

### 8.1 目标

验证初始 human Gaussians 是否绑定到正确语义 anchor。

### 8.2 输出

```text
outputs/anchor_debug/human_gaussians_anchor_init.ply
outputs/anchor_debug/human_gaussians_anchor_init.csv
```

### 8.3 PLY 内容

每个 human Gaussian 用一个点表示即可，位置使用 canonical position。

颜色：

- 使用 top-1 anchor id 上色；
- anchor 颜色固定，便于跨 iteration 对比；
- 可选：opacity 较低的点降低亮度。

同时导出 anchors points，最好与 human Gaussian 同一 PLY 中显示。

### 8.4 验收标准

人工检查：

- 脚底 Gaussians 是否大多绑定到 sole/toe/heel；
- 手部 Gaussians 是否绑定到 palm/fingers；
- back / buttocks 是否区分合理；
- 关节附近是否出现明显错绑；
- 是否需要 top-2 soft assignment。

---

## 9. Densification 后 Binding 继承

### 9.1 目标

验证 clone / split / prune 之后 anchor binding 是否仍然稳定。

### 9.2 实现要求

当 HUGS 执行 clone 或 split 时，新 Gaussian 继承 parent Gaussian 的 anchor binding。

伪代码：

```python
new_anchor_ids = parent_anchor_ids.clone()
new_anchor_weights = parent_anchor_weights.clone()
```

当 HUGS prune Gaussians 时，同步 prune binding：

```python
gaussian_anchor_ids = gaussian_anchor_ids[keep_mask]
gaussian_anchor_weights = gaussian_anchor_weights[keep_mask]
```

### 9.3 需要插入的位置

在 HUGS 代码中找到 human Gaussian densification / clone / split / prune 的实现位置。通常会有类似函数：

```text
densify_and_clone
densify_and_split
prune_points
adaptive_control
```

请在这些函数中同步更新 anchor binding。

### 9.4 输出 checkpoint

每个关键 iteration 导出：

```text
outputs/anchor_debug/human_gaussians_anchor_iter0000.ply
outputs/anchor_debug/human_gaussians_anchor_iter3000.ply
outputs/anchor_debug/human_gaussians_anchor_iter6000.ply
outputs/anchor_debug/human_gaussians_anchor_iter12000.ply
```

具体 iteration 根据当前 HUGS 训练 schedule 调整。

---

## 10. Exp C: Densification Inheritance Visualization

### 10.1 目标

观察训练过程中按 anchor 上色的人体 Gaussians 是否保持语义连续。

### 10.2 输出统计

每个 iteration 同时导出 CSV：

```text
outputs/anchor_debug/anchor_stats_iterXXXX.csv
```

CSV columns：

```text
iter
anchor_name
num_gaussians
mean_opacity
max_opacity
mean_scale_x
mean_scale_y
mean_scale_z
mean_dist_to_anchor_canon
mean_assignment_weight
```

若支持 world frame，可额外记录：

```text
mean_dist_to_anchor_world
```

### 10.3 验收标准

人工检查：

- split/clone 后颜色分布是否仍然连续；
- 是否有 anchor 完全失去 Gaussians；
- 是否有明显漂浮 Gaussians 继承错误 anchor；
- sole/palm 等接触区域是否有足够 Gaussian 覆盖；
- densification 后 anchor count 是否异常爆炸。

---

## 11. Exp D: Anchor-local Gaussian Statistics

### 11.1 目标

验证 anchor 聚合前的局部 Gaussian 特征是否有区分度，为后续 anchor token / attention 提供依据。

### 11.2 对每个 anchor 计算统计

在 canonical 或 selected frame world space 中计算：

```text
num_gaussians
mean_opacity
max_opacity
mean_log_scale
mean_covariance_diag_local
mean_extent_along_anchor_normal
mean_distance_to_anchor
weighted_mean_distance_to_anchor
mean_lbs_entropy
```

其中：

```text
extent_along_anchor_normal = sqrt(n_anchor^T Sigma_g n_anchor)
```

### 11.3 输出

```text
outputs/anchor_debug/anchor_feature_stats_iterXXXX.csv
```

可选输出：

```text
outputs/anchor_debug/anchor_feature_embeddings_iterXXXX.npy
```

其中每个 anchor 一个 pooled vector，用于后续 PCA / t-SNE 可视化。

### 11.4 验收标准

- 不同 semantic anchors 的统计应该有明显差异；
- foot sole / palm / back 等接触相关区域应具有可解释的 opacity / scale / covariance 特征；
- 如果所有 anchor 统计几乎一样，说明 binding 或 feature 设计可能有问题。

---

## 12. Exp E: Anchor -> Scene Gaussian Neighbor Query

### 12.1 目标

验证每帧 anchor world position 能否查询到合理的 scene Gaussians。

这是后续 human-scene attention 的基础。

### 12.2 输入

- 某一训练 checkpoint；
- frame id；
- anchor world positions；
- scene Gaussian positions / opacity / scale / rotation。

### 12.3 查询方式

第一版使用 KNN 即可：

```python
scene_idx = knn(anchor_world, scene_mu_world, K_s)
```

推荐：

```text
K_s = 32
```

可选过滤：

```text
opacity > opacity_threshold
within radius r_max
```

如果 KNN 被 floaters 干扰，可加入 opacity filtering 或 radius filtering。

### 12.4 输出

对指定 frame 和 anchor，导出：

```text
outputs/anchor_debug/scene_neighbors_frame0000_right_sole.ply
```

PLY 中包含：

- scene Gaussians，可选全部灰色；
- queried neighbor scene Gaussians，高亮；
- anchor point，大球或特殊颜色；
- human Gaussians，可选半透明或按 anchor 色。

### 12.5 批量导出

建议支持一次导出多个关键 anchors：

```text
right_sole
left_sole
right_palm
left_palm
buttocks
back
```

### 12.6 验收标准

人工检查：

- foot sole anchor 是否查询到地面 Gaussians；
- hand/palm anchor 是否查询到附近墙面、物体或背景；
- back/buttocks 是否查询到支撑面；
- 查询结果是否被 scene floaters 干扰；
- 是否需要 opacity threshold / radius threshold / direction filtering。

---

## 13. Anchor Token 聚合网络，后续实现参考

本阶段可先只做统计和可视化，不必训练网络。但为了后续代码迁移，建议按以下接口预留。

### 13.1 Anchor-local feature

对绑定到 anchor 的 human Gaussian g，构造：

```text
rel_mu                         3
||rel_mu||                     1
normal_distance                1
log_scale                      3
covariance_local_upper_tri     6
opacity                        1
low_dim_color_or_SH            C
projected_lbs_feature          16
anchor_assignment_weight       1
optional_visibility_residual   r
```

其中：

```text
rel_mu = R_anchor^T (mu_g_world - anchor_world)
Sigma_local = R_anchor^T Sigma_g R_anchor
```

### 13.2 GaussianEncoder

```text
GaussianEncoder:
  Linear(input_dim, 128)
  GELU
  LayerNorm(128)
  Linear(128, 128)
  GELU
```

输出：

```python
e_g: Tensor[K, 128]
```

### 13.3 AnchorQueryMLP

```text
AnchorQueryMLP:
  input = anchor_embed + part_embed + optional pose feature
  Linear(input_dim, 128)
  GELU
  Linear(128, 128)
```

输出：

```python
q_a: Tensor[128]
```

### 13.4 BiasMLP

```text
BiasMLP:
  input = [distance, normal_distance, opacity, assignment_weight]
  Linear(4, 32)
  GELU
  Linear(32, 1)
```

### 13.5 Attention pooling

```text
k_g = W_k(e_g)
v_g = W_v(e_g)
alpha_ag = softmax((q_a · k_g) / sqrt(128) + bias_ag)
human_anchor_token_a = sum_g alpha_ag * v_g
```

第一版可先实现 mean pooling 和 attention pooling 两种模式，便于 ablation：

```text
--pooling mean
--pooling attention
```

### 13.6 Multi-slot pooling，可选

对脚和手等复杂区域，可以支持：

```text
S = 2
```

即一个 anchor 输出两个 learned query / token：

```python
q_a: Tensor[S, 128]
token_a: Tensor[S, 128]
```

第一版可以只实现接口，默认 `S=1`。

---

## 14. 输出目录规范

推荐统一输出到：

```text
<experiment_output>/anchor_debug/
```

建议文件：

```text
anchor_debug/
  anchor_vertices.json
  anchors.pt
  template_anchors.ply
  anchors.csv
  anchor_bindings_init.pt
  human_gaussians_anchor_init.ply
  human_gaussians_anchor_iter0000.ply
  human_gaussians_anchor_iter3000.ply
  human_gaussians_anchor_iter6000.ply
  anchor_stats_iter0000.csv
  anchor_stats_iter3000.csv
  anchor_stats_iter6000.csv
  anchor_feature_stats_iter0000.csv
  scene_neighbors_frame0000_right_sole.ply
  scene_neighbors_frame0000_left_sole.ply
```

---

## 15. PLY 导出要求

PLY 中至少需要包含：

```text
x, y, z
red, green, blue
```

对于 Gaussian 可选额外字段：

```text
opacity
scale_x, scale_y, scale_z
anchor_id
```

如果现有可视化工具支持 Gaussian ellipsoid，可以导出 scale / rotation；否则第一版用 colored points 即可。

颜色要求：

- anchor id 对应固定颜色；
- 不同 iteration 颜色表必须一致；
- scene neighbors 使用高亮颜色；
- 普通 scene Gaussians 使用浅灰色。

---

## 16. 与 HUGS / StM 的迁移关系

本阶段功能应尽量只依赖以下数据：

```text
human Gaussian canonical position
human Gaussian world position
human Gaussian opacity / scale / rotation
human Gaussian LBS weights
SMPL / SMPL-X template vertices
SMPL / SMPL-X template LBS weights
scene Gaussian position / opacity / scale / rotation
frame pose / camera pose
```

这些数据在 HUGS 和 StM 中都应存在。因此实现时尽量不要绑定到 HUGS 特有的训练 loss 或 renderer 内部逻辑。

建议将 anchor 相关代码写成独立工具函数，这样后续迁移到 StM 时只需要适配数据读取接口。

---

## 17. 最小完成标准

服务器端 Codex 完成后，至少应提供：

1. `template_anchors.ply`：能看到 anchor 在人体 template 上的位置；
2. `human_gaussians_anchor_init.ply`：初始 human Gaussians 按 anchor 着色；
3. densification 后若干 iteration 的按 anchor 着色 PLY；
4. `anchor_stats_iterXXXX.csv`：每个 anchor 的 Gaussian 数量、opacity、scale 等统计；
5. 至少一个 frame 的 `scene_neighbors_frameXXXX_<anchor>.ply`；
6. 一段简短 README 或命令说明，说明如何运行这些 debug 导出。

如果以上结果显示 anchor 定义和绑定稳定，再继续实现 anchor-conditioned StM / attention。

---

## 18. 暂不需要实现的内容

当前阶段不需要实现：

- contact attention 训练；
- contact loss；
- Gaussian boundary gap loss；
- shared attribute mapping；
- 修改主训练目标；
- 大规模网络训练。

本阶段只做 anchor 机制的可视化和统计验证。

