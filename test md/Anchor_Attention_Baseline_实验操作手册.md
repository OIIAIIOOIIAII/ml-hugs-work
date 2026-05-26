# Anchor-Attention Baseline 实验操作手册

本文档面向 coding agent，用于在 HUGS / HUGS-like 代码基础上，逐步实现并验证一个最小版本的 HSI 感知单目动态 4DGS baseline。

当前目标不是一次性实现最终完整方法，而是按照可控实验逐步验证：

1. HUGS baseline 是否稳定；
2. Human3R 初始化是否改善人-场相对位置；
3. semantic interaction anchors 是否能稳定组织人体 Gaussians；
4. anchor 是否能查询合理的局部 scene Gaussians；
5. anchor-scene cross-attention 是否能通过渲染优化产生有效 correction；
6. Gaussian volume-aware features / geometry bias / weak ellipsoid regularization 是否进一步改善接触区域。

---

## 0. 整体技术路线

我们的当前 baseline 技术路线如下：

```text
Monocular Video
  -> HUGS baseline reconstruction
  -> Human3R-based initialization
  -> HUGS warm-up training
  -> semantic interaction anchor construction
  -> bind human Gaussians to anchors
  -> anchor token encoder
  -> local scene Gaussian query
  -> anchor-scene cross-attention
  -> interaction correction head
  -> corrected human Gaussians
  -> Gaussian rendering and losses
  -> optional Gaussian volume-aware features / geometry bias / weak ellipsoid gap regularization
```

当前不考虑 StM。也就是说，本阶段只验证：

```text
anchor + local scene query + cross-attention + Gaussian correction
```

核心思想：

- HUGS learned LBS 继续负责人高斯运动；
- anchor 不驱动运动，只作为人体语义区域的信息路由器；
- 每个 anchor 聚合绑定的人体高斯特征，形成 human anchor token；
- anchor 在当前帧 world space 查询附近 scene Gaussians；
- human anchor token 与 scene Gaussian tokens 做 cross-attention；
- attention 输出 interaction context；
- interaction context 通过 correction head 小幅修正人体 Gaussian 参数；
- 修正后的 Gaussians 进入原 HUGS renderer；
- rendering loss 是主监督；
- Gaussian ellipsoid gap 后续作为几何特征、attention bias 和弱安全正则。

---

## 1. 代码组织建议

coding agent 可根据实际仓库结构调整文件位置。推荐新增或修改以下模块。

```text
configs/
  anchor_attention_baseline.yaml

utils/
  anchor_utils.py
  anchor_io.py
  anchor_visualization.py
  scene_query.py
  gaussian_geometry.py

models/
  anchor_token_encoder.py
  scene_token_encoder.py
  anchor_scene_attention.py
  interaction_correction.py

scripts/
  debug_anchors.py
  train_anchor_attention.py 或在原 train.py 中加开关
  eval_interaction_regions.py
```

如果仓库已有对应目录，请遵循原有风格，不强行创建新结构。

所有新增功能必须提供 config 开关，保证可以退回原始 HUGS baseline：

```yaml
use_human3r_init: false
use_anchors: false
use_anchor_token_encoder: false
use_scene_query: false
use_cross_attention: false
use_interaction_correction: false
use_volume_features: false
use_geometry_bias: false
use_gap_regularization: false
```

---

## 2. 实验总表

推荐按以下顺序实现和实验。

```text
Exp 0: HUGS baseline 跑通
Exp 1: HUGS + Human3R initialization
Exp 2: Anchor debug experiments
Exp 3: Anchor token encoder dry-run
Exp 4: Scene Gaussian local query
Exp 5: Anchor-scene cross-attention dry-run
Exp 6: Attention context -> human Gaussian correction
Exp 7: Add Gaussian volume-aware features
Exp 8: Add geometry bias in attention
Exp 9: Add weak ellipsoid gap regularization
Exp 10: Full ablation and evaluation
```

每一步都要能单独开关、单独保存输出，方便定位问题。

---


### 2.1 当前实现状态与补全路线（2026-05-19）

本节用于把 Exp 0-10 的工程状态说清楚：哪些已经在代码中落地，哪些只是 debug/dry-run，哪些还需要继续做，避免把“实验设计”误认为“完整方法已经完成”。当前 baseline 以原版 HUGS 初始化和原版 HUGS 训练流程为主线，Human3R 初始化和更复杂 token fusion 暂时作为后续替换点保留。

#### Exp 0: HUGS baseline 跑通

当前已实现：

- 原版 HUGS/NeuMan `human_scene` 路线已经可以训练、验证、保存 checkpoint、保存 full-sequence render。
- 本地已有无 AMASS 辅助脚本 `scripts/run_neuman_human_scene_noamass.py`，可跳过 animation 数据集。
- 已有 6000-step 原版 HUGS/NeuMan `lab` 对照记录，原版路线质量明显高于当前 Human3R 初始化路线。

已经验证：

- 原版 `lab`、低分辨率 `lab_h3rres` 都能跑通 6000 步并导出 `render_all`。
- 当前新接入的 anchor-attention dry-run 不破坏 HUGS 训练主流程，1-step GPU smoke 已通过。

还需要做：

- 为正式论文/项目结果重跑标准步数对照，例如 6000、15000 steps。
- 固定一套统一评估脚本，确保原版 HUGS、dry-run、correction 实验用同样 train/val/all split 和同样指标。
- 增加 interaction crop 评估区域，例如 feet-ground、hand-scene、back/buttocks。

完成判据：

- 原版 HUGS baseline 有稳定可复现实验目录、配置、指标和视频。
- 后续所有方法都以这个结果作为主对照。

#### Exp 1: HUGS + Human3R initialization

当前已实现：

- 已实现 Human3R 输出到 HUGS 数据目录的转换脚本：`scripts/convert_human3r_to_hugs.py`。
- 已实现严格拟合 Human3R SMPL-X 到 HUGS SMPL 的脚本：`scripts/fit_smpl_to_human3r_smplx.py`。
- 已实现 `Human3RDataset` 和 `cfg_files/release/human3r/hugs_human_scene.yaml`。
- 已完成多轮 Human3R 初始化诊断，确认当前 Human3R 路线质量低于原版 HUGS 初始化。

已经验证：

- Human3R-HUGS 转换数据可以被 HUGS 读取并训练。
- 严格拟合 SMPL 脚本可以 CPU smoke 和 GPU 完整拟合。
- 6000-step Human3R 初始化训练可跑通，但指标低于原版 HUGS。

还需要做：

- 当前 baseline 暂不依赖 Human3R 初始化，正式效果提升 pipeline 中需要重新解决 Human3R camera/depth/scene scale/SMPL 对齐问题。
- 需要明确 Human3R 初始化是只提供 camera/scale/human transform，还是替换 scene point cloud 和 SMPL 参数。
- 需要建立 Human3R init debug 输出：camera trajectory、human-scene alignment、初始投影 bbox、scene point cloud scale 分布。
- 若继续使用 Human3R depth 点云，需要做 scene prior 清理、尺度归一、点云稀疏化或 COLMAP/Human3R 混合初始化。

完成判据：

- Human3R 初始化至少不显著低于原版 HUGS 初始化。
- 人体/场景/相机在 world space 的粗接触关系更合理。
- 该初始化能通过配置切换接入当前 anchor-attention baseline，而不改 interaction 模块。

#### Exp 2: Anchor debug experiments

当前已实现：

- 已实现 anchor 生成、保存、读取、可视化：`hugs/utils/anchor_utils.py`、`anchor_io.py`、`anchor_visualization.py`。
- 已实现 `scripts/debug_anchors.py`，覆盖模板 anchor、初始绑定、densification 继承、feature stats、token dry-run、scene neighbor/query overlay 等调试功能。
- 已在 `hugs_trimlp.py` 和 `hugs_wo_trimlp.py` 中加入 anchor binding 保存、加载、clone/split/prune 继承逻辑。

已经验证：

- Exp A-E 已有输出和记录，anchor 使用 HUGS 自身 vitruvian template，修复过外部 SMPL 模板错位问题。
- 真实 HUGS normal 3000-step 输出上，anchor-scene nearest-neighbor 可视化已完成。
- 原 HUGS 初始化和 Human3R 初始化都做过 ExpE 对比导出。

还需要做：

- 将 `debug_anchors.py` 的部分能力进一步拆成更轻量的独立检查脚本，例如 token stats、attention stats、scene query stats。
- 为每次训练自动导出少量 anchor overlay 图片，而不仅仅保存 `.pt/.csv`。
- 建立固定的关键 anchor 列表和关键帧列表，用于所有实验横向对比。

完成判据：

- 每个正式实验都能导出 anchor binding summary、anchor scene query 诊断、关键帧 overlay。
- anchor binding 在 densification 后保持连续，不出现大面积随机错绑。

#### Exp 3: Anchor token encoder dry-run

当前已实现：

- 已实现 `hugs/models/anchor_attention.py::AnchorSceneAttentionBaseline` 中的 human Gaussian encoder 和 anchor token pooling。
- Human Gaussian 输入特征当前为 16 维：相对 anchor 位置、距离、normal distance、opacity、log scale、world normal、canonical relative position、assignment weight。
- 当前支持 `human_pooling=mean` 和 `human_pooling=attention`。
- attention pooling 使用 anchor embedding 作为 query，并带一个 4 维 bias MLP。

已经验证：

- 纯 CPU 随机张量前向 smoke 通过。
- GPU 1-step HUGS smoke 中已生成 `human_tokens` debug tensor。

还需要做：

- 增加独立 token 诊断脚本，读取 `anchor_attention_iter*.pt`，输出 token norm、cosine similarity、PCA/heatmap。
- 将当前 token 特征设计作为 v0 baseline，后续系统测试 volume-aware feature、SH/color feature、LBS projected feature、多 slot token。
- 判断 token 是否真的参与下游优化：dry-run 只能证明链路可用，不能证明 token 有效。

完成判据：

- `human_tokens` shape 稳定为 `[num_anchors, hidden_dim]`，默认 `[16,128]`。
- token 无 NaN，不同 anchors 有区分度。
- mean pooling 和 attention pooling 都能稳定训练，mean 可作为 fallback。

#### Exp 4: Scene Gaussian local query + Scene token encoder

当前已实现：

- 已在 `AnchorSceneAttentionBaseline._query_scene_tokens()` 中实现 opacity filter + `torch.cdist` + topK scene query。
- 当前 scene token 输入为 11 维：相对 anchor 位置、距离、opacity、log scale、低维 SH/color。
- scene encoder 是轻量 MLP，输出 `[A,K,hidden_dim]`。
- 支持配置：`scene_topk`、`scene_opacity_threshold`、`max_scene_candidates`。

已经验证：

- CPU smoke 和 GPU 1-step smoke 均通过。
- debug 输出中已保存 `scene_knn_idx`、`scene_knn_dist`，CSV 中记录 nearest distance 和 attention weighted distance。

还需要做：

- 增加 PLY/PNG 可视化，把每个 anchor 查询到的 scene Gaussians 高亮导出。
- 优化查询性能：当前 topK 对大规模 scene 依赖 `max_scene_candidates` 采样，后续可替换 KD-tree、voxel hash 或 CUDA KNN。
- 加入可见性/相机 frustum 过滤，避免查询到当前帧不可见或低质量 floaters。
- 对 foot/palm/back 等 anchor 建立固定 query 质量诊断。

完成判据：

- 每个 anchor 的 `scene_knn_dist` 无 NaN/Inf。
- foot sole anchor 能稳定查询到地面附近 scene Gaussians。
- 查询耗时可接受，长训练中不会成为主要瓶颈。

#### Exp 5: Anchor-scene cross-attention dry-run

当前已实现：

- 已实现单层、单 query 的 per-anchor cross-attention。
- `Q` 来自 human anchor token，`K/V` 来自 local scene tokens。
- 输出 `context: [A,D]` 和 `attention_weights: [A,K]`。
- 默认 baseline 配置开启 cross-attention，但关闭 correction，因此不应改变渲染结果。

已经验证：

- CPU 前向 smoke 通过，attention shape 正确。
- GPU 1-step dry-run 通过，训练/验证/render_full_sequence 均完成。
- 已生成 `anchor_attention_iter000000.{pt,csv}` 和 `anchor_attention_iter000001.{pt,csv}`。

还需要做：

- 跑 3000/6000-step dry-run，确认长期训练时 dry-run 不改变原版 HUGS 指标。
- 增加 attention visualization：将 scene neighbors 按 attention weight 上色。
- 检查 attention entropy、max attention 是否塌缩或完全均匀。
- 加入 geometry bias 前，先确认纯 attention 的行为可解释。

完成判据：

- Dry-run 与原版 HUGS 指标接近，证明模块旁路无副作用。
- attention weights 无 NaN，entropy 和 max weight 在合理范围。
- 对接触相关 anchors，attention 或 nearest-distance 诊断具有可解释性。

#### Exp 6: Attention context -> human Gaussian correction

当前已实现：

- 已实现可选 position correction head：`MLP(16+128 -> 128 -> 3)`。
- 已实现可选 opacity correction head：`MLP(1+128 -> 64 -> 1)`。
- correction 通过配置控制，默认关闭。
- 已实现 `delta_loss = mean(delta_mu^2) + mean(delta_opacity^2)`，并在训练器中按 `delta_loss_w` 加入总 loss。
- validation 和 render_full_sequence 也会走同样 correction 路径。

已经验证：

- CPU random forward 中 position + opacity correction 路径可用。
- GPU 1-step dry-run 默认 correction 关闭；correction 训练链路尚未做长步数验证。

还需要做：

- 先跑 position-only 6000-step：`use_interaction_correction=true, correct_opacity=false`。
- 监控 `delta_mu` mean/max、`l_anchor_delta`、human render 是否漂移或破碎。
- 再跑 position+opacity，检查 opacity 是否大量夹到 0/1。
- 如果不稳定，降低 `gamma_mu/gamma_opacity`，提高 `delta_loss_w`，推迟 `correction_start_iter`。
- 增加 correction magnitude CSV，当前 delta 只在 `.pt` 中保存。

完成判据：

- position-only 不崩，指标不明显低于 dry-run。
- correction 后 interaction crop 有视觉改善。
- opacity correction 不导致人体透明、过浓或边界发糊。

#### Exp 7: Gaussian volume-aware features

当前已实现：

- 尚未完整实现。
- 当前 token 特征中只有 `log_scale`，还没有显式 covariance local 6D、extent along anchor normal、approx gap 等 volume-aware features。

已经验证：

- 当前基础 token 可以稳定 forward，为后续替换 feature 提供接口。

还需要做：

- 在 human token 和 scene token 中加入 covariance/local frame 表达。
- 计算 `extent_along_anchor_normal`。
- 增加 approximate anchor-scene gap feature，但先不加 gap loss。
- 做无 volume vs volume-aware token 的 ablation。

完成判据：

- 加入 volume features 后 forward/训练稳定。
- attention 更关注合理表面，低 opacity floaters 权重下降。
- interaction crop 不变差，最好改善。

#### Exp 8: Geometry bias in attention

当前已实现：

- 尚未实现正式 geometry bias。
- 当前 human pooling 内有一个局部 bias MLP，但 anchor-scene cross-attention logits 还没有加入 `B_geo`。

已经验证：

- 无 geometry bias 的 cross-attention dry-run 可以跑通。

还需要做：

- 在 cross-attention logits 中加入 `B_geo`。
- bias 输入建议从 distance、scene opacity、normal distance、approx gap、scene extent 开始。
- 做 no bias / distance-only bias / volume-aware bias 消融。
- 导出 bias value 和 attention weight 的关系，用于判断 bias 是否主导 attention。

完成判据：

- geometry bias 不导致 attention 塌缩。
- foot sole/back/palm anchor 对合理 scene surface 的权重提高。
- low-opacity floaters 权重下降。

#### Exp 9: Weak ellipsoid gap regularization

当前已实现：

- 尚未实现。
- 当前只实现了 correction 正则 `delta_loss`，没有 pairwise human-scene ellipsoid gap loss。

已经验证：

- scene query 已能提供 anchor 局部 topK scene candidates，为 gap regularization 提供候选集。

还需要做：

- 基于 anchor 局部 human-scene pairs 计算 ellipsoid radius 和 approximate gap。
- 实现 `softplus((margin-gap)/tau)^2` 弱非穿透正则。
- scene side 第一版建议 detach。
- 记录 negative gap ratio、mean penetration depth、anchor-wise gap statistics。
- 防止人体通过缩小 scale 或 opacity 逃避约束。

完成判据：

- negative gap 统计下降。
- 渲染质量不明显下降。
- 接触区域几何更稳定，不拉坏 scene。

#### Exp 10: Full ablation and evaluation

当前已实现：

- 已有可运行的 baseline 配置和 debug 输出格式。
- 已有原版 HUGS/Human3R 初始化历史对照记录。
- 当前已经能保存 full rendering、human crop validation、anchor attention debug 和 anchor-attention checkpoint。

已经验证：

- 1-step anchor-attention dry-run 全链路通过。

还需要做：

- 跑完整 ablation：HUGS baseline、anchors only、cross-attention dry-run、position correction、position+opacity、mean pooling、attention pooling。
- 增加 interaction crop metrics。
- 增加 attention maps、scene neighbor PLY/PNG、gap curves。
- 记录训练时间、显存、query time、attention forward time。
- 在至少一个序列上跑 6000/15000 steps，之后扩展多序列。

完成判据：

- 有完整表格对比全图指标、人体区域指标、interaction crop 指标、几何诊断指标和运行开销。
- 至少一个 correction/feature 组合相对 HUGS baseline 在接触区域或几何诊断上有稳定改善，同时全图指标不明显下降。

### 2.2 当前最短可执行路线

如果目标是尽快判断 baseline 是否可能带来效果提升，建议按下面顺序执行：

```text
1. Exp 0: HUGS baseline 6000 steps
2. Exp 5: anchor cross-attention dry-run 3000/6000 steps
3. Exp 6: position-only correction 6000 steps
4. Exp 6: position + opacity correction 6000 steps
5. Exp 3/7: mean pooling vs attention pooling token 消融
6. Exp 7/8/9: volume feature -> geometry bias -> weak gap regularization
```

当前代码最适合先推进第 1-4 步。第 5 步开始会涉及你后续最关心的 token fusion 设计，需要在 `hugs/models/anchor_attention.py` 中有计划地替换 feature 和 pooling。



### 2.3 Exp0 baseline 复现执行记录（2026-05-19）

为了满足“完全作为后续质量评估 baseline”的要求，已完成原版 HUGS/NeuMan `lab` baseline 复现：

```text
6000-step: exp0_hugs_original_6000_20260519
full-step: exp0_hugs_original_15000_20260519
```

执行原则：

- 使用 `cfg_files/release/neuman/hugs_human_scene.yaml`。
- 不开启 Human3R，不开启 anchors，不开启 anchor-attention。
- 非 quick 模式，保留原版 `optimize_init`。
- 仅因为当前环境没有 AMASS，使用 `scripts/run_neuman_human_scene_noamass.py` 禁用 animation；这不改变训练本身。
- full-step 组使用 release 默认 `train.num_steps=14998`，对应原版约 15000 步完整训练。

最终输出：

```text
6000-step:
output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_6000_20260519/2026-05-19_23-10-00

full-step:
output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54
```

验证集指标：

```text
6000-step: PSNR 25.6634, SSIM 0.9012, LPIPS 0.0843, human PSNR 18.7363, human SSIM 0.7432, human LPIPS 0.1652
full-step: PSNR 25.9582, SSIM 0.9147, LPIPS 0.0710, human PSNR 18.8005, human SSIM 0.7580, human LPIPS 0.1532
```

all split 统一评估指标：

```text
6000-step: all_psnr 25.9781, all_ssim 0.8945, human_crop_psnr 20.1687, human_crop_ssim 0.7642, human_mask_psnr 19.3641, bg_psnr 27.3815
full-step: all_psnr 26.4259, all_ssim 0.9103, human_crop_psnr 20.1585, human_crop_ssim 0.7810, human_mask_psnr 19.0400, bg_psnr 28.2458
```

统一评估脚本已新增并通过 GPU smoke / 正式评估：

```text
scripts/evaluate_gaussian_outputs.py
```

它将作为后续所有实验的统一评估入口，避免不同实验之间指标口径不一致。正式质量 baseline 优先使用 full-step 结果，6000-step 作为中期训练对照。

## 3. Exp 0: HUGS Baseline 跑通

### 目标

确认原始 HUGS 代码在目标数据上可以稳定训练、渲染、评估。

### 最小可落地做法

不改模型，只运行原始训练脚本。

需要保存：

```text
outputs/hugs_baseline/
  checkpoints/
  renders/
  human_only_renders/
  scene_only_renders/
  metrics.json
```

### 需要记录

- 训练总 iteration；
- densification schedule；
- 最终 human Gaussian 数量；
- 最终 scene Gaussian 数量；
- 每个 iteration time；
- 显存占用；
- PSNR / SSIM / LPIPS。

### 验收标准

- training 不崩；
- human-only rendering 合理；
- scene-only rendering 合理；
- full rendering 与 HUGS 预期质量接近；
- 有可复现配置。

---

## 4. Exp 1: HUGS + Human3R Initialization

### 目标

验证 Human3R 初始化是否改善人体、场景、相机的相对位置，尤其是粗接触关系。

### 最小可落地做法

将 Human3R 输出转换为 HUGS 可用初始化：

```text
camera extrinsics / intrinsics
scene pointmap or scene point cloud
SMPL / SMPL-X pose and global transform
human-scene metric scale
```

如果完整替换初始化太复杂，第一版至少完成：

```text
Human3R camera / scale / human global transform -> HUGS init
```

### 需要实现

```text
human3r_to_hugs.py 或 initializer/human3r_initializer.py
```

功能：

1. 读取 Human3R 输出；
2. 对齐坐标系；
3. 生成 HUGS 初始化文件；
4. 保留 fallback 到原 HUGS 初始化。

### 输出

```text
outputs/hugs_human3r_init/
  init_debug/
    camera_trajectory.ply
    human_scene_alignment.ply
  renders/
  metrics.json
```

### 对比

```text
HUGS baseline
HUGS + Human3R init
```

### 验收标准

- 人体和场景在 world space 中相对位置更合理；
- 脚底/地面、人体/背景边界的粗位置不明显错开；
- 渲染指标不显著下降；
- 初始化可复现。

---

## 5. Exp 2: Anchor Debug Experiments

### 目标

验证 semantic interaction anchors 能否稳定组织人体 Gaussians，并查询合理的局部 scene Gaussians。

这一步对应独立文档：

```text
Anchor_Experiments_Implementation.md
```

### 最小可落地做法

先不接网络、不改训练 loss，只做可视化和统计。

需要完成 5 个 debug 功能：

```text
A. Anchor template visualization
B. Initial human Gaussian -> anchor binding visualization
C. Densification inheritance visualization
D. Anchor-local Gaussian statistics export
E. Anchor -> scene Gaussian neighbor query visualization
```

### 关键实现

#### Anchor 初始化

从 SMPL / SMPL-X template semantic region 计算：

```text
anchor_pos = mean(region vertices)
anchor_normal = mean(region normals)
anchor_lbs = mean(region LBS weights)
```

#### Human Gaussian 绑定

在 HUGS 初始化 human Gaussians 时，在 canonical space 绑定到 top-m anchors：

```text
score(g, a) = -distance^2 / sigma^2 - lambda_lbs * LBS_distance + body_part_bonus
```

推荐：

```text
m = 1 or 2
```

#### Densification 继承

```text
clone/split: new Gaussian inherits parent anchor binding
prune: delete corresponding binding
```

### 输出

```text
outputs/anchor_debug/
  template_anchors.ply
  human_gaussians_anchor_init.ply
  human_gaussians_anchor_iter3000.ply
  anchor_stats_iterXXXX.csv
  scene_neighbors_frameXXXX_right_sole.ply
```

### 验收标准

- anchors 位于正确人体语义区域；
- human Gaussians 按 anchor 着色后区域连续；
- split/clone/prune 后 anchor binding 不随机；
- foot sole anchor 能查询到地面附近 scene Gaussians；
- palm/back/buttocks 等 anchor 查询结果可解释。

---

## 6. Exp 3: Anchor Token Encoder Dry-run

### 目标

实现将 anchor 绑定的人体 Gaussians 聚合为 human anchor token 的网络，但先不影响渲染。

### 最小可落地做法

先实现 forward-only，并导出 token 统计。不要接 correction，不改 renderer。

### 输入

对每个 anchor a：

```text
bound human Gaussians G_a
anchor_world position
anchor local frame
anchor id embedding
body part embedding
```

每个 Gaussian 的基础特征：

```text
rel_mu_local: 3
||rel_mu||: 1
normal_distance: 1
opacity: 1
log_scale: 3
covariance_local_6d: 6
projected_lbs_feature: 16
anchor_assignment_weight: 1
```

### 网络结构

#### GaussianEncoder

```text
Linear(input_dim, 128)
GELU
LayerNorm(128)
Linear(128, 128)
GELU
```

#### AnchorQueryMLP

```text
input = anchor_embed + part_embed + optional pose feature
Linear(input_dim, 128)
GELU
Linear(128, 128)
```

#### Attention Pooling

```text
k_g = W_k(e_g)
v_g = W_v(e_g)
score = q_a dot k_g / sqrt(D) + bias_ag
alpha = softmax(score over bound Gaussians)
human_anchor_token = sum alpha * v_g
```

#### BiasMLP

```text
input = [distance, normal_distance, opacity, assignment_weight]
Linear(4, 32)
GELU
Linear(32, 1)
```

### 最小版本

同时支持：

```text
pooling = mean
pooling = attention
```

先让 `mean` 跑通，再启用 `attention`。

### 输出

```text
outputs/token_debug/
  human_anchor_tokens_iterXXXX.pt
  token_norms.csv
  pooling_weights_anchor_right_sole.pt
```

### 验收标准

- forward 不崩；
- token 没有 NaN；
- 不同 anchors 的 token 有差异；
- attention pooling weights 不完全均匀，也不塌缩到单个异常 Gaussian；
- 训练时间增加可接受。

---

## 7. Exp 4: Scene Gaussian Local Query + Scene Token Encoder

### 目标

每个人体 anchor 查询附近 scene Gaussians，并编码成 scene tokens。

### 最小可落地做法

先用简单 top-K 查询跑通。

```text
K_s = 16 or 32
```

如果 scene Gaussians 数量不大，可先用 torch 距离 topk；如果过慢，再换 KD-tree / voxel hash。

### 查询输入

```text
anchor_world: [A, 3]
scene_mu: [Ns, 3]
scene_opacity: [Ns, 1]
scene_scale: [Ns, 3]
scene_rot: [Ns, ...]
scene_color_or_SH: [Ns, C]
```

### 候选过滤

最小版本：

```text
opacity > 0.01 or 0.05
```

可选：

```text
distance < radius
scale not too large
```

### Scene Token Feature

每个 scene Gaussian 相对 anchor 编码：

```text
rel_mu_local
||rel_mu||
normal_distance
opacity
log_scale
covariance_local_6d
low_dim_color_or_SH
```

### SceneGaussianEncoder

可以与 GaussianEncoder 同结构，但参数独立：

```text
Linear(input_dim, 128)
GELU
LayerNorm(128)
Linear(128, 128)
GELU
```

输出：

```text
scene_tokens: [A, K_s, D]
```

### 输出

```text
outputs/scene_query_debug/
  scene_neighbors_frameXXXX_anchorXXXX.ply
  scene_tokens_iterXXXX.pt
  query_time.csv
```

### 验收标准

- right_sole 查询到地面区域；
- palm 查询到附近场景表面；
- 查询不被大量 low-opacity floaters 干扰；
- 查询耗时可接受。

---

## 8. Exp 5: Anchor-Scene Cross-Attention Dry-run

### 目标

实现 human anchor token 与 local scene tokens 的 cross-attention，但先不修正 Gaussians。

### 最小可落地做法

只 forward attention，保存 attention weights 和 contact context。

### 网络结构

```text
D = 128
num_heads = 4
num_layers = 1
```

形式：

```text
Q = human_anchor_token: [A, D]
K,V = scene_tokens: [A, K_s, D]
context = CrossAttention(Q, K, V): [A, D]
attention_weights: [A, K_s]
```

### 输出

```text
outputs/attention_debug/
  attention_weights_iterXXXX.pt
  contact_context_iterXXXX.pt
  attention_scene_neighbors_frameXXXX_right_sole.ply
```

### 可视化

对每个 anchor，将 scene neighbors 按 attention weight 上色。

### 验收标准

- attention weights 无 NaN；
- right_sole 等 anchor 对附近地面有较高权重；
- attention 不完全均匀；
- attention 不长期塌缩到 low-opacity floater；
- 训练时间增加可接受。

---

## 9. Exp 6: Attention Context -> Human Gaussian Correction

### 目标

让 attention 真正进入重建闭环：contact context 小幅修正 human Gaussian 参数，然后进入 renderer。

### 最小可落地做法

第一版只修正：

```text
position / mu
opacity
```

不要先修正 scale / rotation。

### Context 分配

每个 human Gaussian g 根据 anchor binding 得到 context：

```text
ctx_g = sum_a bind_weight(g, a) * contact_context_a
```

### Correction Head

#### Position correction

```text
DeltaMuMLP:
  input = [gaussian_feature_g, ctx_g]
  Linear(input_dim, 128)
  GELU
  Linear(128, 3)
```

```text
mu'_g = mu_g + gamma_mu * delta_mu_g
```

推荐：

```text
gamma_mu = 0.02 - 0.1
```

#### Opacity correction

```text
DeltaOpacityMLP:
  input = [opacity_g, ctx_g]
  Linear(input_dim, 64)
  GELU
  Linear(64, 1)
```

```text
opacity'_g = opacity_g + gamma_opacity * delta_opacity_g
```

推荐：

```text
gamma_opacity = 0.05 - 0.1
```

### Loss

主 loss 保持 HUGS 原始：

```text
L_render = L_rgb + L_ssim + L_perceptual + L_human_mask + L_lbs_reg
```

新增 correction 正则：

```text
L_delta = ||delta_mu||^2 + ||delta_opacity||^2
```

总损失：

```text
L = L_render + lambda_delta * L_delta
```

### 训练调度

不要从第 0 iter 开启。建议：

```text
0 - warmup_iters: HUGS + Human3R only
warmup 后: enable attention correction
```

### 实验对比

```text
HUGS + Human3R
+ anchor cross-attention forward only
+ anchor cross-attention correction position only
+ anchor cross-attention correction position + opacity
```

### 验收标准

- training 不崩；
- delta_mu 不爆炸；
- opacity 不大量饱和到 0 或 1；
- interaction crop 或边界可视化有改善；
- scene-only / human-only 渲染不变差。

---

## 10. Exp 7: Add Gaussian Volume-aware Features

### 目标

把 Gaussian 的体积属性显式纳入 token，使 attention 能感知 ellipsoid shape。

### 最小可落地做法

先不加 gap loss，只把 volume features 加入 token。

### Human token 增强

在人高斯特征中加入：

```text
covariance_local_6d
extent_along_anchor_normal = sqrt(n_anchor^T Sigma n_anchor)
scale statistics
opacity statistics
```

### Scene token 增强

在 scene token 中加入：

```text
covariance_local_6d
extent_along_anchor_normal
opacity
```

### Relation feature，可选第一版

对 anchor 与 scene Gaussian 计算近似 gap：

```text
anchor_scene_distance - scene_radius_along_direction - pooled_human_radius
```

先不用 pairwise human-scene gap，降低复杂度。

### 实验对比

```text
attention correction without volume features
attention correction with covariance / scale features
attention correction with approximate gap features
```

### 验收标准

- token forward 稳定；
- 训练速度没有明显不可接受下降；
- attention 可视化更关注合理表面；
- interaction crop / 接触边界不变差，最好改善。

---

## 11. Exp 8: Add Geometry Bias in Attention

### 目标

让 attention logits 显式受到几何关系引导。

### 最小可落地做法

在 attention logit 中加入 `B_geo`：

```text
logit = QK^T / sqrt(D) + B_geo
```

### Geometry Bias 输入

第一版输入：

```text
distance
scene_opacity
normal_distance
approx_gap
scene_extent_along_anchor_normal
```

### BiasMLP

```text
Linear(input_dim, 32)
GELU
Linear(32, 1)
```

输出：

```text
B_geo: [A, K_s]
```

### 实验对比

```text
no geometry bias
distance-only bias
volume-aware geometry bias
```

### 验收标准

- attention 不塌缩；
- right_sole 更关注地面；
- low-opacity floaters 权重降低；
- interaction crop 或几何诊断有改善。

---

## 12. Exp 9: Weak Ellipsoid Gap Regularization

### 目标

在 correction 后的局部 human-scene Gaussian pairs 上加入弱非穿透安全正则。

注意：这不是主方法，不应替代 attention correction。它只是防止明显椭球穿透。

### 最小可落地做法

只对 anchor 局部 top-K pair 计算。

对 human Gaussian h 和 scene Gaussian s：

```text
Sigma = R diag(scale^2) R^T
r_G(n) = kappa * sqrt(n^T Sigma n)
n = normalize(mu_s - mu_h)
gap = ||mu_s - mu_h|| - r_h(n) - r_s(-n)
```

使用 correction 后的 human Gaussian 属性计算 `gap'`。

### Loss

```text
L_gap = sum w_ij * softplus((margin - gap'_ij) / tau)^2
```

权重：

```text
w_ij = anchor_bind_weight * attention_weight * opacity_h * opacity_s
```

第一版建议：

```text
scene side detach
lambda_gap very small
warm-up 后再开启
```

### 实验对比

```text
attention correction without L_gap
attention correction with weak L_gap
```

### 需要记录的几何指标

```text
negative_gap_ratio
mean_negative_gap
max_penetration_depth
anchor-wise gap statistics
```

### 验收标准

- negative gap 统计下降；
- 渲染质量不明显下降；
- human Gaussians 不通过缩小 scale 逃避约束；
- scene geometry 不被拉坏。

---

## 13. Exp 10: Full Ablation and Evaluation

### 必做 ablation

```text
A. HUGS baseline
B. HUGS + Human3R init
C. + anchor token + scene query + cross-attention, no correction
D. + position correction
E. + position + opacity correction
F. + volume-aware token features
G. + geometry bias
H. + weak ellipsoid gap regularization
```

### 可选 ablation

```text
mean pooling vs attention pooling
K_s = 16 vs 32
D = 64 vs 128
with / without opacity filtering
position-only vs position+opacity correction
with / without scene side detach in L_gap
```

### 全图指标

```text
PSNR
SSIM
LPIPS
```

### 人体区域指标

```text
human-mask PSNR / SSIM / LPIPS
```

### interaction crop 指标

对 feet-ground、hand-scene、buttocks/back 等区域 crop：

```text
crop PSNR
crop SSIM
crop LPIPS
```

### 几何诊断指标

```text
negative_gap_ratio
mean_penetration_depth
anchor_scene_nearest_distance
temporal_gap_stability
```

### 可视化输出

```text
full rendering
human-only rendering
scene-only rendering
anchor attention maps
anchor scene neighbors
anchor-wise gap curves
```

### 运行时间指标

必须记录：

```text
training time per iteration
GPU memory
total training time
anchor query time
attention forward time
```

---

## 14. 推荐训练调度

第一版建议：

```text
0 - 3k iters:
  HUGS + Human3R warm-up
  no anchor attention correction

3k - 6k iters:
  enable anchor token encoder
  enable scene query
  enable cross-attention
  enable position / opacity correction
  no volume gap loss

6k - 10k iters:
  enable volume-aware token features
  enable geometry bias

10k - end:
  enable weak ellipsoid gap regularization
  small lambda_gap
```

具体 iteration 根据实际 HUGS schedule 调整。

如果 HUGS densification 从 3k 开始或在 600 iter 间隔进行，需要保证：

```text
anchor binding 在 clone/split/prune 时同步更新
scene spatial index 在 scene densification 后周期性更新
```

---

## 15. 最小可行 Baseline 定义

如果时间有限，先实现这个最小版本：

```text
HUGS + Human3R init
+ semantic anchors
+ human anchor token encoder
+ scene local query
+ anchor-scene cross-attention
+ position / opacity correction
+ rendering loss + delta regularization
```

暂时不做：

```text
StM
scale / rotation correction
pairwise ellipsoid gap loss
complex temporal module
multi-layer transformer
```

该版本能回答最关键问题：

```text
anchor-based cross-attention 是否能通过 rendering loss 学到有用的人体-场景交互修正？
```

---

## 16. 失败情况与排查

### attention 学不到东西

检查：

```text
anchor binding 是否合理
scene query 是否查到正确场景区域
attention weights 是否均匀或塌缩
correction gamma 是否过小或过大
```

### 训练变慢太多

简化：

```text
减少 anchors
K_s 从 32 降到 16
D 从 128 降到 64
只对 feet / hands / buttocks anchors 开启
scene index 降频更新
先用 mean pooling
```

### 渲染质量下降

检查：

```text
delta_mu 是否过大
opacity 是否饱和
correction 是否过早开启
lambda_delta 是否过小
scene side 是否被错误修改
```

### gap loss 导致形状异常

处理：

```text
降低 lambda_gap
detach scene side
先不修 scale / rotation
增加 scale / rotation regularization
只惩罚强穿透，不做吸引接触
```

---

## 17. Coding Agent 实现要求

1. 每个模块都必须有 config 开关。
2. 每个实验必须能单独运行或复现实验配置。
3. 新增模块不要破坏原 HUGS baseline。
4. 所有 debug 输出保存到独立目录。
5. 训练日志中记录新增模块耗时。
6. 出现 NaN 时保存当前 batch 的 anchor / scene query / attention debug 文件。
7. 优先实现最小可行版本，再逐步加 volume features、geometry bias 和 gap regularization。

---

## 18. 当前不做的内容

本阶段明确不做：

- StM 集成；
- 完整 temporal HexPlane；
- 大规模预训练 contact prior；
- dense per-Gaussian contact label；
- 全局 human-scene Gaussian pairwise collision；
- 复杂物理模拟；
- 直接替换 HUGS 的 learned LBS deformation。

---

## 19. 最终期望结果

如果 baseline 有效，应该至少观察到：

1. interaction crop 的视觉质量改善；
2. feet-ground / hand-scene 边界更干净；
3. scene-only rendering 中人体残影减少；
4. attention 可视化能关注合理 scene neighbors；
5. weak gap regularization 后 negative gap ratio 降低；
6. 全图 PSNR / SSIM 不明显下降，最好有小幅提升；
7. 训练额外开销可接受。

这说明 anchor-based cross-attention 作为 HSI 感知模块是有继续扩展价值的。

---

## 20. 2026-05-19 baseline 落地状态

当前已按“原版 HUGS 初始化 + 原版 HUGS 训练流程”为主线，搭建第一版最小 anchor-attention baseline。

### 已落地模块

```text
hugs/models/anchor_attention.py
  AnchorSceneAttentionBaseline
    - human Gaussian -> anchor token
    - scene Gaussian KNN query
    - scene token encoder
    - per-anchor cross-attention
    - optional delta_mu / delta_opacity correction
    - pt/csv debug export

hugs/trainer/gs_trainer.py
  setup_anchor_attention_baseline()
  maybe_apply_anchor_attention()
  train/validate/render_full_sequence 接入

cfg_files/release/neuman/hugs_anchor_attention_baseline.yaml
  原版 HUGS human_scene 配置 + anchor_attention 开关
```

### 当前默认实验模式

`hugs_anchor_attention_baseline.yaml` 默认开启：

```yaml
use_anchors: true
use_anchor_token_encoder: true
use_scene_query: true
use_cross_attention: true
use_interaction_correction: false
```

因此默认是 Exp3-5 的 dry-run/debug baseline：会 forward token/query/attention 并导出统计，但不会改变渲染结果。Exp6 correction 需要通过命令行显式开启：

```bash
anchor_attention.use_interaction_correction=true
anchor_attention.correct_opacity=false  # position only
```

或：

```bash
anchor_attention.use_interaction_correction=true
anchor_attention.correct_opacity=true   # position + opacity
```

### 保留的灵活替换点

- 初始化方案暂不修改，当前使用 HUGS 原版初始化；后续 Human3R/其他初始化只需接到 HUGS 数据/模型初始化层。
- token 融合方式集中在 `AnchorSceneAttentionBaseline._pool_human_tokens()` 和 encoder MLP，后续可替换为新的 Gaussian feature/token fusion。
- scene query 集中在 `_query_scene_tokens()`，后续可替换为 KD-tree、voxel hash、visibility-aware query。
- correction 集中在 `_context_per_gaussian()` 和 correction heads，后续可扩展 scale/rotation/geometry gap。

### 已完成验证

- `py_compile` 通过：`hugs/models/anchor_attention.py`、`hugs/trainer/gs_trainer.py`、`hugs/cfg/config.py`。
- 纯 CPU 随机张量前向烟测通过：attention 输出形状 `[A,K]`，可写出 `.pt/.csv` debug 文件。
- 完整 HUGS 训练 smoke 需要在 `conda activate hugs` 且 `simple_knn` CUDA 扩展可导入的环境中执行。

GPU 1-step 训练 smoke 也已在 `gpu-l40-1` 通过：

```bash
ssh gpu-l40-1 'source /home/u202420081000003/anaconda3/etc/profile.d/conda.sh && \
  conda activate hugs && \
  cd /hdd/u202420081000003/ml-hugs && \
  python scripts/run_neuman_human_scene_noamass.py \
    --seq lab \
    --cfg-file cfg_files/release/neuman/hugs_anchor_attention_baseline.yaml \
    --quick \
    --quick-steps 1 \
    --exp-name anchor_attention_1step_smoke \
    train.val_interval=1 \
    train.save_ckpt_interval=1 \
    train.anim_interval=-1 \
    human.canon_nframes=2 \
    train.save_progress_images=false \
    scene.densify_from_iter=1000 \
    human.densify_from_iter=1000 \
    anchor_attention.debug_interval=1'
```

输出目录：

```text
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_1step_smoke/2026-05-19_02-12-59
```

该 smoke 确认：

- `Created anchor bindings: torch.Size([110210, 2])`；
- `Anchor-attention baseline initialized with 16 anchors`；
- 训练、validation、final validation、`render_full_sequence()` 均完成；
- 生成 `anchor_attention_iter000000.{pt,csv}`、`anchor_attention_iter000001.{pt,csv}`；
- 生成 `ckpt/anchor_attention_000001.pth` 和 `ckpt/anchor_attention_final.pth`。

1-step 指标只用于链路验证，不代表方法质量：

```text
final hugs_psnr:        9.8982
final hugs_ssim:        0.5529
final hugs_lpips:       0.6635
final hugs_human_psnr:  11.1877
final hugs_human_ssim:  0.5218
final hugs_human_lpips: 0.6817
```

## 21. 接触质量评估方案修订（2026-05-20）

### 问题背景

当前 baseline 的主要监督仍然来自 HUGS 原始渲染损失：L1、SSIM、LPIPS patch、human crop loss、LBS regularization 等。这些损失可以约束图像重建质量，但不能直接证明“人体-场景接触质量”变好了。

同时，HUGS 的 scene Gaussians 是面向渲染优化的 radiance representation，不是显式、干净、可物理解释的 scene surface。它可能存在漂浮点、离散噪声、尺度不稳定、局部 geometry 不闭合等问题。因此不能直接把 HUGS scene Gaussians 当作接触监督或接触评估真值。

本项目后续需要补充一个独立于 HUGS 渲染表示的 contact evaluation channel。该通道用于判断接触相关实验是否真的改善了人体-场景关系，而不是只改善了全图 PSNR/SSIM/LPIPS。

### 参考相关工作后的结论

相关 human-scene / human-object interaction 工作通常不会只用渲染指标评价接触质量，而会引入以下信息之一：

- 可信 3D scene geometry：例如 PROX 使用 3D scene scan，并在优化中加入 contact 与 interpenetration constraints。
- 身体表面 contact label：例如 RICH / BSTRO 提供或学习 dense body-scene contact annotation。
- 图像到身体表面的 contact predictor：例如 DECO/DAMON 预测 SMPL/SMPL-X 顶点级 contact probability。
- body-centric contact representation：例如 POSA 在 SMPL-X 顶点上建模 contact probability 和 semantic scene label。
- Gaussian human-object 工作中的外部 contact prior：例如 HOGS 类工作会借助 pretrained contact predictor 或 contact/separation losses 来约束物理合理性。

因此，我们后续实验不能只说“渲染指标提高所以接触更好”，而需要至少引入 contact pseudo-label、独立 scene proxy、局部接触指标中的一类或多类。

### 本项目接触评估的基本原则

1. 不使用 HUGS scene Gaussians 作为唯一接触真值。
2. 接触评估和 HUGS 训练表示解耦：评估可以用独立 scene proxy，训练仍然可以使用 HUGS scene Gaussians 和 anchor-attention。
3. 没有显式 GT contact 时，先使用 pseudo-label 和 proxy metrics，但必须记录置信度。
4. 接触指标只在高置信区域解释，低置信区域只作为可视化和人工检查。
5. 任何 contact/correction 实验都必须同时报告渲染指标和接触 proxy 指标。

### 评估通道设计

后续建议新增两个脚本：

```text
scripts/build_contact_eval_proxy.py
scripts/evaluate_contact_quality.py
```

其中 `build_contact_eval_proxy.py` 负责构建独立 scene proxy，不能直接依赖训练后的 HUGS scene Gaussians。可选来源包括：

```text
1. NeuMan / COLMAP 静态场景点云
2. DepthPro / DepthAnything 等 monocular depth 估计
3. 多帧 depth fusion 后的静态场景点云
4. 利用 human mask 去除人体区域后的背景 depth / point cloud
5. 如果未来有更干净的 scan / mesh，可直接作为 eval-only geometry
```

`evaluate_contact_quality.py` 负责读取 HUGS 输出、人体 SMPL/SMPL-X 状态、contact pseudo-label 和 scene proxy，输出逐帧逐 anchor / vertex 的接触指标。

### Contact pseudo-label 来源

短期最低可行版本：

```text
feet-only contact candidates:
  foot / toe / heel anchors
  low vertical velocity
  long temporal persistence
  near ground or near static scene proxy
```

中期版本：

```text
full-body contact candidates:
  DECO / BSTRO / POSA-like pretrained contact predictor
  输出 SMPL/SMPL-X vertex contact probability
```

长期版本：

```text
if available:
  真实 contact annotation
  高质量 scene scan / mesh
  multi-view depth fused scene surface
```

### 指标定义

推荐先实现以下指标：

```text
Contact Distance:
  被判定为 contact 的人体顶点 / anchor 到 independent scene proxy 的距离。

Contact Recall @ tau:
  contact candidate 中距离 scene proxy 小于 tau 的比例，例如 tau=2cm/5cm/10cm。

Floating Rate:
  contact candidate 离 scene proxy 过远的比例。

Penetration Proxy:
  如果 scene proxy 可拟合局部平面或 TSDF，则统计人体点进入 scene proxy 内部的比例和深度。
  如果局部 geometry 置信度低，则不报告 penetration，只记录 low confidence。

Contact Stability:
  接触距离、contact anchor、contact probability 在相邻帧上的 temporal variance / flicker。

Contact ROI Rendering:
  将 contact anchor / contact vertices 投影回图像，在局部 crop 上计算 PSNR / SSIM / LPIPS。
```

注意：Contact Distance 和 Penetration Proxy 必须带有 scene proxy confidence。不能在 geometry 明显散乱的区域强行解释为接触变好或变坏。

### 第一阶段最小实验

建议从 feet-only 开始，因为脚底-地面接触最容易定义，也最容易可视化。

```text
输入:
  Exp0 full baseline 输出目录
  原始 NeuMan lab 图像和 mask
  HUGS / SMPL 人体状态
  eval-only ground / scene proxy

输出:
  contact_eval_summary.json
  per_frame_contact_metrics.csv
  contact_visualization.ply
  contact_roi_images/
```

第一阶段只需要回答：

```text
1. 原版 HUGS 在脚底区域是否存在 floating / penetration proxy？
2. 接触相关帧的局部 render crop 是否是误差高发区域？
3. 后续 correction 是否能在不牺牲全图渲染的情况下改善这些 proxy？
```

### 对当前 anchor-attention baseline 的影响

当前 anchor-attention 代码仍然可以继续作为 interaction module 使用，但接触质量评估需要外接上述 eval channel。也就是说：

```text
training path:
  HUGS human/scene Gaussians -> anchor query -> scene token -> correction -> render loss

evaluation path:
  HUGS output + independent scene proxy + contact pseudo-label -> contact metrics
```

这可以避免“用 HUGS 自己不可靠的 scene Gaussians 评价自己”的循环论证。

### 后续优先级

1. 先实现 `evaluate_contact_quality.py` 的 feet-only / ground-proxy 版本。
2. 再尝试接入 monocular depth fusion 构建更通用的 scene proxy。
3. 再调研和接入 DECO / BSTRO / POSA 类 pretrained contact predictor。
4. 最后把 contact metrics 纳入所有 correction / token fusion ablation 表格中。

## 22. 从最小 baseline 到可提升质量 pipeline 的路线图（2026-05-20）

### 22.1 当前出发点

当前已经具备：

```text
1. Exp0 原版 HUGS baseline：6000-step 和 full-step 已复现并统一评估。
2. Anchor-attention 最小工程骨架：anchor binding、scene query、token encoder、cross-attention、debug tensor/csv、可选 correction。
3. 统一渲染评估脚本：scripts/evaluate_gaussian_outputs.py。
4. 接触质量评估方案：需要新增 independent scene proxy + contact pseudo-label + contact metrics。
```

当前还没有证明：

```text
1. scene token 是否真正包含对接触有用的信息。
2. correction 是否能稳定改善接触区域，而不是只改变渲染误差。
3. HUGS scene Gaussians 的几何噪声是否会误导 probe / attention。
4. human-scene 初始化关系不好是否会让 anchor 查询到错误场景区域。
```

因此，后续目标不是直接做大模型，而是逐步建立“可评估、可诊断、可对比、可提升”的 pipeline。

### 22.2 总体阶段

建议按下面顺序推进：

```text
Stage A: 建立接触评估通道
Stage B: 诊断当前 anchor probe / attention 是否可信
Stage C: 验证初始化质量对 probe / attention 的影响
Stage D: 开启 zero-init learnable correction，做小步可控训练
Stage E: token fusion / scene query ablation
Stage F: 引入更强初始化或 scene proxy，对比完整 pipeline
Stage G: 多序列和正式 ablation 表格
```

### 22.3 Stage A：接触评估通道

先实现：

```text
scripts/build_contact_eval_proxy.py
scripts/evaluate_contact_quality.py
```

第一版只做 feet-only / ground-proxy：

```text
输入:
  Exp0 full baseline 输出目录
  原始图像和 mask
  SMPL / HUGS human 状态
  eval-only scene proxy

输出:
  contact_eval_summary.json
  per_frame_contact_metrics.csv
  contact_visualization.ply
  contact_roi_images/
```

第一阶段指标：

```text
Contact Distance
Contact Recall @ 2cm/5cm/10cm
Floating Rate
Penetration Proxy with confidence
Contact Stability
Contact ROI PSNR / SSIM / LPIPS
```

目的：先让“接触是否更好”有独立评价依据，而不是只看全图渲染指标。

### 22.4 Stage B：诊断当前 probe / attention

当前 scene query 是：每个 human anchor 查询附近 top-k scene Gaussians，逐点编码后 attention pooling 成一个 anchor-level scene context。需要先确认它有没有查询到合理区域。

需要输出：

```text
anchor_attention_iter*.pt
anchor_attention_iter*.csv
anchor_scene_probe.ply
anchor_scene_probe_summary.json
```

诊断指标：

```text
nearest_scene_distance
trimmed_scene_distance
attention_entropy
attention_weighted_distance
top-k scene opacity / scale / color statistics
scene candidate count
probe temporal stability
```

目的：判断每个 anchor 的 probe 是否稳定、是否被漂浮点或背景噪声带偏、attention 是否塌缩到单个噪声高斯点。

### 22.5 Stage C：初始化质量对 probe / attention 的影响

这是新增重点实验。当前有两个初始化相关风险：

```text
Risk 1: 背景 / scene Gaussians 几何差且杂乱。
  影响：anchor probe 可能查询到漂浮点、噪声点、非表面点，scene token 变成错误上下文。

Risk 2: 人体和场景的初始相对关系不好。
  影响：anchor_world 位置偏移，query 半径 / top-k 结果偏离真实接触区域，attention 学到错误 human-scene relation。
```

因此，后续处理好初始化后，必须做对比实验：

```text
Exp Init-0: 原版 HUGS 初始化 + 原版 scene Gaussians
Exp Init-1: 清理/重建后的 scene 初始化 + 原版 human 初始化
Exp Init-2: 原版 scene 初始化 + 改进 human-scene 对齐
Exp Init-3: 改进 scene 初始化 + 改进 human-scene 对齐
```

每组都不先开启 correction，只跑 dry-run probe / attention，比较：

```text
probe nearest / trimmed distance
attention entropy
attention weighted distance
scene candidate opacity / scale distribution
contact proxy metrics
contact ROI render metrics
```

判断标准：

```text
1. scene 初始化更干净后，probe 是否更稳定，attention 是否更少被离群点吸引。
2. human-scene 初始关系更好后，contact anchors 是否查询到更合理的局部 scene context。
3. 如果初始化变化显著改变 attention/probe 统计，则说明后续 correction 必须和初始化共同评估。
```

这组实验要在 correction 之前做，因为如果 probe 本身已经错了，learnable correction 可能只是学习补偿错误上下文，泛化和解释性都会变差。

### 22.6 Stage D：zero-init learnable correction

在 Stage A-C 建立后，再开启可学习 correction。第一版只改 xyz，不动 opacity / scale / rotation。

推荐结构：

```text
human gaussian feature
+ anchor context
+ scene context token
-> gate_head
-> bounded delta_xyz_head
```

约束：

```text
delta head 最后一层 zero-init
gate bias 初始化为负值，例如 -4
bounded_delta = tanh(raw_delta) * max_delta
max_delta = 0.01 或 0.02
correction_start_iter = 3000
delta_loss_w > 0
```

第一轮实验：

```text
Exp Corr-0: dry-run, correction off
Exp Corr-1: xyz correction, all anchors, max_delta=0.01
Exp Corr-2: xyz correction, contact candidate anchors only
Exp Corr-3: xyz correction + temporal/delta regularization
```

评价：

```text
全图 val/all PSNR/SSIM/LPIPS
human crop metrics
Contact Distance / Floating / Penetration proxy
Contact ROI metrics
delta_mu spatial distribution
attention/probe statistics before vs after training
```

### 22.7 Stage E：token fusion / scene query ablation

在确认 correction 可稳定训练后，再做 token 方式对比。

候选：

```text
Scene token fusion:
  mean pooling
  attention pooling
  distance-aware attention
  opacity/scale-aware attention
  multi-token per anchor

Human token fusion:
  mean pooling
  anchor attention pooling
  contact-candidate weighted pooling

Scene query:
  top-k nearest
  opacity filtered top-k
  scale filtered top-k
  visibility-aware query
  eval-proxy-guided query, if available
```

每次只改一个变量，使用同一 Exp0 baseline、同一 contact eval、同一 render eval。

### 22.8 Stage F：初始化与最终 pipeline 对比

当 initialization 改进完成后，需要和 Stage D/E 中最好的 correction/token 组合交叉验证。

建议矩阵：

```text
Baseline:
  original HUGS init + no correction

Method A:
  original HUGS init + best correction/token

Method B:
  improved init + no correction

Method C:
  improved init + best correction/token
```

如果 Method B 已经改善很多，说明初始化是主要瓶颈；如果 Method C 继续改善，说明 interaction module 有额外贡献。

### 22.9 最终可行 baseline 的定义

只有同时满足下面条件，才认为从“最小可运行 baseline”推进到了“实际可提高质量 baseline”：

```text
1. 原版 HUGS full-step baseline 可复现。
2. 接触评估通道可运行，并输出独立 contact proxy 指标。
3. probe / attention 诊断证明 scene context 不是随机噪声。
4. correction 训练稳定，不破坏全图渲染质量。
5. contact ROI 或 contact proxy 指标有稳定改善。
6. 初始化对 probe / attention 的影响被单独对比过。
7. 至少一个序列上 full-step 结果优于 Exp0，再扩展到多序列。
```

### 22.10 推荐最近三步

```text
Step 1: 实现 feet-only contact eval，先让接触指标能跑。
Step 2: 实现 anchor probe diagnostic，对 Exp0 full baseline 导出 probe/attention 可视化和统计。
Step 3: 做 Init-0 vs improved init 的 dry-run probe 对比，确认初始化是否显著影响 scene token。
```

这三步完成后，再开启 learnable correction 会更有依据。

### 22.11 Stage A-C 首版实现记录（2026-05-20）

已完成 Stage A-C 的首版工程实现，并在 Exp0 full baseline 上跑通小规模 smoke。

新增脚本：

```text
scripts/contact_eval_lib.py
scripts/build_contact_eval_proxy.py
scripts/evaluate_contact_quality.py
scripts/diagnose_anchor_probe_attention.py
scripts/compare_probe_diagnostics.py
```

功能对应关系：

```text
Stage A:
  build_contact_eval_proxy.py
    构建 eval-only scene proxy，当前支持 COLMAP points3D.txt 和可选 depth npy fusion。

  evaluate_contact_quality.py
    feet-only contact proxy 评估，输出 contact distance、contact recall、ROI PSNR/SSIM、anchor trajectory PLY、render/GT/absdiff crops。

Stage B:
  diagnose_anchor_probe_attention.py
    对 anchor scene query / attention 做诊断，输出 nearest/trimmed distance、attention entropy、attention weighted distance、scene opacity/scale 统计、overlay/render/PLY。

Stage C:
  compare_probe_diagnostics.py
    后续不同初始化方案完成后，可把多个 summary.json 汇总成对比 CSV。
```

Smoke 使用的 Exp0 full baseline：

```text
output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54
```

Smoke 输出：

```text
output/contact_eval_proxy/neuman_lab_exp0_full_smoke/
  scene_proxy_points.npz
  scene_proxy_points.ply
  scene_proxy_summary.json

output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54/contact_eval/smoke/
  contact_eval_summary.json
  per_frame_contact_metrics.csv
  contact_anchor_trajectory.ply
  scene_proxy_sample.ply
  overlays/*.png
  renders/*_gt.png
  renders/*_render.png
  contact_rois/*/*_gt.png
  contact_rois/*/*_render.png
  contact_rois/*/*_absdiff.png

output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54/anchor_probe_diagnostics/smoke/
  anchor_probe_summary.json
  anchor_probe_metrics.csv
  anchor_probe_scene_samples.ply
  overlays/*.png
  renders/*_gt.png
  renders/*_render.png
```

Smoke 观察：

```text
contact_eval smoke 使用 COLMAP sparse proxy，feet anchors 到 proxy 的 nearest distance 均值约 0.8787m，2/5/10cm contact recall 均为 0。
anchor_probe smoke 中 attention entropy 接近 log(16)，说明当前随机初始化 attention 基本接近均匀分布；这符合 dry-run 预期，不代表已经学到接触关系。
```

重要结论：当前 COLMAP sparse proxy 对接触评估过稀、尺度/覆盖不一定适合作为最终 contact surface，只能作为 Stage A 链路验证。后续应优先接入 depth fusion / ground proxy / 更干净 scene proxy，再做正式 contact 指标解释。

### 22.12 DECO 接触评估接入记录（2026-05-20）

已将 lab 序列的 DECO 输出接入 Stage A-C 的接触评估通道。DECO 输出位置：

```text
/hdd/u202420081000003/deco/demo_out_lab
```

读取结果确认：

```text
Preds/<frame>/pred.obj
Preds/<frame>/<frame>.png

pred.obj 是 6890 顶点 SMPL mesh。
灰色 [130, 130, 130] 表示非接触。
绿色 [0, 255, 0] 表示 DECO inference 中 cont >= 0.5 的接触顶点。
当前目录没有额外 npy / npz / json 概率文件，因此本次使用 OBJ 顶点颜色恢复二值 contact label。
```

代码落实：

```text
scripts/contact_eval_lib.py
  read_deco_pred_obj()
  deco_contact_obj_path()
  deco_contact_png_path()
  aggregate_deco_contact_to_anchors()

scripts/evaluate_contact_quality.py
  新增 --contact-source deco-anchor
  新增 --deco-dir
  新增 --deco-contact-threshold
  新增 --deco-top-anchors
  新增 --deco-anchor-verts
```

当前映射方式：

```text
1. 从 DECO pred.obj 读取 6890 个 SMPL 顶点的接触二值标签。
2. 使用 hugs.utils.anchor_utils 在原始 SMPL t-pose 上生成同一套 16 个语义 anchor 的 vertex id。
3. 对每个 anchor 的 64 个 SMPL 顶点求 mean(contact)，得到 deco_contact_prob。
4. deco_contact_prob >= threshold 的 anchor 被视为本帧 DECO contact anchor。
5. HUGS 侧仍使用当前训练结果中的 anchor_world 位置，计算投影 overlay、ROI render/GT/absdiff、到 scene proxy 的 nearest/median/trimmed distance。
```

注意：DECO 只负责回答“人体哪个区域可能接触”。它不直接提供场景几何，也不直接评价 HUGS scene Gaussian 是否真实接触。因此本方法比 feet-only 更合理，但最终接触质量仍依赖一个可信 scene proxy / depth surface / ground-object proxy。

Smoke 命令：

```bash
python scripts/evaluate_contact_quality.py \
  -o output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54 \
  --proxy-npz output/contact_eval_proxy/neuman_lab_exp0_full_smoke/scene_proxy_points.npz \
  --tag deco_contact_smoke_20260520 \
  --contact-source deco-anchor \
  --deco-dir /hdd/u202420081000003/deco/demo_out_lab \
  --max-frames 8 \
  --frame-stride 10 \
  --save-every 1 \
  --proxy-max-points 50000 \
  --deco-top-anchors 6
```

完整 103 帧评估命令：

```bash
python scripts/evaluate_contact_quality.py \
  -o output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54 \
  --proxy-npz output/contact_eval_proxy/neuman_lab_exp0_full_smoke/scene_proxy_points.npz \
  --tag deco_contact_full_20260520 \
  --contact-source deco-anchor \
  --deco-dir /hdd/u202420081000003/deco/demo_out_lab \
  --save-every 10 \
  --proxy-max-points 50000 \
  --deco-top-anchors 6
```

完整评估输出：

```text
output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54/contact_eval/deco_contact_full_20260520/
  contact_eval_summary.json
  per_frame_contact_metrics.csv
  contact_anchor_trajectory.ply
  scene_proxy_sample.ply
  overlays/*.png
  renders/*_gt.png
  renders/*_render.png
  deco_preds/*.png
  contact_rois/*/*_gt.png
  contact_rois/*/*_render.png
  contact_rois/*/*_absdiff.png
```

完整 103 帧结果摘要：

```text
num_rows: 1648
num_contact_rows: 642
missing_deco_frames: 0
DECO vertex count: 每帧 6890

DECO contact anchor 分布：
  left_sole: 103
  right_sole: 103
  left_toe: 103
  right_toe: 103
  left_heel: 103
  right_heel: 103
  left_fingers: 12
  left_palm: 9
  right_palm: 1
  right_fingers: 1
  back: 1

contact_deco_contact_prob mean: 0.7829
contact nearest proxy distance mean: 0.8281m
contact nearest proxy distance median: 0.7226m
contact recall @ 2cm: 0.0000
contact recall @ 5cm: 0.0016
contact recall @ 10cm: 0.0031
contact ROI PSNR mean: 26.5095
contact ROI SSIM mean: 0.8746
```

阶段性结论：

```text
1. DECO 接入链路可用，且语义结果合理：主要稳定预测脚底、脚趾、脚跟接触，少量帧预测手/背接触。
2. DECO-contact 比固定 feet-only 更适合作为人体接触区域选择器，后续建议默认采用 --contact-source deco-anchor。
3. 当前到 COLMAP sparse proxy 的接触距离仍很大，5cm/10cm recall 几乎为 0。这不应直接解释为 HUGS 接触完全失败，更可能说明当前 scene proxy 对接触表面过稀、过乱、覆盖不足。
4. 下一步应把 improved initialization / depth fusion / cleaned scene proxy 接进同一脚本，用同一 DECO contact anchor 集合做 A/B 对比。若 improved proxy 后 contact distance 明显下降，同时 ROI 渲染不下降，才可以作为接触质量提升证据。
```

### 22.13 Overlay 投影 bug 修复记录（2026-05-21）

发现问题：`contact_eval/*/overlays/*.png` 中彩色 anchor 点明显落在人体头顶/上半身附近，而 DECO 预测的是脚底、脚趾、脚跟接触区域。

排查结论：

```text
不是 anchor 绑定主问题，而是 scripts/contact_eval_lib.py::project_points() 的 3D->2D 投影 y 轴方向错了。
```

诊断方法：同一帧 frame 00000，对比旧 `full_proj_transform` NDC 投影和 HUGS 内部使用的 camera intrinsics 投影：

```text
image height: 717
left_toe old_xy: [911.06, 43.14]
left_toe K_xy:   [911.06, 673.86]
old_y_flip:      673.86

left_sole old_xy: [909.21, 98.30]
left_sole K_xy:   [909.21, 618.70]
old_y_flip:       618.70
```

这说明旧 overlay 的 y 坐标正好被上下翻转，所以脚部 anchor 被画到了头顶附近。

修复内容：

```text
scripts/contact_eval_lib.py
  project_points()
    现在优先使用 data['world_view_transform'] + data['cam_intrinsics']：
      cam = [xyz, 1] @ world_view_transform
      u = fx * cam_x / cam_z + cx
      v = fy * cam_y / cam_z + cy

    这与 hugs/trainer/gs_trainer.py 中 depth prior / pruning 的投影写法保持一致。
```

影响范围：

```text
受影响：
  overlays/*.png 中的彩色点位置
  contact_rois/* 的 crop 位置
  ROI PSNR / ROI SSIM
  使用 project_points() 的 probe/attention overlay 可视化

不受影响或基本不受影响：
  DECO contact anchor 选择
  anchor 3D world position
  anchor 到 scene proxy 的 3D nearest/median/trimmed distance
  contact recall @ distance threshold
```

修复后 smoke 输出：

```text
output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54/contact_eval/deco_contact_projection_fix_smoke_20260521/
```

修复后完整 103 帧输出：

```text
output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54/contact_eval/deco_contact_projection_fix_full_20260521/
```

修复后 frame 00000 的脚部 anchor 投影示例：

```text
left_sole:  x=909.2, y=618.7
right_sole: x=829.4, y=607.2
left_toe:   x=911.1, y=673.9
right_toe:  x=808.7, y=645.4
left_heel:  x=909.8, y=616.9
right_heel: x=842.3, y=599.7
```

修复后完整 103 帧摘要：

```text
num_rows: 1648
num_contact_rows: 642
missing_deco_frames: 0

contact nearest proxy distance mean: 0.8281m
contact nearest proxy distance median: 0.7226m
contact recall @ 2cm: 0.0000
contact recall @ 5cm: 0.0016
contact recall @ 10cm: 0.0031

contact ROI PSNR mean: 17.8720
contact ROI SSIM mean: 0.7352
```

重要记录：旧目录 `deco_contact_full_20260520` 的 overlay 和 ROI 指标作废；后续引用 DECO-contact 评估时，应使用 `deco_contact_projection_fix_full_20260521` 或更新后的重新评估结果。

### 22.14 ExpE-style anchor overlay 复现与 contact overlay 修正（2026-05-21）

根据复查，`deco_contact_projection_fix_*` 中点位仍然和 ExpE 结果有差异，不是新的投影方向问题，而是 anchor 位置定义不一致。

已新增对比脚本：

```text
scripts/export_anchor_overlay_compare.py
```

导出命令示例：

```bash
python scripts/export_anchor_overlay_compare.py \
  -o output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54 \
  --frames 0,10,20,30 \
  --render-pose dataset \
  --semantic-pose dataset
```

输出：

```text
output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54/anchor_overlay_compare/expe_compare_dataset_render_dataset_semantic/
  overlays/*.png
  anchor_overlay_compare.csv
```

图中含义：

```text
green filled point: ExpE-style semantic anchor center
magenta circle: binding-center anchor currently used by older contact overlay
white line: semantic center -> binding center offset
```

frame 00000 对比：

```text
left_sole  semantic [900.7, 667.0], binding [909.2, 618.7], delta [ 8.5, -48.3]
right_sole semantic [818.2, 644.0], binding [829.4, 607.2], delta [11.3, -36.8]
left_toe   semantic [909.1, 667.6], binding [911.1, 673.9], delta [ 2.0,   6.3]
right_toe  semantic [805.1, 644.0], binding [808.7, 645.4], delta [ 3.6,   1.4]
left_heel  semantic [914.8, 635.8], binding [909.8, 616.9], delta [-5.0, -18.9]
right_heel semantic [847.7, 617.6], binding [842.3, 599.7], delta [-5.4, -17.9]
```

解释：toe 区域紧凑，所以 binding center 与 semantic center 接近；sole/heel 的绑定高斯数量很多，且包含鞋面/脚踝附近高斯，binding center 被拉离真正脚底/脚跟语义位置。因此旧 contact overlay 中 sole/heel 会轻微偏上。

已修改：

```text
scripts/contact_eval_lib.py
  新增 posed_semantic_anchor_world()

scripts/evaluate_contact_quality.py
  新增 --overlay-anchor-position semantic|binding
  默认 semantic
```

现在 contact overlay / ROI crop 默认使用 ExpE-style semantic anchor center；CSV 中仍保留 binding center 的 `anchor_x/y/z`，并新增 overlay anchor 的 `overlay_anchor_x/y/z`。

新的 smoke 输出：

```text
output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54/contact_eval/deco_contact_semantic_overlay_smoke_20260521/
```

frame 00000 semantic overlay 坐标：

```text
left_sole  [900.7, 667.0]
right_sole [818.2, 644.0]
left_toe   [909.1, 667.6]
right_toe  [805.1, 644.0]
left_heel  [914.8, 635.8]
right_heel [847.7, 617.6]
```

额外发现：当前 `exp0_hugs_original_15000` 的 `human_final.pth` 中没有保存 `global_orient/body_pose/transl` 这类优化 pose 参数；因此 optimized render 分支会回退到 dataset pose。这个 original baseline 和旧 ExpE normal 训练输出不是完全同一个 pose 设置来源。

### 22.15 Render-loss-only anchor-attention correction 完整流程实验（2026-05-21）

目标：在不引入显式接触监督的前提下，先验证一个完整可训练 pipeline：

```text
HUGS original training / optimization
-> anchor token encoding
-> scene Gaussian token query and aggregation
-> cross attention
-> human Gaussian parameter correction
-> render loss backprop
```

重要原则：这里的 attention 不作为接触质量估计器，也不作为独立接触监督；它只作为人体 anchor 与场景高斯之间的上下文聚合模块，后续通过 correction 网络隐式影响人体高斯参数，并由图像渲染损失优化。

已实现代码：

```text
hugs/models/anchor_attention.py
  AnchorSceneAttentionBaseline
    - anchor token encoder
    - scene KNN/query token encoder
    - cross-attention context aggregation
    - delta_mu correction head
    - optional delta_opacity correction head
    - module_start_iter: correction 阶段前完全跳过 attention 计算
    - correction_start_iter: correction 生效起点
    - correction_warmup_iters: correction gamma 线性 warmup
    - zero_init_delta: correction head 最后一层零初始化，保证刚开启时接近 HUGS 原行为

hugs/cfg/config.py
  cfg.anchor_attention.module_start_iter
  cfg.anchor_attention.correction_start_iter
  cfg.anchor_attention.correction_warmup_iters
  cfg.anchor_attention.zero_init_delta

hugs/trainer/gs_trainer.py
  maybe_apply_anchor_attention()
  anchor_attention optimizer step
  anchor_attention debug / checkpoint save
```

本轮配置：

```text
cfg_files/release/neuman/hugs_anchor_attention_correction_smoke.yaml
cfg_files/release/neuman/hugs_anchor_attention_correction_15000.yaml
```

关键策略：

```text
前 12000 步：保持原 HUGS 训练流程，不执行 attention/correction
后 3000 步：打开 anchor-attention correction
总训练步数：约 15000 步，与原版 HUGS baseline 对齐
correction head: zero-init
correction warmup: 1000 steps
gamma_mu: 0.005
delta_loss_w: 0.001
correct_opacity: false
anim_interval: -1
```

`anim_interval=-1` 是因为当前环境缺少默认 AMASS/SFU 动画文件；该设置只跳过动画导出，不改变训练集、验证集、HUGS 优化项或渲染质量评估。

先运行 smoke 验证：

```bash
python main.py --cfg_file cfg_files/release/neuman/hugs_anchor_attention_correction_smoke.yaml
```

smoke 输出：

```text
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_smoke_20260521/2026-05-21_02-36-51/
```

smoke 结果：

```text
final HUGS_PSNR:       22.9653
final HUGS_SSIM:        0.8276
final HUGS_LPIPS:       0.1755
final HUMAN_PSNR:      18.6924
final HUMAN_SSIM:       0.7132
final HUMAN_LPIPS:      0.2144
```

smoke 检查结论：

```text
1. anchor-attention 模块可正常初始化。
2. correction 开启后无 NaN / shape error / optimizer error。
3. l_anchor_delta 从 0 逐步增长，说明 zero-init + warmup 后 correction 路径参与了优化。
4. 输出了 train/val/render_all/canonical 视频和图像。
5. anchor_attention_debug 在 1200 和 1600 步正常保存 csv/pt。
```

smoke 关键可视化：

```text
train/000000.png
train/001000.png
val/full_final_*.png
val/human_final_*.png
render_all/*.png
render_all_neuman_lab_final.mp4
canon_neuman_lab_final_a_pose.mp4
canon_neuman_lab_final_da_pose.mp4
anchor_attention_debug/anchor_attention_iter001200.csv
anchor_attention_debug/anchor_attention_iter001600.csv
```

完整实验命令：

```bash
python main.py --cfg_file cfg_files/release/neuman/hugs_anchor_attention_correction_15000.yaml
```

完整实验输出目录：

```text
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_15000_20260521/2026-05-21_02-45-33/
```

完整实验已完成。

```text
final HUGS_PSNR:       26.0527
final HUGS_SSIM:        0.9149
final HUGS_LPIPS:       0.0705
final HUMAN_PSNR:      18.9251
final HUMAN_SSIM:       0.7609
final HUMAN_LPIPS:      0.1491
```

与 Exp0 原版 HUGS 15000 baseline 对比：

```text
Exp0 HUGS original 15000:
  HUGS_PSNR:       25.9582
  HUGS_SSIM:        0.9147
  HUGS_LPIPS:       0.0710
  HUMAN_PSNR:      18.8005
  HUMAN_SSIM:       0.7580
  HUMAN_LPIPS:      0.1532

Anchor-attention correction 15000:
  HUGS_PSNR:       26.0527   (+0.0945)
  HUGS_SSIM:        0.9149   (+0.0002)
  HUGS_LPIPS:       0.0705   (-0.0005, better)
  HUMAN_PSNR:      18.9251   (+0.1246)
  HUMAN_SSIM:       0.7609   (+0.0029)
  HUMAN_LPIPS:      0.1491   (-0.0041, better)
```

关键中间节点：

```text
012000, correction 刚开启:
  HUGS_PSNR: 25.9258
  HUGS_SSIM: 0.9127
  HUGS_LPIPS: 0.0719
  HUMAN_PSNR: 18.7633
  HUMAN_SSIM: 0.7488
  HUMAN_LPIPS: 0.1537

014000, correction 已进入稳定阶段:
  HUGS_PSNR: 26.0629
  HUGS_SSIM: 0.9135
  HUGS_LPIPS: 0.0724
  HUMAN_PSNR: 18.9984
  HUMAN_SSIM: 0.7631
  HUMAN_LPIPS: 0.1502
```

完整实验关键输出：

```text
results:
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_15000_20260521/2026-05-21_02-45-33/results_train.json

train progress:
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_15000_20260521/2026-05-21_02-45-33/train/*.png
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_15000_20260521/2026-05-21_02-45-33/train_neuman_lab.mp4

validation images:
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_15000_20260521/2026-05-21_02-45-33/val/full_final_*.png
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_15000_20260521/2026-05-21_02-45-33/val/human_final_*.png

full sequence render:
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_15000_20260521/2026-05-21_02-45-33/render_all/*.png
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_15000_20260521/2026-05-21_02-45-33/render_all_neuman_lab_final.mp4

canonical videos:
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_15000_20260521/2026-05-21_02-45-33/canon_neuman_lab_final_a_pose.mp4
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_15000_20260521/2026-05-21_02-45-33/canon_neuman_lab_final_da_pose.mp4

anchor attention debug:
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_15000_20260521/2026-05-21_02-45-33/anchor_attention_debug/anchor_attention_iter012000.csv
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_15000_20260521/2026-05-21_02-45-33/anchor_attention_debug/anchor_attention_iter013000.csv
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_15000_20260521/2026-05-21_02-45-33/anchor_attention_debug/anchor_attention_iter014000.csv
```

实验结论：

```text
1. 该 render-loss-only anchor-attention correction pipeline 已经可以完整训练到 15000 步。
2. 前 12000 步行为接近原版 HUGS；12000 步后 attention/correction 路径打开。
3. 在 lab 验证集上，final 渲染指标相对 Exp0 原版 HUGS 有小幅但一致提升，尤其 human crop 的 LPIPS/PSNR/SSIM 均变好。
4. 当前提升幅度不大，但足以说明“token encoding + attention + correction + render loss”的完整优化链路是可行的。
5. 这不是接触质量指标，也不能证明接触一定变好；它只说明该模块可以在不破坏 HUGS 主流程的情况下通过图像损失获得收益。
6. 性能瓶颈明显在 correction 阶段：每步需要在约 2.1M scene Gaussians 上做场景候选查询/聚合，后续需要考虑候选缓存、局部 scene subset 或更强采样策略。
7. 后续替换初始化方案时，应保持该配置作为对照，只替换初始化输入并复用同一套 12000/15000 训练与评估流程。
```

### 22.16 Bike 序列原版 HUGS 与 anchor-attention correction 对照（2026-05-21）

为检查 `lab` 上的小幅提升是否能迁移到不同场景，按相同策略在 NeuMan `bike` 序列上顺序跑了两组 15000-step 实验：

```text
Original HUGS config:
cfg_files/release/neuman/hugs_human_scene_bike_original_15000.yaml

Anchor-attention correction config:
cfg_files/release/neuman/hugs_anchor_attention_correction_15000_bike.yaml
```

原版 HUGS 输出：

```text
output/human_scene/neuman/bike/hugs_trimlp/exp0_hugs_original_15000_bike_20260521/2026-05-21_12-32-06/
```

anchor-attention correction 输出：

```text
output/human_scene/neuman/bike/hugs_trimlp/anchor_attention_correction_15000_bike_20260521/2026-05-21_13-17-51/
```

关键配置保持一致：

```text
train.num_steps: 14998
train.val_interval: 1000
train.save_progress_images: true
train.progress_save_interval: 1000
train.anim_interval: -1
```

attention/correction 配置：

```text
module_start_iter: 12000
correction_start_iter: 12000
correction_warmup_iters: 1000
zero_init_delta: true
gamma_mu: 0.005
delta_loss_w: 0.001
```

6000-step 对照：

```text
Original HUGS 6000:
  HUGS_PSNR:       24.6140
  HUGS_SSIM:        0.7919
  HUGS_LPIPS:       0.1555
  HUMAN_PSNR:      19.6295
  HUMAN_SSIM:       0.6551
  HUMAN_LPIPS:      0.1933

Anchor-attention run 6000:
  HUGS_PSNR:       24.6360
  HUGS_SSIM:        0.7919
  HUGS_LPIPS:       0.1546
  HUMAN_PSNR:      19.6729
  HUMAN_SSIM:       0.6491
  HUMAN_LPIPS:      0.1907
```

由于 attention/correction 在 12000 步后才开启，6000 步两者接近，符合预期。

anchor-attention correction 关键节点：

```text
012000, correction 刚开启:
  HUGS_PSNR:       25.4867
  HUGS_SSIM:        0.8326
  HUGS_LPIPS:       0.1143
  HUMAN_PSNR:      19.7824
  HUMAN_SSIM:       0.6648
  HUMAN_LPIPS:      0.1601

014000:
  HUGS_PSNR:       25.6300
  HUGS_SSIM:        0.8388
  HUGS_LPIPS:       0.1072
  HUMAN_PSNR:      19.7889
  HUMAN_SSIM:       0.6719
  HUMAN_LPIPS:      0.1542
```

final 对照：

```text
Original HUGS final:
  HUGS_PSNR:       25.6923
  HUGS_SSIM:        0.8419
  HUGS_LPIPS:       0.1027
  HUMAN_PSNR:      19.7592
  HUMAN_SSIM:       0.6765
  HUMAN_LPIPS:      0.1520

Anchor-attention correction final:
  HUGS_PSNR:       25.7943   (+0.1020)
  HUGS_SSIM:        0.8411   (-0.0008)
  HUGS_LPIPS:       0.1031   (+0.0004, worse)
  HUMAN_PSNR:      20.0542   (+0.2950)
  HUMAN_SSIM:       0.6800   (+0.0035)
  HUMAN_LPIPS:      0.1483   (-0.0037, better)
```

完整输出文件：

```text
Original HUGS:
output/human_scene/neuman/bike/hugs_trimlp/exp0_hugs_original_15000_bike_20260521/2026-05-21_12-32-06/results_train.json
output/human_scene/neuman/bike/hugs_trimlp/exp0_hugs_original_15000_bike_20260521/2026-05-21_12-32-06/train_neuman_bike.mp4
output/human_scene/neuman/bike/hugs_trimlp/exp0_hugs_original_15000_bike_20260521/2026-05-21_12-32-06/render_all_neuman_bike_final.mp4
output/human_scene/neuman/bike/hugs_trimlp/exp0_hugs_original_15000_bike_20260521/2026-05-21_12-32-06/canon_neuman_bike_final_a_pose.mp4
output/human_scene/neuman/bike/hugs_trimlp/exp0_hugs_original_15000_bike_20260521/2026-05-21_12-32-06/canon_neuman_bike_final_da_pose.mp4

Anchor-attention correction:
output/human_scene/neuman/bike/hugs_trimlp/anchor_attention_correction_15000_bike_20260521/2026-05-21_13-17-51/results_train.json
output/human_scene/neuman/bike/hugs_trimlp/anchor_attention_correction_15000_bike_20260521/2026-05-21_13-17-51/train_neuman_bike.mp4
output/human_scene/neuman/bike/hugs_trimlp/anchor_attention_correction_15000_bike_20260521/2026-05-21_13-17-51/render_all_neuman_bike_final.mp4
output/human_scene/neuman/bike/hugs_trimlp/anchor_attention_correction_15000_bike_20260521/2026-05-21_13-17-51/canon_neuman_bike_final_a_pose.mp4
output/human_scene/neuman/bike/hugs_trimlp/anchor_attention_correction_15000_bike_20260521/2026-05-21_13-17-51/canon_neuman_bike_final_da_pose.mp4
output/human_scene/neuman/bike/hugs_trimlp/anchor_attention_correction_15000_bike_20260521/2026-05-21_13-17-51/ckpt/anchor_attention_final.pth
```

运行观察：

```text
1. 该方法在 bike 上也能完整跑到 final，并保存 human / scene / anchor_attention final checkpoint。
2. 12000 步后 l_anchor_delta 从 0 开始，warmup 期间逐步增长，后段大多在 0.05-0.18 左右波动，未出现 NaN 或发散。
3. bike 上 final 整体 PSNR 和 human crop 指标变好，但整体 SSIM 和 LPIPS 基本持平或轻微变差。
4. 这说明当前方法的收益不是纯 lab 特例，但提升仍然小，不能只凭单次 bike 结果下强结论。
5. correction 阶段耗时明显增加：12000 前约 4.7-5.1 it/s，12000 后约 1.6-2.0 it/s；后续需要优化 scene candidate 查询或缓存。
```

### 22.17 Lab/Bike 质量实验指标总表（2026-05-21）

本节把当前质量实验中最关键的 `lab` 与 `bike` 指标集中到一张表里，便于横向比较。指标均为 validation final；`HUGS_*` 是 full image 指标，`HUMAN_*` 是 human crop 指标，LPIPS 越低越好。

| 场景   | 实验                             |  seed | 输出目录                                                                   | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUMAN_PSNR | HUMAN_SSIM | HUMAN_LPIPS |
| ---- | ------------------------------ | ----: | ---------------------------------------------------------------------- | --------: | --------: | ---------: | ---------: | ---------: | ----------: |
| lab  | 原版 HUGS                        |     0 | `exp0_hugs_original_15000_20260519/2026-05-19_23-26-54`                |   25.9582 |    0.9147 |     0.0710 |    18.8005 |     0.7580 |      0.1532 |
| lab  | anchor-attention correction    |     0 | `anchor_attention_correction_15000_20260521/2026-05-21_02-45-33`       |   26.0527 |    0.9149 |     0.0705 |    18.9251 |     0.7609 |      0.1491 |
| lab  | anchor-attention correction    |     1 | `anchor_attention_correction_15000_seed1_20260521/2026-05-21_09-22-56` |   26.0068 |    0.9140 |     0.0724 |    18.8995 |     0.7611 |      0.1510 |
| lab  | anchor-attention correction    |     2 | `anchor_attention_correction_15000_seed2_20260521/2026-05-21_10-34-44` |   26.1772 |    0.9165 |     0.0707 |    19.0855 |     0.7654 |      0.1477 |
| lab  | anchor-attention correction 平均 | 0/1/2 | 三次 seed 实验均值                                                           |   26.0789 |    0.9151 |     0.0712 |    18.9700 |     0.7625 |      0.1492 |
| bike | 原版 HUGS                        |     0 | `exp0_hugs_original_15000_bike_20260521/2026-05-21_12-32-06`           |   25.6923 |    0.8419 |     0.1027 |    19.7592 |     0.6765 |      0.1520 |
| bike | anchor-attention correction    |     0 | `anchor_attention_correction_15000_bike_20260521/2026-05-21_13-17-51`  |   25.7943 |    0.8411 |     0.1031 |    20.0542 |     0.6800 |      0.1483 |

相对同场景原版 HUGS 的变化：

判定规则：PSNR/SSIM 越高越好；LPIPS 越低越好。表中 **粗体** 表示相对原版 HUGS 更好，*斜体* 表示相对原版 HUGS 更差。

| 场景   | 实验                             |  seed |  ΔHUGS_PSNR |  ΔHUGS_SSIM | ΔHUGS_LPIPS | ΔHUMAN_PSNR | ΔHUMAN_SSIM | ΔHUMAN_LPIPS |
| ---- | ------------------------------ | ----: | ----------: | ----------: | ----------: | ----------: | ----------: | -----------: |
| lab  | anchor-attention correction    |     0 | **+0.0945** | **+0.0002** | **-0.0006** | **+0.1246** | **+0.0029** |  **-0.0042** |
| lab  | anchor-attention correction    |     1 | **+0.0486** |   *-0.0007* |   *+0.0013* | **+0.0990** | **+0.0031** |  **-0.0023** |
| lab  | anchor-attention correction    |     2 | **+0.2190** | **+0.0018** | **-0.0003** | **+0.2850** | **+0.0074** |  **-0.0055** |
| lab  | anchor-attention correction 平均 | 0/1/2 | **+0.1207** | **+0.0004** |   *+0.0001* | **+0.1695** | **+0.0045** |  **-0.0040** |
| bike | anchor-attention correction    |     0 | **+0.1020** |   *-0.0008* |   *+0.0004* | **+0.2950** | **+0.0035** |  **-0.0037** |

指标来源：

```text
lab original:
output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54/results_train.json

lab seed0:
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_15000_20260521/2026-05-21_02-45-33/results_train.json

lab seed1:
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_15000_seed1_20260521/2026-05-21_09-22-56/eval_unified_final.json

lab seed2:
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_15000_seed2_20260521/2026-05-21_10-34-44/results_train.json

bike original:
output/human_scene/neuman/bike/hugs_trimlp/exp0_hugs_original_15000_bike_20260521/2026-05-21_12-32-06/results_train.json

bike correction:
output/human_scene/neuman/bike/hugs_trimlp/anchor_attention_correction_15000_bike_20260521/2026-05-21_13-17-51/results_train.json
```

简要观察：

```text
1. lab 三个 seed 的 human crop 指标均优于原版 HUGS，平均 HUMAN_PSNR +0.1695、HUMAN_SSIM +0.0045、HUMAN_LPIPS -0.0040。
2. lab full image 指标整体小幅正向，但 seed1 的 HUGS_SSIM 和 HUGS_LPIPS 略差于原版，说明 full image 收益仍有随机性。
3. bike 单次实验中 human crop 三项均改善，full image PSNR 改善，但 SSIM/LPIPS 基本持平或轻微变差。
4. 当前结果支持“anchor-attention correction 对人体区域更稳定有益”，但全图收益仍需要更多场景和 seed 验证。
```

### 22.18 Correction start time 消融：12000 / 9000 / 6000（2026-05-21）

本节记录一次针对 anchor-attention correction 介入时机的消融。目标是回答：当前网络 correction 是否应该只在 HUGS 基本收敛后做后段微调，还是可以更早参与优化。

实验设置：

```text
共同设置：dataset=lab, total schedule=15000, 原版 HUGS 初始化与训练流程不变。
只改变 anchor_attention.module_start_iter / correction_start_iter：
A. start12000：12000 -> 15000，已有主实验配置。
B. start9000：9000 -> 15000，本次完整跑完。
C. start6000：6000 -> 15000，本次跑到约 11200 后根据劣化趋势手动停止。

保持相同的 correction 超参：
correction_warmup_iters=1000
gamma_mu=0.005
delta_loss_w=0.001
zero_init_delta=True
```

配置与输出目录：

```text
A start12000 config:
cfg_files/release/neuman/hugs_anchor_attention_correction_15000.yaml
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_15000_20260521/2026-05-21_02-45-33/

B start9000 config:
cfg_files/release/neuman/hugs_anchor_attention_correction_start9000_15000.yaml
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_start9000_15000_20260521/2026-05-21_17-13-07/

C start6000 config:
cfg_files/release/neuman/hugs_anchor_attention_correction_start6000_15000.yaml
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_start6000_15000_20260521/2026-05-21_18-37-06/
```

完整 final 对照：

| 实验 | correction start | 是否完整跑完 | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUMAN_PSNR | HUMAN_SSIM | HUMAN_LPIPS |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 原版 HUGS | none | 是 | 25.9582 | 0.9147 | 0.0710 | 18.8005 | 0.7580 | 0.1532 |
| A anchor-attention | 12000 | 是 | 26.0527 | 0.9149 | 0.0705 | 18.9251 | 0.7609 | 0.1491 |
| B anchor-attention | 9000 | 是 | 26.0416 | 0.9132 | 0.0721 | 18.9643 | 0.7576 | 0.1512 |
| C anchor-attention | 6000 | 否，约 11200 停止 | - | - | - | - | - | - |

关键中间验证点：

| 实验 | iter | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUMAN_PSNR | HUMAN_SSIM | HUMAN_LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| A start12000 | 12000 | 25.9258 | 0.9127 | 0.0719 | 18.7633 | 0.7488 | 0.1537 |
| A start12000 | 14000 | 26.0629 | 0.9135 | 0.0724 | 18.9984 | 0.7631 | 0.1502 |
| B start9000 | 9000 | 26.0062 | 0.9082 | 0.0767 | 19.0329 | 0.7488 | 0.1558 |
| B start9000 | 12000 | 26.0707 | 0.9124 | 0.0722 | 19.0377 | 0.7510 | 0.1488 |
| B start9000 | 14000 | 26.2107 | 0.9135 | 0.0728 | 19.2379 | 0.7656 | 0.1505 |
| C start6000 | 6000 | 25.7734 | 0.9022 | 0.0842 | 18.9807 | 0.7451 | 0.1643 |
| C start6000 | 8000 | 26.0296 | 0.9111 | 0.0758 | 19.0639 | 0.7614 | 0.1527 |
| C start6000 | 9000 | 25.0503 | 0.9027 | 0.0835 | 17.5841 | 0.6917 | 0.2126 |
| C start6000 | 10000 | 26.0590 | 0.9124 | 0.0737 | 18.9955 | 0.7591 | 0.1536 |
| C start6000 | 11000 | 26.0409 | 0.9136 | 0.0730 | 18.9277 | 0.7602 | 0.1524 |

运行观察：

```text
1. start12000 仍是当前最稳的默认选择：final 的 full image 与 human crop 指标都优于原版 HUGS，且 correction 只在后 3000 步介入，扰动较小。
2. start9000 可以完整跑完，中途 14000 的 PSNR/HUMAN_PSNR 较高，但 final 回落；最终 full SSIM、LPIPS、human SSIM、human LPIPS 都不如 start12000，说明提前到 9000 并没有形成稳定收益。
3. start6000 在 8000 时看起来还正常，但 9000 验证出现明显劣化：HUMAN_PSNR 从 19.0639 掉到 17.5841，HUMAN_LPIPS 从 0.1527 变成 0.2126。虽然 10000/11000 有恢复，但训练中 l_anchor_delta 后续多次接近或超过 1.0，说明过早 correction 会造成较大不稳定。
4. start6000 没有 final 结果：本次按用户指令在约 11200 停止，因此没有 results_train.json、final ckpt 或 final video。该结果只作为“早介入不稳定”的失败/负向消融证据。
5. 当前建议：保留 start12000 作为默认 baseline；如果后续要探索更早介入，应先加强 correction 约束，例如更小 gamma_mu、更长 warmup、更小 delta_loss_w 或分阶段冻结/解冻，而不是直接从 6000 开始。
```

结论：

```text
本次消融说明，当前 anchor-attention correction 网络不是越早介入越好。
在没有额外接触监督、几何初始化仍不理想的情况下，过早让 correction 参与会放大人体/场景未收敛阶段的噪声，导致人体区域指标显著波动。
因此当前可行 baseline 应采用“原版 HUGS 先优化到较稳定状态，再在最后 3000 步做小幅 correction”的策略，即 start12000。
```

### 22.19 Attention 延长训练消融：从 15000 checkpoint 继续 +3000（2026-05-21）

本节记录一次针对“后 3000 步 attention/correction 是否足够”的验证。用户的问题是：如果保持 HUGS 初始优化阶段不变，继续增加 attention/correction 训练步数，指标是否还能提升；以及是否必须从头训练。

实现方式：

```text
本次不是严格意义上的 HUGS resume。
当前代码可以加载 human / scene / anchor_attention 的 final 权重，但训练循环的 global iteration 会重新从 0 开始，
optimizer state 与原训练日程也没有完整连续继承。因此这次实验应理解为：

从当前最好的 start12000 final checkpoint 出发，
冻结/弱化常规高斯学习率，关闭 densification，
让 anchor-attention correction 从 local step 0 继续以小学习率微调 3000 步。
```

配置与输出目录：

```text
config:
cfg_files/release/neuman/hugs_anchor_attention_extend_from15000_plus3000.yaml

source checkpoint:
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_15000_20260521/2026-05-21_02-45-33/ckpt/

output:
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_extend_from15000_plus3000_20260521/2026-05-21_19-49-01/
```

主要设置：

```text
train.num_steps=2998
anchor_attention.module_start_iter=0
anchor_attention.correction_start_iter=0
anchor_attention.correction_warmup_iters=0
anchor_attention.lr=5e-5
anchor_attention.gamma_mu=0.003
anchor_attention.delta_loss_w=0.002
human.densify_until_iter=0
scene.densify_until_iter=0

human / scene 的 position、opacity、scaling、rotation、feature 等学习率均做了明显下调，
目的是避免继续训练阶段破坏已经收敛的 HUGS 表示。
```

指标对比：

| 实验                  |          阶段 | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUMAN_PSNR | HUMAN_SSIM | HUMAN_LPIPS |
| ------------------- | ----------: | --------: | --------: | ---------: | ---------: | ---------: | ----------: |
| start12000 baseline | final 15000 |   26.0527 |    0.9149 |     0.0705 |    18.9251 |     0.7609 |      0.1491 |
| extend + attention  |       +1000 |   26.3764 |    0.9203 |     0.0652 |    19.2497 |     0.7711 |      0.1421 |
| extend + attention  |       +2000 |   26.3918 |    0.9208 |     0.0645 |    19.2469 |     0.7705 |      0.1413 |
| extend + attention  | +3000 final |   26.4029 |    0.9210 |     0.0640 |    19.2466 |     0.7701 |      0.1408 |
|                     |             |           |           |            |            |            |             |

可视化与结果文件：

```text
results_train.json
train_neuman_lab.mp4
render_all_neuman_lab_final.mp4
canon_neuman_lab_final_a_pose.mp4
canon_neuman_lab_final_da_pose.mp4
ckpt/human_final.pth
ckpt/scene_final.pth
ckpt/anchor_attention_final.pth
```

观察：

```text
1. 继续 +3000 后，full image 与 human crop 的 LPIPS/SSIM 都继续改善，PSNR 也明显高于原 start12000 final。
2. +1000 已经带来大部分收益，+2000/+3000 仍有小幅提升，说明原来的后 3000 步 attention/correction 未必完全饱和。
3. HUMAN_PSNR 在 +1000 后基本持平略降，但 HUMAN_LPIPS 持续下降，说明更长微调主要改善感知相似度和结构一致性，而不是单纯像素误差。
4. 本次结果只证明渲染指标层面继续微调有收益，不能直接证明接触质量改善；接触质量仍需要后续独立指标或 DECO/几何辅助评估。
5. 由于这不是严格 resume，若要作为正式论文/报告实验，建议后续补充两种更干净的流程：
   A. 在 12000 保存 checkpoint，然后原地继续训练到 18000；
   B. 给 trainer 增加 global_step 与 optimizer state 的完整 resume 支持。
```

结论：

```text
当前证据支持“attention/correction 网络在 3000 步内可能没有完全收敛”。
在不从头训练的情况下，可以从已有 final checkpoint 继续 fine-tune，作为快速消融是可行的；
但严格比较不同训练步数时，应该实现真正 resume 或从 12000 checkpoint 接续。

当前建议：
保留 start12000 作为稳定介入点；
将 attention/correction 总训练长度从 3000 扩展到 6000 作为下一版强 baseline 候选，
同时记录 +1000/+2000/+3000 checkpoint，用验证集指标选择最优停止点。
```

#### Lab 汇总表：HUGS 与 attention 延长训练对照

下表单独汇总 lab 数据集上的关键结果。PSNR/SSIM 越高越好，LPIPS 越低越好；每一列最优值用粗体标出。

| 实验设置                        | HUGS_PSNR ↑ | HUGS_SSIM ↑ | HUGS_LPIPS ↓ | HUMAN_PSNR ↑ | HUMAN_SSIM ↑ | HUMAN_LPIPS ↓ |
| --------------------------- | ----------: | ----------: | -----------: | -----------: | -----------: | ------------: |
| 12000 HUGS                  |     25.9258 |      0.9127 |       0.0719 |      18.7633 |       0.7488 |        0.1537 |
| 15000 HUGS                  |     25.9582 |      0.9147 |       0.0710 |      18.8005 |       0.7580 |        0.1532 |
| 12000 HUGS + 3000 attention |     26.0527 |      0.9149 |       0.0705 |      18.9251 |       0.7609 |        0.1491 |
| 12000 HUGS + 4000 attention |     26.3764 |      0.9203 |       0.0652 |  **19.2497** |   **0.7711** |        0.1421 |
| 12000 HUGS + 5000 attention |     26.3918 |      0.9208 |       0.0645 |      19.2469 |       0.7705 |        0.1413 |
| 12000 HUGS + 6000 attention | **26.4029** |  **0.9210** |   **0.0640** |      19.2466 |       0.7701 |    **0.1408** |

说明：

```text
1. 12000 HUGS 是 start12000 主实验中 attention/correction 开启前的验证点。
2. 15000 HUGS 是原版 HUGS 完整 15000 步 baseline。
3. 12000 HUGS + 3000 attention 是当前 start12000 anchor-attention baseline 的 final 结果。
4. 12000 HUGS + 4000/5000/6000 attention 是便于阅读的累计写法；实际后续 3000 步是从 12000 HUGS + 3000 attention 的 final checkpoint 加载权重后继续 fine-tune，不是严格无缝 resume。
5. full image 的最优指标集中在 12000 HUGS + 6000 attention；human crop 的 PSNR/SSIM 在 +4000 attention 最优，HUMAN_LPIPS 在 +6000 attention 最优。
```

### parkinglot / seattle 跨场景验证

实验时间：2026-05-22  
实验机器：`gpu-l40-1`  
远端总汇总目录：`gpu-l40-1:/tmp/ml-hugs-output/parkinglot_seattle_pipeline_20260522`

本轮实验用于验证当前 anchor-attention correction pipeline 是否能从 lab 扩展到 `parkinglot` 和 `seattle`。每个场景串行运行以下设置：

1. 纯原版 HUGS 12000 步。
2. 纯原版 HUGS 15000 步。
3. 纯原版 HUGS 12000 步 + xyz-only attention correction 6000 步，并记录 attention 第 3000 步和最终第 6000 步。
4. Sugar-HUGS 12000 步。
5. Sugar-HUGS 12000 步 + xyz-only attention correction 6000 步，并记录 attention 第 3000 步和最终第 6000 步。

当前 correction 设置：只优化人体高斯 `xyz`，不优化 opacity、scale、rotation、SH/color。token pooling 使用 attention pooling。这里的 `003000` 表示 correction 阶段的第 3000 步；`final` 表示 correction 阶段第 6000 步。

远端汇总文件：

```text
gpu-l40-1:/tmp/ml-hugs-output/parkinglot_seattle_pipeline_20260522/summary.md
gpu-l40-1:/tmp/ml-hugs-output/parkinglot_seattle_pipeline_20260522/summary.json
gpu-l40-1:/tmp/ml-hugs-output/parkinglot_seattle_pipeline_20260522/runner.log
```

#### parkinglot 结果

| 实验 | HUGS PSNR | HUGS SSIM | HUGS LPIPS | HUMAN PSNR | HUMAN SSIM | HUMAN LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HUGS12000 003000 | 26.4396 | 0.7838 | 0.2263 | 19.7341 | 0.6758 | 0.2232 |
| HUGS12000 final | 26.7238 | 0.8257 | 0.1550 | 19.3489 | 0.7032 | 0.1709 |
| HUGS15000 003000 | 26.4913 | 0.7874 | 0.2247 | 19.7445 | 0.6824 | 0.2170 |
| HUGS15000 final | 26.6424 | 0.8347 | 0.1450 | 19.1697 | 0.7091 | 0.1616 |
| HUGS12000+6000xyzAttention 003000 | 27.2721 | 0.8418 | 0.1390 | 19.9058 | 0.7257 | 0.1436 |
| HUGS12000+6000xyzAttention final | 27.2009 | 0.8444 | 0.1352 | 19.7643 | 0.7260 | 0.1438 |
| SugarHUGS12000 003000 | 26.3779 | 0.7859 | 0.2282 | 19.5552 | 0.6764 | 0.2209 |
| SugarHUGS12000 final | 26.6344 | 0.8246 | 0.1539 | 19.2717 | 0.7000 | 0.1702 |
| SugarHUGS12000+6000xyzAttention 003000 | 27.2752 | 0.8423 | 0.1365 | 19.8842 | 0.7242 | 0.1435 |
| SugarHUGS12000+6000xyzAttention final | 27.1805 | 0.8449 | 0.1320 | 19.6977 | 0.7227 | 0.1426 |

parkinglot 输出目录：

```text
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/parkinglot/hugs_trimlp/exp0_hugs_original_12000_parkinglot_20260522/2026-05-22_17-07-02
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/parkinglot/hugs_trimlp/exp0_hugs_original_15000_parkinglot_20260522/2026-05-22_17-49-15
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/parkinglot/hugs_trimlp/hugs12000_parkinglot_xyz_attentionpool_xyzonly_plus6000_20260522/2026-05-22_18-32-31
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/parkinglot/hugs_trimlp/scene_sugar_hugs_12000_parkinglot_20260522/2026-05-22_19-15-09
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/parkinglot/hugs_trimlp/sugarhugs12000_parkinglot_xyz_attentionpool_xyzonly_plus6000_20260522/2026-05-22_19-50-40
```

#### seattle 结果

| 实验 | HUGS PSNR | HUGS SSIM | HUGS LPIPS | HUMAN PSNR | HUMAN SSIM | HUMAN LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HUGS12000 003000 | 24.4765 | 0.7706 | 0.1865 | 18.6657 | 0.6083 | 0.1992 |
| HUGS12000 final | 25.7050 | 0.8300 | 0.1122 | 18.9170 | 0.6541 | 0.1425 |
| HUGS15000 003000 | 24.6444 | 0.7734 | 0.1803 | 18.4851 | 0.6059 | 0.1985 |
| HUGS15000 final | 26.0878 | 0.8426 | 0.1044 | 18.7852 | 0.6618 | 0.1389 |
| HUGS12000+6000xyzAttention 003000 | 26.2260 | 0.8450 | 0.1004 | 19.3294 | 0.6797 | 0.1330 |
| HUGS12000+6000xyzAttention final | 26.2756 | 0.8477 | 0.0964 | 19.2921 | 0.6771 | 0.1317 |
| SugarHUGS12000 003000 | 24.5716 | 0.7698 | 0.1853 | 18.5136 | 0.6007 | 0.1994 |
| SugarHUGS12000 final | 25.7089 | 0.8271 | 0.1128 | 18.9410 | 0.6529 | 0.1449 |
| SugarHUGS12000+6000xyzAttention 003000 | 26.2566 | 0.8432 | 0.1007 | 19.4184 | 0.6756 | 0.1332 |
| SugarHUGS12000+6000xyzAttention final | 26.3009 | 0.8462 | 0.0971 | 19.3517 | 0.6752 | 0.1316 |

seattle 输出目录：

```text
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/seattle/hugs_trimlp/exp0_hugs_original_12000_seattle_20260522/2026-05-22_20-35-33
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/seattle/hugs_trimlp/exp0_hugs_original_15000_seattle_20260522/2026-05-22_21-10-21
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/seattle/hugs_trimlp/hugs12000_seattle_xyz_attentionpool_xyzonly_plus6000_20260522/2026-05-22_21-55-59
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/seattle/hugs_trimlp/scene_sugar_hugs_12000_seattle_20260522/2026-05-22_22-49-30
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/seattle/hugs_trimlp/sugarhugs12000_seattle_xyz_attentionpool_xyzonly_plus6000_20260522/2026-05-22_23-25-26
```

阶段性观察：

- 在 `parkinglot` 和 `seattle` 上，`12000 HUGS + xyz-only attention correction` 的图像指标均明显好于对应的纯 HUGS 12000/15000，说明当前 correction pipeline 至少在跨场景上不是只对 lab 偶然有效。
- `003000` 到 `final` 的变化显示，继续 correction 到 6000 步通常可以改善 LPIPS/SSIM，但 HUMAN PSNR 不一定继续上升；这说明更长 correction 不是单调收益，后续需要保留中间 checkpoint 做选择。
- Sugar-HUGS 起点接 attention correction 的表现与纯 HUGS 起点接 correction 很接近，部分指标略好，部分略差；目前不能只凭这一轮判断 Sugar 初始化一定更优。
- 这些结果仍然只说明渲染指标提升；接触质量还需要后续引入更可靠的接触评估或人工/DECO 辅助可视化判断。
