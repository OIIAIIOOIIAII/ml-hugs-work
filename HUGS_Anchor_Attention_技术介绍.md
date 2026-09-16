# HUGS Anchor Attention 方案技术介绍

## 整体 Pipeline

本方案解决的核心问题：**HUGS 和 STM 均依赖 GT `smpl_optimized_aligned_scale.npz`，实际场景中不可得。** 我们构建了完整的"无 GT"pipeline，从单目视频出发，经粗对齐得到 SMPL 参数，再通过 Anchor Attention 精细修正人-场对齐。

```
单目视频
    │
    ├─ 粗对齐（离线预处理）
    │       ├─ Scale 估计（v3 足底 COLMAP 深度）
    │       └─ Translation 估计（VIMO 时序 HMR）
    │               ↓  smpl_optimized_aligned_scale.npz（误差 ~93mm）
    │
    └─ HUGS 训练 + Anchor Attention（端到端，18k 步）
            ├─ step 0-2000：HUGS 正常训练（背景+人体初始化）
            └─ step 2000-18000：Inline Anchor Attention
                    ├─ 渲染 loss（L1 + SSIM + LPIPS + Pearson depth）
                    └─ Anchor Attention correction
                            ├─ delta_transl：人体整体平移修正
                            └─ delta_mu：单个 GS 位置微调
```

**两套 pipeline 对比：**

| Pipeline | 对齐来源 | 方案 | 人体 PSNR（lab）|
|---|---|---|---:|
| GT 最优 | GT（0mm）| depth_sup12k + transl_xyz attn 6k（两阶段）| **19.78** |
| 无 GT 最优 | VIMO（93mm）| Inline Attention 18k（单阶段，Exp G）| **18.14**（peak）|
| STM 对比 | VIMO（93mm）| StM 复现（20k）| 17.45 |

---

## 一、粗对齐（Coarse Alignment）

### 1.1 背景与问题

NeuMan 数据集提供精细拟合的 `smpl_optimized_aligned_scale.npz`，包含：
- 每帧 SMPL pose/shape 参数
- **全局 scale**：COLMAP 单位 / meter 换算比例（lab 约 8.62）
- **per-frame global translation**：SMPL root 在 COLMAP 世界坐标系中的位置

替代方案迭代过程：

| 版本 | 方法 | 平均 3D 误差 |
|---|---|---|
| v1 ROMP | 逐帧独立估计 + focal 修正 | **222 mm** |
| v2 DepthPro | 背景深度标定 scale | **251 mm**（更差）|
| v3 足底 COLMAP | 足底投影附近 COLMAP 点估计 scale | scale 误差 4.6%（vs v1 的 8.6%）|
| **VIMO** | 时序感知 HMR，接入 NeuMan COLMAP 相机 | **93 mm**（改善 2.4×）|

### 1.2 Scale 估计（v3 足底 COLMAP 深度法）

**核心思路**：SMPL 模型给出人体在相机坐标系中的物理尺度（单位：meter），COLMAP 场景的单位是抽象坐标系。两者之间有一个全局线性换算因子 `scale_colmap_per_meter`。足底踩在地面上这一物理约束提供了最可靠的 scale 信号。

**实现步骤**（`scripts/coarse_align_smpl_neuman_v3.py`）：

1. 对每帧 SMPL forward，获得脚底顶点的 camera-space 位置（取 Y 最大的顶点，OpenCV 约定 Y 向下）
2. 将脚底顶点投影到 2D 图像平面
3. 在投影区域 ±80px 半径内，查找 COLMAP 3D 点中最近 10%（地面点）
4. 计算 scale 初始值：

```python
scale_init = median(colmap_z_foot / smpl_foot_z_m)
```

5. 优化阶段加入 foot-contact loss 进一步精化：

```python
loss_contact = (scale × foot_z_m − colmap_z_foot)²    # 权重 10.0
```

**Lab 场景结果**：scale = 8.218（vs GT 8.617，误差 4.6%），`t_world` 系统性偏移 [-0.45, 0.65, -0.69]（COLMAP units）。

### 1.3 Translation 估计（VIMO 时序 HMR）

**问题根因**：ROMP 逐帧独立估计 SMPL 参数，输出在相机坐标系下，且使用了错误的 focal length 假设（`focal=image_W/2=621`，实际 focal=1099.98），导致 X/Y 方向乘以了 `focal_romp/focal_actual ≈ 0.565` 的错误缩放，需要事后修正。帧间随机噪声（最大跳变 2.49 COLMAP units ≈ 289mm）无法通过优化消除。

**VIMO 改善机制**：VIMO（Video-based IMU-cues Optimization）是时序感知的 HMR 方法，在多帧联合估计时直接使用正确的 COLMAP focal（1099.98），多帧全局约束使相机-人体几何鲁棒，绝对位置估计更准（mean error 58% 降低：1.826→0.765 COLMAP units）。

**接入流程**（`scripts/run_vimo_neuman.py` + `scripts/gen_coarse_npz_vimo.py`）：

1. 将 NeuMan COLMAP 相机转换为 VIMO 输入格式（`camera.npy`），SAM 分割图转为 `tracks.npy`
2. VIMO 推理（无需 DROID-SLAM，直接用 COLMAP 相机），lab 103 帧约 30 秒
3. 输出 VIMO SMPL 参数（相机坐标系），利用 v3 scale（8.218）转换到 COLMAP 世界坐标系
4. 生成 `smpl_optimized_aligned_scale_vimo.npz`

**VIMO 误差分析**：

| 维度 | 误差 |
|---|---|
| 3D transl（均值）| 90mm（0.765 COLMAP units）|
| 3D transl（最大）| 340mm（2.96 COLMAP units）|
| 2D pelvis 投影（均值）| **11.1px**（极小，主要误差在深度而非侧向）|
| Scale 误差 | 4.85%（VIMO 8.218 vs GT 8.617）|

**帧间噪声分析（多帧监督信号不一致是核心瓶颈）**：VIMO 逐帧噪声方向各异（最大帧间跳变 1.84 COLMAP units，平滑后）。训练时每帧从不同方向拉扯人体高斯，最终高斯球"摊平"做空间平均 → 模糊。Scale 误差（4.85%）使深度方向偏近，进一步引入多帧不一致。

**轨迹平滑（已实施，Plan B）**（`scripts/smpl_traj_smooth.py`）：

```python
from scipy.signal import savgol_filter
smpl_transl_smoothed = savgol_filter(smpl_transl, window=11, polyorder=3, axis=0)
```

- 帧间最大跳变：2.49 → 1.84 COLMAP units（−26%）
- 平均修正幅度：0.34 COLMAP units（40mm）
- 输出：`smpl_optimized_aligned_scale_smoothed.npz`

---

## 二、Anchor Attention 优化

### 2.1 背景：为什么要 Attention 修正

粗对齐（93mm 误差）+ 原版 HUGS 训练后，人体 PSNR 仅 16-17 dB（GT 下 19.78 dB）。瓶颈：

1. SMPL `optim_trans`（渲染 loss 梯度驱动 transl 优化）对 90mm 量级的偏移驱动力极弱，高斯球倾向于用外观变化拟合而非整体移位
2. 单目深度监督（depth_w loss）提供几何约束，但对人-场接触边界的精细对齐信号仍不足
3. 多帧噪声方向各异，渲染梯度相互抵消，correction 需要具备"感知场景几何"的能力

**解决方案**：构建 Anchor Attention 模块——通过语义锚点将人体局部状态与邻近场景高斯进行 cross-attention 融合，输出与场景几何感知的修正量（delta_transl + delta_mu）。

### 2.2 背景优化：深度监督

Phase 1（或 inline 的早期阶段）在标准 HUGS 训练基础上加入**Pearson 相关系数深度 loss**（参考 DASR）：

```python
def _pearson_corrcoef(x, y):
    x = x - x.mean(); y = y - y.mean()
    return (x * y).sum() / (x.norm() * y.norm() + 1e-8)

# 两个深度变换取 min，尺度/偏移无关
loss_depth = -min(pearson(render_depth, mono_depth),
                  pearson(render_depth, 1/(mono_depth + 200)))
```

- **深度图来源**：预计算单目深度，16-bit PNG，路径 `mono_depth/*.png`，除以 10000 得米
- **作用对象**：human+scene 合并渲染的整幅深度图
- **权重**：`depth_w=0.05`
- **效果**：depth_sup 15k（无 attention）HUMAN PSNR 19.39，相比 HUGS baseline +0.59 dB；depth_sup 12k + attention 6k → 19.78（+0.98 vs baseline）

### 2.3 Anchor 绑定

#### 语义锚点定义

在 SMPL canonical 空间预定义 **16 个语义锚点**，覆盖接触风险最高的人体部位：

```python
["left_sole", "right_sole",       # 脚底（接触地面的核心区域）
 "left_toe",  "right_toe",        # 脚尖
 "left_heel", "right_heel",       # 脚跟
 "left_palm", "right_palm",       # 手掌
 "left_fingers", "right_fingers", # 指尖
 "buttocks",  "back",             # 臀部、背部
 "left_knee", "right_knee",       # 膝盖
 "left_elbow","right_elbow"]      # 肘部
```

每个锚点通过 SMPL LBS 权重（`lbs_weights`）定位到对应解剖区域的顶点集合（每个锚点 64 个候选顶点）。例如脚底锚点选取逻辑（`anchor_utils.py: generate_semantic_anchor_vertices`）：

- 筛选 `ankle + foot` 关节 LBS 权重总和 ≥ 0.20 且位于足部 Y 方向最低 38% 的顶点
- 从候选顶点中选取 Y 值最小（最低）、法向倾向于 `(0, -1, 0)` 的 64 个点
- 取平均位置 `pos_canon`、平均法向 `normal_canon`、平均 LBS 权重 `lbs_a`、覆盖半径 `radius`

#### Gaussian 到 Anchor 的绑定（`bind_gaussians_to_anchors`）

每个 human Gaussian 在 canonical 空间绑定到 top-m=2 个锚点，打分方式结合**几何距离**和 **LBS 权重相似度**：

```python
sigma = anchor_radius * sigma_scale    # sigma_scale=1.8, min_sigma=0.035

dist2     = cdist(mu_canon,      anchor_pos)²     # 几何距离平方
lbs_dist2 = cdist(gaussian_lbs,  anchor_lbs)²     # LBS 权重 L2 距离平方
scores    = -(dist2 / sigma²) - lambda_lbs * lbs_dist2   # lambda_lbs=0.35

weights, ids = topk(scores, k=2, dim=1)           # top-2 锚点
soft_weights = softmax(weights, dim=1)             # softmax 归一化
```

几何距离项保证 Gaussian 优先绑定空间邻近的锚点，LBS 距离项保证绑定到动作语义一致的锚点（避免跨关节混淆）。绑定在训练前一次性计算并缓存，forward 时直接查表。

#### Anchor 世界坐标实时更新（`_anchor_world_from_bindings`）

每次 forward，根据已经过 LBS 变形的 Gaussian 世界坐标，加权平均反算各锚点的实时位置：

```python
anchor_world.scatter_add_(0, ids, xyz_rep * weights)
anchor_world /= denom.clamp_min(1e-6)
```

anchor 位置随人体姿态实时变化，始终对应当前帧的解剖位置（脚底锚点在地面附近，手部锚点在手的位置）。

### 2.4 特征融合

#### Human 特征提取（16维，`_human_features`）

对每个 Gaussian，以其主绑定锚点为参考系提取局部特征：

| 分量 | 维度 | 含义 |
|---|:---:|---|
| `rel` | 3 | Gaussian 中心相对锚点世界坐标的偏移向量 |
| `dist` | 1 | 欧氏距离 |
| `normal_dist` | 1 | 沿锚点法线方向的投影距离（正值 = 朝向法线方向）|
| `opacity` | 1 | 当前不透明度 |
| `log_scale` | 3 | 各轴 log scale |
| `normals` | 3 | Gaussian 表面法线 |
| `rel_canon` | 3 | Canonical 空间相对 body center 的偏移（encode 身体位置先验）|
| `top_weight` | 1 | 主锚点绑定权重 |

#### Human Token 聚合（Attention Pooling，`_pool_human_tokens`）

对每个锚点，将属于该锚点的所有 Gaussian 聚合为一个 128 维 token：

1. 先经 `human_encoder`（MLP: 16→128）得到每个 Gaussian 的 encoded 特征 `local`
2. 以该锚点的可学习 embedding（`anchor_embed: Embedding(16, 128)`）作为 query
3. 计算 attention score，加入几何感知 bias：

```python
q      = self.anchor_embed(anchor_id)                           # [128]
logits = (local @ q) / sqrt(128)                               # [N_k,]

# 几何感知 bias：(dist, normal_dist, opacity, top_weight)
bias_in = cat([f[:, 3:5], f[:, 5:6], f[:, -1:]], dim=-1)      # [N_k, 4]
logits  = logits + self.human_attn_bias(bias_in).squeeze(-1)   # MLP: 4→32→1

alpha = softmax(logits, dim=0)
token = (alpha[:, None] * local).sum(dim=0)                    # [128]
```

`human_attn_bias` 让脚底等近接触区域、高不透明度的 Gaussian 在聚合中获得更大权重。实验证明 attention pooling 比 mean pooling 有显著优势。

结果：16 个锚点各一个 128 维 token，形成 `[16, 128]` 的 human token 矩阵。

### 2.5 Cross-Attention 计算（`_cross_attention`）

#### Scene 特征查询（`_query_scene_tokens`）

对每个锚点，在世界坐标系中用 KNN 查询最近的 K=32 个场景 Gaussian（过滤 `opacity < 0.01`，最多采样 200000 个候选）：

```python
d = cdist(anchor_world, scene_xyz_valid)              # [16, N_scene]
knn_dist, knn_idx = topk(d, k=32, largest=False)
```

每个邻居 Gaussian 的 11 维特征：`rel(3) + knn_dist(1) + opacity(1) + log_scale(3) + DC_color(3)`，形成 `[16, 32, 11]` 张量。

- `rel`：邻居相对锚点的位移，反映场景几何的局部分布（地面平整 → 相对位移集中在水平面）
- `opacity`：区分有效 Gaussian（地面/墙面）和透明占位符
- `DC_color`：0 阶球谐系数（漫反射颜色），帮助识别地面纹理

#### Cross-Attention

以 human token 为 Query，场景邻居 token 为 Key/Value，计算 scaled dot-product attention：

```python
scene_tokens = self.scene_encoder(scene_features)     # [16, 32, 128]
q = self.query(human_tokens)[:, None, :]              # [16,  1, 128]
k = self.key(scene_tokens)                            # [16, 32, 128]
v = self.value(scene_tokens)                          # [16, 32, 128]

logits  = (q * k).sum(dim=-1) / sqrt(128)             # [16, 32]
weights = softmax(logits, dim=-1)                     # [16, 32]
context = (weights[..., None] * v).sum(dim=1)         # [16, 128]
context = self.context_norm(context)
```

`context [16, 128]` 编码了每个锚点感知到的邻近场景几何语义：
- 脚底锚点附近有高不透明度的平整地面 → context 感知到地面存在，驱动脚部向地面靠近
- 锚点附近空旷（稀疏/低不透明度）→ context 反映无接触约束，修正量趋近于 0

`attn_weights [16, 32]` 输出用于诊断（每个锚点的 attention entropy、最大注意力权重、attention 加权距离）。

### 2.6 修正量输出

**Context 插值到每个 Gaussian：**

anchor-level context 通过绑定权重插值回每个 Gaussian：

```python
ctx_g = context[anchor_ids[:, :2]]                           # [N, 2, 128]
ctx_g = (ctx_g * anchor_weights[:, :2, None]).sum(dim=1)     # [N, 128]
```

**delta_transl — 人体整体平移修正（核心收益来源）：**

```python
global_ctx   = context.mean(dim=0)                   # [128]，全部锚点平均
delta_transl = self.delta_transl(global_ctx)          # [3]
delta_transl = clamp(delta_transl, -clamp_val, +clamp_val)
corrected["xyz"] = corrected["xyz"] + gamma_transl * delta_transl.reshape(1, 3)
```

`delta_transl` 是一个全局量，对**所有**人体 Gaussian 施加相同位移，修正 VIMO 全局定位偏差（如整体浮空或深度偏近）。

**GT pipeline**：`gamma_transl=0.05`，`transl_delta_clamp=0.2`（固定）。

**VIMO inline pipeline（Exp G）**：cosine curriculum 衰减：

```yaml
gamma_transl_init: 2.0      # 启动时允许最大 1.0 COLMAP unit (≈120mm) 修正
gamma_transl_final: 0.05    # densification 结束后收敛到精调模式
gamma_transl_decay_start: 2000
gamma_transl_decay_end: 15000
transl_delta_clamp: 0.5     # 与大 gamma_init 匹配
```

初期大修正量（最大 1.0 COLMAP unit）覆盖 VIMO 93mm 误差；后期精调模式防止振荡。

**delta_mu — per-Gaussian xyz 微调：**

```python
delta_mu = self.delta_mu(cat([human_features, ctx_g], dim=-1))   # [N, 3]
corrected["xyz"] = human_out["xyz"] + gamma_mu * delta_mu        # gamma_mu=0.005
```

每个 Gaussian 根据自身局部特征 + 锚点场景上下文独立预测微调量，修正因 LBS 近似引起的局部位置偏差。幅度远小于 delta_transl（gamma_mu=0.005 vs gamma_transl=0.05-2.0）。

**消融结论**：

| 修正方式 | Lab HUMAN PSNR | vs baseline |
|---|---:|---:|
| 无 correction（HUGS 12k）| 18.92 | — |
| xyz-only（delta_mu）| 19.48 | +0.56 |
| transl-only（delta_transl）| ~19.4 | ~+0.5 |
| **transl+xyz（两者组合）** | **19.78** | **+0.86** |

transl correction 是主要收益来源，xyz correction 在其基础上略有提升。

**正则化与稳定训练：**

```python
delta_loss = (delta_mu.pow(2).mean() + delta_transl.pow(2).mean())
# delta_loss_w = 0.001，防止修正量无限膨胀
```

所有 MLP 最后一层 `zero_init_last=True`（初始 delta=0，不破坏已收敛的粗对齐）；`correction_warmup_iters=500`（线性 warmup，防止初期不稳定的大 delta 破坏训练）。

---

## 三、Inline Attention（单阶段方案，针对粗对齐场景）

**两阶段方案的缺陷**：Phase 1（depth_sup 12k，无 correction）期间人体高斯在错误位置"定型"，densification 在错误梯度方向积累，Phase 2 再做 correction 时高斯分布已不理想。

**Inline Attention（Exp G）**：将 attention correction 内嵌进主训练，从 step 2000 就参与 densification：

```yaml
# 单阶段 18k
module_start_iter: 2000      # attention 模块从 step 2000 启动
correction_start_iter: 2000  # correction 头从 step 2000 输出修正量
correction_warmup_iters: 500 # 前 500 步 warmup
smpl_trans lr: 0.0005        # 适度提高，加速 transl 参数学习
```

人体高斯在 correction 修正下的正确位置上做 densification，避免在错误位置积累大量错位高斯。

**Exp G 结果（VIMO 93mm）：**

| step | 人体 PSNR |
|---:|---:|
| 4000 | 18.08 |
| **8000（峰值）** | **18.14** |
| 18000（final）| 17.79 |

vs 两阶段 Exp D（depth_sup12k+attn6k，VIMO 93mm）：**17.54**，提升 **+0.60 dB**。
vs StM F2 复现（VIMO 93mm，20k steps）：**17.45**，final-vs-final 差距 **+0.34 dB**。

---

## 四、指标汇总

### GT 对齐 — Lab 场景完整消融

| 方案 | 人体 PSNR↑ | 全图 PSNR↑ | 步数 |
|---|---:|---:|---:|
| HUGS 原版（baseline）| 18.92 | 26.04 | 15k |
| xyz-only attention | 19.48 | 26.40 | 18k |
| ADC12k + transl_xyz attn stage2 | 19.55 | 26.44 | 18k |
| **depth_sup12k + transl_xyz attn 6k** | **19.78** | 26.38 | 18k |
| STM 论文 | 19.73 | **26.60** | — |

### 粗对齐 — Lab 场景（VIMO 93mm）

| 方案 | 对齐方式 | 人体 PSNR（peak）| 人体 PSNR（final）|
|---|---|---:|---:|
| HUGS baseline | VIMO | — | 16.78 |
| Exp D（两阶段）| VIMO | — | 17.54 |
| StM F2 复现 | VIMO | — | 17.45 |
| **Exp G（Inline Attention）** | VIMO | **18.14** ★ | **17.79** |

### 多场景泛化（GT，depth_sup12k + attn6k vs STM）

| 场景 | 我们 人体 PSNR | STM 人体 PSNR | Ours vs STM |
|---|---:|---:|---:|
| Lab | **19.78** | 19.73 | **+0.05** |
| Bike | **20.43** | 20.41 | **+0.02** |
| Parkinglot | **19.87** | 19.80 | **+0.07** |
| Citron | 20.18 | **20.20** | −0.02 |
| Seattle | 19.64 | **19.93** | −0.29 |
| Jogging | 17.80 | **18.28** | −0.48 |
| **平均** | 19.62 | **19.73** | **−0.11** |
