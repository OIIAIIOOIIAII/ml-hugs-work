# 前馈 3D Gaussian Splatting 学习与面试指南

> 面向本项目的学习文档。最后更新：2026-08-25。  
> 目标：理解“前馈 3DGS”是什么、代表性论文的网络架构和取舍，以及 Human3R/GUSH3R 与本项目接触修正层的关系。  
> 阅读原则：**优化式 3DGS、通用前馈 3DGS、前馈人体-场景重建、我们的接触修正层是四个不同层次**，面试时必须明确区分。

---

## 0. 一页结论：可以先背下来的版本

传统 3D Gaussian Splatting（3DGS）为每个新场景迭代优化一组 Gaussian，质量高但速度慢；前馈 3DGS 用已训练的网络从图像、少量多视图图像或视频中直接预测可渲染的 Gaussian，因此对新场景只需要一次或少量前向推理。

前馈网络通常遵循：

```text
图像/视频
  -> CNN/ViT 图像特征
  -> 跨视图匹配、时序记忆或几何 Transformer
  -> 深度/点图/相机/SMPL 等几何中间量
  -> Gaussian 参数头（位置、尺度、旋转、透明度、颜色）
  -> 融合/去重/置信度过滤
  -> Gaussian splatting rasterizer
```

本项目不是重新训练一个通用前馈 3DGS backbone，而是在 **Human3R/GUSH3R 的前馈人体-场景重建之后**，增加接触感知的低维控制层：预测左右脚接触、地面几何、root/foot residual 和不确定性；再通过 SMPL-X 的 LBS 更新人体 Gaussian，静态场景 Gaussian 保持不变。它解决的是前馈重建中常见但 backbone 不会显式保证的滑步、悬空和穿透问题。

---

## 1. 3DGS 基础：网络到底要预测什么？

一个 3D Gaussian 常写为：

\[
G_i=(\mu_i, q_i, s_i, \alpha_i, c_i)
\]

| 参数 | 含义 | 实现中的常见形式 |
|---|---|---|
| \(\mu_i\in\mathbb{R}^3\) | Gaussian 中心 | 世界坐标点；常由深度反投影或点图给出 |
| \(q_i\) | 三维旋转 | 单位四元数，或转换为 \(R_i\in SO(3)\) |
| \(s_i\in\mathbb{R}_+^3\) | 三轴尺度 | 通常预测 log-scale 后再指数化/截断 |
| \(\alpha_i\) | 不透明度 | sigmoid 后的 alpha，控制可见性和融合 |
| \(c_i\) | 外观 | RGB，或球谐（SH）系数以建模视角相关颜色 |

协方差可由旋转和尺度得到：

\[
\Sigma_i=R_i\operatorname{diag}(s_i^2)R_i^T.
\]

渲染时每个 Gaussian 被投影为屏幕空间椭圆。按深度排序后，对像素做近似的 alpha 合成：

\[
C(\mathbf p)=\sum_i T_i(\mathbf p)\,\alpha_i(\mathbf p)c_i,
\qquad
T_i=\prod_{j<i}(1-\alpha_j(\mathbf p)).
\]

这就是 3DGS 快的原因：它不是像 NeRF 那样对每个像素沿光线密集采样并查询 MLP，而是把显式 primitive 直接 rasterize。

### 1.1 优化式与前馈式的根本区别

| | 优化式 3DGS | 前馈 3DGS |
|---|---|---|
| 给新场景时 | 从初始化开始反复优化 Gaussian | 网络直接预测 Gaussian |
| 时间 | 秒到小时，依赖训练步数 | 毫秒到秒级，取决于输入帧数和 backbone |
| 优势 | 对该序列拟合充分，质量往往更高 | 泛化、在线和大规模部署更友好 |
| 风险 | 慢、每个场景要重训/重优化 | 受训练分布限制，几何尺度和遮挡易错 |
| 人体动态问题 | 可通过逐帧/变形优化修正 | 易有时序抖动、接触不一致、姿态/场景坐标失配 |

不要说“前馈方法不优化”。严格说，它在训练阶段仍通过重建损失优化网络参数；区别是**测试时不再为每个新场景大量优化 Gaussian 参数**。

---

## 2. 前馈 3DGS 的通用网络范式

### 2.1 从几何中间量到 Gaussian

最稳定的一类结构不是直接从全局 latent 生成一堆无序的 3D 点，而是先得到像素对齐的几何：

```text
输入像素 u=(x,y)
    -> 特征 f(u)
    -> 深度 d(u) 或世界点 X(u)
    -> 用相机 K, [R|t] 反投影得到 Gaussian mean μ(u)
    -> 小型 MLP/卷积头预测 scale、rotation、opacity、color/SH
```

深度反投影为：

\[
X_c(u)=d(u)K^{-1}[x,y,1]^T,
\qquad X_w=R^T(X_c-t).
\]

它把最难的“3D 中心在哪里”交给多视图几何/深度估计，把外观与形状局部参数交给 Gaussian head。实践中还要配合：

- confidence/visibility：过滤错误的深度或遮挡像素；
- multi-view fusion：同一表面被多帧重复预测，需要合并或选择；
- scale clamp：防止巨大 Gaussian 遮住整张图；
- quaternion normalization：防止无效旋转；
- opacity regularization：防止所有 Gaussian 变得半透明。

### 2.2 三种主流 backbone

**A. 像素对齐 Gaussian image。** CNN/U-Net/ViT 解码成与图像同分辨率的参数图，每个像素对应一个或多个 Gaussian。优点是简单和快；缺点是单视图不可见面没有几何证据。

**B. 多视图匹配 + Gaussian head。** 每张图先提取 feature，再沿 epipolar line 做 cross-attention、cost volume 或 plane sweep，得到有多视图约束的 depth/point map，最后生成 Gaussian。pixelSplat、MVSplat 属于此类。

**C. 视频时序记忆 + 人体先验。** 对视频不断更新 state/memory token，联合预测点图、相机、人体参数和 Gaussian。它可在线处理长视频，但长时序的递归融合误差会积累。CUT3R/Human3R/GUSH3R 的关系接近此类。

### 2.3 训练信号

常见训练损失不只是一项 RGB loss：

\[
\mathcal L=
\lambda_{rgb}\mathcal L_{rgb}+
\lambda_{ssim}\mathcal L_{ssim}+
\lambda_{lpips}\mathcal L_{lpips}
+\lambda_{geo}\mathcal L_{depth/point}
+\lambda_{pose}\mathcal L_{camera/pose}
+\lambda_{reg}\mathcal L_{Gaussian}.
\]

- RGB/SSIM/LPIPS 监督可渲染质量；
- depth、point map、normal、相机 pose 监督几何；
- Gaussian regularization 限制尺度、opacity 或不合理的 offset；
- 动态视频还需要 temporal consistency、flow consistency 或 state consistency。

对于接触问题，仅有重建 loss 不够：脚部被遮挡时，视觉 loss 对“脚是否真的贴地”几乎没有直接约束，这正是本项目增加显式接触层的动机。

---

## 3. 代表性论文精读

下面按“读它能学到什么”组织。论文版本可能有不同实现/配置；面试应说明的是架构思想和数据流，不要杜撰某个未核对的层数或通道数。

### 3.1 3D Gaussian Splatting（Kerbl et al., 2023）：全部方法的基座

**定位**：不是前馈方法，而是后续所有 Gaussian 表达、可微 splatting 和 densification 的基础。

```text
SfM sparse points
  -> 初始化 Gaussian
  -> 可微 splatting render
  -> RGB 重建梯度回传
  -> 更新 μ / covariance / opacity / SH
  -> densify、split、prune
```

要点：显式可编辑的 Gaussian 表达、高效 tile-based rasterizer、训练中 densification/pruning。前馈方法经常省略或大幅弱化“每个测试场景的 densification”，以换取速度。

**面试可说**：3DGS 的速度来自 render 表达，不等于它天然前馈；原始 3DGS 对新场景依旧是优化式的。

### 3.2 pixelSplat（Charatan et al., 2024）：概率深度的多视图前馈 Gaussian

**输入/输出**：少量带相机信息的源视图，输出用于目标视图渲染的一组 Gaussian。

**架构思想**：

```text
两张源图像 + 相机
  -> shared image encoder
  -> epipolar-aware cross-view Transformer
  -> 每像素离散/连续深度分布
  -> 从深度分布采样 3D mean（而非只取单一深度）
  -> per-pixel Gaussian heads: opacity / scale / rotation / color
  -> splatting 到 target view
```

最重要的思想是 **probabilistic depth**。遮挡、低纹理和多视图歧义会使一个像素对应多个可能深度。若只回归一个 depth，错误 mean 会产生明显的漂浮物；pixelSplat 将深度表示为分布并从中采样，将几何不确定性转为多个可能的 Gaussian placement。

**你应理解的组件**：

- epipolar attention：query 像素只在另一视图与其对应的极线上搜索，减少全局 attention 的无效匹配；
- depth distribution head：预测深度 bin/logit，而不是一个标量；
- unprojection：利用已知相机把 sampled depth 转为 3D mean；
- attribute heads：从融合 feature 回归 Gaussian 的其余属性。

**优缺点**：泛化快、适合稀疏多视图；但通常假设输入相机/相对位姿可用，且不直接解决长视频中动态人体的非刚体运动。

### 3.3 MVSplat（Chen et al., 2024）：通用多视图 stereo-to-splat 路线

**输入/输出**：少量多视图图像与相机，直接生成可新视角渲染的 Gaussian 表达。

**核心架构**：

```text
多源视图
  -> 2D feature encoder
  -> geometry-aware multi-view feature aggregation
     （常见实现为 plane sweep / cost volume / epipolar matching）
  -> 深度、置信度与像素特征
  -> 反投影为 3D means
  -> Gaussian parameter decoder
  -> differentiable Gaussian renderer
```

与 pixelSplat 相比，MVSplat 的学习重点是把多视图 stereo 作为可靠几何入口；输出则仍是显式 Gaussian，而不是体素体或 NeRF MLP。它是很好的面试对比对象：**前馈 3DGS 的关键往往不是“生成颜色”，而是怎样可靠地得到可融合的 3D geometry。**

**应追问自己**：如果相机 pose 不准怎么办？答案是几何 matching 会直接退化；因此视频系统通常还要预测/优化相机，或把 pose uncertainty 纳入融合策略。

### 3.4 Splatter Image（Szymanowicz et al., 2024）：Gaussian 参数图

**输入/输出**：单图或条件图像，输出二维网格形式的 Gaussian parameter image；网格中每个位置存一个 3D Gaussian 的参数。

```text
条件图像（可含相机 ray / pose 条件）
  -> U-Net 风格 image-to-image backbone
  -> splatter image：每像素一组 Gaussian 参数
  -> reshape 为 Gaussian set
  -> 3DGS rasterizer
```

它的启发在于把“不定长点集生成”改写为图像到图像预测：网络保留卷积/U-Net 的空间归纳偏置，训练和并行输出都容易。

**局限**：单视图看不见的背面需要依赖数据先验；Gaussian 虽在 3D，但输出网格仍强烈受输入视角约束。因此它更适合对象级生成/重建，而不是要求严格世界坐标一致的长视频 SLAM。

### 3.5 LGM（Tang et al., 2024）：生成式 multi-view 到 Gaussian

**定位**：面向 image-to-3D 内容生成的快速 Gaussian generator，不等于真实世界视频重建。

**典型管线**：

```text
单张条件图
  -> 多视图生成模型得到固定数量的视图
  -> 将多视图堆叠输入 asymmetric U-Net
  -> cross-view attention / multi-view feature fusion
  -> 每个视图输出 Gaussian image（常见为 14D 属性）
  -> 合并所有视图的 Gaussians 并渲染
```

常见的 14D 参数划分为：position 3、opacity 1、scale 3、rotation quaternion 4、RGB 3。LGM 的工程价值是把复杂 3D 生成转成规则的多视图张量预测；其视觉质量高度依赖上游多视图生成是否跨视图一致。

**与本项目的区别**：LGM 主要回答“从一张图生成一个合理的 3D 物体”；GUSH3R/本项目回答“从真实视频恢复同一个动态人体和真实场景，并维持坐标、时序和接触一致性”。

### 3.6 CUT3R（Wang et al., 2025）：前馈视频几何与递归状态

**定位**：重要的前馈视频 3D 重建 backbone，主要输出点图、置信度、相机/状态，而不是最终以 Gaussian 为主要输出。

```text
视频帧
  -> patch embedding + CroCo/ViT encoder
  -> state tokens / local recurrent memory
  -> decoder 对当前帧与历史状态 cross-attend
  -> point map、confidence、camera/pose 等 heads
  -> 将新状态写回 memory，继续处理下一帧
```

它解决“每处理一帧都重跑全序列 Transformer”的问题，使在线/长视频处理成为可能。但 recurrent state 也带来 drift 和错误累积。GUSH3R 的长时序背景问题正是很实际的提醒：即使每帧 backbone 有输出，后续融合后的背景若反过来用于重渲所有历史帧，也会造成 future-to-past contamination。

### 3.7 Human3R（2025）：人体-场景前馈几何基础

**定位**：从单目人体-场景视频中联合预测人体和场景相关几何/人体参数的前馈基础。对本项目而言，它是 P0 几何预验证平台：提供 FrameInput 中的 depth、point map、camera、SMPL-X 和置信度等字段。

应抓住的不是某个实现细节，而是它把传统“人体姿态估计”和“场景几何恢复”放在同一个时序重建系统中。它的输出让我们能够定义：脚 anchor 在哪、场景局部表面在哪里、二者是否真的处于同一坐标系。

**重要边界**：Human3R 本身不等于最终人体 Gaussian 表达，所以不能把 Human3R 上的 mesh/point proxy 结果直接当作 GUSH3R Gaussian 主实验结果。

### 3.8 GUSH3R（Abe et al., 2026）：本项目直接使用的前馈人体-场景 Gaussian

**目标**：从单目人体-场景视频在线重建静态场景和动态人，并输出可渲染的 3D Gaussian。它继承/采用 Human3R、CUT3R 系的时序几何能力，并增加 background Gaussian 与 human Gaussian 分支。

#### 本地代码对应的总体结构

```text
video frames
  -> CroCo/CUT3R-style patch encoder + ray-map encoder
  -> state decoder / local memory / cross-attention
  -> downstream geometry heads
       point/depth/confidence/camera + human mask + SMPL-related features
  ├-> background GS branch
  │    point map + raw GS feature
  │      -> Gaussian decoder
  │      -> confidence/mask filtering
  │      -> voxel merge + max-Gaussian cap
  └-> human GS branch
       DINO image tokens + raw RGB features + SMPL-X query points
         -> Human Transformer
         -> GSLayer
         -> posed human Gaussians by SMPL-X LBS
  -> Gaussian rasterizer -> RGB/depth/alpha
```

#### 背景 Gaussian 分支

本地 `ARCroco3DStereo` 中，背景使用 `DecoderSplattingCUDA`。模型维护/融合背景 Gaussian，并受到如下运行时参数影响：

- `bg_gaussian_max`：背景 Gaussian 最大数；
- `bg_mask_threshold` 与 `bg_mask_dilation`：用人体 mask 排除人体附近背景点；
- `bg_voxel_size`：背景融合时的 voxel 尺寸；
- point confidence threshold：过滤低置信度点；
- 可选 `bg_point_offset_head`：从 raw Gaussian feature 预测小的 3D offset，最后一层零初始化，避免一开始破坏基础点图。

背景 rasterizer 接收 mean、rotation、scale、opacity、SH/RGB feature 和 covariance，输出 RGB、depth、alpha。这里一定要理解：它是“预测/融合后的显式 Gaussian 再渲染”，而不是 NeRF 式隐式场函数。

#### 人体 Gaussian 分支：最值得掌握的架构细节

本地 `HumanGSHead` 的可验证配置如下：

```text
SMPL-X shape/pose
  -> canonical SMPL-X mesh sampling
     1000 groups × 10 Gaussians/group = 10,000 canonical queries
  -> PointEmbed（canonical query geometry）

DINO/CUT3R image tokens (768D)
  + raw RGB Conv(7×7) feature（pool 后加回 image tokens）
  -> projection to 512D

SMPL query feature (1792D)
  -> projection to 512D

query/image cross-attention Human Transformer
  (4 layers, hidden 512, 8 heads)
  -> GSLayer
  -> local Gaussian attributes

SMPL-X LBS transforms
  -> 将 canonical/zero-pose query points 变到当前人体姿态
  -> human Gaussian rendering
```

这套设计有两个关键归纳偏置：

1. **canonical human prior**：Gaussian 不从空中自由出现，而是锚定于 SMPL-X 表面采样点附近；
2. **appearance-geometry 解耦**：SMPL-X/LBS 给出大体人体运动，图像 token 通过 attention 补充衣物、局部外观和 Gaussian 属性。

这也是为什么本项目可以优先修正低维 SMPL-X，而无需每帧重新预测十万个 Gaussian：姿态变化可由 LBS 传播到人体 Gaussian。

#### GUSH3R 的已知工程风险

本地 2026-08 的诊断发现，官方长视频流程若使用“最终融合的 background Gaussian”重渲全部历史帧，会将晚期 fusion error 污染早期帧。32/64/128/229 帧窗口和 per-frame causal render 的对照支持这一结论。已实现 bounded/chunked background rendering，但完整 229 帧质量回归尚待完成。

面试时这不是负面信息；它说明你理解前馈时序系统的真实难点：**backbone 前馈不代表后续 map fusion 和 render policy 自动时序一致。**

---

## 4. 本项目：前馈接触修正层的网络与数据流

### 4.1 研究问题

前馈重建的优化目标主要是“新视角看起来正确”，但脚底接触是局部物理/运动学约束。它常被遮挡，且小尺度误差对视觉重建损失影响有限，因此会出现：

- foot sliding：标记为接触后脚还在地面上横向漂移；
- floating：脚与地面有可见间隙；
- penetration：脚/人体进入地面或物体；
- jitter：接触状态在相邻帧频繁切换；
- coordinate mismatch：人体、point map 和场景并不处于同一可靠坐标系。

### 4.2 我们的系统架构

```text
Human3R / GUSH3R 输出
  -> coordinate convention conversion + robust Sim(3)
  -> SMPL-X forward：左右脚踝/脚 anchor、速度、朝向
  -> 从 depth/point map 得到 local surface proxy
     （human mask 过滤 + KNN/radius 邻域 + 加权 PCA 拟合平面）
  -> 58D causal feature
  -> 规则 hysteresis baseline 或轻量 ContactGRU/ContactTCN
  -> 38D 输出：contact、surface、root/foot residual、risk、uncertainty
  -> 低置信度回退；高置信度时 root correction + 后续 IK/projection
  -> corrected SMPL-X
  -> LBS 更新 human Gaussian；scene Gaussian 不动
  -> renderer / 62-byte contact packet
```

### 4.3 真实的特征与网络接口

第一版只做左右脚，\(K=2\)。每帧输入为 58D：

| 组成 | 每脚维度 | 两脚 |
|---|---:|---:|
| anchor position / velocity / orientation | 3 + 3 + 6 | 24 |
| local surface point / normal / distance / confidence | 3 + 3 + 1 + 1 | 16 |
| alignment confidence、previous contact、previous residual | 1 + 1 + 3 | 10 |
| 每脚合计 | 25 | 50 |
| root position、root velocity、alignment residual/confidence | - | 8 |
| **总计** | - | **58** |

网络统一输出 38D：

| 输出 | 维度 |
|---|---:|
| 左右脚 contact logits | 2 |
| contact point | 6 |
| surface distance | 2 |
| surface normal | 6 |
| foot-related pose residual | 12 |
| root residual（平移/旋转） | 6 |
| penetration risk | 2 |
| uncertainty logits | 2 |
| **总计** | **38** |

已实现的网络为：

```text
ContactGRU:
  58D input -> linear projection -> causal GRU -> MLP head -> 38D

ContactTCN:
  58D temporal window -> causal dilated Conv1D residual blocks -> MLP head -> 38D
```

默认使用过去 5--8 帧、hidden size 128。选择 GRU/TCN 的理由不是它们一定优于 Transformer，而是：接触模块只需短历史、必须因果低延迟、数据量尚有限，它们是可靠且可部署的基线。

### 4.4 接触状态机与不确定性

规则 baseline 使用距离、速度、局部表面置信度与 hysteresis：进入接触阈值、保持阈值、退出阈值不同，避免临界帧抖动。系统状态有 Tracking、Uncertain、Reset。

这是必要的安全设计：若 Sim(3) 对齐残差大，或局部 surface proxy 支撑点太少，不能强行执行 IK。最合理的行为是输出高 uncertainty 并回退到 backbone 原始姿态。

### 4.5 训练目标与指标

初始损失组合：

```text
state/contact BCE
+ surface point/distance SmoothL1
+ normal cosine loss
+ penetration BCE/regression
+ root/pose residual loss
+ temporal/velocity consistency
+ uncertainty calibration
```

正式指标不能只用 render PSNR：

- contact precision/recall/F1 与 transition F1；
- foot sliding、contact jitter；
- penetration ratio/depth；
- surface-distance MAE；
- contact ROI 与全图的 PSNR/SSIM/LPIPS；
- human Gaussian temporal flicker；
- 接触模块 FPS、端到端 latency、packet bitrate；
- 高不确定性帧是否正确回退。

---

## 5. 当前完成度：面试中必须如实说

### 已完成

- Human3R 与 GUSH3R 的单帧 GPU 前向、Gaussian 渲染和所需环境/权重准备；
- 标准 FrameInput 导出：图像、depth、point map、camera、SMPL-X、mask/confidence、时间戳；
- robust Sim(3) 接口、SMPL-X 左右脚 anchor、局部加权 PCA surface proxy；
- 规则接触状态机、58D feature、38D schema、GRU/TCN、训练/验证/因果推理、LBS adapter、62-byte packet；
- 合成数据上 GRU/TCN 的训练到推理全链路，核心单元测试 4/4 通过；
- NeuMan `lab` 103 帧的 Human3R 导出、SMPL 顶点/camera Sim(3) 对照和 proxy 诊断；
- GUSH3R 长窗口背景融合问题的受控诊断与 chunked renderer 实现。

### 尚未完成，不能夸大

- 真实 NeuMan 对齐/proxy 置信度尚未达到接触标签验收条件；不能据此声称真实接触质量已改善；
- 可微 SMPL-X IK、脚底 vertex ROI 与物理 penetration projection 尚未完成；
- BEDLAM/PROX 的真实监督训练尚未完成；
- GUSH3R 内部 corrected SMPL-X 重算和正式 renderer 主实验尚未完成；
- 因而尚未得到“原始 GUSH3R vs GUSH3R+接触修正”的正式主表。

正确表述是：**完成了可运行的系统骨架、合成闭环和真实几何失败诊断；下一步在可信坐标与标签之上完成真实验证。**

---

## 6. 面试高频问题速答

### Q1：为什么不用端到端网络直接输出修正后的 Gaussian？

因为接触误差本质上是低维运动学问题，而人体 Gaussian 数量大且外观属性多。直接改全量 Gaussian 的参数空间大、缺乏可解释性、容易破坏衣物/纹理并增加传输成本。SMPL-X residual + LBS 保持了人体结构先验，只需传少量连续变量。

### Q2：为什么 surface proxy 不直接使用 Gaussian center？

Gaussian center 是渲染 primitive 的位置，不保证位于真实表面；它会受 opacity、scale、视角和融合策略影响。第一版应从 depth/point map 的局部邻域拟合平面或 MLS surface，并用 support/confidence 判断可靠性；后续可增加 Gaussian-aware proxy，而不是把 center 当作真值表面。

### Q3：为什么 Sim(3) 而不是只做平移？

Human3R、SMPL-X、NeuMan/HUGS scene world 可能同时存在尺度、旋转和平移误差。接触距离对尺度非常敏感，因此必须估计 scale、rotation、translation，并在 hold-out 对应点上报告误差；只有 translation fallback 不能作为正式几何校准。

### Q4：为什么需要规则 baseline，直接训 GRU 不行吗？

规则 baseline 可验证坐标、anchor、surface proxy 和接触定义是否正确，也能定位失败源。若无可信几何闭环，神经网络会拟合伪标签或坐标偏差，表面上 loss 下降但没有真实接触改善。

### Q5：前馈 3DGS 与 diffusion 3D generation 的区别？

前馈重建强调对给定输入的度量几何、相机一致性和快速新视角渲染；diffusion image-to-3D 更强调从不完整观测生成合理且多样的 3D 内容。二者可结合，例如 LGM 用生成的多视图驱动 Gaussian，但真实视频接触任务更需要前者的几何可验证性。

### Q6：你从 GUSH3R 长视频问题学到了什么？

前馈 backbone 的单帧输出好，不代表后续长期 fusion/render 一定好。若使用最后时刻融合出的场景 Gaussian 重渲所有历史帧，未来误差会污染过去。应采用因果 snapshot、有限窗口/chunked fusion，且将背景质量、人体质量和接触 ROI 分开评估。

---

## 7. 建议学习顺序（7 天）

1. **第 1 天**：读原始 3DGS，能推导 Gaussian 参数、投影和 alpha compositing。
2. **第 2 天**：读 pixelSplat，重点理解 epipolar matching 与 probabilistic depth。
3. **第 3 天**：读 MVSplat，比较 cost volume/多视图特征融合与直接 parameter image。
4. **第 4 天**：读 Splatter Image 与 LGM，理解“Gaussian image”和生成式 multi-view 路线。
5. **第 5 天**：读 CUT3R/Human3R，画出 video state/memory、point map、camera、SMPL 输出关系。
6. **第 6 天**：对照本地 GUSH3R 代码阅读 `src/dust3r/model.py`、`human_gs_head.py`、Gaussian decoder；自己画出 background/human 双分支。
7. **第 7 天**：读本项目 `contact_streaming/`、`TODO.md` 和两份报告，准备“问题—设计—验证—边界—下一步”的三分钟讲解。

---

## 8. 关键阅读链接

- 3DGS: *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (2023), https://arxiv.org/abs/2308.04079
- pixelSplat: *3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction* (2024), https://arxiv.org/abs/2312.15173
- MVSplat: *Efficient 3D Gaussian Splatting from Sparse Multi-View Images* (2024), https://arxiv.org/abs/2403.14627
- Splatter Image: *Ultra-Fast Single-View 3D Reconstruction* (2024), https://arxiv.org/abs/2312.13150
- LGM: *Large Multi-View Gaussian Model for High-Resolution 3D Content Creation* (2024), https://arxiv.org/abs/2402.05054
- CUT3R: *Continuous 3D Perception Model with Persistent State* (2025), https://arxiv.org/abs/2501.12387
- Human3R: https://arxiv.org/abs/2510.06219
- GUSH3R: *Everyone Everywhere All at Once as Gaussians* (2026), https://arxiv.org/abs/2607.05243

本项目内部推荐阅读顺序：

- `forward_contact_pipeline/TODO.md`：明确项目阶段与未完成项；
- `forward_contact_pipeline/reports/neuman_lab_p0_20260804.md`：真实坐标/proxy 为什么尚未验收；
- `forward_contact_pipeline/reports/gush3r_quality_diagnosis_20260804.md`：长视频背景融合诊断；
- `GUSH3R/src/dust3r/model.py`：总模型、background/human branch 入口；
- `GUSH3R/src/dust3r/heads/human_gs/human_gs_head.py`：SMPL-X 引导的人体 Gaussian head；
- `contact_streaming/models.py` 与 `schema.py`：本项目 GRU/TCN 与 38D 输出定义。

---

## 9. 三分钟项目陈述模板

> 我研究的是前馈人体-场景 Gaussian 重建中的接触一致性。传统 3DGS 通常针对一个序列优化大量 Gaussian；前馈方法则通过视觉 Transformer、跨视图或时序记忆，直接预测点图、相机和 Gaussian，因此推理快，但容易出现人体与场景之间的滑步和穿透。  
>
> 我们以 Human3R/GUSH3R 为 backbone：前者提供人体、深度、点图和 SMPL-X 等几何中间量，后者将动态人体和静态场景表达为 Gaussian。我的工作不重新预测全量 Gaussian，而是从 SMPL-X 得到左右脚 anchor，从局部点图拟合表面，通过规则状态机或轻量 causal GRU/TCN 预测接触、root/foot residual 和不确定性。修正只作用于低维 SMPL-X，再通过 LBS 更新人体 Gaussian，静态场景不变，因此更实时、可解释，也适合低带宽传输。  
>
> 目前我完成了数据 schema、坐标对齐、局部 surface proxy、时序网络、packet 和 LBS 接口，并在合成数据上跑通端到端链路。真实 NeuMan 数据提示跨模型坐标对齐仍是瓶颈，所以我没有把低置信度 proxy 当作真标签；下一步是完成重投影与场景 proxy 校准，再用 PROX/BEDLAM 做真实监督，并在 GUSH3R 上测量 contact F1、滑步、穿透、接触 ROI 渲染质量和端到端延迟。

