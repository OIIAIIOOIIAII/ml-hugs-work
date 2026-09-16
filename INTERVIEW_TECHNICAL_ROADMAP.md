# 两个研究点技术路线（面试深挖版）

> 更新：2026-09-10。用途：技术面试中的系统讲解、方法追问与边界说明。
>
> 原则：研究点一是已完成的 HUGS 人—场联合 Gaussian 优化结果；研究点二是以前馈人体—场景 Gaussian 为目标接口、已完成机制验证但真实前端仍受几何门槛约束的接触一致性路线。两者不能混讲，也不能把第二点的 oracle/proxy 结果表述为真实前馈结果。

---

## 0. 30 秒总述

我的工作可以概括为两个连续的问题。

1. **研究点一：离线高质量的人—场联合重建。** 在 HUGS 的 canonical human Gaussian + static scene Gaussian 框架中，用 SMPL-LBS 保证人体运动结构，用 Anchor Attention 从人体语义锚点查询局部场景，预测接触相关的 Gaussian 位置修正，从而降低人体与场景的错位和遮挡冲突。
2. **研究点二：在线前馈重建的接触一致性。** 前馈模型即使能快速重建人和场景，也常有脚底滑动、穿透、局部表面不可靠的问题。我们不逐帧优化几十万 Gaussian，而是在冻结前馈 backbone 后，利用局部人—场关系和因果状态，仅修正 SMPL-X 的 root/foot 低维变量，再经 LBS 更新人体 Gaussian；几何不可靠时应主动 abstain，而不是强行吸附到错误表面。

二者共享的核心思想是：**人体使用结构化、低维的运动控制；场景使用高容量的静态 Gaussian 表示；人—场交互通过局部关系模块建模，而不是靠无约束地移动全部 Gaussian。**

---

## 1. 研究点一：Anchor Attention 的人—场联合 Gaussian 优化

### 1.1 要解决的具体问题

普通 HUGS 将人体和场景分别表示，但它们会在同一像素、同一空间区域竞争解释图像：

- 人体姿态/平移有误差时，人体 Gaussian 与地面、墙面或物体错位；
- scene Gaussian 可能侵入人体区域并遮挡人体；
- 单纯提高人体 LBS 或场景点数并不能建立“脚应该贴地、身体不应穿墙”这种关系；
- 在粗对齐条件下，人体 root translation/scale 误差会进一步放大冲突。

所以问题不是“再加一个 MLP 提升 PSNR”，而是：**如何从人体当前的可解释语义部位出发，获得与其有关的场景局部证据，并稳定地修正人体表示。**

### 1.2 表示与基本公式

人体和场景的分工：

```text
Canonical human Gaussians G_h^c
  -- SMPL pose/root/scale + LBS --> posed human Gaussians G_h(t)
  -- Anchor Attention correction --> G'_h(t)

Static scene Gaussians G_s
  --------------------------------> G_s

G'_h(t) + G_s -- Gaussian rasterizer --> image_t
```

人体 canonical Gaussian 在所有帧是同一套点及其 appearance/geometry 参数；时间变化主要来自 SMPL 的 pose、root translation、scale，经 LBS 变形得到。训练期间可以 densify/prune，但训练结束后推理时点数和 canonical 参数不随时间改变。

对第 `i` 个人体 Gaussian，核心位置关系可写为：

```text
x_i(t) = LBS(x_i^c; theta_t, beta, root_t, scale_t)
x'_i(t) = x_i(t) + gamma_mu * Delta_mu_i(t)
                    + gamma_transl(t) * Delta_root(t)
```

其中 `Delta_mu` 和 `Delta_root` 不是自由逐帧点优化，而是由人体锚点与 scene Gaussian 局部特征计算得到的前馈 correction。

### 1.3 Anchor Attention：为什么从锚点出发

我们从 SMPL mesh 选取 16 个有稳定语义的锚点（脚、手、肩、腰等）。对每个锚点：

1. 根据当前 LBS 后的人体 Gaussian，求该锚点的世界坐标；
2. 在 scene Gaussian 中查询 top-K 空间邻居（默认 `K=32`）；
3. 将场景点的相对位置、尺度、opacity、外观等编码为 token；
4. 人体 anchor token 作为 query，场景 token 作为 key/value 做 cross-attention；
5. 将 anchor context 按 Gaussian 的 anchor binding/weights 回传到每个 human Gaussian；
6. 输出 `Delta_mu`（局部修正）与 `Delta_root`（整体平移修正）。

因此 attention 的意义不是“Transformer 很强”，而是建立了一个明确归纳偏置：**一个脚附近的修正主要参考脚附近的场景，而不是用全局 pooled feature 猜。**

### 1.4 为什么人体不直接逐帧独立 Gaussian 优化

面试中容易被问“为什么不让每帧各有一套人体 Gaussian”。回答是：

- 独立逐帧点集容易获得表观拟合，却丢失 identity consistency 和拓扑对应；
- 点的数量、排序和语义会漂移，难做 LBS、压缩、实时传输和交互控制；
- 逐帧大规模优化慢，且会把图像噪声写入几何；
- canonical 点 + LBS 将高维动态问题压到 SMPL 的低维状态，Anchor Attention 只补偿 LBS/对齐无法解释的局部残差。

这也是研究点二可以只传/修正低维 state 的基础。

### 1.5 训练路线与关键设计选择

#### GT 对齐下的当前最佳路线（论文主表 Ours）

这是应优先介绍的结果：`depth_sup12k + AnchorAttention_6k`，两阶段。

| 阶段 | 做什么 | 目的 |
|---|---|---|
| Stage 1，约12k步 | HUGS 人—场训练 + mono depth supervision；Attention 关闭 | 先稳定学习 scene/human 的基础外观和几何 |
| Stage 2，约6k步 | 从 Stage 1 热启动；Anchor Attention 从 step 0 开启，warmup 1000步 | 在已有稳定表征上学习人—场局部 correction |

关键超参：`human_pooling=attention`、`scene_topk=32`、`lbs_w=1000`、depth weight `0.05`。Stage 2 采用 warmup，避免初始 correction 破坏已训练的 Gaussian。

**lab 最佳 run：**

```text
output/human_scene/neuman/lab/hugs_trimlp/
depth_sup12k_transl_xyz_attn_6000_lab_20260528/2026-05-28_23-44-41/
```

其 HUMAN PSNR 为 **19.7831**。这是当前 lab 的 GT 对齐最优 Ours；不要误称为旧 ADC 19.5512，也不要用 VIMO-v4 的 lab 17.7588 代替。

#### 粗对齐下的当前最佳路线

`v4_correct_inline_attn_18k` 是 VIMO v4 粗对齐的单阶段最优路线：18k 步，attention 在 step 2000 开始、500 步 warmup。六场景 final HUMAN PSNR 平均 **17.2944**，相对 STM **+0.39 dB**。

它回答的是更困难、更接近自动部署的问题：当 root translation/scale 只能粗估时，Anchor Attention 是否仍能修复人—场关系。

### 1.6 结果如何讲

GT 对齐 Ours 在 NeuMan 6 场景的 HUMAN PSNR 平均 **19.6174**，高于 STM 的 **19.4725**。代表性结果：lab 19.7831、bike 20.4297、seattle 19.6412、parkinglot 19.8686、citron 20.1794。

粗对齐 Ours 在 6 场景平均 17.2944，STM 16.9020；同时报告全图/人体指标，避免只挑 crop PSNR。

面试时应强调：

- HUMAN PSNR 是核心，因为问题发生在人体—场交互区域；
- 同时保留全图 PSNR/SSIM/LPIPS，防止以牺牲背景换人体提升；
- GT 对齐与粗对齐是两套实验条件，不能混合比较；
- lab 的旧 ADC 19.5512 是历史强基线，但当前 GT 最优是两阶段 depth+attention 的 19.7831。

### 1.7 失败分析比“成功”更能体现深度

可以主动讲三个关键负结论：

1. **Global Scene Gate 不足以解决主问题。** opacity gate 在 bike 基本无增益，说明仅抑制 opacity 不能代替几何关系建模。
2. **错误的 scene densification 梯度会造成灾难。** scene densification 若只拿 scene 部分梯度，会让 scene GS 在人体区域无约束增殖；恢复原版 full `viewspace_points` 的梯度流后，训练曲线回到成功结果。这说明实现细节与表示竞争强相关。
3. **粗对齐失败不能全怪 attention。** VIMO 的 pose 本身可正确，但 root translation/scale 误差会达到数个身体高度；这种超出 correction receptive range 的误差必须靠对齐/初始化改善，不能硬靠网络修正。

### 1.8 高帧率/新视角渲染如何解释

因为人体是 canonical Gaussian + LBS，时序插帧应在结构化状态空间完成：

- SMPL joint rotation 用 quaternion SLERP；
- root translation、shape/scale 线性插值；
- 每个中间时刻重新执行 LBS 和 Anchor Attention；
- GT camera 的 rotation/center 同样插值，否则静态背景会跳帧；
- 若 attention 有离散 frame embedding，必须对相邻 correction 连续混合，不能直接切到 nearest frame。

这不是“把 10 FPS 图片重复编码成 30 FPS”，而是重算中间时刻的 3D human state 和 camera state。小幅 novel view 则以当帧人体中心做 look-at target，场景保持原坐标。

---

## 2. 研究点二：前馈 Gaussian 的不确定性感知接触一致性层

### 2.1 问题与创新位置

研究点一主要是离线优化后的高质量重建；研究点二针对的是在线前馈系统：

```text
RGB -> feed-forward human/scene reconstruction -> fast render
```

这类系统的短板不是“缺少一套 Gaussian”，而是人和场景通常独立预测：脚看起来落地，3D 中却会滑动/穿透；scene Gaussian 能渲染，未必是可用于接触的真实 surface。

目标是增加一个轻量、因果、安全的中间层：

```text
I_t + model state_{t-1}
  -> frozen human/scene Gaussian front-end
  -> RGB-mesh-point local relation encoder
  -> contact + proximity + geometry reliability
  -> causal controller
  -> bounded root/foot correction, uncertainty/abstain
  -> corrected SMPL-X -> LBS -> human Gaussian
  -> static scene Gaussian + renderer
```

关键创新不是再训练一个全局 reconstruction backbone，而是：

1. 将**接触**表述为局部 mesh—scene relation 与时序控制问题；
2. 只修正低维 SMPL-X root/foot，保留 canonical human Gaussian 的一致性；
3. 不把 Gaussian center 当表面；通过 local proxy/point patch 表达 surface evidence；
4. 几何不可靠时显式 abstain/reset，避免“错误修正比不修正更糟”。

### 2.2 GUSH3R 前端的准确分工：Human3R 先验，不是外接 HumanGS

研究点二的当前前馈候选是 GUSH3R。这里必须把三个名称分清：

```text
Human3R（已有 foundation backbone，GUSH3R 冻结使用）
  -> human token、image token、SMPL-X mesh、scene point cloud

GUSH3R（作者自主提出的两个 decoder）
  -> Scene Gaussian Decoder：scene point cloud + image token -> static scene Gaussians
  -> Human Gaussian Decoder / HGT：SMPL-X mesh + human token + image token -> dynamic human Gaussians

LHM / 其他同名 HumanGS 工作（外部已有工作）
  -> GUSH3R 论文中的分解式 baseline 或相关工作，不是 GUSH3R 主模型的 Human Decoder
```

GUSH3R 论文 Sec.3.1 明确称 Scene Gaussian Decoder 与 Human Gaussian Decoder 为其新引入的两条分支。官方代码将后者实现为 `HumanGSHead`，但这是**代码类名**，不应误说成“GUSH3R 调用了外部 HumanGS”。论文里真正作为外部人体重建基线的是 `AnySplat + LHM + Human3R`：先用 Human3R 估计人体 mask/SMPL-X，再由 LHM 重建人、AnySplat 重建背景，最后做后处理拼接；GUSH3R 的意义正是避免这种分离拼接。

Human Gaussian Decoder 的内部过程是：以 SMPL-X mesh 的固定语义表面点作 anchor，在 canonical body space 建立 vertex token；Human Gaussian Transformer（HGT）以 human token、vertex token 和按人维护的 appearance-memory token 为 query，以当前 image token 为 key/value；最后预测每个 anchor 周围 Gaussian 的 offset、scale、rotation、opacity 和颜色。SMPL-X/LBS 将这些 body-bound points 变到当前姿态，Gaussian renderer 输出人体 RGB 和 alpha mask。该 alpha mask 经 dilation 后用于排除背景候选，因此人景分离由 GUSH3R 内部完成，不依赖部署时的外部 GT segmentation。

这也界定了我们与 GUSH3R 的关系：我们冻结其 Human3R backbone 与两条 GUSH3R decoder，把 `SMPL-X + human Gaussian + scene Gaussian + image/human token` 当作粗前端；我们的贡献从这里开始，负责接触关系、因果状态、低维修正和安全回退。它不是重新训练 Human Gaussian Decoder。

要落地 `corrected SMPL-X -> human Gaussian`，还需要导出每个 Human Gaussian 对应的 SMPL-X query point / LBS transform 或建立等价 binding。GUSH3R 当前 inference export 已给出 Gaussian 与 SMPL-X，但没有把这一 binding 作为稳定外部接口导出；因此它是 E4 的明确工程任务，不能说已经完成。

#### 2.2.1 GUSH3R 的完整前向图

```text
causal RGB video frame I_t
  -> frozen Human3R recurrent foundation model
       -> camera pose T_t, scene point map X_t
       -> image token F'_t
       -> detected people: human token h_t,k, SMPL-X parameters and mesh V_t,k
       -> shared recurrent state

  -> GUSH3R Scene Gaussian Decoder
       X_t as Gaussian centers
       + DPT-decoded image token + CNN raw-image feature
       -> per-pixel {opacity, rotation, scale, color}
       -> confidence + predicted human-region filtering
       -> causal voxel aggregation with G^S_(t-1)
       -> static scene Gaussian map G^S_t

  -> GUSH3R Human Gaussian Decoder, for each person k
       canonical SMPL-X sampled anchors + h_t,k + F'_t + memory m_k
       -> Human Gaussian Transformer (HGT)
       -> per-anchor {offset, opacity, rotation, scale, color}
       -> SMPL-X LBS to posed space
       -> dynamic human Gaussians G^H_(t,k) + alpha mask

  -> merge G^S_t and all G^H_(t,k) in the same metric frame
       -> Gaussian rasterizer -> final render
```

官方实现中，Human Gaussian Decoder 并不在每个 SMPL-X 的 10,475 个顶点上都直接放一个 token；它把 canonical mesh 采样为 `1000` 个空间 group、每 group `10` 个 query point，即约 `10,000` 个人体 Gaussian query。每个 group 先形成 vertex/point embedding。HGT 是 4 层、512 hidden、8 heads 的 cross-attention：`human token + 1000 group token + optional person memory token` 作 query，当前帧 image token 作 key/value。group feature 被复制给 group 内 query point，再和 point positional feature送入 Gaussian MLP，预测 offset、opacity、rotation、scale、颜色。实际姿态由 SMPL-X LBS 施加到 canonical A-pose Gaussian 上。

这里有三种时序状态，不能混为一谈：Human3R 的共享 recurrent state 用于视频几何；scene Gaussian map 以 voxelization 只积累到当前帧；每个人还有按 SMPL-X matching 对应的 appearance-memory token，用于衣服/外观在遮挡和视角变化下保持一致。它们都不是接触状态。

#### 2.2.2 Human Decoder 如何训练

训练时冻结整个 Human3R foundation backbone，只训练 GUSH3R 新增的 Scene Gaussian Decoder 和 Human Gaussian Decoder，且两条 decoder **分开训练**。因此 GUSH3R 的“统一”主要是共享冻结几何先验、坐标系和最终 Gaussian 渲染表示，不是以人—场接触损失端到端联合训练。

| 分支 | 训练数据 | 监督与损失 | 论文训练配置 |
|---|---|---|---|
| Scene Gaussian Decoder | BEDLAM；另加 DL3DV 提升真实场景泛化 | 仅 GT background mask 区域：RGB MSE、LPIPS、GT depth、Gaussian scale anisotropy regularizer | 100k iterations，batch 2，A100 80GB，约1天 |
| Human Gaussian Decoder | BEDLAM；另加 Motion-X++ 提升真实人体动作/外观泛化 | human RGB MSE、human silhouette BCE、完整图+upper-body+face crop 的 partial LPIPS、Gaussian shape regularizer | 150k iterations，batch 1，A100 40GB，约2天 |

两支训练都输入顺序图像，长边缩放至 512。附录给出的权重为：scene `MSE=1, LPIPS=.2, depth=.1, regularizer=.05`；human `MSE=1, partial-LPIPS=.5, silhouette=1, regularizer=100`。人分支使用 silhouette loss，说明它学习人体覆盖范围；scene 分支用 GT background mask，只在背景处计算 RGB/depth 监督，避免把动态人写入静态 scene Gaussian。

#### 2.2.3 GUSH3R 内部“人—场关系”到底是什么

它存在三层关系，但没有接触关系：

1. **共享几何先验。** Human3R 同时预测相机、metric point map 和 SMPL-X；因此人体与场景理论上被放进同一坐标系。
2. **分离与合成。** scene decoder 用人体 detection score / human mask 过滤人体区域，human decoder 用 SMPL-X anchors 表示动态人；二者生成 Gaussian 后在同一坐标系合成渲染。
3. **图像级弱耦合。** HGT 的 key/value 是全图 image token，人体外观可以间接利用图像中的环境线索；但它不查询 scene Gaussian、scene surface、SDF 或接触标签。

因此 GUSH3R 没有显式的 foot-ground distance、penetration penalty、contact probability、contact episode memory 或接触期间零切向速度约束。它优化的是渲染质量、depth、silhouette、Gaussian shape 与外观时序一致性。视觉上“像落地”不代表几何上满足接触。这也是当前 GUSH3R 在 PROX held-out 接触尺度门槛失败后，不能直接作为接触 teacher 的根本原因。

研究点二应把接触层放在这个缺口上：从 GUSH3R 输出的 `SMPL-X + local scene Gaussian proxy + image/human token` 构造 AnchorAttention-style implicit interaction field，估计 contact/proximity/reliability，再用因果状态输出低维修正。这样使用 GUSH3R 的重建能力，但不把其渲染 alpha 或 Gaussian center 错当成物理接触。

#### 2.2.4 从 Human3R 到 GUSH3R：完整技术路线（学习与面试版）

这一条路线要先分清“**几何理解**”与“**可渲染表示**”。Human3R 解决前者：从单目视频在线恢复相机、米制场景点图和人体 SMPL-X；它输出的是几何状态，不是照片级 Gaussian 渲染器。GUSH3R 在冻结的 Human3R 几何状态上增加两个 Gaussian decoder，才把“人 + 场景”的结构化几何变为可以实时渲染的表示。

```text
单目、按时间到达的 RGB 流 I_1, I_2, ..., I_t
        │
        ├─ Human3R：在线几何底座
        │    ├─ CUT3R recurrent scene reconstruction：相机 + point map + scene memory
        │    └─ Multi-HMR human prior + human prompt：多人 SMPL-X、mask、track
        │
        ├─ GUSH3R Scene Gaussian Decoder：静态背景的 Gaussian map
        ├─ GUSH3R Human Gaussian Decoder / HGT：随 SMPL-X 运动的人体 Gaussian
        │
        └─ 同一米制坐标系合成 → Gaussian rasterizer → RGB / alpha / depth render

在我们的工作中：上述全部冻结为 coarse front-end；
人—场局部 relation/contact controller 再修正 SMPL-X，最后通过 LBS 更新人体 Gaussian。
```

##### A. Human3R：为什么它是 GUSH3R 的必要前端

普通单帧 HMR 只会给出“相机坐标里的一个人”。这不足以做 human-centered reconstruction：当相机自己在动时，不知道脚是否真的落在同一块地面，也无法累积场景。Human3R 的目标是对每一帧同时给出三类量：

| 输出 | 记号 | 用途 | GUSH3R 怎样使用 |
|---|---|---|---|
| 相机位姿 | `T_t` | 将当前观测放回全局米制坐标 | scene/human Gaussian 合成和渲染相机 |
| metric point map | `X_t` | 每个像素对应的 3D 场景点及置信度 | scene Gaussian 的初始中心/几何证据 |
| 多人 SMPL-X + human token | `Y_t,k`, `h_t,k` | 人的姿态、形状、全局 root 和外观/身份语义 | human Gaussian 的 canonical anchor、HGT 条件和跨帧身份 |

它建立在 CUT3R 的在线重建机制上。当前 RGB 的 ViT patch token 与一个固定长度的 recurrent scene state 交互；该 state 汇总过去帧但不保存无限长图像序列。decoder 更新 state 并读出当前帧的 image token、camera token、world metric point map 和 camera pose。因此它是 **causal online**：到 `t` 时只使用 `I_1...I_t`，不是把整段视频拿到后再做 batch optimization。

可以把每一帧的几何主干理解为：

```text
I_t
  → CUT3R image tokenizer → F_t
  → F_t 与 S_(t-1) 交互，两个 transformer decoder 更新 S_t
  → {refined image token F'_t, camera token z'_t}
  → {metric point map X_t^cam / X_t^world, confidence, camera pose T_t}
```

这里的 “point map” 是带像素对应关系的三维点图，不是已经清理、闭合的 scene mesh，也不是 Gaussian surface；这个区别对后面的接触问题非常重要。

##### B. Human3R 人体分支：Multi-HMR 不是简单外接检测器

Human3R 用 CUT3R 的 3D/时序表征，但单靠通用 scene token 不擅长精细人体。它因而引入冻结的 Multi-HMR ViT-DINO encoder 作为 human-specific prior。流程是 bottom-up、多人的，而不是对每个人裁一个图再跑一次 HMR：

```text
F'_t 的每个 patch
  → head detection score
  → 检出的 head patch 位置 u_(t,k)
  → 取该位置的 CUT3R token + 同位置的 Multi-HMR token
  → projection MLP → human prompt H_(t,k)
  → human prompt 与全图 image token self-attention
  → human prompt 与 recurrent scene state cross-attention
  → refined human token h_(t,k)
  → human MLP → SMPL-X 参数、local camera/root transform
  → 用 T_t 变换到 world frame → 10475-vertex SMPL-X mesh V_(t,k)
```

human prompt 的意义是把“这个空间位置是某一个人”显式写入已有的 scene-memory 推理，而不是把 Human3R 粗暴地拆成 CUT3R 和 Multi-HMR 的后处理拼接。它让人体 root、相机和 surrounding point map 在同一世界坐标假设下被预测。人 token 的跨帧特征匹配（Sinkhorn/optimal transport，含 dustbin 处理新出现或消失的人）给出 track ID；该 ID 再让下游的人体 appearance memory 不会把不同人混在一起。

**mask 从哪里来？** Human3R 不是在部署时另跑 SAM。它将 patch-level image feature 送入 MLP、sigmoid 和 PixelShuffle，预测 dense human mask；该 mask/检测分数能帮助把动态人体从 background reconstruction 中排除。它是视觉分离线索，不是三维接触标签，也不能据此推出地面接触。

##### C. Human3R 怎样训练，以及它留下什么局限

Human3R 在 BEDLAM 上做人类提示微调。BEDLAM 约有 6k 序列，提供世界坐标的 scene depth、camera pose 与多人的 SMPL-X。训练目标同时守住通用 3D 能力和人体能力：

```text
L_Human3R = L_confident-pointmap + L_camera-pose
           + L_head-detection(BCE)
           + L_SMPL-parameter(L1) + L_mesh + L_reprojection
```

Multi-HMR encoder 冻结，CUT3R 主干主要保持其已有的场景时空先验，只对 human-prompt 相关部分做高效微调；论文报告单张 48GB GPU 约一天。这里的监督教会它“几何、相机和人体在共同世界系中一致”，但没有 RGB render loss 来学习 Gaussian 外观，也没有 contact/penetration/foot-lock loss。

应主动说明它的边界：recurrent state 的训练上下文较短，超长序列会出现 memory forgetting；严重遮挡时若两人落在同一个 head token，身份区分会退化。更关键的是，我们的 PROX held-out 审计中 Human3R 的几何误差 median 为 7.37 cm，仍大于约 2 cm 的接触尺度。因此“世界系一致”是正确的系统接口，不等于已经达到可直接做物理接触的精度。

##### D. 由 Human3R 过渡到 GUSH3R：冻结、分叉，而非重训一个端到端 backbone

GUSH3R 不重新学习相机、人体姿态和稠密场景几何。它冻结 Human3R，把其输出看作具语义与米制坐标的条件输入，再训练两个相互独立的解码器：一个负责不随人体运动的 scene Gaussian，一个负责可由 SMPL-X 驱动的 human Gaussian。

这一步的设计选择很关键：

| 若直接从 RGB 预测全部 Gaussian | GUSH3R 的选择 |
|---|---|
| 人体和背景会竞争同一批点，动态人容易写进静态地图 | 按 human mask/detection 分离背景，scene map 只累积静态证据 |
| 人体点没有稳定对应，跨帧外观和动作容易漂移 | 在 canonical SMPL-X body space 放置语义一致的 query，再由 LBS 变形 |
| 需要为每帧重新推断人体几何 | 复用 Human3R 的 pose/shape/root，只预测体表附近的 Gaussian 属性 |

##### E. Scene Gaussian Decoder：从点图到可积累的静态背景

scene branch 的原理可分四步：

```text
Human3R point map X_t（每个可信像素的三维点）
  + DPT 解码的 F'_t（高层视觉特征）
  + raw RGB CNN feature（局部纹理）
  → MLP 预测每个候选点的 {opacity, rotation, scale, color}
  → 去除低置信点和 human-region 候选
  → 将新 Gaussian 与历史 G^S_(t-1) 按 voxel 聚合
  → static causal scene map G^S_t
```

point map 直接给中心，使 decoder 不必从零猜三维位置；网络重点补全 Gaussian 的体积、方向、透明度和颜色。voxel aggregation 是在线地图维护，不是全序列 bundle adjustment：重访的静态区域可累积，人体区域被过滤后不应固化到背景。它对动态场景的假设是“背景近似静态”；移动椅子、镜子、强反光、持续遮挡仍会造成失败。

scene training 用 BEDLAM，并加 DL3DV 改善真实场景泛化。只在 GT background mask 内计算 RGB MSE、LPIPS 与 GT depth，另加 scale anisotropy regularizer；论文配置为 100k iterations、batch 2、A100 80GB 约一天。这解释了为什么它有比较好的背景 render，却没有从损失中学习人—场接触。

##### F. Human Gaussian Decoder / HGT：从 SMPL-X 到动态人体 Gaussian

人体分支绝不是“把 SMPL-X mesh 原样渲染”。SMPL-X 只给出可控的 body coordinate system，HGT 学的是衣物、头发、人体表面外观和相对 body surface 的微小几何残差。其结构为：

```text
canonical A-pose SMPL-X surface
  → 1000 spatial groups × 10 query points/group ≈ 10000 body-bound queries
  + person token h_(t,k)
  + current full-image token F'_t
  + this person's appearance-memory token m_(t-1,k)
  → 4-layer HGT cross-attention (hidden 512, 8 heads)
  → 每个 query 的 {offset, opacity, rotation, scale, color}
  → canonical human Gaussians
  → SMPL-X LBS(theta_t,k, beta_k, root_t,k)
  → posed human Gaussians G^H_(t,k) 与 alpha mask
```

更精确地说，human/group/optional-memory token 构成 query，当前帧完整 image tokens 是 key/value；每个 group 的 context 复制给该组的 10 个 point query，再与其 canonical positional feature 一起送入 Gaussian attribute head。这样既能让衣服颜色参考当前图像，也保留“某个 Gaussian 属于脚、手或躯干哪个语义部位”的固定绑定。appearance memory 是按 track ID 维护的外观状态，解决遮挡和视角变化下的纹理不稳定；它不等价于 Human3R 的 shared scene memory，更不等价于我们的 contact state。

训练时 Human3R 仍冻结。human decoder 在 BEDLAM + Motion-X++ 上单独训练，以人体 RGB MSE、silhouette BCE、full/upper-body/face crop 的 partial LPIPS 及 Gaussian shape regularizer 为损失；论文配置为 150k iterations、batch 1、A100 40GB 约两天。silhouette loss 只约束人体投影轮廓，不会自动产生可靠的脚—地面距离或阻止三维穿透。

##### G. 最终合成与“人—场关系”的强弱

最后两类 Gaussian 已处于同一个 Human3R metric frame：

```text
G^S_t（静态背景） + Σ_k G^H_(t,k)（当前姿态的人）
      → Gaussian rasterizer(camera T_t) → rendered RGB, alpha, depth
```

这使 GUSH3R 比 `AnySplat + LHM + Human3R` 的后处理拼接更统一，但“统一坐标 + 合成渲染”不等于“理解 interaction”。它有三种真实耦合：共享 world coordinate、human mask 对 scene branch 的显式排除、HGT 对全图 image token 的弱上下文查询；它**没有** human Gaussian 到 scene Gaussian 的显式 cross-attention，也没有 SDF、最小表面距离、穿透、接触概率、接触持续状态或零滑动约束。

所以面试中可用一句话区分：

> Human3R 保证“人、相机、点图有机会在同一世界系说话”；GUSH3R 保证“这些结构可以被拆成人体与背景 Gaussian 并渲染”；我们的第二研究点才要解决“在这个不完美前端下，接触是否真实、是否持续、以及不可靠时是否应拒绝修正”。

##### H. 与我们接触层的精确接口：已有、待实现、不能夸大

| 接口项 | 当前状态 | 我们怎样用 | 不能误说成 |
|---|---|---|---|
| `SMPL-X / mesh / root / track` | GUSH3R 前端已有 | foot/hand ROI、低维 root/ankle/foot correction | 已有接触真值 |
| `F'_t, h_t,k, m_t,k` | 前端内部语义/外观特征 | 输入 AnchorAttention-style implicit relation field | 已做 scene-contact attention |
| scene GS 的 center/covariance/opacity/color | 可作局部 evidence | 构造 reliability-aware local proxy，并与 point patch/RGB 一起输入 | Gaussian center 就是零厚度 scene surface |
| `corrected SMPL-X → G^H` | 原理上应由 LBS 支持 | 纠正后重算 human GS 的 posed position/orientation | 当前 export 已稳定提供 query-to-LBS binding |

最后一行是当前 E4 的具体工程缺口：必须在 GUSH3R decoder/export 中保存每个 human Gaussian 的 canonical query point、skin/LBS weights 或等价的 per-Gaussian transform。拿到 binding 后，controller 修正 `theta/root` 才能以一次 LBS 前向更新人体 Gaussian，而不是把数万 Gaussian 当自由点硬移动。现有 inference export 虽有 Gaussian 与 SMPL-X，但没有把此 binding 作为稳定接口，故只能把它写为待实现 adapter。

当前还存在更早的科学门槛：GUSH3R→PROXD held-out coarse SMPL-X 误差 median 19.95 cm、p95 54.69 cm，远大于接触约 2 cm 的尺度。这不否定它作为 render/coarse-evidence front-end 的价值，但意味着接触头不能把它当作“已经准确的三维表面 teacher”。因此实验路径必须先用 oracle SDF 证明 controller 机制，再做退化鲁棒性和真实前端几何契约；不可用漂亮 render 替代接触精度证明。

##### I. 一分钟面试表述

> 我们把系统拆成两层。底层 Human3R 是一个因果的 video geometry foundation model：CUT3R 的 recurrent memory 输出相机和 metric point map，Multi-HMR 的 human prompt 从同一状态中读出多人的 world-space SMPL-X、mask 和 track。它解决的是“人和场景如何在同一坐标系被理解”，但不能直接给照片级渲染。GUSH3R 冻结它，再分出两个 decoder：scene decoder 用 point map 加图像特征预测并 voxel-累积静态 Gaussian；human decoder 在 canonical SMPL-X 上布置约一万个 body-bound query，用 HGT 从全图与外观 memory 预测 Gaussian 属性，再用 LBS 跟随姿态运动。两类 Gaussian 最后在同一坐标渲染。现有方法的缺口是：它只做分离和合成，没有显式人—场接触约束。我们的工作不重训整个 renderer，而是在其 SMPL-X、local scene evidence 与 token 接口上增加 AnchorAttention-style interaction field 和因果低维控制；只修 root/foot/ankle，再经 LBS 更新人体 Gaussian，并在前端几何不可靠时主动 abstain。

### 2.3 为什么不直接优化所有 Gaussian

逐帧移动数十万 human/scene Gaussian 的问题：高延迟、易漂移、不可解释、破坏场景静态性，而且很难保证接触期足端速度为零。

我们的可控变量是约 20–40 维：左右脚 contact、切向 action、root/ankle/foot residual、uncertainty/reset。执行时 action 被投影到局部切平面，并做 `tanh` 限幅。场景 GS 不随接触模块移动；纠正只经 SMPL-X/LBS 传给人体 GS。

### 2.4 接触状态与因果控制

每只脚维护一个 contact episode anchor：脚首次可靠接触时记录世界坐标 anchor；持续接触时，预测切向 action 使脚相对该 anchor 的速度降低；离开时释放 anchor。

部署状态仅包含模型自身量：上一帧接触概率、已施加 action、episode anchor、uncertainty/reset。**绝不能使用上一帧 teacher residual**，否则训练/测试会发生 oracle leakage。

控制损失可概括为：

```text
L = contact BCE
  + anchor position loss
  + contact-period stick/velocity loss
  + non-contact release loss
  + action slew regularization
  + uncertainty calibration (E2 onward)
```

这解释了为什么只做 penetration clearance 不够：法向修正可把脚推出地面，但不能保证接触期间的切向零速度。

### 2.5 分阶段技术路线

| 阶段 | 要验证的命题 | 数据/输入 | 前进门槛 |
|---|---|---|---|
| E0 | 坐标、SDF、脚 anchor 是否可信 | PROXD + PROX SDF/scene scan | 坐标/单位/法向/覆盖审计通过 |
| E1 | 接触控制机制是否成立 | Oracle PROXD + Oracle SDF + 注入 drift | penetration、sliding、jitter 同时改善 |
| E2 | 局部场景退化时能否安全 | point-map/proxy 噪声、normal误差、dropout、delay | uncertainty/abstain 不让风险显著恶化 |
| E3 | 真实人体前端能否接入 | Human3R 或其他前馈 human state | held-out 几何到接触尺度（约2cm） |
| E4 | Gaussian/scene-state 接入是否不伤视觉 | frozen controller + local Gaussian proxy | contact ROI 几何改善且 render/flicker/latency 达标 |

这套路线的好处是可证伪：前端不达门槛就停在 diagnostic，而不是用 oracle 标签伪造在线结果。

### 2.6 已完成的、可以讲的实证结果

以下都应明确标为 **oracle-scene mechanism result**：使用 PROXD + 官方 PROX SDF，人工注入 root/foot drift；不是 Human3R/GUSH3R 的真实在线结果。

1. **E0 契约已通过。** 四个 PROX 序列、720 帧，SDF、法向、坐标系 `PROX_scene_world` 有效。
2. **法向 clearance 只能部分解决问题。** GRU 将 penetration depth 从 18.57 mm 降至 13.95 mm、contact |distance| 从 29.28 mm 降至 24.58 mm，但 sliding 从 0.643 升到 0.927 m/s。结论：法向 clearance 与 tangential stick 必须分开。
3. **contact anchor 表示成立。** oracle anchor lock 可将 sliding 从 0.385 m/s 降至 0.028 m/s，证明接触 episode anchor + foot-lock 的执行表达正确。
4. **因果 rollout 机制通过。** 使用模型自身状态、切平面 action、bounded soft gate 的 control-first checkpoint，把 sliding 从 0.385 降至 **0.218 m/s（-43.4%）**，contact F1=0.817。state-first checkpoint 的 contact F1=0.863、transition F1=0.550，但 sliding=0.281 m/s。应报告 Pareto，而不是 cherry-pick 单一“最好”。
5. **中等场景退化暴露关键风险。** 冻结 controller 在 normal 18°、distance noise 2cm、bias 1cm、20% dropout、2-frame delay 条件下，sliding 恶化到 0.463 m/s；简单 threshold 只有 100% abstain 才能回到基线。几何增强能将 sliding 降到 0.276 m/s，但 F1=0.722、transition F1=0.142，仍不够安全。
6. **连续 proximity 是比伪 dense binary contact 更可靠的 Stage-A 目标。** local relation encoder 在严格 split 上 signed proximity MAE=**1.27 mm**，好于 constant-zero 的 1.79 mm 和 nearest-point 的 12.64 mm；但 ROI contact head all-positive，不能拿 F1 当成功结果。

### 2.7 为什么现在不能说“GUSH3R 接触系统已完成”

这是最重要的诚实边界。

- Human3R 到 PROXD 的 held-out Sim(3) 几何误差：median **7.37 cm**、p95 **45.1 cm**；
- GUSH3R coarse SMPL-X 到 PROXD 的 held-out 误差：median **19.95 cm**、p95 **54.69 cm**；
- 接触所需几何尺度约为 2 cm。

因此当前 GUSH3R 可以提供 per-frame Gaussian evidence（center/covariance/opacity）和 renderer/degraded-map diagnostic，但不能作为可靠 surface、contact teacher 或真实前端主表。正确的表述是：

> 已完成前馈 Gaussian 接口与退化诊断；已验证 oracle/degraded geometry 下的接触机制；真实前端接入被 held-out 几何门槛明确阻断，下一步需要更强前端或完成 camera-local/reprojection 误差契约。

这比忽略误差、宣称“前馈 Gaussian 已实现物理接触”更能体现研究严谨性。

### 2.8 研究点二最终评测表应如何设计

不能只报一个 PSNR 或 contact F1。建议至少四张表：

1. **Mechanism table：** oracle SDF + injected drift。contact/transition F1、sliding、penetration depth、contact distance。
2. **Robustness table：** E2 corruption。风险随 abstention 的 curve、coverage-risk、错误修正率、回退率。
3. **Real-front-end table：** 仅 E3 坐标门槛通过才出现。与原始 Human3R/GUSH3R、规则、学习方法比较。
4. **Renderer/system table：** contact ROI PSNR/SSIM/LPIPS、human Gaussian flicker、全图变化、FPS、latency、packet bytes。

---

## 3. 两个研究点的衔接

| 维度 | 研究点一 | 研究点二 |
|---|---|---|
| 核心目标 | 优化人—场重建与对齐 | 在线接触、时序与安全修正 |
| 人体表示 | canonical human GS + SMPL LBS | canonical human GS/SMPL-X + 低维 residual + LBS |
| 场景表示 | 静态 scene GS | 静态 scene GS / local proxy，不逐帧移动 |
| 关系建模 | semantic anchor -> local scene cross-attention | foot ROI mesh-point/Gaussian relation + causal state |
| 修正对象 | human Gaussian local/root correction | root/foot/ankle low-dimensional action |
| 不确定性 | 训练中 scene interaction 的稳定化 | 执行时决定 correction / abstain / reset |
| 最终愿景 | 高质量离线/可重渲人景重建 | 可实时部署的前馈接触一致性层 |

最自然的衔接叙事是：研究点一已经说明“人—场关系不能只靠独立表示，需要局部 interaction”；研究点二进一步将这种 interaction 从离线的 Gaussian correction 推向在线、可控、带不确定性门控的接触状态修正。

---

## 4. 高频追问与建议回答

### Q1：Anchor Attention 和普通 cross-attention 的区别？

不是对所有 human/scene token 做全局注意力。它以 SMPL 语义锚点为查询中心，先做空间邻域 top-K，再将 context 按 Gaussian—anchor binding 回传到点。计算上更局部，语义上更可解释，也更适合接触/遮挡问题。

### Q2：为什么 correction 不直接预测完整 pose 或所有 Gaussian offset？

完整 pose/root/foot 的分解存在 gauge ambiguity；所有 Gaussian offset 则高维、易漂移、慢。第一点中以 LBS 保留人体结构、attention 预测残差；第二点中先限制为可辨识的法向 clearance 与切向 foot-lock，再逐步加入 root/ankle IK。

### Q3：为什么 Global Scene Gate 不作为主贡献？

它是合理的 opacity suppression 对照，但 bike 的结果几乎不变，说明主矛盾不在 opacity gate 本身，而在初始化、densification、粗对齐和局部几何关系。负结果被保留以避免错误归因。

### Q4：如何证明不是仅靠更大模型提升？

第一点使用同一 HUGS human/scene 表示，与 STM 主要差别是 interaction mechanism；第二点用 oracle mechanism、退化鲁棒性、真实前端分别分表。消融围绕 correction 开关、anchor start/warmup、局部 proxy corruption、causal state、soft/hard gate，而非只比较参数量。

### Q5：为什么第二点现在不直接上端到端训练？

真实前端 human geometry 的 held-out 误差远大于接触尺度，端到端会把前端对齐错误、scene surface 错误和 controller 错误混在一起，得到不可解释的结果。先用 oracle 拆机制、再加退化、最后接真实前端，才知道失败来自哪里。

### Q6：Gaussian center 为什么不能当 surface？

Gaussian 是体积/辐射场元，不是零厚度 mesh surface。center 到人体的距离不等于接触距离，尤其在 opacity、scale、遮挡和视角变化下。因此第二点用 local point/normal/proximity proxy，并把 Gaussian 属性作为 evidence，不直接作为几何 teacher。

### Q7：怎样保证在线性？

不做每帧全 Gaussian 反优化；backbone 冻结，controller 是小型 causal GRU/TCN（H=5–8），输出几十维 low-dimensional state。场景不动、人体通过 LBS 更新；后续系统表会实测 FPS、端到端 latency 和 packet bytes，而不是只口头估算。

### Q8：第二点的创新和 IK/物理投影有什么区别？

纯 IK/投影只在给定可信接触表面和接触状态时求解约束。我们的重点在前端不可靠时：从 RGB/mesh/point relation 估计 contact、surface reliability 与 uncertainty，用 causal state 维持 contact episode，并在不可信时 abstain。IK 是可选执行后端，不是完整解决方案。

---

## 5. 面试展示建议

推荐按下面顺序讲，避免被带到尚未完成的第二点主表：

1. 一页系统总图：canonical human GS + static scene GS + interaction；
2. 研究点一：问题、Anchor Attention、两阶段训练、GT/粗对齐结果；
3. 一页失败归因：gate 无效、densification gradient、粗对齐上限；
4. 转场：离线优化虽好，前馈系统仍有接触和时序问题；
5. 研究点二：接口、低维控制、anchor foot-lock、uncertainty；
6. 一页已验证 mechanism/robustness 数字；
7. 一页严格边界：Human3R/GUSH3R held-out geometry 未过 2cm，真实前端表尚未声称完成；
8. 给出 E2→E4 的明确门槛与系统评测计划。

这套叙事的优势是既能展示已完成的 quantitative contribution，也能展示面对负结果时如何建立可证伪的研究路线。

---

## 6. 事实来源与继续维护

- 研究点一最优结果与路径：[BEST_PIPELINE.md](BEST_PIPELINE.md)
- 研究点一历史归因：[CLAUDE_SESSION_LOG.md](CLAUDE_SESSION_LOG.md)
- 研究点二实验协议：[forward_contact_pipeline/CONTACT_CORRECTION_EXPERIMENT_PLAN.md](forward_contact_pipeline/CONTACT_CORRECTION_EXPERIMENT_PLAN.md)
- 研究点二系统路线：[forward_contact_pipeline/OVERALL_EXECUTION_ROADMAP_20260909.md](forward_contact_pipeline/OVERALL_EXECUTION_ROADMAP_20260909.md)
- GUSH3R 门槛报告：[forward_contact_pipeline/reports/a3_gush3r_prox_feasibility_20260909.md](forward_contact_pipeline/reports/a3_gush3r_prox_feasibility_20260909.md)

更新本文件时，所有数字需先以对应报告/`results_train.json` 核验；尤其要保持“GT 对齐、粗对齐、oracle mechanism、proxy robustness、real front-end、renderer diagnostic”六种口径的分离。
