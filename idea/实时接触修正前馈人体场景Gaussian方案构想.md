# 实时接触修正的前馈人体-场景 Gaussian 重建方案

> 状态：Idea / 方案构想
>
> 更新日期：2026-07-29
>
> 目标：将第二个研究点从单纯的 Gaussian 传输压缩，升级为面向直播的“前馈重建 + 实时接触修正 + 低延迟传输”系统。

## 0. 最新方案定位与修订说明

### 0.1 主基线已经明确

经过对 Human3R、GUSH3R、JOSH/JOSH3R、GRAFT 和 CoGS 的进一步分析，本方案采用：

```text
Human3R：上游人体-场景几何模型
GUSH3R：主基线和最终高斯实验平台
我们的工作：GUSH3R 后的实时接触修正与低维传输层
```

Human3R 输出 SMPL-X、场景点图和相机等结构化几何，可以作为接触模块的**网格/点图预验证平台**，用于验证接触状态估计、局部场景表面提取和人体姿态修正是否成立。但 Human3R 不直接输出人体高斯，因此不能作为最终高斯接触效果的主实验平台。

GUSH3R 基于 Human3R 的人体网格和场景几何生成动态人体高斯与静态场景高斯，能够直接评价人体高斯与场景高斯的接触、穿透和渲染质量。因此，正式主实验必须比较：

```text
原始 GUSH3R
    vs.
GUSH3R + 我们的接触修正层
```

### 0.2 研究合理性

本方案延续了已有三维前馈研究的发展路线：

```text
MASt3R：三维匹配和几何特征
  -> Human3R：人体-场景点图、相机和 SMPL-X
  -> JOSH3R：人体-场景运动的前馈预测
  -> GRAFT：基于几何探针的快速人体-场景修正
  -> GUSH3R：前馈人体-场景高斯生成
  -> 我们的工作：前馈人体高斯上的显式实时接触修正
```

JOSH3R 和 GRAFT 说明人体-场景接触关系可以通过前馈或快速修正方式学习，但它们主要处理人体运动、人体网格或点图几何，没有解决“接触修正如何直接作用于前馈人体高斯并保持高斯形变一致性”的问题。

因此，本方案不是简单重复一个接触分类器，而是研究：

```text
局部人体-场景几何
  + 历史接触状态
  -> 接触状态和低维姿态/根节点残差
  -> SMPL-X + LBS 更新人体高斯
  -> 接触感知的实时传输
```

### 0.3 面向直播传输的必要性

人体-场景视频重建的应用目标包括远程人体直播、数字人传输、沉浸式交互和边缘渲染。这些场景要求低延迟、低带宽和连续运行。

迭代式接触优化虽然能够提高离线几何质量，但不适合直播：每个新场景都需要重复优化，接触状态切换时容易产生延迟，且优化过程和高维状态难以编码成稳定的数据流。我们的方案将接触信息压缩为：

```text
接触概率 + 接触目标 + 根节点/关节残差 + 不确定性
```

客户端只需保存基础人体/场景高斯，并利用 SMPL-X/LBS 执行低维修正。因此，实时接触修正同时是几何问题，也是人体-场景高斯传输中的必要系统组件。

### 0.4 研究范围调整

第一阶段只做脚-地面接触，采用“Human3R 预验证、GUSH3R 主实验”的路线。手-物体接触、坐姿接触和全身碰撞放到后续扩展，避免第一阶段同时引入物体检测、刚体运动和复杂物理动力学。

## 1. 一句话概述

现有的前馈人体-场景重建模型能够快速生成可渲染的 Gaussian 表示，但不能保证人体与场景之间的真实接触关系。我们提出一个轻量、因果、低维的实时接触修正模块：前馈模型负责生成场景和人体 Gaussian，接触网络根据人体运动、局部场景几何和历史状态预测接触关系，再通过 IK/姿态残差修正人体状态，客户端仅需接收低维修正参数即可保持接触一致性。

核心原则是：

```text
前馈模型负责“看起来是什么”
接触模块负责“应该如何贴合”
Gaussian renderer 负责“如何实时显示”
```

## 2. 研究背景与问题

### 2.1 前馈重建的优势

近期的 Human3R、GUSH3R 等方法将人体、相机和场景几何统一到一个前馈框架中，能够从单目视频快速输出人体和场景表示。GUSH3R 进一步使用两个 Gaussian decoder，将静态场景和动态人体直接转换为可渲染的 3D Gaussian。

这类方法适合直播和在线应用，因为它们不需要对每个场景执行数千到数万步的测试时优化。

### 2.2 现有前馈方法的不足

现有方法主要优化：

- 人体和场景的整体空间一致性；
- 图像重建质量；
- 新视角渲染质量；
- 在线速度。

但它们通常没有显式建模：

- 脚-地面接触；
- 手-物体接触；
- 臀部-椅面接触；
- 人体与场景的穿透；
- 接触状态的时间连续性。

因此可能出现：

- 脚部滑动；
- 人体浮空；
- 脚部或身体穿入地面；
- 手与物体有明显间隙；
- 接触状态在相邻帧之间闪烁。

### 2.3 为什么不能使用现有迭代式方案

我们当前 HUGS 管线中的 Anchor Attention 和 interaction correction 可以通过 Gaussian 位置修正改善人体-场景关系，但它们是在训练/优化过程中发挥作用的。直播场景要求：

- 每帧或每几帧快速输出结果；
- 不能对 50 万级人体 Gaussian 进行逐帧迭代优化；
- 接触状态变化时需要立即响应；
- 网络受限时仍能传输低维状态。

因此，需要将接触修正从“高维 Gaussian 优化”改造成“低维人体状态预测与投影”。

## 3. 研究目标

### 3.1 总体目标

构建一个针对单目直播视频的实时接触修正系统，在不进行逐帧 Gaussian 优化的情况下，保持人体与静态场景之间的接触一致性。

### 3.2 具体目标

1. 从前馈人体-场景 Gaussian 或场景 proxy 中实时估计接触状态。
2. 预测脚、手、臀部等接触区域的几何误差和穿透风险。
3. 通过低维 root/joint residual 和 IK 修正人体姿态。
4. 使用因果时序模型避免接触闪烁和脚部滑动。
5. 将接触修正编码为低维 packet，支持客户端本地 LBS 和 Gaussian 渲染。
6. 在 NeuMan、PROX、BEDLAM 和人体-物体交互数据上同时验证视觉质量、接触质量和实时性。

### 3.3 不作为第一阶段目标的内容

- 不实时优化全部 Gaussian 属性；
- 不立即联合微调 GUSH3R/Human3R backbone；
- 不一开始处理所有人体-物体动力学；
- 不将传输协议本身作为唯一创新点；
- 不把“视觉上贴合”直接等同于“物理上接触正确”。

## 4. 研究假设

### 假设 H1：接触修正可以在低维人体状态空间完成

人体 Gaussian 的动态位置主要由 SMPL/SMPL-X 姿态和 LBS 决定。对于脚-地面接触，修正 root translation、脚部相关关节旋转和少量 body pose residual，通常比逐点修正 Gaussian 更稳定、更低成本。

### 假设 H2：接触状态具有明显时序持续性

脚接触地面后通常持续多个帧，接触状态不会随机地每帧切换。历史姿态、anchor 速度和上一帧 contact state 可以有效降低接触抖动。

### 假设 H3：局部场景几何足以支持接触判断

接触估计不需要查询全部场景 Gaussian。人体 anchor 附近的局部 scene proxy、深度、法向和可见性信息，足以判断脚底/手部的距离、法向和穿透风险。

### 假设 H4：接触区域比全局场景更值得优先传输

在有限带宽下，人体本身和直接接触区域对用户感知更重要。将接触修正和局部场景块优先传输，能够比固定比例的人体/场景传输获得更好的 rate-distortion 性能。

## 5. 系统总体架构

```text
                 服务器端或直播采集端
┌──────────────────────────────────────────┐
│ 单目视频流                               │
│   ↓                                      │
│ 前馈人体-场景重建器                       │
│ (Human3R / GUSH3R / 现有前馈替代模块)     │
│   ├── camera pose                         │
│   ├── scene proxy / scene Gaussians       │
│   ├── SMPL/SMPL-X pose                    │
│   ├── human anchors                       │
│   └── human Gaussian appearance           │
│   ↓                                      │
│ Contact State Estimator                   │
│   ├── contact probability                 │
│   ├── contact type                        │
│   ├── local surface distance/normal       │
│   └── uncertainty                          │
│   ↓                                      │
│ Contact Correction Network                │
│   ├── root translation residual            │
│   ├── joint pose residual                 │
│   ├── contact point target                │
│   └── penetration correction              │
│   ↓                                      │
│ IK / geometric projection                 │
│   ↓                                      │
│ corrected SMPL/SMPL-X + contact packet    │
└──────────────────────────────────────────┘
                    │
                    │ low-dimensional stream
                    ↓
                 客户端
┌──────────────────────────────────────────┐
│ cached scene proxy / scene chunks         │
│ canonical human Gaussian + LBS            │
│ corrected pose and contact residual        │
│   ↓                                      │
│ LBS deformation + local Gaussian update   │
│   ↓                                      │
│ real-time Gaussian renderer               │
└──────────────────────────────────────────┘
```

## 6. 输入与输出定义

### 6.1 前馈重建器输出

每一帧至少需要：

- 相机内外参；
- 人体 SMPL/SMPL-X 参数；
- 人体 root translation；
- 人体语义 anchor 的世界坐标；
- 人体 anchor 速度；
- 人体 mask 或置信度；
- 局部场景点、Gaussian 或深度 proxy；
- 局部场景法向或平面估计；
- Gaussian/深度置信度。

在现有 HUGS 系统中，可以直接复用：

- canonical human Gaussian；
- LBS weights；
- SMPL pose；
- AnchorAttention 的语义 anchor；
- scene Gaussian KNN 查询；
- `contact_eval_lib.py` 的接触 ROI 评价接口。

### 6.2 接触模块输出

每个接触候选点输出：

- `p_contact`：接触概率；
- `contact_type`：接触类型；
- `d_surface`：到场景表面的距离；
- `n_surface`：场景表面法向；
- `penetration_risk`：穿透风险；
- `uncertainty`：预测不确定性。

人体修正输出：

- `delta_root_translation`；
- `delta_root_rotation`；
- `delta_body_pose`；
- `delta_hand_pose`，可选；
- `contact_target`；
- `contact_state`；
- `keyframe_required`，用于异常状态恢复。

## 7. 接触候选点设计

### 7.1 第一阶段：脚-地面接触

优先实现：

- 左脚底 anchor；
- 右脚底 anchor；
- 左脚踝；
- 右脚踝；
- 左右脚底的若干 SMPL 顶点区域。

脚-地面是最适合作为第一版的原因：

- 接触语义明确；
- 场景通常近似静态；
- 具有明显的距离和法向约束；
- NeuMan、PROX、BEDLAM 都容易构造标签；
- foot sliding 和 contact jitter 容易量化。

### 7.2 第二阶段：手-物体接触

扩展到：

- 左手掌；
- 右手掌；
- 手指/物体接触区域；
- 物体 Gaussian 或物体 proxy。

手-物体需要额外的物体检测、刚体估计或交互数据，不建议作为第一版的必需目标。

### 7.3 第三阶段：坐姿和身体接触

扩展到：

- 臀部-椅面；
- 膝盖-地面或物体；
- 背部-墙面。

该阶段需要更可靠的场景表面和人体接触标签。

## 8. 实时接触估计模块

### 8.1 几何特征

对每个 contact anchor，从局部场景 proxy 查询：

- 最近邻距离；
- trimmed mean 距离；
- 局部点云高度分布；
- 局部表面法向；
- 场景 Gaussian opacity；
- scene confidence；
- 投影可见性。

人体侧特征包括：

- 当前 anchor 世界坐标；
- 最近 K 帧 anchor 位置；
- 线速度和加速度；
- 脚底朝向；
- ankle/foot joint rotation；
- SMPL body pose；
- 人体 mask 置信度；
- 前一帧接触状态。

### 8.2 网络结构

第一版使用轻量因果 TCN 或 GRU：

```text
每帧输入：
  [pose_t, root_t, anchor_t, velocity_t,
   local_scene_t, contact_{t-1}, confidence_t]

过去 5-8 帧
  ↓
2-4 层 TCN 或 1 层 GRU
  ↓
contact probability
contact type
distance residual
pose residual
uncertainty
```

推荐初始配置：

- history length：5-8 帧；
- hidden dimension：128；
- 2-4 层 1D temporal convolution；
- 参数量：2M-10M；
- causal inference；
- 不使用未来帧，避免直播延迟。

### 8.3 接触状态稳定化

网络预测之后增加轻量状态机：

- 接触进入阈值高于接触保持阈值；
- 接触退出需要连续若干帧低概率；
- 接触状态变化时提高修正权重；
- 高不确定性时触发 I-frame 或重新初始化。

示例：

```text
enter contact: p > 0.75，连续 2 帧
maintain contact: p > 0.45
exit contact: p < 0.25，连续 3 帧
```

阈值需要在验证集上校准，不作为固定物理常数。

## 9. 接触修正模块

### 9.1 为什么采用低维修正

现有训练后人体 Gaussian 约为 45-60 万点，每帧逐点优化会带来：

- 大量 GPU 计算；
- 高显存占用；
- 不稳定的渲染梯度；
- 不可接受的直播延迟。

因此只修正：

- root translation；
- root rotation；
- 脚部相关关节；
- 手部相关关节；
- 少量局部 body pose。

修正后通过 LBS 自动更新所有人体 Gaussian。

### 9.2 可微 IK / 几何投影

推荐使用“网络预测 + 几何投影”组合：

```text
网络预测接触点和状态
  ↓
选择目标表面点 p* 与法向 n*
  ↓
foot/hand IK
  ↓
penetration projection
  ↓
姿态限幅与时序平滑
  ↓
corrected SMPL/SMPL-X
```

脚-地面约束可以使用：

```text
foot height ≈ surface height + contact offset
foot velocity ≈ 0      （站立接触时）
foot normal ≈ surface normal
penetration depth ≤ 0
```

不建议直接把脚强行投影到最近 Gaussian 中心，因为 Gaussian 中心不等于真实表面。应使用局部点云/平面拟合得到接触 surface proxy，并保留置信度和截断距离。

### 9.3 穿透修正

对于场景 proxy 能估计法向的区域：

- 计算人体 contact vertices 到场景表面的 signed distance；
- 当 signed distance 小于负阈值时，沿场景法向推出人体；
- 将推出量分配到 root 和相关关节；
- 通过姿态限幅避免突然跳变。

第一版可以只实现脚部局部穿透修正，不立即实现完整人体-场景碰撞检测。

## 10. 训练数据与标签

### 10.0 2026-08-26 自动研究执行约束

实际 PROX assets 已就位后，研究执行固定采用四层证据：PROX oracle SDF 教师 → 可控 synthetic front-end drift → Human3R-on-PROX → GUSH3R Gaussian。前两层仅验证标签、校正机制与网络归纳偏置，不能替代后两层的前馈/Gaussian 主结论。

新增 `forward_contact_pipeline/RESEARCH_PROTOCOL_20260826.md` 作为实验预注册：固定完整 recording split、Raw/EMA/Rule/GRU/TCN/IK baseline、接触/渲染/系统三类指标和停止条件。PROX 标签由 `build_prox_geometry_labels.py` 从 PROXD SMPL-X、官方 cam2world 和场景 SDF 构造；脚速度必须由固定 SMPL-X foot joint 轨迹而不是逐帧最近顶点计算。该 GT 只提供接触/表面/穿透监督；pose/root residual 仅能由可控扰动或真实前馈预测误差构造，禁止以全零 residual 伪训练。

### 10.1 数据集分工

| 数据集 | 用途 | 说明 |
|---|---|---|
| BEDLAM | 人体运动和脚接触预训练 | 有 SMPL/SMPL-X、姿态和合成场景 |
| PROX | 真实室内人体-场景接触 | 脚-地面、坐姿、身体接触 |
| BEHAVE | 人体-物体交互 | 手-物体和遮挡 |
| GRAB/ARCTIC | 手部和物体接触 | 第二阶段扩展 |
| NeuMan | 真实 Gaussian 人体-场景验证 | 六场景、与现有 HUGS 管线一致 |
| HOSNeRF | 人体-物体-场景交互验证 | 适合物体分支扩展 |

### 10.2 第一版最小数据组合

```text
BEDLAM + PROX 训练
NeuMan 测试
```

原因：第一阶段只做脚-地面接触，避免手-物体标注和物体轨迹成为额外变量。

### 10.3 标签构造

接触标签可以由几何规则生成：

```text
contact =
  distance(foot, scene surface) < threshold
  AND foot velocity < velocity threshold
  AND normal compatibility
```

同时生成：

- contact state；
- contact point；
- surface normal；
- signed distance；
- penetration depth；
- foot sliding ground truth。

NeuMan 没有完整接触标签时，可以使用 SMPL 脚底顶点、局部 scene proxy 和人工检查构造 pseudo labels。所有 pseudo labels 必须标记来源和置信度，不应与真实人工标注混为一谈。

## 11. 训练目标

整体损失：

```text
L = λstate Lstate
  + λcontact Lcontact
  + λdist Ldistance
  + λpen Lpenetration
  + λpose Lpose
  + λtemp Ltemporal
  + λrender Lrender
  + λunc Luncertainty
```

### 11.1 接触状态损失

- binary cross entropy 或 focal loss；
- 处理接触/非接触类别不平衡；
- 对接触转换帧提高权重。

### 11.2 几何接触损失

- 脚到场景 surface 的距离损失；
- 接触时脚速度损失；
- 脚底与场景法向一致性损失；
- 人体 penetration penalty。

### 11.3 时序损失

- 接触状态变化平滑；
- root/joint velocity 连续；
- acceleration regularization；
- 接触期间脚部位置稳定。

### 11.4 渲染损失

修正后的 pose 通过 LBS 生成 Gaussian，再进行局部渲染：

- contact ROI RGB loss；
- contact ROI silhouette loss；
- human-region LPIPS；
- 全图质量保持约束。

渲染损失不应压过几何接触损失，否则网络可能学会“看起来正确但仍然穿透”的解。

## 12. 与现有 HUGS 系统的对接

### 12.1 可直接复用的模块

现有项目中已具备：

- `hugs/models/anchor_attention.py`：人体 anchor 和 scene KNN 查询；
- `scripts/contact_eval_lib.py`：接触 anchor、局部场景和 ROI 评价；
- `scripts/evaluate_contact_quality.py`：接触质量评估入口；
- HUGS SMPL/LBS deformation；
- HUGS Gaussian renderer；
- Human3R 到 HUGS 的转换和对齐脚本；
- 真实 `delta_mu` 序列和 receiver-side rendering；
- 传输系统中的人体优先和 scene chunk baseline。

### 12.2 推荐新增模块

```text
contact_streaming/
├── features.py              # pose/anchor/local-scene features
├── surface_proxy.py         # local scene surface and normal
├── contact_state.py         # causal contact classifier/state machine
├── correction_net.py        # GRU/TCN residual predictor
├── ik_projection.py         # foot/hand IK and penetration projection
├── packet.py                # low-dimensional contact packet
├── runtime.py               # server/client real-time loop
└── evaluate.py              # contact, rendering, latency metrics
```

### 12.3 与现有 Anchor Attention 的关系

现有 Anchor Attention 可以作为离线 teacher 或第一版特征提取器，但不应直接承担直播时的高维 correction。

推荐转化关系：

```text
现有 AnchorAttention
  → 提供 anchor-world、scene KNN、attention/context 特征

新 Contact Correction Network
  → 只预测 contact state 和低维 SMPL residual

HUGS LBS
  → 将修正后的 pose 传播到所有人体 Gaussian
```

## 13. 传输协议设计

### 13.1 静态资产

延续现有传输系统：

- canonical human Gaussian：约 5.9 MB；
- LBS weights：约 10.1 MB；
- scene Gaussian：约 486 MB，支持分块渐进加载；
- SMPL metadata：约 60 KB。

### 13.2 接触 packet

第一版每帧可传：

- frame id；
- root translation residual，3 floats；
- root rotation residual，3 或 4 floats；
- left/right foot residual；
- contact state bitmask；
- contact confidence；
- uncertainty；
- optional keyframe flag。

使用 fp16 或 int16 后，单帧通常可控制在数百字节到数 KB，远小于现有逐 Gaussian correction。

### 13.3 传输策略

- 接触状态不变：只发送 pose/contact residual；
- 接触状态切换：提高 packet 优先级；
- 预测不确定性高：发送更完整的 pose keyframe；
- 接触区域附近 scene chunk 未到达：优先调度该 chunk；
- 远景 scene chunk：低优先级渐进传输。

### 13.4 与现有 correction 流的关系

可比较三种模式：

1. 原有 Gaussian `delta_mu` 流；
2. pose-conditioned `delta_mu` predictor + residual；
3. 低维 contact correction 流。

最终目标不是证明低维修正永远替代 Gaussian correction，而是验证在接触区域和直播低延迟场景中，低维接触流具有更高的单位 bit 质量收益。

## 14. 评价指标

### 14.1 接触几何指标

- Foot-ground distance error；
- Contact point error；
- Contact jitter；
- Foot sliding；
- Penetration depth；
- Penetration ratio；
- Contact consistency；
- Contact transition F1。

### 14.2 视觉指标

- Full-frame PSNR/SSIM/LPIPS；
- Human-region PSNR/SSIM/LPIPS；
- Contact-ROI PSNR/SSIM/LPIPS；
- Temporal flicker；
- 修正前后质量变化。

### 14.3 系统指标

- 接触估计 latency；
- correction inference FPS；
- end-to-end FPS；
- first human visible time；
- scene first-contact visible time；
- packet bitrate；
- packet late ratio；
- contact packet recovery time；
- GPU memory；
- CPU/GPU utilization。

### 14.4 必须比较的 baseline

```text
1. 前馈模型原始输出
2. 前馈输出 + temporal smoothing
3. 前馈输出 + 几何 foot projection
4. 前馈输出 + TCN/GRU contact correction
5. 前馈输出 + contact correction + IK
6. 现有 HUGS/AnchorAttention 离线优化结果

### 14.5 参考 JOSH 与 GRAFT 的评价口径

JOSH 主要使用物理合理性指标：

- 脚部滑动：在真实脚部相邻帧位移小于 10 mm 的接触帧中，统计预测脚部的相邻帧位移；
- 脚部浮空率：计算脚部到场景点云的最小距离，超过 20 cm 的帧视为浮空；
- 抖动：统计三维关节加速度变化率。

JOSH 的接触损失包括接触场景损失和接触静止损失，前者约束人体接触顶点接近场景，后者约束持续接触的脚在场景坐标中保持稳定。它们适合评价我们方案中的脚部滑动、浮空和时序稳定性。

GRAFT 使用 PROX 接触顶点集合，并报告：

- 接触精确率、召回率和 F1；
- V2S：人体接触顶点到最近场景点的位移向量误差；
- D2S：人体到场景的接触方向误差。

GRAFT 的接触图主要由人体顶点与场景点之间的空间邻近关系得到，训练时不使用单独的接触损失或穿透损失，而是通过几何探针和人体顶点监督隐式学习接触关系。

我们的指标应结合两类工作：

```text
GRAFT：接触区域是否正确、距离和方向是否正确
JOSH：脚是否滑动、是否浮空、运动是否抖动
我们：接触修正后人体高斯是否稳定、可渲染、可传输
```

此外，研究点二还必须增加高斯特有指标：接触区域高斯穿透比例、接触区域 PSNR/SSIM/LPIPS、人体高斯时序闪烁、接触数据包大小和端到端传输延迟。
```

## 15. 实验矩阵

### 15.1 接触模块消融

| 实验 | 时序模型 | 几何投影 | IK | 目的 |
|---|---|---|---|---|
| A | 无 | 无 | 无 | 前馈 baseline |
| B | EMA | 无 | 无 | 简单时序稳定 |
| C | GRU | 无 | 无 | 学习式 contact state |
| D | GRU | 有 | 无 | 几何穿透修正 |
| E | GRU | 有 | 有 | 完整低维接触修正 |
| F | GRU | 有 | 有 | 加入 uncertainty/keyframe |

### 15.2 数据与泛化实验

| 训练 | 测试 | 目的 |
|---|---|---|
| BEDLAM | PROX | 合成到真实接触泛化 |
| BEDLAM + PROX | NeuMan | 真实人体-场景 Gaussian 验证 |
| BEDLAM + PROX | HOSNeRF | 交互场景扩展 |
| 跨场景训练 | NeuMan 留一场景 | 场景泛化 |

### 15.3 网络条件实验

- 10 Mbps；
- 50 Mbps；
- 100 Mbps；
- 1 Gbps；
- 带宽波动；
- packet late；
- burst loss；
- contact transition burst。

## 16. 计算资源估计

### 16.1 最小可行版本

冻结 Human3R/GUSH3R 和 HUGS，只训练 Contact State Estimator 与 Correction Network：

- GPU：1 张 24 GB；
- CPU：8-16 核；
- 内存：32-64 GB；
- 模型参数：约 2M-10M；
- 训练时间：约 1-3 天；
- 推理：目标 1-5 ms/帧；
- 目标速度：30-60 FPS。

### 16.2 前馈 Gaussian decoder 版本

- GPU：1 张 40-80 GB；
- Scene decoder：约 1 天级别训练；
- Human decoder：约 2 天级别训练；
- 接触网络：额外 1-3 天；
- 需要缓存 foundation model 中间特征和场景 proxy。

### 16.3 联合微调版本

不建议作为第一阶段：

- 80 GB 级 GPU 或多卡；
- 更高数据和训练管理成本；
- backbone 错误会传递到 Gaussian 和接触模块；
- 实验变量过多，难以证明接触模块的独立贡献。

## 17. 分阶段实施计划

### Phase 0：数据和评估闭环

目标：不训练新模型，先确定接触质量是否可测。

- 建立脚底 anchor 和局部 scene proxy；
- 生成 NeuMan pseudo contact labels；
- 跑 foot-ground distance、sliding、jitter、penetration；
- 建立前馈/现有 HUGS 输出 baseline；
- 确认坐标系和场景表面法向。

### Phase 1：几何规则 baseline

- 最近场景表面查询；
- 脚底高度投影；
- 简单速度阈值 contact state；
- foot IK；
- 不使用学习式网络。

目标是回答：纯几何修正能改善多少，哪些错误必须由时序模型解决。

### Phase 2：轻量接触网络

- GRU/TCN 接触分类；
- contact distance residual；
- uncertainty output；
- 接触状态 hysteresis；
- 低维 packet。

目标：实时速度和接触质量同时达到可用水平。

### Phase 3：接入 HUGS/Gaussian renderer

- corrected SMPL/LBS；
- 接触区域 Gaussian 渲染；
- 评价 contact ROI；
- 对比完整 `delta_mu` 流和低维 contact stream。

### Phase 4：场景块预取

- 接触状态驱动 chunk priority；
- 人体运动轨迹预测；
- 不确定性感知 I-frame；
- 带宽变化和丢包实验。

### Phase 5：手-物体与坐姿扩展

- BEHAVE/GRAB/ARCTIC；
- 物体 rigid proxy；
- hand-object contact；
- seated contact。

## 18. 基于 Human3R/GUSH3R 的具体化 Pipeline

本节将前面的通用方案固定为一个可以直接实现和做实验的版本。建议不要一开始联合训练 Human3R、GUSH3R 和接触模块，而是采用“冻结前馈模型、先训练接触修正、最后再做轻量联合微调”的路线。

### 18.1 两个前馈模型的职责

```text
Human3R:
  RGB video
    -> camera pose / intrinsics
    -> scene depth and point map
    -> SMPL-X pose and translation
    -> image/human tokens and recurrent memory

GUSH3R:
  Human3R-like features and geometry
    -> scene Gaussian decoder
    -> human Gaussian decoder
    -> static scene accumulation
    -> online renderable human-scene Gaussians

Ours:
  Human3R/GUSH3R outputs
    -> contact-aware local geometry representation
    -> causal contact estimator
    -> low-dimensional pose/root correction
    -> IK and penetration projection
    -> corrected human Gaussian deformation
```

Human3R 和 GUSH3R 不应被当作两个完全独立的 backbone。GUSH3R 继承了 Human3R 的人体、场景和相机估计，因此 Human3R 中的尺度、坐标和人体-场景相对位置误差也可能传递到 GUSH3R。我们的模块必须显式估计几何置信度，并允许在低置信度时回退到原始前馈姿态。

### 18.2 推荐的实际研究路线

#### 路线 A：Human3R 几何预验证版本

这是第一阶段最容易完成、也最适合验证研究假设的版本：

```text
RGB frame t
  -> Human3R inference
  -> saved depth/conf/camera/smpl outputs
  -> SMPL-X foot anchors
  -> local scene surface proxy
  -> contact GRU/TCN
  -> foot/root residual
  -> SMPL-X forward kinematics and IK
  -> corrected human mesh / Gaussian deformation
```

该版本不要求立刻复现 GUSH3R 的全部 Gaussian decoder，主要验证：在相同前馈输入下，接触修正可以降低脚部滑动、接触抖动、浮空和网格穿透。该版本评价的是网格/点图层面的接触几何，不能替代最终的人体高斯渲染评价。

#### 路线 B：GUSH3R 主实验版本

在路线 A 的接触模块稳定后，再接入 GUSH3R，完成正式主实验：

```text
RGB frame t
  -> GUSH3R
  -> scene Gaussians + human Gaussians + SMPL-X state
  -> local scene proxy from scene Gaussians/depth
  -> contact estimator
  -> corrected SMPL-X state
  -> LBS/deformation of human Gaussians
  -> scene Gaussians unchanged
  -> renderer
```

接触修正只改变人体的低维运动状态和由 LBS 驱动的人体 Gaussian 位置，不重新优化场景 Gaussian 的颜色、尺度和 opacity。主表必须比较原始 GUSH3R 与 GUSH3R 加接触修正，Human3R 只用于中间表示消融和模块预验证。

### 18.3 明确的模块接口

建议将接触模块的输入固定为以下结构，而不是直接把全部 feature map 输入网络：

```text
FrameInput[t] = {
    image_feature:       [C, Hf, Wf] or pooled tokens,
    smplx_pose:           [J, 3] or rotation representation,
    root_translation:     [3],
    anchor_position:      [A, 3],
    anchor_velocity:      [A, 3],
    anchor_orientation:   [A, 6],
    local_surface_points: [M, 3],
    local_surface_normals: [M, 3],
    local_surface_conf:   [M],
    human_confidence:     [A],
    previous_contact:     [K],
    previous_residual:    [D]
}
```

第一版可以把 `image_feature` 去掉，先完成几何 baseline；正式版本应加入人体 crop 或 image tokens，因为仅凭 SMPL 和深度无法可靠区分遮挡、离地和估计错误。

网络输出固定为：

```text
FrameOutput[t] = {
    contact_logits:       [K],
    contact_point:        [K, 3],
    surface_distance:     [K],
    surface_normal:       [K, 3],
    pose_residual:        [D_pose],
    root_residual:        [6],
    penetration_risk:     [K],
    uncertainty:          [K],
    reset_flag:           [1]
}
```

其中第一阶段令 `K=2`，只表示左右脚；`D_pose` 只包含左右 ankle、足部相关关节和 root，不直接预测完整 SMPL-X pose。这样可以将网络输出限制在 20-40 个连续变量，便于训练、传输和稳定化。

### 18.4 局部场景 proxy 的实现顺序

不建议直接使用最近的 scene Gaussian center 作为接触表面。建议按以下顺序实现三个版本：

1. **平面 proxy**：从脚部 anchor 周围的深度点或 point map 拟合局部平面，输出高度和法向。
2. **局部点云 proxy**：保留 KNN 点云，使用加权 PCA 或 MLS 估计表面和 signed distance。
3. **Gaussian-aware proxy**：使用 Gaussian 的 opacity、scale 和 viewing confidence 过滤场景点，再拟合局部表面。

实验中需要单独比较三种 proxy 的接触误差。这样可以回答一个关键问题：性能提升来自接触网络，还是仅仅来自更好的场景表面估计。

### 18.5 坐标对齐必须成为独立模块

Human3R 到 HUGS/GUSH3R 的直接坐标转换已经表现出明显的质量下降。因此接触 pipeline 不能假设各模块坐标天然一致，建议加入：

```text
Human3R camera/world outputs
  -> convention conversion
  -> robust Sim(3) alignment
  -> SMPL-X anchor projection check
  -> local contact coordinate frame
  -> contact estimator
```

具体实现：

- 使用场景点云和人体 anchor 估计全局 scale、rotation、translation；
- 用地面候选平面检查 root 高度和脚底高度；
- 计算 SMPL-X 投影与人体 mask 的重叠率；
- 当对齐残差超过阈值时，只输出 contact confidence，不强行做 IK；
- 记录每帧 alignment residual，作为网络输入和评价指标。

这一步既是工程必需，也可以形成方法中的“uncertainty-aware alignment”组件。

### 18.6 训练分成三个阶段

#### Stage 1：几何教师和伪标签

冻结 Human3R/GUSH3R，只运行离线推理并保存：

- SMPL-X pose、root translation；
- depth、confidence 和 local point map；
- scene proxy；
- ground-truth 或 pseudo contact labels。

BEDLAM 提供较干净的合成接触标签，PROX 提供真实室内场景接触，NeuMan 用于真实 Gaussian 管线验证。伪标签必须保存 `label_source` 和 `label_confidence`，避免将不可靠的 NeuMan 标签当作真实 GT。

#### Stage 2：接触估计和修正网络

训练一个 2M-10M 参数的 causal GRU 或 TCN：

```text
input window: 5-8 frames
hidden size: 128
output: contact state + geometry + residual + uncertainty
optimizer: AdamW
training: mixed precision, frozen upstream backbone
```

损失建议先使用：

```text
L = 1.0 L_state
  + 1.0 L_surface
  + 0.5 L_velocity
  + 0.5 L_penetration
  + 0.2 L_pose
  + 0.2 L_temporal
  + 0.1 L_uncertainty
```

这些权重只是初始配置，最终应在验证集上根据 contact F1、foot sliding 和 ROI LPIPS 联合调节。渲染损失不应在第一阶段占主导，否则网络可能只学习视觉贴合而没有真实接触。

#### Stage 3：轻量联合微调

只有当 Stage 2 已经稳定后，才允许解冻：

- GUSH3R human Gaussian decoder 的最后一层；或
- 人体 Gaussian 的 appearance/deformation adapter。

Human3R/GUSH3R backbone 仍保持冻结。联合微调只用于恢复接触修正造成的局部外观变化，不应让 backbone 重新学习全部重建任务。

### 18.7 在线推理时的时序逻辑

每帧执行顺序固定为：

```text
1. 读取前馈输出和上一帧状态
2. 完成坐标转换和 alignment residual 计算
3. 更新局部 scene proxy
4. GRU/TCN 预测 contact 和 residual
5. 通过 hysteresis 更新 contact state
6. 对高置信度接触执行 IK
7. 执行 penetration projection 和 residual clamp
8. 通过 LBS 更新人体 Gaussian
9. 生成 contact packet 或本地渲染
10. 写入时序 memory
```

建议设置三种运行状态：

- **Tracking**：接触置信度高，发送低维 residual；
- **Uncertain**：几何或网络不确定性高，降低修正幅度；
- **Reset**：连续多帧对齐失败或人体突然离开场景，发送 keyframe 并重建局部状态。

### 18.8 论文实验应固定的主表

主实验不要只比较 PSNR。建议固定以下四组：

| 方法 | 前馈来源 | 接触模块 | 主要目的 |
|---|---|---|---|
| M1 | Human3R/GUSH3R | 无 | 原始前馈 baseline |
| M2 | Human3R/GUSH3R | EMA/规则投影 | 几何 baseline |
| M3 | Human3R/GUSH3R | GRU/TCN，无 IK | 学习接触估计贡献 |
| M4 | Human3R/GUSH3R | GRU/TCN + IK + penetration | 完整方法 |
| M5 | HUGS | 离线优化 | 与迭代式方法比较 |

每组都报告：

- Foot-ground distance；
- Contact F1 和 consistency；
- Foot sliding；
- Penetration ratio/depth；
- Human PSNR/SSIM/LPIPS；
- Contact ROI PSNR/LPIPS；
- 每帧接触模块延迟；
- 端到端 FPS；
- packet bitrate。

这样可以清楚地区分“接触质量提升”“视觉质量变化”和“实时性代价”。

### 18.9 实施优先级

建议按以下顺序推进：

```text
P0: Human3R 输出保存、坐标统一、脚 anchor 和局部平面
P1: 规则 contact + foot projection + 指标闭环
P2: GRU/TCN contact estimator，冻结前馈模型
P3: corrected SMPL-X -> Gaussian LBS -> contact ROI rendering
P4: GUSH3R 接入和前馈 Gaussian 对比
P5: 低维 packet、带宽、丢包和客户端渲染
P6: 手-物体、坐姿和多接触扩展
```

当前最重要的里程碑不是立刻训练一个大模型，而是完成 `Human3R 输出 -> 脚接触 proxy -> 规则修正 -> Gaussian 渲染 -> 接触指标` 的闭环。这个闭环成立后，GRU/TCN 和 GUSH3R 接入才有明确的实验收益可验证。

## 19. 主要风险与应对

### 风险 1：场景 Gaussian 不是可靠表面

应对：

- 使用局部 voxel/point proxy；
- 对局部点云拟合平面或小型 surface proxy；
- 引入场景置信度；
- 不直接把 Gaussian center 当作表面。

### 风险 2：NeuMan 接触标签不完整

应对：

- 第一阶段用 PROX/BEDLAM 训练；
- NeuMan 只做真实渲染验证；
- pseudo labels 标记置信度；
- 人工抽查接触关键帧。

### 风险 3：姿态修正破坏视觉质量

应对：

- pose residual 设置幅度上限；
- 增加 contact ROI 和全图渲染保持损失；
- 先做 root/foot correction，再扩展到全身；
- 高不确定性时保持原始前馈姿态。

### 风险 4：接触状态闪烁

应对：

- causal GRU/TCN；
- hysteresis 状态机；
- contact transition loss；
- 接触状态最短保持时间；
- uncertainty-triggered reset。

### 风险 5：实时速度被前馈 backbone 限制

应对：

- 将 foundation model 放在服务器端；
- 客户端只运行接触修正和 LBS；
- 使用轻量 checkpoint；
- 对人体 anchor 特征缓存；
- 采用关键帧/增量更新。

## 20. 预期贡献

如果 Phase 1-3 验证成功，可以形成以下贡献：

1. 提出一个面向前馈人体-场景 Gaussian 重建的实时接触修正问题设置。
2. 设计基于局部场景几何、人体语义 anchor 和历史状态的因果接触估计器。
3. 用低维 SMPL/SMPL-X residual 和 IK 修正替代逐帧 Gaussian 优化。
4. 提出不确定性感知的接触 packet 和接触驱动传输策略。
5. 同时报告视觉重建质量、几何接触质量和直播系统延迟。

预期最有辨识度的研究结论是：

> 对于实时人体-场景 Gaussian 直播，前馈 Gaussian 生成解决了表示生成速度，但接触一致性需要一个独立的、因果的低维状态修正层。该修正层不需要重新优化高维 Gaussian，也可以通过低带宽 packet 在客户端实时执行。

## 21. 第一阶段验收标准

在不联合微调前馈 backbone 的条件下，第一版至少应达到：

- 接触估计推理速度：30 FPS 以上；
- 接触状态 F1：相对几何 baseline 有明显提升；
- Foot sliding：相对原始前馈输出降低 30% 以上；
- Contact jitter：相对原始输出降低 30% 以上；
- penetration ratio：明显下降；
- Contact ROI PSNR：不下降或提升；
- 全图 PSNR：下降不超过 0.1 dB；
- 每帧接触 packet：小于 10 KB；
- 客户端 LBS + 修正 + 渲染：达到 30 FPS；
- 高不确定性帧：能够自动回退到原始姿态或请求 keyframe。

## 22. 当前推荐决策

建议把第二个研究点暂定为：

> **面向实时人体-场景 Gaussian 直播的前馈接触估计与低维姿态修正**

第一版只做脚-地面接触，采用：

```text
Human3R/GUSH3R 或现有前馈输出
→ SMPL anchor + local scene proxy
→ causal GRU/TCN
→ contact state + root/foot residual
→ foot IK + penetration projection
→ LBS Gaussian rendering
→ low-dimensional contact packet streaming
```

不要从一开始加入手-物体、全身碰撞、foundation model 联合微调和复杂网络协议。先证明“实时接触修正比原始前馈输出更稳定，并且不需要逐帧 Gaussian 优化”，再逐步扩大任务范围。
