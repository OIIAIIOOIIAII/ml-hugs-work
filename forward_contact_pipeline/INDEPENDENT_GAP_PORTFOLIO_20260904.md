# 独立技术缺口分析与研究组合（2026-09-04）

## 1. 从现有约束出发，而非从模型名出发

### 现有资源与硬约束

- 输入：单目、人—场景视频；可用 Human3R/GUSH3R 前端，前端输出并非接触级可靠。
- 表示：静态 scene Gaussian + SMPL-X 驱动 human Gaussian；人体高斯可经 LBS 更新。
- 数据：PROX 的 RGB-D、PROXD、scene SDF/cam2world 已有；NeuMan 已有；没有可无限扩张的大规模、多样化接触数据。
- 算力：共享单张 4090；不能把方法建立在训练大型视频生成/世界模型或多轮全序列优化上。
- 已知失败：Human3R→PROXD hold-out 对齐 median 7.37 cm；旧跨场景 synthetic split 接触先验塌缩。

### 因此真正的技术缺口

> 前馈 human-scene 3D 重建缺少一个**在线、可校准、可拒绝的世界坐标状态估计层**：它应区分“人体前端错误”“相机/尺度错误”“局部表面不可靠”和“真实接触变化”，并把不确定性传递给修正、渲染和网络传输。

如果这个层不存在，继续扩大 GRU、使用世界模型或接入通信预测都会放大错误，而不是解决错误。

这也是第一研究点和第二研究点最自然的连接：

- 第一研究点的 Anchor Attention 已验证“人体语义 anchor 应查询局部 scene Gaussian”；
- 第二点不应立即学习完整 contact policy，而应先把 anchor-query 变为**在线 human-to-scene state estimation**；
- 接触修正、短期预测和低带宽传输都是这个可信状态的下游应用。

## 2. 候选方案组合

评分：5=高，1=低；“资源适配”越高越适合当前数据与单卡。“论文风险”越高表示越不确定。

| 编号 | 方案 | 新数据 | 算力 | 资源适配 | 创新潜力 | 论文风险 | 建议 |
|---|---|---:|---:|---:|---:|---:|---|
| A | Anchor Calibration Filter（在线可信对齐） | 无 | 低 | 5 | 4 | 2 | **主线必做** |
| B | Contact-aware State Estimation & Control（局部接触控制） | PROX | 低/中 | 5 | 4 | 3 | **主线必做** |
| C | Gaussian-derived Contact Surface（从 scene GS 建立可信表面） | 无 | 中 | 4 | 4 | 3 | 与 A/B 联合 |
| D | Event-triggered Semantic Streaming（接触驱动传输） | 无 | 低 | 5 | 4 | 2 | **可快速形成系统贡献** |
| E | Short-horizon Contact Predictor（结构化微型世界模型） | PROX | 中 | 4 | 4 | 4 | A/B 成功后做 |
| F | Test-time Anchor Adaptation（测试时轻量自适应） | 无 | 中 | 3 | 4 | 4 | 备选探索 |
| G | Contact Reliability Benchmark / Stress Test | 无 | 低 | 5 | 3 | 2 | 所有方案的共同基础 |
| H | 大型生成式世界模型/视频扩散 | 大量 | 极高 | 1 | 3 | 5 | **当前不做** |

## 3. 方案 A：Anchor Calibration Filter（推荐为新核心）

### 问题

当前把 Human3R 的 SMPL-X 与 PROXD/GUSH3R scene 直接对齐，采用一次性 Sim(3) 后仍出现厘米级 hold-out 误差。原因很可能不是“接触网络不够大”，而是 camera、尺度、root、局部姿态和局部表面噪声被混成了一个误差。

### 方法

维护一个小的在线状态，而不是每帧重新拟合全身：

\[
x_t = \{s, R, t, \Delta root_t, b_t, \Sigma_t\}
\]

- `s,R,t`：人体前端到 scene-world 的全局 Sim(3)，在短窗口内缓慢变化或固定；
- `Δroot`：帧级低维 root bias；
- `b`：局部脚/锚点 bias；
- `Σ`：协方差/可信度。

观测由四类残差组成：

1. SMPL/人体 mask 的重投影一致性；
2. anchor 到局部 metric surface 的距离/法向一致性；
3. anchor 到 local scene-Gaussian token 的稳定性/可见性；
4. 时序运动先验（速度、加速度、已确认 stance 的 foot lock）。

使用 robust EKF/UKF、因子图或固定窗口 Gauss-Newton 均可；第一版可从可解释的 robust filter 开始，学习模块只预测观测噪声或权重，而非直接预测位姿。

### 创新点与第一点的联系

把 Anchor Attention 从“直接输出 per-Gaussian 位移”转换成“为在线 world-state filter 提供场景条件观测”。这比再堆一个 attention 更有明确问题：**怎样判断人体—场景对齐是否可信，并在不可信时拒绝修正。**

### 数据/算力

- 只需已有 PROX、Human3R 输出和 scene SDF；不需要训练大模型。
- 可在 CPU 或单卡运行；滤波状态维度很小。
- 用 PROXD 提供 oracle 对齐评测，按 train/hold-out frames 验证 scale/root/foot-anchor error。

### 关键实验

Raw Sim(3) vs sequence-level Sim(3) vs online filter vs oracle PROXD；报告全身/脚 anchor hold-out error、mask IoU、surface distance、uncertainty calibration、每帧耗时。

## 4. 方案 B：Contact-aware State Estimation & Control

这是接触模块的正确重述：它不是“预测 GT pose”，而是基于 A 输出的可信世界状态，估计接触模式并进行最小充分修正。

### 设计

- `stance / transition / swing / slip / uncertain` 显式离散状态；
- 连续控制量只在 contact frame 中定义：法向修正消除 float/penetration，切向速度修正抑制 foot skate；
- root/ankle IK 是执行器，不是网络的高维输出；
- 观测噪声大时只输出 uncertainty/不执行，不强制贴地。

### 为什么不同于普通 pose refinement

- 目标是最小化 scene-relative constraint violation，而不是全身 MPJPE；
- 接触状态改变 loss/约束：stance 约束切向速度，swing 不应被粘在地上；
- 输出是可解释的 contact event 与控制量，能进入网络协议。

### 数据/算力

PROX 足够启动。先做 feet；手/臀部需要 RICH/BEHAVE 或可靠新标签，不能在当前阶段承诺。

## 5. 方案 C：Gaussian-derived Contact Surface

### 问题

GUSH3R 最终只有 scene Gaussian，并没有 PROX 那样的官方 SDF。直接把 Gaussian center 当表面是错误的，因此当前方法无法自然推广到正式 GUSH3R。

### 可行路线

在 anchor 邻域，从 scene Gaussian 构建局部、带置信度的 surface surrogate：

- 过滤低 opacity、过大 scale、不可见/跨帧不稳定的 Gaussian；
- 用 Gaussian covariance/normal proxy 形成局部 anisotropic density field；
- 在 contact frame 内求局部零水平面、法向和 signed-distance approximation；
- 仅在多帧一致、support 足够时输出可用 surface，否则交给 uncertainty gate。

核心不是宣称 Gaussian 是物理 SDF，而是学习/校准何时可把局部 Gaussian 视为可靠表面。

### 验证

先在 PROX mesh/SDF 人为采样、再退化成 Gaussian-like primitives，评估 surrogate distance/normal 与真实 SDF 的误差；通过后才接 GUSH3R。这样无需一开始就有 GUSH3R 接触 GT。

### 风险

若 surrogate 在遮挡、薄物、透明/低覆盖区域不可靠，必须以拒绝率和误差呈现，而不是强行对全部位置输出距离。

## 6. 方案 D：Event-triggered Semantic Streaming

### 技术缺口

LapisGS、GIFStream、3DGStream、D-FCGS/InterGS 等关注分层/预测式 Gaussian 传输，但通常以渲染质量、视角或运动相干作为重要性。对于 human-scene interaction，少量 root/foot 错误的感知和物理代价远高于大量背景 Gaussian 的小外观误差。

### 方法

保留现有 `delta_mu` 外观流，另建立超小控制流：

```text
scene/base-GS progressive stream
human appearance residual stream
contact semantic stream: mode, anchor id, Δroot/Δfoot, uncertainty, scene version, reset
```

使用 A/B 的 risk state 做 event trigger：

- 稳定 stance：低频 P-packet；
- 触地/离地、滑动、穿透、置信度突变：优先 I-packet；
- 包迟到/丢失：client 基于最后可信 control 做短时 hold，不做无界预测；
- scene version 变更：停止应用 contact residual，先同步 anchor context。

### 贡献与资源

这是低算力、可测量的系统贡献。现有 62-byte packet 与 network simulator 可直接扩展。无需声称新的通用 codec，而是提出 **contact-critical quality of experience** 和语义重要性调度。

### 必须的反证

与固定 I/P、等间隔采样、普通 delta-only、基于可见性/视角的优先级比较；在相同平均 bitrate、同样 trace 下报告 contact violation、恢复时延、flicker、PSNR/LPIPS。

## 7. 方案 E：Short-horizon Contact Predictor

仅当 A/B 的在线状态可靠后，才做短期预测。此处可借鉴 world-model 思路，但不训练生成式视频模型：

\[
p(x_{t+1:t+H}\mid x_{\le t}, G^{local})
\]

预测的是 contact transition、root/foot residual 和方差；用途是网络延迟 concealment、提前发 I-frame、检测不可能的状态跃迁。

### 数据/算力

- 先用 PROX 的状态轨迹，H=2/4/8；GRU/state-space model 足够。
- 训练重点是 calibrated uncertainty 和 prediction-vs-constant-velocity baseline，不是大模型容量。
- 若预测不能超过 last-value/constant-velocity，则停止，不强行贴“world model”标签。

## 8. 方案 F：Test-time Anchor Adaptation

### 想法

在一个新视频开始的数十帧内，仅优化极少参数：global Sim(3)、root bias、anchor-query temperature/observation noise，而冻结 Human3R/GUSH3R 和接触网络。损失来自 mask reprojection、可信局部 surface、跨帧 foot-lock，不使用未来 GT。

### 优点

适配新场景尺度/相机，直接针对当前对齐失败；比全模型 test-time tuning 算力低、可解释。

### 风险

与 PhysDynPose 等优化式方法需要清楚区分：仅校准全局/低维状态、有预算上限、严格因果、并以 timeout/failure mode 报告。若每帧优化过久，就不再是 online。

## 9. 方案 G：Contact Reliability Stress Test

无论选哪条主线，都应建立统一评测，而不是只做一张 PSNR 表：

- controlled perturbations：global scale、root bias、camera drift、local foot pose、surface dropout、occlusion、network loss；
- 真实切分：recording-level / scene-level / subject-level；
- 评测：correction gain、false correction rate、uncertainty risk-coverage、latency、packet recovery；
- 产物：可复现 manifest、trace、video overlays、failure taxonomy。

这是低风险资产：即使复杂学习模型失败，也能形成可复用的可信评测协议和负结果。

## 10. 推荐的研究决策

### 近期主线（最符合现有资源）

`A -> B -> C -> D`：

1. **A** 先把 7.37 cm 对齐失败拆解、压低或可信拒绝；
2. **B** 在可信 anchor state 上完成接触状态/IK 控制；
3. **C** 把 PROX SDF 的能力迁移为 scene-GS local surface surrogate；
4. **D** 将控制状态变成网络可调度语义流，形成在线系统主贡献。

它具有清晰的论文问题、合理数据需求、单卡可运行的 compute budget，而且每一步都可独立发表为消融/模块。

### 中期高收益扩展

在 A/B/C 成立后加入 **E**，将可靠状态做短期预测。这时“world-model”是有证据的结构化预测，而不是概念包装。

### 当前不推荐

- 从头训练视频扩散或通用 world model；
- 直接下载 TB 级 BEDLAM/RICH 并假设数据会解决坐标问题；
- 先做手物、多人体、移动物体、完整物理仿真；
- 在真实前端对齐失败前训练更大的 GRU/Transformer；
- 将 packet 原型描述为完整通信系统。

## 11. 两种可写论文的边界清晰定位

### 定位一：视觉/图形主导

**Online Confidence-Calibrated Human-Scene Gaussian Alignment and Contact Control**

贡献重点：A+B+C，强调 first-point anchor principle 的在线化，正式 GUSH3R 渲染闭环。

### 定位二：系统/沉浸通信主导

**Contact-Critical Semantic Streaming for Human-Scene Gaussian Telepresence**

贡献重点：A+B+D(+E)，强调语义 packet、风险调度、延迟丢包下的 contact QoE。必须使用真实 network trace/replay 和端到端测量。

两者可共享核心代码和数据，但投稿时必须选一个主叙事，不能把所有方向各做一点而没有完整证据。
