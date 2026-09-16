# 第二轮迭代：从接触修正到预测式在线 Human-Scene Gaussian 系统（2026-09-04）

## 1. 这轮迭代解决什么问题

第一轮把方案从“GRU 修脚”提升为 Contact-Conditioned Anchor Gaussian Controller（CAGC），但它仍主要是**当前帧纠错器**。这轮将竞争对象扩大到：

1. **世界模型 / 4D world reconstruction**：从历史观测预测后续世界状态或生成后续视觉内容；
2. **在线动态 Gaussian/SLAM**：持续维护一个动态、可渲染的 3D/4D 表示；
3. **Gaussian 流式传输/压缩**：在带宽、延迟、丢包下递送可交互自由视点内容；
4. **实时人体运动/接触系统**：在不可靠前端观测下保持人—场景物理一致性。

结论不是把一个大视频世界模型硬接进系统。我们的单目输入没有可靠动作控制信号，也没有资源训练大生成模型；直接宣称通用 world model 会超出证据范围。更准确、可发表的定位是：

> **一个以 scene-Gaussian 为外部记忆、以接触状态为结构化 latent、能够短期预测并进行网络自适应控制的 task-oriented contact world model。**

它只预测对“人体是否稳定地站/贴在场景中”最关键的低维状态，而不试图凭空生成整段 RGB 视频。

## 2. 扩展文献地图

| 方向 | 代表工作 | 对我们的含义 |
|---|---|---|
| 4D/世界重建 | GFlow, AAAI 2025, DOI `10.1609/aaai.v39i8.32847`; RenderWorld, ICRA 2025, DOI `10.1109/icra55743.2025.11127609` | 竞争点是长期时空一致性；它们不等价于可度量的 human-scene contact control。我们需要报告预测误差和物理违例，而不仅是视觉观感。 |
| 动态 GS 表示 | 4DGS, arXiv:2310.08528; Deformable 3DGS, CVPR 2024; SplineGS/MoDec-GS, CVPR 2025 | 运动表示、时间连续性和紧凑动态编码已经很强；我们不能只贡献“让 Gaussian 随 SMPL 动”。 |
| 在线 GS | 3DGStream, CVPR 2024, DOI `10.1109/cvpr52733.2024.01954`; SplatMAP, TOG 2025; LongSplat, AAAI 2026 | 在线系统的门槛是 bounded memory、时延、观测不一致和实时 map update。我们的 controller 必须有清晰的端到端 latency 预算。 |
| 自适应/4D GS 流传输 | LapisGS, 3DV 2025, DOI `10.1109/3dv66043.2025.00096`; GIFStream, CVPR 2025, DOI `10.1109/cvpr52734.2025.02027`; HiCoM, arXiv:2411.07541; PRoGS, WACV 2025 | 渐进层级、运动相干和可视优先传输已有工作。我们的区分点不能只是“delta 压缩”，而应是 **contact-critical state 的任务驱动调度**。 |
| 动态 GS 编码 | D-FCGS, AAAI 2026, DOI `10.1609/aaai.v40i19.38674`; InterGS, VCIP 2025; ADC-GS, IJCAI 2025 | 预测式/帧间 Gaussian 编码是直接通信竞品；需要与其比较 rate-distortion，同时引入 contact distortion。 |
| 接触/物理人体 | RICH/BSTRO, DECO, SA-HMR, PhysCap, PhysDynPose | 这些工作覆盖接触检测、场景感知 HMR 和物理优化；我们必须证明“预测 + Gaussian + 网络鲁棒”产生了单独价值。 |

截至本轮搜索，没有检索到同时满足“前馈 human-scene 3DGS + 显式接触状态预测 + 接触驱动低维控制 + 网络自适应传输”的直接同类方法；这只能作为检索范围内的定位，写论文前仍需完成系统性 related-work 检索，不能写成绝对首创。

## 3. 修订后的总方案：PCWM-GS

暂定名称：**Predictive Contact World Model for Human-Scene Gaussians（PCWM-GS）**。

```text
                 server / encoder
RGB video -> GUSH3R -> static scene GS + human GS + SMPL-X
                            |
                 contact-anchor scene query
            (metric SDF/proxy + local scene-GS tokens)
                            |
      causal predictive contact state model (PCWM)
    -> state, residual distribution, H-step future prediction,
       uncertainty, keyframe/packet priority
                            |
          contact-gated IK + SMPL-X LBS
                            |
         corrected human GS + fixed scene GS renderer
                            |
           semantic I/P packets over network
                            v
                 client / decoder
  cached scene GS + cached human canonical GS + last SMPL-X
      -> predicted state during delay/loss -> LBS -> render
      -> correction/keyframe arrival -> confidence-gated reconcile
```

### 3.1 状态不是像素，而是可控的 contact latent

每帧状态：

\[
z_t = \{q^{root}_t, q^{foot}_t, a_t, s_t, d_t, n_t, v_t, c_t, u_t\}
\]

- `q`：有限的 root/foot SMPL-X 参数；
- `a`：语义 foot anchor 在 scene world 的位置/朝向；
- `s`：`swing / approach / stance / slip / uncertain` 接触模式；
- `d,n,v`：距离、法向、速度；
- `c,u`：接触概率与不确定性。

局部 scene Gaussian 不是被当成点云，而是通过研究点一同源的 Anchor-to-Scene cross-attention 写入状态。SDF/proxy 是 metric physical evidence；GS token 是可渲染的局部 world memory。这样既能处理有 SDF 的 PROX，也能在 GUSH3R/NeuMan 中使用 proxy，而不会把 Gaussian center 错当表面。

### 3.2 两个时间尺度：纠错 + 短期预测

PCWM 同时输出当前纠错和未来 `H` 帧分布：

\[
(\mu_{t:t+H}, \sigma_{t:t+H}, p(s_{t:t+H})) = f_\theta(z_{\le t}, G^{local}_{\le t})
\]

- **当前帧**：依据真实观测执行 contact-gated correction；
- **未来 2–8 帧**：只用于 latency concealment、packet scheduling 和提前发现可能的接触转换，不能作为伪真值渲染结果；
- **高方差**：不预测性强修正，退回最后可信状态/等待 keyframe。

这是一种结构化、任务导向的“微型世界模型”，而非生成式视频 world model。它的优点是可用 PROX 的几何真值监督、能解释、可在边端运行；限制是不能生成未见物体或复杂未来行为，论文必须写清。

### 3.3 Contact-aware rate-distortion-control（新增系统贡献）

将既有 `delta_mu` 平滑/差分传输扩展为两条不同的传输层：

1. **外观流**：scene/base human Gaussian 的渐进层、`delta_mu` 或现有 GS codec；
2. **接触控制流**：极小的 contact I/P packet，包含 mode、root/foot residual、uncertainty、anchor hash/scene version、prediction horizon、reset/keyframe request。

系统根据预测的接触风险调度，不按固定帧率盲传：

```text
stance transition / penetration risk high / uncertainty high
  -> 发送 contact I-frame 或提前提高优先级

stable stance + low uncertainty
  -> 只发小 P-residual；客户端用 PCWM 预测

packet loss / scene-version mismatch
  -> 禁止大幅 correction，回退并请求 keyframe
```

优化目标从普通 rate-distortion 变为：

\[
\mathcal L_{RCD} =
D_{render} + \lambda_c D_{contact} + \lambda_t D_{temporal}
+ \lambda_r R + \lambda_l D_{late/loss}
\]

其中 `D_contact` 对 penetration、foot sliding、transition 错误和错误强修正施加更高代价。该任务损失是通信部分的学术核心：在同样 bitrate 下，优先保护最容易破坏人—场景可信度的少量变量。

### 3.4 客户端一致性与重同步

客户端不能无界地把预测当事实。必须有：

- scene-GS version / anchor hash，防止在错误场景几何上应用 residual；
- bounded horizon（例如最多预测 H 帧）；
- 置信度门控的 reconciliation：新包到达时用短窗口平滑回真实状态，但接触转换时允许硬 reset；
- packet late/loss trace 下的真实回放评测，而非只在零延迟离线测 PSNR。

## 4. 与研究点一的真正连接

研究点一的 Anchor Attention 不是简单引用，而是 PCWM 的 scene interface：

| 维度 | 研究点一 | PCWM-GS |
|---|---|---|
| Query | SMPL 语义锚点 | SMPL-X 接触锚点 + 历史 contact state |
| Key/Value | local scene Gaussian | local scene Gaussian + metric geometry/proxy |
| 输出 | per-GS 高维位移/opacity | contact mode、低维 residual 分布、packet priority |
| 时序 | 离线序列训练 | 严格因果，短期预测 |
| 目标 | 渲染对齐 | 几何接触、渲染非劣、抗网络抖动 |

因此可以形成一条统一论文叙事：

> 第一阶段证明 scene-conditioned semantic anchors 能改善离线 human-Gaussian alignment；第二阶段研究如何将同一 scene-query 机制压缩成可预测、可通信、可拒绝的低维 human-scene control state，使其在在线环境中保持物理和视觉一致性。

## 5. 实验路线：从必要到有野心

### P0：先完成当前可信接触闭环

必须先完成第一轮计划的 E0–E3：PROX 多场景标签审计、机制 split、Human3R 对齐、IK/projection。没有真实可靠接触链路，预测和通信只会放大错误。

### P1：Contact State Predictor（不含传输）

在 PROX mechanism split 上训练当前+H 帧预测：

- 比较 Rule、GRU、causal Transformer/state-space model；
- 指标：当前与未来 contact F1/transition F1、root/foot residual RMSE、penetration、foot locking、calibration/coverage；
- 消融：无 scene-GS token、无 metric proxy、无历史、无未来损失、无 uncertainty gate。

只有在 P1 的未来预测误差显著低于 last-value/constant-velocity baseline 时，才有资格宣称 world-model/predictive component 有价值。

### P2：网络仿真与控制包

不需要等待正式 GUSH3R renderer，可先用已实现 packet/runtime 做可复现实验：

- trace：0/30/60/120 ms delay、jitter、1/5/10% loss、burst loss；
- 对比：每帧 full update、固定 I/P 周期、现有 delta-only、预测 I/P、预测+风险优先级；
- 指标：bitrate、late-frame ratio、recovery time、stale contact violation、prediction drift、客户端 FPS。

### P3：GUSH3R 渲染闭环

将同一网络 trace 回放到 corrected human Gaussian renderer：

- contact ROI / human / full render 指标；
- contact-aware rate-distortion 曲线；
- 在延迟/丢包条件下的 temporal flicker 和主观视频对照；
- 一个 server GPU + client GPU/CPU 的端到端 latency breakdown。

### P4：可选的高风险扩展

- 多接触 anchor（手扶桌、坐姿臀部），用 RICH/DECO 类型数据扩展；
- 使用 foundation/world-model latent 作为**辅助观测 token**，而不是替代 metric geometry；
- 学习 packet scheduler，动作空间仅限 `send-P / send-I / defer / request-reset`，通过离线 trace 做 contextual bandit，而不是未验证的在线 RL。

这些扩展只有在 P0–P3 主线成功后才做，避免范围失控。

## 6. 新的主表与反证要求

主表必须同时有“无网络”和“有网络”两部分：

1. **Geometry/render**：Raw GUSH3R、EMA、Rule、CAGC、PCWM current-only、PCWM+IK。
2. **Network robustness**：full update、fixed I/P、delta-only、PCWM prediction、PCWM+risk-aware scheduler。

每行报告：contact F1/transition F1、sliding、penetration、uncertainty risk-coverage、contact ROI/full PSNR/LPIPS、flicker、bitrate、p50/p95 latency、loss recovery time。

必须反证以下风险：

- 预测器是否只是平滑器？与 EMA/constant velocity 比较；
- scene-GS token 是否只是更大模型？与同参数量的无 scene-token 对比；
- 低码率改善是否只是牺牲画质？给出 contact-aware RD 和视觉 RD 曲线；
- 世界模型表述是否过度？只报告短期可验证状态预测，不宣称开放世界生成；
- 传输优势是否只在模拟器？至少加入真实 packet trace/replay 和端到端计时。

## 7. 当前可写入论文的贡献候选

在 P0–P3 都成功前，以下只能称为“方法设计”，不是结果：

1. 首个（需最终系统检索验证）将 semantic anchor-to-scene-Gaussian query 用于**在线接触状态控制**的 human-scene 3DGS 框架；
2. 结合 metric geometry 与 scene-GS rendering memory 的双证据、因果且可拒绝的接触状态预测；
3. Contact-Gated LBS：以低维 SMPL-X 变化稳定地更新 human GS，同时约束 Gaussian 接触 ROI 视觉非劣；
4. 面向丢包/延迟的 contact-aware rate-distortion-control：预测性 I/P 包、风险优先级和安全回退；
5. 与第一点共享 anchor-scene interaction 原理，但明确给出离线高维校正到在线低维控制的统一分析。

## 8. 重要边界

- 不把 Human3R/PROX 对齐未通过的结果拿去训练或宣传；
- 不把 synthetic drift 成功等同于真实前馈成功；
- 不把 packet-only CPU smoke test 等同于端到端实时 Gaussian streaming；
- 不把短期结构化预测宣称为通用生成式 world model；
- 不把固定 scene Gaussian 假设推广到可移动物体交互。手物/动态物体需单独建模、单独数据和协议。
