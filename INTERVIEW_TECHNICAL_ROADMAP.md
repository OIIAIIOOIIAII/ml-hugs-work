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

### 2.2 为什么不直接优化所有 Gaussian

逐帧移动数十万 human/scene Gaussian 的问题：高延迟、易漂移、不可解释、破坏场景静态性，而且很难保证接触期足端速度为零。

我们的可控变量是约 20–40 维：左右脚 contact、切向 action、root/ankle/foot residual、uncertainty/reset。执行时 action 被投影到局部切平面，并做 `tanh` 限幅。场景 GS 不随接触模块移动；纠正只经 SMPL-X/LBS 传给人体 GS。

### 2.3 接触状态与因果控制

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

### 2.4 分阶段技术路线

| 阶段 | 要验证的命题 | 数据/输入 | 前进门槛 |
|---|---|---|---|
| E0 | 坐标、SDF、脚 anchor 是否可信 | PROXD + PROX SDF/scene scan | 坐标/单位/法向/覆盖审计通过 |
| E1 | 接触控制机制是否成立 | Oracle PROXD + Oracle SDF + 注入 drift | penetration、sliding、jitter 同时改善 |
| E2 | 局部场景退化时能否安全 | point-map/proxy 噪声、normal误差、dropout、delay | uncertainty/abstain 不让风险显著恶化 |
| E3 | 真实人体前端能否接入 | Human3R 或其他前馈 human state | held-out 几何到接触尺度（约2cm） |
| E4 | Gaussian/scene-state 接入是否不伤视觉 | frozen controller + local Gaussian proxy | contact ROI 几何改善且 render/flicker/latency 达标 |

这套路线的好处是可证伪：前端不达门槛就停在 diagnostic，而不是用 oracle 标签伪造在线结果。

### 2.5 已完成的、可以讲的实证结果

以下都应明确标为 **oracle-scene mechanism result**：使用 PROXD + 官方 PROX SDF，人工注入 root/foot drift；不是 Human3R/GUSH3R 的真实在线结果。

1. **E0 契约已通过。** 四个 PROX 序列、720 帧，SDF、法向、坐标系 `PROX_scene_world` 有效。
2. **法向 clearance 只能部分解决问题。** GRU 将 penetration depth 从 18.57 mm 降至 13.95 mm、contact |distance| 从 29.28 mm 降至 24.58 mm，但 sliding 从 0.643 升到 0.927 m/s。结论：法向 clearance 与 tangential stick 必须分开。
3. **contact anchor 表示成立。** oracle anchor lock 可将 sliding 从 0.385 m/s 降至 0.028 m/s，证明接触 episode anchor + foot-lock 的执行表达正确。
4. **因果 rollout 机制通过。** 使用模型自身状态、切平面 action、bounded soft gate 的 control-first checkpoint，把 sliding 从 0.385 降至 **0.218 m/s（-43.4%）**，contact F1=0.817。state-first checkpoint 的 contact F1=0.863、transition F1=0.550，但 sliding=0.281 m/s。应报告 Pareto，而不是 cherry-pick 单一“最好”。
5. **中等场景退化暴露关键风险。** 冻结 controller 在 normal 18°、distance noise 2cm、bias 1cm、20% dropout、2-frame delay 条件下，sliding 恶化到 0.463 m/s；简单 threshold 只有 100% abstain 才能回到基线。几何增强能将 sliding 降到 0.276 m/s，但 F1=0.722、transition F1=0.142，仍不够安全。
6. **连续 proximity 是比伪 dense binary contact 更可靠的 Stage-A 目标。** local relation encoder 在严格 split 上 signed proximity MAE=**1.27 mm**，好于 constant-zero 的 1.79 mm 和 nearest-point 的 12.64 mm；但 ROI contact head all-positive，不能拿 F1 当成功结果。

### 2.6 为什么现在不能说“GUSH3R 接触系统已完成”

这是最重要的诚实边界。

- Human3R 到 PROXD 的 held-out Sim(3) 几何误差：median **7.37 cm**、p95 **45.1 cm**；
- GUSH3R coarse SMPL-X 到 PROXD 的 held-out 误差：median **19.95 cm**、p95 **54.69 cm**；
- 接触所需几何尺度约为 2 cm。

因此当前 GUSH3R 可以提供 per-frame Gaussian evidence（center/covariance/opacity）和 renderer/degraded-map diagnostic，但不能作为可靠 surface、contact teacher 或真实前端主表。正确的表述是：

> 已完成前馈 Gaussian 接口与退化诊断；已验证 oracle/degraded geometry 下的接触机制；真实前端接入被 held-out 几何门槛明确阻断，下一步需要更强前端或完成 camera-local/reprojection 误差契约。

这比忽略误差、宣称“前馈 Gaussian 已实现物理接触”更能体现研究严谨性。

### 2.7 研究点二最终评测表应如何设计

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
