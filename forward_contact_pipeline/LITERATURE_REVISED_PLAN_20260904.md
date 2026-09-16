# 研究点二：文献驱动的方案迭代与实验路线（2026-09-04）

## 0. 结论先行

当前原型“局部 SDF/平面 + causal GRU/TCN + SMPL-X residual”**不能**作为最终论文方法：它证明了工程可行性，但单独看缺少与现有 scene-aware HMR、dense-contact detection 和 physics-based pose refinement 的区分度。

建议将正式方法收敛为 **Contact-Conditioned Anchor Gaussian Controller（暂名，CAGC）**：

> 将研究点一的“语义人体锚点查询局部 scene Gaussian”改造成在线的、接触专用的双证据锚点控制器。它以 SMPL-X 接触锚点为 Query，同时读取局部 scene-Gaussian token 和可信的 metric surface/SDF 证据；在因果时序中预测带方差的低维 root/foot 修正，再经 LBS 更新 human Gaussian。最终以接触几何改善和 Gaussian 接触区域渲染非劣性共同验收。

这不是让网络直接移动高维 Gaussian，也不是把普通姿态回归换名为接触：**场景侧证据决定是否、何处、以何方向修正；SMPL-X 是受控执行器；Gaussian 是最终渲染与传输对象。**

## 1. 截至本次审计的真实状态

### 已完成

- `contact_streaming/` 已有统一 58D 输入、38D 输出、局部 surface proxy、规则状态机、GRU/causal TCN、训练/验证、LBS adapter、packet/runtime；核心单元测试 4/4 通过。
- PROX 的 RGB/Depth、PROXD、scene mesh/SDF、`cam2world` 已到位。`build_prox_geometry_labels.py` 已在一个 180 帧序列验证 SDF 有效覆盖，并修复“每帧最近顶点切换导致假低脚速”的标签问题。
- Human3R-on-PROX 已跑通 60 帧真实 RGB 前向，SMPL-X/point map/FrameInput 已落盘。
- GUSH3R 单帧 human/scene Gaussian 推理与渲染已跑通；长序列背景回渲污染已定位，`chunked`/causal background renderer 已实现。

### 已证伪或暂停

- Human3R-to-PROXD 的 45/15 hold-out Sim(3) 顶点误差：median 7.37 cm、p95 45.1 cm。该精度不满足脚部接触训练，禁止将其作为真实前端残差监督。
- 旧 B 层 `scene-disjoint` synthetic-drift split 的 train/val/test 接触正例率为 84.58% / 53.33% / 12.50%。GRU 与 contact-only GRU 都接近全正预测：test F1=0.218、transition F1=0；TCN 回归发散。它只能保留为 domain-generalization 压力测试，不能判断模块不可学习。
- 尚未完成可微 IK、corrected mesh 可视化、GUSH3R 内部 corrected-SMPL-X 重算/正式 renderer 对照。因此不能声称已有 GUSH3R 接触改善。

## 2. 关键文献与对方案的约束

| 工作 | 已做什么 | 对我们的约束/机会 |
|---|---|---|
| HUGS, CVPR 2024, arXiv:2311.17910 | SMPL 驱动的人体 Gaussian 表达与高质量渲染 | 证明 LBS/人体 Gaussian 是合适执行器；本工作不能只重做人体 GS 形变。 |
| Human3R, arXiv:2510.06219；GUSH3R, arXiv:2607.05243 | 因果前馈的人体—场景重建；GUSH3R 输出 human/scene 3DGS | 它们是 frozen backbone / 正式主 baseline；当前没有接触校正目标。 |
| SA-HMR, CVPR 2023, arXiv:2306.03847 | 预扫描场景中，先预测绝对位置和稠密接触，再以 cross-attention 融合 3D scene cues 的单帧 HMR | 不能声称“scene-aware cross-attention”本身新；我们的区别必须是 **Gaussian scene token + causal correction + uncertainty + rendering closure**。 |
| BSTRO/RICH, CVPR 2022, arXiv:2206.09553 | 以 Transformer 从单 RGB 估计稠密全身接触；RICH 提供 vertex-level contact | 不能声称“从图像预测接触”新；RICH 可作为将来手/身体接触扩展数据。 |
| DECO, ICCV 2023 Oral, arXiv:2309.15273 | body-part 与 scene-context attention 的稠密 3D contact detector | 不能只做 body/scene attention contact classifier；我们的输出必须直接控制可渲染的人体 GS。 |
| PhysCap, TOG 2020；PhysDynPose, arXiv:2507.17406 | 通过物理/场景约束修正单目人体运动；后者处理移动相机和非平地 | 不能把“脚底贴地 + 优化”包装为新；需要强调前馈、低延迟、Gaussian 端、可拒绝校正，并与优化式方法比较延迟。 |
| UnderPressure, CGF 2022 | 足接触检测、受力估计与 footskate cleanup | foot locking / contact transition 是必报指标；只报 MPJPE 或 contact F1 不够。 |

本轮已保存关键原文 PDF 于 `references/papers/`，并以 arXiv/OpenAlex 的官方元数据和摘要完成第一轮核对；完整方法细节、页码引用与 BibTeX 仍应在论文写作阶段逐篇复核，不能只依赖搜索摘要。

## 3. 修订后的方法：CAGC

### 3.1 与研究点一的统一科学叙事

```text
研究点一：Anchor Attention（离线、逐序列、高维）
SMPL semantic anchors + local scene Gaussians
  -> per-human-Gaussian transl/xyz/opacity correction
  -> 改善离线 human-scene rendering alignment

研究点二：CAGC（在线、因果、低维、物理语义明确）
SMPL-X contact anchors + local scene-Gaussian tokens + metric surface/SDF
  -> contact state + contact-frame root/foot residual distribution
  -> LBS 更新 human Gaussian，保持 scene Gaussian 不动
  -> 改善脚滑/穿透/抖动，且接触 ROI 渲染不劣化
```

共同点是“以人体语义锚点向局部场景查询证据”；区别是研究点一允许高容量、离线的 per-GS 对齐，研究点二必须在在线约束下只控制少量可解释自由度。

### 3.2 双证据 Contact Anchor Encoder（方法核心一）

第一版只做左右脚，接口保留 K 个 contact anchors，后续可扩展手、臀部和背部。

对第 `k` 个脚锚点，建立局部接触坐标系（surface normal 为法向；历史运动方向定义切向）。它同时读取两类互补证据：

1. **metric geometry token**：SDF 或经验证的局部 surface proxy，含 signed distance、normal、support、planarity、距离场梯度、相机/Sim(3) 对齐残差。
2. **scene Gaussian token**：锚点邻域的 scene GS，含相对接触坐标、scale、rotation/normal proxy、opacity、visibility、颜色/图像特征及时间一致性。以锚点为 Query、局部 GS 为 Key/Value 做 cross-attention。

二者不可互相替代：SDF/proxy 提供尺度可信的物理约束；Gaussian token 提供真实最终渲染表示的边界、遮挡和外观上下文。必须以消融验证二者各自的必要性。

### 3.3 因果接触状态与低维控制（方法核心二）

使用 history 8 的 causal state-space Transformer 或小型 GRU 作为效率 baseline。输入只含当前和历史帧，禁止未来信息：

\[
h_t = f(h_{t-1}, z^{anchor}_{t}, z^{motion}_{t})
\]

不直接预测任意的完整 SMPL-X，而是输出：

- 接触状态 `s_t ∈ {swing, approach, stance, slip, uncertain}`；
- 接触概率与 transition probability；
- contact-frame residual：法向修正、切向速度修正、root 6D residual、脚踝/脚部有限 DOF residual；
- residual 的均值 `μ` 与尺度/方差 `σ`；
- correction acceptance（是否执行）与 reset/keyframe flag。

接触帧参数化的价值是：法向分量用于消除 float/penetration，切向分量用于抑制 stance 时脚滑；它比世界坐标任意 6D 平移更可解释、更难投机。

### 3.4 可信度校准与拒绝机制（方法核心三）

目前的 `uncertainty` 是几何可用性 BCE 标签。正式版本升级为两部分：

- **data uncertainty**：由 SDF/proxy 有效性、support、planarity、visibility、Human3R-to-world alignment residual 构造；
- **predictive uncertainty**：网络预测 residual 的 `log σ²`，使用异方差 NLL 训练；可在小型 ensemble/MC-dropout 校验 epistemic uncertainty 是否必要。

修正只在 `p(contact)` 高、几何置信度高、预测方差低时执行。否则回退 raw backbone，并记录 `Uncertain`/`Reset`。必须报告 coverage-risk 曲线，而不只报告一个平均 uncertainty。

### 3.5 Contact-Gated LBS 与 Gaussian 闭环（方法核心四）

```text
Δroot / Δfoot pose
  -> SMPL-X forward kinematics + joint-limit IK
  -> LBS transforms
  -> posed human Gaussian centers / rotations
  -> 与固定 scene Gaussian 同一相机渲染
```

只更新 human Gaussian；scene Gaussian 固定，避免网络以移动地面或重写外观掩盖接触错误。GUSH3R 端的训练先冻结 backbone；只有几何闭环稳定后，才考虑只解冻 human-GS deformation/appearance adapter 的最后层。

## 4. 训练目标

### 4.1 几何/控制主损失

对可信帧使用 mask `m_t`：

\[
\mathcal L =
\lambda_c L_{state} +
\lambda_g L_{surface} +
\lambda_r L_{residual} +
\lambda_p L_{penetration} +
\lambda_l L_{footlock} +
\lambda_s L_{smooth} +
\lambda_u L_{uncertainty} +
\lambda_a L_{anatomy} +
\lambda_G L_{GS}
\]

- `L_state`：contact/transition 分类；类别权重不能在失衡 test 上临时调，必须先固定 split 和 PR 阈值。
- `L_surface`：contact point、signed distance、normal。
- `L_residual`：修正后 root/foot joints 到 PROXD 教师的 robust loss；B 层以注入 drift 的已知 inverse residual 为监督，C 层以对齐后的前馈—PROXD 差为监督。
- `L_penetration`：可信 SDF 下 `ReLU(-d)`；`L_footlock`：stance 时修正后脚在 contact frame 的切向速度接近零。
- `L_smooth`：只在相同接触状态内约束加速度/jerk，避免把真实起步动作过度平滑。
- `L_uncertainty`：几何可用性 BCE + residual heteroscedastic NLL；低置信度帧不强迫 pose target。
- `L_anatomy`：joint limit、最小修正和左右脚一致性。

### 4.2 仅在 D 层启用的 Gaussian 辅助损失

`L_GS` 不是接触真值的替代物，且只对可信 contact ROI 使用：

- corrected-vs-raw 的 human/contact-ROI RGB、mask、LPIPS/SSIM 非劣约束；
- human Gaussian temporal flicker 约束；
- 将经 LBS 变换的脚部 Gaussian center 查询可靠 SDF，得到 Gaussian-level penetration barrier。

这保证网络不能靠破坏脚部外观、opacity 或渲染来“赢”几何指标。全图 PSNR 下降上限预注册为 0.1 dB。

## 5. 分阶段实验：进入条件不可跳过

### E0：数据与标签审计

1. 在至少 4 个 PROX 场景、多个 subject/take 上构建 manifest。
2. 可视化 RGB + PROXD mesh + SDF zero level + foot anchors + normal；统计有效率、正例率、距离和速度分布。
3. 固定 label version、坐标约定、许可证、split 清单；所有后续 run 只读该 manifest。

**通过条件**：SDF、RGB、Depth、PROXD 在所有选用序列上坐标一致；不确定帧可定位且不会被伪标为接触。

### E1：规则与受控机制数据

1. 从同一 PROX scene/action 的 clean teacher 制作多个互不重叠 AR(1) drift seeds。
2. 每个 scene/action 同时出现在 train/val/test，但 raw recording 与 drift seed 都不泄漏；另保留 scene-disjoint split 作为 domain-generalization 压力测试。
3. 先冻结阈值的 Raw、EMA、Rule baseline，再跑 contact-only、multi-head GRU、causal TCN。

**通过条件**：在 mechanism split 上，GRU/TCN 至少在 contact transition、root/foot residual 和 penetration 的一组联合指标优于 Rule；否则先检查标签/参数化，不扩大模型。

### E2：真实 Human3R 前端（C 层）

1. 将 Human3R SMPL-X 与 PROXD 拆解为 camera、global scale/rotation、root、local pose 四类误差。
2. 使用 train-frame 对齐、hold-out-frame 验证；做 RGB mask/mesh reprojection 和 anchor-to-SDF 可视化。
3. 以**脚 anchor 误差**而非全身平均顶点误差决定是否进入学习；不达标则标 high uncertainty、不得造残差标签。

**通过条件**：对齐达到预注册接触尺度门槛，并在真实前馈输入上 Rule 已不劣于 Raw；否则停在诊断而非宣布学习结果。

### E3：IK/projection 执行器

1. 实现 root correction、局部脚踝/脚部 IK、joint limit 和 contact-frame projection。
2. 逐项比较 Raw → Rule → Rule+projection/IK，输出 mesh overlay、foot trajectory、失败帧归因。

**通过条件**：真实序列至少一个接触指标改善，且不引入明显姿态跳变。

### E4：GUSH3R Gaussian 正式主实验（D 层）

1. 完成 `corrected SMPL-X -> GUSH3R human Gaussian` 的内部 LBS 重算；scene Gaussian 固定。
2. 在同一视频、同一相机、同一 chunked background renderer 下渲染 Raw/EMA/Rule/GRU/GRU+IK。
3. 报告 contact、render、system、安全四类指标及 per-sequence 结果。

**通过条件**：接触指标改善，human/contact ROI 不劣化，全图 PSNR 满足非劣约束，端到端至少 30 FPS 或明确报告未达标原因。

## 6. 必做消融与反证

| 编号 | 去掉/替换项 | 要回答的问题 |
|---|---|---|
| A0 | Raw GUSH3R | 原始前馈误差多大？ |
| A1 | EMA | 改善是否只是平滑？ |
| A2 | Rule + metric geometry | 学习是否超过显式几何规则？ |
| A3 | 无 scene-GS token，只用 SDF/proxy | scene Gaussian 查询是否提供额外价值？ |
| A4 | 无 SDF/proxy，只用 scene-GS token | learned rendering representation 是否足够可靠？ |
| A5 | 无 temporal memory | 因果时序是否降低 slip/jitter/transition error？ |
| A6 | 无 uncertainty gate | 拒绝机制是否降低坏帧风险？ |
| A7 | arbitrary world residual vs contact-frame residual | 接触帧参数化是否更稳定、更小幅？ |
| A8 | 无 Gaussian ROI loss | 几何改善是否以视觉劣化为代价？ |
| A9 | HUGS Anchor Attention（离线参考） | 在线低维方法距离离线质量上界多少？不作为公平实时主表。 |

## 7. 论文级实验与可复现性要求

按 NeurIPS Paper Checklist 和 CVPR 2026 Author Guidelines，主张必须与证据范围匹配。每个 run 必须保存：

- dataset manifest、许可/版本、训练/验证/测试 recording-level split；
- 坐标系、SDF 分辨率、标签规则、阈值冻结时点；
- 模型配置、随机种子、权重、命令、环境、GPU/显存/训练时间；
- 每序列指标、均值和 bootstrap CI 或多 seed error bars，禁止只报 pooled best run；
- failure cases（遮挡、无 SDF、快速动作、坐姿、非平地、对齐失败）及 uncertainty coverage；
- Raw/EMA/Rule/learned/IK 的同帧可视化和 contact ROI 结果；
- 人体数据许可、隐私/再识别风险及模型使用边界。

避免以下不成立的表述：

- “在 GUSH3R 上有效”：在 E4 前不能说；
- “真实前馈监督”：在 E2 hold-out 对齐通过前不能说；
- “跨场景泛化”：旧 split 的分布偏移修复并报告 per-scene/PR calibration 前不能说；
- “实时”：必须同时报告前端、接触模块、LBS、renderer 的端到端延迟。

## 8. 文献与官方规范入口

- SA-HMR: `https://arxiv.org/abs/2306.03847`
- BSTRO/RICH: `https://arxiv.org/abs/2206.09553`
- DECO: `https://arxiv.org/abs/2309.15273`
- PhysDynPose/MoviCam: `https://arxiv.org/abs/2507.17406`
- HUGS: `https://arxiv.org/abs/2311.17910`
- Human3R: `https://arxiv.org/abs/2510.06219`
- GUSH3R: `https://arxiv.org/abs/2607.05243`
- NeurIPS Paper Checklist: `https://neurips.cc/public/guides/PaperChecklist`
- CVPR 2026 Author Guidelines: `https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines`
