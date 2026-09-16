# 前馈接触修正自动研究协议（2026-08-26）

## 1. 研究目标与可证伪主张

本研究的正式对象是：在冻结的前馈人体-场景 Gaussian backbone（最终为 GUSH3R）之后，使用局部人体-场景 anchor、低维 SMPL-X residual 和不确定性回退，修正脚部接触错误。

主假设 H1：相对于原始前馈输出，在相同的输入视频、Gaussian 表达和渲染器下，接触感知的低维修正可降低 foot sliding、penetration、contact jitter，并保持接触 ROI 与全图的渲染质量。

非劣约束 H2：接触模块不能通过过度平滑或破坏姿态换取几何指标；全图 PSNR 的下降不得超过 0.1 dB，human/contact ROI 的视觉指标必须单独报告。

系统假设 H3：修正只传 root/foot/contact 的低维 packet，仍可在至少 30 FPS 运行；其 packet 和延迟优势应与既有 Gaussian `delta_mu` 传输系统并列报告。

本协议不接受以下替代结论：

- 只在 Human3R mesh/point map 上有效，即宣称 GUSH3R Gaussian 已有效；
- 只在由同一几何规则生成的伪标签上取得 F1=1；
- 只报告 PSNR，未报告脚部接触、穿透和不确定性回退；
- 在训练、验证、测试中混入同一原始视频的相邻帧。

## 2. 与第一研究点的统一叙事

第一研究点的 Anchor Attention/Global Scene Gate 是**离线、逐序列、高容量**的人体-场景 Gaussian 校正：从 SMPL 语义 anchor 查询局部 scene Gaussian，以高维 per-Gaussian transl/xyz 修正和 global opacity gate 提升人-场景对齐与渲染质量。

本研究点将其改写为**在线、前馈、低维、物理语义明确**的校正：

```text
Anchor Attention (研究点一)
  semantic anchors + scene context
  -> per-human-Gaussian high-dimensional displacement / opacity
  -> offline rendering-quality optimisation

Contact Correction (研究点二)
  feet anchors + local surface/SDF + temporal state
  -> contact state + root/foot low-dimensional residual + uncertainty
  -> SMPL-X LBS propagates correction to human Gaussians
  -> online contact quality, latency and packet efficiency
```

共同科学问题是“人体 Gaussian 如何与场景几何一致”；差别是校正自由度、监督和部署场景。正式实验应将 HUGS/Anchor Attention 作为质量上界/离线 teacher 讨论，而非声称两种方法可在不相同的 backbone、输入和训练预算下直接数值公平比较。

## 3. 数据与监督层级

| 层级 | 数据 | 输入 | 监督 | 能回答的问题 |
|---|---|---|---|---|
| A | PROX oracle | PROXD SMPL-X + official scene SDF | SDF contact/distance/normal/penetration | 坐标与标签是否可靠；规则和时序模块的上限 |
| B | PROX synthetic error | A 的 GT 加可控 root/foot drift | 已知 clean correction residual | 网络能否从受控前端误差恢复接触 |
| C | PROX front-end | Human3R on RGB-D/RGB + PROX GT | predicted-vs-GT contact | 真实前馈几何误差下是否有效 |
| D | NeuMan/GUSH3R | GUSH3R human/scene Gaussian | 仅高置信度 pseudo/人工审计 + rendering | 正式 Gaussian 质量、速度和传输影响 |

层 A/B 不能替代 C/D。A/B 的作用是防止在“坐标不对、标签不对、训练目标不对”时浪费 GPU；正式论文主结论必须至少在 C/D 成立。

## 4. PROX 标签构造的固定定义

标签由 `scripts/build_prox_geometry_labels.py` 生成：

1. 从 PROXD 读取每帧拟合的 SMPL-X 参数；
2. 将 SMPL-X 顶点通过该场景官方 `cam2world` 变换到 PROX scene world；
3. 在左/右腿顶点区域查询官方 256^3 signed-distance field；
4. 选取离零水平集最近的脚部顶点，以 SDF gradient 获得 surface normal 和投影 surface point；
5. 使用固定 SMPL-X left/right foot joint 的世界轨迹计算速度，禁止用每帧被选中的最近顶点计算速度；
6. `contact = |signed distance| <= 2 cm AND foot speed <= 0.15 m/s`；
7. `penetration = signed distance < -5 mm`；场景外/SDF gradient 无效时 `uncertainty=1`。

该构建器只输出有真实几何依据的 `contact`、`contact_point`、`surface_distance`、`surface_normal`、`penetration_risk` 和 `uncertainty`。它**不输出全零 pose/root residual**；残差监督只能在层 B 的可控扰动或层 C 的真实前端预测误差中构造。

## 5. 实验设计与比较

### 5.1 固定主 baseline

1. Raw：前馈 backbone 原始 SMPL-X/Gaussian；
2. EMA：仅对 root/foot trajectory 平滑；
3. Rule：距离 + 速度 + normal + hysteresis，带 uncertainty fallback；
4. GRU：冻结 backbone 的 causal GRU；
5. TCN：同等输入/输出维度的 causal dilated TCN；
6. GRU/TCN + projection/IK：完整方法；
7. HUGS Anchor Attention：只作为离线人-场景校正参考或 teacher，不混入同一主表的实时声称。

### 5.2 消融顺序（禁止跳步）

| 阶段 | 唯一变量 | 进入条件 | 停止/回滚条件 |
|---|---|---|---|
| E0 | PROX 标签与坐标 | SDF 覆盖、帧名匹配、人工可视化审计通过 | SDF/mesh 坐标错或标签塌缩 |
| E1 | Rule vs Raw | oracle/synthetic 数据 | Rule 未改善至少一个接触指标，先查标签，不训网络 |
| E2 | GRU vs Rule | B 层有 train/val/test 严格 split | val 指标未优于 Rule，停止扩模型 |
| E3 | TCN vs GRU | E2 成立 | 仅保留更优或更快者 |
| E4 | projection/IK | E2/E3 成立且无不连续跳变 | 渲染或 jitter 恶化则回滚 |
| E5 | Human3R front-end | GPU 可用、C 层对齐通过 | 对齐/GT 失败时不报告学习结果 |
| E6 | GUSH3R LBS renderer | E5 成立 | render 非劣约束不满足则仅报告几何结果 |

### 5.3 数据 split

- split 的最小单位是完整 PROX recording，绝不按帧随机 split；
- 默认按场景和 subject/take group 划分，训练/验证/测试不共享同一原始视频；
- 所有 split、阈值、随机种子、标签版本、源文件清单写入 run directory；
- 开发集可用于选阈值；测试集在模型和阈值冻结后只运行一次主报告。

### 5.4 必报指标

**接触**：contact precision/recall/F1、transition F1、foot sliding、contact jitter、signed distance MAE、penetration ratio/depth、float rate。  
**渲染**：full/human/contact-ROI PSNR、SSIM、LPIPS，human Gaussian temporal flicker。  
**系统**：模块 FPS、end-to-end latency、GPU memory、packet bitrate、late/loss recovery。  
**安全**：uncertainty coverage、低置信度回退率、回退帧上的大幅 residual 发生率。

## 6. 当前证据与下一步

2026-08-26 的 180 帧 `BasementSittingBooth_00142_01` CPU 审计显示：

- PROXD、RGB、Depth 按 frame name 一一对应；
- PROXD body 经过 `cam2world` 后，左右脚 SDF query 有效率为 100%；
- 脚部最近 surface 的 signed-distance 中位数为毫米级；
- 初版最近顶点速度会因顶点身份切换而低估 contact；已修正为固定 foot-joint 轨迹，修正后接触率为左 10.6%、右 7.8%；
- 这只证明标签管线在一个短窗口可用，不证明整个 PROX 或最终 GUSH3R 主张。

下一步：跨至少四个场景重复 E0，输出 manifest 与质量报告；之后构造 B 层可控扰动，先在 CPU 上跑 Rule/GRU/TCN 的可复现小规模研究，GPU 空闲后再进行 Human3R/GUSH3R 的 C/D 层实验。

更新：已新增 `scripts/make_prox_synthetic_drift.py` 并用 60 帧 MPH112 smoke test 验证。它以 AR(1) 时序相关的 root/foot 误差扰动 clean teacher anchor，输出 58D features、已知的 6D root 与 12D foot correction target；该 smoke 的 root/foot correction RMS 分别约为 2.3 cm/1.9 cm，且会产生可测的输入穿透。此数据只用于 E2 前的机制测试。

更新：Human3R-on-PROX MPH112 60 帧真实导出已完成（0.312 s/frame，1 人/帧）。按同帧、同 SMPL-X topology 的 45/15 train/hold-out Sim(3) 评估，hold-out vertex median=7.37 cm、p95=45.1 cm、mean=12.4 cm，confidence=0.384。这未达到脚部接触训练的几何验收，C 层训练暂停；下一自动阶段是分解相机、root 和局部姿态误差并做重投影/anchor-SDF 可视化，禁止将该结果接入 contact learner。

更新：以四个互斥 PROX 场景构造 B 层 synthetic-drift（2 train / 1 val / 1 test）后，30-epoch GRU 在未见 BasementSittingBooth 测试序列上 contact F1=0.218、transition F1=0、surface distance MAE=9.07 cm、root residual RMSE=5.98 cm、预测 penetration ratio=50%。这不是可接受的泛化结果；它说明当前 60 帧/场景、单一固定噪声和多任务未加权目标不足，不能作为方法有效性主张。TCN 作为同预算结构对照已启动，结果将与此负基线并列记录。

更新（B 层数据集诊断）：`diagnostics/prox_synth_aug_contact_separability.json` 对 24 条多 seed 序列执行了只用训练 scene 选择的 distance/speed/confidence rule grid，再冻结阈值报告 val/test。发现接触正例比例在 train/val/test 分别为 84.58% / 53.33% / 12.50%；训练选择的最高 F1 规则几乎全预测接触（test predicted positive 92.92%，test F1=0.237）。这解释了此前 GRU 的 test F1=0.218 接近 all-positive 类先验，且表明当前“一 scene 一动作片段”的 scene split 同时改变了动作/接触分布。该 split 仍应保留为困难的跨场景泛化诊断，但**不能单独用于判断 B 层机制是否可学习**，也不能用阈值或 class weighting 美化结果。后续预注册为两条独立实验：(i) 受控机制 split：每个 scene/动作在 train/val/test 都有独立 drift seed，检验能否恢复已知 residual；(ii) 跨场景泛化 split：保持 scene-disjoint，额外报告 prevalence、PR/ROC、calibration 和 per-scene 指标，并只将其称作 domain generalization。正在运行的 contact-only GRU 只用于隔离 multi-task negative transfer；其几何回归指标不解释为有效性。

更新（contact-only 对照完成）：`runs/prox_synth_aug_gru_contactonly/` 以与首轮 GRU 相同的 history=8、hidden=128、seed=42、30 epochs 运行；仅将 surface/velocity/penetration/pose/temporal/uncertainty loss 权重设为零。best val loss=0.692049（epoch 18），固定 test 为 precision=0.12264、recall=1.0、F1=0.21849、transition F1=0，**与多任务 GRU F1=0.218 等同**；分类 head 仍预测近乎全部为接触。故否定“仅多任务负迁移是首要原因”的解释，且不再在旧 scene split 上做 class weighting/focal/结构调参。下一项是构建预注册的同 scene/动作、互斥 drift seed 的 B-mechanism split；旧 split 仅作为 domain-generalization 压力测试保留。
