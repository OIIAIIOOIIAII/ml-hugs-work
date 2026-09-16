# A4：UniCon3R 候选审计与 Gaussian 接触证据诊断

## UniCon3R

官方仓库已确认（commit `9b248489d3febfb52244f7287c32399a6f502542`），但只有项目页、README 与
网页 demo；README 明确声明代码、checkpoints、安装和 inference scripts “will
be released soon”。故当前不能复现、不能在 PROX 上做前端门槛测试，也不能作
可运行 baseline。

网页 demo JSON 确实包含每帧 `contactIndices`（SMPL-X 10475 顶点索引）和 temporal
contact evolution，说明 UniCon3R 的公开展示是 dense vertex contact；但没有模型
输出、训练标签来源、坐标/scene geometry 或权重。这些 JSON 仅可指导未来 RICH/EMDB
稠密监督的数据 schema，不能拿来训练或对比 HUGS。

## GUSH3R Gaussian center vs. PROX SDF

使用 GUSH3R 60 帧 per-frame causal state（每帧 top-2048 scene Gaussian）与前45帧
拟合、后15帧失败的 Human Sim(3) 进行 SDF support audit：

- SDF coverage：22.4%
- 有效 center 的 |SDF| median：33.9 cm
- |SDF| <= 2cm：9.6%
- SDF < -5mm：95.6%

该实验的 coordinate contract 已知失败，故不能把上述数值解释为真实 scene quality；
它严格证明了反方向的命题：Gaussian center + coordinate alignment + SDF threshold
不构成可用的显式接触规则。Gaussian 只能进入学习到的 local relation/reliability
token，不能替代 surface。

## Oracle occupancy representation test

在现有 Stage-A near-contact ROI 上，99.997% vertex 已满足 |SDF|<=2cm，无法进行
二分类比较；nearest-point 与手工 anisotropic Gaussian kernel 的 continuous proximity
Spearman 分别为 .986 / .964。扩到同一 PROX L_Leg segment 的 full-leg FPS ROI 后，
仍全部 |SDF|<=2cm（MPH112 max 1.15cm）。因此当前 PROX segment/SDF sampling contract
不具备 “contact vs non-contact vertex” 支持，停止调 Gaussian kernel；这不是 Gaussian
表示失败结论，而是监督分布不适合该问题。

## 当前决策

1. 研究点二保留 implicit Gaussian interaction field：AnchorAttention relation token
   + RGB + SMPL semantics；Gaussian covariance/opacity 用作软 evidence/reliability，
   不用 hard occupancy threshold 判接触。
2. 真实前端实验被两项外部条件阻塞：现有 Human3R/GUSH3R 未过2cm gate，UniCon3R
   尚无代码/weights。
3. 下一可验证数据工作不是继续调网络，而是接入带真正 dense vertex contact 的
   RICH/PhySIC 类公开/获许可数据，或等待 UniCon3R release 后在 PROX 统一协议复测。
