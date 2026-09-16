# 前馈 Gaussian 工作对第二研究点的接口更新（2026-09-09）

本记录核验 arXiv 原始摘要，不把论文方法或其未复现实验写成我们的结果。

| 工作 | 已核验设定 | 对我们的可用接口 | 不应混淆的边界 |
|---|---|---|---|
| Splat-SAP（AAAI 2026，arXiv:2511.22704） | 大稀疏度双目 human-centered scene；scale-aware point map、target-plane stereo refinement、Gaussian anchor | 用 point map/置信度/法向建立接触模块的 local scene proxy；将其 point-map confidence 作为 uncertainty gate 的观测项 | 双目与目标视图平面依赖强；不是长时 online map，也不解决人体姿态或接触 |
| HumanGS（arXiv:2604.10259） | 稀疏多视图 RGB + SMPL-X；反投影多尺度特征到 SMPL-X vertices，Transformer 聚合，MLP 输出 canonical Gaussians，LBS 动画 | 作为 GUSH3R 人体分支的更稳候选：接触模块只修正 SMPL-X root/foot，HumanGS canonical human Gaussians 经 LBS 更新 | 不重建周围场景；需要多视图和关联 SMPL-X pose；尚未在本机复现 |
| Hand-4DGS（arXiv:2606.19156） | 第一人称视频，mesh-guided prior + temporal convolution，约 60 FPS | 后续手-物接触扩展时采用 mesh-guided token、时间卷积和遮挡/visibility 状态 | 对象是手，不可替代完整人体—场景 baseline |
| PointSplat（arXiv:2606.32036） | input point set -> coarse proxy -> ray casting pruning -> compact human-centric Gaussians，面向直播带宽 | 与 HUGS 第一研究点连接：把 corrected SMPL-X / contact state 与 compact point/Gaussian packet 共同传输；ROI 做预算分配 | 人体为主，不提供场景几何教师或接触监督 |
| GaussianLens（arXiv:2509.25603） | 按需局部高分辨率 Gaussian densification | contact ROI（鞋底、手、交互物）触发局部细化，接触不确定性同时作为 densification/bitrate gate | 不是人景前馈重建或接触方法；不应在 E1 先引入 |

## 更新后的分层方案

```text
双目/多视图 RGB
  ├─ Splat-SAP-style scale-aware point map -> local scene proxy (point, normal, confidence)
  └─ HumanGS-style SMPL-X canonical human Gaussians -> corrected SMPL-X LBS
                         │                         │
                         └─ causal contact controller ┘
                                  │
                       uncertainty / contact ROI gate
                                  │
              PointSplat-style compact packet + GaussianLens-style ROI budget
```

因此第二研究点的核心创新仍是跨表示的在线接触控制：**不改变场景 Gaussian 来消除接触错误，而是用不确定性感知的 local scene proxy 校正低维 SMPL-X，再以 canonical/LBS 人体 Gaussian 和紧凑 packet 实现稳定显示与传输。**

## 优先级

1. 完成 E1 因果 controller 的机制闭环（当前进行）。
2. E2 中将 oracle SDF 替换为 Splat-SAP-style noisy point-map proxy，评估 confidence-aware abstention；这是最直接、数据/算力可承受的论文接口。
3. 比较/接入 HumanGS 作为 human branch；若可复现且数据设定匹配，则它比当前 GUSH3R 更适合 corrected-SMPL-X -> canonical Gaussian -> LBS。
4. 最后才加入 PointSplat/GaussianLens 的传输与接触 ROI 自适应预算。
