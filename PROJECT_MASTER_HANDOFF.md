# 项目总交接与阅读地图

这是新机器、新 agent 的最高层入口。先读本页，再按下方顺序进入具体文档；不要从数千行历史日志或旧 TODO 中猜测当前任务。

## 2026-09-18 当前结论

项目的近期主线是：冻结 GUSH3R 前端，利用 RICH 的真实稠密人体—场景接触监督，训练局部 RGB--mesh--point Stage-A 接触估计器；随后才研究时序控制、人体低维修正与 human Gaussian LBS 回写。

首个真实全量 RICH Stage-A 实验 `rich_full_contact_v1_seed42_v2` 已完成 30/30 epoch。内部 ParkingLot2 开发验证的最佳 checkpoint 在 epoch 5：all-candidate AP=0.94421、F1=0.86902、precision=0.90091、recall=0.83932；成功匹配条件下 AP=0.97444、F1=0.92168；目标关联覆盖率=0.89676。它优于 all-positive 基线 AP=0.70045、F1=0.82384，但不是官方 val/test，也没有证明接触修正、时序稳定性或渲染改善。

唯一可用于该结论的数值证据是本机受忽略 run 的 `history.json`、`contract.json`、`best.pt` 与 `last.pt`；Git 只保存重建代码与文档，绝不保存 RICH、权重、cache、checkpoint 或凭据。

## 新 agent 的阅读顺序

| 顺序 | 文档 | 目的 |
|---:|---|---|
| 1 | [AGENT_HANDOFF.md](AGENT_HANDOFF.md) | 工作约束、数据/GT 边界、当前 run 和首次现场检查 |
| 2 | [forward_contact_pipeline/reports/rich_full_contact_v1_v2_20260918.md](forward_contact_pipeline/reports/rich_full_contact_v1_v2_20260918.md) | 全量 Stage-A 的数据、模型、协议、结果、限制和下一步 |
| 3 | [forward_contact_pipeline/CONTACT_ESTIMATOR_ARCHITECTURE.md](forward_contact_pipeline/CONTACT_ESTIMATOR_ARCHITECTURE.md) | 最终系统目标与当前已实现部分的边界 |
| 4 | [forward_contact_pipeline/TODO.md](forward_contact_pipeline/TODO.md) 顶部 | 当前待办与禁止事项；旧条目仅作追溯 |
| 5 | [forward_contact_pipeline/CONTACT_CORRECTION_EXPERIMENT_PLAN.md](forward_contact_pipeline/CONTACT_CORRECTION_EXPERIMENT_PLAN.md) | 可执行研究协议与历史失败原因 |
| 6 | [forward_contact_pipeline/RICH_DATASET.md](forward_contact_pipeline/RICH_DATASET.md) 和 [RICH_PREPARATION.md](forward_contact_pipeline/RICH_PREPARATION.md) | RICH 数据契约、处理和许可边界 |
| 7 | [MIGRATION.md](MIGRATION.md) | 有数据/权重/cache 时的受控迁移或续训 |
| 8 | [COLD_START_NEW_MACHINE.md](COLD_START_NEW_MACHINE.md) | 无任何资产时的重新下载、处理与新 run |
| 9 | [CLAUDE_SESSION_LOG.md](CLAUDE_SESSION_LOG.md) 顶部及关键词搜索 | 决策时间线；不应替代上述状态文档 |
| 10 | [GIT_ASSET_INVENTORY.md](GIT_ASSET_INVENTORY.md) | 已上传/未上传资产、大小、许可边界及重建方式 |

## 当前任务优先级

1. 保存并核对完成 run 的四个非 Git 资产：`best.pt`、`last.pt`、`history.json`、`contract.json`；记录 SHA256、Git commit、GUSH3R patch/revision、checkpoint SHA256 和 RICH plan hash。
2. 在没有官方 test 的前提下，不再调 epoch、阈值或选择第 30 轮。用 epoch 5 的 `best.pt` 做冻结的内部误差分析；若下载官方 val/test，必须将它们作为全新冻结评测而非调参数据。
3. 做能证伪的 Stage-A 消融：几何/局部 RGB/全局 RGB/相机条件/预测 point map 的移除或替换。每项使用同一 split、同一全候选指标和固定选择规则。
4. 仅当 Stage-A 在独立数据上完成接触、覆盖率、校准/abstention 与错误分析后，将 ROI token 接到 Stage-B；Stage-B 不能读取教师过去状态或 GT 输入。
5. Stage-B 通过因果 rollout 后，才接受 bounded root/foot/ankle correction、SMPL-X FK/IK 和 human Gaussian LBS；scene Gaussian 永远冻结。

## 不可违反的研究边界

- RICH GT body/contact 只能用于标签、目标关联和审计，不能进入 Stage-A 或部署输入。
- 不把 matched-only 指标作为主结果；必须同时报告 all-candidate、matched、coverage 和 all-positive baseline。
- ParkingLot2 是从官方 train 划分出的内部开发集，不称官方 validation/test。
- 不能将 GUSH3R/PROX 的 2cm 几何门槛失败、oracle 接触结果或合成 drift controller 结果描述为真实可部署改善。
- 不强行 resume source/config/plan/cache hash 不一致的 run；语义变化一律新建版本化 run。
- 不提交数据、权重、cache、checkpoint、cookie、token、私钥或本机环境。

## 非本次主线但需保留的工作

传统 HUGS/NeuMan 渲染、Anchor Attention、VIMO、STM 对照和传输系统仍在仓库中。它们的历史结论见 `CLAUDE_SESSION_LOG.md`、`VALIDATION_PROTOCOL.md`、`transmission_system/` 与相关报告；除非用户明确改变优先级，不应与 RICH Stage-A 的数据、指标或结论混写。
