# HUGS Agent Handoff

新机器上的 agent 开始工作时，先阅读本文件、`CLAUDE_SESSION_LOG.md` 顶部、
`MIGRATION.md` 和 `forward_contact_pipeline/TODO.md` 顶部。旧记录只作背景；当前
进程、Git 提交、run 的 `progress.json` 与 `history.json` 才是运行状态的依据。

## 当前工作目录与规则

- 项目根目录：`ml-hugs-work`。
- 用户使用中文，要求持续维护 `CLAUDE_SESSION_LOG.md`，最新内容写在最上方。
- 运行 GPU 工作前先检查 `nvidia-smi`；此项目使用单张共享 RTX 4090，不能结束非本项目任务。
- 不读取或展示 RICH 原图来做人工判断；数值特征提取和数值评估可以执行。
- Git 只放源码、配置、测试、文档和小型清单。数据、模型权重、cookie/私钥、缓存、检查点和渲染产物都不能提交。

## 进行中的实验：RICH Stage-A v2

`forward_contact_pipeline/runs/rich_full_contact_v1_seed42_v2` 是唯一有效的全量运行。

- 任务：冻结 GUSH3R，训练新的 RGB--mesh--point Stage-A 脚底逐顶点接触分类头；不训练 GUSH3R，不训练旧几何残差模型。
- 范围：165,920 train 人物--图像样本、74,264 ParkingLot2 内部 validation 样本，共 30 epoch。官方 val/test 尚未下载，内部 validation 不能作为正式 test。
- 输入：冻结预测的 SMPL-X/scene point map、局部及全局 DINO token、pose/query、物理相机条件。GT body/contact 仅用于唯一目标关联和监督，绝不进入模型输入。
- v1 在 epoch 1 validation 的退化预测脚底法向处崩溃，因源码哈希改变不可续跑；v2 对退化法向或非有限的冻结输入跳过该预测人体，并把对应目标作为未匹配项在全候选验证中计负，避免伪造样本或隐藏失败。
- v2 的完整指标以 `history.json` 为准，最佳 checkpoint 以全候选 histogram AP 选择。运行时状态、可续训模型和特征契约分别在 `progress.json`、`last.pt`、`contract.json`。
- 缓存契约与源码绑定。要继续同一 run，必须迁移相同的 `datasets/RICH/processed/full_contact_v1/cache`、run 目录、GUSH3R revision/patch、DINO cache 和 RICH 数据根；如改变源码、模型、配置或数据语义，建立新版本 run，不能覆盖 resume。

## 当前科学边界

该实验只验证 Stage-A 接触估计。它没有接入 Stage-B 时序控制、SMPL-X 修正、LBS human Gaussian 回写、可靠性校准或官方 test。预测关联覆盖率约 89.7%，因此必须同时报告 matched 和“遗漏按负预测”的 all-candidate 指标，不能只汇报较高的 matched 分数。

## 首次交接核查

```bash
git status --short
nvidia-smi
cat forward_contact_pipeline/runs/rich_full_contact_v1_seed42_v2/progress.json
cat forward_contact_pipeline/runs/rich_full_contact_v1_seed42_v2/history.json
```

确认训练未运行且要在同一版本恢复时，按 `MIGRATION.md` 的恢复命令启动；先比较
`contract.json` 的 hash，任何不一致都不要强行 `--resume`。
