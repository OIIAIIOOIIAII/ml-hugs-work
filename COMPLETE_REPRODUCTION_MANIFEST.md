# 两个研究点的完整复刻资产清单

这里的“完整复刻”指能够在新机器加载已得到的最终结果、运行既有评测，并在资产许可允许时重新训练；不是只获得模型源码或少量模块权重。

## 研究点一：NeuMan v4 Anchor Attention / 场景校正

| 层级 | 必需资产 | 交付方式 | 状态 |
|---|---|---|---|
| 代码与配置 | HUGS/Anchor Attention 源码、submodule、六场景配置 | Git | 已同步 |
| 最终可加载结果 | 6×`scene_final.pth`、6×`human_final.pth`、6×`anchor_attention_final.pth`、6×`config_train.yaml`，24 files / 约9.3GB | GitHub Release `research-point-1-v4-final-results` | 清单/校验已同步，待 GitHub 登录后上传 |
| 完整运行目录 | 中间 checkpoint、日志、debug、render 产物，约19GB | 可选受控 NAS/rsync | 不影响加载最终结果；继续历史训练时才需要 |
| 原始场景数据与评测输入 | 对应 NeuMan 数据、相机/图像/标注及运行环境 | 原始数据来源或受控迁移 | 重新训练、重新评测必须具备 |

最终 Release 资产用 [`releases/research_point_1_v4_final_results/SHA256SUMS`](releases/research_point_1_v4_final_results/SHA256SUMS) 校验。`model_weights/research_point_1/` 的六个 Anchor Attention 文件只是 Release 的小型重复副本，不能替代 scene/human Gaussian。

## 研究点二：RICH Stage-A Contact

| 层级 | 必需资产 | 交付方式 | 状态 |
|---|---|---|---|
| 代码、配置、评测器、实验记录 | `forward_contact_pipeline/`、reproducibility JSON、交接文档 | Git | 已同步 |
| 最佳模型 | `model_weights/research_point_2/rich_full_contact_v1_seed42_v2_best.pt`，6.58MB，SHA256记录在 README | Git | 已同步 |
| 精确内部验证 | `datasets/RICH/processed/full_contact_v1/plan/` 与 `cache/`（约35GB） | 受控 NAS/`rsync --append-verify` | 必须迁移；否则只能重建 |
| 冻结特征重建/重新训练 | RICH 原始与处理数据（约558GB原始，另有处理输出）、GUSH3R checkpoint、SMPL/SMPL-X、DINO | 许可授权下载或受控 NAS/rsync | 必须具备 |

已同步的 `best.pt` 可复现模型参数；要复现报告中的内部评测数值，还必须使用相同的 plan/cache contract。运行时评测器会拒绝不匹配的组合。完整命令和哈希检查见 [MIGRATION.md](MIGRATION.md) 与 [COLD_START_NEW_MACHINE.md](COLD_START_NEW_MACHINE.md)。

## 交接验收

新机器完成以下检查后，才可称两个研究点均可承接：

1. `git rev-parse HEAD` 与交接记录的 commit 一致，`git submodule update --init --recursive` 成功。
2. 研究点一：下载 Release 后，按 SHA256 清单校验 24/24 assets；将文件放回相对路径并加载六个最终 scene/human/anchor state。
3. 研究点二：比较 `best.pt`、plan 和 cache contract 的 SHA256；运行 `evaluate_rich_full_contact.py` 并复现记录的内部指标。
4. 若需要重训而非只加载结果，按每项数据许可证从官方来源或受控 NAS 获取原始资产；不能用不同数据/缓存强行 resume。
