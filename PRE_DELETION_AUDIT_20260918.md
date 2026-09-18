# 本机删除前验收（2026-09-18）

本文件记录 `ml-hugs-work` 在删除前的最后一次可恢复性核验。它只说明本机删除后如何承接，不授权删除操作。

## 已验证可从远端恢复

- Git 本地 `HEAD` 与 `origin/main` 均为 `889827373e34e8eb8f41768eca63cde62a46d6d9`；远端包含总代码、固定子模块指针、交接入口、迁移/冷启动文档、实验记录、处理和训练脚本。
- [PROJECT_MASTER_HANDOFF.md](PROJECT_MASTER_HANDOFF.md)、[AGENT_HANDOFF.md](AGENT_HANDOFF.md)、[COMPLETE_REPRODUCTION_MANIFEST.md](COMPLETE_REPRODUCTION_MANIFEST.md)、[GIT_ASSET_INVENTORY.md](GIT_ASSET_INVENTORY.md)、[MIGRATION.md](MIGRATION.md)、[COLD_START_NEW_MACHINE.md](COLD_START_NEW_MACHINE.md) 和 `CLAUDE_SESSION_LOG.md` 都已在 `origin/main` 实测存在。
- 研究点一 VIMO v4 粗对齐 Release：<https://github.com/OIIAIIOOIIAII/ml-hugs-work/releases/tag/research-point-1-v4-final-results>。27 项均为 uploaded：六场景18k final 的24项，以及 parkinglot 11k峰值三件套；每项已与本地 SHA256/大小核验。
- 研究点一 GT 对齐最佳 Release：<https://github.com/OIIAIIOOIIAII/ml-hugs-work/releases/tag/research-point-1-gt-alignment-best-results>。24 项均为 uploaded，且每项已与本地 SHA256/大小核验。
- 研究点二的自研 Stage-A best checkpoint 与两条研究点一路线的轻量 Anchor Attention state dict 已由 Git 跟踪；哈希和加载边界见 [model_weights/README.md](model_weights/README.md)。

## 删除会丢失、但未进入 Git/Release 的本机资产

- `datasets/RICH/raw/`、`extracted/`、`processed/`、full-contact plan/cache 和 `forward_contact_pipeline/runs/`。其中 raw train JPG 归档约559GB；RICH 受许可证约束，必须由新机器持有自己的授权重新下载，或在许可范围内受控迁移。删除后不能从 GitHub 恢复精确的 RICH 内部验证 cache。
- `output/` 约97.6GB，含历史中间结果和渲染产物。两条路线的最终可加载 scene/human/anchor/config 已在 Release；继续历史训练、取得未发布中间 checkpoint 或原始渲染，仍需保留/迁移本目录。
- 上游 GUSH3R、SMPL/SMPL-X、DINO 等许可资产，以及本机工具、凭据和构建缓存，均不能放入公开 Git；重新获取方式在迁移与冷启动文档中。

## 当前唯一未同步的项目差异

1. `INTERVIEW_TECHNICAL_ROADMAP.md`：用户已有未提交修改（6行新增、248行删除）。它未被提交，删除项目会丢失该版本。
2. `submodules/diff-gaussian-rasterization`：仅 `build/` 与 `__pycache__/` 本机构建缓存未跟踪。
3. `submodules/simple-knn`：仅 `simple_knn.egg-info/` 本机构建元数据未跟踪。

后两项可在新机器按固定子模块 commit 重新构建；第一项须在删除前决定保留到 Git 还是放弃。
