# 2026-09-18 项目承接验证

本记录验证 Git 交接内容可在一份没有本机工作树改动的干净目录中取得和启动。它验证的是代码、文档、配置、子模块和资产获取说明；RICH、cache、上游权重、按场景 Gaussian checkpoint 与 GPU renderer 仍受许可和硬件环境约束，不能也不应被 Git clone 替代。用户指定的 `model_weights/` 小型自研模块是明确例外。

## 实测条件与结果

| 检查 | 结果 |
|---|---|
| HTTPS 从 `origin/main` 重新 clone | 通过；HEAD=`df56f36f000a99c7f10c2812cb06bd1c5273cb6a` |
| 顶层子模块 | 通过；`diff-gaussian-rasterization=59f5f77`、`simple-knn=f155ec0` |
| 嵌套 `glm` 子模块 | 通过；固定在 `5c46b9c` |
| 必需交接入口与依赖/补丁清单 | 11/11 文件存在 |
| 7 个 RICH 获取、处理、计划、训练、评测入口的 `--help` | 7/7 通过 |
| 顶层交接文档的本地相对链接 | 通过 |
| Git 工作树 | 干净；未依赖源机器未提交文件 |
| non-Git 资产声明 | 通过；RICH、cache、best/last checkpoint、GUSH3R/SMPL-X 均在资产清单中列出获取或受控迁移路径 |

## 新机器承接顺序

1. 通过 HTTPS clone 后，记录 HEAD，并先读 [PROJECT_MASTER_HANDOFF.md](PROJECT_MASTER_HANDOFF.md)、[AGENT_HANDOFF.md](AGENT_HANDOFF.md) 和 [COLD_START_NEW_MACHINE.md](COLD_START_NEW_MACHINE.md)。
2. 有受控 NAS/资产时，严格按照 [MIGRATION.md](MIGRATION.md) 复制或挂载 RICH、cache、checkpoint，并逐项比对 contract 和 SHA256；无资产时完全按冷启动文档重新下载、处理和新建 run。
3. 重建 GUSH3R、SMPL-X、DINO 与 CUDA 扩展后，先跑 `python -m unittest forward_contact_pipeline.tests.test_rich_full`，再建立 cache 或启动新 run。
4. 复评既有 checkpoint 前，运行 `scripts/evaluate_rich_full_contact.py`；它会拒绝 plan、cache 与 checkpoint contract 不一致的组合。

## 当前不可由 Git 自动完成的事项

- RICH、SMPL/SMPL-X 与 GUSH3R 权重各自需要新机器使用者按许可证取得授权。
- 旧 checkpoint/cache 若不能受控迁移，必须创建新的命名 run，不能伪造或强行 resume `rich_full_contact_v1_seed42_v2`。
- 新机器需要其自己的 Git 写权限；HTTPS clone 不依赖源机器的 deploy 私钥。
