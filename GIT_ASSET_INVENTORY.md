# Git 资产清单与非 Git 资产边界

本页列出截至 `2775a11` 的项目资产。它回答两件事：哪些已经同步到 Git，哪些没有，以及新机器应如何取得未同步资产。清单不是删除建议；任何未同步数据都保留在当前机器，直到用户选择受许可的迁移或重建方式。

## 已提交到 Git

| 资产 | 路径 | 内容与用途 |
|---|---|---|
| 顶层交接 | `PROJECT_MASTER_HANDOFF.md`、`AGENT_HANDOFF.md`、`CLAUDE_SESSION_LOG.md` | 阅读顺序、当前任务、研究边界、决策时间线 |
| 迁移与冷启动 | `MIGRATION.md`、`COLD_START_NEW_MACHINE.md` | 有资产时迁移/续训，以及无资产时从授权下载重建 |
| 代码与测试 | `forward_contact_pipeline/`、主项目源码、14个 contact 测试 | 训练、处理、评估、配置和可执行回归 |
| 第三方可复现性 | `migration/dependencies.json`、`migration/patches/GUSH3R.patch` | 固定 Human3R/GUSH3R/submodule revision 与本项目5个 GUSH3R 修改 |
| RICH 公开清单与处理配置 | `forward_contact_pipeline/configs/data/` | 文件名/URL、split 和处理配置；不含账号、cookie 或下载数据 |
| 技术路线与结论 | `forward_contact_pipeline/reports/*.md`、架构、TODO、训练文档 | 设计、成功与失败实验、当前限制和下一步 |
| 全量 Stage-A v2 机器可读证据 | `forward_contact_pipeline/reproducibility/rich_full_contact_v1_seed42_v2/` | 完整30轮 `history.json`、resolved config、contract、最终 progress；总计约54KB |
| 全量 Stage-A v2 人读报告 | `forward_contact_pipeline/reports/rich_full_contact_v1_v2_20260918.md` | 数据、模型、修复、hash、指标、局限与后续实验 |

上述内容可由普通 Git clone 获得。它们足以解释和重建实验流程，但**不含**运行资产本体。

## 已上传的小型最终运行证据

下列文件是本机完成 run 的逐字节副本，提交时间的 SHA256 应与原 run 一致：

| Git 文件 | SHA256 | 作用 |
|---|---|---|
| `reproducibility/rich_full_contact_v1_seed42_v2/config.json` | `6302e3fa817ee14e09ec917f46fde3ea009cbae37cf70e4fd7a9da6bcfb13283` | resolved hyperparameters |
| `reproducibility/rich_full_contact_v1_seed42_v2/contract.json` | `8ac1ffd323c0fb0c5b568670f5a29ad48e866a6e11dc4100d3cbad8e51aebfc8` | plan/source/frontend/checkpoint hashes |
| `reproducibility/rich_full_contact_v1_seed42_v2/history.json` | `f781170ef13eb49f5ad5a8f39f864188285c354f976e4bee0877f8776a3c7aa3` | 30 epoch 全量指标 |
| `reproducibility/rich_full_contact_v1_seed42_v2/progress.json` | `c2d7a15ed7333c345fdc624899e5e18951f3bc9eb0a337c42e5abb1716883382` | completed 状态、step/sample 计数 |

## 未提交：受许可或大体积运行资产

| 本机路径 | 约大小 | 为什么不进 Git | 新机器取得方式 | 是否影响冷启动 |
|---|---:|---|---|---|
| `datasets/RICH/raw/` | 558GB | RICH 原始归档受个人非商业许可约束，且远超 Git/LFS 合理规模 | 在新机器用 `acquire_rich.py --configure/--start` 以自己的授权下载，或挂载许可允许的受控 NAS | 必须重下/挂载 |
| `datasets/RICH/extracted/` | 约560GB 图片加公共资产 | RICH 许可与大体积 | 从原始归档重新处理 | 冷启动会重建 |
| `datasets/RICH/processed/` | 约49GB | 是由 RICH 生成的标注、监督、计划和 cache，包含受限数据派生物 | 按 RICH 准备脚本重建；同一许可范围内可受控迁移 | 训练前必须重建或迁移 |
| `datasets/RICH/processed/full_contact_v1/cache/` | 35GB | 可再生冻结特征，且与 contract 严格绑定 | 新机器首轮训练自动生成，或受控 rsync/NAS | 可重建；不必迁移 |
| `forward_contact_pipeline/runs/` | 约63MB | checkpoints、原始日志、旧 run 产物不进入源码历史 | best/last 可受控迁移；冷启动则新建 run | 仅续训需要 |
| `model_weights/` | 约12MB | 用户明确指定的本项目小型最佳模块权重：研究点一六个 Anchor Attention state dict 和研究点二 RICH Stage-A best checkpoint | Git 已跟踪，SHA256和加载边界见 `model_weights/README.md` | 直接复评/加载自研模块需要 |
| `runs/rich_full_contact_v1_seed42_v2/best.pt` | 6.3MB | 原 run 内 checkpoint；相同内容的可迁移副本在 `model_weights/research_point_2/` | SHA256 `645ef6e02c075a4da5578bffd61bea0561f26cf5cb46f613cd76568dc5b87406` | 复用最佳模型需要 |
| `runs/rich_full_contact_v1_seed42_v2/last.pt` | 6.3MB | 同上 | 当前 SHA256 `391c3068581acbff89d132c7ca851821c6441603dea315c84c3c32629e9ad698` | 续训需要；本 run 已完成 |
| `GUSH3R/checkpoints/gush3r.pth` | 4.9GB | 上游模型权重，不属于本仓库 | 从官方 `abkeito/GUSH3R` 下载；run contract checkpoint SHA256=`1e390dcf65f440dea4527378af1ef6bd7859b74465ab1925e58534c9e9208fe1` | 必须获取 |
| `GUSH3R/src/models/` | 3.3GB | SMPL/SMPL-X/MultiHMR 各有许可 | 按 GUSH3R README 的 `fetch_body_models.sh` 与官方许可重新获取 | 必须获取 |
| `/home/admin/.cache/torch/hub/` | DINO源码约5MB，另有权重缓存 | 环境级 cache，不随项目 Git 迁移 | PyTorch/HF 在新机器重新下载，或按许可使用本地 cache | 必须可用，但可重下 |
| `.tools/` | 约75MB | 含下载工具、临时验证和 Git deploy 私钥 | 新机器自行安装 aria2/创建 Git key；绝不复制私钥/cookie | 工具可重建；凭据须重建 |
| `datasets/RICH/download_lists/` | 极小 | 可能含 RICH credentials/cookie | 新机器运行 `acquire_rich.py --configure` | 必须重新配置 |

## 未提交：历史原始证据和非主线结果

`forward_contact_pipeline/reports/` 本机有约221个文件，而 Git 追踪24个结论报告。未追踪部分主要是下载检索网页缓存、搜索响应、JSON 中间统计、处理过程日志和大型实验原始输出。它们不影响重建主流程；追踪的 Markdown/摘要 JSON 已保存可复核结论。若未来要发表或复审某个历史实验，应在受控存储保留其对应原始目录，再把必要的小型摘要加入 Git。

传统 HUGS/NeuMan 的 output、渲染图、scene/human Gaussian checkpoint 和历史 run 不进 Git；其代码、配置、文档和关键结论已同步。用户指定的六个 Anchor Attention state dict 是唯一例外，已作为 `model_weights/research_point_1/` 保存。新机器复现完整场景时仍须按相应数据与环境说明获取大体积资产。

## 未提交但不是迁移缺口的本机修改

| 路径 | 状态 | 处理 |
|---|---|---|
| `INTERVIEW_TECHNICAL_ROADMAP.md` | 用户已有未提交修改 | 保留原样；需要发布前由用户或后续 agent 审阅其内容 |
| `submodules/diff-gaussian-rasterization/build/`、`diff_gaussian_rasterization/__pycache__/` | CUDA 构建产物 | 新机器按固定 submodule commit 重新编译 |
| `submodules/simple-knn/simple_knn.egg-info/` | Python 构建元数据 | 新机器重新安装即可 |

## 资产策略

普通 Git 承载代码、文档、配置、patch、manifest、小型数值证据和用户明确选择的 `model_weights/` 自研模块。受许可的数据和上游模型应从官方来源或受控 NAS 获取；可再生 cache 和未列入 `model_weights/` 的 checkpoint 只在受控存储传递。不要为“迁移方便”把 RICH、SMPL/SMPL-X、上游权重、cookie 或私钥上传到 Git/LFS、公开 Hugging Face 或公开网盘。
