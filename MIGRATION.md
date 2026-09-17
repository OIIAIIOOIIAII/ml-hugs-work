# HUGS代码与双机迁移

## 当前同步（2026-09-18）

通过Git迁移代码可行，而且应当作为后续持续迭代的主方式。当前origin为
`https://github.com/OIIAIIOOIIAII/ml-hugs-work.git`，既有远端为公开仓库，默认分支`main`。
本次源码快照收录本地HUGS改动、接触训练与RICH处理模块、配置、测试、方案说明和第三方补丁。
仓库端已授权title为`deploy`的部署密钥，源码提交`8ba0dcf`已推送至`origin/main`。
后续继续通过此仓库的SSH push配置同步；同步结束以本地HEAD和远端main哈希一致验收。
本次没有改变远端可见性，也没有将数据或模型上传。为使新机器上的 agent 能无缝接手，
仓库根目录的`AGENT_HANDOFF.md`与`CLAUDE_SESSION_LOG.md`同步当前实验边界、失败原因、
恢复规则和工作偏好；新 agent 应先读二者，再以运行中的`progress.json`验证实时状态。

验证：主工作区45项CPU软件测试通过；从暂存内容导出的干净副本44项通过、
1项aria2相关测试因副本环境未安装该工具跳过。源码语法与敏感信息模式检查通过，
第三方revision/补丁一致。现有文档硬换行和历史源码空白保留；未进行GPU训练验收。

代码路径、依赖环境、数据资产分别处理：

| 内容 | 迁移方式 |
|---|---|
| 自有Python/Shell、配置、测试、方案文档 | Git，同一主仓库 |
| Human3R/GUSH3R源码 | `migration/dependencies.json`固定revision |
| GUSH3R本地5处改动 | `migration/patches/GUSH3R.patch`，已验证重建字节一致 |
| rasterizer/simple-knn | 原Git submodule固定提交，新机器重新编译 |
| RICH/PROX、SMPL(-X)、checkpoint/特征缓存 | NAS或受控传输，遵守各自许可 |
| 登录凭据/Cookie/token、本机路径、conda环境 | 不进Git；目标机器单独配置 |

## 主项目结构

保留`hugs/`原训练链；`forward_contact_pipeline/contact_streaming/`是独立包，
`training/`处理可替换的数值缓存实验；`configs/experiments/`记录实验参数，
`configs/paths.local.yaml`记录本机路径。后续更换算法主要新增组件/配置，不要在
`main.py`上堆叠所有接触实验开关。具体接口、命令与已知缺口见
[TRAINING.md](forward_contact_pipeline/TRAINING.md)。

## 选择性源码提交

```bash
python scripts/audit_git_migration.py --output forward_contact_pipeline/reports/git_migration_review
```

输出`audit.json`与`source.paths`。它识别源码候选、常见token/private-key模式、
大文件、符号链接、遗留硬编码机器路径；不会git add或上传。
它不是完整历史密钥审计：主仓库已经跟踪的历史资产仍随clone迁移。
`.gitignore`不会从已有提交中移除文件。

审阅`source.paths`、阻塞项与旧路径列表后，选择性暂存：

```bash
git --literal-pathspecs add --pathspec-from-file=forward_contact_pipeline/reports/git_migration_review/source.paths
git diff --cached --stat
git diff --cached
```

本次沿用现有main分支，非强制推送。审计覆盖必要的RICH元数据TSV、requirements-data.txt、
根目录导出脚本与设计文档；下载论文、检索缓存、数据集和测量输出保留在本机。
不要运行`git add .`；不要force push或把600GB数据放入Git/LFS。
历史脚本中的机器绝对路径由审计报告逐项列出；新入口已配置化，旧脚本没有全量重写，
迁移后应按实际使用到的旧实验逐项修改/通过参数覆盖，而非做盲目全局替换。

## 本机推送认证

已完成认证：仓库管理员已添加title为`deploy`的专用公钥并授予写入权限，SSH推送成功。
此前HTTPS没有配置认证、旧SSH密钥未获授权，这些已不再阻塞当前仓库推送。
本机已准备独立的仓库部署密钥，公钥位于`.tools/git-sync/github_deploy_ed25519.pub`，
私钥位于同目录且被Git忽略。GitHub主机公钥从其官方HTTPS API获取并严格校验。
仓库管理员在`Settings → Deploy keys`添加公钥并开启`Allow write access`后，
本机可以使用该密钥推送到原仓库；不要把私钥或token放进提交或对话。
这项认证只适用于当前机器，其他机器需要自己的GitHub认证。

## 目标机器

发布上述代码后：

```bash
git clone --recurse-submodules https://github.com/OIIAIIOOIIAII/ml-hugs-work.git
cd ml-hugs-work
```

如果代码在迁移分支，clone时用`--branch`指定该分支或随后checkout对应commit。
再按`migration/dependencies.json`克隆需要的frontend并checkout固定revision：

```bash
git clone https://github.com/fanegg/Human3R.git Human3R
git -C Human3R checkout 402f2b2c7f20514e99cb42e4126c46b4ff75593f
git -C Human3R submodule update --init --recursive
git clone https://github.com/abkeito/GUSH3R.git GUSH3R
git -C GUSH3R checkout bac8d88405ff62453b033c5a2b5709f42fcf50be
git -C GUSH3R submodule update --init --recursive
git -C GUSH3R apply --check ../migration/patches/GUSH3R.patch
git -C GUSH3R apply ../migration/patches/GUSH3R.patch
```

patch只应应用一次，先核对patch SHA256和base revision。未来对第三方代码的修改也需要
更新patch和固定版本，或使用自己的维护fork；不能仅更新主仓库里一个目录名。

先按TRAINING.md重建独立数值训练环境并运行CPU检查，再按HUGS/Human3R/GUSH3R各自
说明配置GPU环境、授权模型文件及CUDA扩展。尚未在第二台机器实际验证GPU/renderer。

数据挂载回本机paths配置指向的根目录。checkpoint/缓存单独转移并核对hash；
仍在下载的`.part/.aria2`不是可训练数据，也不复制`download_lists/`的凭据。
公开RICH下载清单已放在`configs/data/rich_download_assets.json`，新clone不再依赖
源机器未跟踪的私有manifest；认证仍由`acquire_rich.py --configure`在终端本地录入。

## 实验继续变化时

每次改变模型或数据语义：新增命名配置、更新feature_version/index、记录Git commit。
避免覆盖旧结果；科学配置改变开新run，保持同配置的机器迁移才使用resume。
优先上传已测试的小步代码，不把一次性结果目录与临时第三方源码同步进主仓库。

## 当前 RICH Stage-A v2 的可续训迁移

目前有效运行目录为`forward_contact_pipeline/runs/rich_full_contact_v1_seed42_v2`。要在新机器
继续同一个实验，必须在停止源任务或确保源/目标不同时写同一 run 后，成组复制如下资产：

| 资产 | 用途 | 迁移要求 |
|---|---|---|
| `datasets/RICH/` | 原图、标注、scan、计划 | 许可受限；优先同一受控 NAS 挂载，不公开上传 |
| `datasets/RICH/processed/full_contact_v1/cache/` | 35GB 左右冻结特征与契约 | 与源码/config/checkpoint hash 必须匹配 |
| `forward_contact_pipeline/runs/rich_full_contact_v1_seed42_v2/` | `last.pt`、`best.pt`、状态和历史 | 复制时保留原子 checkpoint 文件，不要只复制 `progress.json` |
| `GUSH3R/checkpoints/`、SMPL-X、DINO hub cache | 冻结前端依赖 | 受各自许可约束；目标机本地配置，不进 Git |

如果目标机能通过 NAS 访问同一绝对路径，最稳妥的方式是仅 clone 代码、重建环境，并复用
同一数据目录；无需复制约 560GB 的 RICH JPG。若必须通过 SSH 复制，源任务运行时可先
反复同步只增不删的数据与缓存，最终停任务后做一次收尾同步：

```bash
# 在目标机执行；将 SOURCE_HOST 和源目录替换为实际值。
rsync -aHAX --partial --append-verify --info=progress2 \
  SOURCE_HOST:/workspace/nas_auto_backup/yuzilang/ml-hugs-work/datasets/RICH/ \
  /data/RICH/
rsync -aHAX --partial --append-verify --info=progress2 \
  SOURCE_HOST:/workspace/nas_auto_backup/yuzilang/ml-hugs-work/forward_contact_pipeline/runs/rich_full_contact_v1_seed42_v2/ \
  forward_contact_pipeline/runs/rich_full_contact_v1_seed42_v2/
rsync -aHAX --partial --append-verify --info=progress2 \
  SOURCE_HOST:/workspace/nas_auto_backup/yuzilang/ml-hugs-work/datasets/RICH/processed/full_contact_v1/cache/ \
  /data/RICH/processed/full_contact_v1/cache/
```

迁移后先比较源/目标 run 的`contract.json`、`last.pt` SHA256 和 cache 中`contract.json`；再
确认目标机可读取 RICH 与 GUSH3R checkpoint，最后才带`--resume`启动。不要两个机器同时
写同一个 cache 或 run 目录。

如果无法传递任何 RICH、权重、cache 或 checkpoint，则不使用本节的 resume 流程。新机器应按
[`COLD_START_NEW_MACHINE.md`](COLD_START_NEW_MACHINE.md) 从本地授权下载、归档处理、计划/cache
重建和新的命名 run 开始；该文档也包含环境、验证和结果口径。
