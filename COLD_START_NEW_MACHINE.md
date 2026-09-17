# 新机器冷启动：重新下载数据、权重并重新训练 RICH Stage-A

适用于不能从旧机器复制 RICH、模型权重、冻结特征 cache 或训练 checkpoint 的情况。代码、配置和交接信息来自 Git；有许可限制的资产在新机器由已授权使用者自行下载。

开始前阅读 [AGENT_HANDOFF.md](AGENT_HANDOFF.md)。当前实验边界、失败历史和指标口径以该文件和 `CLAUDE_SESSION_LOG.md` 顶部为准。

## 0. 权限、硬件与空间

| 项目 | 要求 |
|---|---|
| GPU | NVIDIA GPU；推荐 CUDA 12.1 兼容驱动，先运行 `nvidia-smi` |
| 磁盘 | 至少 1.5 TB 可用；RICH train JPG 解压后约560GB，另需归档、GT、处理数据、cache 和工作余量 |
| RICH | 在官网接受非商业研究许可，并在新机器录入自己的账号密码 |
| GUSH3R | Hugging Face 账号能访问 `abkeito/GUSH3R` |
| SMPL/SMPL-X | 在官方页面接受各自许可 |
| GitHub | 在新机器创建自己的 SSH key 或 credential；不要复制旧机器 deploy 私钥 |

不要把 RICH、SMPL/SMPL-X、模型权重或任何 token 上传到公开 Git、Hugging Face 或网盘。

## 1. 克隆固定代码

```bash
git clone --recurse-submodules git@github.com:OIIAIIOOIIAII/ml-hugs-work.git
cd ml-hugs-work
git checkout e2ecef3
git submodule update --init --recursive
git status --short
```

此时工作树应为空。需要更新代码时，先阅读提交历史与 `AGENT_HANDOFF.md`，再显式 checkout 确认的 commit；不可混用未提交源码与旧 feature cache。

## 2. 重建 GUSH3R 和权重

主仓库不包含第三方前端、编译产物和权重。固定 GUSH3R 源码及本项目 patch：

```bash
git clone https://github.com/abkeito/GUSH3R.git GUSH3R
git -C GUSH3R checkout bac8d88405ff62453b033c5a2b5709f42fcf50be
git -C GUSH3R submodule update --init --recursive
git -C GUSH3R apply --check ../migration/patches/GUSH3R.patch
git -C GUSH3R apply ../migration/patches/GUSH3R.patch
```

创建新环境。若目标驱动不兼容 CUDA 12.1，先依 PyTorch 官方兼容表选择 wheel，不能复制旧机器 conda 目录或 `.so`：

```bash
conda create -n gush3r python=3.10 -y
conda activate gush3r
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r GUSH3R/requirements.txt
pip install -r forward_contact_pipeline/requirements-data.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()"
```

下载合并推理权重：

```bash
pip install -U "huggingface_hub[cli]"
hf auth login
cd GUSH3R
mkdir -p checkpoints
hf download abkeito/GUSH3R checkpoints/gush3r.pth --local-dir .
sha256sum checkpoints/gush3r.pth
cd ..
```

按 `GUSH3R/README.md` 的 **SMPL and SMPL-X Related Body Models** 小节运行其官方下载脚本，并在交互时录入自己的 SMPL/SMPL-X 凭据。完成后检查 `GUSH3R/src/models/` 中的 `body_models/smpl/`、`body_models/smplx/`、`smplx2smpl.pkl` 与 `smplx2smpl_joints.npy`。它们都不进 Git。

若 DINOv2 torch hub 下载受网络限制，可先让 PyTorch 在新机器正常下载，或设置 `GUSH3R_DINO_HUB_LOCAL_REPO` 指向新机器的本地 DINO 源码目录。

## 3. 在新机器重新获取和处理 RICH

下载器不保存旧机器认证。它使用 Git 中的资产清单，登录信息只保存在新机器受忽略的 `.tools/` 目录：

```bash
conda activate gush3r
python forward_contact_pipeline/scripts/acquire_rich.py --configure
python forward_contact_pipeline/scripts/acquire_rich.py --start --engine auto --connections 16 --other-limit 500K
python forward_contact_pipeline/scripts/acquire_rich.py --status
```

保留 `.part` 与 `.aria2`，它们用于断点续传；不要把稀疏文件显示的逻辑大小当作实际完成。下载完成后处理归档和 GT：

```bash
python forward_contact_pipeline/scripts/process_rich_assets.py --start --wait-hours 36 --stream-jpg
python forward_contact_pipeline/scripts/prepare_rich_annotations.py --start
python forward_contact_pipeline/scripts/process_rich_assets.py --status
python forward_contact_pipeline/scripts/prepare_rich_annotations.py --status
python forward_contact_pipeline/scripts/prepare_rich_dataset.py \
  --root datasets/RICH \
  --config forward_contact_pipeline/configs/data/rich_dataset_v1.json \
  --workers 8
```

确认 `datasets/RICH/processed/annotations_v1/state.json` 与 `datasets/RICH/processed/dataset_v1/report.json` 已完成。不要删除完整归档或正式处理目录；失败时查看日志后用相同命令续跑。详细字段和归档规则见 `forward_contact_pipeline/RICH_PREPARATION.md` 与 `RICH_DATASET.md`。

## 4. 从零建立 cache 并重新训练

冷启动没有 checkpoint，因此新建输出目录，绝不带 `--resume`。首轮会生成冻结特征并训练，最慢；后续轮次复用 cache。

```bash
conda activate gush3r
export DINO_REPO="$HOME/.cache/torch/hub/facebookresearch_dinov2_main"
python forward_contact_pipeline/scripts/make_rich_full_plan.py \
  --rich-root datasets/RICH \
  --output datasets/RICH/processed/full_contact_v1/plan
python -u forward_contact_pipeline/scripts/train_rich_full_contact.py \
  --repo GUSH3R \
  --rich-root datasets/RICH \
  --plan datasets/RICH/processed/full_contact_v1/plan \
  --cache datasets/RICH/processed/full_contact_v1/cache \
  --output forward_contact_pipeline/runs/rich_full_contact_v1_seed42_newmachine \
  --config forward_contact_pipeline/configs/experiments/rich_full_contact_v1.json \
  --dino-repo "$DINO_REPO"
```

建议在 `tmux new -s rich_full_contact` 中启动，并重定向日志到数据盘。运行状态、检查点和指标分别在新 run 目录中的 `progress.json`、`last.pt`/`best.pt` 与 `history.json`。只有源码、配置、plan 和 cache 的 `contract.json` 都不变时才可在同一命令末尾加 `--resume`；任何模型、数据、标签、split、前端或配置语义变动都要新建版本化输出目录。

## 5. 验收与指标解释

先运行 CPU 回归：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python -m unittest forward_contact_pipeline.tests.test_rich_full
```

训练完成后，`history.json` 必须同时报告：

1. `all_candidates_missing_as_negative`：主开发指标，漏检/退化前端输出按负预测计入。
2. `matched`：仅成功关联人体的条件指标，不能单独报告。
3. `candidate_coverage`：前端目标关联覆盖率。

这是从 RICH train 中划出的 ParkingLot2 内部开发验证，不能称为官方 validation/test。该训练仅为 Stage-A；不含 Stage-B、SMPL-X 修正或 Gaussian LBS 回写。跨机器实验要记录 Git commit、GPU/驱动、权重 SHA256、RICH 处理报告 hash 和 run 的 `contract.json`。

## 常见问题

| 现象 | 处理 |
|---|---|
| RICH 下载文件只有几 KB | 多为认证错误页，重新执行 `--configure` |
| 下载中断 | 保留 `.part/.aria2`，重新执行 `--start` |
| 无法访问 GUSH3R 权重 | 在网页取得仓库访问权后重新 `hf auth login`；不可用未授权镜像替代 |
| SMPL-X 缺失 | 在官方页面接受许可后按 GUSH3R README 的目录布局放置 |
| `--resume` hash 不一致 | 新建 run/cache 版本，不能删除 contract 绕过保护 |
| 退化脚底法向 | 本 commit 的 v2 会将其按漏检记录，不应使全任务崩溃；若仍崩溃，保存 clip ID 和 commit 再修复 |
| 空间不足 | 先扩容；删除数据、cache 或 checkpoint 后不能继续同一 run |
