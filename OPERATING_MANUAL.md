# HUGS 项目操作手册

本文档是本仓库的本地操作手册，会随着后续代码修改持续更新。内容基于当前仓库 README、配置文件、训练入口、数据加载器、训练器、评估脚本以及当前本地未提交改动整理。

最后更新：2026-05-17

## 1. 项目定位

HUGS（Human Gaussian Splats）从单目视频中重建可动画化人体与背景场景。仓库当前只实现并启用 NeuMan 数据集流程，核心方法包括：

- 人体：`HUGS_TRIMLP` 或 `HUGS_WO_TRIMLP`，由 SMPL 参数驱动人体高斯。
- 场景：`SceneGS`，基本沿用 3D Gaussian Splatting 的场景高斯表示。
- 联合渲染：`render_human_scene` 将人体高斯和场景高斯合成到同一相机视角。
- 训练模式：`human_scene`、`human`、`scene` 三种。

## 2. 目录结构速览

```text
.
├── README.md                         # 官方快速说明
├── OPERATING_MANUAL.md               # 本操作手册
├── main.py                           # 训练、验证、动画、canonical 渲染主入口
├── cfg_files/
│   ├── release/neuman/               # 发布版训练配置
│   └── ablation/neuman/              # 消融实验配置
├── hugs/
│   ├── cfg/                          # 默认配置和常量路径
│   ├── datasets/                     # NeuMan 数据集读取与相机/SMPL 工具
│   ├── losses/                       # L1、SSIM、LPIPS 等损失
│   ├── models/                       # 人体高斯和场景高斯模型
│   ├── renderer/                     # Gaussian rasterizer 渲染封装
│   ├── trainer/                      # GaussianTrainer 训练/验证/动画逻辑
│   └── utils/                        # 配置、图像、视频、几何、可视化工具
├── scripts/
│   ├── conda_setup.sh                # 官方 conda 环境安装脚本
│   ├── prepare_data_models.sh        # 下载 NeuMan 数据和预训练模型
│   ├── evaluate.py                   # 评估已有输出目录
│   └── run_neuman_human_scene_noamass.py
│                                      # 本地新增：无需 AMASS 动画数据的训练辅助脚本
├── submodules/
│   ├── diff-gaussian-rasterization/  # 3DGS CUDA rasterizer
│   └── simple-knn/                   # CUDA KNN 距离扩展
├── data/                             # 本地数据目录
├── output/                           # 训练输出和预训练模型
└── run_logs/                         # 本地 shell 运行日志
```

## 3. 环境准备

官方测试环境是 Ubuntu 22.04.3、CUDA 11.7 兼容 GPU、Python 3.8。

从零安装：

```bash
git clone --recursive git@github.com:apple/ml-hugs.git
cd ml-hugs
source scripts/conda_setup.sh
```

如果已经克隆但子模块缺失：

```bash
git submodule update --init --recursive
```

`scripts/conda_setup.sh` 会创建 `hugs` conda 环境，并安装：

- `pytorch==1.13.1`、`torchvision==0.14.1`、`torchaudio==0.13.1`、`pytorch-cuda=11.7`
- `pytorch3d`
- `submodules/diff-gaussian-rasterization`
- `submodules/simple-knn`
- `requirements.txt` 中的 `open3d`、`lpips`、`smplx`、`omegaconf` 等依赖
- `chumpy`

日常进入环境：

```bash
conda activate hugs
cd /hdd/u202420081000003/ml-hugs
```

## 4. 数据准备

### 4.1 SMPL 模型

需要从 SMPL 官网下载 neutral body model v1.1.0 和 SMPL UV obj。放置结构：

```text
data/smpl/
├── SMPL_NEUTRAL.pkl
└── smpl_uv.obj
```

代码里的默认 SMPL 根目录是 `hugs/cfg/constants.py` 中的：

```python
SMPL_PATH = 'data/smpl'
```

### 4.2 NeuMan 数据和预训练模型

官方脚本：

```bash
source scripts/prepare_data_models.sh
```

它会下载并解压：

- `neuman_data.zip`
- `hugs_pretrained_models.zip`

当前代码固定从 `data/neuman/dataset` 读取 NeuMan：

```python
NEUMAN_PATH = 'data/neuman/dataset'
```

当前本地已有的 NeuMan 序列包括：

```text
bike
citron
jogging
lab
parkinglot
seattle
```

单个序列至少需要包含这些训练会直接访问的内容：

```text
data/neuman/dataset/<seq>/
├── images/
├── sparse/
├── depth_maps/
├── mono_depth/
├── 4d_humans/
│   ├── smpl_optimized_aligned_scale.npz
│   └── sam_segmentations/
└── ...
```

`NeumanDataset` 会把图像、mask、相机参数、SMPL 参数缓存到 CUDA，所以启动训练时显存会先被数据缓存占用一部分。

### 4.3 AMASS 动画数据

官方 README 中 AMASS 只用于 novel-pose animation，不是训练和验证的必要输入。原始 `main.py` 在 `human` 和 `human_scene` 模式下会创建 `anim_dataset`，因此如果没有 AMASS，最终或中途动画渲染会失败。

官方期望结构：

```text
data/
├── MPI_mosh/
└── SFU/
```

如果当前只想训练和验证，不想准备 AMASS，可使用本地新增脚本：

```bash
python scripts/run_neuman_human_scene_noamass.py --seq lab --quick --quick-steps 100
```

该脚本会禁用动画数据集，并在 quick 模式跳过 `optimize_init` 的 7000 步初始化，适合烟测。

### 4.4 Human3R 输出转 HUGS 数据集

本仓库现在提供一个转换脚本，用于把 Human3R `demo.py --save` 结果转成 HUGS 可读取的数据目录：

```bash
python scripts/convert_human3r_to_hugs.py \
  --result-dir /path/to/human3r/output/my_run \
  --out-dir data/human3r_hugs/my_run \
  --conf-threshold 1.5 \
  --point-stride 2 \
  --max-scene-points 500000 \
  --smpl-params-frame camera
```

输入目录需要包含 Human3R 保存的：

```text
color/
depth/
conf/
camera/
smpl/
```

输出目录结构：

```text
data/human3r_hugs/my_run/
├── images/
├── masks/
├── cameras.npz
├── metadata.json
├── scene/
│   ├── points.npz
│   └── points.ply
└── 4d_humans/
    └── smpl_optimized_aligned_scale.npz
```

转换逻辑：

- 用 Human3R `depth/conf/camera/color` 反投影生成世界坐标点云。
- 默认用 Human3R `smpl/<frame>.npz` 中的 `msk` 或 `scores` 估计人体 mask，并从场景点云中剔除人体区域。
- 默认将 Human3R SMPL-X 参数近似转成 HUGS 所需 SMPL 参数：
  - `global_orient = c2w[:3, :3] @ rotvec[0]`
  - `body_pose = rotvec[1:24]`
  - `betas = shape[:10]`
  - `transl = c2w[:3, :3] @ transl + c2w[:3, 3]`
  - `scale = 1`

默认 `--smpl-params-frame camera`，因为 Human3R 保存的 `smpl/*.npz::transl` 通常是相机坐标；脚本会将 `global_orient/transl` 转到 world frame 再写给 HUGS。如果确认输入已经是 world frame，使用 `--smpl-params-frame world`。

重要：这个 SMPL-X 到 SMPL 的映射是用于快速接通流程的保守近似。Human3R 的 `smpl/*.npz::transl` 在默认输出里是 `person_center='head'` 下的头部中心坐标，而 HUGS 使用的是 `SMPL vertices * smpl_scale + transl` 的整体平移，因此不能把 Human3R 的 `transl` 直接当作 HUGS SMPL 平移长期使用。正式实验建议先运行下面的严格拟合流程。

### 4.5 严格拟合 Human3R SMPL-X 到 HUGS SMPL

本仓库新增：

```text
scripts/fit_smpl_to_human3r_smplx.py
```

用途：复现 Human3R 的 head-centered SMPL-X forward，把 SMPL-X mesh 映射到 SMPL topology，然后优化 HUGS 可直接读取的 `global_orient/body_pose/betas/transl/scale`。图像、mask、相机和场景点云保持不变，只替换 `4d_humans/smpl_optimized_aligned_scale.npz`。

推荐流程：

```bash
python scripts/convert_human3r_to_hugs.py \
  --result-dir data/neuman/dataset/lab/human3r \
  --out-dir data/human3r_hugs/lab_c2w \
  --conf-threshold 1.5 \
  --point-stride 2 \
  --max-scene-points 500000 \
  --smpl-params-frame camera

python scripts/fit_smpl_to_human3r_smplx.py \
  --human3r-result-dir data/neuman/dataset/lab/human3r \
  --hugs-dir data/human3r_hugs/lab_c2w \
  --out-dir data/human3r_hugs/lab_fit_smpl \
  --smpl-params-frame camera \
  --iters 1500 \
  --batch-size 16 \
  --vertex-sample 4096
```

默认会使用本机 Human3R 模型资产：

```text
/hdd/u202420081000003/Human3R/src/models/
```

其中需要：

```text
smplx/SMPLX_NEUTRAL.npz 或 SMPLX_NEUTRAL.pkl
smplx/smplx2smpl.pkl
```

输出：

```text
data/human3r_hugs/lab_fit_smpl/
├── images/ masks/ cameras.npz scene/ ...
└── 4d_humans/
    ├── smpl_optimized_aligned_scale.npz
    └── fit_smpl_to_human3r_smplx_report.json
```

拟合报告里的重点字段：

- `final_vertex_l1_m`：拟合后 SMPL 顶点到 Human3R SMPL-X 映射顶点的平均 L1，单位米。
- `final_joint_l1_m`：身体关节平均 L1，单位米。
- `final_bbox_l1_px`：SMPL 投影 bbox 与 Human3R mask bbox 的平均像素误差。

CPU 最小烟测命令：

```bash
python scripts/fit_smpl_to_human3r_smplx.py \
  --human3r-result-dir data/neuman/dataset/lab/human3r \
  --hugs-dir data/human3r_hugs/lab_c2w \
  --out-dir /tmp/hugs_fit_smpl_smoke \
  --max-frames 1 \
  --iters 1 \
  --batch-size 1 \
  --vertex-sample 64 \
  --target-chunk-size 1 \
  --device cpu
```

当前烟测结果：脚本可以在 CPU 上完成 1 帧/1 步并写出结果，最终顶点 L1 约 `0.017 m`、关节 L1 约 `0.018 m`、bbox L1 约 `3.8 px`。完整序列建议在 GPU 上运行；如果 `ssh gpu-l40-1` 报 `pam_slurm_adopt: you have no active jobs on this node`，需要先申请/进入对应 Slurm job。

## 5. 配置系统

默认配置定义在 `hugs/cfg/config.py`，实验配置在 `cfg_files/`。

主入口用法：

```bash
python main.py --cfg_file <yaml> [--cfg_id <id>] [OmegaConf overrides...]
```

配置合并顺序：

1. `hugs/cfg/config.py` 的 `default_cfg`
2. `--cfg_file` 指定 YAML 展开的单个实验项
3. 命令行 extras，例如 `dataset.seq=lab train.num_steps=1000`

重要行为：`hugs/utils/config.py:get_cfg_items` 会把 YAML 中所有列表值当作超参搜索维度。例如 release 配置里的：

```yaml
dataset:
  seq: ["lab", "citron", "seattle", "bike", "jogging", "parkinglot"]
```

如果不传 `--cfg_id`，会按列表展开并依次跑多个实验。注意：列表展开发生在命令行 extras 合并之前，所以只在命令行写 `dataset.seq=lab` 仍会生成多项实验，只是每一项最后都会被覆盖成 `lab`，结果就是同一个序列被重复跑多次。

日常单序列训练建议两种做法二选一：

1. 使用 `--cfg_id` 选择展开后的某一个实验：

```bash
python main.py \
  --cfg_file cfg_files/release/neuman/hugs_human_scene.yaml \
  --cfg_id 0 \
  dataset.seq=lab
```

2. 复制一份 YAML，把 `dataset.seq` 从列表改成单个字符串，然后用这份单序列配置运行：

```bash
python main.py --cfg_file cfg_files/release/neuman/hugs_human_scene_lab.yaml
```

如果忘记加 `--cfg_id`，第一个实验完成后会继续进入下一项。已经保存完目标序列结果时，可以停止多余进程，避免继续占用 GPU。

## 6. 常用训练命令

### 6.1 联合训练人体和场景

```bash
python main.py \
  --cfg_file cfg_files/release/neuman/hugs_human_scene.yaml \
  --cfg_id 0 \
  dataset.seq=lab
```

特点：

- `mode=human_scene`
- 同时优化 `HUGS_TRIMLP` 人体和 `SceneGS` 场景。
- 默认 `human.loss.humansep_w=1.0`，包含人体/场景分离相关损失。

### 6.2 仅训练人体

```bash
python main.py \
  --cfg_file cfg_files/release/neuman/hugs_human.yaml \
  --cfg_id 0 \
  dataset.seq=lab
```

特点：

- `mode=human`
- 不创建场景高斯。
- 会训练人体高斯、验证人体区域指标，并输出 canonical 视频。

### 6.3 仅训练场景

```bash
python main.py \
  --cfg_file cfg_files/release/neuman/hugs_scene.yaml \
  --cfg_id 0 \
  dataset.seq=lab
```

特点：

- `mode=scene`
- 近似原 3DGS 场景训练路径。
- 不依赖人体动画输出。

### 6.4 快速烟测

原始入口的短步数命令：

```bash
python main.py \
  --cfg_file cfg_files/release/neuman/hugs_human_scene.yaml \
  --cfg_id 0 \
  dataset.seq=lab \
  train.num_steps=100 \
  train.val_interval=50 \
  train.save_ckpt_interval=100 \
  train.anim_interval=-1
```

如果没有 AMASS，推荐本地辅助脚本：

```bash
python scripts/run_neuman_human_scene_noamass.py \
  --seq lab \
  --quick \
  --quick-steps 100 \
  --exp-name smoke_lab
```

该脚本会输出：

```text
LOGDIR=<实际输出目录>
```

并把结果指标打印到终端。

### 6.5 使用 Human3R 初始化 HUGS

先转换 Human3R 输出：

```bash
python scripts/convert_human3r_to_hugs.py \
  --result-dir /path/to/human3r/output/my_run \
  --out-dir data/human3r_hugs/my_run
```

再用新增 Human3R 配置训练：

```bash
python main.py \
  --cfg_file cfg_files/release/human3r/hugs_human_scene.yaml \
  dataset_path=data/human3r_hugs/my_run \
  dataset.seq=my_run \
  train.num_steps=100 \
  train.val_interval=50 \
  train.save_ckpt_interval=100
```

Human3R 数据集当前不提供 AMASS novel-pose 动画，配置中默认 `train.anim_interval=-1`，最终 `trainer.animate()` 会跳过动画数据集。

## 7. 代码执行流程总览

本节记录当前代码中“原版 HUGS/NeuMan 训练”和“Human3R 初始化训练”的真实执行路径。两条路线共用 `main.py` 和 `GaussianTrainer`，主要差异在配置文件、数据集 loader、初始场景点云、SMPL 参数来源，以及 animation/all split 的处理。

### 7.1 `main.py` 公共入口

入口命令统一是：

```bash
python main.py --cfg_file <yaml> [--cfg_id <id>] [OmegaConf overrides...]
```

执行顺序：

1. `OmegaConf.load(args.cfg_file)` 读取 YAML。
2. `get_cfg_items(cfg_file)` 把 YAML 中所有列表字段展开为实验列表。
3. 如果传了 `--cfg_id`，只跑指定实验；否则按顺序跑完展开后的所有实验。
4. 对每个实验执行 `OmegaConf.merge(default_cfg, cfg_item, OmegaConf.from_cli(extras))`。
5. `safe_state(seed=cfg.seed)` 设置随机种子。
6. `get_logger(cfg)` 创建输出目录、`train.log`、`config_train.yaml`。
7. 创建 `GaussianTrainer(cfg)`。
8. 非 eval 模式下执行 `trainer.train()`，随后 `trainer.save_ckpt()` 保存 final ckpt。
9. 执行最终 `trainer.validate()`。
10. 执行 `trainer.render_full_sequence(keep_images=True)`，保存 GT 视角全帧 PNG 和 MP4。
11. 写入 `results_train.json` 或 `results_eval.json`。
12. `human`/`human_scene` 模式下执行 `trainer.animate()`、`render_canonical(a_pose)`、`render_canonical(da_pose)`。

注意：当前 `render_full_sequence()` 在写 `results_train.json` 之前执行，因此如果全帧视频写入失败，结果 JSON 也可能还没落盘。

### 7.2 原版 HUGS/NeuMan 训练流程

典型命令：

```bash
python main.py \
  --cfg_file cfg_files/release/neuman/hugs_human_scene.yaml \
  --cfg_id 0 \
  dataset.seq=lab \
  exp_name=neuman_lab_original_6000_vis \
  train.num_steps=6000 \
  train.anim_interval=-1
```

关键配置：

```yaml
dataset.name: neuman
dataset.seq: ["lab", "citron", "seattle", "bike", "jogging", "parkinglot"]
mode: human_scene
human.name: hugs_trimlp
train.anim_interval: 15000
```

单独跑 `lab` 时建议加 `--cfg_id 0` 或把 YAML 里的 `dataset.seq` 改成单值，否则主程序会循环多个实验。

`GaussianTrainer.__init__()` 中的数据集构造：

- `get_train_dataset(cfg)` 创建 `NeumanDataset(seq, 'train')`。
- `get_val_dataset(cfg)` 创建 `NeumanDataset(seq, 'val')`。
- `get_anim_dataset(cfg)`：如果 `train.anim_interval < 0`，当前本地代码直接返回 `None`；否则创建 `NeumanDataset(seq, 'anim')` 并需要 AMASS/SFU 动作文件。
- `get_all_dataset(cfg)` 创建 `NeumanDataset(seq, 'all')`，这是本地新增能力，用于最终 GT 视角全帧视频。

`NeumanDataset` 读取路径：

```text
data/neuman/dataset/<seq>/
├── sparse/                         # COLMAP 相机和点云
├── images/                         # 原始图像
└── 4d_humans/
    ├── smpl_optimized_aligned_scale.npz
    └── sam_segmentations/
```

`NeumanDataset` 的主要处理：

- 用 `NeuManReader.read_scene(..., smpl_type='optimized')` 读取相机、图像、COLMAP 点云和场景对象。
- 从 `4d_humans/smpl_optimized_aligned_scale.npz` 读取 HUGS 原版 SMPL 参数。
- `train/val` split 来自 `get_data_splits(scene)`，`lab` 当前训练帧 82、验证帧 10。
- `all` split 使用 `idx=i`，按原始 0..102 帧顺序渲染，不参与训练。
- `anim` split 会读取 `data/SFU` 或 `data/MPI_mosh` 的 AMASS 动作，再套用 `alignment(seq)`，只用于 novel-pose animation。
- `init_pcd` 来自 NeuMan/COLMAP 场景点云，例如 `lab` 初始约 9K 点，随后训练中逐步 densify。
- 每个 datum 包含 `rgb/mask/bbox`、相机矩阵、SMPL `betas/global_orient/body_pose/transl/scale`、near/far，并缓存到 CUDA。

模型初始化：

- 人体模型为 `HUGS_TRIMLP`。
- 初始 `betas` 来自 val dataset 第一帧。
- 非 eval 模式会先 `human_gs.initialize()`，再执行 `optimize_init(self.human_gs, num_steps=7000)`。
- 训练用 SMPL pose/trans 来自 train dataset，并创建可优化的 `body_pose/global_orient/transl` 参数。
- 场景模型为 `SceneGS`，如果没有指定 `scene.ckpt`，通过 `scene_gs.create_from_pcd(train_dataset.init_pcd, train_dataset.radius)` 从 COLMAP 点云初始化。

训练循环与 Human3R 路线相同，见 7.4。

最终输出示例：

```text
output/human_scene/neuman/lab/hugs_trimlp/<exp_name>/<timestamp>/
├── render_all_neuman_lab_final.mp4
├── render_all/00000.png ... 00102.png
├── results_train.json
├── val/full_final_*.png
├── val/human_final_*.png
└── canon_neuman_lab_final_*.mp4
```

### 7.3 Human3R 初始化训练流程

Human3R 路线先把 Human3R 输出转换成 HUGS 能读取的数据目录，再用 `dataset.name=human3r` 训练。推荐使用严格拟合后的目录，而不是直接使用粗映射目录：

```text
data/human3r_hugs/lab_fit_smpl/
```

典型命令：

```bash
python main.py \
  --cfg_file cfg_files/release/human3r/hugs_human_scene.yaml \
  dataset_path=data/human3r_hugs/lab_fit_smpl \
  dataset.seq=lab_fit_smpl \
  exp_name=human3r_fit_smpl_6000_vis \
  train.num_steps=6000 \
  train.anim_interval=-1
```

关键配置：

```yaml
dataset.name: human3r
dataset.seq: converted
dataset_path: ''
mode: human_scene
human.name: hugs_trimlp
train.anim_interval: -1
```

`dataset_path` 必须指向转换后的 Human3R-HUGS 目录。`dataset.seq` 只影响输出路径和视频文件名，不负责定位数据。

`GaussianTrainer.__init__()` 中的数据集构造：

- `get_train_dataset(cfg)` 创建 `Human3RDataset(cfg.dataset_path, 'train')`。
- `get_val_dataset(cfg)` 创建 `Human3RDataset(cfg.dataset_path, 'val')`。
- `get_anim_dataset(cfg)` 因 `train.anim_interval < 0` 返回 `None`；Human3R 当前没有 AMASS animation split。
- `get_all_dataset(cfg)` 创建 `Human3RDataset(cfg.dataset_path, 'all')`，最终按原始 103 帧输出 GT 视角全帧视频。

`Human3RDataset` 读取路径：

```text
data/human3r_hugs/<seq>/
├── images/*.png
├── masks/*.png
├── cameras.npz
├── scene/points.npz
└── 4d_humans/smpl_optimized_aligned_scale.npz
```

`Human3RDataset` 的主要处理：

- 从 `cameras.npz` 读取 `c2w/intrinsics/image_sizes`。
- 从 `scene/points.npz` 读取 Human3R depth/conf 反投影得到的场景点云，作为 `SceneGS` 初始化点云。
- 从 `4d_humans/smpl_optimized_aligned_scale.npz` 读取 HUGS 需要的 SMPL 参数。
- split 由本地 `get_data_splits(num_frames)` 生成：`val=list(range(2, num_frames, 5))`，其余为 train，`all` 为全部帧。
- 每帧从 mask 计算 bbox。mask 为空时 bbox 回退到整图。
- 每个 datum 包含 `rgb/mask/bbox`、相机矩阵、SMPL `betas/global_orient/body_pose/transl/scale`、near/far，并缓存到 CUDA。

Human3R 的 SMPL 参数来源有两种：

- 粗转换：`scripts/convert_human3r_to_hugs.py` 直接从 Human3R `smpl/*.npz` 近似取前 24 个 SMPL-X joints 映射到 SMPL。
- 严格拟合：`scripts/fit_smpl_to_human3r_smplx.py` 复现 Human3R head-centered SMPL-X forward，经 `smplx2smpl.pkl` 映射到 SMPL topology，再优化 `global_orient/body_pose/betas/transl/scale`。

当前正式实验应使用严格拟合结果。原因是 Human3R `smpl/*.npz::transl` 默认是 `person_center='head'` 下的头部中心坐标，不能直接当作 HUGS SMPL 的整体平移；直接粗映射容易导致人体相对场景位置或人体质量异常。

模型初始化和训练循环与原版 HUGS 相同：

- 人体仍是 HUGS 的 `HUGS_TRIMLP`/SMPL，不是 SMPL-X。
- 场景仍是 `SceneGS`。
- 区别只在人体 SMPL 参数和场景点云由 Human3R 转换目录提供。

最终输出示例：

```text
output/human_scene/human3r/lab_fit_smpl/hugs_trimlp/<exp_name>/<timestamp>/
├── render_all_human3r_lab_fit_smpl_final.mp4
├── render_all/00000.png ... 00102.png
├── results_train.json
├── val/full_final_*.png
├── val/human_final_*.png
└── canon_human3r_lab_fit_smpl_final_*.mp4
```

### 7.4 `GaussianTrainer.train()` 公共训练循环

原版 NeuMan 和 Human3R 路线进入训练后使用同一段循环：

1. 创建 `RandomIndexIterator(len(train_dataset))`，每步随机取一个训练帧。
2. 更新 `SceneGS` 和人体模型学习率。
3. 从缓存中取 `data = train_dataset[rnd_idx]`。
4. 人体分支执行 `human_gs.forward(smpl_scale=data['smpl_scale'][None], dataset_idx=rnd_idx, is_train=True)`。
5. 场景分支在 `t_iter >= scene.opt_start_iter` 后执行 `scene_gs.forward()`；release 配置中 `scene.opt_start_iter=-1`，所以从一开始训练场景。
6. 调用 `render_human_scene()` 联合渲染人体和场景。
7. `HumanSceneLoss` 计算 L1/SSIM/LPIPS patch、人像分离、LBS 等损失。
8. `loss.backward()`。
9. 根据 viewspace gradient 对场景和人体高斯执行 densify/prune。
10. 分别 step/zero `human_gs.optimizer` 和 `scene_gs.optimizer`。
11. 到 `save_ckpt_interval` 或最后一步时保存 ckpt 和 scene ply；当前本地代码会捕获磁盘写入异常并 warning。
12. 到 `val_interval` 时执行 `validate(t_iter)`。
13. `t_iter == 0` 时保存初始 mesh，并渲染初始 canonical 视频。
14. 若 `anim_interval > 0`，到间隔时执行 animation/canonical。
15. 每 1000 步调用 `oneupSHdegree()` 提升人体和场景 SH degree。

### 7.5 `validate()`、`render_full_sequence()` 和视频保存

`validate(iter)`：

- 遍历 `val_dataset`。
- 评估时人体 forward 显式传入当前验证帧的 `global_orient/body_pose/betas/transl/smpl_scale`，`dataset_idx=-1`。
- 渲染全图，保存 `val/full_<iter>_<idx>.png`。
- 根据 `bbox` 裁剪人体区域，保存 `val/human_<iter>_<idx>.png`。
- 统计 `hugs_psnr/ssim/lpips` 和 `hugs_human_psnr/ssim/lpips`。
- 写入 `val/eval_<iter>.pth`，并把均值放入 `self.eval_metrics`。

`render_full_sequence(iter=None, keep_images=True)`：

- 依赖 `all_dataset`。当前 NeuMan 和 Human3R 都支持 `all` split。
- 按原始帧顺序逐帧 forward 人体、forward 场景、联合渲染。
- PNG 写入 `<logdir>/render_all/00000.png` 等。
- MP4 写入 `<logdir>/render_all_<dataset.name>_<dataset.seq>_<iter>.mp4`。
- 当前 `main.py` 调用时传 `keep_images=True`，所以逐帧 PNG 会保留。

`animate()`：

- 如果 `anim_dataset is None`，只打印 `No animation dataset found` 并返回。
- NeuMan 原版如果 `train.anim_interval >= 0` 且 AMASS/SFU 数据存在，可创建 novel-pose animation。
- Human3R 当前没有 animation dataset。

## 8. 输出目录说明

`main.py` 自动生成输出目录。

`human` 或 `human_scene`：

```text
output/<mode>/<dataset.name>/<dataset.seq>/<human.name>/<exp_name>/<timestamp>/
```

`scene`：

```text
output/<mode>/<dataset.name>/<dataset.seq>/<exp_name>/<timestamp>/
```

典型内容：

```text
<logdir>/
├── train.log 或 eval.log
├── config_train.yaml 或 config_eval.yaml
├── results_train.json 或 results_eval.json
├── ckpt/
│   ├── human_XXXXXX.pth
│   ├── scene_XXXXXX.pth
│   ├── human_final.pth
│   └── scene_final.pth
├── train/
│   └── 000000.png 等训练对比图
├── val/
│   ├── full_<iter>_<idx>.png
│   ├── human_<iter>_<idx>.png
│   └── eval_<iter>.pth
├── meshes/
│   ├── human_<iter>_splat.ply
│   └── scene_<iter>_splat.ply
├── anim/
├── anim_<dataset>_<seq>_<iter>.mp4
├── render_all_<dataset>_<seq>_<iter>.mp4
├── canon/
└── canon_<dataset>_<seq>_<iter>.mp4
```

当前本地已有预训练模型示例：

```text
output/pretrained_models/lab/
├── config_train.yaml
├── human_final.pth
└── scene_final.pth
```

## 9. 评估已有模型

使用 `scripts/evaluate.py`：

```bash
python scripts/evaluate.py -o output/pretrained_models/lab
```

脚本行为：

1. 读取 `<output_dir>/config_train.yaml`。
2. 自动查找 human 和 scene ckpt。
3. 设置 `cfg.eval=true`。
4. 创建 `GaussianTrainer`。
5. 执行 `validate()`。
6. 执行 `animate()`。
7. 写入 `results_eval.json`。

注意：如果评估 `human` 或 `human_scene` 且没有 AMASS，`animate()` 仍可能失败。需要临时禁用动画时，可参考 `scripts/run_neuman_human_scene_noamass.py` 的 monkey patch 思路。

输出指标：

- `hugs_psnr`
- `hugs_ssim`
- `hugs_lpips`
- `hugs_human_psnr`
- `hugs_human_ssim`
- `hugs_human_lpips`

其中 `hugs_human_*` 是根据 mask bbox 裁剪人体区域后计算。

## 10. 关键配置字段

### 10.1 全局字段

| 字段 | 说明 |
| --- | --- |
| `seed` | 随机种子 |
| `mode` | `human_scene`、`human`、`scene` |
| `output_path` | 输出根目录，默认 `output` |
| `exp_name` | 实验名，会进入输出路径 |
| `dataset.name` | 当前支持 `neuman`、`human3r` |
| `dataset.seq` | 序列名 |
| `eval` | 是否评估模式 |
| `bg_color` | 验证/动画背景色，`white` 或 `black` |

### 10.2 训练字段

| 字段 | 说明 |
| --- | --- |
| `train.num_steps` | 训练总步数，release 默认约 15000 |
| `train.save_ckpt_interval` | ckpt 保存间隔 |
| `train.val_interval` | 验证间隔 |
| `train.anim_interval` | 动画/canonical 渲染间隔；小于等于 0 可避免中途动画 |
| `train.optim_scene` | 是否优化场景 optimizer |
| `train.save_progress_images` | 是否保存 canonical 训练进度图 |

### 10.3 人体字段

| 字段 | 说明 |
| --- | --- |
| `human.name` | `hugs_trimlp` 或 `hugs_wo_trimlp` |
| `human.ckpt` | 指定人体 ckpt；为空时尝试从当前 logdir ckpt 自动恢复 |
| `human.sh_degree` | 人体 SH 最大阶数 |
| `human.n_subdivision` | SMPL 细分层数 |
| `human.use_deformer` | 是否使用 deformer |
| `human.disable_posedirs` | 是否禁用 posedirs |
| `human.optim_pose` | 是否优化 SMPL pose |
| `human.optim_betas` | 是否优化 betas |
| `human.optim_trans` | 是否优化平移 |
| `human.canon_nframes` | canonical 视频帧数 |
| `human.loss.*` | 人体训练损失权重 |
| `human.densify_*` | 人体高斯 densification/prune 策略 |

### 10.4 场景字段

| 字段 | 说明 |
| --- | --- |
| `scene.ckpt` | 指定场景 ckpt |
| `scene.sh_degree` | 场景 SH 最大阶数 |
| `scene.add_bg_points` | 是否为背景球添加点 |
| `scene.num_bg_points` | 背景点数量 |
| `scene.clean_pcd` | 是否用 Open3D 去除离群点 |
| `scene.opt_start_iter` | 场景开始优化的 iteration |
| `scene.loss.*` | 场景损失权重 |
| `scene.densify_*` | 场景高斯 densification/prune 策略 |

## 11. 消融实验配置

`cfg_files/ablation/neuman/` 中配置会通过列表值展开实验：

- `abl_deformer.yaml`：比较 `human.use_deformer=true/false`。
- `abl_densify.yaml`：比较 `human.densify_from_iter=3000/20000`。
- `abl_lhuman.yaml`：关闭 `humansep_w`。
- `abl_trimlp.yaml`：使用 `hugs_wo_trimlp`，关闭 TriMLP 相关能力。

运行方式示例：

```bash
python main.py \
  --cfg_file cfg_files/ablation/neuman/abl_deformer.yaml \
  dataset.seq=lab \
  --cfg_id 0
```

注意：`--cfg_id` 是展开后的索引；也可以用命令行覆盖列表字段避免一次跑多组。

## 12. 本地修改说明

当前工作区有本地改动，本手册按这些改动记录当前行为：

### 12.1 `hugs/utils/general.py:create_video`

当前本地版本已从 ffmpeg shell 命令改为 OpenCV `VideoWriter`：

- 自动查找 `img_folder/*.png`。
- 没有 PNG 时跳过视频导出并 warning。
- 第一帧读不到时跳过导出。
- 自动把奇数宽高 pad 到偶数，方便 MP4 编码。
- MP4 writer 打不开时 fallback 到 AVI。

这意味着运行环境不再强依赖 `/usr/bin/ffmpeg`，但仍依赖 `opencv-python`。

### 12.2 `scripts/run_neuman_human_scene_noamass.py`

本地新增脚本用于没有 AMASS 数据时训练 NeuMan：

- 将 `gst.get_anim_dataset` 替换为返回 `None`。
- quick 模式将 `gst.optimize_init` 替换为 no-op，跳过 7000 步人体初始化。
- 默认配置文件是 `cfg_files/release/neuman/hugs_human_scene.yaml`。
- 默认 `--seq lab`。
- 支持追加 OmegaConf dotlist override。

常用：

```bash
python scripts/run_neuman_human_scene_noamass.py --seq lab --quick --quick-steps 100
```

追加覆盖示例：

```bash
python scripts/run_neuman_human_scene_noamass.py \
  --seq lab \
  --exp-name full_lab_noamass \
  train.num_steps=15000 \
  train.val_interval=1000
```

### 12.3 Human3R adapter

新增文件：

```text
scripts/convert_human3r_to_hugs.py
hugs/datasets/human3r.py
cfg_files/release/human3r/hugs_human_scene.yaml
```

用途：

- `convert_human3r_to_hugs.py` 将 Human3R 保存结果转换为 HUGS 可读取目录。
- `Human3RDataset` 读取转换后的图像、mask、相机、SMPL 参数和场景点云。
- `GaussianTrainer` 现在支持 `dataset.name=human3r`。
- 2026-05-14 修正：Human3R `smpl/*.npz` 的 `global_orient/transl` 默认按 camera frame 处理，并通过该帧 `c2w` 转换到 HUGS world frame。旧版本转换数据如果没有该转换，full-sequence 渲染中人体会明显漂移、变淡或缺失。

已知限制：

- 当前 Human3R SMPL-X 到 HUGS SMPL 的转换是近似映射，适合流程验证，不等价于严格模型转换。
- 当前只支持单人，默认 `--person-index 0`。
- 当前 Human3R 数据集没有动画 split，`anim_dataset=None`。
- 相机投影沿用 HUGS 现有 fov-based 投影路径；如果 Human3R 输出的 principal point 明显偏离中心，可能需要改成 full intrinsic projection。

### 12.4 Rasterizer 返回值兼容

当前环境的 `diff_gaussian_rasterization` 返回 tuple 中包含超过两个元素。`hugs/renderer/gs_renderer.py` 已兼容该情况：

- 渲染图像从 tuple 中选择形状类似 `(3,H,W)` 或 `(4,H,W)` 的 tensor。
- `radii` 从 tuple 中选择长度等于高斯数量的一维 tensor。
- 如果找不到 `radii`，回退为全可见 mask。

这个修改避免了旧代码 `rendered_image, radii = rasterizer(...)` 在当前 CUDA 扩展下触发 `ValueError: too many values to unpack`。

### 12.5 完整序列渲染视频

当前 `GaussianTrainer` 已新增 `render_full_sequence()`，并在 `main.py` 和 `scripts/run_neuman_human_scene_noamass.py` 的最终验证后自动调用。行为如下：

- 当前支持 `dataset.name=human3r`，会创建 `Human3RDataset(split='all')`。
- 当前也支持 `dataset.name=neuman`，会创建 `NeumanDataset(split='all')`，按原始 NeuMan 相机顺序渲染全帧。
- 按原始帧顺序渲染全部帧，而不是只渲染 train/val 子集。
- 临时帧写入 `<logdir>/render_all/00000.png` 等文件。
- 默认导出 `<logdir>/render_all_<dataset>_<seq>_<iter>.mp4`。
- 当前 `main.py` 调用 `render_full_sequence(keep_images=True)`，所以生成视频后会保留 `<logdir>/render_all/*.png` 全帧图片。

这项功能用于对比 Human3R 初始化和原版 HUGS/NeuMan 训练结果。典型输出：

```text
render_all_human3r_lab_fit_smpl_final.mp4
render_all_neuman_lab_final.mp4
```

## 13. 常见问题

### 13.1 `rg: command not found`

当前环境没有 `rg`，可以用 `find`、`grep` 替代。

### 13.2 没有 AMASS 导致动画失败

训练、验证和 GT 视角全帧视频不需要 AMASS。当前本地代码中，如果设置 `train.anim_interval < 0`，`get_anim_dataset()` 会直接返回 `None`，最终 `animate()` 会打印 `No animation dataset found` 并跳过。

如果需要 novel-pose animation，仍然要准备 AMASS/SFU 数据。可选方案：

- 准备 `data/SFU` 和 `data/MPI_mosh`。
- 或使用 `scripts/run_neuman_human_scene_noamass.py`。
- 或在 `main.py` 入口使用 `train.anim_interval=-1` 禁用 animation 数据加载。

### 13.3 一次跑了很多序列

原因通常是 YAML 中 `dataset.seq` 是列表，并且没有传 `--cfg_id`。由于本仓库先展开 YAML 列表、再合并命令行 extras，只写 `dataset.seq=lab` 不足以阻止循环。使用：

```bash
--cfg_id 0 dataset.seq=lab
```

或复制一份单序列 YAML，把 `dataset.seq` 改成字符串而不是列表。

### 13.4 CUDA 扩展导入失败

确认已经安装子模块：

```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

并确认 PyTorch/CUDA 版本与编译扩展一致。

### 13.5 显存不足

可尝试：

- 减少训练步数先烟测。
- 降低 `human.max_n_gaussians` 或 `scene.max_n_gaussians`。
- 推迟或减少 densification。
- 只跑 `human` 或 `scene` 模式定位问题。

## 14. 推荐工作流

1. 确认环境：

```bash
conda activate hugs
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

2. 确认数据：

```bash
find data/neuman/dataset/lab -maxdepth 2 -type d | sort
```

3. 先跑无 AMASS quick smoke：

```bash
python scripts/run_neuman_human_scene_noamass.py --seq lab --quick --quick-steps 100
```

4. 看 `LOGDIR`、`results_train.json`、`val/` 图像。

5. 再拉长训练：

```bash
python scripts/run_neuman_human_scene_noamass.py \
  --seq lab \
  --exp-name lab_noamass_15000 \
  train.num_steps=15000 \
  train.val_interval=1000 \
  train.save_ckpt_interval=5000
```

6. 若要完整官方动画，补齐 AMASS 后用原始 `main.py` 或 `scripts/evaluate.py`。

## 16. 当前 lab Human3R 运行记录

### 16.1 2026-05-15 原版 HUGS 与 Human3R 初始化 6000 步对照

Human3R 严格拟合 SMPL 初始化 6000 步：

```bash
python main.py \
  --cfg_file cfg_files/release/human3r/hugs_human_scene.yaml \
  dataset_path=data/human3r_hugs/lab_fit_smpl \
  dataset.seq=lab_fit_smpl \
  exp_name=human3r_fit_smpl_6000_vis \
  train.num_steps=6000 \
  train.anim_interval=-1
```

输出：

```text
output/human_scene/human3r/lab_fit_smpl/hugs_trimlp/human3r_fit_smpl_6000_vis/2026-05-14_23-59-57/
├── render_all_human3r_lab_fit_smpl_final.mp4
├── render_all/                 # 103 帧 PNG
└── results_train.json
```

最终指标：

```text
hugs_psnr:       18.9311
hugs_ssim:       0.6439
hugs_lpips:      0.1806
hugs_human_psnr: 15.2528
hugs_human_ssim: 0.5131
hugs_human_lpips:0.1648
```

原版 HUGS/NeuMan `lab` 6000 步（历史记录，已被 2026-05-19 Exp0 标准 baseline 替换）：

```bash
python main.py \
  --cfg_file cfg_files/release/neuman/hugs_human_scene.yaml \
  --cfg_id 0 \
  dataset.seq=lab \
  exp_name=neuman_lab_original_6000_vis \
  train.num_steps=6000 \
  train.anim_interval=-1
```

输出：

```text
output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_original_6000_vis/2026-05-15_00-29-45/
├── render_all_neuman_lab_final.mp4
├── render_all/                 # 103 帧 PNG
└── results_train.json
```

最终指标（历史口径，仅作追溯；后续对比以第 20 节统一评估结果为准）：

```text
hugs_psnr:       25.8792
hugs_ssim:       0.9031
hugs_lpips:      0.0838
hugs_human_psnr: 19.0826
hugs_human_ssim: 0.7477
hugs_human_lpips:0.1653
```

2026-05-19 已重新按 release 配置复现 Exp0，并补充 6000-step 与 full-step 统一评估结果，见第 20 节。后续 anchor/correction/Human3R 初始化实验不再引用本段旧原版数值作为主 baseline。

结论记录：

- 原版 HUGS/NeuMan 的整体指标和人体 PSNR/SSIM 明显高于 Human3R 初始化路线。
- Human3R 初始化路线的人体 LPIPS 接近原版，但 PSNR/SSIM 仍明显偏低，说明人体质量问题不能只归因于训练步数。
- 这次原版对照验证了 `/hdd/u202420081000003/ml-hugs/data/neuman/dataset/lab` 可直接跑原版 HUGS，并能保存 GT 视角全帧视频。
- 运行原版配置时必须注意 `dataset.seq` 列表展开问题；建议使用 `--cfg_id 0`。本次若不加 `--cfg_id`，第一个 `lab` 完成后会继续启动下一项实验。

### 16.2 2026-05-15 同分辨率原版路线控制实验

为了判断 Human3R 初始化质量下降是否主要由 512x288 低分辨率造成，新增了一个原版 NeuMan 路线的低分辨率序列：

```bash
python scripts/create_neuman_downsampled_seq.py \
  --src data/neuman/dataset/lab \
  --dst data/neuman/dataset/lab_h3rres \
  --width 512 \
  --height 288
```

该脚本只创建新序列，不修改原始 `lab`。处理内容：

- `images/` 从 1276x717 下采样到 512x288。
- `4d_humans/sam_segmentations`、`4d_humans/masks`、`4d_humans/images` 同步下采样到 512x288。
- `sparse/cameras.txt` 中 `PINHOLE` 相机内参按比例缩放，得到 `PINHOLE 512 288 441.3712853 441.8329707 256 144`。
- `sparse/images.txt` 中 COLMAP 二维观测点同步缩放。
- 场景 3D 点云和 SMPL 参数保持原版 NeuMan/HUGS 路线不变。

运行 6000 步：

```bash
python main.py \
  --cfg_file cfg_files/release/neuman/hugs_human_scene.yaml \
  --cfg_id 0 \
  dataset.seq=lab_h3rres \
  exp_name=neuman_lab_h3rres_6000_vis \
  train.num_steps=6000 \
  train.anim_interval=-1
```

输出：

```text
output/human_scene/neuman/lab_h3rres/hugs_trimlp/neuman_lab_h3rres_6000_vis/2026-05-15_01-06-26/
├── render_all_neuman_lab_h3rres_final.mp4
├── render_all/                 # 103 帧 PNG
└── results_train.json
```

最终指标：

```text
hugs_psnr:       26.3263
hugs_ssim:       0.9208
hugs_lpips:      0.0392
hugs_human_psnr: 19.2890
hugs_human_ssim: 0.7276
hugs_human_lpips:0.0843
```

同分辨率结论：

- Human3R 转换图像和 `lab_h3rres` 图像几乎一致，逐帧平均图像 PSNR 约 55.03 dB。
- Human3R mask 与原版低分辨率 SAM mask 平均 IoU 约 0.90，mask 有差异但不足以解释巨大指标差距。
- 在同样 512x288 下，原版 NeuMan 路线最终 `hugs_human_psnr=19.2890`、`hugs_human_ssim=0.7276`、`hugs_human_lpips=0.0843`；Human3R 严格拟合 SMPL 路线为 `15.2528`、`0.5131`、`0.1648`。
- 因此当前 Human3R 初始化质量下降不能归因于输入分辨率。主要问题更可能来自 Human3R 路线提供给 HUGS 的几何初始化，包括 Human3R 相机/坐标系、depth/conf 反投影点云、SMPL-X 到 SMPL 拟合结果与 HUGS 监督之间的残余不对齐。

2026-05-15 进一步对最终 `render_all/` 的 103 帧重算了统一指标，排除验证集划分差异：

```text
原版低分辨率 render_all vs lab_h3rres GT:
all_psnr:   26.6986
human_psnr: 19.5793
bg_psnr:    28.4181
human_mae:  15.2428

Human3R render_all vs Human3R GT/mask:
all_psnr:   19.0021
human_psnr: 14.4080
bg_psnr:    19.7513
human_mae:  29.6643

Human3R render_all vs lab_h3rres GT/mask:
all_psnr:   19.0332
human_psnr: 14.4248
bg_psnr:    19.7814
human_mae:  29.5904
```

额外诊断：

- Human3R 图像与 `lab_h3rres` 图像平均 PSNR 约 55 dB，输入图像几乎一致。
- Human3R mask 与原版低分辨率 SAM mask 平均 IoU 约 0.90，mask 差异存在但不是主因。
- Human3R 拟合后的 SMPL 初始投影 bbox 与自身 mask 的平均 bbox L1 约 12 px，粗略投影位置并没有严重错位；原版 NeuMan 初始 bbox 反而更粗，但仍训练得好。
- Human3R 的 SMPL-X 到 SMPL 拟合报告中 `final_vertex_l1_m≈0.0083`、`final_joint_l1_m≈0.0115`、`final_bbox_l1_px≈12.66`，说明“SMPL-X 到 SMPL 拟合误差”不是唯一主因。
- Human3R 路线的场景点云来自 monocular depth/conf 反投影，初始化点数为 500000；原版低分辨率路线使用 COLMAP sparse 点云，初始化点数为 9013。两者相机轨迹尺度、点云分布和 scene position lr 明显不同，Human3R 场景最终背景指标也显著低，说明 Human3R depth/camera/scene 初始化质量是当前优先怀疑对象。

### 16.3 2026-05-13 Human3R 输出检查和早期烟测

2026-05-13 检查了 lab 场景 Human3R 输出：

```text
data/neuman/dataset/lab/human3r/
├── camera/  # 103 个 .npz，包含 pose 和 intrinsics
├── color/   # 103 个 .png
├── conf/    # 103 个 .npy
├── depth/   # 103 个 .npy
└── smpl/    # 103 个 .npz
```

抽查 `000000`：

```text
depth: (288, 512), float32
conf:  (288, 512), float32
camera keys: pose (4,4), intrinsics (3,3)
smpl keys: scores, msk, shape, rotvec, transl, expression
shape:  (1, 10)
rotvec: (1, 53, 3)
transl: (1, 3)
msk:    (1, 288, 512)
```

该结构适配 `scripts/convert_human3r_to_hugs.py`。已转换到：

```text
data/human3r_hugs/lab/
```

转换命令：

```bash
python scripts/convert_human3r_to_hugs.py \
  --result-dir data/neuman/dataset/lab/human3r \
  --out-dir data/human3r_hugs/lab \
  --conf-threshold 1.5 \
  --point-stride 2 \
  --max-scene-points 500000
```

转换结果：

```text
Frames: 103
Scene points: 500000
cameras.npz: c2w (103,4,4), intrinsics (103,3,3)
4d_humans/smpl_optimized_aligned_scale.npz:
  global_orient (103,3)
  body_pose     (103,69)
  betas         (103,10)
  transl        (103,3)
  scale         (103,)
  bbox          (103,5)
```

尝试在当前机器运行 Human3R 初始化的 lab 短训练：

```bash
source /home/u202420081000003/anaconda3/etc/profile.d/conda.sh
conda activate hugs
python scripts/run_neuman_human_scene_noamass.py \
  --seq lab \
  --cfg-file cfg_files/release/human3r/hugs_human_scene.yaml \
  --quick \
  --quick-steps 10 \
  --exp-name human3r_lab_smoke \
  dataset_path=data/human3r_hugs/lab \
  train.val_interval=10 \
  train.save_ckpt_interval=10
```

当前环境无法训练，原因是 GPU/driver 不可见：

```text
torch 1.13.1
cuda built 11.7
cuda available False
device count 0
nvidia-smi: command not found
RuntimeError: Found no NVIDIA driver on your system.
```

换到可见 NVIDIA GPU 的节点后，可直接复用上面的训练命令继续跑。

2026-05-13 已通过 ssh 在 `gpu-l40-1` 上重新运行：

```text
GPU: NVIDIA L40, 46068 MiB
torch: 1.13.1
cuda build: 11.7
cuda available: True
```

1 step smoke 命令：

```bash
ssh gpu-l40-1 'source /home/u202420081000003/anaconda3/etc/profile.d/conda.sh && \
  conda activate hugs && \
  cd /hdd/u202420081000003/ml-hugs && \
  python scripts/run_neuman_human_scene_noamass.py \
    --seq lab \
    --cfg-file cfg_files/release/human3r/hugs_human_scene.yaml \
    --quick \
    --quick-steps 1 \
    --exp-name human3r_lab_smoke_1step_v2 \
    dataset_path=data/human3r_hugs/lab \
    train.val_interval=1 \
    train.save_ckpt_interval=1 \
    human.canon_nframes=2 \
    train.save_progress_images=false \
    scene.densify_from_iter=1000 \
    human.densify_from_iter=1000'
```

1 step 成功输出：

```text
LOGDIR=output/human_scene/human3r/lab/hugs_trimlp/human3r_lab_smoke_1step_v2/2026-05-13_23-17-05
final hugs_psnr: 15.0371
final hugs_ssim: 0.5299
final hugs_lpips: 0.6344
final hugs_human_psnr: 10.5638
final hugs_human_ssim: 0.3618
final hugs_human_lpips: 0.6888
```

10 step smoke 命令：

```bash
ssh gpu-l40-1 'source /home/u202420081000003/anaconda3/etc/profile.d/conda.sh && \
  conda activate hugs && \
  cd /hdd/u202420081000003/ml-hugs && \
  python scripts/run_neuman_human_scene_noamass.py \
    --seq lab \
    --cfg-file cfg_files/release/human3r/hugs_human_scene.yaml \
    --quick \
    --quick-steps 10 \
    --exp-name human3r_lab_smoke_10step \
    dataset_path=data/human3r_hugs/lab \
    train.val_interval=10 \
    train.save_ckpt_interval=10 \
    human.canon_nframes=2 \
    train.save_progress_images=false \
    scene.densify_from_iter=1000 \
    human.densify_from_iter=1000'
```

10 step 成功输出：

```text
LOGDIR=output/human_scene/human3r/lab/hugs_trimlp/human3r_lab_smoke_10step/2026-05-13_23-18-21
final hugs_psnr: 15.2165
final hugs_ssim: 0.5386
final hugs_lpips: 0.6020
final hugs_human_psnr: 10.5683
final hugs_human_ssim: 0.3641
final hugs_human_lpips: 0.6644
```

2026-05-13 追加了完整序列渲染视频功能后，在 `gpu-l40-1` 上先跑过 1 step smoke，确认最终会生成 103 帧组成的全序列视频：

```text
LOGDIR=output/human_scene/human3r/lab/hugs_trimlp/human3r_lab_renderall_smoke/2026-05-13_23-26-29
video=render_all_human3r_lab_final.mp4
```

随后在 `gpu-l40-1` 上跑完整步数 Human3R 初始化训练：

```bash
ssh gpu-l40-1 'source /home/u202420081000003/anaconda3/etc/profile.d/conda.sh && \
  conda activate hugs && \
  cd /hdd/u202420081000003/ml-hugs && \
  python scripts/run_neuman_human_scene_noamass.py \
    --seq lab \
    --cfg-file cfg_files/release/human3r/hugs_human_scene.yaml \
    --exp-name human3r_lab_full_renderall \
    dataset_path=data/human3r_hugs/lab'
```

完整训练输出：

```text
LOGDIR=output/human_scene/human3r/lab/hugs_trimlp/human3r_lab_full_renderall/2026-05-13_23-28-11
results=results_train.json
video=render_all_human3r_lab_final.mp4
video frames: 103
video size: 845K
```

最终指标：

```text
hugs_psnr: 17.3413
hugs_ssim: 0.6211
hugs_lpips: 0.2420
hugs_human_psnr: 11.8584
hugs_human_ssim: 0.4139
hugs_human_lpips: 0.4118
```


## 17. Anchor 实验记录（Exp A-E）

本节记录 2026-05-15 至 2026-05-17 期间围绕 anchor 的实验实现、可视化和当前结果。实验设计参考：

```text
test md/Anchor_Experiments_Implementation.md
```

核心目标：在 HUGS 的人体高斯和场景高斯之间建立可解释的局部语义锚点系统。当前先在 HUGS 项目内完成调试，再迁移到 HUGS 原训练路径中验证。

相关新增/修改文件主要包括：

```text
hugs/utils/anchor_io.py
hugs/utils/anchor_utils.py
hugs/utils/anchor_visualization.py
hugs/utils/anchor_token.py
scripts/debug_anchors.py
scripts/run_neuman_anchor_expe_smoke.py
hugs/models/hugs_trimlp.py
hugs/models/hugs_wo_trimlp.py
```

其中 `scripts/run_neuman_anchor_expe_smoke.py` 是 ExpE 的真实 HUGS 训练/导出入口，支持 `--preset smoke|normal`、`--export-only`、`--skip-train`、`--dataset-path`、`--no-resume` 等参数。

### 17.1 Exp A：模板人体 anchor 定义

目的：在 HUGS 使用的 SMPL/HUGS 模板人体上定义语义 anchor，并检查 anchor 是否落在正确的人体部位。

实现内容：

- 使用 HUGS 当前模板人体，而不是额外外部模板，避免模板坐标不一致。
- 定义语义 anchor，例如 back、buttocks、left/right elbow、left/right knee、left/right heel、left/right sole、toe、palm、fingers 等。
- 输出 anchor 顶点索引、世界/模板坐标、法线方向和区域颜色。
- 修正并补充 anchor normal 可视化，便于确认脚底、手掌、背部等方向是否合理。

输出目录：

```text
output/anchor_debug/expA_template/
```

关键输出：

```text
anchor_vertices.json
anchors.csv
anchors.pt
template_mesh_anchor_spheres.ply
template_mesh_anchor_spheres_normals.ply
template_anchors_with_normals.ply
template_mesh_anchor_regions.ply
template_anchor_region_points.ply
anchor_color_legend.ply
README_expA.md
```

可视化含义：

- `template_mesh_anchor_spheres*.ply`：人体 mesh 上的 anchor 小球。
- `template_mesh_anchor_spheres_normals.ply` / `template_anchors_with_normals.ply`：anchor 小球和法线方向箭头。
- `template_mesh_anchor_regions.ply`：人体局部区域按 anchor/身体部位上色。
- `template_anchor_region_points.ply`：anchor 周围区域点云化检查。

当前结果：ExpA 已通过。anchor 位置、区域颜色和 normal 标注可正常在 PLY 查看器中检查。

### 17.2 Exp B：人体 Gaussian 到 anchor 的绑定

目的：把 canonical space 中的人体高斯绑定到语义 anchor，建立 “哪个 anchor 控制哪些人体 Gaussian” 的关系。

实现内容：

- 对每个人体 Gaussian，结合 canonical 位置、LBS 权重、anchor 距离和局部半径计算绑定。
- 输出 top-1 anchor id、anchor 权重、local mask 和每个 anchor 控制的 Gaussian 统计。
- 可视化人体 Gaussian 按 anchor 上色，以及 local Gaussian 与 anchor 的空间关系。

输出目录：

```text
output/anchor_debug/expB_bind_init/
```

关键输出：

```text
anchor_binding_summary_init.csv
anchor_bindings_init.pt
anchors_hugs_canonical.csv
anchors_hugs_canonical.pt
human_gaussians_anchor_init.csv
human_gaussians_anchor_init.ply
human_gaussians_anchor_init_local.ply
human_gaussians_anchor_init_with_anchors.ply
human_gaussians_anchor_init_local_with_anchors.ply
human_gaussians_anchor_init_on_mesh.ply
human_gaussians_anchor_init_local_on_mesh.ply
renders/
README_expB.md
```

图片可视化：

```text
renders/human_gaussians_anchor_init_front.png
renders/human_gaussians_anchor_init_back.png
renders/human_gaussians_anchor_init_left.png
renders/human_gaussians_anchor_init_right.png
renders/human_gaussians_anchor_init_top.png
renders/human_gaussians_anchor_init_local_*.png
```

可视化含义：

- 彩色点是人体 Gaussian 的位置，不是 anchor 本身。
- 每种颜色表示该 Gaussian 被绑定到的 anchor。
- `*_local*` 只显示通过 local mask 过滤后的 anchor 局部高斯，便于观察 anchor 控制范围。
- `*_with_anchors.ply` 同时显示人体高斯和 anchor 小球。

当前结果：ExpB 效果较好，人体 Gaussian 和 anchor 的 canonical 绑定关系可视化清楚。

### 17.3 Exp C：posed 状态下 anchor 与人体 Gaussian 对齐

目的：验证 anchor 从 canonical space 跟随 HUGS/SMPL pose 变换到训练帧后，是否仍然落在正确人体部位，并检查 densify/迭代后绑定继承是否稳定。

实现内容：

- 修复早期模板错位问题：anchor 必须使用 HUGS 的 `smpl_template` 和 vitruvian template，而不是外部 SMPL 模板。
- 修复投影可视化中的 y 轴方向问题。
- 在真实 rasterizer 渲染图上叠加 anchor、anchor 标签、anchor 控制的人体高斯。
- 检查不同迭代步数下 anchor 是否随人体 pose 和优化后位姿正确移动。

输出目录：

```text
output/anchor_debug/expC_densify_inherit/
output/anchor_debug/expC_raster_overlay/
```

关键输出：

```text
output/anchor_debug/expC_densify_inherit/anchor_bindings_iter0000.pt
output/anchor_debug/expC_densify_inherit/anchor_bindings_iter3000.pt
output/anchor_debug/expC_densify_inherit/anchor_bindings_iter6000.pt
output/anchor_debug/expC_densify_inherit/anchor_bindings_iter12000.pt
output/anchor_debug/expC_densify_inherit/anchor_stats_iter*.csv
output/anchor_debug/expC_densify_inherit/human_gaussians_anchor_iter*.ply
output/anchor_debug/expC_densify_inherit/renders/*.png
output/anchor_debug/expC_raster_overlay/raster_anchor_binding_frame*_overlay.png
```

可视化含义：

- `human_gaussians_anchor_iter*.ply`：不同训练迭代下的人体 Gaussian anchor 绑定。
- `renders/human_gaussians_anchor_init_iter*_local_*.png`：不同视角下局部绑定可视化。
- `raster_anchor_binding_frame*_overlay.png`：3DGS rasterizer 渲染图上的 anchor 和受控 Gaussian overlay。

当前结果：已定位并修复 anchor 与人体 Gaussian 不对应的问题。修复后 buttocks、back、heel 等 anchor 不再错误投影到脖子等部位；anchor 使用优化后的 HUGS pose/transl 导出。

### 17.4 Exp D 第一部分：anchor-local Gaussian feature 构造与统计

目的：为后续 token 网络构造每个 anchor 局部的人体 Gaussian 特征，并分析这些特征是否具有稳定结构。

实现内容：

- 对每个 anchor 收集其控制的 local Gaussian 属性。
- 统计 Gaussian 数量、局部距离、scale、opacity、颜色/SH 相关 embedding 等特征。
- 输出 CSV、npy、pt 和 heatmap，用于判断不同 anchor 的局部高斯分布是否有明显差异。

输出目录：

```text
output/anchor_debug/expD_feature_stats/
```

关键输出：

```text
anchor_feature_stats_init.csv
anchor_feature_stats_heatmap_init.png
anchor_feature_local_gaussians_init.ply
anchor_feature_embeddings_init.npy
anchor_feature_embedding_names.txt
anchor_local_features_init.pt
README_expD.md
```

可视化含义：

- `anchor_feature_stats_heatmap_init.png`：每个 anchor 的 feature 统计热力图。
- `anchor_feature_local_gaussians_init.ply`：按 anchor-local feature 分组的人体高斯可视化。
- `anchor_feature_stats_init.csv`：每个 anchor 的数量、距离、opacity、scale 等统计表。

当前结论：这一步能证明每个 anchor 的局部 Gaussian 集合可以被稳定抽取，并能产生可统计的局部结构；但它本身还不是学习式 token，需要 ExpD 第二部分继续编码。

### 17.5 Exp D 第二部分：anchor-local Gaussian feature 到 token 的编码

目的：设计一个神经网络，把每个 anchor 绑定的 Gaussian 属性编码成固定维度 token。

实现内容：

- 实现 anchor token encoder 的前向结构。
- 支持 mean pooling token 和 attention pooling token 两种聚合方式。
- 当前主要完成结构验证和初始 token 可视化，网络尚未针对下游任务训练。

输出目录：

```text
output/anchor_debug/expD_token_encode/
```

关键输出：

```text
anchor_tokens_mean_init.npy
anchor_tokens_attention_init.npy
anchor_tokens_mean_pca_init.png
anchor_tokens_attention_pca_init.png
anchor_tokens_attention_cosine_init.png
anchor_attention_weights_init.pt
anchor_token_attention_stats_init.csv
anchor_token_names.txt
README_expD_token.md
```

可视化含义：

- `anchor_tokens_mean_pca_init.png`：mean pooling token 的 PCA 2D 投影，每个点是一个 anchor token。
- `anchor_tokens_attention_pca_init.png`：attention pooling token 的 PCA 2D 投影。
- `anchor_tokens_attention_cosine_init.png`：不同 anchor token 之间的 cosine similarity。
- 当前图只能说明未训练网络的初始 token 分布，不能单独证明 token 已学习到场景相关语义。

当前结论：网络结构和输出链路可用。后续若要证明 token 有效，需要在真实任务中训练，例如用 token 预测 anchor-scene contact、重建误差、cross-attention 权重或下游渲染改进。

### 17.6 Exp E：anchor 到场景 Gaussian 的最近邻/交互可视化

目的：在真实 HUGS 训练流程中，观察人体 anchor 与场景 Gaussian 的空间关系，并比较原 HUGS 初始化和 Human3R 初始化。

实现内容：

- 使用 normal 训练配置跑 3000 步，保留 `500/1000/1500/2000/2500/3000` checkpoint。
- 对每个记录步数和多个帧导出完整人体+场景渲染图。
- 在 3DGS rasterizer 渲染图上 overlay anchor、anchor 标签、anchor 到最近场景 Gaussian 的连线，以及被查询到的场景点颜色。
- attention 已改为距离最近邻诊断，不再使用 softmax cross-attention。
- 当前最终版本使用跨 anchor 全局归一化：把当前帧所有 anchor 的 KNN 场景距离放在一起归一化，距离越近越红，越远越蓝。

当前距离强度定义：

```text
strength = 1 - (distance - global_min_distance) / (global_max_distance - global_min_distance)
```

其中 `global_min_distance/global_max_distance` 来自当前帧所有 anchor-scene KNN pair。CSV 中的 `nearest_neighbor_strength` 是每个 anchor 最近场景点在这个全局尺度下的强度。

原 HUGS 初始化输出目录：

```text
output/anchor_debug/expE_real_hugs_normal_3000/lab/anchor_cross_attention/
```

Human3R 初始化输出目录：

```text
output/anchor_debug/expE_human3r_normal_3000/lab_fit_smpl/anchor_cross_attention/
```

每组主要输出：

```text
iter000500_frame0000_render.png
iter000500_frame0000_cross_attention_overlay.png
iter000500_frame0000_cross_attention.csv
iter000500_frame0000_cross_attention.pt
...
iter003000_frame0080_render.png
iter003000_frame0080_cross_attention_overlay.png
iter003000_frame0080_cross_attention.csv
iter003000_frame0080_cross_attention.pt
```

导出帧和步数：

```text
frames: 0, 20, 40, 60, 80
iters: 500, 1000, 1500, 2000, 2500, 3000
```

可视化含义：

- `*_render.png`：HUGS 完整人体+场景渲染结果。
- `*_cross_attention_overlay.png`：在渲染图上叠加 anchor 和附近场景点。
- 彩色场景点：被 anchor KNN 查询到的场景 Gaussian 投影点。红色表示该 anchor-scene 距离在所有 anchor 比较中更近，蓝色表示更远。
- anchor 圆圈和文字：anchor 在当前帧投影后的位置与全局距离强度。
- 连线：该 anchor 到最近场景 Gaussian 的投影连线。
- `*_cross_attention.csv`：每个 anchor 的最近场景距离、全局距离强度、控制人体 Gaussian 数量、top scene opacity 等。
- `*_cross_attention.pt`：保存原始 tensor，包括 `anchor_world`、`knn_idx`、`knn_dist`、`attention`、`scene_strength` 等。

当前验证结果：

```text
原 HUGS 初始化：
  output: output/anchor_debug/expE_real_hugs_normal_3000/lab/anchor_cross_attention
  PNG: 62
  CSV: 31
  PT: 31
  image size: 1276 x 717
  sample: iter003000_frame0040_cross_attention.csv
  nearest_neighbor_strength range: 0.1392 - 1.0000
  nearest_scene_distance range: 0.1590 - 4.5814

Human3R 初始化：
  output: output/anchor_debug/expE_human3r_normal_3000/lab_fit_smpl/anchor_cross_attention
  PNG: 62
  CSV: 31
  PT: 31
  image size: 512 x 288
  sample: iter003000_frame0040_cross_attention.csv
  nearest_neighbor_strength range: 0.0753 - 1.0000
  nearest_scene_distance range: 0.0043 - 0.7238
```

注意：目标导出组合是 5 帧 x 6 步，每组应至少有 60 张目标 PNG（render 和 overlay 各 30 张）。目录中额外的 2 张 PNG 来自早期保留的 iter0/调试导出，不影响当前 3000 步 normal 结果。

当前结论：ExpE 已经可以在真实 HUGS normal 训练输出上比较人体 anchor 和场景 Gaussian 的空间接近程度。原 HUGS 初始化和 Human3R 初始化两组均完成 3000 步 normal checkpoint 导出。跨 anchor 全局归一化后，不会再出现每个 anchor 的最近点都显示为红色的问题。

### 17.7 重要问题和修复记录

1. 模板空间错位问题：早期用外部 SMPL 模板会导致 anchor 与 HUGS 人体 Gaussian 不对应，例如 buttocks 投到脖子附近。已改为使用 HUGS 自身模板和 HUGS vitruvian template。
2. pose/transl 问题：ExpC/ExpE 导出 anchor 时必须使用优化后的 HUGS pose/transl，否则训练过程中 anchor 会和人体 Gaussian 逐渐偏离。当前导出已使用优化后参数。
3. 投影方向问题：修复了可视化中的 y 轴投影方向，使 anchor overlay 与 rasterizer 渲染图对齐。
4. attention 解释问题：早期 softmax cross-attention 会被密集地面点稀释，脚底附近点可能仍偏蓝。现在 ExpE 使用最近邻距离诊断，并采用跨 anchor 全局归一化。
5. normal 训练质量问题：快速诊断版 smoke/preset 会降低人体质量；用于最终可视化的 ExpE 结果使用 normal preset 的 3000 步 checkpoint。

### 17.8 后续建议

- ExpD token encoder 需要进入真实训练目标，不能只看未训练 token PCA。
- ExpE 当前是几何距离诊断，还不是可学习 cross-attention；后续可把距离、opacity、normal、anchor token、scene Gaussian feature 合并成可训练 attention。
- 若要证明方法有效，建议比较：anchor-scene contact 预测、脚底/手部接触一致性、不同初始化的接触稳定性、以及加入 anchor token 后的渲染或重建指标变化。

## 18. 后续维护规则

后续我们修改本项目时，请同步更新本文档：

- 新增脚本：更新“目录结构”“常用命令”“本地修改说明”。
- 改配置：更新“关键配置字段”和对应运行示例。
- 改输出目录或指标：更新“输出目录说明”和“评估已有模型”。
- 改数据格式：更新“数据准备”和“常见问题”。
- 修复重要 bug：在“本地修改说明”中记录行为差异。

## 19. Anchor-Attention Baseline 搭建记录（2026-05-19）

本轮在原版 HUGS 初始化和原版 `human_scene` 训练流程上，新增了一个最小可行的 anchor-attention baseline 骨架。当前不默认使用 Human3R 初始化；Human3R 或其他初始化方案后续只需要替换数据/初始化入口，不影响本模块的训练接入方式。

### 19.1 新增/修改文件

```text
hugs/models/anchor_attention.py
cfg_files/release/neuman/hugs_anchor_attention_baseline.yaml
hugs/cfg/config.py
hugs/trainer/gs_trainer.py
```

其中：

- `hugs/models/anchor_attention.py`：独立的 interaction 模块，包含 human anchor token encoder、scene local query、anchor-scene cross-attention、position/opacity correction head，以及 debug 导出函数。
- `cfg_files/release/neuman/hugs_anchor_attention_baseline.yaml`：基于原版 NeuMan/HUGS `human_scene` 配置复制出的 baseline 配置，默认开启 anchor/token/query/cross-attention dry-run，默认不开 correction。
- `hugs/cfg/config.py`：新增 `cfg.anchor_attention.*` 默认字段，默认全部关闭，保证原版 HUGS 可回退。
- `hugs/trainer/gs_trainer.py`：在原训练循环中增加最小接入点：初始化 anchors/bindings，forward interaction 模块，保存 debug，保存 `anchor_attention_*.pth`。

### 19.2 当前 baseline 的数据流

```text
HUGS original initialization
  -> HUGS human_gs.forward / scene_gs.forward
  -> semantic anchor binding on human Gaussians
  -> anchor world position from bound human Gaussian weighted centroid
  -> human Gaussian features -> human anchor tokens
  -> anchor KNN query over scene Gaussians
  -> scene Gaussian features -> scene tokens
  -> per-anchor cross-attention
  -> optional delta_mu / delta_opacity correction
  -> original render_human_scene and original HUGS losses
```

注意：当前 `anchor_world` 第一版使用绑定人体高斯的当前帧加权中心估计，没有侵入 HUGS LBS 内部。后续如果需要更严格的 anchor pose，可替换成 SMPL/LBS 显式变换接口。

### 19.3 关键配置

默认配置字段位于 `hugs/cfg/config.py`：

```yaml
anchor_attention:
  use_anchors: false
  anchor_vertices_path: ''
  count_per_anchor: 64
  top_m: 2
  lambda_lbs: 0.35
  sigma_scale: 1.8
  min_sigma: 0.035
  use_anchor_token_encoder: false
  use_scene_query: false
  use_cross_attention: false
  use_interaction_correction: false
  correct_opacity: false
  hidden_dim: 128
  human_pooling: attention
  scene_topk: 32
  scene_opacity_threshold: 0.01
  max_scene_candidates: 200000
  correction_start_iter: 3000
  gamma_mu: 0.02
  gamma_opacity: 0.05
  delta_loss_w: 0.001
  lr: 0.0001
  debug_interval: 1000
  debug_dir: ''
  ckpt: ''
```

推荐分阶段打开：

```text
Dry-run:   use_anchors=true, use_anchor_token_encoder=true, use_scene_query=true, use_cross_attention=true, use_interaction_correction=false
Position:  上述基础上 use_interaction_correction=true, correct_opacity=false
Pos+Occ:   上述基础上 correct_opacity=true
```

### 19.4 运行命令

原版 HUGS 初始化 + anchor-attention dry-run：

```bash
python main.py \
  --cfg_file cfg_files/release/neuman/hugs_anchor_attention_baseline.yaml \
  --cfg_id 0 \
  dataset.seq=lab \
  exp_name=anchor_attention_lab_dryrun \
  train.num_steps=3000 \
  train.anim_interval=-1
```

开启 position correction 的最小实验：

```bash
python main.py \
  --cfg_file cfg_files/release/neuman/hugs_anchor_attention_baseline.yaml \
  --cfg_id 0 \
  dataset.seq=lab \
  exp_name=anchor_attention_lab_poscorr \
  train.num_steps=6000 \
  train.anim_interval=-1 \
  anchor_attention.use_interaction_correction=true \
  anchor_attention.correct_opacity=false \
  anchor_attention.correction_start_iter=3000
```

开启 position + opacity correction：

```bash
python main.py \
  --cfg_file cfg_files/release/neuman/hugs_anchor_attention_baseline.yaml \
  --cfg_id 0 \
  dataset.seq=lab \
  exp_name=anchor_attention_lab_posocc \
  train.num_steps=6000 \
  train.anim_interval=-1 \
  anchor_attention.use_interaction_correction=true \
  anchor_attention.correct_opacity=true \
  anchor_attention.correction_start_iter=3000
```

如果没有 AMASS，也可以沿用本地无 AMASS 辅助脚本，并指定新的 cfg：

```bash
python scripts/run_neuman_human_scene_noamass.py \
  --seq lab \
  --cfg-file cfg_files/release/neuman/hugs_anchor_attention_baseline.yaml \
  --exp-name anchor_attention_lab_smoke \
  --quick \
  --quick-steps 100 \
  train.anim_interval=-1
```

### 19.5 输出文件

训练输出目录中会新增：

```text
<logdir>/anchor_attention_debug/
├── anchor_vertices.json
├── anchors.pt
├── anchors.csv
├── anchor_binding_summary_init.csv
├── anchor_attention_iter000000.pt
├── anchor_attention_iter000000.csv
├── anchor_attention_iter001000.pt
└── anchor_attention_iter001000.csv
```

`anchor_attention_iter*.pt` 保存 anchor world、human tokens、scene KNN、attention weights、context、delta 等 tensor。`anchor_attention_iter*.csv` 是轻量可读诊断表，包含：

```text
iter
anchor_id
anchor_name
nearest_scene_distance
attention_entropy
max_attention
attention_weighted_distance
```

checkpoint 目录中会额外保存：

```text
ckpt/anchor_attention_XXXXXX.pth
```

### 19.6 当前验证结果

已完成轻量验证：

```bash
python -m py_compile hugs/models/anchor_attention.py hugs/trainer/gs_trainer.py hugs/cfg/config.py
```

结果：通过。

还完成了不依赖 CUDA 数据集的 CPU 前向烟测：构造随机 human/scene Gaussian 字典和随机 anchor bindings，验证 `AnchorSceneAttentionBaseline` 可以输出 corrected human dict、`[A,K]` attention weights，并写出 `/tmp/hugs_anchor_attention_smoke/anchor_attention_iter000001.{pt,csv}`。

当前 shell 环境直接 import HUGS 训练包时缺少 `simple_knn` CUDA 扩展，因此完整训练 smoke 需要在 `conda activate hugs` 且 CUDA 扩展可用的环境中运行。

### 19.7 当前实现限制和替换点

- 初始化：当前依托原版 HUGS 初始化。后续 Human3R 或其他初始化方案只需保证 `human_gs` 和 `scene_gs` 正常 forward，anchor-attention 模块无需改动。
- anchor world：当前用绑定人体高斯当前帧加权中心估计。后续可替换为显式 SMPL/LBS anchor transform。
- token 融合：当前支持 `human_pooling=mean|attention`，scene token 是 KNN scene Gaussian feature MLP。后续可替换为更复杂 Gaussian feature、volume-aware feature、Transformer、多 slot pooling。
- scene query：当前是 opacity filter + torch KNN/topK，`max_scene_candidates` 控制候选数。后续可替换 KD-tree、voxel hash 或可见性过滤。
- correction：当前只支持 world-space `delta_mu` 和可选 opacity correction，默认关闭。后续可扩展 scale/rotation correction、geometry bias、gap regularization。

## 20. Exp0 原版 HUGS baseline 复现与统一评估（2026-05-19）

本节记录用于后续所有方法对比的原版 HUGS baseline 复现实验。目标是尽量保持与 release 配置 `cfg_files/release/neuman/hugs_human_scene.yaml` 一致，只在无 AMASS 环境下跳过 novel-pose animation 数据集，不改变训练初始化、loss、optimizer、densification 或渲染主路径。

### 20.1 原版配置核对

已对比 git 原始版本：

```bash
git show HEAD:cfg_files/release/neuman/hugs_human_scene.yaml
sed -n '1,220p' cfg_files/release/neuman/hugs_human_scene.yaml
```

关键 release 设置：

```text
mode: human_scene
dataset.name: neuman
dataset.seq: [lab, citron, seattle, bike, jogging, parkinglot]
train.num_steps: 14998
train.val_interval: 1000
train.save_ckpt_interval: 15000
human.name: hugs_trimlp
human.n_subdivision: 2
human.use_deformer: true
human.disable_posedirs: true
human.loss.lbs_w: 1000.0
human.loss.humansep_w: 1.0
human.densify_from_iter: 3000
human.densification_interval: 600
scene.opt_start_iter: -1
scene.densify_from_iter: 500
scene.densification_interval: 100
```

当前工作区新增的 `anchor_attention` 默认全部关闭，`scene` 里新增的 prior/depth/prune 字段默认权重或 interval 为 0，因此不会进入原版 Exp0 baseline。训练日志中也确认：

```text
anchor_attention.use_anchors: false
scene.anchor_loss_w: 0.0
scene.depth_prior_w: 0.0
scene.depth_prune_interval: 0
```

### 20.2 已完成的 baseline

已在 `gpu-l40-1` 后台顺序启动两组实验：

```bash
ssh gpu-l40-1 'cd /hdd/u202420081000003/ml-hugs && mkdir -p run_logs && nohup bash -lc '\''source /home/u202420081000003/anaconda3/etc/profile.d/conda.sh && conda activate hugs && cd /hdd/u202420081000003/ml-hugs && python scripts/run_neuman_human_scene_noamass.py --seq lab --cfg-file cfg_files/release/neuman/hugs_human_scene.yaml --exp-name exp0_hugs_original_6000_20260519 train.num_steps=6000 train.save_ckpt_interval=6000 train.val_interval=1000 train.save_progress_images=false human.canon_nframes=2 && python scripts/run_neuman_human_scene_noamass.py --seq lab --cfg-file cfg_files/release/neuman/hugs_human_scene.yaml --exp-name exp0_hugs_original_15000_20260519 train.save_progress_images=false human.canon_nframes=2'\'' > run_logs/exp0_hugs_original_6000_15000_20260519.log 2>&1 & echo $!'
```

说明：

- 6000-step 组只覆盖 `train.num_steps/save_ckpt_interval/val_interval`，用于快速中期 baseline。
- 15000-step 组不覆盖 `train.num_steps`，使用 release 配置默认 `14998`，即原版约 15000 步完整训练。
- 使用 `scripts/run_neuman_human_scene_noamass.py` 的非 quick 模式；该模式不跳过 `optimize_init`，只禁用 AMASS animation，训练流程仍是原版 HUGS。

日志：

```text
run_logs/exp0_hugs_original_6000_15000_20260519.log
```

训练输出与验证指标：

```text
6000-step:
  output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_6000_20260519/2026-05-19_23-10-00
  val hugs_psnr:       25.6634
  val hugs_ssim:       0.9012
  val hugs_lpips:      0.0843
  val hugs_human_psnr: 18.7363
  val hugs_human_ssim: 0.7432
  val hugs_human_lpips:0.1652

full-step / release default 14998:
  output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54
  val hugs_psnr:       25.9582
  val hugs_ssim:       0.9147
  val hugs_lpips:      0.0710
  val hugs_human_psnr: 18.8005
  val hugs_human_ssim: 0.7580
  val hugs_human_lpips:0.1532
```

统一评估结果：

```text
6000-step eval_unified_final.json:
  all_psnr:        25.9781
  all_ssim:        0.8945
  human_crop_psnr: 20.1687
  human_crop_ssim: 0.7642
  human_mask_psnr: 19.3641
  bg_psnr:         27.3815
  human_mae_255:   15.7290
  bg_mae_255:       6.1503

full-step eval_unified_final.json:
  all_psnr:        26.4259
  all_ssim:        0.9103
  human_crop_psnr: 20.1585
  human_crop_ssim: 0.7810
  human_mask_psnr: 19.0400
  bg_psnr:         28.2458
  human_mae_255:   15.6149
  bg_mae_255:       5.4488
```

结论：full-step 按原版 release 配置完成后，全图 val/all 指标均优于 6000-step，人体 crop SSIM 与 LPIPS 也提升；后续质量评估默认以 full-step 作为正式 Exp0 baseline，同时保留 6000-step 作为中期训练对照。

### 20.3 统一评估脚本

新增：

```text
scripts/evaluate_gaussian_outputs.py
```

用途：给后续原版 HUGS、anchor dry-run、correction、Human3R 初始化等所有实验提供统一 checkpoint 评估入口。它会从训练输出目录读取：

```text
config_train.yaml
ckpt/human_*.pth
ckpt/scene_*.pth
ckpt/anchor_attention_*.pth  # 如果存在
```

然后可执行：

```text
trainer.validate()
all split full-sequence metric evaluation
```

输出：

```text
<logdir>/eval_unified_<tag>.json
<logdir>/eval_all_frames_<tag>.json
<logdir>/eval_unified_<tag>.log
```

当前支持指标：

```text
validation split:
  hugs_psnr / hugs_ssim / hugs_lpips
  hugs_human_psnr / hugs_human_ssim / hugs_human_lpips

all split:
  all_psnr / all_ssim
  human_crop_psnr / human_crop_ssim
  human_mask_psnr
  bg_psnr
  human_mae_255
  bg_mae_255
```

使用示例：

```bash
python scripts/evaluate_gaussian_outputs.py \
  -o output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_6000_20260519/<timestamp> \
  --tag final \
  --keep-images
```

快速烟测：

```bash
python scripts/evaluate_gaussian_outputs.py \
  -o <logdir> \
  --tag smoke_eval \
  --max-frames 2 \
  --skip-val
```

已验证：

- `python -m py_compile scripts/evaluate_gaussian_outputs.py` 通过。
- 在 `gpu-l40-1` 上用已有 1-step 输出目录跑 `--max-frames 2 --skip-val` 通过，并写出 `eval_unified_smoke_eval.json`。

### 20.4 后续替换规则

旧的 2026-05-15 `neuman_lab_original_6000_vis` 只作为历史排障记录保留，不再作为正式 baseline。后续所有 anchor/correction 实验必须和本节记录的 baseline 使用同一评估脚本、同一数据 split 和同一指标口径。

## 21. 接触质量评估方案修订（2026-05-20）

### 21.1 为什么需要独立接触评估

当前 baseline 的主要监督仍然来自 HUGS 原始渲染损失。渲染损失可以评价图像重建质量，但不能直接证明人体-场景接触关系变好了。

同时，HUGS scene Gaussians 是面向渲染优化的 radiance representation，不是显式、干净、可物理解释的 scene surface。它可能存在漂浮点、离散噪声、局部 geometry 不闭合等问题。因此不能直接把 HUGS scene Gaussians 当作接触监督或唯一接触评估真值。

后续需要补充一个独立于 HUGS Gaussian 表示的 contact evaluation channel，用于评估 contact/correction/token fusion 实验是否真的改善人体-场景接触质量。

### 21.2 相关工作启发

调研 human-scene / human-object interaction 与 Gaussian human-object reconstruction 后，形成以下参考结论：

- PROX 类方法依赖可信 3D scene scan，并显式使用 contact / interpenetration constraints。
- RICH / BSTRO 类工作提供或学习 dense body-scene contact annotation，可进行顶点级 contact 评估。
- DECO / DAMON 类工作从 RGB 图像预测 SMPL/SMPL-X 顶点级 contact probability。
- POSA 在 body-centric 表示中预测每个 SMPL-X 顶点的 contact probability 和 semantic scene label。
- HOGS 类 Gaussian human-object 工作也会引入 pretrained contact predictor 或 contact/separation losses，而不是只依赖渲染指标。

因此，本项目后续不能只用 PSNR/SSIM/LPIPS 声称接触质量提升，必须至少补充 contact pseudo-label、independent scene proxy、局部接触指标中的一类或多类。

### 21.3 本项目采用的评估原则

```text
1. 不使用 HUGS scene Gaussians 作为唯一接触真值。
2. 接触评估和 HUGS 训练表示解耦。
3. 没有真实 contact GT 时，使用 pseudo-label + proxy metrics，并记录置信度。
4. 低置信 scene proxy 区域只做可视化，不强行解释 penetration/contact。
5. 后续 correction 实验必须同时报告渲染指标和接触 proxy 指标。
```

### 21.4 拟新增脚本

```text
scripts/build_contact_eval_proxy.py
scripts/evaluate_contact_quality.py
```

`build_contact_eval_proxy.py` 用于构建 eval-only scene proxy，可选输入：

```text
NeuMan / COLMAP 静态场景点云
DepthPro / DepthAnything monocular depth
多帧 depth fusion 后的静态场景点云
去除人体 mask 后的背景 depth / point cloud
未来可能获得的 scan / mesh
```

`evaluate_contact_quality.py` 用于读取 HUGS 输出、人体状态、contact pseudo-label 和 scene proxy，并输出逐帧接触指标。

### 21.5 Contact pseudo-label 来源

短期先做 feet-only：

```text
foot / toe / heel anchors
low vertical velocity
temporal persistence
near ground or near scene proxy
```

中期接入 full-body predictor：

```text
DECO / BSTRO / POSA-like pretrained contact predictor
SMPL/SMPL-X vertex contact probability
```

长期如果有更强数据，可使用真实 contact annotation、高质量 scene scan 或 multi-view depth fused surface。

### 21.6 指标清单

```text
Contact Distance:
  contact vertices / anchors 到 independent scene proxy 的距离。

Contact Recall @ tau:
  contact candidate 中距离 scene proxy 小于 tau 的比例，例如 2cm/5cm/10cm。

Floating Rate:
  contact candidate 离 scene proxy 过远的比例。

Penetration Proxy:
  如果可拟合局部平面或 TSDF，则统计进入 scene proxy 内部的比例和深度。
  低置信局部 geometry 不报告 penetration。

Contact Stability:
  接触距离、contact anchor、contact probability 在时间上的抖动。

Contact ROI Rendering:
  contact anchor / contact vertices 投影图像后的局部 crop PSNR / SSIM / LPIPS。
```

### 21.7 第一阶段落地目标

优先实现 feet-only / ground-proxy 版本：

```text
输入:
  Exp0 full baseline 输出目录
  原始 NeuMan lab 图像和 mask
  HUGS / SMPL 人体状态
  eval-only ground / scene proxy

输出:
  contact_eval_summary.json
  per_frame_contact_metrics.csv
  contact_visualization.ply
  contact_roi_images/
```

第一阶段只回答三个问题：

```text
1. 原版 HUGS 在脚底区域是否存在 floating / penetration proxy？
2. 接触相关帧的局部 render crop 是否是误差高发区域？
3. 后续 correction 是否能在不牺牲全图渲染的情况下改善这些 proxy？
```

训练路径和评估路径保持解耦：

```text
training path:
  HUGS human/scene Gaussians -> anchor query -> scene token -> correction -> render loss

evaluation path:
  HUGS output + independent scene proxy + contact pseudo-label -> contact metrics
```

## 22. 从最小 baseline 到可提升质量 pipeline 的路线图（2026-05-20）

### 22.1 当前能力和缺口

当前已经具备原版 HUGS Exp0 baseline、anchor-attention 最小工程骨架、统一渲染评估脚本，以及接触评估方案设计。当前还缺少：

```text
1. 独立可运行的 contact evaluation channel。
2. probe / attention 是否可信的诊断结果。
3. 初始化质量对 probe / attention 的影响对比。
4. 稳定、zero-init、强约束的 learnable correction 实验。
5. token fusion / scene query 的系统 ablation。
```

### 22.2 阶段路线

```text
Stage A: 建立接触评估通道
Stage B: 诊断当前 anchor probe / attention 是否可信
Stage C: 验证初始化质量对 probe / attention 的影响
Stage D: 开启 zero-init learnable correction，做小步可控训练
Stage E: token fusion / scene query ablation
Stage F: 引入更强初始化或 scene proxy，对比完整 pipeline
Stage G: 多序列和正式 ablation 表格
```

### 22.3 初始化影响验证

新增重点：初始化会直接影响当前 anchor-attention 的 probe 和 attention 计算。

```text
Risk 1: 背景 / scene Gaussians 几何差且杂乱。
  可能导致 anchor 查询到漂浮点、噪声点、非表面点，使 scene token 成为错误上下文。

Risk 2: 人体和场景初始相对关系不好。
  可能导致 anchor_world 偏移，top-k scene query 偏离真实接触区域，使 attention 学到错误关系。
```

因此，初始化改进后必须增加 dry-run 对比实验：

```text
Exp Init-0: 原版 HUGS 初始化 + 原版 scene Gaussians
Exp Init-1: 清理/重建后的 scene 初始化 + 原版 human 初始化
Exp Init-2: 原版 scene 初始化 + 改进 human-scene 对齐
Exp Init-3: 改进 scene 初始化 + 改进 human-scene 对齐
```

这些实验先不打开 correction，只比较 probe 和 attention：

```text
nearest_scene_distance
trimmed_scene_distance
attention_entropy
attention_weighted_distance
scene candidate opacity / scale distribution
probe temporal stability
contact proxy metrics
contact ROI render metrics
```

如果初始化变化显著改变 probe/attention 统计，则后续 correction 必须和初始化作为耦合因素一起评估。

### 22.4 Learnable correction 的进入条件

不再把 rule-based correction 作为必须成功的前置条件。rule-based 只作为 sanity check。真正的改进模块采用 zero-init learnable correction：

```text
human gaussian feature + anchor context + scene context token
-> gate_head
-> bounded delta_xyz_head
```

约束：

```text
delta head zero-init
gate bias 初始化为负值
bounded_delta = tanh(raw_delta) * max_delta
max_delta = 0.01 或 0.02
correction_start_iter = 3000
只先改 xyz，不动 opacity / scale / rotation
```

### 22.5 最终可提升 baseline 的判定标准

```text
1. 原版 HUGS full-step baseline 可复现。
2. contact evaluation channel 可运行。
3. probe / attention 诊断证明 scene context 有稳定信息。
4. 初始化影响已通过 Init-0/1/2/3 对比实验验证。
5. correction 训练稳定，不破坏全图渲染质量。
6. contact ROI 或 contact proxy 指标有稳定改善。
7. 至少一个序列 full-step 优于 Exp0，再扩展到多序列。
```

### 22.6 最近优先级

```text
Step 1: 实现 feet-only contact eval。
Step 2: 实现 anchor probe diagnostic，对 Exp0 full baseline 导出 probe/attention 可视化和统计。
Step 3: 等初始化方案改进后，做 Init-0 vs improved init 的 dry-run probe 对比。
Step 4: 在可信 probe 基础上开启 zero-init xyz correction。
```

### 22.11 Stage A-C 首版实现记录（2026-05-20）

已完成 Stage A-C 的首版工程实现，并在 Exp0 full baseline 上跑通小规模 smoke。

新增脚本：

```text
scripts/contact_eval_lib.py
scripts/build_contact_eval_proxy.py
scripts/evaluate_contact_quality.py
scripts/diagnose_anchor_probe_attention.py
scripts/compare_probe_diagnostics.py
```

功能对应关系：

```text
Stage A:
  build_contact_eval_proxy.py
    构建 eval-only scene proxy，当前支持 COLMAP points3D.txt 和可选 depth npy fusion。

  evaluate_contact_quality.py
    feet-only contact proxy 评估，输出 contact distance、contact recall、ROI PSNR/SSIM、anchor trajectory PLY、render/GT/absdiff crops。

Stage B:
  diagnose_anchor_probe_attention.py
    对 anchor scene query / attention 做诊断，输出 nearest/trimmed distance、attention entropy、attention weighted distance、scene opacity/scale 统计、overlay/render/PLY。

Stage C:
  compare_probe_diagnostics.py
    后续不同初始化方案完成后，可把多个 summary.json 汇总成对比 CSV。
```

Smoke 使用的 Exp0 full baseline：

```text
output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54
```

Smoke 输出：

```text
output/contact_eval_proxy/neuman_lab_exp0_full_smoke/
  scene_proxy_points.npz
  scene_proxy_points.ply
  scene_proxy_summary.json

output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54/contact_eval/smoke/
  contact_eval_summary.json
  per_frame_contact_metrics.csv
  contact_anchor_trajectory.ply
  scene_proxy_sample.ply
  overlays/*.png
  renders/*_gt.png
  renders/*_render.png
  contact_rois/*/*_gt.png
  contact_rois/*/*_render.png
  contact_rois/*/*_absdiff.png

output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54/anchor_probe_diagnostics/smoke/
  anchor_probe_summary.json
  anchor_probe_metrics.csv
  anchor_probe_scene_samples.ply
  overlays/*.png
  renders/*_gt.png
  renders/*_render.png
```

Smoke 观察：

```text
contact_eval smoke 使用 COLMAP sparse proxy，feet anchors 到 proxy 的 nearest distance 均值约 0.8787m，2/5/10cm contact recall 均为 0。
anchor_probe smoke 中 attention entropy 接近 log(16)，说明当前随机初始化 attention 基本接近均匀分布；这符合 dry-run 预期，不代表已经学到接触关系。
```

重要结论：当前 COLMAP sparse proxy 对接触评估过稀、尺度/覆盖不一定适合作为最终 contact surface，只能作为 Stage A 链路验证。后续应优先接入 depth fusion / ground proxy / 更干净 scene proxy，再做正式 contact 指标解释。

### 22.12 DECO 接触评估接入记录（2026-05-20）

已将 lab 序列的 DECO 输出接入 Stage A-C 的接触评估通道。DECO 输出位置：

```text
/hdd/u202420081000003/deco/demo_out_lab
```

读取结果确认：

```text
Preds/<frame>/pred.obj
Preds/<frame>/<frame>.png

pred.obj 是 6890 顶点 SMPL mesh。
灰色 [130, 130, 130] 表示非接触。
绿色 [0, 255, 0] 表示 DECO inference 中 cont >= 0.5 的接触顶点。
当前目录没有额外 npy / npz / json 概率文件，因此本次使用 OBJ 顶点颜色恢复二值 contact label。
```

代码落实：

```text
scripts/contact_eval_lib.py
  read_deco_pred_obj()
  deco_contact_obj_path()
  deco_contact_png_path()
  aggregate_deco_contact_to_anchors()

scripts/evaluate_contact_quality.py
  新增 --contact-source deco-anchor
  新增 --deco-dir
  新增 --deco-contact-threshold
  新增 --deco-top-anchors
  新增 --deco-anchor-verts
```

当前映射方式：

```text
1. 从 DECO pred.obj 读取 6890 个 SMPL 顶点的接触二值标签。
2. 使用 hugs.utils.anchor_utils 在原始 SMPL t-pose 上生成同一套 16 个语义 anchor 的 vertex id。
3. 对每个 anchor 的 64 个 SMPL 顶点求 mean(contact)，得到 deco_contact_prob。
4. deco_contact_prob >= threshold 的 anchor 被视为本帧 DECO contact anchor。
5. HUGS 侧仍使用当前训练结果中的 anchor_world 位置，计算投影 overlay、ROI render/GT/absdiff、到 scene proxy 的 nearest/median/trimmed distance。
```

注意：DECO 只负责回答“人体哪个区域可能接触”。它不直接提供场景几何，也不直接评价 HUGS scene Gaussian 是否真实接触。因此本方法比 feet-only 更合理，但最终接触质量仍依赖一个可信 scene proxy / depth surface / ground-object proxy。

Smoke 命令：

```bash
python scripts/evaluate_contact_quality.py \
  -o output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54 \
  --proxy-npz output/contact_eval_proxy/neuman_lab_exp0_full_smoke/scene_proxy_points.npz \
  --tag deco_contact_smoke_20260520 \
  --contact-source deco-anchor \
  --deco-dir /hdd/u202420081000003/deco/demo_out_lab \
  --max-frames 8 \
  --frame-stride 10 \
  --save-every 1 \
  --proxy-max-points 50000 \
  --deco-top-anchors 6
```

完整 103 帧评估命令：

```bash
python scripts/evaluate_contact_quality.py \
  -o output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54 \
  --proxy-npz output/contact_eval_proxy/neuman_lab_exp0_full_smoke/scene_proxy_points.npz \
  --tag deco_contact_full_20260520 \
  --contact-source deco-anchor \
  --deco-dir /hdd/u202420081000003/deco/demo_out_lab \
  --save-every 10 \
  --proxy-max-points 50000 \
  --deco-top-anchors 6
```

完整评估输出：

```text
output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54/contact_eval/deco_contact_full_20260520/
  contact_eval_summary.json
  per_frame_contact_metrics.csv
  contact_anchor_trajectory.ply
  scene_proxy_sample.ply
  overlays/*.png
  renders/*_gt.png
  renders/*_render.png
  deco_preds/*.png
  contact_rois/*/*_gt.png
  contact_rois/*/*_render.png
  contact_rois/*/*_absdiff.png
```

完整 103 帧结果摘要：

```text
num_rows: 1648
num_contact_rows: 642
missing_deco_frames: 0
DECO vertex count: 每帧 6890

DECO contact anchor 分布：
  left_sole: 103
  right_sole: 103
  left_toe: 103
  right_toe: 103
  left_heel: 103
  right_heel: 103
  left_fingers: 12
  left_palm: 9
  right_palm: 1
  right_fingers: 1
  back: 1

contact_deco_contact_prob mean: 0.7829
contact nearest proxy distance mean: 0.8281m
contact nearest proxy distance median: 0.7226m
contact recall @ 2cm: 0.0000
contact recall @ 5cm: 0.0016
contact recall @ 10cm: 0.0031
contact ROI PSNR mean: 26.5095
contact ROI SSIM mean: 0.8746
```

阶段性结论：

```text
1. DECO 接入链路可用，且语义结果合理：主要稳定预测脚底、脚趾、脚跟接触，少量帧预测手/背接触。
2. DECO-contact 比固定 feet-only 更适合作为人体接触区域选择器，后续建议默认采用 --contact-source deco-anchor。
3. 当前到 COLMAP sparse proxy 的接触距离仍很大，5cm/10cm recall 几乎为 0。这不应直接解释为 HUGS 接触完全失败，更可能说明当前 scene proxy 对接触表面过稀、过乱、覆盖不足。
4. 下一步应把 improved initialization / depth fusion / cleaned scene proxy 接进同一脚本，用同一 DECO contact anchor 集合做 A/B 对比。若 improved proxy 后 contact distance 明显下降，同时 ROI 渲染不下降，才可以作为接触质量提升证据。
```

### 22.13 Overlay 投影 bug 修复记录（2026-05-21）

发现问题：`contact_eval/*/overlays/*.png` 中彩色 anchor 点明显落在人体头顶/上半身附近，而 DECO 预测的是脚底、脚趾、脚跟接触区域。

排查结论：

```text
不是 anchor 绑定主问题，而是 scripts/contact_eval_lib.py::project_points() 的 3D->2D 投影 y 轴方向错了。
```

诊断方法：同一帧 frame 00000，对比旧 `full_proj_transform` NDC 投影和 HUGS 内部使用的 camera intrinsics 投影：

```text
image height: 717
left_toe old_xy: [911.06, 43.14]
left_toe K_xy:   [911.06, 673.86]
old_y_flip:      673.86

left_sole old_xy: [909.21, 98.30]
left_sole K_xy:   [909.21, 618.70]
old_y_flip:       618.70
```

这说明旧 overlay 的 y 坐标正好被上下翻转，所以脚部 anchor 被画到了头顶附近。

修复内容：

```text
scripts/contact_eval_lib.py
  project_points()
    现在优先使用 data['world_view_transform'] + data['cam_intrinsics']：
      cam = [xyz, 1] @ world_view_transform
      u = fx * cam_x / cam_z + cx
      v = fy * cam_y / cam_z + cy

    这与 hugs/trainer/gs_trainer.py 中 depth prior / pruning 的投影写法保持一致。
```

影响范围：

```text
受影响：
  overlays/*.png 中的彩色点位置
  contact_rois/* 的 crop 位置
  ROI PSNR / ROI SSIM
  使用 project_points() 的 probe/attention overlay 可视化

不受影响或基本不受影响：
  DECO contact anchor 选择
  anchor 3D world position
  anchor 到 scene proxy 的 3D nearest/median/trimmed distance
  contact recall @ distance threshold
```

修复后 smoke 输出：

```text
output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54/contact_eval/deco_contact_projection_fix_smoke_20260521/
```

修复后完整 103 帧输出：

```text
output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54/contact_eval/deco_contact_projection_fix_full_20260521/
```

修复后 frame 00000 的脚部 anchor 投影示例：

```text
left_sole:  x=909.2, y=618.7
right_sole: x=829.4, y=607.2
left_toe:   x=911.1, y=673.9
right_toe:  x=808.7, y=645.4
left_heel:  x=909.8, y=616.9
right_heel: x=842.3, y=599.7
```

修复后完整 103 帧摘要：

```text
num_rows: 1648
num_contact_rows: 642
missing_deco_frames: 0

contact nearest proxy distance mean: 0.8281m
contact nearest proxy distance median: 0.7226m
contact recall @ 2cm: 0.0000
contact recall @ 5cm: 0.0016
contact recall @ 10cm: 0.0031

contact ROI PSNR mean: 17.8720
contact ROI SSIM mean: 0.7352
```

重要记录：旧目录 `deco_contact_full_20260520` 的 overlay 和 ROI 指标作废；后续引用 DECO-contact 评估时，应使用 `deco_contact_projection_fix_full_20260521` 或更新后的重新评估结果。

### 22.14 ExpE-style anchor overlay 复现与 contact overlay 修正（2026-05-21）

根据复查，`deco_contact_projection_fix_*` 中点位仍然和 ExpE 结果有差异，不是新的投影方向问题，而是 anchor 位置定义不一致。

已新增对比脚本：

```text
scripts/export_anchor_overlay_compare.py
```

导出命令示例：

```bash
python scripts/export_anchor_overlay_compare.py \
  -o output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54 \
  --frames 0,10,20,30 \
  --render-pose dataset \
  --semantic-pose dataset
```

输出：

```text
output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54/anchor_overlay_compare/expe_compare_dataset_render_dataset_semantic/
  overlays/*.png
  anchor_overlay_compare.csv
```

图中含义：

```text
green filled point: ExpE-style semantic anchor center
magenta circle: binding-center anchor currently used by older contact overlay
white line: semantic center -> binding center offset
```

frame 00000 对比：

```text
left_sole  semantic [900.7, 667.0], binding [909.2, 618.7], delta [ 8.5, -48.3]
right_sole semantic [818.2, 644.0], binding [829.4, 607.2], delta [11.3, -36.8]
left_toe   semantic [909.1, 667.6], binding [911.1, 673.9], delta [ 2.0,   6.3]
right_toe  semantic [805.1, 644.0], binding [808.7, 645.4], delta [ 3.6,   1.4]
left_heel  semantic [914.8, 635.8], binding [909.8, 616.9], delta [-5.0, -18.9]
right_heel semantic [847.7, 617.6], binding [842.3, 599.7], delta [-5.4, -17.9]
```

解释：toe 区域紧凑，所以 binding center 与 semantic center 接近；sole/heel 的绑定高斯数量很多，且包含鞋面/脚踝附近高斯，binding center 被拉离真正脚底/脚跟语义位置。因此旧 contact overlay 中 sole/heel 会轻微偏上。

已修改：

```text
scripts/contact_eval_lib.py
  新增 posed_semantic_anchor_world()

scripts/evaluate_contact_quality.py
  新增 --overlay-anchor-position semantic|binding
  默认 semantic
```

现在 contact overlay / ROI crop 默认使用 ExpE-style semantic anchor center；CSV 中仍保留 binding center 的 `anchor_x/y/z`，并新增 overlay anchor 的 `overlay_anchor_x/y/z`。

新的 smoke 输出：

```text
output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54/contact_eval/deco_contact_semantic_overlay_smoke_20260521/
```

frame 00000 semantic overlay 坐标：

```text
left_sole  [900.7, 667.0]
right_sole [818.2, 644.0]
left_toe   [909.1, 667.6]
right_toe  [805.1, 644.0]
left_heel  [914.8, 635.8]
right_heel [847.7, 617.6]
```

额外发现：当前 `exp0_hugs_original_15000` 的 `human_final.pth` 中没有保存 `global_orient/body_pose/transl` 这类优化 pose 参数；因此 optimized render 分支会回退到 dataset pose。这个 original baseline 和旧 ExpE normal 训练输出不是完全同一个 pose 设置来源。

### 22.15 Render-loss-only anchor-attention correction 完整流程实验（2026-05-21）

目标：在不引入显式接触监督的前提下，先验证一个完整可训练 pipeline：

```text
HUGS original training / optimization
-> anchor token encoding
-> scene Gaussian token query and aggregation
-> cross attention
-> human Gaussian parameter correction
-> render loss backprop
```

重要原则：这里的 attention 不作为接触质量估计器，也不作为独立接触监督；它只作为人体 anchor 与场景高斯之间的上下文聚合模块，后续通过 correction 网络隐式影响人体高斯参数，并由图像渲染损失优化。

已实现代码：

```text
hugs/models/anchor_attention.py
  AnchorSceneAttentionBaseline
    - anchor token encoder
    - scene KNN/query token encoder
    - cross-attention context aggregation
    - delta_mu correction head
    - optional delta_opacity correction head
    - module_start_iter: correction 阶段前完全跳过 attention 计算
    - correction_start_iter: correction 生效起点
    - correction_warmup_iters: correction gamma 线性 warmup
    - zero_init_delta: correction head 最后一层零初始化，保证刚开启时接近 HUGS 原行为

hugs/cfg/config.py
  cfg.anchor_attention.module_start_iter
  cfg.anchor_attention.correction_start_iter
  cfg.anchor_attention.correction_warmup_iters
  cfg.anchor_attention.zero_init_delta

hugs/trainer/gs_trainer.py
  maybe_apply_anchor_attention()
  anchor_attention optimizer step
  anchor_attention debug / checkpoint save
```

本轮配置：

```text
cfg_files/release/neuman/hugs_anchor_attention_correction_smoke.yaml
cfg_files/release/neuman/hugs_anchor_attention_correction_15000.yaml
```

关键策略：

```text
前 12000 步：保持原 HUGS 训练流程，不执行 attention/correction
后 3000 步：打开 anchor-attention correction
总训练步数：约 15000 步，与原版 HUGS baseline 对齐
correction head: zero-init
correction warmup: 1000 steps
gamma_mu: 0.005
delta_loss_w: 0.001
correct_opacity: false
anim_interval: -1
```

`anim_interval=-1` 是因为当前环境缺少默认 AMASS/SFU 动画文件；该设置只跳过动画导出，不改变训练集、验证集、HUGS 优化项或渲染质量评估。

先运行 smoke 验证：

```bash
python main.py --cfg_file cfg_files/release/neuman/hugs_anchor_attention_correction_smoke.yaml
```

smoke 输出：

```text
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_smoke_20260521/2026-05-21_02-36-51/
```

smoke 结果：

```text
final HUGS_PSNR:       22.9653
final HUGS_SSIM:        0.8276
final HUGS_LPIPS:       0.1755
final HUMAN_PSNR:      18.6924
final HUMAN_SSIM:       0.7132
final HUMAN_LPIPS:      0.2144
```

smoke 检查结论：

```text
1. anchor-attention 模块可正常初始化。
2. correction 开启后无 NaN / shape error / optimizer error。
3. l_anchor_delta 从 0 逐步增长，说明 zero-init + warmup 后 correction 路径参与了优化。
4. 输出了 train/val/render_all/canonical 视频和图像。
5. anchor_attention_debug 在 1200 和 1600 步正常保存 csv/pt。
```

smoke 关键可视化：

```text
train/000000.png
train/001000.png
val/full_final_*.png
val/human_final_*.png
render_all/*.png
render_all_neuman_lab_final.mp4
canon_neuman_lab_final_a_pose.mp4
canon_neuman_lab_final_da_pose.mp4
anchor_attention_debug/anchor_attention_iter001200.csv
anchor_attention_debug/anchor_attention_iter001600.csv
```

完整实验命令：

```bash
python main.py --cfg_file cfg_files/release/neuman/hugs_anchor_attention_correction_15000.yaml
```

完整实验输出目录：

```text
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_correction_15000_20260521/2026-05-21_02-45-33/
```

完整实验已完成。

```text
final HUGS_PSNR:       26.0527
final HUGS_SSIM:        0.9149
final HUGS_LPIPS:       0.0705
final HUMAN_PSNR:      18.9251
final HUMAN_SSIM:       0.7609
final HUMAN_LPIPS:      0.1491
```

与 Exp0 原版 HUGS 15000 baseline 对比：

```text
Exp0 HUGS original 15000:
  HUGS_PSNR:       25.9582
  HUGS_SSIM:        0.9147
  HUGS_LPIPS:       0.0710
  HUMAN_PSNR:      18.8005
  HUMAN_SSIM:       0.7580
  HUMAN_LPIPS:      0.1532

Anchor-attention correction 15000:
  HUGS_PSNR:       26.0527   (+0.0945)
  HUGS_SSIM:        0.9149   (+0.0002)
  HUGS_LPIPS:       0.0705   (-0.0005, better)
  HUMAN_PSNR:      18.9251   (+0.1246)
  HUMAN_SSIM:       0.7609   (+0.0029)
  HUMAN_LPIPS:      0.1491   (-0.0041, better)
```

关键中间节点：

```text
012000, correction 刚开启:
  HUGS_PSNR: 25.9258
  HUGS_SSIM: 0.9127
  HUGS_LPIPS: 0.0719
  HUMAN_PSNR: 18.7633
  HUMAN_SSIM: 0.7488
  HUMAN_LPIPS: 0.1537

014000, correction 已进入稳定阶段:
  HUGS_PSNR: 26.0629
  HUGS_SSIM: 0.9135
  HUGS_LPIPS: 0.0724
  HUMAN_PSNR: 18.9984
  HUMAN_SSIM: 0.7631
  HUMAN_LPIPS: 0.1502
```

完整实验关键输出：

```text
results:
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_correction_15000_20260521/2026-05-21_02-45-33/results_train.json

train progress:
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_correction_15000_20260521/2026-05-21_02-45-33/train/*.png
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_correction_15000_20260521/2026-05-21_02-45-33/train_neuman_lab.mp4

validation images:
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_correction_15000_20260521/2026-05-21_02-45-33/val/full_final_*.png
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_correction_15000_20260521/2026-05-21_02-45-33/val/human_final_*.png

full sequence render:
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_correction_15000_20260521/2026-05-21_02-45-33/render_all/*.png
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_correction_15000_20260521/2026-05-21_02-45-33/render_all_neuman_lab_final.mp4

canonical videos:
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_correction_15000_20260521/2026-05-21_02-45-33/canon_neuman_lab_final_a_pose.mp4
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_correction_15000_20260521/2026-05-21_02-45-33/canon_neuman_lab_final_da_pose.mp4

anchor attention debug:
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_correction_15000_20260521/2026-05-21_02-45-33/anchor_attention_debug/anchor_attention_iter012000.csv
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_correction_15000_20260521/2026-05-21_02-45-33/anchor_attention_debug/anchor_attention_iter013000.csv
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_correction_15000_20260521/2026-05-21_02-45-33/anchor_attention_debug/anchor_attention_iter014000.csv
```

实验结论：

```text
1. 该 render-loss-only anchor-attention correction pipeline 已经可以完整训练到 15000 步。
2. 前 12000 步行为接近原版 HUGS；12000 步后 attention/correction 路径打开。
3. 在 lab 验证集上，final 渲染指标相对 Exp0 原版 HUGS 有小幅但一致提升，尤其 human crop 的 LPIPS/PSNR/SSIM 均变好。
4. 当前提升幅度不大，但足以说明“token encoding + attention + correction + render loss”的完整优化链路是可行的。
5. 这不是接触质量指标，也不能证明接触一定变好；它只说明该模块可以在不破坏 HUGS 主流程的情况下通过图像损失获得收益。
6. 性能瓶颈明显在 correction 阶段：每步需要在约 2.1M scene Gaussians 上做场景候选查询/聚合，后续需要考虑候选缓存、局部 scene subset 或更强采样策略。
7. 后续替换初始化方案时，应保持该配置作为对照，只替换初始化输入并复用同一套 12000/15000 训练与评估流程。
```

### 22.16 Bike 序列原版 HUGS 与 anchor-attention correction 对照（2026-05-21）

为检查 `lab` 上的小幅提升是否能迁移到不同场景，按相同策略在 NeuMan `bike` 序列上顺序跑了两组 15000-step 实验：

```text
Original HUGS config:
cfg_files/release/neuman/hugs_human_scene_bike_original_15000.yaml

Anchor-attention correction config:
cfg_files/release/neuman/hugs_anchor_attention_correction_15000_bike.yaml
```

原版 HUGS 输出：

```text
output/human_scene/neuman/bike/hugs_trimlp/exp0_hugs_original_15000_bike_20260521/2026-05-21_12-32-06/
```

anchor-attention correction 输出：

```text
output/human_scene/neuman/bike/hugs_trimlp/anchor_attention_xyz_correction_15000_bike_20260521/2026-05-21_13-17-51/
```

关键配置保持一致：

```text
train.num_steps: 14998
train.val_interval: 1000
train.save_progress_images: true
train.progress_save_interval: 1000
train.anim_interval: -1
```

attention/correction 配置：

```text
module_start_iter: 12000
correction_start_iter: 12000
correction_warmup_iters: 1000
zero_init_delta: true
gamma_mu: 0.005
delta_loss_w: 0.001
```

6000-step 对照：

```text
Original HUGS 6000:
  HUGS_PSNR:       24.6140
  HUGS_SSIM:        0.7919
  HUGS_LPIPS:       0.1555
  HUMAN_PSNR:      19.6295
  HUMAN_SSIM:       0.6551
  HUMAN_LPIPS:      0.1933

Anchor-attention run 6000:
  HUGS_PSNR:       24.6360
  HUGS_SSIM:        0.7919
  HUGS_LPIPS:       0.1546
  HUMAN_PSNR:      19.6729
  HUMAN_SSIM:       0.6491
  HUMAN_LPIPS:      0.1907
```

由于 attention/correction 在 12000 步后才开启，6000 步两者接近，符合预期。

anchor-attention correction 关键节点：

```text
012000, correction 刚开启:
  HUGS_PSNR:       25.4867
  HUGS_SSIM:        0.8326
  HUGS_LPIPS:       0.1143
  HUMAN_PSNR:      19.7824
  HUMAN_SSIM:       0.6648
  HUMAN_LPIPS:      0.1601

014000:
  HUGS_PSNR:       25.6300
  HUGS_SSIM:        0.8388
  HUGS_LPIPS:       0.1072
  HUMAN_PSNR:      19.7889
  HUMAN_SSIM:       0.6719
  HUMAN_LPIPS:      0.1542
```

final 对照：

```text
Original HUGS final:
  HUGS_PSNR:       25.6923
  HUGS_SSIM:        0.8419
  HUGS_LPIPS:       0.1027
  HUMAN_PSNR:      19.7592
  HUMAN_SSIM:       0.6765
  HUMAN_LPIPS:      0.1520

Anchor-attention correction final:
  HUGS_PSNR:       25.7943   (+0.1020)
  HUGS_SSIM:        0.8411   (-0.0008)
  HUGS_LPIPS:       0.1031   (+0.0004, worse)
  HUMAN_PSNR:      20.0542   (+0.2950)
  HUMAN_SSIM:       0.6800   (+0.0035)
  HUMAN_LPIPS:      0.1483   (-0.0037, better)
```

完整输出文件：

```text
Original HUGS:
output/human_scene/neuman/bike/hugs_trimlp/exp0_hugs_original_15000_bike_20260521/2026-05-21_12-32-06/results_train.json
output/human_scene/neuman/bike/hugs_trimlp/exp0_hugs_original_15000_bike_20260521/2026-05-21_12-32-06/train_neuman_bike.mp4
output/human_scene/neuman/bike/hugs_trimlp/exp0_hugs_original_15000_bike_20260521/2026-05-21_12-32-06/render_all_neuman_bike_final.mp4
output/human_scene/neuman/bike/hugs_trimlp/exp0_hugs_original_15000_bike_20260521/2026-05-21_12-32-06/canon_neuman_bike_final_a_pose.mp4
output/human_scene/neuman/bike/hugs_trimlp/exp0_hugs_original_15000_bike_20260521/2026-05-21_12-32-06/canon_neuman_bike_final_da_pose.mp4

Anchor-attention correction:
output/human_scene/neuman/bike/hugs_trimlp/anchor_attention_xyz_correction_15000_bike_20260521/2026-05-21_13-17-51/results_train.json
output/human_scene/neuman/bike/hugs_trimlp/anchor_attention_xyz_correction_15000_bike_20260521/2026-05-21_13-17-51/train_neuman_bike.mp4
output/human_scene/neuman/bike/hugs_trimlp/anchor_attention_xyz_correction_15000_bike_20260521/2026-05-21_13-17-51/render_all_neuman_bike_final.mp4
output/human_scene/neuman/bike/hugs_trimlp/anchor_attention_xyz_correction_15000_bike_20260521/2026-05-21_13-17-51/canon_neuman_bike_final_a_pose.mp4
output/human_scene/neuman/bike/hugs_trimlp/anchor_attention_xyz_correction_15000_bike_20260521/2026-05-21_13-17-51/canon_neuman_bike_final_da_pose.mp4
output/human_scene/neuman/bike/hugs_trimlp/anchor_attention_xyz_correction_15000_bike_20260521/2026-05-21_13-17-51/ckpt/anchor_attention_final.pth
```

运行观察：

```text
1. 该方法在 bike 上也能完整跑到 final，并保存 human / scene / anchor_attention final checkpoint。
2. 12000 步后 l_anchor_delta 从 0 开始，warmup 期间逐步增长，后段大多在 0.05-0.18 左右波动，未出现 NaN 或发散。
3. bike 上 final 整体 PSNR 和 human crop 指标变好，但整体 SSIM 和 LPIPS 基本持平或轻微变差。
4. 这说明当前方法的收益不是纯 lab 特例，但提升仍然小，不能只凭单次 bike 结果下强结论。
5. correction 阶段耗时明显增加：12000 前约 4.7-5.1 it/s，12000 后约 1.6-2.0 it/s；后续需要优化 scene candidate 查询或缓存。
```

### 22.18 Correction start time 消融：12000 / 9000 / 6000（2026-05-21）

本节记录一次针对 anchor-attention correction 介入时机的消融。目标是回答：当前网络 correction 是否应该只在 HUGS 基本收敛后做后段微调，还是可以更早参与优化。

实验设置：

```text
共同设置：dataset=lab, total schedule=15000, 原版 HUGS 初始化与训练流程不变。
只改变 anchor_attention.module_start_iter / correction_start_iter：
A. start12000：12000 -> 15000，已有主实验配置。
B. start9000：9000 -> 15000，本次完整跑完。
C. start6000：6000 -> 15000，本次跑到约 11200 后根据劣化趋势手动停止。

保持相同的 correction 超参：
correction_warmup_iters=1000
gamma_mu=0.005
delta_loss_w=0.001
zero_init_delta=True
```

配置与输出目录：

```text
A start12000 config:
cfg_files/release/neuman/hugs_anchor_attention_correction_15000.yaml
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_correction_15000_20260521/2026-05-21_02-45-33/

B start9000 config:
cfg_files/release/neuman/hugs_anchor_attention_correction_start9000_15000.yaml
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_correction_start9000_15000_20260521/2026-05-21_17-13-07/

C start6000 config:
cfg_files/release/neuman/hugs_anchor_attention_correction_start6000_15000.yaml
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_correction_start6000_15000_20260521/2026-05-21_18-37-06/
```

完整 final 对照：

| 实验 | correction start | 是否完整跑完 | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUMAN_PSNR | HUMAN_SSIM | HUMAN_LPIPS |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 原版 HUGS | none | 是 | 25.9582 | 0.9147 | 0.0710 | 18.8005 | 0.7580 | 0.1532 |
| A xyz-only anchor-attention | 12000 | 是 | 26.0527 | 0.9149 | 0.0705 | 18.9251 | 0.7609 | 0.1491 |
| B xyz-only anchor-attention | 9000 | 是 | 26.0416 | 0.9132 | 0.0721 | 18.9643 | 0.7576 | 0.1512 |
| C xyz-only anchor-attention | 6000 | 否，约 11200 停止 | - | - | - | - | - | - |

关键中间验证点：

| 实验 | iter | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUMAN_PSNR | HUMAN_SSIM | HUMAN_LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| A start12000 | 12000 | 25.9258 | 0.9127 | 0.0719 | 18.7633 | 0.7488 | 0.1537 |
| A start12000 | 14000 | 26.0629 | 0.9135 | 0.0724 | 18.9984 | 0.7631 | 0.1502 |
| B start9000 | 9000 | 26.0062 | 0.9082 | 0.0767 | 19.0329 | 0.7488 | 0.1558 |
| B start9000 | 12000 | 26.0707 | 0.9124 | 0.0722 | 19.0377 | 0.7510 | 0.1488 |
| B start9000 | 14000 | 26.2107 | 0.9135 | 0.0728 | 19.2379 | 0.7656 | 0.1505 |
| C start6000 | 6000 | 25.7734 | 0.9022 | 0.0842 | 18.9807 | 0.7451 | 0.1643 |
| C start6000 | 8000 | 26.0296 | 0.9111 | 0.0758 | 19.0639 | 0.7614 | 0.1527 |
| C start6000 | 9000 | 25.0503 | 0.9027 | 0.0835 | 17.5841 | 0.6917 | 0.2126 |
| C start6000 | 10000 | 26.0590 | 0.9124 | 0.0737 | 18.9955 | 0.7591 | 0.1536 |
| C start6000 | 11000 | 26.0409 | 0.9136 | 0.0730 | 18.9277 | 0.7602 | 0.1524 |

运行观察：

```text
1. start12000 仍是当前最稳的默认选择：final 的 full image 与 human crop 指标都优于原版 HUGS，且 correction 只在后 3000 步介入，扰动较小。
2. start9000 可以完整跑完，中途 14000 的 PSNR/HUMAN_PSNR 较高，但 final 回落；最终 full SSIM、LPIPS、human SSIM、human LPIPS 都不如 start12000，说明提前到 9000 并没有形成稳定收益。
3. start6000 在 8000 时看起来还正常，但 9000 验证出现明显劣化：HUMAN_PSNR 从 19.0639 掉到 17.5841，HUMAN_LPIPS 从 0.1527 变成 0.2126。虽然 10000/11000 有恢复，但训练中 l_anchor_delta 后续多次接近或超过 1.0，说明过早 correction 会造成较大不稳定。
4. start6000 没有 final 结果：本次按用户指令在约 11200 停止，因此没有 results_train.json、final ckpt 或 final video。该结果只作为“早介入不稳定”的失败/负向消融证据。
5. 当前建议：保留 start12000 作为默认 baseline；如果后续要探索更早介入，应先加强 correction 约束，例如更小 gamma_mu、更长 warmup、更小 delta_loss_w 或分阶段冻结/解冻，而不是直接从 6000 开始。
```

结论：

```text
本次消融说明，当前 anchor-attention correction 网络不是越早介入越好。
在没有额外接触监督、几何初始化仍不理想的情况下，过早让 correction 参与会放大人体/场景未收敛阶段的噪声，导致人体区域指标显著波动。
因此当前可行 baseline 应采用“原版 HUGS 先优化到较稳定状态，再在最后 3000 步做小幅 correction”的策略，即 start12000。
```

### 22.19 XYZ-only attention 延长训练消融：从 15000 checkpoint 继续 +3000（2026-05-21）

本节记录一次针对“后 3000 步 xyz-only attention/correction 是否足够”的验证。用户的问题是：如果保持 HUGS 初始优化阶段不变，继续增加 attention/correction 训练步数，指标是否还能提升；以及是否必须从头训练。

实现方式：

```text
本次不是严格意义上的 HUGS resume。
当前代码可以加载 human / scene / anchor_attention 的 final 权重，但训练循环的 global iteration 会重新从 0 开始，
optimizer state 与原训练日程也没有完整连续继承。因此这次实验应理解为：

从当前最好的 start12000 final checkpoint 出发，
冻结/弱化常规高斯学习率，关闭 densification，
让 anchor-attention correction 从 local step 0 继续以小学习率微调 3000 步。
```

配置与输出目录：

```text
config:
cfg_files/release/neuman/hugs_anchor_attention_extend_from15000_plus3000.yaml

source checkpoint:
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_correction_15000_20260521/2026-05-21_02-45-33/ckpt/

output:
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_extend_from15000_plus3000_20260521/2026-05-21_19-49-01/
```

主要设置：

```text
train.num_steps=2998
anchor_attention.module_start_iter=0
anchor_attention.correction_start_iter=0
anchor_attention.correction_warmup_iters=0
anchor_attention.lr=5e-5
anchor_attention.gamma_mu=0.003
anchor_attention.delta_loss_w=0.002
human.densify_until_iter=0
scene.densify_until_iter=0

human / scene 的 position、opacity、scaling、rotation、feature 等学习率均做了明显下调，
目的是避免继续训练阶段破坏已经收敛的 HUGS 表示。
```

指标对比：

| 实验 | 阶段 | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUMAN_PSNR | HUMAN_SSIM | HUMAN_LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| start12000 baseline | final 15000 | 26.0527 | 0.9149 | 0.0705 | 18.9251 | 0.7609 | 0.1491 |
| extend + xyz-only attention | +1000 | 26.3764 | 0.9203 | 0.0652 | 19.2497 | 0.7711 | 0.1421 |
| extend + xyz-only attention | +2000 | 26.3918 | 0.9208 | 0.0645 | 19.2469 | 0.7705 | 0.1413 |
| extend + xyz-only attention | +3000 final | 26.4029 | 0.9210 | 0.0640 | 19.2466 | 0.7701 | 0.1408 |

可视化与结果文件：

```text
results_train.json
train_neuman_lab.mp4
render_all_neuman_lab_final.mp4
canon_neuman_lab_final_a_pose.mp4
canon_neuman_lab_final_da_pose.mp4
ckpt/human_final.pth
ckpt/scene_final.pth
ckpt/anchor_attention_final.pth
```

观察：

```text
1. 继续 +3000 后，full image 与 human crop 的 LPIPS/SSIM 都继续改善，PSNR 也明显高于原 start12000 final。
2. +1000 已经带来大部分收益，+2000/+3000 仍有小幅提升，说明原来的后 3000 步 xyz-only attention/correction 未必完全饱和。
3. HUMAN_PSNR 在 +1000 后基本持平略降，但 HUMAN_LPIPS 持续下降，说明更长微调主要改善感知相似度和结构一致性，而不是单纯像素误差。
4. 本次结果只证明渲染指标层面继续微调有收益，不能直接证明接触质量改善；接触质量仍需要后续独立指标或 DECO/几何辅助评估。
5. 由于这不是严格 resume，若要作为正式论文/报告实验，建议后续补充两种更干净的流程：
   A. 在 12000 保存 checkpoint，然后原地继续训练到 18000；
   B. 给 trainer 增加 global_step 与 optimizer state 的完整 resume 支持。
```

结论：

```text
当前证据支持“xyz-only attention/correction 网络在 3000 步内可能没有完全收敛”。
在不从头训练的情况下，可以从已有 final checkpoint 继续 fine-tune，作为快速消融是可行的；
但严格比较不同训练步数时，应该实现真正 resume 或从 12000 checkpoint 接续。

当前建议：
保留 start12000 作为稳定介入点；
将 xyz-only attention/correction 总训练长度从 3000 扩展到 6000 作为下一版强 baseline 候选，
同时记录 +1000/+2000/+3000 checkpoint，用验证集指标选择最优停止点。
```

#### Lab 汇总表：HUGS 与 xyz-only attention 延长训练对照

下表单独汇总 lab 数据集上的关键结果。PSNR/SSIM 越高越好，LPIPS 越低越好；每一列最优值用粗体标出。

| 实验设置 | HUGS_PSNR ↑ | HUGS_SSIM ↑ | HUGS_LPIPS ↓ | HUMAN_PSNR ↑ | HUMAN_SSIM ↑ | HUMAN_LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|
| 12000 HUGS | 25.9258 | 0.9127 | 0.0719 | 18.7633 | 0.7488 | 0.1537 |
| 15000 HUGS | 25.9582 | 0.9147 | 0.0710 | 18.8005 | 0.7580 | 0.1532 |
| 12000 HUGS + 3000 xyz-only attention | 26.0527 | 0.9149 | 0.0705 | 18.9251 | 0.7609 | 0.1491 |
| 12000 HUGS + 4000 xyz-only attention | 26.3764 | 0.9203 | 0.0652 | **19.2497** | **0.7711** | 0.1421 |
| 12000 HUGS + 5000 xyz-only attention | 26.3918 | 0.9208 | 0.0645 | 19.2469 | 0.7705 | 0.1413 |
| 12000 HUGS + 6000 xyz-only attention | **26.4029** | **0.9210** | **0.0640** | 19.2466 | 0.7701 | **0.1408** |

说明：

```text
1. 12000 HUGS 是 start12000 主实验中 attention/correction 开启前的验证点。
2. 15000 HUGS 是原版 HUGS 完整 15000 步 baseline。
3. 12000 HUGS + 3000 xyz-only attention 是当前 start12000 xyz-only anchor-attention baseline 的 final 结果。
4. 12000 HUGS + 4000/5000/6000 xyz-only attention 是便于阅读的累计写法；实际后续 3000 步是从 12000 HUGS + 3000 xyz-only attention 的 final checkpoint 加载权重后继续 fine-tune，不是严格无缝 resume。
5. full image 的最优指标集中在 12000 HUGS + 6000 xyz-only attention；human crop 的 PSNR/SSIM 在 +4000 xyz-only attention 最优，HUMAN_LPIPS 在 +6000 xyz-only attention 最优。
```

#### 当前已跑 attention/correction 结果的参数范围标注

```text
截至 2026-05-21，本文档中所有已完成的 anchor-attention/correction 训练结果均为 xyz-only correction。

含义：
1. correction 模块只显式作用于 human Gaussian 的 xyz / mean / center。
2. correct_opacity=false，因此没有启用 opacity correction。
3. 没有显式 correction human scale / rotation / SH feature / color，也没有显式 correction scene Gaussian。
4. 训练循环中原版 HUGS 的 human/scene optimizer 仍会继续 step，因此最终 checkpoint 里的 human/scene base parameters 可能仍随渲染 loss 变化；
   但新增 interaction correction 分支本身只输出并施加 human xyz delta。

为了避免和后续 opacity / scale / rotation / feature correction 实验混淆，
相关输出目录已统一改名为 anchor_attention_xyz_*，表格中也统一标注为 xyz-only attention。
```

---

## Lab Quality Record: SuGaR Background and Anchor-Attention Continuation

This record uses NeuMan `lab` validation metrics from HUGS training outputs. Metric direction: higher PSNR/SSIM is better, lower LPIPS is better. Bold values indicate improvement relative to the paired baseline or previous stage; italic values indicate degradation.

### 15000-step Background Reconstruction

| Method | Steps | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUGS_HUMAN_PSNR | HUGS_HUMAN_SSIM | HUGS_HUMAN_LPIPS | Output |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| HUGS original | 15000 | 25.9582 | 0.9147 | 0.0710 | 18.8005 | 0.7580 | 0.1532 | `output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54` |
| SuGaR-HUGS background | 15000 | **26.1260** | *0.9143* | **0.0707** | **18.9424** | **0.7606** | **0.1494** | `output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_15000_lab_20260521/2026-05-21_21-21-45` |

### Attention-stage Metrics Under Different Background Reconstructions

| Background / Attention pipeline | Stage | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUGS_HUMAN_PSNR | HUGS_HUMAN_SSIM | HUGS_HUMAN_LPIPS | Output |
|---|---|---:|---:|---:|---:|---:|---:|---|
| HUGS background + attention | 12000 before attention | 25.9258 | 0.9127 | 0.0719 | 18.7633 | 0.7488 | 0.1537 | `output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_correction_15000_20260521/2026-05-21_02-45-33` |
| HUGS background + attention | 12000 + 3000 attention | 26.0527 | 0.9149 | 0.0705 | 18.9251 | 0.7609 | 0.1491 | same as above |
| HUGS background + attention | 12000 + 3000 + 3000 attention | 26.4029 | 0.9210 | 0.0640 | 19.2466 | 0.7701 | 0.1408 | `output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_extend_from15000_plus3000_20260521/2026-05-21_19-49-01` |
| SuGaR-HUGS background + attention | 12000 before attention | **25.9340** | **0.9130** | *0.0721* | **18.8669** | **0.7588** | **0.1512** | `output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_hugs_12000_lab_20260521_rerun/2026-05-21_22-55-25` |
| SuGaR-HUGS background + attention | 12000 + 3000 attention | **26.2388** | **0.9191** | **0.0659** | **19.1240** | **0.7689** | **0.1422** | `output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_hugs12000_anchor_attention_plus3000_stage1_20260521/2026-05-21_23-34-40` |
| SuGaR-HUGS background + attention | 12000 + 3000 + 3000 attention | *26.2989* | *0.9201* | *0.0648* | *19.1753* | *0.7697* | *0.1414* | `output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_hugs12000_anchor_attention_plus3000_stage2_20260522/2026-05-22_00-05-47` |

### Attention-stage Metric Growth

| Background | Attention interval | Delta PSNR | Delta SSIM | Delta LPIPS | Delta HUMAN_PSNR | Delta HUMAN_SSIM | Delta HUMAN_LPIPS |
|---|---|---:|---:|---:|---:|---:|---:|
| HUGS background | 12000 -> 15000 | **+0.1269** | **+0.0022** | **-0.0014** | **+0.1618** | **+0.0121** | **-0.0046** |
| HUGS background | 15000 -> 18000 | **+0.3502** | **+0.0061** | **-0.0065** | **+0.3215** | **+0.0092** | **-0.0083** |
| HUGS background | 12000 -> 18000 | **+0.4771** | **+0.0083** | **-0.0079** | **+0.4833** | **+0.0213** | **-0.0129** |
| SuGaR-HUGS background | 12000 -> 15000 | **+0.3049** | **+0.0061** | **-0.0061** | **+0.2572** | **+0.0101** | **-0.0090** |
| SuGaR-HUGS background | 15000 -> 18000 | **+0.0600** | **+0.0010** | **-0.0011** | **+0.0512** | **+0.0009** | **-0.0009** |
| SuGaR-HUGS background | 12000 -> 18000 | **+0.3649** | **+0.0071** | **-0.0072** | **+0.3084** | **+0.0110** | **-0.0099** |

### Takeaway

At 15000-step background reconstruction, SuGaR-HUGS improves full-image PSNR/LPIPS and human-region metrics over original HUGS, with a small SSIM drop. When attention starts from the 12000-step background checkpoint, SuGaR-HUGS gives a stronger first 3000-step attention gain, suggesting cleaner scene Gaussians help the attention/correction module converge faster. The second 3000-step attention stage still improves LPIPS slightly but largely reaches a PSNR/SSIM plateau; the original HUGS background extension reaches the stronger final 18000-step metric.

### 22.20 实验 B：12000 HUGS + 3000 xyz+opacity attention（2026-05-22）

本节记录一次在同源 12000 HUGS baseline 上进行的 correction 参数扩展实验。目标是比较：在原有 xyz-only attention correction 的基础上，额外开启 opacity correction 是否能带来更好的渲染质量。

实验设置：

```text
baseline checkpoint:
output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_hugs_12000_lab_20260521_rerun/2026-05-21_22-55-25/ckpt/

xyz-only 对照：
output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_hugs12000_xyz_attention_plus3000_stage1_20260521/2026-05-21_23-34-40/

xyz+opacity 实验 B：
config: cfg_files/release/neuman/hugs_scene_sugar_hugs12000_xyz_opacity_attention_plus3000_lab.yaml
output: output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_hugs12000_xyz_opacity_attention_plus3000_20260522/2026-05-22_00-46-26/
```

关键超参：

```text
train.num_steps=2998
anchor_attention.module_start_iter=0
anchor_attention.correction_start_iter=0
anchor_attention.correction_warmup_iters=0
anchor_attention.correct_opacity=true
anchor_attention.gamma_mu=0.003
anchor_attention.gamma_opacity=0.01
anchor_attention.delta_loss_w=0.002
anchor_attention.lr=5e-5
human.densify_until_iter=0
scene.densify_until_iter=0
```

同源对照指标：

| 实验 | 阶段 | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUMAN_PSNR | HUMAN_SSIM | HUMAN_LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 12000 HUGS baseline | final | 25.9340 | 0.9130 | 0.0721 | 18.8669 | 0.7588 | 0.1512 |
| 12000 HUGS + 3000 xyz-only attention | +1000 | 26.2079 | 0.9184 | 0.0674 | 19.1233 | 0.7701 | 0.1453 |
| 12000 HUGS + 3000 xyz+opacity attention | +1000 | 26.2282 | 0.9184 | 0.0674 | 19.1539 | 0.7704 | 0.1449 |
| 12000 HUGS + 3000 xyz-only attention | +2000 | 26.2116 | 0.9188 | 0.0665 | 19.0971 | 0.7686 | 0.1432 |
| 12000 HUGS + 3000 xyz+opacity attention | +2000 | 26.2089 | 0.9188 | 0.0665 | 19.0935 | 0.7689 | 0.1435 |
| 12000 HUGS + 3000 xyz-only attention | final | 26.2388 | 0.9191 | 0.0659 | 19.1240 | 0.7689 | 0.1422 |
| 12000 HUGS + 3000 xyz+opacity attention | final | 26.2270 | 0.9191 | 0.0660 | 19.1043 | 0.7686 | 0.1426 |

输出文件：

```text
results_train.json
train_neuman_lab.mp4
render_all_neuman_lab_final.mp4
canon_neuman_lab_final_a_pose.mp4
canon_neuman_lab_final_da_pose.mp4
ckpt/human_final.pth
ckpt/scene_final.pth
ckpt/anchor_attention_final.pth
```

观察：

```text
1. xyz+opacity 在 +1000 时略优于 xyz-only，full PSNR 与 human crop 指标均有微小正向变化。
2. 到 +2000 和 final 时，xyz+opacity 没有超过 xyz-only；final 的 PSNR、LPIPS、HUMAN_PSNR、HUMAN_LPIPS 均略差于 xyz-only。
3. 训练没有崩溃，l_anchor_delta 后半段偶尔升到约 0.1-0.13，但整体可控。
4. 当前 gamma_opacity=0.01 的 opacity correction 不带来稳定收益；它可能在早期有帮助，但后续会被原 HUGS 参数优化和 xyz correction 吸收，或者对透明度产生轻微扰动。
```

结论：

```text
在当前设置下，xyz+opacity 不是比 xyz-only 更强的默认 baseline。
下一步如果继续探索 opacity，建议做更保守的 opacity 分支消融：
A. gamma_opacity=0.003 或 0.005；
B. 增加 opacity delta clamp；
C. 只在接触 anchor 附近/高 attention scene token 附近启用 opacity correction；
D. 或者先只训练 1000 步并选择早停 checkpoint。

当前默认推荐仍保留 xyz-only attention correction。
```

## 22.21 多参数 correction 消融：xyz / opacity / scale / feature_dc（2026-05-22）

### 实验目标

在已经搭好的 anchor-attention baseline 上，比较 correction head 改变不同人体 Gaussian 参数时的效果。所有实验都从同一个 lab `12000-step HUGS` checkpoint 继续训练 3000 步 anchor-attention，因此表中所有 `+3000 attention` 分支都不是从零训练，也不是完整重跑 HUGS。

本轮只修正人体 Gaussians，不修正 scene Gaussians。场景高斯仍然保持 HUGS/SceneGS 原训练结果，attention 只用于生成人体 Gaussian correction。

### 已实现的 correction 参数开关

代码位置：

- `hugs/models/anchor_attention.py`
- `hugs/cfg/config.py`

当前 correction head 支持：

- `correct_mu`: 修正人体 Gaussian `xyz`，当前主线 baseline。
- `correct_opacity`: 修正人体 Gaussian opacity logit，使用小幅 residual。
- `correct_scale`: 修正人体 Gaussian scale，使用 `scale * exp(delta)` 并做 clamp。
- `correct_feature_dc`: 修正人体 SH 的 DC 颜色项 `shs[:, 0, :]`。

新增配置字段：

```yaml
anchor_attention:
  correct_scale: false
  correct_feature_dc: false
  gamma_scale: 0.01
  gamma_feature_dc: 0.01
  scale_delta_clamp: 0.05
  feature_delta_clamp: 0.05
  scale_max: 1.0
```

注意：旧的 anchor-attention checkpoint 可能不包含新增 head 的参数。当前这些消融都使用 `ckpt: null` 从 12000-step HUGS checkpoint 初始化 anchor-attention 模块，所以不会触发旧权重加载问题；如果后续要 resume 旧 anchor-attention，需要做 `strict=False` 或 checkpoint migration。

### 实验配置与输出

| 分支 | 配置文件 | 输出目录 |
|---|---|---|
| `xyz-only` | `cfg_files/release/neuman/hugs_anchor_attention_extend_from12000_plus3000_lab.yaml` | `output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_hugs12000_xyz_attention_plus3000_stage1_20260521/2026-05-21_23-34-40/` |
| `xyz+opacity` | `cfg_files/release/neuman/hugs_scene_sugar_hugs12000_xyz_opacity_attention_plus3000_lab.yaml` | `output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_hugs12000_xyz_opacity_attention_plus3000_20260522/2026-05-22_00-46-26/` |
| `xyz+scale` | `cfg_files/release/neuman/hugs_scene_sugar_hugs12000_xyz_scale_attention_plus3000_lab.yaml` | `output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_hugs12000_xyz_scale_attention_plus3000_20260522/2026-05-22_01-30-44/` |
| `xyz+opacity+scale` | `cfg_files/release/neuman/hugs_scene_sugar_hugs12000_xyz_opacity_scale_attention_plus3000_lab.yaml` | `output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_hugs12000_xyz_opacity_scale_attention_plus3000_20260522/2026-05-22_02-05-32/` |
| `xyz+feature_dc` | `cfg_files/release/neuman/hugs_scene_sugar_hugs12000_xyz_featuredc_attention_plus3000_lab.yaml` | `output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_hugs12000_xyz_featuredc_attention_plus3000_20260522/2026-05-22_02-36-31/` |

每个输出目录中都有：

- `results_train.json`: 1000/2000/final 指标。
- `val/`: validation 渲染图。
- `render_all/` 和 `render_all_neuman_lab_final.mp4`: 全序列渲染。
- `canon/`、`canon_neuman_lab_final_a_pose.mp4`、`canon_neuman_lab_final_da_pose.mp4`: canonical 可视化。
- `anchor_attention_debug/`: attention/debug 图像和数据。

### lab final 指标对比

下表指标均来自 `results_train.json` 的 `final`。PSNR/SSIM 越高越好，LPIPS 越低越好。

| 方法 | HUGS PSNR ↑ | HUGS SSIM ↑ | HUGS LPIPS ↓ | HUMAN PSNR ↑ | HUMAN SSIM ↑ | HUMAN LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|
| 12000 HUGS | 25.9340 | 0.9130 | 0.0721 | 18.8669 | 0.7588 | 0.1512 |
| `12000 HUGS + 3000 attention, xyz-only` | **26.2388** | **0.9191** | **0.0659** | **19.1240** | **0.7689** | **0.1422** |
| `12000 HUGS + 3000 attention, xyz+opacity` | 26.2270 | 0.9191 | 0.0660 | 19.1043 | 0.7686 | 0.1426 |
| `12000 HUGS + 3000 attention, xyz+scale` | 26.1812 | 0.9187 | 0.0665 | 19.0372 | 0.7663 | 0.1468 |
| `12000 HUGS + 3000 attention, xyz+opacity+scale` | 26.1885 | 0.9187 | 0.0665 | 19.0484 | 0.7662 | 0.1465 |
| `12000 HUGS + 3000 attention, xyz+feature_dc` | 26.2259 | 0.9191 | 0.0660 | 19.1044 | 0.7684 | 0.1431 |

### 过程指标观察

`xyz+feature_dc` 在 1000 步时有轻微优势：

- `xyz-only` 1000: HUGS PSNR 26.2079，HUMAN PSNR 19.1233，HUMAN LPIPS 0.1453。
- `xyz+feature_dc` 1000: HUGS PSNR 26.2177，HUMAN PSNR 19.1381，HUMAN LPIPS 0.1449。

但到 2000/final 后没有持续扩大收益，最终仍低于 `xyz-only`。这说明颜色 DC 的 correction 可能能更快拟合局部 appearance，但在当前 learning rate、delta clamp 和 loss 下没有带来稳定 final 提升。

### 当前结论

1. `xyz-only` 仍然是本轮最稳、final 指标最好的分支。
2. `opacity` 没有带来 final 提升，只在早期 1000 步略微提高部分指标；它可能更适合和更强的可见性/遮挡约束一起使用。
3. `scale` 在当前 clamp 和训练策略下明显不如 `xyz-only`，说明直接改人体高斯尺度容易破坏已有的人体外观/几何平衡。
4. `xyz+opacity+scale` 比 `xyz+scale` 略好，但仍然不能抵消 scale correction 的负面影响。
5. `feature_dc` 分支比 scale 稳定，但 final 与 `xyz+opacity` 接近，仍未超过 `xyz-only`。

因此，下一阶段建议先把 `xyz-only` 作为主 baseline，保留 opacity/feature_dc 的代码开关，但不要默认打开 scale。若要继续试多参数 correction，优先路径是：

- `xyz + very small opacity`，降低 `gamma_opacity` 或加入 opacity delta regularization。
- `xyz + feature_dc`，降低 `gamma_feature_dc` 并只在后 1000-1500 步打开，避免早期 appearance shortcut。
- scale 只作为后续有更好初始化/更好几何约束之后的二阶段实验，不建议当前作为主线。



## 2026-05-22 记录：anchor-attention 多参数 correction 消融

本次在 `lab` 上继续从同一个原版 HUGS 12000-step checkpoint 出发，后接 3000 step anchor-attention fine-tuning，验证多参数 correction 是否优于当前 `xyz-only` 主 baseline。所有 correction 都只作用于人体高斯，不直接修改场景高斯。

新增配置：

- `cfg_files/release/neuman/hugs_scene_sugar_hugs12000_xyz_scale_featuredc_attention_plus3000_lab.yaml`
- `cfg_files/release/neuman/hugs_scene_sugar_hugs12000_xyz_opacity_scale_featuredc_attention_plus3000_lab.yaml`

新增结果：

| correction 参数 | final HUGS PSNR | final HUGS SSIM | final HUGS LPIPS | final HUMAN PSNR | final HUMAN SSIM | final HUMAN LPIPS | 输出目录 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `xyz + scale + feature_dc` | 26.1769 | 0.9187 | 0.0665 | 19.0292 | 0.7660 | 0.1466 | `output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_hugs12000_xyz_scale_featuredc_attention_plus3000_20260522/2026-05-22_09-11-42/` |
| `xyz + opacity + scale + feature_dc` | 26.1807 | 0.9187 | 0.0665 | 19.0365 | 0.7662 | 0.1469 | `output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_hugs12000_xyz_opacity_scale_featuredc_attention_plus3000_20260522/2026-05-22_09-43-25/` |

对照：当前最好的 `xyz-only` attention final 为 HUGS 26.2388 / 0.9191 / 0.0659，HUMAN 19.1240 / 0.7689 / 0.1422。

结论：多参数联合优化可以稳定跑完，但目前没有超过 `xyz-only`。尤其 `scale` 相关组合持续拖低 HUMAN PSNR/SSIM/LPIPS，说明仅靠现有图像监督和当前 delta 正则，还不足以安全释放高斯形状自由度。后续默认实验仍使用 `xyz-only`，若继续测试 scale，应优先尝试更小 `gamma_scale`、更强 shape regularization、按 anchor/接触区域局部启用 scale。


## 2026-05-22 human token attention pooling 实验：xyz-only correction

本次针对 human token pooling 做单独验证：在同一个 `lab` 12000-step HUGS checkpoint 上，后接 3000 step anchor-attention fine-tuning，只开启 `xyz` correction，并明确设置 `human_pooling: attention`。

配置文件：

- `cfg_files/release/neuman/hugs_scene_sugar_hugs12000_xyz_attentionpool_xyzonly_plus3000_lab.yaml`

关键设置：

- `human.ckpt`: `output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_hugs_12000_lab_20260521_rerun/2026-05-21_22-55-25/ckpt/human_final.pth`
- `scene.ckpt`: `output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_hugs_12000_lab_20260521_rerun/2026-05-21_22-55-25/ckpt/scene_final.pth`
- `correct_opacity: false`
- `correct_scale: false`
- `correct_feature_dc: false`
- `human_pooling: attention`
- `gamma_mu: 0.003`
- `delta_loss_w: 0.002`
- `anchor_attention.lr: 5e-05`

输出目录：

- `output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_hugs12000_xyz_attentionpool_xyzonly_plus3000_20260522/2026-05-22_12-26-49/`

指标：

| step | HUGS PSNR | HUGS SSIM | HUGS LPIPS | HUMAN PSNR | HUMAN SSIM | HUMAN LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 26.2177 | 0.9184 | 0.0674 | 19.1376 | 0.7700 | 0.1452 |
| 2000 | 26.2198 | 0.9188 | 0.0665 | 19.1107 | 0.7689 | 0.1434 |
| final | 26.2398 | 0.9191 | 0.0660 | 19.1266 | 0.7688 | 0.1430 |

与此前记录的 `12000 HUGS + 3000 attention, xyz-only` 对比：

- 旧 `xyz-only`: HUGS 26.2388 / 0.9191 / 0.0659，HUMAN 19.1240 / 0.7689 / 0.1422。
- 本次 `xyz-only + attention pooling`: HUGS 26.2398 / 0.9191 / 0.0660，HUMAN 19.1266 / 0.7688 / 0.1430。

当前判断：attention pooling 至少不劣于 mean/旧 xyz-only 记录，在 PSNR 上有极小提升，但 LPIPS 没有同步变好。这个差异很小，不能单独证明 pooling 改动带来稳定质量提升；不过它说明 attention pooling 可以作为后续默认 token 聚合方式继续使用，尤其适合后面做 anchor-specific 或 contact-specific token 诊断。
