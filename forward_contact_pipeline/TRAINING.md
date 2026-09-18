# 可持续迭代的接触实验工程

2026-09-18更新：RICH全量监督与冻结前端 Stage-A contact v2 已完成30 epoch内部开发训练；最佳epoch5的all-candidate AP=.94421、F1=.86902。完整实验事实与限制以`reports/rich_full_contact_v1_v2_20260918.md`为准。GUSH3R前端的2cm几何门槛、official test、Stage-B、修正与LBS仍未解决，不能将该分类结果写成端到端部署结论。
独立的粗几何研究已实现非线性残差MLP、可微SMPL-X损失、训练内选模与分阶段新序列评价，
配置为`configs/experiments/rich_geometry_refinement_v1.json`，入口为`scripts/train_rich_neural_refiner.py`。
它需要含`roma/smplx/pytorch3d`的GUSH3R环境，不能只装`requirements-data.txt`；
它不是下文的正式接触训练cache或可部署校正器，研究脚本尚无中途epoch续训。结果与命令见
[真实几何校正报告](reports/rich_neural_refinement_20260916.md)。

当前版本提供**能执行并可续训的 Stage-A 数值缓存训练层**。旧 GRU/TCN、Stage-B
rollout 和诊断脚本保留原入口。2026-09-15 已完成 RICH 原始归档的 GT 标注整理，
全量 37669 人体帧的顶点/面对应核验通过，生成 1206 个数值分片，见 [RICH_PREPARATION.md](RICH_PREPARATION.md)。
这些 GT 分片还不是下文的训练缓存：真实冻结前端局部特征 exporter、RGB/帧/顶点关联、
真实前端精度审计，以及 corrected SMPL-X 到 GUSH3R Gaussian 的完整 binding 仍需接通。
本工程不把 GT 顶点或旧 PROX 几何教师当成真实前端输入。

## 模块边界

```text
RICH/PROX 原始数据 + 冻结前端
  -> 数据集/前端 exporter（各自处理帧、顶点、相机、坐标）
  -> 分片 inputs.npz + targets.npz + catalog.json
  -> build_training_index.py -> 不可变 index.json + 独立几何审计报告
  -> NumericCache -> inputs / targets / metadata
  -> 可替换 model(inputs) -> outputs
  -> 可替换 objective(outputs, targets)
  -> train + validation -> best.pt / last.pt
  -> 显式 evaluate --split test
```

- `contact_streaming/training/config.py`：YAML继承、严格命令行覆盖、本机路径。
- `data.py`：数值缓存适配、跨划分泄漏检查、有效性/版本检查、有界分片缓存。
- `components.py`：复用既有几何模型；可选RGB cross-attention、Gaussian属性融合；掩码损失和指标。
- `engine.py`：训练、验证选模、独立测试、checkpoint与随机状态恢复。
- `scripts/run_experiment.py`：统一 `preflight/train/evaluate`。

数据、模型、损失通过 `factory: package.module:callable` 注入；新实验通常只需一个新组件
和一份继承配置。配置文件是可信代码入口，不执行不明来源的factory。
优化器当前使用AdamW，评测当前实现接触指标；更换任务时在engine对应接口扩展。
当前引擎为单进程 CPU/单 GPU 路径，尚未接入 DDP、AMP 或学习率调度器；真实 RICH
GPU 的显存、吞吐和 DataLoader 参数仍需在数据接口接通后测量。
新增数据适配器实现 `audit/load/dataset/sha256` 接口；当前训练采样器要求dataset暴露
`items`中每个样本的分片编号，以减少NAS上反复解压同一NPZ。

Stage-B仍使用已有 `train_e1b_rollout.py` / `evaluate_e1b_rollout.py`。
不在这个版本把未通过的uncertainty head或尚未实现的LBS binding伪装成可部署组件。

## 安装与机器路径

推荐单独的Python 3.10环境用于数值缓存训练，避免与旧HUGS或GUSH3R的CUDA扩展冲突。
从仓库根执行：

```bash
python -m pip install -r forward_contact_pipeline/requirements-training.txt
python -m pip install --no-deps -e forward_contact_pipeline
cp forward_contact_pipeline/configs/paths.example.yaml forward_contact_pipeline/configs/paths.local.yaml
```

`requirements-training.txt`记录本机CPU回归所用版本。GPU机器需先按其CUDA/驱动选择
PyTorch wheel；还没有宣称跨GPU型号逐位复现。不要复制旧机器的conda目录或编译好的`.so`。

设置`HUGS_CONTACT_DATA_ROOT`、`HUGS_CONTACT_RUN_ROOT`，或直接在`paths.local.yaml`
写本机目录。相对路径相对于paths文件所在目录；cache内文件引用相对于data_root。
本机路径文件忽略进Git，实验配置里不写`/workspace/...`或用户名。

## 可运行的软件检查

以下从仓库根执行，数据都是数值合成fixture，不解码图像、不占GPU、不产生科研结论。
目录已存在时fixture生成器和新训练都会拒绝覆盖。

```bash
python forward_contact_pipeline/scripts/make_training_fixture.py --output forward_contact_pipeline/runs/fixture_data
export HUGS_CONTACT_DATA_ROOT="$PWD/forward_contact_pipeline/runs/fixture_data"
export HUGS_CONTACT_RUN_ROOT="$PWD/forward_contact_pipeline/runs/fixture_experiments"
python forward_contact_pipeline/scripts/run_experiment.py preflight --config forward_contact_pipeline/configs/experiments/smoke.yaml --paths forward_contact_pipeline/configs/paths.local.yaml
python forward_contact_pipeline/scripts/run_experiment.py train --config forward_contact_pipeline/configs/experiments/smoke.yaml --paths forward_contact_pipeline/configs/paths.local.yaml
python forward_contact_pipeline/scripts/run_experiment.py train --config forward_contact_pipeline/configs/experiments/smoke.yaml --paths forward_contact_pipeline/configs/paths.local.yaml --set trainer.epochs=3 --resume "$HUGS_CONTACT_RUN_ROOT/software_smoke/last.pt"
python forward_contact_pipeline/scripts/run_experiment.py evaluate --config forward_contact_pipeline/configs/experiments/smoke.yaml --paths forward_contact_pipeline/configs/paths.local.yaml --checkpoint "$HUGS_CONTACT_RUN_ROOT/software_smoke/best.pt" --split test
python -m unittest discover -s forward_contact_pipeline/tests -v
```

真实运行使用`base.yaml`或`rich_rgb_gaussian.yaml`复制出的命名配置，GPU运行显式
`--set trainer.device=cuda`。fusion配置中的768/8维是接口示例，必须换成实际缓存维数。
preflight会检查缺字段和通道不匹配，不会自动下载或执行backbone。

## 数值缓存契约 v1

每个shard建议不超过32帧，覆盖一个sequence、一个scene、一个subject；多人的录像
应按人体track拆分，scene/sequence身份保持原始身份，不能重命名以绕过隔离。
同一拍摄序列的不同摄像机也共享sequence_id，不能落到不同split。

`inputs.npz`：

| 字段 | 形状 | 含义 |
|---|---|---|
| vertex_local / vertex_normal | T,R,V,3 | 真实前端人体局部顶点/单位法向 |
| point_local / point_normal | T,R,V,P,3 | 每顶点场景邻居/单位法向 |
| point_valid | T,R,V,P | 0/1掩码，无邻居时全0 |
| rgb_tokens（可选） | T,R,K,D | 同帧或仅历史可见的冻结视觉token |
| gaussian_features（可选） | T,R,V,P,G | 与邻居一一对应的属性，语义由feature_version定义 |

`targets.npz`：`contact/contact_valid`为T,R,V；可选`proximity/proximity_valid`为T,R,V。
所有浮点数组须finite；无效位置填有限值并通过valid排除。有效法向为单位向量。
R为区域，第一版0/1对应左/右脚；相同index内保持相同区域与顶点布局。
GT只能在targets或独立审计资料中，禁止通过局部坐标系/选点偷渡GT进入部署输入。
缓存exporter须保留帧号、相机ID、SMPL-X顶点ID与原始文件索引作为独立sidecar，供审计追溯。

`catalog.json`/生成的`index.json`示意（路径均相对于data_root）：

```json
{
  "schema_version": 1,
  "dataset": "RICH",
  "provenance": {
    "kind": "real",
    "input_source": "frozen_backbone",
    "backbone_id": "GUSH3R",
    "backbone_revision": "实际代码及模型版本",
    "feature_version": "rich-local-v1",
    "label_source": "RICH official human-scene contact",
    "units": "m",
    "coordinate_frame": "contact_local",
    "temporal_context": "current_and_past",
    "topology": "smplx"
  },
  "audit_report": "rich_v1/geometry_audit.json",
  "records": [
    {
      "id": "sequence-camera-track-shard",
      "scene_id": "真实场景ID",
      "sequence_id": "真实拍摄序列ID",
      "subject_id": "真实人物ID",
      "split": "train",
      "input": "rich_v1/shards/example.inputs.npz",
      "target": "rich_v1/shards/example.targets.npz"
    }
  ]
}
```

用`build_training_index.py --data-root DATA_ROOT --catalog CATALOG --output rich_v1/index.json`
检查数值字段和划分身份并填充每个payload的SHA256。它不替用户生成“已通过”几何审计。
独立审计报告须记录`manifest_sha256`，并包含`checks.coordinates/topology/frame_alignment/
label_distribution/frontend_geometry`，全部为true才进入真实训练；具体误差、阈值、
held-out划分与审计程序版本应在报告中记录。布尔值是已完成测量的记录，不能手动改为true绕过失败。
训练会复核输入/标签文件hash，避免旧审计对应新数据。synthetic只能运行kind=smoke。

当前下载仅train+公共资产；有效scene/subject隔离无法从train构成时先补官方val。
test由显式evaluate命令读取，训练从不以test选模。数据检查拒绝整个split只有一种
接触类别；不会丢弃合法的全接触/全非接触帧。

## 结果与续训

每个运行目录保存resolved config、manifest摘要、环境与代码hash、每轮history、best/last。
best按验证集的分桶average precision选择；同时报告固定阈值F1、precision/recall、Brier
及逐区域指标。`average_precision_histogram`是1024桶近似AP，不能冒充论文精确PR-AUC；
最终评测可在独立评测器追加精确指标。proximity指标默认关闭；打开对应监督时同时设置
`objective.kwargs.proximity_weight`与`metrics.proximity=true`。

last保存模型、AdamW、Python/NumPy/Torch/DataLoader随机状态，支持epoch边界恢复。
断在epoch中间时重跑该epoch；非逐batch恢复。新运行不覆盖旧输出；续训必须指定该运行
的last.pt。可以增加总epoch、改变机器路径或设备；更改模型/损失/数据/hash/科学配置
会拒绝resume，应创建新的实验名称。跨设备/CUDA版本不保证逐位相同。

迁移整个运行文件夹即可在新run_root下继续；checkpoint通过`weights_only=True`读取。
运行文件夹和checkpoint通过NAS/rsync保存，代码/config通过Git保存。
