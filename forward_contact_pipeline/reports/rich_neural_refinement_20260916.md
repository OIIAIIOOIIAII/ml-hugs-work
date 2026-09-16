# RICH非线性几何校正与新序列验证（2026-09-16）

**本轮完成了训练与新序列评估，2cm目标仍未解决。** 可微几何损失使模型更好地拟合训练数据，但没有推广到新序列。全部候选均不用于部署，正式接触网络仍未启动。本轮证据进一步指向误差分布变化与小样本覆盖问题，而不是仅凭训练误差决定成功。

## 实验设计与输入边界

在上一轮ridge研究基础上，复用17序列、832个图像样本的真实冻结GUSH3R缓存。其中13序列/624样本用于梯度训练，4序列/208样本用于选模型和epoch。总缓存来自553个不同人体帧、7位参与者；连续帧和不同机位并非独立样本。

三种配置，各用seed7/19，共6组训练：

1. `pose_parameter`：211维预测姿态/体型/头部位置，监督79维参数残差。
2. `visual_parameter`：另加入完整1792维SMPL/视觉query，不再先压缩到PCA32，仍用参数损失。
3. `visual_geometry`：相同视觉输入，加可微SMPL-X顶点/脚底/头部位置损失。

这里的“完整query”是保留该人体query的全部维度，**不是完整图像patch或局部场景特征**。没有新增场景表面输入，也没有接触标签、真值人体或相机外参作为部署输入。身体真值仅作为监督；原预测相机用于重建。

模型为hidden128的两层MLP，带LayerNorm、GELU、dropout0.1，末层零初始化。AdamW，lr0.001、weight decay0.001、batch32、60epochs。参数输入模型约5.4万参数，视觉版约28.3万参数。只训练MLP，上游GUSH3R与SMPL资产冻结。

粗几何研究范围扩大为每关节旋转≤1.2rad、beta分量≤3、位置长度≤预测深度25%；这不是接触控制器的动作范围，也没有改变2cm验收标准。几何损失包含512固定人体顶点与128脚底顶点的head-relative SmoothL1（脚底权重3）、按深度归一化的头部位置损失、0.01参数损失和小残差正则。

选模指标为训练内部留出集上原相机全身/脚底、诊断对齐后全身/脚底四项误差相对基线的归一化均值。零修正是可选回退；所有6组候选完整报告。锁定`visual_geometry_seed7`、epoch60后，才加载新确认集的GT缓存并评价。

## 新序列与覆盖失败

预先固定以下6个新序列，各32帧：

- ParkingLot2_008_pushup2、ParkingLot2_008_phonetalk1
- ParkingLot2_014_burpeejump2、ParkingLot2_014_takingphotos2
- ParkingLot2_015_overfence1、ParkingLot2_016_pushup2

初始选择中间有效机位，六段均为camera3。推理发现5段输出多人，不满足当前单人匹配规则，仅takingphotos2整段可测。这不等价于5段都发生误检：场景可能确有其他人，当前评估尚无可靠目标身份匹配。所有这些不覆盖片段仍计入分母。

在读取其几何误差和本轮训练结果之前，统一补充**全部相同序列、相同帧号的camera0**。六段均满足单人规则，192帧可测。原camera3的32帧仍单列；总计12视角片段、7个可测，仍只有6个独立序列，不能称12个独立样本。

确认参与者与训练互斥，但这些参与者出现在以前看过的验证序列；因此这是新序列确认，不能称完全未见参与者的研究历史、官方val/test或最终主表。新窗口前24帧只拟合诊断Sim(3)，后8帧评估；不把对齐写回输入。

初始计划SHA256：`3ab9b5fde6bd4c01d34ec4565325fcb3b08a1b0343a14056dba1a6858ff1aad1`。
补充机位计划SHA256：`6487aaca14dffce1845fc80fcf480b922dfb061ab0627484f256f17487b7e5c4`。

## 新camera0结果

单位cm，先取每片段误差中位数，再跨6片段取中位数。原相机指标含整体位置偏差，对齐指标仍含局部姿态及片段内变化；二者不能相互替代。

| 方法 | 原相机全身 | 原相机脚底 | 对齐后全身 | 对齐后脚底 | 同时≤2cm |
|---|---:|---:|---:|---:|---:|
| 冻结native896，无修正 | 23.84 | 25.58 | 8.84 | 10.02 | 0/6 |
| pose_parameter，seed7 | 41.12 | 48.76 | 18.80 | 26.92 | 0/6 |
| pose_parameter，seed19 | 33.74 | 34.22 | 8.85 | 25.80 | 0/6 |
| visual_parameter，seed7 | 47.91 | 45.69 | 9.07 | 19.37 | 0/6 |
| visual_parameter，seed19 | 44.26 | 38.41 | 10.31 | 27.57 | 0/6 |
| **已锁定visual_geometry，seed7** | **39.26** | **37.71** | **6.66** | **22.47** | **0/6** |
| visual_geometry，seed19 | 38.28 | 31.83 | 7.77 | 20.08 | 0/6 |

几何损失模型改善了部分对齐全身指标，但脚底和绝对位置退化，不能采用。原camera3唯一可测片段的对齐脚底误差从3.69cm变为22.20cm，同样失败；其余5片段保留单人匹配失败记录。计入全部声明视角，2cm通过数仍为0/12。

## 为何训练效果不能代表泛化

选中模型在梯度训练的624样本上：原相机全身2.74cm、脚底2.70cm；对齐后全身2.58cm、脚底2.91cm。训练内部留出208样本的原相机脚底已升至16.12cm，换到新camera0片段又升至37.71cm。

独立误差分布审计发现，训练624帧需要的头部深度修正中位数为**+41.98cm**，新camera0帧为**−11.70cm**；深度归一化后仍为+0.1064和−0.0295。修正方向已经改变，支持“模型学到训练数据特有偏差”的解释，但不能只用这个统计证明全部失败原因。

这次更强模型确实可以拟合训练样本，不能再把小训练误差当成功证据。小样本、高相关性、动作/人物/机位覆盖、已压缩上游query的信息限制均可能影响泛化；本轮确认还把窗口从16帧扩为32帧，因此不能从这一次对照分离各个变化的因果贡献。本实验不能证明只要扩大数据就必然成功，也不能否定更充分数据的作用。

另做了GT仅供审计的投影足迹统计：新camera0人体在896输入上中位包围盒高度约237–323像素。没有证据把这些片段统一描述为只有几十像素的小人；局部脚部细节与遮挡仍需独立检查。

## 相机标定的解析对照

附加一个不训练、不修改姿态/体型的固定假设：把模型在默认60°虚拟相机下自己预测的像素，投到实际传感器K，并通过线性最小二乘求整体平移。只使用预测顶点、裁剪变换和传感器内参，GT身体只用于评价；它不能提供真正观测到的图像关键点，也不能修复姿态。

camera0结果：原相机全身23.84→17.73cm，脚底25.58→22.78cm；对齐后全身8.84→8.46cm、脚底10.02→10.69cm。说明K一致性可能帮助整体位置，但此方案没有解决脚底精度。camera3唯一片段的对齐脚底3.69→3.72cm。没有据此改选神经网络或宣称已获得接触改善。

## 实现验证

- 发现并处理HumanGS `get_target_transform`上的`no_grad`包装：训练路径绕过这一包装，保留完整头部FK对姿态/体型的梯度。
- 6个旋转、体型、平移方向通过有限差分；零初始化所有梯度finite。
- TF32批量与逐帧矩阵内核会产生最高约3.92mm差异。训练使用FP32，批量与逐帧FP32重建max-abs差为**1.43e-6m**。最终评价始终用原发布的逐帧TF32路径；代表训练样本和全部224个可测确认帧零修正max-abs均为**0.0m**。没有放宽原0.1mm导出校验或2cm精度门槛。
- 全部6组训练约199秒，不含缓存/模型加载、冻结前端导出。额外对已锁定的seed7几何模型独立复训，模型全部参数数组、训练记录及选中指标逐值相同，完成审计已落盘。
- 全CPU套件58项：55通过、3项因默认环境缺可选`roma`跳过；这3项在GUSH3R环境另跑全部通过。包括残差幅度/旋转合法性、零初始化梯度、GT额外字段不影响预测参数，以及相机解析已知解。测试通过不表示真实精度通过。

## 下一轮边界

保留可微实现、缓存契约和新验证证据，拒绝部署本轮MLP。优先扩大训练动作/人物/机位覆盖，并强化按参与者与机位留出的训练内验证；目前只有624个图像样本参与梯度训练，远少于已处理的数据规模。若继续更改输入，需独立比较局部视觉/场景条件的作用，不能将未使用的场景特征写成已有能力。新的方案需要重新预留确认序列，避免在本轮6序列上选模型。正式contact训练保持`training_ready=false`。

## 复现入口

新增代码：`contact_streaming/rich_refinement.py`、`camera_translation.py`，以及`scripts/make_rich_geometry_holdout.py`、`prepare_rich_refinement_cache.py`、`train_rich_neural_refiner.py`、`evaluate_rich_camera_translation.py`。

受控配置已纳入Git：`configs/experiments/rich_geometry_refinement_v1.json`。在仓库根目录，先检查GPU，再用包含torch、roma、smplx、pytorch3d的GUSH3R环境执行：

```bash
# 缓存准备使用CPU，不重新运行前端。
python forward_contact_pipeline/scripts/prepare_rich_refinement_cache.py \
  --rich-root datasets/RICH --models GUSH3R/src/models \
  --plan datasets/RICH/processed/geometry_experiments_v3/training_plan.json \
  --features datasets/RICH/processed/geometry_experiments_v3/training/native896 \
  --output datasets/RICH/processed/geometry_experiments_v4/train_cache

python forward_contact_pipeline/scripts/train_rich_neural_refiner.py train \
  --cache datasets/RICH/processed/geometry_experiments_v4/train_cache \
  --repo GUSH3R --rich-root datasets/RICH \
  --output datasets/RICH/processed/geometry_experiments_v4/neural_new_run \
  --protocol forward_contact_pipeline/configs/experiments/rich_geometry_refinement_v1.json \
  --inner-split datasets/RICH/processed/geometry_experiments_v3/calibrator_v3/report.json

# selection.json生成后才进行新序列评价；新输出目录防止覆盖。
python forward_contact_pipeline/scripts/train_rich_neural_refiner.py evaluate \
  --cache datasets/RICH/processed/geometry_experiments_v4/holdout_camera0_cache \
  --repo GUSH3R --rich-root datasets/RICH \
  --output datasets/RICH/processed/geometry_experiments_v4/evaluation_new_run \
  --training-run datasets/RICH/processed/geometry_experiments_v4/neural_new_run
```

初始确认计划由`make_rich_geometry_holdout.py`生成，补测可用`--camera-zero-of`复现。缓存manifest校验源计划、模型、标定和数据hash；训练与评估分阶段，模型选择/权重hash在新集评价前锁定。此有界研究脚本保存各完成配置的权重和日志，尚无中途epoch续训入口，重跑使用新目录。

本地证据：`datasets/RICH/processed/geometry_experiments_v4/`中的train/holdout缓存、`neural_v1/selection.json`、两个`evaluation_camera{0,3}_v1/report.json`、复训目录；以及`reports/rich_geometry_neural_20260916/`中的协议、梯度、测试、相机标定和误差分布审计。数据/权重/生成缓存不上传Git，源码、配置、测试和本报告随Git迁移。
