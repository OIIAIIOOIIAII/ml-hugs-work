# RICH真实前端几何：消融、校正与数据量实验（2026-09-16）

目前没有达到全身与脚底同时≤2cm的几何准入条件。数据处理、真实特征导出、复现校验和本轮小型校正实验已完成；正式接触估计/控制器训练没有启动。此次失败模型不接入部署。

## 1. 学了什么，使用了多少数据

之前报告的2.7–5.7cm是两个片段上的冻结GUSH3R误差，不是接触修复网络训练后的误差。较早PROX控制实验使用几何教师加合成扰动。本轮首次研究真实RICH输入上的小型SMPL-X参数校正，**尚未训练RICH接触估计器**。

- 输入：冻结前端预测的root/body旋转、体型、头部位置；另比较1792维视觉/人体query经训练内PCA32压缩后的特征。
- 监督：RICH身体姿态、性别模型向neutral模型投影后的形状系数、相机坐标头部位置。真值只参与训练目标与独立审计，不作为部署输入。没有使用接触标签训练本校正器。
- 输出：22个root/body旋转残差、10个形状系数残差、3个按深度归一化的位置残差，共79维。模型为标准化ridge线性回归。
- 计划：17训练序列×4片段×16帧=1088个图像样本。实际排除16个检测数量不唯一的片段，保留52片段、832个图像样本，来自553个不同人体帧、7位参与者。失败清单保留，不伪称1088个可用样本。
- 超参只在训练内部按序列留出选择：pose-only/pose+token，alpha=1/10/100/1000。选中pose-only、alpha=1000；确认集不参与拟合和选参。
- 确认集12个新序列，每段16帧：6个训练场景中的新序列，6个来自内部验证场景ParkingLot2。该内部验证与训练场景/参与者隔离，但不是官方val/test。

标签总量足够开始研究，不等于本轮校正器已经用过全部数据。现有监督缓存包含37585个有效人体帧；本轮只用553个不同训练人体帧。

## 2. 先修复实验可比性

1. HumanGS的`smpl_transl`表示头部位置，导出必须遵循其独立FK头部路径，不能直接当SMPL pelvis translation。普通SMPL joints路径仅作诊断。
2. CROCO导入时在`GUSH3R/src/croco/models/croco.py`开启TF32。校正脚本原先在CPU/默认FP32下重建，与上游TF32输出相差约1–2mm；改用相同CUDA+TF32后，代表样本零修正max-abs差为**0.0m**，没有放宽原0.1mm复现校验。
3. geometry-only冻结基线与之前完整HumanGS导出的32帧世界网格max-abs差为**0.0m**。关闭Gaussian分支没有改变上游几何预测。
4. RICH参数按gender、PCA12、non-flat-hand重建与官方顶点误差小于1e-6m；错误flat-hand约定可制造约8cm手部差异。
5. 本机GUSH3R与Human3R权重在去掉`module.`包装后，共有1631个张量、约11.77亿参数逐值相同，直接换这份权重不构成独立几何模型。

TF32与严格FP32的差异需要记录，但不足以解释本次厘米到数十厘米误差。

## 3. 冻结前端消融

每段前12帧拟合一个诊断Sim(3)，仅在后4帧测误差；绝不把真值对齐写回模型输入。表中为片段内误差中位数再跨片段取中位数，单位cm。world与camera-local是不同评测坐标路径，不可混为一个数字。

两个开发片段的world脚底误差：默认4.18、原生896输入3.31、768编码器+原生896为8.90、每帧重置+原生896为9.66、默认每帧重置15.06；全部未通过2cm。

| 12片段确认方法 | world全身 | world脚底 | camera-local脚底 | 可测片段 | 同时通过2cm |
|---|---:|---:|---:|---:|---:|
| 默认前处理 | 4.13 | 8.52 | 9.02 | 12/12 | 0/12 |
| 原始图直接采样到MHMR 896 | 4.19 | 8.19 | 8.83 | 12/12 | 0/12 |
| 已知相机内参重采样+原生896 | 6.01 | 6.67 | 6.94 | 11/12 | 0/12 |
| 原生896+额外32个过去帧 | 4.65 | 8.82 | 9.07 | 12/12 | 0/12 |

标定重采样在`ParkingLot2_014_phonetalk2`检测/身份失败，不能用减少后的分母宣称稳定胜出。原生896在8/12片段脚底优于默认，但最差片段仍约47cm。共完成58个开发/确认输出片段、928帧，另384个过去上下文帧；不包括早期实现修正后的重跑。

## 4. 真实监督校正结果

以下全部从相同native896 camera-local输出出发。原相机误差反映绝对位置/尺度等误差；诊断对齐后的误差观察去除一个片段级相似变换后剩余的几何和时间变化误差。两者都需要报告。

| 方法，12个确认片段 | 原相机全身 | 原相机脚底 | 对齐后全身 | 对齐后脚底 | 同时通过2cm |
|---|---:|---:|---:|---:|---:|
| 无修正 | 40.73 | 49.07 | 4.08 | 8.83 | 0/12 |
| 训练平均残差控制 | 35.96 | 37.32 | 4.18 | 9.82 | 0/12 |
| 学习全部79维残差 | 36.76 | 46.16 | 6.59 | 13.72 | 0/12 |
| 训练内选出的分量/强度方案 | 41.02 | 47.96 | 4.28 | 11.36 | 0/12 |

全部残差限制：每关节旋转≤0.35rad、每个beta修正≤2、位置修正长度≤预测深度的10%。可见，位置误差下降不代表脚底变准，学习全部参数反而使局部误差恶化。

最后一行是后续探索：预先声明旋转/体型/位置/全部四类，学习残差/训练平均残差两种来源，强度0.25/0.5/1共24候选，外加不修正回退。只在训练内选择，四个误差均不能超过其基线1.01倍，再最小化归一化均值。选中半强度学习旋转，但确认仍失败。该追加实验复用了已经看过的12片段，应称探索验证，不能称新的盲测。

## 5. 增加数据是否有效

固定模型/选参规则，以种子7/19/42按场景交错抽取4、8、17序列；每个预算独立拟合特征统计，确认集固定。表中只列未见ParkingLot2场景的6片段，报告三次子集的中位结果及范围，单位cm。

| 训练序列 | 图像样本数 | 原相机脚底，中位〔范围〕 | 对齐后脚底，中位〔范围〕 |
|---|---:|---:|---:|
| 4 | 192–224 | 57.32〔47.37–62.04〕 | 12.70〔11.78–19.01〕 |
| 8 | 400–432 | 50.82〔39.15–53.56〕 | 10.92〔7.43–11.76〕 |
| 17 | 832 | 46.16 | 12.93 |
| 不修正的原前端 | — | 57.52 | 6.56 |

8序列在这组实验的对齐脚底误差优于4序列，但17序列没有继续改善；所有预算均0/12通过。17序列三次使用完全相同的数据与确定性模型，因此数值相同，**不是三次独立重复证据**。范围只是子集变化，不是置信区间。

结论：目前不能把修复偏差简单归因于数据少，也不能排除数据不足。样本仍小、帧之间高度相关，且使用受限线性模型、固定小预算选参，不足以判断大模型或更充分覆盖数据的上限。现有证据支持同时处理泛化、特征/目标设计与修正幅度问题；没有依据立即扩大同一种模型的训练并承诺成功。

## 6. 真值残差上限：仅作诊断，不可部署

将正确残差直接交给重建器，用来区分表示能力与预测能力。仍使用neutral模型，手部等未全部替换；这不是学出来的结果，也没有写入部署缓存。

| 真值修正方式 | 原相机全身 | 原相机脚底 | 对齐后全身 | 对齐后脚底 | 同时通过2cm |
|---|---:|---:|---:|---:|---:|
| 姿态+体型+位置，保留幅度限制 | 2.81 | 6.61 | 1.86 | 4.92 | 4/12 |
| 姿态+体型+位置，取消幅度限制 | 0.47 | 1.15 | 0.39 | 0.54 | 12/12 |
| 仅姿态，取消限制 | 39.98 | 40.94 | 3.06 | 3.88 | 0/12 |
| 仅位置，取消限制 | 3.20 | 17.49 | 2.43 | 7.72 | 0/12 |
| 仅体型，取消限制 | 40.60 | 47.75 | 4.25 | 8.92 | 0/12 |

旋转单独审计也发现，每个确认片段均有帧需要超过0.35rad（约20°）的某个关节修正；这不意味着所有关节都错这么多。

表示本身具备达到门槛的能力；但原始误差不是单一位置或体型偏差，部分错误超出了“小幅修正”范围。直接放宽真实模型输出范围不能由oracle结果证明有效。

下一轮应把**较大初始误差的几何初始化**与**较小、连续的接触控制**分开：利用更充分的图像/局部场景证据学习几何残差，以顶点/关节几何损失检查脚部，采用置信度与回退；再讨论接触控制。要检验该方向，需要新的序列/参与者留出确认集，避免继续在这12片段上选方案。正式contact门槛保持，当前`training_ready=false`。

## 7. 复现与证据

代码：

- `scripts/run_rich_geometry_experiments.py`：冻结、纯数值前端消融。
- `contact_streaming/rich_geometry.py`：图像数值重采样与HumanGS几何重建。
- `scripts/evaluate_rich_geometry_experiments.py`：保留全部检测失败、原相机/对齐指标。
- `scripts/diagnose_rich_geometry.py`：GT模型约定与误差分解，普通SMPL joints路径只作近似诊断。
- `scripts/make_rich_geometry_training_plan.py`：固定训练片段，排除确认序列。
- `scripts/train_rich_geometry_calibrator.py`：训练内选参、学习曲线、可选分量与oracle诊断；每次写入新输出目录。

在仓库根目录运行校正研究：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/workspace/nas_auto_backup/nas/yuzilang/miniconda3/envs/gush3r/bin/python \
  forward_contact_pipeline/scripts/train_rich_geometry_calibrator.py \
  --rich-root datasets/RICH --repo GUSH3R \
  --train-plan datasets/RICH/processed/geometry_experiments_v3/training_plan.json \
  --train-features datasets/RICH/processed/geometry_experiments_v3/training/native896 \
  --confirmation-plan datasets/RICH/processed/geometry_experiments_v1/confirmation_plan.json \
  --confirmation-features datasets/RICH/processed/geometry_experiments_v3/confirmation/native896 \
  --output datasets/RICH/processed/geometry_experiments_v3/calibrator_new_run \
  --component-study
```

先检查GPU占用。TF32重建依赖CUDA和同版本SMPL资产，零修正校验失败时脚本停止，不能直接忽略。迁移时更换Python环境路径，并单独迁移数据/权重；它们不上传Git。

本地数值证据：

- `reports/rich_geometry_20260916/`：protocol、development_results、confirmation_results、error_decomposition、checkpoint_comparison_v2、reconstruction_device_probe、reconstruction_precision_probe、component_protocol、rotation_cap_audit及源文件快照。
- `datasets/RICH/processed/geometry_experiments_v3/calibrator_v2/report.json`：首轮完整校正与学习曲线。
- `datasets/RICH/processed/geometry_experiments_v3/calibrator_v3/report.json`：复测与追加分量/oracle实验；同目录含模型、源码快照及哈希。
- `calibrator/`保留早期零修正校验失败记录；`geometry_experiments_v1/development`保留早期辅助导出差异，不替代v2的精确基线结果。
- 冻结确认计划SHA256：`79944abf8125439952912fc335fd06872dfa3a7e8f7046876fe560c622e1ac79`。
- 冻结训练计划SHA256：`87679eb939530fb0a137d8b5a3c5231e1411ced2795ac847d58c5b49007c726a`。

验证：此前完整CPU套件52项通过，本次2项相关图像变换测试通过；真实GPU零修正max-abs=0、两次校正主实验结果一致、8组初次超参比较、9组预算/种子、24组追加分量候选及8组oracle均已完成。测试通过只保证对应软件契约，不代表2cm几何目标通过。
