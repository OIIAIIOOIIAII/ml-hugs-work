# RICH 全量 Stage-A Contact v1 v2：最终运行报告

## 结论

`rich_full_contact_v1_seed42_v2` 已在单张 RTX 4090 上完成 30 个 epoch。最佳 checkpoint 是 epoch 5，而非最后一轮。它在 RICH official train 中划出的 ParkingLot2 内部开发集上，显著优于 all-positive 分类基线；这证明冻结前端的局部特征可支持接触分类，但不证明官方泛化、接触修正、时序控制或 Gaussian 渲染改善。

## 研究问题与范围

目标是训练 Stage-A：对每个已关联预测人体的左右脚底各 64 个顶点输出接触 logit。GUSH3R 完全冻结；不训练其人体/场景分支，也不训练先前失败的全局几何残差模型。当前 run 不包含 Stage-B、low-D correction、SMPL-X FK/IK、LBS 回写、uncertainty calibration 或 official val/test。

RICH official train 被按 scene/sequence/subject 严格划分：非 ParkingLot2 为 train，ParkingLot2 为内部 validation。计划包含 165,920 train 人体—图像样本（158,428 unique images）和 74,264 internal-val 样本。所有合格样本恰好进入一个 split；官方 val/test 未下载。

## 输入、监督与模型

每个预测人物先由其预测头部与 5% padding 的 GT 人体投影框做互斥关联。GT 只在该关联、接触标签和指标计算中使用，绝不进入网络输入。关联失败、歧义、退化预测或非有限输入被保留为未匹配目标；validation 以固定负 logit 计入 all-candidate 指标，训练不制造假输入或假标签。

每个脚 ROI 的输入为：64 个预测 SMPL-X 脚底顶点的局部位置/法向、每顶点最多 16 个预测 point-map 邻居的位置/法向/valid mask、25 个局部 DINO patch token、16 个全局 DINO token、SMPL pose/shape/head、冻结 person query、物理相机条件、预测头部 UV 和左右脚 region ID。局部输入均来自当前帧冻结预测；没有使用未来帧。

`FullContactModel` 使用共享 point MLP 和 masked mean pooling、vertex MLP、person pose/query/sensor context、每顶点 learned semantic embedding，以及对 RGB token 的 cross-attention，输出每个脚底 vertex 的 contact logits。配置：hidden=128、4 heads、dropout=.1、AdamW lr=3e-4、weight decay=1e-4、batch=64 foot ROIs、seed=42、30 epoch。

## 运行版本与完整性

| 项目 | 值 |
|---|---|
| 有效 run | `forward_contact_pipeline/runs/rich_full_contact_v1_seed42_v2` |
| 状态 | complete，30/30 epoch，156,150 optimizer steps |
| 配置 | `configs/experiments/rich_full_contact_v1.json` |
| plan SHA256 | `8da5750b3decac182b24521d80a6422d8d66f38eaa587bb99a340691b5dfa141` |
| GUSH3R checkpoint SHA256 | `1e390dcf65f440dea4527378af1ef6bd7859b74465ab1925e58534c9e9208fe1` |
| feature cache | 7,525 NPZ shards，约35GB |
| best.pt SHA256 | `645ef6e02c075a4da5578bffd61bea0561f26cf5cb46f613cd76568dc5b87406` |
| last.pt SHA256 | `391c3068581acbff89d132c7ca851821c6441603dea315c84c3c32629e9ad698` |

v1 在 epoch 1 validation 的 `ParkingLot2_016_burpeejump2` camera 3（clip 7044）发现退化预测脚底法向并抛出异常，因而作废。v2 修改前端，使退化法向或 non-finite 输入的预测人体跳过并作为未匹配目标记录。v2 已实际跨过该 clip 并完成 30 轮。该修复防止崩溃，同时让失败惩罚留在主指标中。

## 验证指标与选择规则

主指标是 `all_candidates_missing_as_negative.average_precision_histogram`（1024-bin AP 近似）；best.pt 仅在完整 internal validation 后按该指标更新。每轮还报告 fixed-threshold F1/precision/recall、matched-only 指标、coverage 和 all-positive baseline。AP 为直方图近似，不能冒充精确 PR-AUC。

| epoch | all-candidate AP | all-candidate F1 | precision | recall | matched AP | matched F1 | coverage |
|---:|---:|---:|---:|---:|---:|---:|---:|
| all-positive baseline | 0.70045 | 0.82384 | — | 1.00000 | — | — | — |
| 1 | 0.92736 | 0.84757 | 0.86751 | 0.82853 | 0.95551 | 0.89827 | 0.89676 |
| 5 **best** | **0.94421** | **0.86902** | 0.90091 | 0.83932 | **0.97444** | **0.92168** | 0.89676 |
| 10 | 0.94340 | 0.86099 | 0.92100 | 0.80832 | 0.97352 | 0.91475 | 0.89676 |
| 20 | 0.94181 | 0.86342 | 0.92063 | 0.81290 | 0.97172 | 0.91717 | 0.89676 |
| 30 | 0.93940 | 0.85196 | 0.92943 | 0.78641 | 0.96899 | 0.90613 | 0.89676 |

第 5 轮后 AP 进入平台并略有回落，后期 precision 升高而 recall 下降。选择第 5 轮不是 cherry-pick：它遵循训练开始前写入代码的全候选 AP 选择规则，且 `best.pt` 在后来 epoch 未超过该数值时未更新。

## 冻结 checkpoint 的细粒度内部复评（2026-09-18）

已用新增的 `scripts/evaluate_rich_full_contact.py` 对冻结 epoch-5 `best.pt` 完整重评 2,426 个 ParkingLot2 internal-validation cache shards（74,264 candidates、9,505,792 valid vertices），并逐片校验计划、缓存和 checkpoint contract。结果与训练历史复现一致：all-candidate AP=0.944209、F1@0.5=0.869023；matched-only AP=0.974443、F1@0.5=0.921677；coverage=0.896760。数值差异仅由浮点累计顺序造成，均小于 2e-6。

细粒度证据、左右脚/主体/序列分解、阈值曲线、概率分箱和可复现命令见 [`../reproducibility/rich_full_contact_v1_seed42_v2/internal_validation_detailed.md`](../reproducibility/rich_full_contact_v1_seed42_v2/internal_validation_detailed.md) 及同目录机器可读 JSON。该复评仍是从 official train 划分的内部开发验证，绝非 official val/test。其最明确的工程发现是：7,667 个未唯一关联候选占有效顶点 10.32%，正例率 74.87%，使 matched recall 0.943429 降至 all-candidate 0.839319；关联覆盖是下一步独立于分类 head 的主瓶颈。

## 已知限制与严禁外推

1. 这是内部 train split，不是 official val/test；不能报告为最终泛化。
2. 约 10.32% validation 候选没有被唯一关联，coverage 是系统瓶颈的一部分，不能只看 matched 分数。
3. 评价的是 RICH vertex contact 分类；没有证明可减少 foot sliding、penetration、jitter 或 render error。
4. GUSH3R 粗人体几何在既有独立诊断中未达到 2cm contact-scale 门槛；该事实没有因 Stage-A 分类分数提升而解除。
5. 该模型没有 uncertainty/calibration 输出；不可据此自动执行 correction 或安全 abstain。
6. 官方 val/test、跨机器重跑、不同 GPU/驱动都不保证逐位相同；每次须记录 code/contract/checkpoint/data hash。

## 下一阶段的可证伪顺序

1. 冻结 epoch-5 best.pt，做分 scene/sequence/subject、左右脚、关联失败原因和接触 prevalence 的误差报告。
2. 在同一 split 和固定选择规则上进行消融：移除 RGB、local point patch、global token、physical camera context；不同时改架构与数据。
3. 获取并冻结官方 validation/test；仅在开发集完成方案选择后做一次官方评测。
4. 为 Stage-A 增加可靠性/校准与 abstention，报告 calibration 和 coverage-risk，达标后才进入 Stage-B。
5. 将 Stage-A ROI token 与模型自身过去状态接到 Stage-B causal rollout，比较 no-correction、现有机制 baseline 和 Stage-A+B；禁止 teacher previous state。
6. 仅在 rollout 通过后接 bounded correction、SMPL-X 和 human Gaussian LBS；scene Gaussian 保持冻结。
