# RICH Stage-A v2：ParkingLot2 内部细粒度验证

## 结论与边界

本报告复评冻结的 epoch-5 `best.pt`，覆盖 ParkingLot2 内部开发划分的全部 2,426 个缓存分片、74,264 个候选人体—图像样本和 9,505,792 个有效脚底顶点标签。ParkingLot2 来自 official RICH train，因此这**不是** official validation 或 test；它只可用于当前方案的开发诊断，不能用于最终泛化结论或后续阈值调参。

复评与训练时保存的 epoch-5 指标一致，差异仅来自浮点累计顺序（AP `+2.02e-7`，F1 `-1.80e-6`）。脚本只将 `input_*` 的冻结预测特征送入模型；GT body/contact 只用于既有的关联标签和评测标签，未进入预测输入。

## 可复现命令与资产

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/workspace/nas_auto_backup/nas/yuzilang/miniconda3/envs/gush3r/bin/python -u \
forward_contact_pipeline/scripts/evaluate_rich_full_contact.py \
  --plan datasets/RICH/processed/full_contact_v1/plan \
  --cache datasets/RICH/processed/full_contact_v1/cache \
  --checkpoint forward_contact_pipeline/runs/rich_full_contact_v1_seed42_v2/best.pt \
  --config forward_contact_pipeline/configs/experiments/rich_full_contact_v1.json \
  --output forward_contact_pipeline/reproducibility/rich_full_contact_v1_seed42_v2/internal_validation_detailed.json \
  --device cuda
```

机器可读完整统计在同目录的 `internal_validation_detailed.json`。评测脚本逐片校验 plan SHA256、缓存 shard SHA256 和 checkpoint/cache contract；checkpoint SHA256 为 `645ef6e02c075a4da5578bffd61bea0561f26cf5cb46f613cd76568dc5b87406`。JSON 不含图像、RICH 数据或权重。

## 主结果

| 口径 | 有效顶点 | AP（1024-bin 近似） | Brier | precision@0.5 | recall@0.5 | F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|
| all candidate；未关联/歧义候选预测为无接触 | 9,505,792 | 0.944209 | 0.150313 | 0.900906 | 0.839319 | 0.869023 |
| 仅唯一关联成功的样本 | 8,524,416 | 0.974443 | 0.081422 | 0.900906 | 0.943429 | 0.921677 |
| 未关联或歧义样本，固定零预测 | 981,376 | 0.748710 | 0.748710 | 0.000000 | 0.000000 | 0.000000 |
| all-positive 基线 | 9,505,792 | 0.700453 | — | — | 1.000000 | 0.823843 |

唯一关联成功 66,597 / 74,264 个候选，coverage 为 **0.896760**；其余 7,667 个候选占有效顶点的 10.32%，且其中正例率为 74.87%。因此 matched-only 到 all-candidate 的 recall 落差（0.943429 到 0.839319）主要是关联覆盖，而不是该 contact head 在已关联样本上的分类能力。任何后续比较必须继续同时报告这两种口径和 coverage。

## 左右脚与主体分解

| 分组 | AP | Brier | precision@0.5 | recall@0.5 | F1@0.5 | coverage（主体） |
|---|---:|---:|---:|---:|---:|---:|
| 左脚（matched） | 0.974926 | 0.076926 | 0.896813 | 0.946946 | 0.921198 | — |
| 右脚（matched） | 0.974465 | 0.085918 | 0.904633 | 0.940276 | 0.922110 | — |
| subject 008（all candidate） | 0.951890 | 0.160211 | 0.920280 | 0.826662 | 0.870963 | 0.874500 |
| subject 014（all candidate） | 0.952339 | 0.119757 | 0.902359 | 0.867813 | 0.884749 | 0.941932 |
| subject 015（all candidate） | 0.944002 | 0.170253 | 0.882303 | 0.831330 | 0.856059 | 0.885802 |
| subject 016（all candidate） | 0.924560 | 0.151373 | 0.879381 | 0.839019 | 0.858726 | 0.893143 |

左右脚的 matched 指标接近；右脚的 Brier 略高。主体 014 的 coverage 最高、全候选 F1 也最高；主体 008 的 AP 很高但 coverage 最低之一，说明 pooled AP 不能替代关联诊断。

## 序列诊断

23 个序列都被评测。以下列出值得优先排查的边界样本，数值均为 all-candidate：

| 序列 | coverage | AP | F1@0.5 | 说明 |
|---|---:|---:|---:|---|
| `ParkingLot2_016_pushup2` | 0.83407 | 0.88376 | 0.81852 | 本次最低 AP/F1，且 coverage 最低 |
| `ParkingLot2_016_pushup1` | 0.86436 | 0.90337 | 0.83258 | 低 AP/F1，关联覆盖也偏低 |
| `ParkingLot2_008_overfence3` | 0.86491 | 0.90963 | 0.83421 | 低覆盖的跨障动作 |
| `ParkingLot2_014_pushup2` | 0.94167 | 0.91503 | 0.83115 | 覆盖较高仍有低 F1，需分辨分类与标签/动作差异 |
| `ParkingLot2_014_takingphotos2` | 0.95331 | 0.97033 | 0.91445 | 本次最高 F1 |
| `ParkingLot2_008_eating1` | 0.82546 | 0.97600 | 0.87358 | 最低 coverage 但 matched 分类可能很强；不能只看 pooled F1 |

完整 per-sequence 指标在 JSON 的 `sequences` 字段；下一轮错误分析应首先按 `status != matched` 的原因细分，并对上述序列固定抽取数值输入/关联记录审计，不读图选例、不以标签修补输入。

## 阈值与校准

在 all-candidate 的同一内部开发集上，F1 在阈值 0.45 时为 0.869186，0.50 时为 0.869023，差仅 0.000163；matched-only 的最佳 F1 恰在 0.50。由于这里复用了模型选择集，0.45 只是描述性统计，不能据此更改部署阈值或做对外结论。

概率分箱显示需要单独做校准：0--0.1 分箱共 2,432,200 顶点，平均预测概率 0.0126，但经验正例率 0.3319。这一分箱混入了所有未关联样本的强制零预测；所以它同时反映关联失败和概率失校准，不能直接用于温度缩放。高置信 0.9--1.0 分箱的平均概率 0.9847、经验正例率 0.9622，偏差较小。后续应在训练独立的 calibration split 上，对 matched 输出和关联/abstention 分支分别建模并报告 coverage-risk。

## 结论后的行动

1. 固定 epoch-5 checkpoint、0.5 阈值和本 split，不再以此评测结果调 epoch 或阈值。
2. 将唯一关联失败按检测缺失、多人歧义、退化/非有限输入等原因分桶；目标是提升 coverage，同时保持 all-candidate 主指标。
3. 对 RGB、point-map、global token、相机条件做一次一个变量的消融，沿用完全相同的 split、样本和 all-candidate 选择规则。
4. 下载并冻结 official RICH val/test 后，先在内部集完成选择，再对 official 数据做一次不可回看的评测。
