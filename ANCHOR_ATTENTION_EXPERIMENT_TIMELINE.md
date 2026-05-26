# Anchor Attention 实验结果时间线

本文档只记录实验设置、指标结果和输出位置，不展开代码实现细节。早期 Human3R 初始化相关实验不纳入本文档。

指标说明：

- `HUGS_*`：完整图像 validation 指标。
- `HUMAN_*`：human crop validation 指标。
- PSNR / SSIM 越高越好，LPIPS 越低越好。
- `xyz-only attention correction` 表示只修正人体 Gaussian 的 `xyz/mean/center`，不修正 scene Gaussian，也不修正人体 opacity、scale、rotation、SH/color。
- `mean pooling` / `attention pooling` 指 human Gaussian 聚合为 anchor token 的 pooling 方式。

## 2026-05-19：原版 HUGS baseline 复现

目标：先获得完全依托原版 HUGS 训练流程的可评估 baseline，作为后续 attention / SuGaR-HUGS 对比基准。

共同设置：

```text
dataset: NeuMan lab
training: 原版 HUGS human_scene
attention/correction: off
主要步数: 6000 / 15000
```

| 场景 | 实验设置 | 步数 | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUMAN_PSNR | HUMAN_SSIM | HUMAN_LPIPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lab | 原版 HUGS | 6000 | 25.6634 | 0.9012 | 0.0843 | 18.7363 | 0.7432 | 0.1652 |
| lab | 原版 HUGS | 12000 checkpoint | 25.9258 | 0.9127 | 0.0719 | 18.7633 | 0.7488 | 0.1537 |
| lab | 原版 HUGS | 15000 final | 25.9582 | 0.9147 | 0.0710 | 18.8005 | 0.7580 | 0.1532 |

输出位置：

```text
output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_6000_20260519/2026-05-19_23-10-00
output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_15000_20260519/2026-05-19_23-26-54
```

## 2026-05-21：lab 上的首个完整 xyz-only attention baseline

目标：在不引入显式接触监督的情况下，验证完整链路是否能稳定训练并带来渲染指标提升。

共同设置：

```text
dataset: NeuMan lab
0-12000: 原版 HUGS 正常训练
12000-15000: 开启 anchor-attention correction
correction: xyz-only
human token pooling: mean pooling
correction_warmup_iters: 1000
gamma_mu: 0.005
delta_loss_w: 0.001
zero_init_delta: true
```

| 场景 | 实验设置 | seed | 步数/阶段 | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUMAN_PSNR | HUMAN_SSIM | HUMAN_LPIPS |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lab | 原版 HUGS | 0 | 15000 final | 25.9582 | 0.9147 | 0.0710 | 18.8005 | 0.7580 | 0.1532 |
| lab | HUGS12000 + 3000 xyz-only attention | 0 | final | 26.0527 | 0.9149 | 0.0705 | 18.9251 | 0.7609 | 0.1491 |
| lab | HUGS12000 + 3000 xyz-only attention | 1 | final | 26.0068 | 0.9140 | 0.0724 | 18.8995 | 0.7611 | 0.1510 |
| lab | HUGS12000 + 3000 xyz-only attention | 2 | final | 26.1772 | 0.9165 | 0.0707 | 19.0855 | 0.7654 | 0.1477 |
| lab | HUGS12000 + 3000 xyz-only attention | 0/1/2 | mean | 26.0789 | 0.9151 | 0.0712 | 18.9700 | 0.7625 | 0.1492 |

输出位置：

```text
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_correction_15000_20260521/2026-05-21_02-45-33
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_correction_15000_seed1_20260521/2026-05-21_09-22-56
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_correction_15000_seed2_20260521/2026-05-21_10-34-44
```

阶段性结论：三次 seed 的 human crop 指标均优于原版 HUGS；full image 指标总体小幅提升，但幅度较小。

## 2026-05-21：bike 跨场景初测

目标：检查 lab 上的小幅收益是否能迁移到另一个 NeuMan 场景。

共同设置：

```text
dataset: NeuMan bike
原版 HUGS: 15000
attention run: 0-12000 HUGS, 12000-15000 xyz-only attention correction
human token pooling: mean pooling
```

| 场景 | 实验设置 | 步数/阶段 | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUMAN_PSNR | HUMAN_SSIM | HUMAN_LPIPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bike | 原版 HUGS | 6000 checkpoint | 24.6140 | 0.7919 | 0.1555 | 19.6295 | 0.6551 | 0.1933 |
| bike | HUGS attention run | 6000 checkpoint | 24.6360 | 0.7919 | 0.1546 | 19.6729 | 0.6491 | 0.1907 |
| bike | HUGS attention run | 12000 checkpoint | 25.4867 | 0.8326 | 0.1143 | 19.7824 | 0.6648 | 0.1601 |
| bike | 原版 HUGS | 15000 final | 25.6923 | 0.8419 | 0.1027 | 19.7592 | 0.6765 | 0.1520 |
| bike | HUGS12000 + 3000 xyz-only attention | final | 25.7943 | 0.8411 | 0.1031 | 20.0542 | 0.6800 | 0.1483 |

输出位置：

```text
output/human_scene/neuman/bike/hugs_trimlp/exp0_hugs_original_15000_bike_20260521/2026-05-21_12-32-06
output/human_scene/neuman/bike/hugs_trimlp/anchor_attention_correction_15000_bike_20260521/2026-05-21_13-17-51
```

阶段性结论：bike 的 human crop 三项指标提升，full image PSNR 提升，但 full SSIM / LPIPS 基本持平或轻微变差。

## 2026-05-21：correction start time 消融

目标：确认 correction 应该在 HUGS 基本收敛后再介入，还是可以更早介入。

共同设置：

```text
dataset: NeuMan lab
total schedule: 15000
correction: xyz-only
human token pooling: mean pooling
只改变 module_start_iter / correction_start_iter
```

| 场景 | 实验设置 | correction start | 是否完整跑完 | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUMAN_PSNR | HUMAN_SSIM | HUMAN_LPIPS |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lab | 原版 HUGS | none | 是 | 25.9582 | 0.9147 | 0.0710 | 18.8005 | 0.7580 | 0.1532 |
| lab | xyz-only attention | 12000 | 是 | 26.0527 | 0.9149 | 0.0705 | 18.9251 | 0.7609 | 0.1491 |
| lab | xyz-only attention | 9000 | 是 | 26.0416 | 0.9132 | 0.0721 | 18.9643 | 0.7576 | 0.1512 |
| lab | xyz-only attention | 6000 | 否，约 11200 停止 | - | - | - | - | - | - |

关键中间点：

| 场景 | 实验设置 | iter | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUMAN_PSNR | HUMAN_SSIM | HUMAN_LPIPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lab | start12000 | 12000 | 25.9258 | 0.9127 | 0.0719 | 18.7633 | 0.7488 | 0.1537 |
| lab | start12000 | 14000 | 26.0629 | 0.9135 | 0.0724 | 18.9984 | 0.7631 | 0.1502 |
| lab | start9000 | 9000 | 26.0062 | 0.9082 | 0.0767 | 19.0329 | 0.7488 | 0.1558 |
| lab | start9000 | 12000 | 26.0707 | 0.9124 | 0.0722 | 19.0377 | 0.7510 | 0.1488 |
| lab | start9000 | 14000 | 26.2107 | 0.9135 | 0.0728 | 19.2379 | 0.7656 | 0.1505 |
| lab | start6000 | 6000 | 25.7734 | 0.9022 | 0.0842 | 18.9807 | 0.7451 | 0.1643 |
| lab | start6000 | 8000 | 26.0296 | 0.9111 | 0.0758 | 19.0639 | 0.7614 | 0.1527 |
| lab | start6000 | 9000 | 25.0503 | 0.9027 | 0.0835 | 17.5841 | 0.6917 | 0.2126 |
| lab | start6000 | 10000 | 26.0590 | 0.9124 | 0.0737 | 18.9955 | 0.7591 | 0.1536 |
| lab | start6000 | 11000 | 26.0409 | 0.9136 | 0.0730 | 18.9277 | 0.7602 | 0.1524 |

输出位置：

```text
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_15000_20260521/2026-05-21_02-45-33
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_start9000_15000_20260521/2026-05-21_17-13-07
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_correction_start6000_15000_20260521/2026-05-21_18-37-06
```

阶段性结论：start12000 最稳；start9000 可跑完但最终不如 start12000 稳；start6000 中途明显劣化并停止。

## 2026-05-21：attention/correction 延长训练

目标：验证 12000 HUGS + 3000 attention 后，继续 attention fine-tune 是否还能提升。

设置：

```text
dataset: NeuMan lab
source: HUGS12000 + 3000 xyz-only attention 的 final checkpoint
continuation: 再训练 3000 步
correction: xyz-only
human token pooling: mean pooling
注意: 不是严格 trainer resume，optimizer state 和 global step 没有完整连续继承
```

| 场景 | 实验设置 | 累计阶段 | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUMAN_PSNR | HUMAN_SSIM | HUMAN_LPIPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lab | HUGS12000 + 3000 xyz-only attention | 15000 final | 26.0527 | 0.9149 | 0.0705 | 18.9251 | 0.7609 | 0.1491 |
| lab | HUGS12000 + 4000 xyz-only attention | +1000 | 26.3764 | 0.9203 | 0.0652 | 19.2497 | 0.7711 | 0.1421 |
| lab | HUGS12000 + 5000 xyz-only attention | +2000 | 26.3918 | 0.9208 | 0.0645 | 19.2469 | 0.7705 | 0.1413 |
| lab | HUGS12000 + 6000 xyz-only attention | +3000 final | 26.4029 | 0.9210 | 0.0640 | 19.2466 | 0.7701 | 0.1408 |

输出位置：

```text
output/human_scene/neuman/lab/hugs_trimlp/anchor_attention_xyz_extend_from15000_plus3000_20260521/2026-05-21_19-49-01
```

阶段性结论：继续 fine-tune 主要改善 LPIPS / SSIM，HUMAN_PSNR 在 +1000 后基本持平。

## 2026-05-21 到 2026-05-22：SuGaR-HUGS 背景与 attention 对比

目标：比较原版 HUGS 背景与轻量 SuGaR-HUGS 背景作为 attention 起点时的表现。

设置：

```text
dataset: NeuMan lab
SuGaR-HUGS: 只对 scene/background Gaussians 加轻量 SuGaR-style regularization
attention correction: xyz-only
主要比较: SuGaR-HUGS 12000 / 15000, 以及 SuGaR-HUGS12000 + 3000/6000 attention
```

| 场景 | 实验设置 | pooling | 步数/阶段 | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUMAN_PSNR | HUMAN_SSIM | HUMAN_LPIPS |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lab | SuGaR-HUGS | - | 12000 final | 25.9340 | 0.9130 | 0.0721 | 18.8669 | 0.7588 | 0.1512 |
| lab | SuGaR-HUGS | - | 15000 final | 26.1260 | 0.9143 | 0.0707 | 18.9424 | 0.7606 | 0.1494 |
| lab | SuGaR-HUGS12000 + 3000 xyz-only attention | mean | final | 26.2388 | 0.9191 | 0.0659 | 19.1240 | 0.7689 | 0.1422 |
| lab | SuGaR-HUGS12000 + 6000 xyz-only attention | mean | final | 26.2989 | 0.9201 | 0.0648 | 19.1753 | 0.7697 | 0.1414 |
| lab | SuGaR-HUGS12000 + 3000 xyz-only attention | attention | final | 26.2398 | 0.9191 | 0.0660 | 19.1266 | 0.7688 | 0.1430 |

输出位置：

```text
output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_15000_lab_20260521/2026-05-21_21-21-45
output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_hugs_12000_lab_20260521_rerun/2026-05-21_22-55-25
output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_hugs12000_xyz_attention_plus3000_stage1_20260521/2026-05-21_23-34-40
output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_hugs12000_xyz_attention_plus3000_stage2_20260522/2026-05-22_00-05-47
output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_hugs12000_xyz_attentionpool_xyzonly_plus3000_20260522/2026-05-22_12-26-49
```

阶段性结论：SuGaR-HUGS12000 + attention 的第一个 3000 步提升明显；继续到 6000 步后 LPIPS 继续改善，但 PSNR/SSIM 接近平台。lab 上 pure HUGS 背景 + 6000 attention 的 full metrics 更高。

## 2026-05-22：纯 HUGS12000 + attention pooling 对照

目标：在完全原版 HUGS12000 起点上，比较 attention pooling 版本的 xyz-only attention。

设置：

```text
dataset: NeuMan lab
background: 原版 HUGS 12000
attention correction: xyz-only
human token pooling: attention pooling
attention steps: 3000
```

已保存输出：

```text
output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_12000_20260522/2026-05-22_13-30-32
output/human_scene/neuman/lab/hugs_trimlp/hugs12000_xyz_attentionpool_xyzonly_plus3000_20260522/2026-05-22_14-16-38
```

说明：该组目录当前保留 checkpoint、训练日志和 val 可视化文件，但未发现 `results_train.json`。对比时优先使用重新统一评估脚本补齐指标后，再纳入主结果表。

## 2026-05-22：parkinglot / seattle 跨场景验证

目标：验证当前 pipeline 是否能从 lab 扩展到不同场景。

共同设置：

```text
datasets: NeuMan parkinglot, NeuMan seattle
实验 1: 原版 HUGS 12000
实验 2: 原版 HUGS 15000
实验 3: 原版 HUGS12000 + 6000 xyz-only attention correction
实验 4: SuGaR-HUGS 12000
实验 5: SuGaR-HUGS12000 + 6000 xyz-only attention correction
human token pooling: attention pooling
记录点: 003000 表示 correction 第 3000 步；final 表示 correction 第 6000 步
```

远端汇总位置：

```text
gpu-l40-1:/tmp/ml-hugs-output/parkinglot_seattle_pipeline_20260522/summary.md
gpu-l40-1:/tmp/ml-hugs-output/parkinglot_seattle_pipeline_20260522/summary.json
gpu-l40-1:/tmp/ml-hugs-output/parkinglot_seattle_pipeline_20260522/runner.log
```

### parkinglot

| 场景 | 实验设置 | 阶段 | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUMAN_PSNR | HUMAN_SSIM | HUMAN_LPIPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| parkinglot | HUGS12000 | 003000 | 26.4396 | 0.7838 | 0.2263 | 19.7341 | 0.6758 | 0.2232 |
| parkinglot | HUGS12000 | final | 26.7238 | 0.8257 | 0.1550 | 19.3489 | 0.7032 | 0.1709 |
| parkinglot | HUGS15000 | 003000 | 26.4913 | 0.7874 | 0.2247 | 19.7445 | 0.6824 | 0.2170 |
| parkinglot | HUGS15000 | final | 26.6424 | 0.8347 | 0.1450 | 19.1697 | 0.7091 | 0.1616 |
| parkinglot | HUGS12000 + 6000 xyz-only attention | 003000 | 27.2721 | 0.8418 | 0.1390 | 19.9058 | 0.7257 | 0.1436 |
| parkinglot | HUGS12000 + 6000 xyz-only attention | final | 27.2009 | 0.8444 | 0.1352 | 19.7643 | 0.7260 | 0.1438 |
| parkinglot | SuGaR-HUGS12000 | 003000 | 26.3779 | 0.7859 | 0.2282 | 19.5552 | 0.6764 | 0.2209 |
| parkinglot | SuGaR-HUGS12000 | final | 26.6344 | 0.8246 | 0.1539 | 19.2717 | 0.7000 | 0.1702 |
| parkinglot | SuGaR-HUGS12000 + 6000 xyz-only attention | 003000 | 27.2752 | 0.8423 | 0.1365 | 19.8842 | 0.7242 | 0.1435 |
| parkinglot | SuGaR-HUGS12000 + 6000 xyz-only attention | final | 27.1805 | 0.8449 | 0.1320 | 19.6977 | 0.7227 | 0.1426 |

输出位置：

```text
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/parkinglot/hugs_trimlp/exp0_hugs_original_12000_parkinglot_20260522/2026-05-22_17-07-02
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/parkinglot/hugs_trimlp/exp0_hugs_original_15000_parkinglot_20260522/2026-05-22_17-49-15
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/parkinglot/hugs_trimlp/hugs12000_parkinglot_xyz_attentionpool_xyzonly_plus6000_20260522/2026-05-22_18-32-31
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/parkinglot/hugs_trimlp/scene_sugar_hugs_12000_parkinglot_20260522/2026-05-22_19-15-09
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/parkinglot/hugs_trimlp/sugarhugs12000_parkinglot_xyz_attentionpool_xyzonly_plus6000_20260522/2026-05-22_19-50-40
```

### seattle

| 场景 | 实验设置 | 阶段 | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUMAN_PSNR | HUMAN_SSIM | HUMAN_LPIPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seattle | HUGS12000 | 003000 | 24.4765 | 0.7706 | 0.1865 | 18.6657 | 0.6083 | 0.1992 |
| seattle | HUGS12000 | final | 25.7050 | 0.8300 | 0.1122 | 18.9170 | 0.6541 | 0.1425 |
| seattle | HUGS15000 | 003000 | 24.6444 | 0.7734 | 0.1803 | 18.4851 | 0.6059 | 0.1985 |
| seattle | HUGS15000 | final | 26.0878 | 0.8426 | 0.1044 | 18.7852 | 0.6618 | 0.1389 |
| seattle | HUGS12000 + 6000 xyz-only attention | 003000 | 26.2260 | 0.8450 | 0.1004 | 19.3294 | 0.6797 | 0.1330 |
| seattle | HUGS12000 + 6000 xyz-only attention | final | 26.2756 | 0.8477 | 0.0964 | 19.2921 | 0.6771 | 0.1317 |
| seattle | SuGaR-HUGS12000 | 003000 | 24.5716 | 0.7698 | 0.1853 | 18.5136 | 0.6007 | 0.1994 |
| seattle | SuGaR-HUGS12000 | final | 25.7089 | 0.8271 | 0.1128 | 18.9410 | 0.6529 | 0.1449 |
| seattle | SuGaR-HUGS12000 + 6000 xyz-only attention | 003000 | 26.2566 | 0.8432 | 0.1007 | 19.4184 | 0.6756 | 0.1332 |
| seattle | SuGaR-HUGS12000 + 6000 xyz-only attention | final | 26.3009 | 0.8462 | 0.0971 | 19.3517 | 0.6752 | 0.1316 |

输出位置：

```text
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/seattle/hugs_trimlp/exp0_hugs_original_12000_seattle_20260522/2026-05-22_20-35-33
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/seattle/hugs_trimlp/exp0_hugs_original_15000_seattle_20260522/2026-05-22_21-10-21
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/seattle/hugs_trimlp/hugs12000_seattle_xyz_attentionpool_xyzonly_plus6000_20260522/2026-05-22_21-55-59
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/seattle/hugs_trimlp/scene_sugar_hugs_12000_seattle_20260522/2026-05-22_22-49-30
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/seattle/hugs_trimlp/sugarhugs12000_seattle_xyz_attentionpool_xyzonly_plus6000_20260522/2026-05-22_23-25-26
```

阶段性结论：parkinglot 和 seattle 上，HUGS12000 + xyz-only attention correction 均明显优于对应的纯 HUGS12000/15000。SuGaR-HUGS 起点接 attention correction 与纯 HUGS 起点接 correction 很接近，不能只凭这轮判断 SuGaR-HUGS 一定更优。

## 2026-05-26：transl correction 消融

目标：验证新讨论的 canonical-to-world / 人体整体位置 correction 是否能接入 anchor-attention correction，并比较 `transl-only` 与 `transl+xyz` 在 3000-step correction 下的效果。

共同设置：

```text
dataset: NeuMan lab
source checkpoint: 原版 HUGS 12000
attention steps: 3000
human token pooling: attention pooling
scene_global_aware_adc: false
scene_sugar: false
correction_start_iter: 0
warmup: 0
```

实验设置：

```text
P1 transl-only:
  correct_xyz=false
  correct_transl=true
  gamma_transl=0.05
  transl_delta_clamp=0.2

P3 transl+xyz:
  correct_xyz=true
  correct_transl=true
  gamma_mu=0.005
  gamma_transl=0.05
  transl_delta_clamp=0.2
```

| 场景 | 实验设置 | 步数/阶段 | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUMAN_PSNR | HUMAN_SSIM | HUMAN_LPIPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lab | 原版 HUGS | 12000 checkpoint | 25.9258 | 0.9127 | 0.0719 | 18.7633 | 0.7488 | 0.1537 |
| lab | 原版 HUGS | 15000 final | 25.9582 | 0.9147 | 0.0710 | 18.8005 | 0.7580 | 0.1532 |
| lab | HUGS12000 + 3000 xyz-only attention | final | 26.0527 | 0.9149 | 0.0705 | 18.9251 | 0.7609 | 0.1491 |
| lab | HUGS12000 + 3000 transl-only attention | final | 26.2609 | 0.9187 | 0.0666 | 19.1296 | 0.7689 | 0.1430 |
| lab | HUGS12000 + 3000 transl+xyz attention | final | 26.2644 | 0.9187 | 0.0666 | 19.1331 | 0.7686 | 0.1428 |

输出位置：

```text
output/human_scene/neuman/lab/hugs_trimlp/transl_only_attention_plus3000_lab_20260526/2026-05-26_01-17-55
output/human_scene/neuman/lab/hugs_trimlp/transl_xyz_attention_plus3000_lab_20260526/2026-05-26_02-20-24
```

Debug 观察：

```text
P1 iter1000 delta_transl ~= [-0.1437, 0.0757, 0.1886]
P1 iter2000 delta_transl hits clamp ~= [-0.2, 0.2, 0.2]
P3 iter1000 delta_transl ~= [-0.0934, 0.0793, 0.1269], delta_mu_mean_norm ~= 0.2829
P3 iter2000 delta_transl hits clamp ~= [-0.2, 0.2, 0.2], delta_mu_mean_norm ~= 0.3872
```

阶段性结论：

1. `transl-only` 已经明显优于旧的 `xyz-only 3000`，说明 lab 当前问题中确实存在可由人体整体 world-space 对齐修正解释的一部分。
2. `transl+xyz` 只比 `transl-only` 略好，说明在这组超参下主要收益来自整体平移，而不是额外 xyz 微调。
3. `delta_transl` 在 2000 step 已触达 clamp，说明当前 `gamma_transl=0.05 / transl_delta_clamp=0.2` 设置偏保守但已饱和；后续应做更小/更大 clamp 与 gamma 的消融，或者加入 temporal/pose prior 后再放宽。
4. 该组是 3000-step correction 的公平对比；不能直接和已有 `HUGS12000 + 6000 attention` continuation 混合比较。

## 已做但未纳入主表的设置类消融

这些实验或设置在讨论/运行中出现过，但当前不是主推荐 baseline，或结果没有形成完整可复用主表。

| 时间 | 实验方向 | 设置摘要 | 当前状态 |
| --- | --- | --- | --- |
| 2026-05-21 | xyz + opacity correction | 在 xyz-only 基础上额外修正人体 opacity | 已做过对照，当前记录显示 final 略弱于 xyz-only，未作为主线 |
| 2026-05-21 | 更多 correction 参数组合 | 讨论过 xyz + opacity、xyz + scale/shape、xyz + color 等方向 | 当前主线仍是 xyz-only；多参数组合需要更强 regularization 和单独表格 |
| 2026-05-22 | mean pooling vs attention pooling | lab 上 SuGaR-HUGS 起点已比较，二者非常接近 | attention pooling 可用于跨场景主线，但 lab 上没有显著超过 mean pooling |
| 2026-05-22 | 纯 HUGS12000 + attention pooling | 已有输出目录，但未找到结果 JSON | 需要补跑统一评估后再加入主指标表 |

## 当前推荐对照组

后续如果继续做新初始化、新 token fusion 或新 correction 参数，建议至少保留以下对照：

```text
1. 原版 HUGS 15000
2. 原版 HUGS 12000
3. 原版 HUGS12000 + 3000 xyz-only attention
4. 原版 HUGS12000 + 6000 xyz-only attention
5. SuGaR-HUGS 12000 / 15000
6. SuGaR-HUGS12000 + 3000 / 6000 xyz-only attention
```

当前最稳定的工作假设：

```text
先训练 HUGS 或 SuGaR-HUGS 到 12000 步，
再对人体 Gaussian xyz 做 3000 到 6000 步 anchor-attention correction。
```

## 2026-05-25/26 lab - Global-aware Voxelized ADC + xyz-only attention

实验目的：验证新的 Global-aware Voxelized ADC 背景重建方案作为 HUGS 背景/人体联合重建起点后，继续接入后续 anchor-attention xyz-only correction 的效果。

实验设置：

```text
场景: lab
起点: Global-aware Voxelized ADC + HUGS 12000
后续 correction: xyz-only anchor-attention, attention pooling
阶段: +3000 stage1, 再 +3000 stage2
opacity/scale/color correction: disabled
scene_global_aware_adc.enabled: false during attention stages
scene_sugar.enabled: false
scene GS: 1138.0K
human GS: 110.2K
```

| 场景 | 实验设置 | 阶段 | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUMAN_PSNR | HUMAN_SSIM | HUMAN_LPIPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lab | ADC-HUGS12000 | final | 25.8364 | 0.9082 | 0.0794 | 19.2437 | 0.7594 | 0.1472 |
| lab | ADC-HUGS12000 + 3000 xyz-only attention | 001000 | 26.2303 | 0.9169 | 0.0696 | 19.4142 | 0.7697 | 0.1431 |
| lab | ADC-HUGS12000 + 3000 xyz-only attention | 002000 | 26.2732 | 0.9179 | 0.0681 | 19.4061 | 0.7693 | 0.1408 |
| lab | ADC-HUGS12000 + 3000 xyz-only attention | final | 26.3161 | 0.9185 | 0.0671 | 19.4328 | 0.7696 | 0.1396 |
| lab | ADC-HUGS12000 + 6000 xyz-only attention | 001000 | 26.3965 | 0.9198 | 0.0660 | 19.5060 | 0.7718 | 0.1395 |
| lab | ADC-HUGS12000 + 6000 xyz-only attention | 002000 | 26.3883 | 0.9198 | 0.0658 | 19.4723 | 0.7709 | 0.1384 |
| lab | ADC-HUGS12000 + 6000 xyz-only attention | final | 26.3999 | 0.9199 | 0.0656 | 19.4782 | 0.7702 | 0.1384 |

输出位置：

```text
gpu-l40-1:/hdd/u202420081000003/ml-hugs/output/human_scene/neuman/lab/hugs_trimlp/global_aware_voxel_adc_12000_lab_20260525/2026-05-25_08-31-58
gpu-l40-1:/hdd/u202420081000003/ml-hugs/output/human_scene/neuman/lab/hugs_trimlp/global_adc12000_xyz_attention_plus3000_stage1_20260525/2026-05-25_21-42-34
gpu-l40-1:/hdd/u202420081000003/ml-hugs/output/human_scene/neuman/lab/hugs_trimlp/global_adc12000_xyz_attention_plus3000_stage2_20260525/2026-05-25_23-29-01
```

阶段性结论：ADC-HUGS12000 接 xyz-only attention 后明显提升。相对 ADC-HUGS12000 final，6000 步 attention final 提升约 +0.5635 PSNR、+0.0117 SSIM、-0.0138 LPIPS；human 区域提升约 +0.2345 PSNR、+0.0108 SSIM、-0.0088 LPIPS。stage2 相比 stage1 仍有小幅增益，其中 stage2@1000 的 human PSNR/SSIM 最好，stage2 final 的 full PSNR/SSIM/LPIPS 最好。

## 2026-05-25 lab - Global-aware Voxelized ADC background-only

实验目的：先不启用 anchor-attention correction，只验证 Global-aware Voxelized ADC 背景重建方案本身作为 HUGS scene branch 替代路径的效果。

实验设置：

```text
场景: lab
方案: Global-aware Voxelized ADC + HUGS
attention correction: disabled
scene_sugar.enabled: false
scene_global_aware_adc.enabled: true
```

| 场景 | 实验设置 | 步数 | Scene GS | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUMAN_PSNR | HUMAN_SSIM | HUMAN_LPIPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lab | Global-aware Voxelized ADC + HUGS | 12000 | 1,138,038 | 25.8364 | 0.9082 | 0.0794 | 19.2437 | 0.7594 | 0.1472 |
| lab | Global-aware Voxelized ADC + HUGS | 15000 | 1,236,705 | 25.7032 | 0.9071 | 0.0809 | 19.0630 | 0.7549 | 0.1530 |

参考对照：

| 场景 | 实验设置 | 步数 | Scene GS | HUGS_PSNR | HUGS_SSIM | HUGS_LPIPS | HUMAN_PSNR | HUMAN_SSIM | HUMAN_LPIPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lab | 原版 HUGS | 15000 | 2,136,961 | 25.9582 | 0.9147 | 0.0710 | 18.8005 | 0.7580 | 0.1532 |
| lab | SuGaR-HUGS | 15000 | 2,156,810 | 26.1260 | 0.9143 | 0.0707 | 18.9424 | 0.7606 | 0.1494 |

输出位置：

```text
gpu-l40-1:/hdd/u202420081000003/ml-hugs/output/human_scene/neuman/lab/hugs_trimlp/global_aware_voxel_adc_12000_lab_20260525/2026-05-25_08-31-58
gpu-l40-1:/hdd/u202420081000003/ml-hugs/output/human_scene/neuman/lab/hugs_trimlp/global_aware_voxel_adc_15000_lab_20260525/2026-05-25_09-18-26
```

阶段性结论：ADC12000 优于 ADC15000，且 scene GS 数量明显低于原版 HUGS/SuGaR 15000。虽然 ADC-only full-image 指标低于 HUGS/SuGaR 15000，但 compact scene 表示和较好的 human 区域指标使 ADC12000 更适合作为后续 xyz-only attention correction 起点。

## 2026-05-26 lab - transl+xyz correction across background schemes

实验目的：在已有三种背景重建起点上，验证 `transl + xyz` attention correction 的兼容性和步数收益。这里的 `transl` 表示对人体整体 world-space 平移的 correction，`xyz` 表示对人体 Gaussian 位置的 correction；不修正 opacity/scale/color/LBS。

排队脚本：

```text
/hdd/u202420081000003/ml-hugs/run_logs/run_lab_transl_xyz_bg_queue_20260526.sh
/hdd/u202420081000003/ml-hugs/run_logs/lab_transl_xyz_bg_queue_20260526.log
```

共同设置：

```text
dataset: lab
human_pooling: attention
correct_xyz: true
correct_transl: true
gamma_mu: 0.005
gamma_transl: 0.05
transl_delta_clamp: 0.2
correct_opacity / scale / feature_dc: false
val_interval: 1000
save_progress_images: true
```

计划/队列：

| 场景 | 背景起点 | correction 步数 | exp_name | 状态 |
|---|---|---:|---|---|
| lab | 原版 HUGS 12000 | 5998 | `hugs12000_transl_xyz_attention_plus6000_lab_20260526` | queued, waiting for GPU memory <= 12000 MiB |
| lab | SuGaR-HUGS 12000 | 2998 | `sugarhugs12000_transl_xyz_attention_plus3000_lab_20260526` | queued |
| lab | ADC-HUGS 12000 | 2998 | `adc12000_transl_xyz_attention_plus3000_lab_20260526` | queued |

起点 checkpoint：

```text
原版 HUGS12000:
output/human_scene/neuman/lab/hugs_trimlp/exp0_hugs_original_12000_20260522/2026-05-22_13-30-32/ckpt/{human,scene}_011998.pth

SuGaR-HUGS12000:
output/human_scene/neuman/lab/hugs_trimlp/scene_sugar_hugs_12000_lab_20260521_rerun/2026-05-21_22-55-25/ckpt/{human,scene}_011998.pth

ADC-HUGS12000:
output/human_scene/neuman/lab/hugs_trimlp/global_aware_voxel_adc_12000_lab_20260525/2026-05-25_08-31-58/ckpt/{human,scene}_011998.pth
```

备注：启动时 `gpu-l40-1` 上存在其他用户进程和约 24GB 显存占用，本脚本先等待，不抢占 GPU。

