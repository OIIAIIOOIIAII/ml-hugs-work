# Anchor Attention 当前 Pipeline 总结

本文档根据当前实验记录整理 Anchor Attention baseline 的 pipeline、各模块实现程度、实现方法与当前实验结论。

## 1. 当前主线

当前 Anchor Attention pipeline 可以概括为：

```text
HUGS / SuGaR-HUGS background-human training
-> human anchors / anchor bindings
-> anchor token encoding
-> scene Gaussian KNN query
-> anchor-scene cross attention
-> human Gaussian xyz correction
-> render loss 反传优化
-> validation metrics / debug / 可视化
```

当前最稳定、正式在用的是：

```text
前 12000 步：原版 HUGS 或 SuGaR-HUGS 正常训练
后 3000 步：开启 anchor-attention xyz-only correction
可选再 +3000 步：继续 attention/correction fine-tune
```

也就是：

```text
HUGS background:
12000 HUGS -> +3000 xyz attention -> +3000 xyz attention

SuGaR-HUGS background:
12000 SuGaR-HUGS -> +3000 xyz attention -> +3000 xyz attention
```

## 2. Background / Scene Reconstruction

实现程度：已跑通两条背景路线。

### HUGS 原版背景

使用原始 HUGS human_scene 训练流程，作为 baseline。

### SuGaR-HUGS 背景

已实现轻量 SuGaR-style scene regularization，只作用于背景/scene Gaussians，不作用于人体 Gaussians。

当前实现包含：

```text
opacity regularization
scale / flatten regularization
low-opacity pruning
scene_sugar_debug PLY 导出
```

对应代码主要是：

```text
hugs/utils/scene_sugar_regularizer.py
hugs/trainer/gs_trainer.py
hugs/cfg/config.py
```

当前不是完整 SuGaR mesh pipeline，而是把 SuGaR 的“让背景高斯更贴近表面、更少 floaters”的思想接进 HUGS scene 分支。

## 3. Human Anchors / Anchor Binding

实现程度：已实现并用于训练。

方法：基于人体 SMPL / human Gaussian 与预定义 anchor 的绑定关系，得到每个 anchor 对应的人体局部 Gaussian group。训练中会得到 `anchor_world`，作为人体-场景交互查询的空间起点。

早期已经完成：

```text
anchor debug
anchor-local Gaussian statistics
anchor-to-scene nearest neighbor visualization
```

当前训练中的 anchor binding 已能支撑后续 token 和 attention。

## 4. Anchor Token Encoder

实现程度：已实现。

方法：对每个 anchor 绑定的人体 Gaussian 做 feature encoding 和 pooling，形成 anchor-level human token。

支持两种 pooling：

```text
mean pooling
attention pooling
```

attention pooling 使用 anchor embedding 作为 query，并带局部 bias MLP。当前主线可用，但正式质量实验主要关注 correction pipeline 是否能通过 render loss 带来收益，token fusion 还不是最终形态。

核心代码：

```text
hugs/models/anchor_attention.py
AnchorSceneAttentionBaseline
```

## 5. Scene Gaussian Query

实现程度：已实现并用于训练。

方法：每个 human anchor 在 scene Gaussians 中做局部 KNN / topK 查询，得到附近 scene candidates，再经过 scene token encoder。

当前实现方式：

```text
opacity filter
torch.cdist
topK scene query
scene local token MLP
```

debug 中会保存：

```text
scene_knn_idx
scene_knn_dist
nearest distance
attention weighted distance
scene opacity / scale stats
```

当前性能瓶颈也在这里：scene Gaussians 大约 200 万级，correction 阶段速度会从约 `4.7-5.1 it/s` 降到 `1.6-2.0 it/s`。后续需要 cache、局部 subset 或更强采样策略。

## 6. Anchor-Scene Cross Attention

实现程度：已实现。

方法：每个 anchor 的 human token 作为 query，附近 scene Gaussian tokens 作为 key/value，做 per-anchor cross attention。

输出：

```text
context: [A, D]
attention_weights: [A, K]
```

早期 dry-run 已验证 attention shape、无 NaN、能保存 pt/csv debug。当前正式 pipeline 里 attention 不作为接触监督，也不直接当接触质量指标，而是作为 context aggregation 模块，服务后面的 correction head。

## 7. Human Gaussian Correction

实现程度：xyz-only correction 已跑通并作为当前主 baseline；opacity correction 做过一次对照，但不是当前最优主线。

当前主线：

```text
correct_opacity = false
只修正 human Gaussian xyz / mean / center
不直接修正 scene Gaussian
不直接修正 human scale / rotation / SH / color
```

方法：cross-attention 得到 context 后，correction head 输出 `delta_mu`，对 human Gaussian center 做小幅修正，再进入 renderer。这个修正通过 render loss 学习。

关键稳定策略：

```text
module_start_iter = 12000
correction_start_iter = 12000
correction_warmup_iters = 1000
zero_init_delta = true
gamma_mu = 0.005 或 continuation 中 0.003
delta_loss_w = 0.001 / 0.002
```

`zero_init_delta` 很重要：correction 刚开启时接近原 HUGS 行为，避免突然破坏已收敛表示。

### Opacity correction

已在 SuGaR-HUGS 12000 checkpoint 上跑过 `xyz+opacity` 对照，但 final 略弱于 xyz-only，因此当前更推荐 xyz-only。

## 8. Training Schedule

当前最稳策略：

```text
0-12000:
  原版 HUGS / SuGaR-HUGS 正常优化
  不执行 attention/correction

12000-15000:
  开启 anchor-attention correction
  xyz-only
  render loss 优化

15000-18000:
  可选 continuation
  从 final checkpoint 继续 fine-tune attention/correction
```

start time 消融结论：

```text
start12000 最稳
start9000 可以跑完，但 final 不如 start12000 稳
start6000 中途出现明显劣化，已作为负向消融
```

所以当前推荐 pipeline 是“先让 HUGS/SuGaR-HUGS 收敛，再在后段小幅 correction”。

## 9. Evaluation / Debug

实现程度：已比较完整。

已有输出包括：

```text
results_train.json
train progress png / mp4
validation full / human crop images
render_all video
canonical videos
anchor_attention_debug/*.pt
anchor_attention_debug/*.csv
anchor_attention_final.pth
```

debug csv/pt 中记录 attention、scene KNN、距离、entropy、delta 等信息。另有 probe diagnostic、contact eval、overlay 可视化，但当前质量结论主要还是 validation PSNR/SSIM/LPIPS。

## 10. 当前实验结论

### HUGS 背景下

```text
12000 -> 15000 attention:
PSNR +0.1269
SSIM +0.0022
LPIPS -0.0014
Human PSNR +0.1618
Human SSIM +0.0121
Human LPIPS -0.0046

继续 15000 -> 18000:
PSNR +0.3502
SSIM +0.0061
LPIPS -0.0065
Human LPIPS -0.0083
```

### SuGaR-HUGS 背景下

```text
12000 -> 15000 attention:
PSNR +0.3049
SSIM +0.0061
LPIPS -0.0061
Human PSNR +0.2572
Human SSIM +0.0101
Human LPIPS -0.0090
```

这说明更干净的 SuGaR-HUGS 背景让第一个 3000-step attention 收益更明显。第二个 3000-step 在 SuGaR-HUGS 上继续小幅改善 LPIPS，但 PSNR/SSIM 基本进入平台期；而 HUGS 背景的 +6000 attention 最终 full metrics 更高。

### parkinglot / seattle 跨场景验证

实验时间：2026-05-22  
实验机器：`gpu-l40-1`  
远端总汇总目录：`gpu-l40-1:/tmp/ml-hugs-output/parkinglot_seattle_pipeline_20260522`

本轮实验用于验证当前 anchor-attention correction pipeline 是否能从 lab 扩展到 `parkinglot` 和 `seattle`。每个场景串行运行以下设置：

1. 纯原版 HUGS 12000 步。
2. 纯原版 HUGS 15000 步。
3. 纯原版 HUGS 12000 步 + xyz-only attention correction 6000 步，并记录 attention 第 3000 步和最终第 6000 步。
4. Sugar-HUGS 12000 步。
5. Sugar-HUGS 12000 步 + xyz-only attention correction 6000 步，并记录 attention 第 3000 步和最终第 6000 步。

当前 correction 设置：只优化人体高斯 `xyz`，不优化 opacity、scale、rotation、SH/color。token pooling 使用 attention pooling。这里的 `003000` 表示 correction 阶段的第 3000 步；`final` 表示 correction 阶段第 6000 步。

远端汇总文件：

```text
gpu-l40-1:/tmp/ml-hugs-output/parkinglot_seattle_pipeline_20260522/summary.md
gpu-l40-1:/tmp/ml-hugs-output/parkinglot_seattle_pipeline_20260522/summary.json
gpu-l40-1:/tmp/ml-hugs-output/parkinglot_seattle_pipeline_20260522/runner.log
```

#### parkinglot 结果

| 实验 | HUGS PSNR | HUGS SSIM | HUGS LPIPS | HUMAN PSNR | HUMAN SSIM | HUMAN LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HUGS12000 003000 | 26.4396 | 0.7838 | 0.2263 | 19.7341 | 0.6758 | 0.2232 |
| HUGS12000 final | 26.7238 | 0.8257 | 0.1550 | 19.3489 | 0.7032 | 0.1709 |
| HUGS15000 003000 | 26.4913 | 0.7874 | 0.2247 | 19.7445 | 0.6824 | 0.2170 |
| HUGS15000 final | 26.6424 | 0.8347 | 0.1450 | 19.1697 | 0.7091 | 0.1616 |
| HUGS12000+6000xyzAttention 003000 | 27.2721 | 0.8418 | 0.1390 | 19.9058 | 0.7257 | 0.1436 |
| HUGS12000+6000xyzAttention final | 27.2009 | 0.8444 | 0.1352 | 19.7643 | 0.7260 | 0.1438 |
| SugarHUGS12000 003000 | 26.3779 | 0.7859 | 0.2282 | 19.5552 | 0.6764 | 0.2209 |
| SugarHUGS12000 final | 26.6344 | 0.8246 | 0.1539 | 19.2717 | 0.7000 | 0.1702 |
| SugarHUGS12000+6000xyzAttention 003000 | 27.2752 | 0.8423 | 0.1365 | 19.8842 | 0.7242 | 0.1435 |
| SugarHUGS12000+6000xyzAttention final | 27.1805 | 0.8449 | 0.1320 | 19.6977 | 0.7227 | 0.1426 |

parkinglot 输出目录：

```text
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/parkinglot/hugs_trimlp/exp0_hugs_original_12000_parkinglot_20260522/2026-05-22_17-07-02
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/parkinglot/hugs_trimlp/exp0_hugs_original_15000_parkinglot_20260522/2026-05-22_17-49-15
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/parkinglot/hugs_trimlp/hugs12000_parkinglot_xyz_attentionpool_xyzonly_plus6000_20260522/2026-05-22_18-32-31
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/parkinglot/hugs_trimlp/scene_sugar_hugs_12000_parkinglot_20260522/2026-05-22_19-15-09
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/parkinglot/hugs_trimlp/sugarhugs12000_parkinglot_xyz_attentionpool_xyzonly_plus6000_20260522/2026-05-22_19-50-40
```

#### seattle 结果

| 实验 | HUGS PSNR | HUGS SSIM | HUGS LPIPS | HUMAN PSNR | HUMAN SSIM | HUMAN LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HUGS12000 003000 | 24.4765 | 0.7706 | 0.1865 | 18.6657 | 0.6083 | 0.1992 |
| HUGS12000 final | 25.7050 | 0.8300 | 0.1122 | 18.9170 | 0.6541 | 0.1425 |
| HUGS15000 003000 | 24.6444 | 0.7734 | 0.1803 | 18.4851 | 0.6059 | 0.1985 |
| HUGS15000 final | 26.0878 | 0.8426 | 0.1044 | 18.7852 | 0.6618 | 0.1389 |
| HUGS12000+6000xyzAttention 003000 | 26.2260 | 0.8450 | 0.1004 | 19.3294 | 0.6797 | 0.1330 |
| HUGS12000+6000xyzAttention final | 26.2756 | 0.8477 | 0.0964 | 19.2921 | 0.6771 | 0.1317 |
| SugarHUGS12000 003000 | 24.5716 | 0.7698 | 0.1853 | 18.5136 | 0.6007 | 0.1994 |
| SugarHUGS12000 final | 25.7089 | 0.8271 | 0.1128 | 18.9410 | 0.6529 | 0.1449 |
| SugarHUGS12000+6000xyzAttention 003000 | 26.2566 | 0.8432 | 0.1007 | 19.4184 | 0.6756 | 0.1332 |
| SugarHUGS12000+6000xyzAttention final | 26.3009 | 0.8462 | 0.0971 | 19.3517 | 0.6752 | 0.1316 |

seattle 输出目录：

```text
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/seattle/hugs_trimlp/exp0_hugs_original_12000_seattle_20260522/2026-05-22_20-35-33
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/seattle/hugs_trimlp/exp0_hugs_original_15000_seattle_20260522/2026-05-22_21-10-21
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/seattle/hugs_trimlp/hugs12000_seattle_xyz_attentionpool_xyzonly_plus6000_20260522/2026-05-22_21-55-59
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/seattle/hugs_trimlp/scene_sugar_hugs_12000_seattle_20260522/2026-05-22_22-49-30
gpu-l40-1:/tmp/ml-hugs-output/human_scene/neuman/seattle/hugs_trimlp/sugarhugs12000_seattle_xyz_attentionpool_xyzonly_plus6000_20260522/2026-05-22_23-25-26
```

阶段性观察：

- 在 `parkinglot` 和 `seattle` 上，`12000 HUGS + xyz-only attention correction` 的图像指标均明显好于对应的纯 HUGS 12000/15000，说明当前 correction pipeline 至少在跨场景上不是只对 lab 偶然有效。
- `003000` 到 `final` 的变化显示，继续 correction 到 6000 步通常可以改善 LPIPS/SSIM，但 HUMAN PSNR 不一定继续上升；这说明更长 correction 不是单调收益，后续需要保留中间 checkpoint 做选择。
- Sugar-HUGS 起点接 attention correction 的表现与纯 HUGS 起点接 correction 很接近，部分指标略好，部分略差；目前不能只凭这一轮判断 Sugar 初始化一定更优。
- 这些结果仍然只说明渲染指标提升；接触质量还需要后续引入更可靠的接触评估或人工/DECO 辅助可视化判断。

## 11. 尚未完成或还不是最终方法的部分

当前还没有把 attention 作为显式 contact supervision；没有直接优化 contact loss。

当前主实验还不是强 geometry-biased attention。geometry bias 设计在文档里，但尚未成为当前主 baseline。

Human3R 初始化也不是当前主线。当前 baseline 主要依托原版 HUGS 初始化或 SuGaR-HUGS 背景替换。

完整 SuGaR mesh extraction 也没有接入训练，只做了轻量 scene regularization 和后续白模/点云诊断。

## 12. 一句话总结

当前 pipeline 已经从“可视化/诊断 dry-run”推进到“可训练、可复现、有指标提升的 xyz-only render-loss anchor-attention correction baseline”。

它现在最可靠的用法是：

```text
HUGS 或 SuGaR-HUGS 先训 12000 步，
再用 anchor-scene attention 对人体 Gaussian center 做后 3000 到 6000 步的小幅 correction。
```
