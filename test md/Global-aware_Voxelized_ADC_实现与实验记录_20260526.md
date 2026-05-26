# Global-aware Voxelized ADC 实现与实验记录（2026-05-26）

本文档汇总 Global-aware Voxelized ADC 背景重建方案在 HUGS 项目中的实现位置、运行方式、已完成实验结果，以及和后续 anchor attention xyz-only correction 的衔接结论。

## 1. 目标与定位

Global-aware Voxelized ADC 是一个与原版 HUGS 背景重建、SuGaR 背景重建平行的第三套背景方案。

设计目标：

- 替代原版 HUGS 的 scene densification / pruning 流程，用于背景 Gaussian 优化。
- 保持对现有 HUGS、SuGaR、anchor attention correction 功能的兼容，不破坏默认行为。
- 作为后续 attention xyz-only correction 的初始化背景，先完成 12000 步 ADC + HUGS 重建，再继续 3000 + 3000 步 attention 优化。

当前实现默认关闭，只有配置中显式设置 `scene_global_aware_adc.enabled=true` 时才启用。

## 2. 主要实现文件

### 2.1 ADC 工具实现

文件：`hugs/utils/scene_global_aware_adc.py`

主要内容：

- 相机相对模糊度估计：根据相邻视角图像差异计算 per-camera blur score。
- Global-aware 权重：结合模糊度、viewspace gradient 和 visibility，对背景 Gaussian 的 densification 候选进行加权。
- Scene max viewspace gradient accumulator：按 Gaussian 维护跨 iteration 的最大 viewspace gradient。
- Voxelized ADC densification：将候选 Gaussian 按 voxel 分组，控制局部密度增长。
- 子 Gaussian opacity split：densify 时采用半透明度子节点，避免背景点数过快膨胀。
- Prune 逻辑：结合 opacity、screen size、world scale 和体素统计做背景清理。

关键入口函数：

- `estimate_scene_relative_blur_scores`
- `setup_scene_global_aware_adc`
- `init_scene_global_aware_adc_state`
- `accumulate_scene_global_aware_adc`
- `scene_global_aware_voxel_adc_densify_and_prune`

### 2.2 Trainer 接入

文件：`hugs/trainer/gs_trainer.py`

接入点：

- import ADC 工具：`gs_trainer.py:58-63`
- 初始化配置与状态：`gs_trainer.py:304-306`
- 每轮训练累计 ADC 统计：`gs_trainer.py:616`、`gs_trainer.py:771`
- scene densification 分支：`gs_trainer.py:937-966`

行为约束：

- 仅在 `scene_global_aware_adc.enabled=true` 且 `scene_sugar.enabled=false` 时接管 scene densification。
- human Gaussian、SMPL、anchor attention 模块不被 ADC 逻辑直接修改。
- 原版 HUGS 和 SuGaR 分支保持原有路径。
- attention correction 阶段默认关闭 ADC，只加载 ADC12000 作为初始化 checkpoint。

### 2.3 配置文件

默认配置入口：`hugs/cfg/config.py:207-247`

新增配置组：`cfg.scene_global_aware_adc`

关键字段：

- `enabled`
- `start_iter`
- `stop_iter`
- `densify_interval`
- `opacity_prune_interval`
- `gradient_threshold`
- `voxel_size`
- `max_points_per_voxel`
- `blur_weight`
- `visibility_weight`
- `screen_size_threshold`
- `world_size_threshold`

实验配置：

- ADC 背景方案：`cfg_files/release/neuman/hugs_global_aware_voxel_adc_15000_lab.yaml`
- ADC12000 + xyz-only attention：`cfg_files/release/neuman/hugs_global_adc_hugs12000_xyz_attention_plus3000_lab.yaml`

## 3. 兼容性说明

为了避免破坏现有功能，当前实现采用显式开关和互斥分支：

- 原版 HUGS：`scene_global_aware_adc.enabled=false`、`scene_sugar.enabled=false`
- SuGaR-HUGS：`scene_sugar.enabled=true`、`scene_global_aware_adc.enabled=false`
- ADC-HUGS：`scene_global_aware_adc.enabled=true`、`scene_sugar.enabled=false`
- ADC + attention：第一阶段使用 ADC checkpoint，后续 attention 训练中 `scene_global_aware_adc.enabled=false`

因此 ADC 只作为背景初始化/重建策略，不改变后续 anchor attention correction 的模块接口。

## 4. 已完成验证

### 4.1 静态与 smoke 验证

已完成：

- Python 编译检查。
- CPU 级别 smoke 检查。
- GPU 环境下短流程 smoke 检查。
- 完整 lab 场景训练验证。

GPU 机器：`gpu-l40-1`

## 5. 实验一：ADC 背景-only，lab 场景

目标：只验证新的背景重建方案，不开启后续 attention correction。

场景：`neuman/lab`

实验设置：

- `scene_global_aware_adc.enabled=true`
- `scene_sugar.enabled=false`
- 不开启 anchor attention correction
- 分别训练 12000 步和 15000 步

### 5.1 输出目录与日志

ADC12000：

`output/human_scene/neuman/lab/hugs_trimlp/global_aware_voxel_adc_12000_lab_20260525/2026-05-25_08-31-58`

ADC15000：

`output/human_scene/neuman/lab/hugs_trimlp/global_aware_voxel_adc_15000_lab_20260525/2026-05-25_09-18-26`

日志：

`run_logs/global_aware_voxel_adc_12000_15000_lab_20260525.log`

### 5.2 指标

| 方案 | 步数 | Scene GS | HUGS PSNR | HUGS SSIM | HUGS LPIPS | HUMAN PSNR | HUMAN SSIM | HUMAN LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ADC-HUGS | 12000 | 1,138,038 | 25.8364 | 0.9082 | 0.0794 | 19.2437 | 0.7594 | 0.1472 |
| ADC-HUGS | 15000 | 1,236,705 | 25.7032 | 0.9071 | 0.0809 | 19.0630 | 0.7549 | 0.1530 |
| HUGS 原版参考 | 15000 | 2,136,961 | 25.9582 | 0.9147 | 0.0710 | 18.8005 | 0.7580 | 0.1532 |
| SuGaR-HUGS 参考 | 15000 | 2,156,810 | 26.1260 | 0.9143 | 0.0707 | 18.9424 | 0.7606 | 0.1494 |

### 5.3 结论

- ADC12000 优于 ADC15000，因此后续 attention 实验使用 ADC12000 作为初始化。
- ADC 背景点数明显少于原版 HUGS / SuGaR，大约 1.14M vs 2.14M。
- ADC12000 的 HUMAN 指标优于原版 HUGS15000 和 SuGaR15000 的 human PSNR/LPIPS，但 full-image HUGS 指标仍低于 HUGS/SuGaR 参考方案。
- ADC-only 更适合作为 compact background initialization，再接后续 attention correction。

## 6. 实验二：ADC12000 + anchor attention xyz-only correction，lab 场景

目标：验证 ADC 背景初始化能否接入后续 anchor attention correction，并与 ADC-only 结果形成对比。

实验流程：

1. 前 12000 步：使用 ADC-HUGS 背景重建。
2. 后 3000 步 Stage1：从 ADC12000 checkpoint 继续训练，启用 anchor attention xyz-only correction。
3. 再 3000 步 Stage2：从 Stage1 checkpoint 继续训练，继续 xyz-only correction。

attention 设置：

- `human_pooling=attention`
- `anchor_attention.correct_opacity=false`
- `anchor_attention.correct_scale=false`
- `anchor_attention.correct_feature_dc=false`
- 仅保留 xyz correction
- attention 阶段关闭 ADC：`scene_global_aware_adc.enabled=false`
- attention 阶段关闭 SuGaR：`scene_sugar.enabled=false`

### 6.1 输出目录与日志

Stage1：

`output/human_scene/neuman/lab/hugs_trimlp/global_adc12000_xyz_attention_plus3000_stage1_20260525/2026-05-25_21-42-34`

Stage2：

`output/human_scene/neuman/lab/hugs_trimlp/global_adc12000_xyz_attention_plus3000_stage2_20260525/2026-05-25_23-29-01`

日志：

`run_logs/global_adc12000_xyz_attention_stage1_stage2_20260525.log`

### 6.2 Stage1 指标

| Checkpoint | HUGS PSNR | HUGS SSIM | HUGS LPIPS | HUMAN PSNR | HUMAN SSIM | HUMAN LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Stage1 001000 | 26.2303 | 0.9169 | 0.0696 | 19.4142 | 0.7697 | 0.1431 |
| Stage1 002000 | 26.2732 | 0.9179 | 0.0681 | 19.4061 | 0.7693 | 0.1408 |
| Stage1 final | 26.3161 | 0.9185 | 0.0671 | 19.4328 | 0.7696 | 0.1396 |

### 6.3 Stage2 指标

| Checkpoint | HUGS PSNR | HUGS SSIM | HUGS LPIPS | HUMAN PSNR | HUMAN SSIM | HUMAN LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Stage2 001000 | 26.3965 | 0.9198 | 0.0660 | 19.5060 | 0.7718 | 0.1395 |
| Stage2 002000 | 26.3883 | 0.9198 | 0.0658 | 19.4723 | 0.7709 | 0.1384 |
| Stage2 final | 26.3999 | 0.9199 | 0.0656 | 19.4782 | 0.7702 | 0.1384 |

### 6.4 相对 ADC12000 的收益

以 ADC12000 background-only 为起点，Stage2 final 的提升为：

| 指标组 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| HUGS full image | +0.5635 | +0.0117 | -0.0138 |
| HUMAN | +0.2345 | +0.0108 | -0.0088 |

### 6.5 结论

- ADC12000 可以顺利接入后续 anchor attention xyz-only correction。
- Stage2 final 是当前 full-image 综合最优结果：HUGS `26.3999 / 0.9199 / 0.0656`。
- Stage2 001000 是当前 human PSNR/SSIM 最优点：HUMAN `19.5060 / 0.7718 / 0.1395`。
- 对主表推荐使用 Stage2 final；如果单独讨论 human 指标，可以附注 Stage2 001000。

## 7. 当前推荐对比口径

主线对比建议：

| 方案 | 说明 | 推荐用途 |
|---|---|---|
| HUGS15000 | 原版背景重建参考 | baseline |
| SuGaR-HUGS15000 | SuGaR 背景重建参考 | stronger background baseline |
| ADC12000 | 新背景方案，不加 attention | ADC-only ablation |
| ADC12000 + xyz-only + 3000 + 3000 | 新背景方案接 attention | 当前主结果 |

当前主结果：

`global_adc12000_xyz_attention_plus3000_stage2_20260525/2026-05-25_23-29-01/final`

## 8. 已同步更新的文档

本记录之外，以下文档也已写入相关结果：

- `ANCHOR_ATTENTION_EXPERIMENT_TIMELINE.md`
- `test md/Global-aware_Voxelized_ADC应用于HUGS背景重建方案.md`

## 9. 后续建议

- 将 ADC12000 + xyz-only attention 的结果加入最终实验表。
- 若继续优化 ADC，可优先关注 ADC-only full-image 指标低于 SuGaR/HUGS 的问题。
- 若目标是 human 指标，可保留 Stage2 001000 checkpoint 作为补充比较点。
- 后续如扩展到其他场景，建议先复用 ADC12000 初始化，再做相同的 3000 + 3000 attention 续训。
