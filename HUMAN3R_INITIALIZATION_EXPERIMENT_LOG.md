# Human3R Initialization Experiment Log

本文档记录将 Human3R 引入 HUGS 初始化流程后的实验结果。后续每次实验应补充：

- 训练入口与关键配置
- 初始化来源
- densify / prune 策略
- 指标结果
- 可视化视频路径
- 现象和结论

## 2026-05-18 当前阶段结论

目前最有效的方向不是完全使用 Human3R 的相机、深度和人体参数替换 HUGS，而是：

```text
保留原版 NeuMan/HUGS 相机、图像监督、人体初始化；
只把 Human3R 场景点云通过 SMPL 对齐后作为轻量 scene prior 注入。
```

这样可以保留 Human3R 的两个优势：

1. 人体和场景的相对位置更合理。
2. 背景场景点云更整洁，乱飘杂点更少。

同时可以避免 Human3R 直接路线带来的两个主要问题：

1. Human3R 相机/深度与 HUGS 训练假设不完全匹配，直接替换会使前景和背景质量都下降。
2. Human3R 的 SMPL-X 到 HUGS SMPL 映射误差会污染人体初始化和人体外观学习。

## 实验结果表

说明：这里按用户要求加粗每个指标列中的“数值最高项”。需要注意，LPIPS 指标本身是越低越好，因此 LPIPS 的加粗项只表示数值最大，不表示质量最好。

| ID | 实验 | 初始化 | 步数 | 关键设置 | Overall PSNR | Overall SSIM | Overall LPIPS | Human PSNR | Human SSIM | Human LPIPS | 结论 |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| E00 | 原版 HUGS full-res baseline | NeuMan 原版 | 6000 | 原版流程 | 25.8792 | 0.9031 | 0.0838 | 19.0826 | 0.7477 | 0.1653 | 原版 6000 质量参考线 |
| E01 | 原版 HUGS Human3R 分辨率 baseline | NeuMan 原版，图像下采样到 Human3R 分辨率 | 6000 | 原版流程 | **26.3263** | **0.9208** | 0.0392 | 19.2890 | 0.7276 | 0.0843 | 分辨率不是质量下降主因 |
| E02 | 旧 Human3R 直接初始化 | Human3R camera/depth/SMPL 拟合 | 6000 | 旧转换流程 | 18.9311 | 0.6439 | 0.1806 | 15.2528 | 0.5131 | 0.1648 | 直接替换流程质量明显下降 |
| E03 | Human3R 1024 直接路线 | Human3R 1024 输出 + SAM mask + LCC | 1000 | Human3R camera/depth 路线 | 14.7826 | 0.6235 | **0.5896** | 10.9411 | 0.4827 | **0.6040** | 高分辨率不能单独解决问题，直接路线仍不稳定 |
| E04 | Hybrid 50k smoke | NeuMan 原版 + Human3R SMPL 对齐 scene prior 50k | 1000 | `scene.init_pcd_path` 注入 50k Human3R scene 点 | 22.9476 | 0.8413 | 0.2024 | 16.3763 | 0.6758 | 0.3597 | Human3R prior 过重时早期质量较差 |
| E05 | Hybrid 25k smoke | NeuMan 原版 + Human3R SMPL 对齐 scene prior 25k | 1000 | `scene.init_pcd_path` 注入 25k Human3R scene 点 | 23.4737 | 0.8419 | 0.1948 | 17.3733 | 0.6904 | 0.3180 | 25k 比 50k/100k 更稳 |
| E06 | Hybrid 100k smoke | NeuMan 原版 + Human3R SMPL 对齐 scene prior 100k | 1000 | `scene.init_pcd_path` 注入 100k Human3R scene 点 | 22.4651 | 0.8374 | 0.2071 | 15.6446 | 0.6759 | 0.3514 | Human3R 点云太重会拖累训练 |
| E07 | Hybrid 25k 6000，scene densify 到 6000 | NeuMan 原版 + Human3R scene prior 25k | 6000 | `scene.densify_until_iter=6000`，human 默认剪枝 | 24.5184 | 0.8851 | 0.1112 | 17.4330 | 0.7044 | 0.2631 | scene 过度增殖，human 被剪到过少，人体质量下降 |
| E08 | Hybrid 25k 6000，禁 human densify/prune，scene 到 3000 | NeuMan 原版 + Human3R scene prior 25k | 6000 | `scene.densify_until_iter=3000`; `human.densify_from_iter=10000`; `human.densify_until_iter=10000` | 25.3258 | 0.8943 | 0.1061 | 18.5122 | 0.7374 | 0.2270 | 当前 6000 步最稳，人体不会消失，但细节仍弱于原版 |
| E09 | Hybrid 25k 6000，human densify + no prune | NeuMan 原版 + Human3R scene prior 25k | 6000 | `scene.densify_until_iter=3000`; `human.densify_from_iter=3000`; `human.densify_until_iter=6000`; `human.prune_min_opacity=0.0` | 25.1099 | 0.8939 | 0.1082 | 18.2822 | 0.7319 | 0.2332 | 6000 步内启用 human densify 没有带来收益 |
| E10 | Hybrid 25k 15000，human densify + no prune | NeuMan 原版 + Human3R scene prior 25k | 15000 | `scene.densify_until_iter=3000`; `human.densify_from_iter=3000`; `human.densify_until_iter=12000`; `human.prune_min_opacity=0.0` | 25.7824 | 0.9046 | 0.0877 | 18.9393 | 0.7513 | 0.1572 | 已接近原版 6000 overall 质量，human LPIPS 优于原版 6000，但 human PSNR 仍略低 |
| E11 | Hybrid 25k 15000，禁 human densify | NeuMan 原版 + Human3R scene prior 25k | 15000 | `scene.densify_until_iter=3000`; `human.densify_from_iter=20000`; `human.densify_until_iter=20000` | 25.7375 | 0.9048 | 0.0883 | 18.8328 | 0.7489 | 0.1700 | Overall 接近 E10，但人体感知指标差于 E10，说明适度 human densify 有必要 |
| E12 | Hybrid 25k 15000，human densify cap200k | NeuMan 原版 + Human3R scene prior 25k | 15000 | `scene.densify_until_iter=3000`; `human.densify_from_iter=3000`; `human.densify_until_iter=12000`; `human.prune_min_opacity=0.0`; `human.max_n_gaussians=200000` | 26.0035 | 0.9047 | 0.0901 | **19.3210** | **0.7544** | 0.1709 | 当前最均衡 Hybrid：人体指标优于原版 15000，但 overall SSIM/LPIPS 弱于原版 15000 |
| E13 | 原版 HUGS full-res baseline | NeuMan 原版 | 15000 | 原版流程；同步数公平对照 | 25.6259 | 0.9108 | 0.0790 | 18.3661 | 0.7386 | 0.1870 | 同步数原版对照：overall SSIM/LPIPS 更好，但人体指标低于 E12 |
| E14 | Hybrid 25k 15000，放开 scene densify | NeuMan 原版 + Human3R scene prior 25k | 15000 | `scene.densify_until_iter=15000`; `human.densify_from_iter=3000`; `human.densify_until_iter=12000`; `human.prune_min_opacity=0.0`; `human.max_n_gaussians=200000` | 25.4616 | 0.9102 | 0.0770 | 18.1739 | 0.7450 | 0.1777 | 放开 scene densify 改善 overall LPIPS/SSIM，但人体质量低于 E12，说明 scene-human 竞争仍是核心问题 |

## 关键实验路径

### E08: 当前 6000 步最稳设置

```text
output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_hybrid_orig_plus_h3r_smpl25k_6000_nohprune_scene3000/2026-05-18_01-59-13
```

最终视频：

```text
output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_hybrid_orig_plus_h3r_smpl25k_6000_nohprune_scene3000/2026-05-18_01-59-13/render_all_neuman_lab_final.mp4
```

### E10: 当前 15000 步最强设置

```text
output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_hybrid_orig_plus_h3r_smpl25k_15000_hdens_noprune_scene3000/2026-05-18_02-23-25
```

最终视频：

```text
output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_hybrid_orig_plus_h3r_smpl25k_15000_hdens_noprune_scene3000/2026-05-18_02-23-25/render_all_neuman_lab_final.mp4
```

预览图：

```text
output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_hybrid_orig_plus_h3r_smpl25k_15000_hdens_noprune_scene3000/2026-05-18_02-23-25/render_all_preview.jpg
```

### E11: 禁 human densify 的 15000 步对照

```text
output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_hybrid_orig_plus_h3r_smpl25k_15000_nohprune_scene3000_rerun/2026-05-18_03-01-50
```

最终视频：

```text
output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_hybrid_orig_plus_h3r_smpl25k_15000_nohprune_scene3000_rerun/2026-05-18_03-01-50/render_all_neuman_lab_final.mp4
```

预览图：

```text
output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_hybrid_orig_plus_h3r_smpl25k_15000_nohprune_scene3000_rerun/2026-05-18_03-01-50/render_all_preview.jpg
```


### E12: 当前最均衡 15000 步设置（human densify cap200k）

```text
output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_hybrid_orig_plus_h3r_smpl25k_15000_hdens_cap200k_scene3000/2026-05-18_03-26-15
```

最终视频：

```text
output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_hybrid_orig_plus_h3r_smpl25k_15000_hdens_cap200k_scene3000/2026-05-18_03-26-15/render_all_neuman_lab_final.mp4
```

预览图：

```text
output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_hybrid_orig_plus_h3r_smpl25k_15000_hdens_cap200k_scene3000/2026-05-18_03-26-15/render_all_preview.jpg
```


### E13: 原版 HUGS 15000 步同步数对照

```text
output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_original_15000_vis/2026-05-18_13-39-52
```

最终视频：

```text
output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_original_15000_vis/2026-05-18_13-39-52/render_all_neuman_lab_final.mp4
```

预览图：

```text
output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_original_15000_vis/2026-05-18_13-39-52/render_all_preview.jpg
```


### E14: 放开 scene densify 到 15000 的 Hybrid 对照

```text
output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_hybrid_orig_plus_h3r_smpl25k_15000_hdens_cap200k_scene15000/2026-05-18_14-07-46
```

最终视频：

```text
output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_hybrid_orig_plus_h3r_smpl25k_15000_hdens_cap200k_scene15000/2026-05-18_14-07-46/render_all_neuman_lab_final.mp4
```

预览图：

```text
output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_hybrid_orig_plus_h3r_smpl25k_15000_hdens_cap200k_scene15000/2026-05-18_14-07-46/render_all_preview.jpg
```

## E10 命令记录

```bash
ssh gpu-l40-1 'source /home/u202420081000003/anaconda3/etc/profile.d/conda.sh && conda activate hugs && cd /hdd/u202420081000003/ml-hugs && CUDA_VISIBLE_DEVICES=0 python scripts/run_neuman_human_scene_noamass.py --seq lab --cfg-file cfg_files/release/neuman/hugs_human_scene.yaml --quick --quick-steps 15000 --exp-name neuman_lab_hybrid_orig_plus_h3r_smpl25k_15000_hdens_noprune_scene3000 scene.init_pcd_path=data/human3r_hugs/hybrid_init/lab_orig_plus_h3r_smpl25k_pcd.npz train.save_progress_images=true train.progress_save_interval=1000 scene.densify_from_iter=100 scene.densify_until_iter=3000 scene.densification_interval=100 human.densify_from_iter=3000 human.densify_until_iter=12000 human.prune_min_opacity=0.0'
```

## 观察

### 1. 分辨率不是主要矛盾

原版 HUGS 在 Human3R 分辨率下仍能跑出较好的结果，所以早期 Human3R 初始化质量下降不能简单归因于低分辨率。更主要的问题是 Human3R 直接路线改变了相机、深度、人体参数和数据几何假设。

### 2. Human3R 直接接管 HUGS 流程风险很高

Human3R 的强项是全局人-场景对齐，而不是直接替代 HUGS 的可优化人体初始化。直接路线会同时影响前景和背景，这与用户观察一致。

### 3. Human3R scene prior 必须轻量

25k Human3R scene 点比 50k 和 100k 更稳。点太多时，Human3R prior 会干扰 HUGS 自己的 scene densification 和优化。

### 4. scene 和 human 的竞争需要控制

E07 中 scene 增殖到约 1.1M，而 human 被剪枝到约 1 万级，导致人体质量明显下降。这说明 scene 容易解释掉前景区域，需要限制 scene densify，同时避免 human 被过早剪掉。

### 5. 15000 步可以明显恢复质量

E10 的 overall 指标已经接近原版 6000。人体指标中：

- Human SSIM: 0.7513，高于原版 full-res 6000 的 0.7477。
- Human LPIPS: 0.1572，优于原版 full-res 6000 的 0.1653。
- Human PSNR: 18.9393，略低于原版 full-res 6000 的 19.0826。

这说明路线是可行的，但还没有完全解决人体锐度问题。

### 6. human densify 不是根本坏点，过量 densify 才是问题

E11 禁用 human densify 后，人体高斯数量保持 110.2K，scene 保持约 114.6K。最终 overall 指标接近 E10，但 human LPIPS 从 E10 的 0.1572 变差到 0.1700，human PSNR 也从 18.9393 降到 18.8328。

这说明：

- 人体 densify 确实能补充一部分感知细节。
- 但 E10 中人体高斯增长到约 553K，存在过度增殖风险。
- 下一步不应完全关闭 human densify，而应限制其时段或上限。


### 7. E12 说明：有限额的人体 densify 是当前最好折中

E12 在 E10 基础上加入 `human.max_n_gaussians=200000`。最终 overall PSNR=26.0035、human PSNR=19.3210、human SSIM=0.7544，均超过原版 6000 步 baseline；human LPIPS=0.1709，仍略差于原版 0.1653 和 E10 0.1572。

这说明当前路线已经能保留 Human3R 的人-场景相对位置优势，并恢复到接近/部分超过原版的数值质量。下一步重点不是回到 Human3R 直接路线，而是在 Hybrid 路线上继续优化人体 perceptual 细节，例如更合理的 densify 停止时机、前景过滤和 human/scene loss 平衡。


### 8. 同步数公平比较：E12 vs E13

E13 补齐了原版 HUGS 15000 步 baseline。与 Hybrid E12 同为 15000 步时：

- E12 overall PSNR 更高：26.0035 vs E13 25.6259。
- E13 overall SSIM/LPIPS 更好：0.9108/0.0790 vs E12 0.9047/0.0901。
- E12 人体三个指标都更好：human PSNR 19.3210 vs 18.3661，human SSIM 0.7544 vs 0.7386，human LPIPS 0.1709 vs 0.1870。

因此同步数下不能说 Human3R Hybrid 初始化显然不如原版。更准确的判断是：Hybrid E12 保住并改善了人体区域，同时全图感知质量/背景细节仍弱于原版 15000。原因很可能不是人体 SMPL-X 到 SMPL 映射本身，而是 Human3R scene prior 与 HUGS scene 优化之间的几何/颜色先验差异，让背景在 LPIPS/SSIM 上吃亏。


### 9. E14 说明：放开背景 densify 会重新引入人体竞争

E14 的目的，是验证 E12 的 overall SSIM/LPIPS 弱于原版，是否主要因为 `scene.densify_until_iter=3000` 让背景容量不足。实验把 scene densify 放开到 15000，其他保持与 E12 接近。

结果：

- E14 overall LPIPS=0.0770，略优于原版 E13 的 0.0790，也优于 E12 的 0.0901。
- E14 overall SSIM=0.9102，接近原版 E13 的 0.9108，高于 E12 的 0.9047。
- 但 E14 human PSNR=18.1739、human LPIPS=0.1777，明显弱于 E12 的 19.3210/0.1709，也弱于 E12 的 human SSIM=0.7544。

训练日志里还出现了关键现象：scene 高斯增长到约 2.15M，human 高斯在训练中段曾从接近 200K 被压到约 57.5K，最终回升到约 112.8K。这说明 E12 的问题不是单纯“背景点不够”，而是背景容量过强时会重新解释前景区域，导致人体细节被 scene 分走。

当前判断：

```text
E12 是更适合作为 Human3R 初始化路线的平衡点；
E14 证明长时间 scene densify 能恢复 overall perceptual 指标，但会牺牲人体。
下一步不应直接采用 scene densify 到 15000，而应寻找中间策略，例如 scene densify 到 7000/9000，或者给 scene 设置上限/前景排斥区。
```

### 10. checkpoint 清理记录

为给后续实验腾出空间，已删除一批早期 smoke/direct/旧路线实验的 `ckpt` 子目录，保留了对应日志、结果 json、渲染视频和预览图。清理对象主要包括：

- 旧 Human3R direct route 的 smoke/full checkpoint。
- Human3R c2w / fit_smpl 早期 checkpoint。
- 1024 direct route 的 1000 步 checkpoint。
- Hybrid 1000/6000 早期探索 checkpoint。

清理后 `output/human_scene` 占用从约 45G 降到约 27G。关键对照实验的视频、指标和预览仍保留；只是早期实验如果要从 checkpoint 继续训练，需要重跑。

## 下一步实验建议

1. 在 E10 基础上限制人体高斯最大数量，例如目标 180k 到 250k，避免 human densify 后增长到 553k。
2. 尝试 `human.densify_until_iter=7500` 或 `9000`，观察是否比 12000 更锐。
3. 跑一个 15000 步的 E08 延长版：禁 human densify/prune，仅延长优化，验证是否比 E10 更锐、更稳。
4. 对 Human3R scene prior 做更严格的前景过滤，避免 scene 点云侵入人体附近。
5. 对比最终视频中的固定帧，单独评估脸、衣服 logo、手脚边界和地面杂点。

E11 已完成第 3 项，结果显示完全禁用 human densify 不是最优。后续优先执行第 1、2 项。



## G 系列：面向接触/attention 的干净 scene 几何实验（2026-05-18）

这组实验重新聚焦核心问题：HUGS 原版 scene 高斯虽然渲染好，但 3D 几何结构差、杂点多，不适合后续人-场景接触识别和交叉 attention。因此本组不再只追求 PSNR/SSIM，而同时记录 scene 点云是否贴近 Human3R/真实深度几何先验、是否远离人体区域。

### G0: clean Human3R scene prior 构建

输入：

```text
data/human3r_hugs/hybrid_init/lab_h3r1024_scene_to_neuman_by_smpl_pcd.npz
```

处理：

```text
1. 使用 E13 posed human frame 0 点云作为人体参考。
2. 删除距离人体高斯中心 0.18 以内的 Human3R scene 点。
3. 删除 99.5 percentile 之外的远端 outlier。
4. voxel/max-points 下采样到 120k。
```

输出：

```text
data/human3r_hugs/hybrid_init/lab_h3r1024_scene_to_neuman_by_smpl_clean_bg120k_r018_pcd.npz
output/human_scene/position_pointclouds/G2G3_clean_h3r_scene_prior_bg120k_r018_xyzrgb.ply
```

构建统计：

```text
input_points: 400000
after_percentile_trim: 398000
removed_near_human: 1091
after_human_exclusion: 396909
final_points: 120000
```

注意：该 Human3R -> NeuMan 的 SMPL Sim3 对齐报告中 `smpl_fit_error_mean=3.64`，说明当前 Human3R/HUGS 人体对齐本身并不非常精确。这是后续使用 Human3R 几何先验时必须警惕的问题。

### G1: 原版 E13 的后处理几何清理

目的：不改训练，只对 E13 训练后的 scene 高斯做后处理过滤，验证“渲染高斯能不能被清理成更适合 contact 的点云”。

输入：

```text
E13 scene_final_splat.ply
E13 posed human point cloud
G0 clean Human3R scene prior
```

过滤规则：

```text
1. sigmoid opacity >= 0.05
2. scene 点到 G0 clean prior 最近距离 <= 2.0
3. scene 点到 posed human 最近距离 > 0.12
4. 最终下采样 scene 到 300k
```

输出：

```text
output/human_scene/position_pointclouds/G1_E13_scene_filtered_by_h3r_prior_frame000_scene_blue_human_red.ply
```

过滤统计：

```text
input_scene_points: 2138428
removed_low_opacity: 471540
removed_far_from_prior: 997455
removed_near_human: 48
after_filters_scene_points: 875381
after_subsample_scene_points: 300000
human_points: 192629
```

结论：G1 不影响渲染质量，因为它只是后处理。它可以显著减少用于 contact/attention 的 scene 点数量，并把点云约束到 Human3R 几何附近；但它不能改善训练过程中的 scene-human 竞争，也不能从源头阻止训练长出 floaters。

### G2/G3 快速训练对照（1000 步）

G2：只用 G0 clean Human3R background prior 初始化 scene，不加几何正则。

```bash
scene.init_pcd_path=data/human3r_hugs/hybrid_init/lab_h3r1024_scene_to_neuman_by_smpl_clean_bg120k_r018_pcd.npz
scene.densify_from_iter=100
scene.densify_until_iter=1000
scene.max_n_gaussians=300000
human.densify_from_iter=3000
human.densify_until_iter=3000
```

G3：在 G2 基础上加入轻量 scene 几何约束。

```bash
scene.anchor_pcd_path=data/human3r_hugs/hybrid_init/lab_h3r1024_scene_to_neuman_by_smpl_clean_bg120k_r018_pcd.npz
scene.anchor_loss_w=0.02
scene.anchor_loss_trunc=0.5
scene.anchor_loss_sample_scene=1024
scene.anchor_loss_sample_prior=4096
scene.human_exclusion_w=0.05
scene.human_exclusion_radius=0.12
scene.human_exclusion_sample_scene=1024
scene.human_exclusion_sample_human=4096
```

实现文件：

```text
hugs/trainer/gs_trainer.py
hugs/cfg/config.py
scripts/build_clean_scene_prior.py
scripts/filter_scene_centers_for_geometry.py
scripts/analyze_scene_pointcloud_geometry.py
```

G3 新增损失：

```text
scene_anchor: sampled scene centers 到 clean Human3R prior 的截断最近邻距离
scene_human_exclusion: sampled scene centers 进入 posed human 半径内时惩罚
```

### G2/G3 渲染指标

| ID | 实验 | 步数 | Overall PSNR | Overall SSIM | Overall LPIPS | Human PSNR | Human SSIM | Human LPIPS | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| G2 | clean Human3R bg120k init | 1000 | 18.8997 | 0.7652 | 0.3672 | 13.6774 | 0.6467 | 0.4027 | 几何干净但外观质量很差，纯 Human3R background prior 覆盖/颜色/投影不足 |
| G3 | G2 + scene anchor + human exclusion | 1000 | 19.0760 | 0.7677 | 0.3604 | 13.6267 | 0.6474 | 0.3982 | 比 G2 略好，但仍远低于 E12/E13；几何约束没有造成崩溃，但不能单独恢复外观 |

路径：

```text
G2 logdir:
output/human_scene/neuman/lab/hugs_trimlp/G2_clean_h3r_bg120k_1000/2026-05-18_23-26-20

G3 logdir:
output/human_scene/neuman/lab/hugs_trimlp/G3_clean_h3r_bg120k_anchor_excl_1000/2026-05-18_23-30-05
```

视频和预览：

```text
G2:
output/human_scene/neuman/lab/hugs_trimlp/G2_clean_h3r_bg120k_1000/2026-05-18_23-26-20/render_all_neuman_lab_final.mp4
output/human_scene/neuman/lab/hugs_trimlp/G2_clean_h3r_bg120k_1000/2026-05-18_23-26-20/render_all_preview.jpg

G3:
output/human_scene/neuman/lab/hugs_trimlp/G3_clean_h3r_bg120k_anchor_excl_1000/2026-05-18_23-30-05/render_all_neuman_lab_final.mp4
output/human_scene/neuman/lab/hugs_trimlp/G3_clean_h3r_bg120k_anchor_excl_1000/2026-05-18_23-30-05/render_all_preview.jpg
```

点云输出：

```text
G2:
output/human_scene/position_pointclouds/G2_clean_h3r_bg120k_1000_final_centers_frame000_scene_blue_human_red.ply

G3:
output/human_scene/position_pointclouds/G3_clean_h3r_bg120k_anchor_excl_1000_final_centers_frame000_scene_blue_human_red.ply

G3 GT camera centered preview:
output/human_scene/position_pointclouds/G3_gt_camera_centered_views/G3_gt_camera_human_centered_sheet.jpg
```

### G1/G2/G3 几何统计

统计文件：

```text
output/human_scene/position_pointclouds/G1G2G3_geometry_stats.json
```

关键指标如下。`scene_to_prior_dist` 越小，说明 scene 点越贴近 clean Human3R 几何先验；`scene_human_frac_lt_*` 越小，说明 scene 点越少侵入人体附近。

| 实验 | scene 点数 | scene->prior p50 | scene->prior p90 | prior 内 0.5 比例 | prior 内 2.0 比例 | scene-human <0.2 | scene-human <0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E13 原版 | 2,138,428 | 1.7869 | 8.8458 | 0.2236 | 0.5328 | 0.00008 | 0.00042 |
| G1 E13 后处理 | 300,000 | 0.6476 | 1.6762 | 0.4190 | 1.0000 | 0.00008 | 0.00048 |
| E12 Hybrid | 122,587 | 0.3985 | 5.9698 | 0.5616 | 0.7445 | 0.00123 | 0.00612 |
| G2 clean prior | 221,556 | 0.1851 | 0.4372 | 0.9284 | 0.9975 | 0.00260 | 0.01245 |
| G3 anchor/excl | 239,633 | 0.1816 | 0.4364 | 0.9265 | 0.9973 | 0.00316 | 0.01161 |

### 当前阶段判断

1. G2/G3 证明：如果 scene 完全从 clean Human3R background prior 出发，最终 scene 几何确实会显著更贴近 Human3R 深度结构，点数也控制在 20 万级，而不是 E13 的 200 万级。
2. 但 G2/G3 1000 步渲染质量很差，说明 Human3R clean prior 不能直接替代 HUGS 原始 scene 初始化。原因可能包括：Human3R -> HUGS 对齐误差、Human3R 点云覆盖不足、颜色/可见性不匹配、早期 RGB 优化难以从纯几何先验恢复外观。
3. G3 的 `scene_anchor` 没有破坏训练，指标略优于 G2；但 `scene_human_exclusion` 在当前 clean prior 下基本为 0，说明 frame 0 附近场景点本来就不太贴人体，或者半径 0.12 太小。后续如果要避免人体附近杂点，需要多帧 SMPL exclusion 或增大/动态调整半径。
4. G1 是当前最安全的工程方案：渲染仍用 E13/E12，contact/attention 使用后处理后的 clean scene 点云。但 G1 不能反向改善训练。
5. 最有潜力的下一步不是纯 G2/G3，而是 G3-mix：使用原版/HUGS scene 保外观，同时用 clean Human3R/Depth prior 约束 scene 高斯不要漂离真实表面，并对人体附近做多帧排斥。

### 下一步建议：G3-mix

建议实验：

```text
scene.init_pcd_path = 原版 NeuMan 点云 + 少量 clean Human3R prior
scene.anchor_pcd_path = clean Human3R prior
scene.anchor_loss_w = 0.005 ~ 0.02
scene.anchor_loss_trunc = 1.0 或 2.0
scene.densify_until_iter = 3000 或 6000
scene.max_n_gaussians = 300k ~ 600k
human_exclusion 使用多帧 posed SMPL/human gaussian，而不是只用当前帧 sampled human
```

这样更符合目标：保留 HUGS 的渲染质量，同时让最终用于 contact/attention 的 scene 高斯更贴近真实几何。


## 2026-05-18 G3-mix 补充实验：原版外观初始化 + clean Human3R 几何约束

### 目的

纯 G2/G3 使用 clean Human3R 背景点云初始化后，几何显著更干净，但 1000 步渲染质量明显下降。为了验证“保留 HUGS 原版外观收敛能力，同时把 scene 高斯拉回 Human3R/深度几何表面”是否可行，补充了 G3-mix：

```text
scene init = 原版 NeuMan scene 点云 9013 点 + clean Human3R scene prior 随机 50000 点
scene anchor = clean Human3R scene prior 120000 点
human init = HUGS 原版人体路线
```

这个实验仍然重点服务 G3 思路：不是完全替换 HUGS 初始化，而是让 scene 高斯训练过程受到 Human3R 几何先验约束，同时尽量保留原版 HUGS 的可优化性。

### 配置

初始化文件：

```text
data/human3r_hugs/hybrid_init/lab_neuman_orig_plus_clean_h3r50k_pcd.npz
```

训练命令关键参数：

```bash
--quick --quick-steps 1000
--exp-name G3mix_orig_cleanh3r50k_anchor_1000
scene.init_pcd_path=data/human3r_hugs/hybrid_init/lab_neuman_orig_plus_clean_h3r50k_pcd.npz
scene.anchor_pcd_path=data/human3r_hugs/hybrid_init/lab_h3r1024_scene_to_neuman_by_smpl_clean_bg120k_r018_pcd.npz
scene.anchor_loss_w=0.01
scene.anchor_loss_trunc=1.0
scene.anchor_loss_sample_scene=1024
scene.anchor_loss_sample_prior=4096
scene.human_exclusion_w=0.05
scene.human_exclusion_radius=0.12
scene.human_exclusion_sample_scene=1024
scene.human_exclusion_sample_human=4096
scene.densify_from_iter=100
scene.densify_until_iter=1000
scene.densification_interval=100
scene.max_n_gaussians=300000
human.densify_from_iter=3000
human.densify_until_iter=3000
human.prune_min_opacity=0.0
human.max_n_gaussians=200000
```

输出目录：

```text
output/human_scene/neuman/lab/hugs_trimlp/G3mix_orig_cleanh3r50k_anchor_1000/2026-05-18_23-41-33
```

### G2/G3/G3-mix 1000 步渲染指标

同为 1000 步时，G3-mix 明显优于纯 clean Human3R 初始化。表中 PSNR/SSIM 越高越好，LPIPS 越低越好；粗体为三者最优。

| ID | 实验 | Overall PSNR | Overall SSIM | Overall LPIPS | Human PSNR | Human SSIM | Human LPIPS | 判断 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| G2 | clean Human3R bg120k init | 18.8997 | 0.7652 | 0.3672 | 13.6774 | 0.6467 | 0.4027 | 几何最贴近先验之一，但外观差 |
| G3 | G2 + anchor + human exclusion | 19.0760 | 0.7677 | 0.3604 | 13.6267 | 0.6474 | 0.3982 | 比 G2 略好，仍外观差 |
| G3-mix | 原版点云 + clean H3R 50k + anchor | **22.9043** | **0.8422** | **0.2036** | **16.4839** | **0.6827** | **0.3423** | 当前更合理的折中路线 |

G3-mix 视频和预览：

```text
output/human_scene/neuman/lab/hugs_trimlp/G3mix_orig_cleanh3r50k_anchor_1000/2026-05-18_23-41-33/render_all_neuman_lab_final.mp4
output/human_scene/neuman/lab/hugs_trimlp/G3mix_orig_cleanh3r50k_anchor_1000/2026-05-18_23-41-33/render_all_preview.jpg
```

### 点云输出

G3-mix 人体/场景高斯中心点云：

```text
output/human_scene/position_pointclouds/G3mix_orig_cleanh3r50k_anchor_1000_final_centers_frame000_scene_blue_human_red.ply
```

点数：

```text
scene_points = 97324
human_points = 110210
scene color = blue
human color = red
```

GT 相机视角、以人体为中心的点云预览：

```text
output/human_scene/position_pointclouds/G3mix_gt_camera_centered_views/G3mix_gt_camera_human_centered_sheet.jpg
```

### G1/G2/G3/G3-mix 几何对比

统计文件：

```text
output/human_scene/position_pointclouds/G1G2G3_geometry_stats_with_G3mix.json
```

`scene->prior` 越小，说明 scene 高斯中心越贴近 clean Human3R 几何先验。`prior 内 0.5/2.0 比例` 越高，说明更多 scene 点落在 Human3R 几何附近。粗体标注当前表内最优值；但要注意，几何最优不等于渲染最优。

| 实验 | scene 点数 | scene->prior p50 | scene->prior p90 | prior 内 0.5 比例 | prior 内 2.0 比例 | scene-human <0.2 | scene-human <0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E13 原版 | 2,138,428 | 1.7869 | 8.8458 | 0.2236 | 0.5328 | **0.00008** | **0.00042** |
| G1 E13 后处理 | 300,000 | 0.6476 | 1.6762 | 0.4190 | **1.0000** | **0.00008** | 0.00048 |
| E12 Hybrid | 122,587 | 0.3985 | 5.9698 | 0.5616 | 0.7445 | 0.00123 | 0.00612 |
| G2 clean prior | 221,556 | 0.1851 | 0.4372 | **0.9284** | 0.9975 | 0.00260 | 0.01245 |
| G3 anchor/excl | 239,633 | **0.1816** | **0.4364** | 0.9265 | 0.9973 | 0.00316 | 0.01161 |
| G3-mix | 97,324 | 0.2464 | 3.2496 | 0.7430 | 0.8646 | 0.00376 | 0.01118 |

### 结论

1. G2/G3 证明了完全使用 clean Human3R 背景 prior 可以显著改善 scene 点云几何，但渲染质量不够，说明 Human3R 点云不能直接替代 HUGS 原始 scene 初始化。
2. G3-mix 是目前更有希望的方向：它比 E13/E12 更贴近 Human3R 几何先验，同时比纯 G2/G3 保留了更多 HUGS 的外观收敛能力。
3. G3-mix 仍未达到原版/E12 的长步数质量，主要原因可能是 anchor loss 在早期就限制了 scene 高斯为重建 RGB 所需的自由移动；同时 Human3R prior 和 NeuMan/HUGS 相机坐标仍存在非零对齐误差。
4. 下一步不建议继续纯 G2/G3；建议围绕 G3-mix 做 6000/15000 步对照，并尝试更温和的几何约束：较小 `anchor_loss_w`、分阶段打开 anchor、只对低 opacity/远离相机表面的点做 anchor、以及使用多帧人体外壳做 scene-human exclusion。


## 2026-05-19 G3-mix 6000 步实验

### 目的

1000 步 G3-mix 已经证明“原版 NeuMan 点云 + clean Human3R 几何 anchor”比纯 Human3R clean prior 更稳定。为了判断它是否能在公平步数下接近原版渲染质量，同时保留更干净的 scene 几何，本次继续跑 6000 步。

### 配置

```text
exp_name = G3mix_orig_cleanh3r50k_anchor_6000_w005
scene init = 原版 NeuMan scene 点云 9013 点 + clean Human3R scene prior 50000 点
scene anchor = clean Human3R scene prior 120000 点
anchor_loss_w = 0.005
anchor_loss_trunc = 1.0
scene densify = 100 -> 3000
scene max_n_gaussians = 300000
human densify = 3000 -> 6000
human max_n_gaussians = 200000
```

输出目录：

```text
output/human_scene/neuman/lab/hugs_trimlp/G3mix_orig_cleanh3r50k_anchor_6000_w005/2026-05-18_23-58-26
```

训练耗时：约 10 分钟，最终全帧视频另用约 30 秒生成。

### 渲染结果

| 实验 | 步数 | Overall PSNR | Overall SSIM | Overall LPIPS | Human PSNR | Human SSIM | Human LPIPS | 说明 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 原版 HUGS 同分辨率 baseline | 6000 | **26.3263** | **0.9208** | **0.0392** | **19.2890** | 0.7276 | **0.0843** | 渲染质量仍最好，但 scene 点云几何杂点最多 |
| G3-mix w0.005 | 3000 | 24.8184 | 0.8809 | 0.1209 | 18.2524 | 0.7246 | 0.2447 | 中途验证，已明显好于纯 G2/G3 |
| G3-mix w0.005 | 6000 | 25.4842 | 0.8957 | 0.0962 | 18.7951 | **0.7387** | 0.2013 | 质量仍低于原版，但差距可控，scene 几何更干净 |

重要观察：G3-mix 6000 的整体 PSNR 比同分辨率原版低约 0.84 dB，SSIM 低约 0.025，LPIPS 高约 0.057；人体 PSNR 低约 0.49 dB，人体 SSIM 反而略高，但人体 LPIPS 明显更差。这说明形状/结构没有完全坏，但外观细节和感知质量仍受 Human3R anchor 或混合初始化影响。

视频与预览：

```text
output/human_scene/neuman/lab/hugs_trimlp/G3mix_orig_cleanh3r50k_anchor_6000_w005/2026-05-18_23-58-26/render_all_neuman_lab_final.mp4
output/human_scene/neuman/lab/hugs_trimlp/G3mix_orig_cleanh3r50k_anchor_6000_w005/2026-05-18_23-58-26/render_all_preview.jpg
```

### 点云输出

```text
output/human_scene/position_pointclouds/G3mix_orig_cleanh3r50k_anchor_6000_w005_final_centers_frame000_scene_blue_human_red.ply
```

点数：

```text
scene_points = 297990
human_points = 181692
scene color = blue
human color = red
```

GT 相机视角、以人体为中心的点云预览：

```text
output/human_scene/position_pointclouds/G3mix6000_gt_camera_centered_views/G3mix6000_gt_camera_human_centered_sheet.jpg
```

### 几何统计

统计文件：

```text
output/human_scene/position_pointclouds/G1G2G3_geometry_stats_with_G3mix6000.json
```

| 实验 | scene 点数 | scene->prior p50 | scene->prior p90 | prior 内 0.5 比例 | prior 内 2.0 比例 | scene-human <0.2 | scene-human <0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E13 原版 15000 | 2,138,428 | 1.7869 | 8.8458 | 0.2236 | 0.5328 | **0.00008** | **0.00042** |
| G1 E13 后处理 | 300,000 | 0.6476 | 1.6762 | 0.4190 | **1.0000** | **0.00008** | 0.00048 |
| E12 Hybrid 15000 | 122,587 | 0.3985 | 5.9698 | 0.5616 | 0.7445 | 0.00123 | 0.00612 |
| G2 clean prior 1000 | 221,556 | 0.1851 | 0.4372 | **0.9284** | 0.9975 | 0.00260 | 0.01245 |
| G3 anchor/excl 1000 | 239,633 | **0.1816** | **0.4364** | 0.9265 | 0.9973 | 0.00316 | 0.01161 |
| G3-mix 1000 | 97,324 | 0.2464 | 3.2496 | 0.7430 | 0.8646 | 0.00376 | 0.01118 |
| G3-mix 6000 w0.005 | 297,990 | 0.2866 | 4.3929 | 0.6876 | 0.8304 | 0.00170 | 0.00781 |

### 当前结论

1. G3-mix 6000 是目前最接近我们目标的训练型方案：渲染质量比纯 Human3R 初始化好很多，同时 scene 点云数量和几何分布明显比原版更适合后续 contact/attention。
2. 它没有完全达到同分辨率原版 HUGS 的渲染质量，尤其 LPIPS 仍有明显差距；这意味着 Human3R prior/anchor 会限制一部分用于外观拟合的自由度。
3. 但它保留了 Human3R 的关键亮点：scene 高斯更贴近深度几何，scene 点数从原版百万级降到 30 万级，人体和场景也仍在同一个合理坐标关系中。
4. 下一步最值得做的是 G3-mix 的约束调度，而不是更强 anchor：例如前 3000 步弱 anchor 或不开 anchor，3000 步后只对低 opacity/远离 prior 的 scene 点施加几何回拉；同时把 human exclusion 从单帧采样改成多帧人体 envelope，避免后续人体附近出现 scene 杂点。


## 2026-05-19 G4 DepthPro 深度监督实验

### 目的

G3-mix 证明了“原版/NeuMan 外观初始化 + Human3R 几何 anchor”是可行方向，但点云仍有厚度和漂浮问题。G4 的目标是在不破坏 G3-mix 渲染质量的前提下，引入单目深度监督，让 scene 高斯更贴近真实深度表面。这里使用 `/hdd/u202420081000003/ml-depth-pro` 的 DepthPro，因为它能直接输出与 lab 原图一致的 `717x1276` 深度图，避免 Human3R `560x1024` 深度带来的分辨率误差。

### 深度尺度诊断

DepthPro raw 深度不能直接用于 HUGS。用 G3-mix 初始化点云投影到 NeuMan/HUGS 相机后，前 20 个训练帧的中位相机深度约为 `35-43`，而 DepthPro 中位深度约为 `4.7-6.5`。估计到的全局比例：

~~~text
global_ratio_median = 7.37146258
~~~

因此生成了对齐到 HUGS 坐标尺度的深度目录：

~~~text
data/neuman/dataset/lab/depthpro_full_hugs_scale_s7p37146
~~~

未缩放的 raw DepthPro 200 步 smoke 中 `l_scene_depth_prior` 基本饱和在 `0.999`，说明尺度错误会给训练施加错误梯度；缩放后该项稳定在约 `0.30-0.55`，可以作为有效的弱几何监督。

### 配置

~~~text
exp_name = G4_depthpro_scaled_g3mix_w005_d002_6000
scene init = 原版 NeuMan scene 点云 9013 点 + clean Human3R scene prior 50000 点
scene anchor = clean Human3R scene prior 120000 点
anchor_loss_w = 0.005
scene depth prior = DepthPro full-res depth, scaled by 7.37146258
depth_prior_w = 0.02
depth_prior_tolerance = 1.5
depth_prior_max_residual = 10.0
scene densify = 100 -> 3000
scene max_n_gaussians = 300000
human densify = 3000 -> 6000
human max_n_gaussians = 200000
~~~

输出目录：

~~~text
output/human_scene/neuman/lab/hugs_trimlp/G4_depthpro_scaled_g3mix_w005_d002_6000/2026-05-19_01-56-48
~~~

训练耗时：约 9 分钟 12 秒；最终全帧视频另用约 33 秒生成。

### 渲染结果

| 实验 | 步数 | Overall PSNR | Overall SSIM | Overall LPIPS | Human PSNR | Human SSIM | Human LPIPS | 说明 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 原版 HUGS 同分辨率 baseline | 6000 | **26.3263** | **0.9208** | **0.0392** | **19.2890** | 0.7276 | **0.0843** | 渲染质量仍最好，但 scene 几何最杂 |
| G3-mix w0.005 | 6000 | 25.4842 | 0.8957 | 0.0962 | 18.7951 | **0.7387** | 0.2013 | 已接近可用，几何比原版更干净 |
| G4 DepthPro scaled w0.02 | 3000 | 24.8004 | 0.8742 | 0.1256 | 18.3747 | 0.7241 | 0.2494 | 中途验证 |
| G4 DepthPro scaled w0.02 | 6000 | 25.5137 | 0.8966 | 0.0946 | 18.7059 | 0.7352 | 0.2008 | 相比 G3-mix 整体指标微升，人体 PSNR/SSIM 微降 |

### 可视化输出

~~~text
output/human_scene/neuman/lab/hugs_trimlp/G4_depthpro_scaled_g3mix_w005_d002_6000/2026-05-19_01-56-48/render_all_neuman_lab_final.mp4
output/human_scene/neuman/lab/hugs_trimlp/G4_depthpro_scaled_g3mix_w005_d002_6000/2026-05-19_01-56-48/render_val_final_preview.jpg
~~~

点云输出：

~~~text
output/human_scene/position_pointclouds/G4_depthpro_scaled_g3mix_w005_d002_6000_final_centers_frame000_scene_blue_human_red.ply
~~~

点数：

~~~text
scene_points = 282447
human_points = 182353
scene color = blue
human color = red
~~~

GT 相机视角、以人体为中心的点云预览：

~~~text
output/human_scene/position_pointclouds/G4_depthpro_scaled_g3mix_w005_d002_6000_gt_camera_centered_views/G4_depthpro_scaled_gt_camera_human_centered_sheet.jpg
~~~

### 几何统计

统计文件：

~~~text
output/human_scene/position_pointclouds/G4_depthpro_scaled_g3mix_w005_d002_6000_geometry_vs_h3r_prior.json
~~~

| 实验 | scene 点数 | scene->prior p50 | scene->prior p90 | scene->prior p95 | prior 内 0.5 比例 | prior 内 2.0 比例 | scene-human <0.2 | scene-human <0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G3-mix 6000 w0.005 | 297,990 | 0.2866 | 4.3929 | 7.9932 | 0.6876 | 0.8304 | **0.00170** | 0.00781 |
| G4 DepthPro scaled w0.02 | 282,447 | **0.2836** | **4.2228** | **7.6124** | **0.6994** | **0.8393** | 0.00205 | **0.00687** |

### 当前结论

1. G4 的尺度校准是必要条件。DepthPro raw 深度虽然分辨率正确，但和 NeuMan/HUGS 坐标尺度不一致；未缩放监督会直接饱和，不能用于正式训练。
2. 在 G3-mix 基础上加入弱 DepthPro 深度监督后，overall PSNR/SSIM/LPIPS 相比 G3-mix 6000 有轻微提升，说明 `depth_prior_w=0.02` 没有破坏整体渲染。
3. 几何上 G4 比 G3-mix 更贴近 Human3R clean prior：p90/p95 距离下降，prior 内 0.5/2.0 比例上升，scene 点数也更少。这说明深度监督确实在把 scene 高斯往更合理表面拉。
4. 人体质量没有明显改善，human PSNR/SSIM 略低于 G3-mix，human LPIPS 只微好。这说明 G4 当前主要改善背景/scene 几何，不解决人体外观细节缺失。
5. 后续可改善方向：做逐帧 scale 或 scale+shift 深度校准，而不是全局单比例；把深度监督限制到高置信背景区域；加入 normal consistency/深度边缘保护；以及把 depth loss 调度为前期弱、中期增强、后期关闭，避免影响最终外观拟合。


### G4 三类点云叠加导出

为了检查最终优化结果和 DepthPro 深度约束之间的空间关系，额外导出了一个包含三类点的 PLY：

~~~text
output/human_scene/position_pointclouds/G4_depthpro_scaled_depth_scene_human_frame000_depthgreen_sceneblue_humanred.ply
~~~

颜色和 `part` 字段：

~~~text
part=0: final scene Gaussian centers, blue
part=1: final posed human Gaussian centers at frame 000, red
part=2: scaled DepthPro background depth point cloud, green
~~~

导出设置：

~~~text
depth source = data/neuman/dataset/lab/depthpro_full_hugs_scale_s7p37146
depth frames = 0:103:3
mask human in depth = true
max depth points per frame = 10000
depth points = 350000
scene points = 282447
human points = 182353
~~~

对应 GT 相机视角预览：

~~~text
output/human_scene/position_pointclouds/G4_depthpro_scaled_depth_scene_human_gt_camera_views/G4_depth_scene_human_gt_camera_human_centered_sheet.jpg
~~~

这个文件用于查看：绿色深度表面是否贴近蓝色 scene 高斯中心，以及红色人体高斯是否处在合理的人-场景相对位置。需要注意：绿色深度点云来自多帧单目深度反投影，默认去掉 SAM 人体区域，因此它不是一个经过多视图融合/去重的干净 mesh，而是用于对比深度监督位置的参考点云。


### G4 深度约束有效性补充评估

用户观察到 G4 的蓝色 scene 高斯仍有大量点偏离绿色 scaled DepthPro 深度点云。为了直接评估“depth loss 是否真的约束住背景高斯分布”，补充了 scene 高斯中心到 scaled DepthPro 深度表面的投影残差统计。

评估脚本：

~~~text
scripts/evaluate_scene_depth_alignment.py
~~~

评估文件：

~~~text
output/human_scene/position_pointclouds/G3mix_G4_depth_alignment_scaled_depth.json
~~~

评估方式：将每个 scene Gaussian center 投影到多帧 GT 相机，与对应帧 scaled DepthPro depth 比较 signed depth residual。对每个 scene 点取所有有效帧里的最小 absolute residual。这个指标已经比较宽松，如果这里仍然偏大，说明点云确实没有被深度表面约束干净。

| 实验 | scene 点数 | supported frac | abs residual p50 | abs residual p90 | abs residual p95 | all points <0.5 | all points <1.0 | all points <2.0 | behind depth frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G3-mix 6000 | 297,990 | 0.9963 | 0.9674 | 8.1317 | 12.2629 | 0.3889 | 0.5029 | 0.6149 | 0.6821 |
| G4 DepthPro scaled 6000 | 282,447 | **0.9965** | **0.8567** | **8.0848** | 12.4122 | **0.4085** | **0.5212** | **0.6324** | **0.6716** |

结论修正：G4 只带来了非常有限的几何改善。虽然 p50 和 <1.0/<2.0 比例略优于 G3-mix，但在很宽松的多帧最小残差标准下，仍有接近一半 scene 点距离深度表面超过 1.0，约 36.8% 超过 2.0，p90 仍超过 8.0。这说明当前 `depth_prior_w=0.02` 的点采样式 depth loss 没有从根本上清理背景高斯分布。之前只强调渲染质量是不充分的；后续实验必须把这个 depth-alignment 指标作为主指标之一。

当前 G4 失败原因判断：

1. depth loss 太弱，每步只采样少量 scene points，且权重远小于 photometric/densification 对外观拟合的驱动力。
2. loss 只比较点中心的深度，不约束高斯尺度、透明度、屏幕 footprint，也不惩罚远离深度表面的低贡献漂浮高斯。
3. densification 会持续复制/生成用于拟合颜色的高斯，新点可能从一开始就不在深度面上；depth loss 没有对新点做硬投影或剪枝。
4. 使用全局 depth scale 仍然粗糙，不同视角/区域的 DepthPro scale/shift 误差会让约束表面变厚。
5. 当前指标显示多数偏离点在 depth 后方 `behind_depth_frac` 约 0.67，说明很多 scene 高斯落在可见深度表面后面，视觉上可能仍能贡献颜色，但几何上不干净。

下一步应该转向“几何硬约束/剪枝 + 渲染恢复”，而不是继续单纯加权弱 depth loss：

1. 训练中或训练后周期性删除/冻结距离 depth surface 过远且 opacity/可见贡献低的 scene 高斯。
2. 对新 densified scene 高斯做 depth projection，把中心拉回当前视角深度表面附近。
3. 使用逐帧或局部 scale+shift 校准 DepthPro，再构建更薄的多视角融合深度点云。
4. depth loss 从 center depth residual 扩展为 opacity-weighted surface consistency，重点惩罚 depth 表面前后厚度。
5. 把最终评价改成双指标：渲染质量 + scene-depth alignment，不再只用 PSNR/SSIM/LPIPS 判断方案优劣。

## G5：原版 COLMAP 初始化 + DepthPro 全流程深度约束实验

目标：不再使用 Human3R 或其它外部点云作为初始化，而是从 HUGS/NeuMan 原版 COLMAP scene 点云出发，在高斯优化全过程中用 DepthPro 深度约束 scene Gaussian 的几何分布。主评估不只看渲染质量，也看 scene Gaussian centers 与 DepthPro 深度表面的投影残差。

### G5 实现改动

新增配置项默认关闭，不影响原版 HUGS：

```text
scene.depth_prune_interval
scene.depth_prune_from_iter
scene.depth_prune_until_iter
scene.depth_prune_abs_threshold
scene.depth_prune_abs_hard_threshold
scene.depth_prune_opacity_threshold
scene.depth_prune_max_frac
scene.depth_prune_mask_human
```

训练中新增 `scene_depth_guided_pruning`：把 scene Gaussian center 投影到当前 GT 相机，读取同像素 scaled DepthPro 深度；在背景 mask 区域中，对 depth residual 过大且 opacity 较低的点做软剪枝，对 residual 极大的点做硬剪枝。该逻辑只有显式设置 `scene.depth_prune_interval > 0` 才会启用。

DepthPro 尺度重新用原版 COLMAP 初始化估计，而不是沿用 Human3R/G3 混合尺度：

```text
raw depth dir: data/neuman/dataset/lab/depthpro_full
COLMAP scale: 8.17764187
scaled depth dir: data/neuman/dataset/lab/depthpro_full_hugs_colmap_scale_s8p17764
scale diagnosis: output/human_scene/position_pointclouds/G5_colmap_depthpro_scale_diagnosis.txt
```

### G5 实验结果表

| 实验 | 初始化 | 关键设置 | Scene 点数 | Overall PSNR ↑ | Overall SSIM ↑ | Overall LPIPS ↓ | Human PSNR ↑ | Human SSIM ↑ | Human LPIPS ↓ | Depth p50 ↓ | Depth p90 ↓ | 全部点 <1.0 ↑ | 全部点 <2.0 ↑ | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| G4 scaled DepthPro | G3-mix/Human3R 相关初始化 | weak depth loss `w=0.02`，无硬剪枝 | 282,447 | 25.5137 | **0.8966** | **0.0946** | **18.7059** | **0.7352** | **0.2008** | 0.8567 | 8.0848 | 0.5212 | 0.6324 | 渲染好，但几何约束很弱，蓝色 scene 仍大量偏离深度面 |
| G5a | 原版 COLMAP | depth loss `w=0.05` + 弱剪枝，全程 densify | 554,239 | 21.5685 | 0.8468 | 0.1609 | 15.8752 | 0.6821 | 0.2707 | **0.1717** | **0.8814** | **0.9106** | **0.9572** | 深度约束非常有效，但 scene 点暴涨，渲染质量明显下降 |
| G5b | 原版 COLMAP | depth loss `w=0.035` + 强剪枝，scene cap 120K，densify 到 3000 | 7,227 | 21.1121 | 0.7874 | 0.3113 | 18.3996 | 0.6963 | 0.2471 | 0.2090 | 0.9172 | 0.8818 | 0.9322 | 点云非常稀疏干净，但背景欠表达，整体渲染不可接受 |
| G5c | 原版 COLMAP | depth loss `w=0.015` + 2000 步后中等剪枝，scene cap 180K，densify 到 5000 | 185,625 | **24.3533** | 0.8689 | 0.1447 | 18.2126 | 0.7058 | 0.2479 | 0.1890 | 0.9512 | 0.8939 | 0.9466 | 当前最好折中：渲染明显优于 G5a/G5b，几何远优于 G4 |

### G5 文件位置

G5a：

```text
logdir: output/human_scene/neuman/lab/hugs_trimlp/G5a_orig_depthloss_prune_w005_6000/2026-05-19_03-28-31
video:  output/human_scene/neuman/lab/hugs_trimlp/G5a_orig_depthloss_prune_w005_6000/2026-05-19_03-28-31/render_all_neuman_lab_final.mp4
ply:    output/human_scene/position_pointclouds/G5a_orig_depthloss_prune_w005_6000_final_centers_frame000_scene_blue_human_red.ply
json:   output/human_scene/position_pointclouds/G5a_depth_alignment_scaled_colmap_depth.json
```

G5b：

```text
logdir: output/human_scene/neuman/lab/hugs_trimlp/G5b_orig_depthprune_cap120k_6000/2026-05-19_03-39-19
video:  output/human_scene/neuman/lab/hugs_trimlp/G5b_orig_depthprune_cap120k_6000/2026-05-19_03-39-19/render_all_neuman_lab_final.mp4
ply:    output/human_scene/position_pointclouds/G5b_orig_depthprune_cap120k_6000_final_centers_frame000_scene_blue_human_red.ply
json:   output/human_scene/position_pointclouds/G5b_depth_alignment_scaled_colmap_depth.json
```

G5c：

```text
logdir: output/human_scene/neuman/lab/hugs_trimlp/G5c_orig_depthloss_midprune_cap180k_6000/2026-05-19_03-50-22
video:  output/human_scene/neuman/lab/hugs_trimlp/G5c_orig_depthloss_midprune_cap180k_6000/2026-05-19_03-50-22/render_all_neuman_lab_final.mp4
ply:    output/human_scene/position_pointclouds/G5c_orig_depthloss_midprune_cap180k_6000_final_centers_frame000_scene_blue_human_red.ply
json:   output/human_scene/position_pointclouds/G5c_depth_alignment_scaled_colmap_depth.json
```

### 当前判断

1. 深度约束确实能从根本上改善 scene Gaussian 几何。G5a/G5c 的 depth alignment 相比 G4 有数量级改善，尤其 p90 从 G4 的 8.08 降到 0.88-0.95。
2. 只加弱 depth loss 不够；必须配合训练过程中的硬剪枝，否则 densification 会不断生成偏离深度表面的高斯。
3. 剪枝过强也不行。G5b 只有 7227 个 scene 点，几何薄但背景欠表达，overall LPIPS 退化到 0.3113。
4. 当前最好的折中是 G5c：它保留原版 COLMAP 初始化，不使用 Human3R；先让外观/颜色拟合建立起来，再从 2000 步开始做中等强度 depth pruning，并限制 scene 点数到 180K 左右。它的渲染质量虽然仍低于 G4/original baseline，但几何干净程度远好于 G4。
5. 后续应继续沿 G5c 方向调参，而不是回到 Human3R 初始化：可以尝试 `scene.depth_prior_w=0.01`、`scene.max_n_gaussians=220000`、`depth_prune_from_iter=2500`、`depth_prune_abs_hard_threshold=8.0`，目标是进一步恢复渲染质量，同时保持 p90 depth residual < 1.5。

