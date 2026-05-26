# Human3R 初始化实验汇总

更新时间：2026-05-18

本文件汇总当前已完成的原版 HUGS、Human3R 直接路线、Hybrid 初始化路线实验，并记录最终视频与抽帧预览图位置。

## 指标总表

| ID | 实验 | Overall PSNR | Overall SSIM | Overall LPIPS | Human PSNR | Human SSIM | Human LPIPS | 预览图 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| E00 | 原版 HUGS full-res 6000 | 25.8792 | 0.9031 | 0.0838 | 19.0826 | 0.7477 | 0.1653 | `output/human_scene/experiment_previews/E00_neuman_lab_original_6000_vis_2026-05-15_00-29-45.jpg` |
| E01 | 原版 HUGS Human3R 分辨率 6000 | 26.3263 | 0.9208 | 0.0392 | 19.2890 | 0.7276 | 0.0843 | `output/human_scene/experiment_previews/E01_neuman_lab_h3rres_6000_vis_2026-05-15_01-06-26.jpg` |
| E02 | 旧 Human3R 直接初始化 6000 | 18.9311 | 0.6439 | 0.1806 | 15.2528 | 0.5131 | 0.1648 | `output/human_scene/experiment_previews/E02_human3r_fit_smpl_6000_vis_2026-05-14_23-59-57.jpg` |
| E03 | Human3R 1024 直接路线 1000 | 14.7826 | 0.6235 | 0.5896 | 10.9411 | 0.4827 | 0.6040 | `output/human_scene/experiment_previews/E03_human3r_1024_full_sammask_lcc_1000_nohdens_2026-05-18_01-01-25.jpg` |
| E04 | Hybrid 50k 1000 | 22.9476 | 0.8413 | 0.2024 | 16.3763 | 0.6758 | 0.3597 | `output/human_scene/experiment_previews/E04_neuman_lab_hybrid_orig_plus_h3r_smpl50k_1000_2026-05-18_01-37-02.jpg` |
| E05 | Hybrid 25k 1000 | 23.4737 | 0.8419 | 0.1948 | 17.3733 | 0.6904 | 0.3180 | `output/human_scene/experiment_previews/E05_neuman_lab_hybrid_orig_plus_h3r_smpl25k_1000_2026-05-18_01-41-13.jpg` |
| E06 | Hybrid 100k 1000 | 22.4651 | 0.8374 | 0.2071 | 15.6446 | 0.6759 | 0.3514 | `output/human_scene/experiment_previews/E06_neuman_lab_hybrid_orig_plus_h3r_smpl100k_1000_2026-05-18_01-44-39.jpg` |
| E07 | Hybrid 25k 6000 scene 到 6000 | 24.5184 | 0.8851 | 0.1112 | 17.4330 | 0.7044 | 0.2631 | `output/human_scene/experiment_previews/E07_neuman_lab_hybrid_orig_plus_h3r_smpl25k_6000_2026-05-18_01-48-23.jpg` |
| E08 | Hybrid 25k 6000 禁 human densify scene 到 3000 | 25.3258 | 0.8943 | 0.1061 | 18.5122 | 0.7374 | 0.2270 | `output/human_scene/experiment_previews/E08_neuman_lab_hybrid_orig_plus_h3r_smpl25k_6000_nohprune_scene3000_2026-05-18_01-59-13.jpg` |
| E09 | Hybrid 25k 6000 human densify no prune | 25.1099 | 0.8939 | 0.1082 | 18.2822 | 0.7319 | 0.2332 | `output/human_scene/experiment_previews/E09_neuman_lab_hybrid_orig_plus_h3r_smpl25k_6000_hdens_noprune_scene3000_2026-05-18_02-11-48.jpg` |
| E10 | Hybrid 25k 15000 human densify no prune | 25.7824 | 0.9046 | 0.0877 | 18.9393 | 0.7513 | 0.1572 | `output/human_scene/experiment_previews/E10_neuman_lab_hybrid_orig_plus_h3r_smpl25k_15000_hdens_noprune_scene3000_2026-05-18_02-23-25.jpg` |
| E11 | Hybrid 25k 15000 禁 human densify | 25.7375 | 0.9048 | 0.0883 | 18.8328 | 0.7489 | 0.1700 | `output/human_scene/experiment_previews/E11_neuman_lab_hybrid_orig_plus_h3r_smpl25k_15000_nohprune_scene3000_rerun_2026-05-18_03-01-50.jpg` |
| E12 | Hybrid 25k 15000 human densify cap200k | 26.0035 | 0.9047 | 0.0901 | 19.3210 | 0.7544 | 0.1709 | `output/human_scene/experiment_previews/E12_neuman_lab_hybrid_orig_plus_h3r_smpl25k_15000_hdens_cap200k_scene3000_2026-05-18_03-26-15.jpg` |

## 关键结论

- 分辨率不是主要原因：原版 HUGS 在 Human3R 分辨率下 E01 仍然表现很好。
- Human3R 直接替换 HUGS 数据路线质量显著下降，核心风险来自相机/深度/SMPL-X 到 SMPL 映射/几何假设同时变化。
- 目前最可行路线是 Hybrid：保留 NeuMan/HUGS 原图像、相机和人体训练流程，只把 Human3R 对齐后的场景点云作为轻量初始化先验。
- Human3R scene prior 不宜过重：25k 点优于 50k 和 100k。
- 需要控制 scene-human 竞争：scene densify 过久或 human 被剪枝，会导致人体质量明显下降。
- E12 是当前最均衡结果：overall PSNR 26.0035，human PSNR 19.3210，human SSIM 0.7544，已经超过原版 6000 的 PSNR/SSIM；human LPIPS 0.1709 仍略差于原版 0.1653。

## 集中预览图目录

`output/human_scene/experiment_previews`

每个有最终视频的实验目录中也保留了同名 `render_all_preview.jpg`。


## 原始实验目录

- E00 原版 HUGS full-res 6000: `output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_original_6000_vis/2026-05-15_00-29-45`
- E01 原版 HUGS Human3R 分辨率 6000: `output/human_scene/neuman/lab_h3rres/hugs_trimlp/neuman_lab_h3rres_6000_vis/2026-05-15_01-06-26`
- E02 旧 Human3R 直接初始化 6000: `output/human_scene/human3r/lab_fit_smpl/hugs_trimlp/human3r_fit_smpl_6000_vis/2026-05-14_23-59-57`
- E03 Human3R 1024 直接路线 1000: `output/human_scene/human3r/lab_size1024_full_sammask_lcc/hugs_trimlp/human3r_1024_full_sammask_lcc_1000_nohdens/2026-05-18_01-01-25`
- E04 Hybrid 50k 1000: `output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_hybrid_orig_plus_h3r_smpl50k_1000/2026-05-18_01-37-02`
- E05 Hybrid 25k 1000: `output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_hybrid_orig_plus_h3r_smpl25k_1000/2026-05-18_01-41-13`
- E06 Hybrid 100k 1000: `output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_hybrid_orig_plus_h3r_smpl100k_1000/2026-05-18_01-44-39`
- E07 Hybrid 25k 6000 scene 到 6000: `output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_hybrid_orig_plus_h3r_smpl25k_6000/2026-05-18_01-48-23`
- E08 Hybrid 25k 6000 禁 human densify scene 到 3000: `output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_hybrid_orig_plus_h3r_smpl25k_6000_nohprune_scene3000/2026-05-18_01-59-13`
- E09 Hybrid 25k 6000 human densify no prune: `output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_hybrid_orig_plus_h3r_smpl25k_6000_hdens_noprune_scene3000/2026-05-18_02-11-48`
- E10 Hybrid 25k 15000 human densify no prune: `output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_hybrid_orig_plus_h3r_smpl25k_15000_hdens_noprune_scene3000/2026-05-18_02-23-25`
- E11 Hybrid 25k 15000 禁 human densify: `output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_hybrid_orig_plus_h3r_smpl25k_15000_nohprune_scene3000_rerun/2026-05-18_03-01-50`
- E12 Hybrid 25k 15000 human densify cap200k: `output/human_scene/neuman/lab/hugs_trimlp/neuman_lab_hybrid_orig_plus_h3r_smpl25k_15000_hdens_cap200k_scene3000/2026-05-18_03-26-15`
