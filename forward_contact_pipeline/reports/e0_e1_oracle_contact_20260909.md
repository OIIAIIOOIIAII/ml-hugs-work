# E0/E1 接触修正审计与机制实验（2026-09-09）

## 结论范围

本报告只使用 PROX 的 PROXD + scene SDF 及其人为注入的 synthetic drift；没有解码 RGB/深度图片，也没有调用 Human3R 或 GUSH3R。所有结果均是 **oracle-scene mechanism result**，不能表述为真实前馈人体或 Gaussian 提升。

这里的 GRU/TCN 是已训练的**合成机制原型**，不等于已经训练完成可部署的接触模块：它没有使用 Human3R/GUSH3R 输出作为训练输入，也没有经过真实前端对齐、SMPL-X IK/LBS 或 Gaussian renderer 验证。

## E0：坐标与标签契约

输入目录：`labels/prox_gt_sdf_e0/`。四个 180 帧序列均通过结构、有限值、法向单位长度、SDF/标签字段及 PROX scene-world 元数据审计：

| 序列 | 有效脚部比例 | 左/右 contact rate | 左/右 median |SDF| |
|---|---:|---:|---:|
| BasementSittingBooth_00142_01 | 1.00 | 10.56% / 7.78% | 0.008 / 0.006 mm |
| MPH112_00034_01 | 1.00 | 26.67% / 23.89% | 0.402 / 0.395 mm |
| N0Sofa_00034_01 | 1.00 | 43.33% / 48.33% | 0.253 / 0.035 mm |
| N3Office_00034_01 | 1.00 | 37.22% / 43.89% | 0.437 / 0.494 mm |

坐标系均为 `PROX_scene_world`，标签来源均为 `PROX_GT_SDF_from_PROXD`，法向单位长度平均误差约 `2.2e-8`。审计 JSON：`experiments/e1_oracle_mechanism_20260909/e0_contract_audit.json`。

**资产缺口**：这些早期 E0 标签没有同目录的 `.teacher.npz`，因此可做标签契约审计，但不能从该目录独立重建完整的时序 feature。E1 使用另行已归档的 `prox_gt_sdf_b/*.teacher.npz`。后续重建 E0 标签时必须把 teacher NPZ 与 JSON 一并保存。

## E1：旧随机分解目标（否定性对照）

旧目标将随机注入的共同 root 误差和 per-foot 误差分别作为唯一真值。该分解对局部接触几何并不唯一：同一脚位置误差可由 root 或 foot residual 的不同组合产生。

在同 scene、drift-seed 互斥的 4 scene / 24 train / 8 val / 8 test 机制 split 上，GRU/TCN 的 contact F1 分别为 `0.615/0.694`，但应用 residual 后穿透深度和滑步变差。这排除“只要分类正确，接触修正就有效”的错误解释；该目标不得用于正式训练。

## E1：可辨识的接触法向投影目标

新数据：`learning/prox_synth_projection_v1/`。保留相同 drift、数据、序列与 seed split，但只在 teacher-contact 脚上定义可观测目标：

```text
delta_foot = - observed_signed_distance * teacher_surface_normal
root_delta = active feet delta 的均值
foot_delta = delta_foot - root_delta
```

这消除了随机 root/foot gauge ambiguity。训练采用 causal history=8、hidden=128、80 epochs；为避免辅助 surface heads 压过毫米级 residual，E1 机制口径只训练 contact + 5x pose/root residual。所有模型都是冻结前端的轻量网络。

测试时在验证集预选 `contact probability >= 0.70` 的安全门控，并对每脚 correction 施加 8 cm 上限。选阈只使用 validation，test 未参与选择。

| Test（宏平均） | 未修正 | GRU + gate | Oracle target |
|---|---:|---:|---:|
| Contact F1 | 0.000 | 0.715 | 1.000 |
| Transition F1 | 0.000 | 0.138 | 1.000 |
| Penetration ratio（>5 mm） | 56.35% | 52.60% | 24.17% |
| Mean penetration depth | 18.57 mm | **13.95 mm** | 6.87 mm |
| Contact |distance| | 29.28 mm | **24.58 mm** | 0.00 mm |
| Foot sliding | 0.643 m/s | **0.927 m/s** | 0.698 m/s |
| Applied correction MAE | 10.60 mm | 10.60 mm | 0.00 mm |

因此 E1 的结论是**部分通过**：网络可以学习到安全门控下的法向 clearance 修正（穿透深度下降 24.9%，接触距离下降 16.1%），但没有学会 contact hold / tangential stick，滑步仍上升约 44.1%。TCN 的分类更高，却会产生更大的 residual，未被选作 correction 候选。

**2026-09-09 口径修正**：最初 `foot_sliding` 未除以相邻 timestamp，单位应为 m/frame 而非 m/s；评测已修为 m/s 并重跑上述数字。相对结论不变，旧的 20.99/30.19 mm/s 数字废弃。

下一步不是接入 GUSH3R，而是为 E1 增加“接触锚点保持”的可辨识切向目标、相邻帧 correction/velocity loss、预测-执行闭环，并把滑步不恶化作为 E1 通过门槛。

## 指标词典与含义

| 指标 | 定义 | 它回答的问题 | 趋势 |
|---|---|---|---|
| Contact precision / recall / F1 | 逐脚二分类接触状态 | 是否识别正确接触 | 高更好；必须同时看 P/R，避免 all-positive |
| Transition F1 | 接触进入/退出时刻的 F1，按序列计算 | 接触开关是否抖动或错时 | 高更好 |
| Penetration ratio (>5 mm) | SDF < -5 mm 的脚帧比例 | 是否真实穿入场景，而非数值噪声 | 低更好 |
| Mean penetration depth | `mean(max(-SDF, 0))` | 穿入有多深 | 低更好 |
| Contact |distance| | 接触 GT 帧上 `abs(SDF)` | 接触脚是否贴到表面 | 低更好 |
| Foot sliding | 接触 GT 帧的相邻 anchor 速度 | 脚接触后是否仍在地面滑动 | 低更好 |
| Correction MAE | 修正后脚 anchor 与 oracle corrected anchor 的误差 | 低维 correction 是否估对 | 低更好 |
| Root correction RMSE | 预测与 teacher root residual 的 RMSE | root 分支是否稳定；须结合 gauge 定义解释 | 低更好 |
| Uncertainty calibration（E2） | 置信度与实际错误/风险的一致性 | 是否会在坏 scene 上拒绝错误修正 | 低风险高覆盖更好 |
| Contact ROI PSNR/SSIM/LPIPS（E4） | 脚—地面区域的渲染质量 | LBS human Gaussian 修正有无伤害局部视觉 | PSNR/SSIM 高、LPIPS 低 |

不能只报告 F1：本轮 TCN 就是反例——分类更高，但执行 correction 后几何更差。E1/E2 的模型选择必须优先满足“penetration、distance、sliding 不恶化”，再比较分类指标。
