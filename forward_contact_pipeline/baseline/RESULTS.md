# Baseline 结果登记

| 方法 | 输入 | 帧数 | 分辨率 | 结果目录 | 状态 |
|---|---|---:|---:|---|---|
| Human3R 896L | `Human3R/examples/GoodMornin1.mp4` | 1 | 原始输出尺寸 | `results/human3r_smoke_gpu/` | 前向完成 |
| GUSH3R | `Human3R/examples/GoodMornin1.mp4` | 1 | 512 | `results/gush3r_smoke/` | 前向与渲染完成 |
| GUSH3R (`bg_mask_dilation=0`) | `Human3R/examples/GoodMornin1.mp4` | 1 | 512 | `results/gush3r_bg_dilation0/` | 白边对照完成 |

## 结果文件说明

Human3R 保存了 `depth`、`conf`、`smpl`、`camera` 和 `color` 五类输出。GUSH3R 保存了单帧合成渲染 PNG 和 MP4。

这些是原始前馈 baseline，不包含脚部接触判断、surface proxy、IK、penetration projection 或姿态修正。

`bg_mask_dilation=0` 对照显示人体周边白边明显减弱，支持“人体 mask 膨胀导致背景 Gaussian 被过滤”的判断；外圈白色仍来自 rasterizer 背景覆盖不足。
