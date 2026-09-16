# 前馈方案 Baseline 归档

本目录保存前馈人体-场景 Gaussian 方案的基线资料和可复现结果。这里的 baseline 指“未进行接触修正”的原始前馈输出，当前包含 Human3R 与 GUSH3R 单帧 smoke test。

## 目录结构

```text
baseline/
├── README.md
└── results/
    ├── human3r_smoke_gpu/
    │   ├── camera/000000.npz
    │   ├── color/000000.png
    │   ├── conf/000000.npy
    │   ├── depth/000000.npy
    │   └── smpl/000000.npz
    └── gush3r_smoke/
        ├── merged_render/frame_000000.png
        └── render.mp4
```

## Human3R baseline

- 输入：`Human3R/examples/GoodMornin1.mp4`
- 测试：单帧 GPU 前向
- 输出：深度、置信度、SMPL-X、相机和颜色图
- 原始结果：`results/human3r_smoke_gpu/`
- 历史临时目录：`/tmp/human3r_smoke_gpu/`

## GUSH3R baseline

- 输入：`Human3R/examples/GoodMornin1.mp4`
- 测试：`--max_frames 1 --size 512`
- 输出：人体/场景 Gaussian 合成渲染结果
- 原始结果：`results/gush3r_smoke/`
- 历史临时目录：`/tmp/gush3r_final_smoke/`
- 最终验证：`Inference finished in 0.79s`

运行时需要使用已经配置好的 `gush3r` 环境、CUDA RoPE2D、xFormers，以及本地 DINOv2 cache。GUSH3R 的 sampling cache 已固定写入模型实际读取的运行时路径，避免启动时重新执行 FPS/KMeans。

## 基线使用约定

后续接触 proxy、规则投影、GRU/TCN 和 IK 结果分别放入 `results/<方法名>/`，不要覆盖这两个原始前馈目录。每个新方法至少附带输入、checkpoint、命令、环境和输出文件说明。

