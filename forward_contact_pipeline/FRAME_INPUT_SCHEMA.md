# FrameInput 数据格式

导出工具：[`export_frame_input.py`](./export_frame_input.py)

## 运行示例

```bash
TORCH_HOME=/tmp/hugs_torchhome \
python export_frame_input.py \
  --human3r-root ../Human3R \
  --model-path ../Human3R/src/human3r_896L.pth \
  --seq-path ../Human3R/examples/GoodMornin1.mp4 \
  --output-dir datasets/frameinput_goodmornin_10f \
  --device cuda --size 512 --max-frames 10
```

## 输出结构

```text
datasets/<sequence>/
├── manifest.json
├── frames/000000.npz
└── human3r_raw/
    ├── color/
    ├── depth/
    ├── conf/
    ├── camera/
    └── smpl/
```

`frames/XXXXXX.npz` 是接触 pipeline 的统一读取接口，包含 `color_rgb`、`depth`、`confidence`、`point_map`、`camera_pose_c2w`、`intrinsics`、`smpl_shape`、`smpl_rotvec`、`smpl_transl`、`smpl_expression`、`smpl_scores` 和 `smpl_mask`。

`manifest.json` 保存输入视频 FPS、源帧号、时间戳、帧数、checkpoint、推理速度和坐标约定。当前 `alignment_status` 为 `raw_unaligned`，后续 P0.3 再加入 Sim(3) 对齐结果。

