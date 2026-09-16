# 前馈人体-场景接触修正 Pipeline

> **2026-09-14新增入口**：可配置的Stage-A训练、验证、续训及缓存契约见
> [TRAINING.md](TRAINING.md)；双机代码/依赖迁移见[迁移说明](../MIGRATION.md)。
> 下方保留既有原型用法，不表示RICH真实前端、dense标签审计或GUSH3R LBS绑定已完成。

本目录实现更新后方案中的 Human3R 几何预验证、规则接触 baseline、GRU/TCN 训练验证、GUSH3R/HUGS Gaussian 适配和低维 contact packet。正式论文主实验仍是：

```text
原始 GUSH3R vs. GUSH3R + 接触修正层
```

## 目录

```text
contact_streaming/              核心 Python package
  alignment.py                  robust Sim(3) 与低置信度 fallback
  anchors.py                    Human3R SMPL-X 脚部 anchor
  surface_proxy.py              局部平面/点云 proxy
  rules.py                      hysteresis、规则 contact、几何修正
  features.py                   58D 因果时序特征
  data.py                       sequence-safe window dataset
  models.py                     GRU 与 causal TCN
  losses.py                     接触/表面/穿透/姿态/不确定性损失
  metrics.py                    接触与几何指标
  gaussian_adapter.py           通用 LBS/Gaussian 接口
  packet.py                     fp16 contact packet
  runtime.py                    在线因果推理
scripts/
  prepare_sequence.py           FrameInput -> 对齐几何/训练序列
  attach_labels.py              合并 BEDLAM/PROX/NeuMan 标签
  build_splits.py               按序列划分 train/val/test
  train_contact.py              GRU/TCN 训练
  evaluate_contact.py           验证/测试
  infer_sequence.py             因果推理和 packet 导出
  run_rule_baseline.py          规则 baseline 指标
  apply_gaussian_correction.py  Gaussian interchange 测试
  simulate_stream.py            带宽、延迟和丢包模拟
  make_synthetic_dataset.py     无真实数据时的端到端测试数据
```

## 1. Human3R 真实序列

先使用现有导出器生成 FrameInput：

```bash
python export_frame_input.py \
  --human3r-root ../Human3R \
  --seq-path ../Human3R/examples/GoodMornin1.mp4 \
  --output-dir datasets/goodmornin
```

再生成几何和训练数据：

```bash
python scripts/prepare_sequence.py \
  --input datasets/goodmornin \
  --output prepared/goodmornin.npz \
  --smplx-model-dir ../Human3R/src/models \
  --device cpu
```

没有显式 3D 对应点时，脚本只使用低置信度 translation fallback，并会进入 `Uncertain`；正式实验应通过 `--correspondences alignment.npz` 提供 `src`/`dst` 对应点，启用 robust Sim(3)。低置信度帧不会强行执行 IK。

### NeuMan P0：有外部场景参考的坐标校准

`GoodMornin1.mp4` 只能做无参考工程 smoke test，不能证明绝对坐标正确。
P0 使用同一 NeuMan 帧的 Human3R 结果与 NeuMan 已对齐的
`4d_humans/smpl_optimized_aligned_scale.npz` 建立对应点；两者的同帧、同
SMPL 顶点是 Sim(3) 参考，这**不是**接触真值或接触标签。

先在可见 GPU 的 Human3R 环境中对 NeuMan 图片运行 `export_frame_input.py`，
并将其 raw 结果转为 HUGS SMPL 参数（必要时用
`scripts/fit_smpl_to_human3r_smplx.py` 做 SMPL-X→SMPL 拟合）。然后生成带
frame-level hold-out 验证的对应点：

```bash
cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work

/workspace/nas_auto_backup/yuzilang/miniconda3/envs/hugs/bin/python \
  forward_contact_pipeline/scripts/build_neuman_p0_correspondences.py \
  --source-smpl data/human3r_hugs/neuman_lab_fitted/4d_humans/smpl_optimized_aligned_scale.npz \
  --target-smpl data/neuman/dataset/lab/4d_humans/smpl_optimized_aligned_scale.npz \
  --out forward_contact_pipeline/datasets/neuman_lab_p0/correspondences.npz
```

该命令同时写出 `correspondences.json`。必须检查其中的
`holdout_vertex_error`，不能只看参与拟合的 training error。将输出传入
`prepare_sequence.py`：

```bash
/workspace/nas_auto_backup/yuzilang/miniconda3/envs/hugs/bin/python \
  forward_contact_pipeline/scripts/prepare_sequence.py \
  --input forward_contact_pipeline/datasets/neuman_lab_frameinput \
  --correspondences forward_contact_pipeline/datasets/neuman_lab_p0/correspondences.npz \
  --output forward_contact_pipeline/prepared/neuman_lab_p0.npz \
  --smplx-model-dir Human3R/src/models
```

`prepare_sequence.py` 会对**anchor 与 point map 同时应用**该 Sim(3)，之后再
拟合局部 surface proxy；若只变换人体、不变换 point map，脚到地面的距离没有
几何意义。P0 接下来仍需检查 mesh/mask 重投影和 proxy 可视化，才允许进入 P1。

## 2. BEDLAM/PROX 数据准备

每个视频序列先离线运行 Human3R，再运行 `prepare_sequence.py`。数据集 GT 统一保存为 NPZ，可包含以下字段：

```text
contact [T,2]
contact_point [T,2,3]
surface_distance [T,2]
surface_normal [T,2,3]
pose_residual [T,12]
root_residual [T,6]
penetration_risk [T,2]
uncertainty [T,2]
```

合并 GT：

```bash
python scripts/attach_labels.py \
  --sequence prepared/bedlam_seq.npz \
  --labels labels/bedlam_seq.npz \
  --label-source BEDLAM_GT \
  --output learning/bedlam__seq.npz
```

按完整序列划分，禁止相邻帧泄漏：

```bash
python scripts/build_splits.py --data learning --group-prefix
```

## 3. 训练与验证

```bash
python scripts/train_contact.py \
  --data learning --output runs/contact_gru \
  --model gru --history 8 --hidden-dim 128 --epochs 30

python scripts/evaluate_contact.py \
  --data learning --checkpoint runs/contact_gru/best.pt \
  --split test --output runs/contact_gru/test_metrics.json
```

TCN 对照只需将 `--model gru` 改为 `--model tcn --layers 3`。

## 4. 在线推理和传输

```bash
python scripts/infer_sequence.py \
  --sequence learning/neuman__lab.npz \
  --checkpoint runs/contact_gru/best.pt \
  --output runs/contact_gru/neuman_lab

python scripts/simulate_stream.py \
  --packets runs/contact_gru/neuman_lab/packets \
  --loss-rate 0.05 --latency-ms 30 --jitter-ms 10
```

当前 packet 固定为 62 bytes/frame，包括 root residual、12D foot residual、左右脚 confidence/uncertainty 和状态 flags，显著低于 10 KB 目标。

## 5. GUSH3R/HUGS 接入边界

`gaussian_adapter.py` 提供通用 LBS transform blending。正式 GUSH3R 接入应复用：

```text
GUSH3R SMPLX_Mesh.get_target_transform
GUSH3R SMPLX_Mesh.pose_points_from_zero
GUSH3R HumanGSHead / GaussianRenderer
```

接触模块只修改 corrected SMPL-X/root，scene Gaussian 颜色、尺度和 opacity 保持不变。`apply_gaussian_correction.py` 只用于检查 NPZ interchange 和 root correction，不替代正式 GUSH3R LBS 渲染。

## 6. Smoke test

```bash
python -m unittest discover -s tests -v
python scripts/make_synthetic_dataset.py --output /tmp/contact_synth
python scripts/train_contact.py --data /tmp/contact_synth --output /tmp/contact_run --epochs 2 --device cpu
python scripts/evaluate_contact.py --data /tmp/contact_synth --checkpoint /tmp/contact_run/best.pt --device cpu
```
