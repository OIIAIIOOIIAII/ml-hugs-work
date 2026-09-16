# A3：GUSH3R 前馈 Gaussian 证据接口与 PROX 几何门槛

**状态：已完成接口 smoke 与短序列可行性诊断；未通过接触尺度门槛。**

## 接口实验

在冻结 GUSH3R `infer.py` 增加默认关闭的 `--export_contact_state`：每帧输出
opacity-ranked scene/human Gaussian 的 `xyz/scale/rotation/opacity`，以及小型
SMPL-X/camera 字段。默认不改变原推理；导出不含 RGB 像素，并明确 Gaussian
center 不是 scene surface。

- `final` renderer mode 只将累计 scene map 放在最后一帧 prediction，前帧没有
  online scene state，不能用于因果接触。
- `per-frame` causal map mode 在 PROX MPH112 的 8 帧和 60 帧 smoke 中均为
  **每帧**导出：250,000 scene GS（保存 top-2048）和 10,000 human GS
  （保存 top-2048）。
- 60 帧 inference 完成，证据结构完整；未读取或评价任何输出图像。

因此，GUSH3R 能提供第二研究点所需的 Gaussian evidence 接口，尤其可为
AnchorAttention/interaction field 提供 local patch 的位置、协方差、opacity 和
因果 map state；但这不表示其几何足以作为接触 surface 或 teacher。

## 几何门槛：GUSH3R coarse SMPL-X → PROXD

用 60 帧中的前 45 帧拟合一个 Sim(3)，最后 15 帧完全 held-out。预测采用
GUSH3R 导出的 full-hand SMPL-X，PROXD 采用其 PCA-hand SMPL-X；两者均为
neutral SMPL-X 同拓扑顶点。该实现细节已验证，不混淆手部参数格式。

| 误差 | Train（45帧） | Held-out（15帧） |
|---|---:|---:|
| 顶点 median | 2.14 cm | **19.95 cm** |
| 顶点 mean | 5.02 cm | 23.64 cm |
| 顶点 p95 | 17.42 cm | 54.69 cm |

接触门槛是 2 cm，因此 held-out 明确失败。短 8 帧 smoke 的 2.11 cm 只是短
窗口近似，60 帧结果显示终段几何/坐标漂移严重；不能以短窗口结果替代长序列。

## 决策

1. 保留 GUSH3R evidence export，作为**退化 Gaussian proxy / renderer
   diagnostic** 和后续 AnchorAttention 输入接口。
2. 禁止把 GUSH3R 当前 coarse SMPL-X 或 Gaussian map 当作 PROX 接触的真实
   supervision、surface 或 reliability teacher；不得据此训练 A3 uncertainty。
3. 想进入真正 A3，需要一个能在 PROX 上通过 contact-scale aligned human
   geometry gate 的前馈 backbone，或获取其训练/评测数据坐标契约；否则第二点
   先保持为 PROX oracle mechanism + GUSH3R degraded-map robustness，不能写成
   完整真实前馈在线接触结果。

Artifacts:

- `runs/a3_gush3r_evidence_perframe_60f_v1_20260909/output/contact_state/`
- `runs/a3_gush3r_proxd_alignment_60f_v1_20260909/report.json`
