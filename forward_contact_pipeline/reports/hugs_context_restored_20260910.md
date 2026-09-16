# HUGS 会话上下文恢复（2026-09-10）

来源会话：`01a06025-0f9a-7ca3-8be1-17ba3a60823f`。已读取其阶段交接、之后至末尾的 RICH 数据讨论，以及当前项目活文档。本文件是可持续引用的工作上下文；不代表修改平台会话归属或将整份历史自动注入模型。

## 交接之后的进展与当前覆盖项

- A5 已审计：原工作区未发现可用 RICH/PhySIC/EMDB 稠密接触数据；优先获取 RICH。
- A1 已通过：12个 PROX relation assets 的局部坐标/normal 数值审计通过，最大单位误差 `2.38e-7`，但不解除 binary contact 标签和真实前端门槛。
- 用户已注册 RICH，并提供官网下载清单。采用 JPG 路线，全部 train/val/test 加人体、接触、公共扫描和标定约1087GB；这些是用户提供的估计量，尚未用真实下载响应核验。
- 既有下载器 `scripts/download_rich_parallel.sh` 使用 wget 文件级并行和断点续传，原版仅做过语法检查；它不能把单个文件拆成多个连接。
- 本轮现场：NAS约32TB可用，系统盘约139GB可用；有wget/curl/tmux/conda，无aria2c。原约定的URL清单与RICH数据目录不存在；这不是整台NAS的全量搜索结论。
- 官网公开页确认数据类型；未登录下载页是登录表单。目前没有真实资源URL或已核验的认证配置，不能报告实测下载速度或Range支持。
- 旧“56GB sample → 全test → train/val”不是最快启动实验的必要顺序。当前建议扫描/标定 + train标注先到位，再接同序列JPG；sample可选，必须检查是否包含完整所需字段，test不用于开发调参。
- 当前具体下载方案见 `rich_fast_download_plan_20260910.md`。本次用户请求是恢复上下文并给出下载方案，不据历史“持续跑实验”自动启动新训练或全量数据传输。

以下保留原会话的阶段交接，属于历史记录；动态环境路径、发布状态与数值结论的适用范围以当前活文档及现场核验为准。

---

## 当前任务

用户要求持续推进 HUGS 第二研究点实验：前馈人体—场景 Gaussian 基础上的在线接触一致性。用户希望阶段结束自动进入下一阶段，不读/展示图片，只看数值和日志；需持续维护 plan、TODO、会话日志。

当前工作目录：

`/workspace/nas_auto_backup/yuzilang/ml-hugs-work`

必维护文档：

- `CLAUDE_SESSION_LOG.md`
- `forward_contact_pipeline/CONTACT_CORRECTION_EXPERIMENT_PLAN.md`
- `forward_contact_pipeline/TODO.md`

## 已收敛的研究目标

最终不是普通 SMPL pose correction，而是：

```text
current RGB I_t + model-owned past state s_(t-1)
→ frozen feed-forward human-scene Gaussian backbone
→ coarse SMPL-X + human/scene Gaussian + visual/relation token
→ Stage A implicit Gaussian interaction field
→ ROI contact / continuous proximity / reliability / uncertainty
→ Stage B causal controller
→ bounded root/foot correction + optional confident contact-ROI Gaussian boundary adaptation
→ SMPL-X LBS updates human Gaussians
→ frozen scene Gaussians render
```

关键原则：

- 研究点一 AnchorAttention 应作为第二点的前端 relation mechanism：SMPL semantic anchors cross-attend local scene Gaussian。
- Gaussian center 不是 surface，不能阈值化 center distance 判接触。
- scene Gaussian 不逐帧移动；只更新 human SMPL-X/LBS Gaussian。
- Gaussian 需作为：
  1. contact evidence（local patch、covariance、opacity、stability）；
  2. execution target（contact ROI human Gaussian adaptation）；
  3. validation object（ROI render、flicker、penetration）。
- `s_(t-1)` 只能是模型自己的 previous contact/action/anchor/uncertainty/reset，禁止 teacher state 或未来帧。

论文定位仍建议：

> Causal Human-Scene Gaussian Reconstruction via Implicit Interaction Fields and Contact-Conditioned Boundary Adaptation.

## 重要已完成实验

### E1/E2：oracle mechanism 与 proxy robustness

- E1b oracle contact anchor causal rollout：
  - sliding `0.385 → 0.218 m/s`（-43.4%），F1 `.817`
  - 是 `PROX oracle SDF + synthetic drift` mechanism result，不是前馈 Gaussian 结果。
- E2 medium proxy：
  - frozen controller 失败：sliding `0.385 → 0.463 m/s`
  - augmented controller恢复至 `.276 m/s`，但 F1 `.722`、transition F1 `.142`
  - 不能称可靠。

### Stage A：PROX 标签问题与连续 proximity

- 当前 PROX hard vertex contact 标签不可信：
  - 64 ROI vertex 在一帧内全同 binary label；
  - strict split F1 `.5965` 等于 all-positive；
  - 禁止用 PROX 报告 dense vertex binary contact F1。
- 改用 vertex continuous signed proximity + ROI contact。
- `stagea_proximity_roi_v2_20260909`：
  - test signed proximity MAE `1.27 mm`
  - zero predictor `1.79 mm`
  - nearest point baseline `12.64 mm`
  - ROI contact仍 all-positive，不能作 contact 结果。
- 位置：
  - `forward_contact_pipeline/reports/stagea_proximity_roi_v2_20260909.md`

### Local-only uncertainty 失败

- heteroscedastic Laplace reliability：
  - v1 hard clamp bug，scale 全饱和，已保留失败资产；
  - v2 unseen distance corruption：lowest 50% MAE `.743 mm`，反而差于全量 `.698 mm`；
  - all-family augmented：50%略好 `.653 vs .716 mm`，但25%更差、uncertainty bins不单调。
- 结论：纯 local mesh/point patch uncertainty 不可靠，禁止接 Stage B。
- 报告：
  - `forward_contact_pipeline/reports/stagea_uncertainty_20260909.md`

## Human3R / GUSH3R 真实前端门槛

### Human3R

已有 Human3R-on-PROX `MPH112_00034_01` 60帧数据：

`forward_contact_pipeline/datasets/prox_mph112_00034_01_h3r_60_v1/`

- Human3R→PROXD held-out Sim(3)：
  - median `7.37 cm`
  - p95 `45.1 cm`
- 即使在拟合训练 correspondence 中，root-aligned local vertex median 仍 `5.78 cm`。
- 禁止作为真实接触监督/真实前端主表。

相关脚本：

- `scripts/audit_h3r_proxd_error_decomposition.py` 需要 `smplx` 默认环境没有，未成功完整运行；
- `scripts/audit_h3r_proxd_stored_correspondence.py` 成功利用已有 correspondence 做 train-frame diagnostic。

### GUSH3R

已为：

`ml-hugs-work/GUSH3R/infer.py`

新增默认关闭接口：

```bash
--export_contact_state
--contact_state_max_gaussians 2048
```

它导出 per-frame numerical evidence：

- scene/human Gaussian：xyz, scale, rotation, opacity
- coarse SMPL-X fields
- 不导出 RGB
- 明确 center 不是 surface。

必须使用历史 GUSH3R 环境：

```text
/workspace/nas_auto_backup/nas/yuzilang/miniconda3/envs/gush3r/bin/python
```

默认 Python 缺 `diff_gaussian_rasterization`，不要重装。

重要发现：

- `--background-render-mode final`：只有最后一帧有累计 scene Gaussian，不可作为 online input。
- `--background-render-mode per-frame`：每帧有因果 scene map，可作为 online evidence interface。
- PROX MPH112 60帧 per-frame export 已完成：
  - 路径：
    `forward_contact_pipeline/runs/a3_gush3r_evidence_perframe_60f_v1_20260909/output/contact_state/`
  - 每帧：250k scene GS（保存 top2048）+ 10k human GS（保存 top2048）。

GUSH3R→PROXD 几何：

- 60帧，前45帧 fit Sim(3)，最后15帧 held-out：
  - median `19.95 cm`
  - p95 `54.69 cm`
- 失败 2cm contact gate。
- 仅能做 Gaussian evidence、renderer diagnostic、degraded-map robustness，不可做接触 teacher。

报告：

- `forward_contact_pipeline/reports/a3_gush3r_prox_feasibility_20260909.md`

### Gaussian center/SDF hard rule 已否定

脚本：

`forward_contact_pipeline/scripts/audit_gush3r_gaussian_sdf_support.py`

结果（已知 failed coordinate contract 下的 diagnostic）：

- held-out scene Gaussian SDF coverage `22.4%`
- valid center |SDF| median `33.9 cm`
- near surface 2cm `9.6%`
- SDF < -5mm `95.6%`

不能把它解释为真实 GUSH scene quality；它严格说明：

> Gaussian center + failed coordinate alignment + SDF threshold 不能定义接触。

报告：

- `forward_contact_pipeline/reports/a4_candidate_and_gaussian_evidence_20260910.md`

## UniCon3R 竞品状态

官方 repo 已确认并 clone：

`ml-hugs-work/baselines/UniCon3R_20260909/`

commit：

`9b248489d3febfb52244f7287c32399a6f502542`

README 明确说：

> Code, checkpoints, installation instructions, and inference scripts will be released soon.

因此：

- 没有可运行代码/weights；
- 不能在 PROX 上复现；
- 不能成为当前可运行 baseline。

其网页 demo JSON 存在：

`docs/static/contact/*/contact_sequence.json`

含每帧 `contactIndices`，约 SMPL-X 10475 vertex dense contact index，能作为未来 RICH/EMDB dense supervision schema 参考；没有预测模型、几何、训练标签或坐标契约，不能用来训练。

## Oracle Gaussian occupancy 表征实验

脚本：

`forward_contact_pipeline/scripts/audit_gaussian_occupancy_evidence.py`

在现有 Stage-A near-contact ROI：

- positive fraction `99.9967%`（|SDF|<=2cm）
- nearest-point Spearman `.986`
- 手工 anisotropic Gaussian occupancy Spearman `.964`
- AUC无意义（全正）。

试图以 `--roi-selection full_leg` 重建 full L_Leg sample：

- 代码已给 `build_stagea_relation_manifest.py` 加 `--roi-selection near_anchor/full_leg`
- MPH112 full-leg 仍全部 |SDF|<=2cm，max约 `1.15cm`。
- 说明当前 PROX segment/SDF sampling contract不提供 contact-vs-noncontact distribution。
- 不要继续调 Gaussian kernel。

## 当前最重要结论

目前真正的阻塞不是接触 controller：

1. Human3R/GUSH3R 人体几何未通过 contact-scale gate；
2. UniCon3R还没发布可运行代码；
3. PROX 当前 ROI 标签/采样不支持稠密 binary contact 研究；
4. local-only uncertainty已失败；
5. Gaussian hard geometry threshold已否定。

因此下一个合理阶段不是继续训练网络，而是：

### A5：真实 dense supervision data gate

需要 RICH / PhySIC 等真正 dense vertex contact 数据：

- 先核验本地是否有 RICH/PhySIC，若无则只做可用性/许可/下载规模审计，不能盲目下TB级数据；
- 建立 manifest：
  - RGB reference
  - SMPL-X vertex correspondence
  - scene mesh/SDF
  - dense contact labels
  - contact label source/confidence
  - sequence/scene split
- 用该数据才可重启：
  - Gaussian evidence vs point/mesh relation comparison；
  - dense contact head；
  - uncertainty calibration。

如果用户要继续自主实验，优先执行本地资产盘点：

```bash
find /workspace/nas_auto_backup/yuzilang -iname '*RICH*' -o -iname '*PhySIC*'
```

但避免读取图片。

## 文件修改提醒

最近新增/修改：

- `GUSH3R/infer.py`：contact-state evidence export。
- `forward_contact_pipeline/scripts/`
  - `audit_gush3r_proxd_alignment.py`
  - `audit_gush3r_gaussian_sdf_support.py`
  - `audit_gaussian_occupancy_evidence.py`
  - `audit_h3r_proxd_error_decomposition.py`
  - `audit_h3r_proxd_stored_correspondence.py`
  - `build_stagea_relation_manifest.py` 新 `--roi-selection`
- 文档：
  - `reports/a3_gush3r_prox_feasibility_20260909.md`
  - `reports/a4_candidate_and_gaussian_evidence_20260910.md`
  - `CONTACT_CORRECTION_EXPERIMENT_PLAN.md`
  - `TODO.md`
  - `CLAUDE_SESSION_LOG.md`

注意：另有其他 agent/用户在 2026-09-10 修改过 `CLAUDE_SESSION_LOG.md`，新增第一研究点面试与高帧率渲染内容。不要覆盖或删除这些内容；新增条目必须插在顶部。

## 用户偏好与约束

- 中文。
- 不读取、不展示图片；只读结构、JSON、日志、数值。
- 不夸大：严格区分 `mechanism result` / `robustness result` / `real-front-end result` / `renderer diagnostic`。
- 新实验目录不可覆盖旧目录。
- GPU共享：启动GPU前检查 `nvidia-smi`。
- 所有工作限制在 `/workspace/nas_auto_backup/yuzilang/`。
- 用户希望自动持续执行，但系统每轮可能需要返回；继续时应直接按 A5 做，不要再问重复问题。
