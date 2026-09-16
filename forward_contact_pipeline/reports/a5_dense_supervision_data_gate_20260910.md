# A5：真实稠密接触监督数据门槛审计

> 2026-09-10后续修订：最新获取顺序及认证/加速方案见 `rich_fast_download_plan_20260910.md`，上下文恢复见 `hugs_context_restored_20260910.md`。以下原始“sample→test→train/val”表格保留为历史方案，不再作为强制顺序；当前先公共资产与train标注约39.2GB，再接同序列JPG。sample内容是否完整、实际认证机制、Range支持和传输量均待真实资源核验；不能假设所有官方脚本都等价于GET URL加netrc。

## 目的

重启最终 Stage-A `vertex contact` 头之前，核验是否存在可合法使用、可与当前
在线 Gaussian 接口对齐的真实稠密人—场景接触数据。这里的“稠密”指每帧 SMPL-X
顶点级、可同时出现正/负样本的接触标注；它不能由 PROX 当前脚底 ROI 的 2 cm
阈值替代。

## 本地资产结果（2026-09-10）

对当前工作区及已知 NAS 项目根目录进行了仅文件名/目录名的审计（未读图、未下载
新数据）：未发现 RICH、PhySIC 或 EMDB 的数据目录、归档、manifest 或预处理产物。

| 资产 | 本地状态 | 是否可立即训练最终 dense contact head |
|---|---|---|
| PROX / PROXD + SDF | 已有 | 否：可监督连续 proximity / ROI 状态，但当前采样的 vertex binary 标签几乎全正 |
| Human3R-on-PROX | 已有 60 帧诊断 | 否：held-out human geometry median 7.37 cm，未过 2 cm gate |
| GUSH3R-on-PROX | 已有 60 帧 causal state | 否：held-out human geometry median 19.95 cm，未过 2 cm gate |
| UniCon3R demo JSON | 已有网页演示格式 | 否：无模型、weights、几何/坐标与监督来源 |
| RICH / PhySIC / EMDB dense contact 数据 | 未发现 | 否：尚无数据/许可/坐标契约 |

这不是“数据不存在”的外部结论，只是当前机器没有可用资产的事实。禁止凭空将
PROX ROI 标签升级为稠密接触监督，或在没有数据许可和坐标契约时启动大规模下载。

## 到位后必须满足的数据契约

每个序列至少要能建立以下 manifest；缺少任一项，数据只能做调研或诊断，不能进入
Stage-A 主实验：

```text
sequence_id / scene_id / frame_id / timestamp
RGB reference + calibrated camera
SMPL-X topology/version + world-space vertices (or recoverable pose/camera)
static scene mesh / signed-distance query / coordinate unit and handedness
dense vertex contact labels [V] + label source + confidence/visibility
frozen backbone coarse output reference: SMPL-X, visual token, human/scene Gaussian patch
split ownership: source sequence, scene, capture subject, error/front-end family
```

入口审计必须逐帧确认：

1. `contact` 同时存在正、负 vertex；报告每 body part、每 scene、每 sequence 的
   positive rate，拒绝 all-positive / all-negative 片段。
2. 标签的 SMPL-X vertex 顺序与当前人体分支一致；若不一致，必须保存确定性
   correspondence，而不是最近点临时匹配。
3. mesh/SDF、SMPL-X 和相机单位统一为米，并以已知点/重投影做坐标检查。
4. train/val/test 至少 scene、source sequence 和 subject 三重隔离；真实前端误差
   family 不得泄漏。
5. 只以 frozen front-end 的真实粗输出作为输入。GT mesh/SDF 只能生成 label 和
   evaluation，不能泄漏到部署输入或 `s_(t-1)`。

## 恢复顺序

1. 获得并核验一个许可明确的 dense-contact 数据子集（先 1--2 scene、短序列 smoke）。
2. 生成上述 manifest，做 label distribution / topology / coordinate audit；通过后才扩展。
3. 在 frozen backbone coarse state 上比较 point/mesh-only、Gaussian relation token、
   `AnchorAttention + RGB` 三种 Stage-A；不把 Gaussian center 阈值作为对照方法。
4. 先验收 dense contact 的 per-part PR-AUC/F1、continuous proximity 和 calibration；
   再把通过门槛的 ROI token 接入已有 Stage-B causal rollout。
5. 最后才评估 corrected SMPL-X -> human Gaussian LBS 的 sliding、penetration、
   contact jitter、ROI render flicker；scene Gaussian 始终冻结。

## 当前结论

目前不存在可诚实启动最终 dense contact 网络训练的本地数据与真实前端组合。因此下一
阶段保持为**数据契约准备与前端候选跟踪**，而不是继续在 PROX 伪稠密标签、GUSH3R
center distance 或 local-only uncertainty 上调参。E1/E2 仍只是 oracle/proxy 的
mechanism 与 robustness 证据。

## 前馈候选复核

同日对官方 UniCon3R snapshot（`9b248489d3febfb52244f7287c32399a6f502542`）作了
只读 README/目录复核：仓库仍仅含项目页资产，README 仍明确说明 code、checkpoints、
installation 与 inference scripts “will be released soon”。因此它是重要竞品和最终
架构参考（contact prompt + contact-guided latent refinement），但截至本次核验仍不能
用作本地 A3/E3 backbone 或可运行 baseline。远程 refs 未返回可用于判定的新 release
信息，故不据此推断网络端发布状态。

## RICH 可用性复核（2026-09-10）

RICH 官方站点仍可访问，并明确说明数据包含：4K indoor/outdoor multiview video、
markerless-mocap ground-truth 3D bodies、3D body scans、high-resolution 3D scene scans，
以及 accurate body vertex-level contact labels；这正好覆盖最终 Stage-A 所缺的 dense
contact 和 scene-surface 两项。下载页要求注册/登录，许可为个人、非转让的
**non-commercial scientific research / education / artistic** 使用，不允许擅自复制、
共享或再分发。

因此 RICH 是当前的**优先数据源**，而不是只可引用的相关工作。获得有权限的下载后，
第一步不是直接全量训练，而是对一个短序列做 adapter smoke：核验实际交付的 RGB/camera、
body model/topology、world coordinate、scene scan/SDF 和 vertex-contact label 的字段与
帧对齐；确认 contact 正负分布后，再生成本项目 manifest。RICH 可完成 Stage-A dense
contact 和 Stage-B causal correction 的训练/评测，但不自动解决 GUSH3R/Human3R 尚未
达到 2 cm contact-scale gate 的真实 Gaussian-backbone 问题。

### 官方下载包到实验资产的映射

用户已从授权下载页确认官方包与体积。采用 JPG 版 images，**不运行**会占 3 TB/6 TB
的原始 image 下载脚本。分三步获取：

| 阶段 | 下载内容 | 估计体积 | 用途 |
|---|---|---:|---|
| P0 adapter smoke | `Gym_010_dips1` sample sequence | 56 GB | 只验证目录、帧对齐、SMPL-X/contact/scan 字段；不训练、不报告指标 |
| P1 held-out evaluation | test JPG images 250 GB + test SMPL-X 7.3 GB + test contact 23 GB + scans/calibration 1 GB + multiview-to-world 4 KB | 约281 GB | 完成 dataset adapter、冻结评测和真实接触标签分布审计 |
| P2 formal training | train JPG images 500 GB + train SMPL-X 9.2 GB + train contact 29 GB；另加 validation JPG 250 GB + SMPL-X 4.3 GB + contact 13.2 GB | 约806 GB（不重复 scan） | train/val 学习、阈值/abstention 选择；test 永不参与训练选择 |

最终完整训练、验证、测试的 JPG 主数据约 `1.09 TB`（含一次 scene scan/calibration），
远低于原始图像路线；当前服务器约 32 TB 可用空间足够。`BSTRO checkpoint` 不是我们
训练所需；`tsv databases`（107 GB）只在需要和 BSTRO 严格公平比较时下载，先不作为
P0--P2 阻塞。geodesic matrix（379 MB）也只在复现 BSTRO 特定指标/损失时需要。

下载后先校验 archive hash/体积，不能把 6 KB 的认证错误 HTML 当成数据；官方 FAQ 也
指出这种小文件常由认证失败导致。下载命令须由持有授权账号的用户在服务器交互式执行，
不得在对话中传递账号、密码或 cookie。
