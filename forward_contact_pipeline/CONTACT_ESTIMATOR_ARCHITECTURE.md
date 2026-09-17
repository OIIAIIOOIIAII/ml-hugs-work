# 最终在线接触估计器架构

> 状态：2026-09-09。本文定义第二研究点的最终系统契约。当前 58D PROX GRU 是其中的 `Temporal Controller` 机制原型，**不是**最终接触估计器。

> **实现状态补充（2026-09-15）**：Stage-A已经有可执行的几何基线与RGB/Gaussian可选融合基线、统一数值训练入口。实装结构是共享point MLP→按有效mask平均汇聚→拼接vertex位置/法向→vertex MLP；融合版追加RGB cross-attention，默认hidden=96、heads=4。图网络/Point Transformer属于候选扩展，当前并未实现。当前统一损失已接contact，proximity头存在但默认权重为0；几何原型的reliability头尚未在统一训练层建立监督/校准，融合版尚无可靠性头。RICH GT整理已完成，真实前端特征export/审计与Stage-B/LBS完整绑定仍未接通。下文“最终网络”是目标契约，不能据此宣称所有输出已经训练可用。

> **全量训练状态补充（2026-09-17）**：`rich_full_contact_v1` 已开始真实全量 RICH Stage-A 训练。它使用冻结 GUSH3R 的每人预测、64 个脚底 vertex、每 vertex 的16个预测 point-map 邻居、25个局部与16个全局 DINO RGB token、pose/query/物理相机条件；输出仅为每脚底 vertex 的 contact logit。首轮边生成并持久化这些非GT输入边训练，随后 epoch 复用缓存。GT body/contact 只用于将预测目标互斥地关联到标签及计算损失；没有作为模型输入。该运行不包含 Stage-B、修正量、LBS 回写或官方 test，因此首轮结果只能作为 Stage-A 内部开发证据。

## 1. 工作目标

给定流式视频的当前 RGB 帧 `I_t`，以及到 `t-1` 为止模型自身的接触控制状态，系统以一个**冻结的前馈人体—场景 Gaussian backbone** 为基础，在线输出当前帧的接触估计与低维修正量。

```text
I_t + state_{t-1}
  -> frozen feed-forward Gaussian reconstruction
  -> coarse aligned scene/human Gaussians + SMPL-X + visual tokens
  -> contact estimator
  -> contact / uncertainty / correction at t
  -> corrected SMPL-X -> LBS update human Gaussians
  -> corrected human Gaussians + unchanged scene Gaussians
```

目标不是重新优化或移动所有 scene Gaussians，也不是取代前馈重建器；目标是修正其在接触区域的低维人体状态，使在线 rendering 具有更少 sliding、penetration 和接触跳变。

### 当前 GUSH3R 具体实例与接口边界（2026-09-10 更正）

GUSH3R 不是“外接 HumanGS”的拼接系统。它冻结已有 Human3R backbone，并由 GUSH3R 作者训练两条自主 decoder：Scene Gaussian Decoder 和 Human Gaussian Decoder。后者在官方代码中命名为 `HumanGSHead`，论文中称 Human Gaussian Decoder / Human Gaussian Transformer (HGT)。它以 Human3R 的 SMPL-X mesh、human token、image token 为输入，在 canonical body space 的 SMPL-X semantic anchors 上预测人体 Gaussian；其 per-person appearance memory 通过 SMPL-X matching 跨帧维护。LHM 是 GUSH3R 论文的外部 baseline，不是该模块。

所以在本架构中，GUSH3R 可提供 `Z_t, G^S_t, G^H_t, q_t`，而我们不改动其 scene decoder。其人体 decoder 内部确实通过 SMPL-X/LBS pose query points，但当前导出的 contact state 尚未包含逐 Gaussian 的 query-point/LBS binding。接触执行层若要只重算 LBS 而不重跑完整 decoder，必须先补充该 binding export；在此之前，`corrected SMPL-X -> human Gaussian` 是待实现接口，不得描述为已完成。

## 2. 因果输入—输出契约

### 当前帧输入（允许）

| 输入 | 来源 | 作用 |
|---|---|---|
| `I_t` 当前 RGB | 采集流 | 提供遮挡、脚底边界、接触外观和细节证据；不使用未来帧 |
| `Z_t` visual tokens | 冻结 backbone 的 image feature / 人体 crop feature | 复用前馈视觉表征，避免从小型 PROX 数据重新训练大视觉编码器 |
| `G^S_t` coarse scene Gaussians 或 point-map/depth proxy | 冻结 scene branch | 在每个 foot/hand ROI 提取局部点、normal、distance、support、opacity/view confidence |
| `G^H_t, q_t` human Gaussians + SMPL-X | 冻结 human branch | 得到 mesh vertices、脚底/手部 ROI、root 与关节状态 |
| `s_{t-1}` | 模型自身持久状态 | 只包含上一帧的 contact probability、已施加 action、episode anchor、uncertainty/reset；禁止 teacher state |

### 当前帧输出

```text
C_t:  vertex/ROI contact probability（先脚—地，后扩至手—物/身体—场景）
P_t:  continuous proximity / signed-distance residual 与局部法向可信度
U_t:  geometry/model uncertainty、abstain/reset probability
Δ_t:  bounded low-dimensional root/foot/ankle correction
```

执行层根据 `C_t, U_t` 将 `Δ_t` 投影到可信局部切平面、限幅、施加或 abstain。然后通过 SMPL-X FK/IK 与 LBS 只更新 `G^H_t`；`G^S_t` 保持不变。

## 3. 最终网络：两层结构

```text
                         current frame t
I_t ── frozen RGB/Gaussian backbone ── Z_t, G^S_t, G^H_t, q_t
                                         │
SMPL-X ROI vertices V_t ────────────────┼───── local contact frame
scene Gaussian / point patch P_t ───────┘
                                         ↓
Stage A: RGB–mesh–point Local Contact Encoder
  human ROI vertex tokens + local scene point/Gaussian tokens + RGB crop tokens
  mesh graph / vertex MLP + PointNet/small Point Transformer + cross-attention
                                         ↓
  per-vertex contact, continuous proximity, normal/distance confidence
                                         ↓ (ROI pooling)
Stage B: Causal Contact Controller
  current ROI contact tokens + q_t/root + s_{t-1}
  small GRU/causal TCN (H=8) + action/uncertainty heads
                                         ↓
  C_t, U_t, Δ_t, abstain/reset -> corrected SMPL-X -> human Gaussian LBS
```

### Stage A：局部接触编码器

第一版仅左右脚。对每个脚底采样 64--128 个 SMPL-X ROI vertices；每个 vertex 取 128--256 个 scene point/Gaussian proxy 邻居，并全部转换到局部接触坐标系（surface normal + tangent basis）。

每个 point token 至少含相对位置、normal、distance、support/opacity/view confidence、point-map confidence；每个 vertex token 含相对位置、法向、body-part embedding、SMPL pose feature。`Z_t` 提供 RGB crop/image token，通过 cross-attention 处理遮挡或仅靠几何无法区分的接触。

Stage A 输出 vertex-level contact logit、continuous proximity、estimated normal/distance reliability。它借鉴 RICH/DECO/GraphiContact 的 dense body-surface contact、CONTHO 的“粗几何指导 contact 再反哺 refinement”、LEXIS 的连续 proximity，而不是把接触简化成两个 anchor 的二分类。

### Stage B：因果控制器

将每个 ROI 的 contact/proximity/confidence token 池化为脚/手 token，再与 root/pose dynamics 和 `s_{t-1}` 输入小型 GRU。它只处理时序 hysteresis、episode anchor、action smoothness 与风险回退；它不负责从绝对 world coordinate 猜测接触。

当前 58D GRU 可以保留为 Stage B 的 ablation，但正式版输入必须改为 contact-local invariant token，去掉可记忆房间/序列的绝对位置捷径。

## 4. 训练策略

1. **冻结 backbone，训练 Stage A+B**：PROX/RICH 等 mesh/SDF 监督产生 vertex contact、proximity、normal/distance reliability；训练 source-sequence、scene、error-family 三重隔离。
2. **前馈误差模拟 + scheduled rollout**：将真实 backbone 误差统计与多族模拟误差混合，训练时逐步用模型自身状态替换 teacher state。
3. **E2 uncertainty calibration**：point-map/Gaussian proxy 的 normal error、distance bias、occlusion/dropout、delay 都有独立 severity；只在 validation 固定 abstention threshold。
4. **轻量 human adapter（最后）**：仅在 Stage A+B 达标后，解冻 human Gaussian decoder/deformation adapter 最后一层；scene branch 始终冻结。

## 5. 运行时状态与传输

`s_t` 必须是小包：contact probabilities、episode anchors、applied low-D action、uncertainty/reset。当前 RGB 和高斯由正常前馈/渲染链处理；接触包只作为低带宽实时增量。可按 PointSplat 类 compact representation 和 GaussianLens 类 contact ROI budget 提升传输/渲染效率，但它们不是接触监督的替代物。

## 6. 现有原型与最终版的差异

| 项目 | 当前 E1/E2 原型 | 最终系统 |
|---|---|---|
| 当前 RGB | 无，使用 PROX 几何教师/退化 proxy | `I_t` 经冻结 backbone token 明确输入 |
| 接触空间 | 两个脚 anchor | mesh ROI vertex-level contact/proximity |
| 场景输入 | 手工 58D point/normal/distance | local Gaussian/point patch + RGB relation encoding |
| 时序 | causal GRU | 保留 causal GRU/TCN，但接收 relation tokens 与 `s_{t-1}` |
| 人体修正 | anchor action 机制 | contact-gated root/ankle IK -> SMPL-X -> LBS human Gaussian |
| 场景更新 | 无 | 无；scene Gaussian 始终冻结 |

因此当前实验只能回答“Stage B 的 anchor-lock controller 能否被训练”；Stage A、真实 RGB 输入、跨 scene/error 泛化和 LBS render 是下一阶段必须完成的工作。
