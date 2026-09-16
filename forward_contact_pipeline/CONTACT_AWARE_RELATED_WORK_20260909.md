# 接触感知前馈重建相关工作与 E1b 路线（2026-09-09）

本笔记按“是否能作为我们 online 人景 Gaussian 的直接 baseline”和“能为接触模块借鉴什么”区分。论文检索截止日期为 2026-09-09；不能因任务表面相似而把离线优化、已知 mesh/SDF 或人-物方法写成公平 online baseline。

## 1. 最相关的技术路线

| 工作 | 表示与接触路线 | 是否纯前馈/在线 | 对我们的可迁移部分 |
|---|---|---|---|
| [GraphiContact, 2026](https://arxiv.org/abs/2603.20310) | 先重建人体 mesh；两个预训练 Transformer 提供人体先验；在 mesh vertex 上预测人-场 contact；训练时模拟遮挡/噪声，并以 token adaptive routing 表示不确定性 | 单图高效 contact perception；不等于人体姿态修正或 scene map | **vertex/contact token + uncertainty-aware routing**；但需我们补 local surface、时序与执行修正 |
| [CRISP, 2026](https://arxiv.org/abs/2512.14696) | 单目视频的点云/深度/normal/flow -> 聚类拟合紧凑平面 primitive；以人体接触推断被遮挡支撑面；RL humanoid controller 做物理 motion tracking | 非纯单一前馈头，含几何管线与 RL；面向 Real2Sim | **不要直接用噪点 point cloud/Gaussian center 当地面；将局部点云转换为稳定平面/primitive，并以 contact 补全遮挡支撑面** |
| [PhySIC, 2025](https://arxiv.org/abs/2510.11649) | 单图 human mesh + dense scene surface + vertex contact；depth/geometry scaffold 后以 contact、penetration、reprojection 联合优化 pose/camera/scale | 非实时；约 9 s joint optimization、<27 s end-to-end | 证明 contact、scale、camera 必须联合考虑；我们把它的慢优化替换为因果低维 residual + uncertainty gate |
| [POSA, 2020](https://arxiv.org/abs/2012.11581) | SMPL-X 每个顶点输出 scene contact probability 与 scene semantic label；VAE 学 body-centric proximity/contact | 已知 scene 下的交互先验，不是视频前馈修正 | **由二元脚接触扩到 body-vertex contact field 的理论基础**；第一版仍限制左右脚 |
| [LEXIS, 2026](https://arxiv.org/abs/2604.20800) | body/object 全表面的连续 proximity `InterFields`，VQ-VAE 离散 interaction signature + diffusion，指导生成/细化 | 重生成/HOI，不是实时视频 | **不要只用 binary contact；保留连续距离/方向场和离散 interaction state** |
| [StackFLOW, 2024](https://arxiv.org/abs/2407.20545) | 人 mesh anchor 与物 mesh anchor 的 dense offset；normalizing-flow 后验；再优化 human pose/object 6D pose | 前端可学习但最后需优化 | **surface-anchor offset 是比单一 distance 更丰富的关系表征**；可用于我们的 contact anchor state |
| [Kinematics-based HOI, 2024](https://arxiv.org/abs/2407.14043) | 单视图视频 CRRNet 识别 contact region；FK + learned IK 将末端驱到接触区域 | 半前馈 IK；人-物 | **contact detector 与 IK/执行层分离**，与我们的 estimator -> gate -> SMPL-X residual 相符 |
| [HOI-TG, 2025](https://arxiv.org/abs/2503.06012) | mesh/object query token + Transformer；图残差聚合 mesh topology，隐式学习局部 interaction | 端到端 mesh 重建；主要人-物 | **局部 mesh topology/graph message passing**；但隐式 contact 不足以提供可靠安全门控 |
| [VolumetricSMPL, 2025](https://arxiv.org/abs/2506.23236) | pose/shape-conditioned人体 SDF，Neural Blend Weights 生成紧凑 decoder，支持可微 contact/collision | body representation，不是完整人景在线系统 | 后期可把脚底 mesh-SDF/scene-SDF 距离替代脆弱最近点；不是本阶段阻塞 |
| [CHORE, 2022](https://arxiv.org/abs/2204.02445) | 人/物双 unsigned distance field + body correspondence/object pose field，再 fit parametric body/template | 单图前端 + 拟合，非实时在线 | joint distance/correspondence field 是强几何表征；但慢 fitting 不符合当前实时约束 |
| [PhysCap, 2020](https://arxiv.org/abs/2008.08880) | CNN contact-event -> IK -> realtime physics optimizer，以环境碰撞、地面、力和动力学稳定运动 | 实时但依赖 physics optimization | **foot contact event + foot-lock + root control** 是 E1b 的关键启发；我们先以轻量 learned residual/投影实现，不引入完整物理仿真 |

## 2. 与 Human3R/GUSH3R 的关系

- **Human3R** 是 online feed-forward human/scene mesh/point-map 前端，但论文定位为“消除 iterative contact-aware refinement”；它没有替我们完成可靠接触执行。
- **GUSH3R** 解决 Gaussian rendering，不解决局部接触状态；其长时背景 state 在本地仍不稳定，因此不应成为 E1 的 scene teacher。
- 上表没有一个现成工作同时满足：单目、online、长时、动态人+场景、可复现 Gaussian、接触修正。因此它们是 component prior 或 baseline family，不是可直接替换的最终 backbone。

## 3. E1b：接触锚点保持与切向 foot-lock（下一实验）

当前 E1a 只学法向 clearance：`delta_n = -d * n`。它可减穿透，却不能阻止沿地面的切向漂移。E1b 明确维护每只脚的接触状态：

```text
state per foot = {contact probability, transition, anchor_world, tangent basis,
                  anchor_age, local-scene confidence, uncertainty}

on contact enter:
    anchor_world <- corrected foot position
while contact hold:
    target foot position <- anchor_world + optional bounded tangential residual
on exit/reset/uncertain:
    release anchor; do not force stick
```

网络仍为 causal GRU/TCN，但输出改为 normal residual、tangent residual/velocity、contact transition 和 uncertainty；执行端只在 `contact & reliable & low-uncertainty` 时启用 foot lock。

建议损失：

```text
L_contact        接触/transition BCE
L_normal         接触帧 |SDF| / penetration
L_stick          contact hold 时 ||Pi_t(p_foot - anchor_world)||
L_tangent_vel    contact hold 时 ||Pi_t(v_foot)||
L_release        non-contact 时不强制 anchor
L_rollout        预测 residual 执行后的多帧几何损失
L_uncertainty    几何退化时正确 abstain（E2 才启用）
```

synthetic 数据必须在**真实 teacher-contact 区间**注入切向 drift，并记录每次 enter 的 clean anchor；不能让网络从不可辨识随机 root/foot 分解中猜答案。E1b 的通过条件是：相对未修正输入，penetration depth、contact distance、foot sliding 三者都下降或至少不恶化；否则不进入 E2/E3。
