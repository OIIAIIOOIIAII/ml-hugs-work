# 在线人体—场景 Gaussian 接触一致性：总体方案与执行路径

> 状态：2026-09-09。本文是第二研究点的系统级路线图；`CONTACT_CORRECTION_EXPERIMENT_PLAN.md` 保留为实验协议，`TODO.md` 保留为工程勾选清单。所有主张必须标注为 mechanism / robustness / real-front-end / renderer diagnostic 四类之一。

## 1. 我们最终要解决的不是“重建”，而是跨表示的在线一致性

前馈重建器可以快速从当前 RGB 给出人体和场景，但它们通常分别优化：人体姿态可能视觉合理却脚底漂移，场景 Gaussian 可能渲染好却不是可靠表面。逐帧优化全部 Gaussian 太慢且会破坏在线性。

因此研究问题固定为：**给定当前 RGB 帧和过去的模型自身 contact/control state，在一个冻结的前馈人体—场景 Gaussian 重建器之后，能否通过 RGB–mesh–point 局部接触估计与因果控制，只修改低维 SMPL-X root/foot，再由 LBS 更新 canonical human Gaussians，从而在场景不可靠时安全回退，并保持实时传输？**

这不是和 Splat-SAP/HumanGS 比“谁的单帧图更清晰”。它们分别解决场景几何与人体 canonical 表示；我们研究的是二者之间、在线且不确定的物理/时序一致性层。

## 2. 最终系统

```text
当前 RGB 流 I_t（按部署版本可为单目、双目或稀疏多视图） + state_{t-1}
  ├─ GUSH3R concrete front-end（当前候选）
  │    frozen Human3R backbone -> human token/image token/SMPL-X mesh/scene point cloud
  │    GUSH3R-native Scene Gaussian Decoder -> scene Gaussians
  │    GUSH3R-native Human Gaussian Decoder (HGT, code name HumanGSHead)
  │      -> human Gaussians + rendered alpha mask + per-person appearance memory
  │    -> left/right foot anchors, local scene proxy, visibility/confidence
  └─ RGB–mesh–point local contact encoder + causal controller (H=8)
       state: previous contact probability, applied tangent action,
              episode anchor, uncertainty/reset state
       output: vertex/ROI contact probability, geometry uncertainty, tangent foot/root correction,
               uncertainty, abstain/reset
             ↓
       corrected SMPL-X -> LBS / deformation -> human Gaussians
             ↓
       renderer + contact-ROI budget + compact state packet
```

执行原则：场景 Gaussian **不被**接触模块逐帧移动；人体仅接受有界的低维修正。局部几何低置信或不确定性高时，系统输出原始前馈姿态并发送 `abstain/reset`，而不是把人体硬吸到错误表面。

## 3. 与近期工作的分工

| 工作 | 在我们系统中的位置 | 采用什么 | 不采用什么 |
|---|---|---|---|
| Splat-SAP（AAAI 2026） | 场景几何候选 | scale-aware point map、两阶段几何 refinement、点图 confidence | 不把 target-view Gaussian plane 当作长时状态，也不声称其解决接触；当前没有官方代码可复现 |
| HumanGS（2026） | 人体表示候选 | SMPL-X vertex back-projection、canonical human Gaussians、LBS 动画 | 不让其承担场景 geometry 或接触决策；接入前先核验代码/数据/许可 |
| Hand-4DGS | 后续手物扩展 | mesh prior、时序建模、遮挡状态 | 不作为完整人景/脚地 baseline |
| PointSplat | 实时传输候选 | compact point/Gaussian 表示、冗余剔除 | 不取代 scene surface teacher |
| GaussianLens | 渲染资源策略 | contact ROI 按需细化与预算 | 不在 E1/E2 以图像质量掩盖几何失败 |
| GUSH3R | 当前人景 renderer diagnostic | 短窗口 Gaussian 产物、局部退化场景 proxy | 不作为当前在线主表：长序列背景 state 漂移尚未解决 |

## 4. 已验证证据与由此锁定的设计

| 证据 | 数值/现象 | 锁定的设计决策 |
|---|---|---|
| E0 | 4 PROX 序列、720 帧，SDF/法向契约有效 | PROX SDF 可作为几何教师；所有样本保留坐标系、source、confidence |
| Human3R -> PROXD | hold-out median 7.37 cm，p95 45.1 cm | 当前 Human3R 不能直接提供真实接触监督，E3 前必须通过 camera-local/重投影门槛 |
| E1 normal clearance | penetration 18.57 -> 13.95 mm，但滑步恶化 | clearance 与 stick 必须解耦；不使用不可辨识的随机 root/foot residual target |
| E1b oracle anchor | sliding 0.385 -> 0.028 m/s | contact episode anchor 与 tangential foot-lock 的表示/执行层成立 |
| E1b causal rollout | 40 mm soft action: 0.385 -> 0.218 m/s，contact F1 0.817 | 使用模型自有历史状态、切平面投影、soft action gate；禁止 teacher previous residual |
| E1b trade-off | BCE×1.5: F1 0.849, transition 0.416, sliding 0.259；BCE×2: F1 0.863, transition 0.550, sliding 0.281 | 默认保留两个 Pareto checkpoint：`control-first`=BCE×1，`state-first`=BCE×2；不要伪造唯一最优 |
| E2 medium frozen | sliding 0.385 -> 0.463 m/s；confidence gate 只能靠 100% abstain 回到基线 | 手工 confidence threshold 不足；必须学习/校准局部几何 reliability |
| E2 medium augmented | sliding 0.385 -> 0.276 m/s，但 F1=0.722、transition F1=0.142 | 增强可恢复部分控制，不能替代 vertex-level relation 与显式 uncertainty |
| GUSH3R 长时序 | anchor handoff 无法消除窗口 seam/background drift | 将 GUSH3R 降级为 E4 renderer/退化 proxy，不绑定主贡献 |

以上 E1 仅是 `PROXD + oracle SDF + injected drift` 的 **mechanism result**，尚不是 Human3R、Splat-SAP、HumanGS 或 GUSH3R 的真实前端结果。

完整输入—输出与两层网络定义见 `CONTACT_ESTIMATOR_ARCHITECTURE.md`。

### GUSH3R 与外部人体 Gaussian 工作的术语边界（2026-09-10）

GUSH3R 论文将 Scene Gaussian Decoder 与 Human Gaussian Decoder 明确为其新引入的两条 decoder；官方实现中的 `HumanGSHead` 只是后者的类名。它基于冻结 Human3R 的 human/image token 和 SMPL-X mesh，在 canonical body anchors 上用 HGT 预测人体 Gaussian，并维护 per-person appearance memory。**它不是把外部同名 HumanGS 工作接入 GUSH3R。** 论文中的外部人体重建 baseline 是 LHM，与 AnySplat、Human3R 组成 `AnySplat+LHM+Human3R` 的后处理拼接管线。

对本项目的含义是：GUSH3R 的两个 decoder 是当前可冻结的 coarse front-end；研究点二从其输出的 local relation 开始。其公开 inference state 还未导出 human Gaussian 到 SMPL-X query point/LBS transform 的稳定 binding，因此接触修正后仅以 LBS 更新 human Gaussian 是待实现的 E4 adapter，不能称为现成能力。

## 5. 模型、状态与损失的具体定义

### 5.1 输入与状态

每帧左右脚各有 anchor position/velocity/orientation、surface point/normal/signed distance/confidence、visibility、人体 alignment confidence；共享 root、时间戳与可选 image token。历史窗口 `H=8`。

可部署状态只有：上一帧 `p_contact`、已施加 action、接触 episode anchor、uncertainty/Reset flag。训练、验证和测试都必须用模型自身状态滚动；teacher state 只用于离线标签，绝不进入部署 feature。

### 5.2 输出与执行

- `p_contact[L,R]`：接触概率；
- `a_tangent[L,R,3]`：经场景法向投影的切向 action，`a_max*tanh(.)` 限幅；
- `u[L,R]` / `abstain`：几何与模型不确定性；
- 后续 E3/E4 才加入可微 root/ankle IK residual。当前 E1b root action 固定为零，以消除 root-foot gauge ambiguity。

默认执行用连续 `p_contact * a_tangent` 的 soft gate，这与训练一致；hard threshold 只能作为单独 deployment ablation，不能混入主结果。

### 5.3 损失

```text
L = λc L_contact(BCE)
  + λa L_anchor(SmoothL1)
  + λs L_stick(contact-period velocity)
  + λr L_release(non-contact action)
  + λΔ L_slew(action difference)
  + λu L_uncertainty(E2 onward)
```

E1b 的已验证 Pareto 说明 `λc` 不应只按分类最优选；选择 checkpoint 时同时约束 contact/transition F1 与 sliding ratio。E2 起采用 heteroscedastic NLL 或 calibrated confidence head，并在验证集固定 abstention threshold。

## 6. 分阶段执行路径、产物与停止门槛

| 阶段 | 实际任务 | 主要数据/算力 | 产物 | 前进门槛 / 停止条件 |
|---|---|---|---|---|
| P0（完成） | 坐标/SDF/anchor 标签审计 | PROX CPU | teacher NPZ、audit report | 已完成；保留资产完整性检查 |
| P1（完成） | oracle scene 下 clearance + anchor-lock | PROX synthetic drift，单 4090 | causal rollout checkpoints、test JSON | 已证实 foot-lock mechanism；冻结两张 Pareto checkpoint |
| P2（进行中） | E2 point-map 退化、Stage-A local relation 与不确定性 | 现有 PROX teacher；无需下载大数据 | relation manifest、local encoder、calibration/coverage-risk report | 在中/重退化下，abstain 能提高且错误修正不恶化；否则先修几何 confidence，不接真实前端 |
| P3 | real human front-end error contract | Human3R-on-PROX，单 4090 | camera-local error decomposition、reprojection report | 若误差仍大于接触尺度，只产生 diagnostic；不能训练真实主表 |
| P4 | scene front-end candidate | Splat-SAP official release 后，或可控 point-map baseline | local-proxy adapter、same-protocol E2 benchmark | 仅当 scene proxy 在接触 ROI 的 normal/distance/confidence 达标才可接 E4 |
| P5 | canonical human Gaussian candidate | HumanGS release/复现，或现有 LBS adapter | corrected-SMPL-X -> LBS unit/integration tests | 人体 visual quality/flicker 不劣于冻结 baseline；否则保持 mesh-only contact result |
| P6 | renderer + online packet | 选定 P4/P5；GUSH3R 仅诊断 | contact-ROI render、latency/FPS/packet report | 接触改善不以全图质量崩坏换取；GUSH3R 长时不稳则明确报告 degraded-map robustness |

## 7. 下一批可直接执行任务（按顺序）

1. **冻结 E1b 机制结论**：将 BCE×1、×1.5、×2 的 soft-gate test 汇总到一个 Pareto 表；主 checkpoint 按任务选择：重控制用 ×1，重状态识别用 ×2，论文主结果同时报二者而非 cherry-pick。
2. **冻结 E2 已完成退化构造与负结果**：medium frozen controller 已失败，增强 controller 虽恢复 sliding 至 0.276 m/s，却有 F1=0.722、transition F1=0.142；不得继续以 gate 调参作为主线。
3. **构建 Stage-A relation manifest**：每帧保存当前 RGB 的 backbone token 引用、64--128 个 foot ROI SMPL-X vertices、每 vertex 128--256 个 contact-local scene point/Gaussian 邻居，以及 vertex contact/proximity/normal-distance reliability 标签；split 必须同时隔离 source sequence、scene 和 corruption/error family。
4. **实现可靠性驱动的 Stage A+B**：小型 mesh-vertex MLP/graph + PointNet/Point Transformer + RGB cross-attention 输出 vertex contact、continuous proximity、normal/distance reliability；ROI token 再进入已有 causal controller。报告 calibration、coverage-risk、sliding/penetration-vs-abstention；阈值只在 validation 选定。
5. **Human3R E3 只做误差分解**：分开 camera/root/local-pose error，完成 foot ROI reprojection；没有通过接触尺度门槛时禁止接触训练。
6. **并行跟踪外部发布**：Splat-SAP 目前无官方代码，HumanGS/PointSplat 的代码和许可待核验；不在当前机器盲目下载或重训。代码可用后先做 10--60 帧接口 smoke，再决定是否替换 GUSH3R 的 scene/human branch。

## 8. 论文叙事与正式比较

论文的主张应是：**uncertainty-aware causal contact consistency layer for feed-forward human-centered Gaussian systems**。它的可迁移性来自接口而不是绑定某个 renderer：Splat-SAP-like point map、HumanGS-like canonical human GS、或任何能输出 SMPL-X 与 local proxy 的前端均可接入。

正式表必须分开：

1. mechanism table：PROX oracle/degraded proxy，报告 contact/transition F1、sliding、penetration、abstention；
2. real-front-end table：只在 E3 coordinate gate 通过后报告；
3. renderer table：只在 P4/P5 接口稳定后报告 contact ROI PSNR/SSIM/LPIPS、human flicker、全图质量、FPS、latency、packet bytes；
4. 失败/限制表：GUSH3R long-horizon drift、Human3R current alignment error、Splat-SAP binocular dependence 与当前无官方代码。

## 9. 明确不做的事

- 不把 Splat-SAP/HumanGS 未复现的视觉效果写成我们的对比或结果；
- 不用 GUSH3R 背景修复代替接触实验；
- 不用 scene-disjoint all-positive failure 继续无止境调分类阈值；
- 不在 Human3R 对齐失败时伪造真实接触标签；
- 不在 E2 前引入大规模 image token、joint renderer fine-tuning 或 TB 级数据下载。
