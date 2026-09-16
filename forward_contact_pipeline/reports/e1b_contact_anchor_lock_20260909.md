# E1b：接触锚点保持 / 切向 foot-lock（2026-09-09）

## 目的与范围

E1a 的法向 SDF correction 可降低 penetration，但无法控制脚沿地面漂移。E1b 在真实 PROX teacher-contact 区间向脚 anchor 注入**仅切向**的 AR drift，并在每个接触 episode 开始处记录 clean `anchor_world`。这是 oracle-scene synthetic mechanism experiment；没有 Human3R、GUSH3R、真实 Gaussian 或真实前端输入。

目标定义：接触期间 `delta_foot = anchor_world - noisy_foot`；root target 固定为零，以消除 root/foot gauge ambiguity。非接触时 target 为零。

## 数据与执行

- 数据：4 PROX scene、same scene / mutually exclusive drift seed split，24 train / 8 val / 8 test sequences。
- 切向 drift：`foot_error - (foot_error·normal) normal`，且只在 teacher contact frame 生效；因此 normal-only SDF projection 无法完成本任务。
- Oracle executor：接触 begin 保存 clean anchor，hold 期间把脚投回该 anchor。
- 学习对照：causal GRU, history=8, 80 epochs, contact + 5x residual loss；训练 feature 含上一帧 **teacher** residual，故模型结果是 teacher-forced diagnostic，不能描述为在线 rollout。
- Foot sliding 已按 `||p_t-p_{t-1}|| / Δt` 以 m/s 计算。

## 结果

| Test macro mean | Zero correction | Oracle anchor lock | GRU teacher-forced, foot-only, gate 0.85 |
|---|---:|---:|---:|
| Contact F1 | 0.000 | 1.000 | 0.847 |
| Transition F1 | 0.000 | 1.000 | 0.673 |
| Foot sliding | 0.385 m/s | **0.028 m/s** | 1.180 m/s |
| Contact |distance| | 0.711 mm | 0.978 mm | 12.7 mm |
| Correction MAE | 6.78 mm | ~0 mm | 11.5 mm |

## 结论

1. **表示/执行层通过**：若 contact episode 和 anchor 正确，显式 foot-lock 可将滑步降低约 92.7%。这验证了接触 anchor state 是 E1a 缺失的必要状态。
2. **当前 learned controller 不通过**：GRU 在 teacher-forced 前一 correction 的有利条件下仍过度修正，滑步增至 1.18 m/s，且接触距离恶化。因此当前网络绝不能接入真实系统。
3. **下一实验**：实施真正的 causal rollout（上帧使用模型自身 applied residual/anchor state，而非 teacher residual），并把输出改为有界 tangent velocity / anchor increment；采用 multi-step execution loss、stick loss、release loss 和 residual slew-rate penalty。通过条件仍是 penetration、contact distance、foot sliding 同时不恶化。
