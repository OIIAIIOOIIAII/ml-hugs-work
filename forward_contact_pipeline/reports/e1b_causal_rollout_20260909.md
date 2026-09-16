# E1b 因果 rollout：执行记录（2026-09-09）

## 目的

将 E1b 从 teacher-forced 的诊断网络改成部署一致的因果控制器：第 `t` 帧只可使用模型在 `<t` 帧产生的 contact probability 与实际施加的 correction，禁止读取 teacher previous residual。

## 已验证的实现契约

- `anchorlock_rollout.py` 会覆盖 58D feature 中左脚 `21, 22:25` 和右脚 `46, 47:50` 的历史状态字段。
- raw residual 先投影到当前局部场景法向的切平面，再以 `max_action_m * tanh(.)` 有界化；hard gate 是 `[foot, 1]`，避免错误广播。
- 随机 GRU 在真实 60 帧测试序列的无梯度 smoke 中输出 `probability=[60,2]`、`action=[60,2,3]`，hard gate 路径正常。
- 训练 checkpoint 同时保存 model、optimizer、normalization、epoch、best validation loss 与 history；控制器中断后以 `--resume` 接续。

## 预注册候选与判定

| id | 单变量 | 门槛 |
|---|---|---|
| `action40mm` | 最大切向 action 40 mm | contact F1 >= 0.70 且 post sliding / pre sliding <= 1 |
| `action20mm` | 最大切向 action 20 mm | 同上 |
| `action20mm_stick5` | 20 mm，stick 权重 5（默认 2） | 同上 |

该表是 oracle-SDF + injected-drift 的 `mechanism result`，不是 Human3R/GUSH3R 真正前端结果。

## 当前状态

控制器已完成首轮三项训练。初始 hard-gate 评测发现训练/执行不一致：训练时 action 按连续 probability 门控，但评测用 0.7 hard gate。已修复 evaluator，且只用 validation 选择 soft gate，再重评 test。

| candidate | test contact F1 | test transition F1 | test sliding (m/s) | 结论 |
|---|---:|---:|---:|---|
| 40 mm, soft gate | 0.817 | 0.310 | 0.385 -> 0.218 | 通过，滑步下降 43.4% |
| 20 mm, hard gate | 0.947 | 0.816 | 0.385 -> 0.551 | 失败，执行滑步增加 |
| 20 mm + stick=5, soft gate | 0.822 | 0.511 | 0.385 -> 0.370 | 通过最小滑步门槛，但不优于 40 mm |
| 40 mm + contact BCE×1.5, soft gate | 0.849 | 0.416 | 0.385 -> 0.259 | contact/control Pareto 中间点 |
| 40 mm + contact BCE×2, soft gate | 0.863 | 0.550 | 0.385 -> 0.281 | 状态识别更强的 Pareto 点 |

以上全是 PROX oracle-SDF + injected-drift 的 mechanism result。

## E2 退化 proxy 后续结果

medium proxy（18° normal error、2 cm distance noise、1 cm bias、20% dropout、2-frame delay）下，冻结的 control-first controller 将 sliding 从未修正的 `0.385` 恶化为 `0.463 m/s`（+20.3%）。仅靠 confidence threshold 的安全回退需要 100% abstention，因此不构成有效修正。

以同类几何退化训练的 `medium_aug_action40_contact15` 已完成，在该 medium test 达到：sliding `0.385 -> 0.276 m/s`（-28.4%）、contact F1 `0.722`、transition F1 `0.142`。结论是几何增强能恢复部分控制，却不能可靠建模接触转变或错误表面风险。下一步不是继续调 gate，而是以当前 RGB token + mesh ROI + local point/Gaussian patch 训练 Stage-A local relation encoder，显式预测 normal/distance reliability 与 uncertainty，再接入现有 causal controller。
