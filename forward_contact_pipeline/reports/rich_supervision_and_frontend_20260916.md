# RICH全量监督整理与真实前端检查（2026-09-16）

本批官方train+公共资产的下载、解压、监督整理已完成。当前未达到真实训练输入门槛。

| 产物 | train | 内部val（ParkingLot2） |
|---|---:|---:|
| 有效SMPL-X人体帧 | 25071 | 12514 |
| v1图像-人体候选样本 | 167710 | 75084 |
| v2画面范围筛选后样本 | 165920 | 74264 |
| 双脚都在512裁剪外的视角 | 1790 | 820 |
| 固定脚底顶点标签positive比例 | 55.07% | 69.93% |

数据：`datasets/RICH/processed/supervision_v2/`。左右脚各64模板固定顶点；
全部1206片及242794候选的标签、真实帧号、投影/画面范围mask和hash独立回读通过。
筛选后全部37585个有效人体帧仍有合格视角，无新增整帧丢失。原来的84个无颜色帧、
10个无contact帧、14987个缺标定camera10图像保持原规则；未删除原始数据。
画面范围mask不证明遮挡可见性；无符号距离不冒充signed proximity。

## 冻结GUSH3R数值pilot

复用模型`bac8d88405ff62453b033c5a2b5709f42fcf50be`及既有本机补丁。
本轮没有更改GUSH3R内部源码，通过head hook和逐帧callback保存真实decoder image tokens
`[1,736,768]`、SMPL-X、point map、当前累积Gaussian诊断和固定LBS索引。
`skip_inference_render_outputs=True`且`keep_outputs=False`，未调用渲染。

每片段独立因果状态、默认FOV先验K、无GT输入；头部平移约定为
`SMPL-X vertices + predicted head translation - posed_head`，flat-hand-mean与原层一致。
32帧总用时78.96秒（含模型加载/校验/落盘，不作全量吞吐保证）。
对齐只在独立审计中使用，未写回前端数据。

| 片段 / cam00 | fit帧号 | held-out帧号 | 全身median | 脚底median | 脚底p95 |
|---|---|---|---:|---:|---:|
| ParkingLot1_002_stretching1 | 5–16 | 17–20 | 3.39cm | 5.69cm | 9.07cm |
| ParkingLot2_016_stretching1 | 100–111 | 112–115 | 4.62cm | 2.66cm | 5.00cm |

每片仅用前12帧的固定512顶点拟合一个Sim(3)，后4帧评估全部顶点及固定脚底ROI。
两片全身和脚底中位误差都高于既定2cm门槛；短片段结果不能代表完整数据集精度。
没有生成全量local relation训练缓存，`training_ready=false`。
先诊断前端几何精度或明确变更实验协议，不能通过手工改pass标记进入正式训练。
scene仅保留每帧最多8192个高opacity Gaussian用于诊断，不是完整scene surface。

## 验证与复现

50项CPU软件回归通过；新增5项覆盖变换、掩码、缺标注、帧/源错误、输出损坏和验证后续跑。
完整脚本与跨机器命令见`../RICH_DATASET.md`。数据/模型不进入Git。

本地证据（均在`reports/rich_processing_20260916/`）：
`data_status.json`、`supervision_completion_audit.json`、`frontend_geometry_audit.json`、`final_tests.txt`。

- supervision报告SHA256：`df081c076da5b4419ce0e8c426be2a1cde45e8482a752ba91b669d3f98037384`
- frontend manifest SHA256：`0b96da9d148ecda70beafc00f006d71896de6d85f271cecf971b0d697a5ad427`
- train筛选清单SHA256：`3b075dd489c81e08203263468ab19cf05edd621d7425a906cfca467eefbbc1b6`
- val筛选清单SHA256：`976ae6dd70880f588309748f8de86323582e2988d089b0c07335cafc256cb327`
