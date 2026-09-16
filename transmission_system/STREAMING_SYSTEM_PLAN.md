# 面向人体-场景联合表征的流式传输方案

> 本文档是 `transmission_system/` 的活文档。系统架构、研究假设、实验结果和待办事项均在此持续更新。

## 1. 研究目标

针对动态人体 Gaussian 与静态场景 Gaussian 的联合表征，构建一个接触感知、人体优先、可渐进加载的流式传输系统。

目标不是传输渲染后的 RGB 视频，而是服务器传输 Gaussian 表征和动态参数，客户端恢复人体-场景联合表征并本地渲染任意视角。

核心研究问题：

1. 如何压缩逐帧人体 correction，降低动态传输带宽？
2. 如何根据人体运动和接触状态动态调度人体流与场景流？
3. 如何在有限带宽下优先保证人体和接触区域的视觉质量？
4. 如何降低首次启动延迟，并使场景渐进补全？

## 2. 数据组成与统一口径

当前论文/实验建议统一采用以下静态初始化口径：

| 组件 | 内容 | 传输频率 | 当前规模 |
|---|---|---:|---:|
| `S-base/chunks` | 场景 Gaussian | 首次/渐进缓存 | 约 486 MB |
| `H-canonical` | canonical 人体 Gaussian | 一次 | 约 5.9 MB |
| `H-lbs` | LBS 权重 | 一次 | 约 10.1 MB |
| `H-smpl` | 全部帧 SMPL 参数 | 一次 | 约 60 KB |
| `H-corr_t` | 第 `t` 帧人体 correction | 逐帧 | 原始约 6 MB/帧 |

静态初始化总量约为 `486 + 5.9 + 10.1 + 0.06 ≈ 502 MB`。

注意：文档中“canonical 人体约 27 MB”和“canonical 5.9 MB + LBS 10.1 MB”是不同统计口径，后续统一使用上表口径。

## 3. 系统基本逻辑

### 3.1 初始化阶段

发送 `H-canonical`、`H-lbs`、`H-smpl`、低质量全场景代理 `S-base` 和首帧完整 correction `H-corr_I`。客户端收到人体资产和首帧 correction 后即可显示可动人体，不必等待完整场景。

### 3.2 运动阶段

持续传输 `H-corr_I/P`，同时根据人体位置和接触状态调度 `S-contact`、`S-near`。人体 correction 保证运动连续性，场景块保证脚、手、身体与环境交互质量。

### 3.3 持续播放阶段

优先级为：当前接触区域 > 即将到达区域 > 人体附近区域 > 远景区域。服务器根据当前人体位置、未来运动轨迹和接触 anchor 预测下一批需要的场景块并预取。

## 4. 数据分层

### 4.1 人体流

```text
H-base       canonical human Gaussian + LBS
H-pose       SMPL pose / shape / camera metadata
H-I-frame    完整 delta_mu correction
H-P-frame    相邻帧 correction residual
```

### 4.2 场景流

```text
S-base       低质量、低 SH 或降采样的全场景代理
S-contact    与人体接触相关的高质量场景块
S-near       人体 bounding box 附近但无直接接触的场景块
S-far        远景场景块
S-enhance    SH、opacity、scale 等渐进增强层
```

场景块初始使用 XYZ 空间分块；正式实验应增加可见性、人体轨迹和接触关系标注。

## 5. Correction 压缩方案

### 5.1 当前离线方案

```text
delta_mu[T,N,3]
  → Savitzky-Golay(window=11, polyorder=2)
  → I/P 帧划分
  → P 帧 temporal difference
  → int8 量化
  → 熵编码
```

当前实验结果：6 个 NeuMan 场景均已测试，ratio 降低约 47%–73%，PSNR 变化约 ±0.01 dB；frame embedding 平滑方案失败，PSNR 下降约 0.48 dB；当前推荐默认 `window=11`。

### 5.2 在线方案

Savgol 需要未来帧，适合离线预编码或有缓冲播放，不能直接作为零延迟在线算法。实时版本候选为：

```text
pose-conditioned predictor
  → 预测 delta_mu_t
  → 只编码 residual
  → int8 + entropy coding
```

预测器输入可以包括 `pose_t - pose_{t-1}`、previous `delta_mu`、anchor trajectory 和 contact state。

## 6. 研究创新主线

不把 TCP、分块、I/P 帧作为主要创新，而将论文主线定义为：

> Contact-aware rate-distortion optimized streaming for dynamic human-scene Gaussian representations

### 6.1 接触感知率失真优化

```text
L = λg L_global + λh L_human + λc L_contact + λt L_temporal + λr R
```

其中 `L_global` 是全图误差，`L_human` 是人体区域误差，`L_contact` 是脚-地面、手-物体等接触区域误差，`L_temporal` 是帧间闪烁，`R` 是传输 bit 数。

### 6.2 人体运动驱动的场景预取

区别于普通 camera-driven streaming，系统根据 `SMPL pose + anchor trajectory + contact state` 预测人体即将到达和接触的场景区域，提前传输相关场景块。

### 6.3 人体流与场景流联合调度

```text
Priority(packet) =
  VisualGain × ContactWeight × Visibility × MotionNeed
  / TransmissionCost
```

固定的 70% 人体 / 30% 场景只作为 baseline，不作为最终方法。

## 7. 推荐系统模块

```text
transmission_system/
├── assets/
│   ├── canonical/
│   ├── smpl/
│   └── scene_chunks/
├── preprocessing/
│   ├── smooth_correction.py
│   ├── build_ip_frames.py
│   ├── partition_scene.py
│   └── label_contact_chunks.py
├── codec/
│   ├── correction_codec.py
│   └── scene_chunk_codec.py
├── scheduler/
│   ├── priority_queue.py
│   ├── contact_scheduler.py
│   └── bandwidth_allocator.py
├── transport/
│   ├── protocol.py
│   ├── sender.py
│   └── receiver.py
├── renderer/
│   └── reconstruction.py
└── evaluate/
    ├── bandwidth.py
    ├── quality.py
    └── contact_quality.py
```

当前根目录的 `codec.py`、`protocol.py`、`file_transport.py`、`sender.py`、`receiver.py`、`evaluate.py` 是底层 smoke-test 骨架，只支持 `delta_mu` 文件/TCP 验证，尚未接入真实 HUGS renderer、scene chunks、contact scheduler 和 SMPL packet。

## 8. 实验矩阵

### 8.1 压缩消融

| 方法 | 目的 |
|---|---|
| raw fp32 | 上限 baseline |
| fp16 | 基础量化 baseline |
| int8 | 量化效果 |
| int8 + temporal diff | 帧间冗余效果 |
| Savgol + diff + int8 + entropy | 当前方法 |
| pose-conditioned prediction + residual | 在线候选方法 |

### 8.2 调度消融

| 方法 | 目的 |
|---|---|
| uniform streaming | 通用传输 baseline |
| scene-distance priority | 空间距离 baseline |
| human-first fixed ratio | 人体优先 baseline |
| contact-aware scheduling | 核心方法 |
| contact-aware + trajectory prefetch | 完整方法 |

### 8.3 评价指标

- 总 bitrate、平均每帧 payload、首帧启动延迟；
- 全图 PSNR / SSIM / LPIPS；
- 人体区域 PSNR；
- 接触区域 PSNR；
- 接触边界几何误差；
- temporal flicker；
- 带宽波动下的恢复时间；
- 丢包后的恢复帧数；
- 客户端解码和渲染耗时。

## 9. 实施路线

### Phase 0：传输闭环

- [x] demo `delta_mu` 文件打包/解包
- [x] localhost TCP sender/receiver
- [x] 数值重建误差评估
- [ ] 接入真实 `delta_mu`

### Phase 1：真实人体流

- [ ] 导出真实 `delta_mu[T,N,3]`
- [ ] 增加 `H-smpl` packet
- [ ] 实现周期性 I-frame
- [ ] 接入 HUGS receiver-side reconstruction
- [ ] 逐帧渲染并对比 reference

### Phase 1.5：基础传输性能 baseline

- [x] 静态资产一次传输 + 动态 correction 逐帧传输模型
- [x] 输出启动延迟、吞吐、按时到达帧比例和 decode FPS
- [ ] 用真实 502 MB 静态资产和真实 correction 序列运行
- [ ] 接入 HUGS GPU renderer，测真实观看 FPS
- [ ] 测试 10/50/100/1000 Mbps 和 15/30/60 FPS

#### 首轮体积 baseline（2026-07-16）

测试配置：静态资产 502,000,000 B，人体高斯点数 500K，raw `delta_mu` 为 6,000,000 B/帧，60 帧，目标 30 FPS，RTT=20 ms。该轮使用体积模型，没有接入 HUGS GPU renderer。

| 带宽 | 静态传输 | 首帧可见 | 动态流到达速率 | 30 FPS 按时帧比例 |
|---:|---:|---:|---:|---:|
| 100 Mbps | 40.18 s | 40.68 s | 约 2.0 FPS | 1/60 |
| 1000 Mbps | 4.04 s | 4.10 s | 约 14.7 FPS | 1/60 |

原始动态流的理论带宽需求为：

```text
6 MB/frame × 30 frame/s × 8 ≈ 1.44 Gbps
```

因此简单的“静态资产一次传输 + raw correction 逐帧传输”方案无法满足 30 FPS 实时播放，即使在 1 Gbps 网络下也不够。该结果支持继续研究 correction 压缩、S-base 渐进初始化、接触区域优先调度和带宽自适应分配。

注意：benchmark 中的 `decode_fps` 只代表 packet 解码吞吐，不代表最终观看 FPS。最终观看 FPS 必须把 HUGS receiver-side reconstruction 和 GPU renderer 接到 receiver 后重新测量。

#### 真实 bike correction 传输结果（2026-07-16）

输入：`output/delta_mu_analysis/bike/delta_mu_f16.npy`，形状 `(104, 515804, 3)`；静态资产按 502 MB 模拟；目标 30 FPS，RTT=20 ms。Savgol 实验使用 `window=11, polyorder=2`，然后使用当前 TCP 骨架中的 int8 per-frame quantization + temporal difference + zlib。这个结果是实际 packet payload，不是理论熵下界。

| 方案 | 网络 | 动态 payload/帧 | 首帧可见 | 接收数据速率 | 30 FPS 按时帧 |
|---|---:|---:|---:|---:|---:|
| raw correction | 1 Gbps | 5.90 MiB | 4.11 s | 14.38 FPS | 1/104 |
| Savgol + int8 diff + zlib | 1 Gbps | 1.12 MiB | 4.07 s | 34.08 FPS | 104/104 |
| Savgol + int8 diff + zlib | 100 Mbps | 1.12 MiB | 40.31 s | 8.81 FPS | 1/104 |

对应结论：

- Savgol 版本将 bike 动态 payload ratio 降到约 `18.9%`，约降低 `81.1%`；
- 1 Gbps 下动态流从 14.38 FPS 提升到 34.08 FPS，超过 30 FPS 目标；
- 100 Mbps 下仍无法实时播放，说明压缩后动态流的实际需求约为 `1.12 MiB × 30 × 8 ≈ 268 Mbps`；
- 首帧延迟几乎没有改善，因为首帧主要由 502 MB 静态场景传输决定；
- `decode_fps` 约 99 FPS，仅表示 correction 解码吞吐，不包含 HUGS GPU 渲染。

结果文件：

- `results/bike_raw_1000mbps.json`
- `results/bike_savgol11_1000mbps.json`
- `results/bike_savgol11_100mbps.json`

#### 6 场景 Savgol 传输汇总（1 Gbps）

| 场景 | 帧数 | payload/帧 | ratio | 30 FPS 所需动态带宽 | 接收速率 | 按时帧 |
|---|---:|---:|---:|---:|---:|---:|
| bike | 104 | 1.116 MiB | 18.9% | 268 Mbps | 34.08 FPS | 104/104 |
| citron | 37 | 1.282 MiB | 18.9% | 308 Mbps | 32.54 FPS | 37/37 |
| jogging | 102 | 1.067 MiB | 20.4% | 256 Mbps | 34.56 FPS | 102/102 |
| lab | 103 | 1.164 MiB | 20.3% | 279 Mbps | 33.61 FPS | 103/103 |
| parkinglot | 42 | 1.106 MiB | 19.7% | 265 Mbps | 34.18 FPS | 42/42 |
| seattle | 41 | 1.136 MiB | 18.4% | 273 Mbps | 33.89 FPS | 41/41 |

6 个场景在当前传输模型下均能在 1 Gbps 网络达到 30 FPS 的 packet 到达速率；但首帧仍约 4.07 秒，且这是接收/解码速率，不包含 HUGS GPU 渲染。Savgol + 当前 zlib packet 的实际动态 payload 约为原始 fp32 correction 的 18.4%–20.4%，比之前只看 ratio 的分析更接近真实传输成本。

#### Receiver-side HUGS 渲染结果（2026-07-16）

已将 Savgol packet 解码结果接入真实 HUGS `render_human_scene`，使用 bike 的全部 104 帧连续序列生成视频。客户端模型加载后，视频按目标 30 FPS 输出；当 correction packet 未在当前 presentation deadline 到达时，receiver 使用最近一次已解码 correction。

| 网络 | 首帧网络延迟 | 网络迟到帧 | correction 复用帧 | GPU 连续渲染能力 | 视频 |
|---:|---:|---:|---:|---:|---|
| 1 Gbps | 4.07 s | 0/104 | 12/104 | 104.96 FPS | `results/received_render_all_1gbps/bike_savgol11_1000mbps/received_stream.mp4` |
| 100 Mbps | 40.31 s | 103/104 | 73/104 | 103.55 FPS | `results/received_render_all_100mbps/bike_savgol11_100mbps/received_stream.mp4` |

两次实验的实际 GPU renderer 平均耗时分别为约 9.53 ms/帧和 9.66 ms/帧，说明当前 bike 配置下客户端渲染能力明显高于 30 FPS。1 Gbps 下网络供给也满足 30 FPS；100 Mbps 下主要问题是 correction packet 到达太慢，而不是渲染速度不足。

运行命令：

```bash
PYTHONUNBUFFERED=1 \
/workspace/nas_auto_backup/yuzilang/miniconda3/envs/hugs/bin/python \
  transmission_system/render_received_stream.py \
  --scene bike --split all --bandwidth-mbps 1000 --fps 30 \
  --out-dir transmission_system/results/received_render_all_1gbps
```

当前视频评测仍只模拟 correction 流，scene GS 作为已缓存静态资产；尚未加入 S-base/S-contact/S-near/S-far 的渐进场景传输。

#### 人体优先 + 中心距离场景渐进 baseline（2026-07-16）

这是不带接触感知的最小可行方案：先传约 16.06 MB 的人体初始化资产，再将场景 GS 按距 SMPL 运动轨迹中位中心的距离分成 10 块，使用 20% 链路带宽渐进传输；其余 80% 用于 correction。

| 带宽 | 人体首帧 | 第一场景块 | 完整场景 | 104 帧窗口内场景块 | 结果视频 |
|---:|---:|---:|---:|---:|---|
| 1 Gbps | 0.15 s | 2.11 s | 19.79 s | 5/10 | `results/progressive_center_1gbps/bike_1000mbps_10chunks/progressive_center.mp4` |
| 100 Mbps | 1.30 s | 20.76 s | 195.90 s | 0/10 | `results/progressive_center_100mbps/bike_100mbps_10chunks/progressive_center.mp4` |

该实验验证了人体优先架构的核心价值：在 100 Mbps 下，客户端约 1.3 秒即可显示人体，不需要等待 502 MB 场景完整下载；代价是场景随后渐进得很慢。下一步应先增加小型 `S-base`，再比较固定中心加载与接触/轨迹感知调度。

## 13. 候选前馈式研究方向（暂不实施）

> 状态：仅记录方案，当前不启动实验、不修改现有传输实现。

### 13.1 Pose-conditioned correction 前馈预测

服务器根据历史 SMPL pose、pose difference 和上一帧 `delta_mu` 预测未来 correction：

```text
pose history + delta_mu_{t-1}
        → predictor
        → delta_mu_hat_t
        → 只传 residual_t = delta_mu_t - delta_mu_hat_t
```

候选模型从简单到复杂：上一帧保持、线性外推、pose-conditioned MLP、TCN/GRU。目标是将“训练后滤波压缩”升级为“面向传输学习的 correction 表征”。

### 13.2 人体运动驱动的场景前馈预取

根据 `SMPL pose + anchor trajectory + contact state` 预测未来人体会经过或接触的场景区域，在真正需要之前预取对应场景块。区别于普通 camera-driven streaming，该方向是 human/contact-driven streaming。

### 13.3 前馈式动态带宽分配

根据未来若干帧的 correction、接触场景块和可见性，计算每个 packet 的预期收益：

```text
utility(packet) =
P(visible) × P(contact) × visual_gain / packet_size
```

在带宽预算下动态选择人体流、接触场景流和远景流，不再使用固定 70%/30% 分配。

### 13.4 不确定性感知关键帧调度

前馈预测存在突然动作和接触状态变化导致的失效风险。预测器同时输出 uncertainty，当预测不可靠或 contact state 变化时，提前发送 I-frame、提高 correction 优先级或预取更多接触场景。

候选关键帧条件：

```text
prediction uncertainty
+ contact state change
+ residual error
```

### 13.5 推荐的统一研究主线

暂定名称：

> Uncertainty-Aware Contact-Conditioned Feed-Forward Streaming for Dynamic Human-Scene Gaussian Representations

潜在组合：

1. pose-conditioned correction predictor；
2. contact-aware scene prefetch；
3. uncertainty-aware keyframe scheduling。

### 13.6 未来验证矩阵

```text
A. raw correction
B. Savgol + temporal difference
C. constant-velocity feedforward prediction
D. pose-conditioned feedforward prediction
```

预测 horizon 可测试 `K=1,3,5,10`，指标包括 residual payload、动态带宽、预测误差、人体/接触区域 PSNR、temporal flicker、突发动作失败率和 I-frame 次数。

#### 新增优化问题：视频不能在人体序列结束时停止

当前 progressive video 只覆盖人体的 104 帧、约 10.4 秒内容时长：

- 100 Mbps 下第一块场景约 20.76 秒才到达，因此视频结束时背景尚未出现；
- 1 Gbps 下完整场景约 19.79 秒才到达，10.4 秒视频结束时只加载了约一半场景块；
- 按人体中心距离排序的场景块不保证当前相机视角可见，已加载块可能对画面贡献很小。

这不是可接受的最终用户体验，后续必须实现：

1. 人体序列播放结束后继续保持/循环人体画面，直到场景渐进加载完成；
2. 输出完整的“人体先出现 → 背景逐步出现 → 场景补全”体验视频；
3. 将纯距离排序升级为至少考虑相机可见性的场景块排序；
4. 后续再加入接触区域和人体轨迹预测，比较其相对于中心加载 baseline 的收益。

该问题正式列为下一阶段优先级：**渐进场景加载期间的持续播放与可见性引导**。

#### 阶段结果：持续播放直到场景补全（2026-07-16）

已修改 progressive baseline：人体 104 帧播放结束后保持最后姿态，继续写出视频帧直到所有场景块到达。

| 带宽 | 视频总时长 | 人体首帧 | 第一场景块 | 完整场景 | 视频 |
|---:|---:|---:|---:|---:|---|
| 1 Gbps | 19.79 s | 0.15 s | 2.11 s | 19.79 s | `results/progressive_center_tail_1gbps/bike_1000mbps_10chunks/progressive_center.mp4` |
| 100 Mbps | 195.90 s | 1.30 s | 20.76 s | 195.90 s | `results/progressive_center_tail_100mbps/bike_100mbps_10chunks/progressive_center.mp4` |

该阶段解决了“视频结束时背景还没加载”的展示问题。当前仍是 center-distance baseline，下一阶段在相同持续播放框架下实现 visibility-aware 和 contact-aware 调度。

重要的帧率口径修正：NeuMan/HUGS 当前全帧可视化默认使用 `cfg.train.render_fps = 10`，bike 的 104 帧原始内容应约持续 10.4 秒。此前用 30 FPS 写出的视频内容段只有 3.47 秒，因此视觉上约为 3 倍速；30 FPS 只能作为额外的高帧率假设，不能作为 NeuMan 原始观看速度。后续视频体验评测默认使用 10 FPS，并将 30 FPS 作为独立压力测试。

因果时序修正：接收端即使提前收到未来 packet，也不能将未来 correction 用于当前帧；benchmark 已限制 `available_packet <= current_frame_index`。

#### 现象记录：黑屏结束后播放流畅

该现象由多个因素共同造成：

1. 黑屏阶段完成约 502 MB 静态 scene GS 初始化；之后不再重复传输场景，只传动态 correction。
2. 1 Gbps 下 Savgol correction 流满足 NeuMan 原始 10 FPS，packet 可按时到达。
3. 100 Mbps 下 correction packet 迟到时，receiver 使用最近已收到的 correction，不会停止渲染。
4. 当前帧的 SMPL pose 仍连续更新，因此人体主体运动不会停顿。
5. Savgol 时域平滑降低了 correction 的帧间抖动，旧 correction 与当前 correction 的差异通常不容易形成明显闪烁。

因此评测时需要同时记录：

```text
主观播放流畅度 + packet 迟到率 + correction 复用率
+ temporal error + 接触区域质量 + GPU render FPS
```

主观上“流畅”不能单独证明网络带宽充足。

正确 10 FPS 体验视频：

| 网络 | 视频总时长 | 首帧等待 | 网络迟到帧 | correction 复用 |
|---:|---:|---:|---:|---:|
| 1 Gbps | 14.47 s | 4.07 s | 0/104 | 0/104 |
| 100 Mbps | 50.71 s | 40.31 s | 103/104 | 13/104 |

输出目录：

- `results/received_render_all_1gbps_10fps_causal/bike_savgol11_1000mbps/received_stream.mp4`
- `results/received_render_all_100mbps_10fps_causal/bike_savgol11_100mbps/received_stream.mp4`

用户体验版视频会显式保留启动等待阶段：

- 1 Gbps：约 4.07 秒黑屏/加载 + 3.47 秒内容，总时长约 7.53 秒；
- 100 Mbps：约 40.31 秒黑屏/加载 + 3.47 秒内容，总时长约 43.77 秒。

体验版文件名为同目录下的 `received_stream_experience.mp4`。之前的 `received_stream.mp4` 是仅内容帧版本，适合分析渲染画质，不代表用户从点击播放开始的完整等待体验。

### Phase 2：场景渐进流

- [ ] scene GS 空间分块
- [ ] S-base 低质量初始化
- [ ] S-contact / S-near / S-far 标注
- [ ] scene chunk packet
- [ ] 客户端缓存和渐进更新

### Phase 3：研究算法

- [ ] 接触质量评价指标
- [ ] rate-distortion priority
- [ ] 轨迹驱动预取
- [ ] 带宽自适应调度
- [ ] 在线 pose-conditioned predictor

## 10. 当前风险与决策

| 问题 | 当前判断 | 后续动作 |
|---|---|---|
| 首次 502 MB 延迟高 | 真实风险 | 增加 S-base 和渐进加载 |
| Savgol 非因果 | 仅适合离线预编码 | 增加 causal predictor |
| 固定 70/30 缺乏依据 | 只能做 baseline | 使用 rate-distortion 调度 |
| XYZ 分块不能准确表达接触 | 需要增强 | 加入 anchor、可见性和轨迹标签 |
| 只做压缩创新不足 | 学术风险 | 聚焦接触感知率失真优化 |
| HUGS renderer 尚未接入 | 工程未闭环 | Phase 1 优先完成 |

## 11. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-07-16 | 创建方案文档；确定“接触感知率失真优化 + 轨迹预取 + 联合调度”为研究主线。 |
| 2026-07-16 | 创建文件模式和 localhost TCP 传输骨架，完成 demo 闭环。 |

## 12. 实验记录模板

后续每次实验按以下格式追加：

```text
### YYYY-MM-DD 实验名称

- 数据集/场景：
- 输入模型：
- 传输配置：
- 带宽/帧率：
- 结果：
- 接触区域结果：
- 结论：
- 下一步：
```
