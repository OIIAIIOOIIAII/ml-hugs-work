# HUGS 传输系统实验报告

日期：2026-07-16  
项目：HUGS 动态人体-场景 Gaussian 表征传输  
场景：NeuMan，主要使用 `bike`，压缩 benchmark 覆盖 6 个场景

本文记录 2026-07-16 完成的全部传输相关实验，包括实验设置、数据口径、编码方式、网络模拟方式、接收端真实 HUGS 渲染结果，以及人体优先渐进场景加载实验。

## 1. 实验目标

昨天的实验分为四个阶段：

1. 建立“静态资产一次传输 + 动态 correction 逐帧传输”的基础网络性能 baseline；
2. 将已经验证有效的 Savgol correction 压缩方案接入传输 benchmark；
3. 将压缩 packet 接入真实 HUGS receiver-side renderer，测量真实 GPU 渲染和 packet 迟到影响；
4. 验证人体优先、场景分块渐进加载的最小可行架构。

研究目标不是传输 RGB 视频，而是传输 Gaussian 表征和动态参数，使客户端恢复人体-场景联合表征并本地渲染。

## 2. 统一数据口径

当前传输模型使用如下数据划分：

| 数据 | 内容 | 传输频率 | 规模 |
|---|---|---:|---:|
| `S-base` | 场景 Gaussian | 首次或渐进缓存 | 约 486 MB |
| `H-canonical` | canonical 人体 Gaussian | 一次 | 约 5.9 MB |
| `H-lbs` | LBS 权重 | 一次 | 约 10.1 MB |
| `H-smpl` | SMPL 参数 | 一次 | 约 60 KB |
| `H-corr_t` | 第 `t` 帧人体 correction | 逐帧 | raw 约 6 MB/帧 |

完整静态初始化口径为：

```text
5.9 MB + 10.1 MB + 0.06 MB + 486 MB ≈ 502 MB
```

动态 correction 的原始理论带宽需求约为：

```text
6 MB/frame × 30 frame/s × 8 ≈ 1.44 Gbps
```

因此 raw correction 在 1 Gbps 网络下也无法稳定满足 30 FPS。

### 2.1 真实输入数据

压缩和 receiver 渲染使用已经训练好的 `v4_correct_inline_attn_18k` 模型与真实 `delta_mu`：

```text
output/delta_mu_analysis/<scene>/delta_mu_f16.npy
```

其中 bike 的输入形状为：

```text
(104, 515804, 3)
```

即 104 帧、515,804 个 human Gaussian、每点 3 个 correction 值。每帧 raw fp32 correction 为约 5.90 MiB。

### 2.2 网络模型

昨天的主要 benchmark 使用解析式网络模拟，而不是实际跨机器网络：

```python
transfer_time = num_bytes * 8 / (bandwidth_mbps * 1e6) + rtt_ms / 1000
```

使用的网络参数：

- 带宽：100 Mbps 或 1000 Mbps；
- RTT：20 ms；
- 静态资产和动态 packet 按串行逻辑时间计算到达时间；
- 没有模拟丢包、重传、jitter、带宽波动或 TCP 拥塞控制。

因此结果表示“固定带宽、固定 RTT 条件下的可复现传输时序”，不是复杂真实网络鲁棒性测试。

## 3. 实验一：体积模型 raw correction baseline

### 3.1 设置

脚本：

```text
transmission_system/baseline_benchmark.py
```

设置：

| 参数 | 数值 |
|---|---:|
| 静态资产 | 502,000,000 B |
| human Gaussian 点数 | 500,000（体积模型） |
| raw correction | 6,000,000 B/帧 |
| 帧数 | 60 |
| 目标帧率 | 30 FPS |
| RTT | 20 ms |
| 编码 | raw |

### 3.2 结果

| 带宽 | 静态传输时间 | 首帧可见时间 | 动态接收速率 | 30 FPS 按时到达 |
|---:|---:|---:|---:|---:|
| 100 Mbps | 40.18 s | 40.68 s | 2.00 FPS | 1/60（1.67%） |
| 1 Gbps | 4.036 s | 4.104 s | 14.71 FPS | 1/60（1.67%） |

完整 JSON：

- `results/baseline_100mbps_raw.json`
- `results/baseline_1000mbps_raw.json`

### 3.3 结论

1. 502 MB 静态场景在 100 Mbps 下需要约 40 秒，在 1 Gbps 下需要约 4 秒；
2. raw correction 的动态带宽需求超过 1 Gbps；
3. 即使静态场景已经传完，raw correction 也无法以 30 FPS 持续供给；
4. 仅优化网络连接不能解决问题，必须压缩 correction，或者改变表示方式。

## 4. 实验二：真实 correction 的 Savgol + int8 差分 + zlib

### 4.1 编码流程

使用的离线压缩流程为：

```text
真实 delta_mu[T,N,3]
    ↓
Savgol filter（window=11, polyorder=2）
    ↓
帧间 temporal difference
    ↓
int8 量化
    ↓
zlib 压缩
    ↓
逐帧 packet
```

代码中的 codec 名称为 `causal_int8_zlib`。Savgol 本身需要时间序列上下文，适合离线预编码或有缓冲播放；TCP sender 的基础接口仍然保留 causal temporal difference。

### 4.2 bike 对比 raw

输入：bike，104 帧，515,804 点，目标 30 FPS，静态资产 502 MB，RTT 20 ms。

| 指标 | raw / 1 Gbps | Savgol / 1 Gbps | Savgol / 100 Mbps |
|---|---:|---:|---:|
| raw dynamic 总量 | 643,723,392 B | 643,723,392 B | 643,723,392 B |
| dynamic payload 总量 | 643,723,392 B | 121,656,802 B | 121,656,802 B |
| payload/帧 | 5.903 MiB | 1.116 MiB | 1.116 MiB |
| payload ratio | 100% | 18.90% | 18.90% |
| 首帧可见 | 4.106 s | 4.067 s | 40.306 s |
| 接收速率 | 14.38 FPS | 34.08 FPS | 8.81 FPS |
| 30 FPS 按时帧 | 1/104 | 104/104 | 1/104 |
| decode FPS | — | 98.68 | 100.89 |

结果文件：

- `results/bike_raw_1000mbps.json`
- `results/bike_savgol11_1000mbps.json`
- `results/bike_savgol11_100mbps.json`

### 4.3 6 个场景的 1 Gbps 结果

| 场景 | 帧数 | raw payload/帧 | 压缩 payload/帧 | ratio | 接收速率 | 按时帧 |
|---|---:|---:|---:|---:|---:|---:|
| bike | 104 | 5.903 MiB | 1.116 MiB | 18.90% | 34.08 FPS | 104/104 |
| jogging | 102 | 5.239 MiB | 1.067 MiB | 20.37% | 34.56 FPS | 102/102 |
| seattle | 41 | 6.180 MiB | 1.136 MiB | 18.38% | 33.89 FPS | 41/41 |
| lab | 103 | 5.738 MiB | 1.164 MiB | 20.29% | 33.61 FPS | 103/103 |
| parkinglot | 42 | 5.623 MiB | 1.106 MiB | 19.66% | 34.18 FPS | 42/42 |
| citron | 37 | 6.795 MiB | 1.282 MiB | 18.86% | 32.54 FPS | 37/37 |

对应 JSON：

```text
results/{bike,citron,jogging,lab,parkinglot,seattle}_savgol11_1000mbps.json
```

### 4.4 结论

- 实际 packet payload 降到 raw 的 18.4%～20.4%；
- 1 Gbps 下 6 个场景都达到 30 FPS packet 到达速率；
- 压缩后的动态流仍需要大约 256～308 Mbps，100 Mbps 仍不足；
- 首帧延迟几乎没有变化，因为首帧主要由 502 MB 静态资产决定；
- 之前独立 PSNR 实验显示 Savgol `window=11` 的 ΔPSNR 约在 ±0.01 dB 内，可以视为无明显质量损失。

## 5. 实验三：真实 HUGS receiver-side 渲染

### 5.1 目的

前一个实验只测 packet 到达和解码吞吐，不能说明客户端真实渲染是否跟得上。因此将压缩 packet 接入：

```text
transmission_system/render_received_stream.py
```

该脚本使用真实 HUGS checkpoint、真实 NeumanDataset 和真实 `render_human_scene`。

### 5.2 receiver 行为

每个 presentation deadline 到来时：

1. 查找已经到达的 correction packet；
2. 只使用不晚于当前显示帧的 packet，保持因果性；
3. 如果当前 packet 未到，复用最近一次已解码 correction；
4. 仍然使用当前帧的 SMPL pose 渲染人体；
5. 输出真实 HUGS 渲染结果。

因此网络迟到不会让渲染进程停止，但会造成 correction stale。

### 5.3 30 FPS bike 结果

| 指标 | 1 Gbps | 100 Mbps |
|---|---:|---:|
| 静态网络启动延迟 | 4.07 s | 40.31 s |
| 帧数 | 104 | 104 |
| 目标播放速率 | 30 FPS | 30 FPS |
| 迟到帧 | 0/104 | 103/104 |
| correction 复用帧 | 12/104 | 73/104 |
| 平均 GPU render | 约 9.53 ms | 约 9.66 ms |
| 连续渲染能力 | 约 104 FPS | 约 103 FPS |

视频：

```text
results/received_render_all_1gbps/bike_savgol11_1000mbps/received_stream.mp4
results/received_render_all_100mbps/bike_savgol11_100mbps/received_stream.mp4
```

### 5.4 10 FPS 因果版本

NeuMan 原始观看速度按 HUGS 配置使用 10 FPS，30 FPS 主要是压力测试。因此又在 10 FPS 下生成因果版本。

| 指标 | 1 Gbps | 100 Mbps |
|---|---:|---:|
| 静态网络启动延迟 | 4.067 s | 40.306 s |
| 视频总时长（含启动黑屏） | 14.467 s | 50.706 s |
| 迟到帧 | 0/104 | 103/104 |
| correction 复用帧 | 0/104 | 13/104 |
| 平均 render | 9.592 ms | 9.592 ms |
| 连续渲染能力 | 104.26 FPS | 104.25 FPS |
| wall-clock 视频生成 FPS | 11.07 FPS | 9.20 FPS |

视频和指标：

```text
results/received_render_all_1gbps_10fps_causal/
results/received_render_all_100mbps_10fps_causal/
```

### 5.5 结论

1. 当前 bike 配置的 GPU renderer 不是瓶颈，连续渲染能力超过 100 FPS；
2. 1 Gbps 下 Savgol correction 可以满足 30 FPS packet 供给；
3. 100 Mbps 下主要问题是动态 packet 到达速度不足，而不是 GPU 渲染速度不足；
4. 100 Mbps 下视频仍能持续播放，是因为 receiver 复用了旧 correction，不能解释为每帧 correction 都实时到达；
5. 这些实验当时仍将 scene GS 视为已经缓存的静态资产，没有模拟场景块渐进传输。

## 6. 实验四：人体优先 + 中心距离场景渐进加载

### 6.1 目的

完整等待 502 MB 场景会造成很长启动延迟。因此测试如下最小架构：

```text
先传 H-canonical + H-lbs + H-smpl + 首帧 correction
        ↓
客户端立即显示人体
        ↓
场景 GS 分块渐进加载
```

该实验是 center-distance baseline，不是最终接触感知方法。

### 6.2 传输设置

| 参数 | 设置 |
|---|---:|
| 场景 | bike |
| 场景总大小 | 486,000,000 B |
| 人体初始化大小 | 16,060,000 B |
| 场景块数 | 10 |
| correction 带宽 | 80% |
| scene chunk 带宽 | 20% |
| 播放速度 | 10 FPS |
| RTT | 20 ms |
| chunk 排序 | 到 SMPL 运动轨迹中位中心的距离 |

代码：

```text
transmission_system/render_progressive_center_stream.py
```

场景点按照到人体运动轨迹中位中心的距离排序，再分为 10 个逻辑块。该方法没有使用 contact label、相机可见性或未来接触预测。

### 6.3 初始版本结果

初始视频在人体 104 帧结束后停止，随后发现这不适合展示场景渐进加载，因此后续进行了 tail 修复。初始实验的主要网络时序为：

| 带宽 | 人体首帧 | 第一场景块 | 完整场景 | 104 帧窗口内到达块数 |
|---:|---:|---:|---:|---:|
| 1 Gbps | 0.148 s | 2.112 s | 19.788 s | 5/10 |
| 100 Mbps | 1.305 s | 20.765 s | 195.905 s | 0/10 |

### 6.4 修复后结果

修复内容：人体 104 帧播放完后，保持最后人体姿态，继续等待和展示 scene chunks，直到完整场景到达。

#### 1 Gbps

| 指标 | 数值 |
|---|---:|
| 人体首帧网络时间 | 0.14848 s |
| receiver 模型加载时间 | 20.120 s |
| 第一场景块到达 | 2.11248 s |
| 完整场景到达 | 19.78848 s |
| 场景块到达时间 | 2.112、4.076、6.040、8.004、9.968、11.932、13.896、15.860、17.824、19.788 s |
| 人体内容帧数 | 104 |
| 尾部等待 slots | 93 |
| 视频总时长 | 19.78848 s |
| frames late | 1 |
| 平均 render | 5.437 ms |
| 连续渲染能力 | 183.92 FPS |

指标和视频：

```text
results/progressive_center_tail_1gbps/bike_1000mbps_10chunks/
```

#### 100 Mbps

| 指标 | 数值 |
|---|---:|
| 人体首帧网络时间 | 1.3048 s |
| receiver 模型加载时间 | 20.813 s |
| 第一场景块到达 | 20.7648 s |
| 完整场景到达 | 195.9048 s |
| 场景块到达时间 | 20.765、40.225、59.685、79.145、98.605、118.065、137.525、156.985、176.445、195.905 s |
| 人体内容帧数 | 104 |
| 尾部等待 slots | 1843 |
| 视频总时长 | 195.9048 s |
| frames late | 104 |
| 平均 render | 4.350 ms |
| 连续渲染能力 | 229.87 FPS |

指标和视频：

```text
results/progressive_center_tail_100mbps/bike_100mbps_10chunks/
```

### 6.5 结论

1. 人体优先显著降低首个可见时间：100 Mbps 下从完整场景约 40 秒降低到人体约 1.3 秒；
2. 1 Gbps 下人体约 0.15 秒即可出现；
3. 低带宽下场景渐进补全仍非常慢，100 Mbps 需要约 196 秒；
4. 场景中心距离策略可以作为 baseline，但不具备接触感知能力；
5. 人体播放结束后保持最后姿态的 tail 机制已经修复，可以完整展示场景逐块加载过程。

## 7. 昨天实验的整体结果汇总

| 实验 | 主要设置 | 关键结果 | 状态 |
|---|---|---|---|
| raw 体积 baseline | 502 MB 静态 + 6 MB/frame raw correction | 1 Gbps 仅约 14.7 FPS，无法 30 FPS | 完成 |
| Savgol compression | window=11 + int8 diff + zlib | 6 场景 payload 约为 raw 的 18.4%～20.4% | 完成 |
| 真实 receiver 渲染 | bike，真实 HUGS renderer | GPU 约 103～104 FPS；1 Gbps packet 可满足 30 FPS | 完成 |
| 10 FPS 因果验证 | bike，真实播放速度 | 1 Gbps 0/104 迟到，100 Mbps 103/104 迟到 | 完成 |
| center-distance progressive | 80% correction / 20% scene | 人体首帧 0.15 s（1 Gbps）或 1.30 s（100 Mbps） | 完成 |
| progressive tail 修复 | 人体结束后保持最后姿态 | 完整展示至所有 scene chunks 到达 | 完成 |

## 8. 当前实验能证明什么

已经证明：

1. 原始 correction 流的动态带宽需求过高；
2. Savgol + temporal difference + int8 + zlib 可以显著降低真实 packet payload；
3. 在固定 1 Gbps、20 ms RTT 的网络模型下，压缩 correction 可以满足 30 FPS packet 到达；
4. HUGS receiver-side GPU 渲染不是当前主要瓶颈；
5. 人体优先可以显著降低首个可见时间；
6. 场景渐进加载可以将完整场景等待转化为逐步补全过程。

## 9. 当前不能证明什么

昨天的实验尚不能证明：

- 在丢包网络下的鲁棒性；
- 在带宽波动下的自适应能力；
- packet 重传和断线恢复能力；
- scene chunk 丢失后的恢复能力；
- 接触感知调度优于中心距离调度；
- 动态带宽分配优于固定 80/20 分配；
- 真实跨机器网络环境下的端到端时延；
- 接触区域 PSNR、接触边界误差和 temporal flicker 的改善。

当前 receiver 的基本容错只有：

```text
当前 correction 未到达
→ 使用最近一次已解码 correction
→ 继续使用当前 SMPL pose 渲染
```

这属于播放连续性机制，不是完整的丢包恢复方案。

## 10. 后续实验接口

基于昨天的结果，下一阶段建议按以下顺序扩展：

1. 将 center-distance 作为固定 baseline 保留；
2. 增加 contact-aware scene chunk 排序；
3. 增加 SMPL trajectory-aware prefetch；
4. 对比 uniform、human-first、center-distance、contact-aware、contact+trajectory；
5. 增加随机丢包、burst loss、jitter 和带宽阶跃变化；
6. 增加周期性 I-frame、packet 重传和自适应带宽分配；
7. 增加人体区域、接触区域、接触边界和 temporal stability 指标。

