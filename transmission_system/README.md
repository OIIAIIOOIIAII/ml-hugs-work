# HUGS Transmission System

HUGS 动态 Gaussian 传输实验目录。目标是验证：静态模型只发送一次，后续逐帧发送人体动态 correction，在接收端解码并重建渲染。

## 当前架构

```text
delta_mu[T,N,3] + pose
        |
        v
  quantize -> temporal diff -> zlib
        |
        +--> file mode: package/*.json + package/frames/*.bin
        |
        +--> TCP mode: sender.py -> 127.0.0.1 -> receiver.py
                                      |
                                      v
                         delta_mu[T,N,3] -> HUGS renderer
```

本目录目前只实现传输、编码、解码和数值重建；HUGS renderer 通过 `receiver.py` 的输出文件接入。这样可以先独立验证通信闭环，再接入 GPU 渲染，避免把协议问题和训练代码混在一起。

## 文件编排

- `protocol.py`：TCP framing 和 frame packet 定义
- `codec.py`：int8 量化、帧间差分、zlib 编码/解码
- `file_transport.py`：文件模式打包与解包
- `sender.py`：localhost TCP sender
- `receiver.py`：localhost TCP receiver
- `evaluate.py`：原始 `delta_mu` 与重建结果的误差、大小统计
- `make_demo_data.py`：生成小型可运行 demo 数据
- `configs/default.json`：传输配置

## 快速验证

在本目录执行：

```bash
python make_demo_data.py --out demo
python file_transport.py pack --input demo/delta_mu.npy --out demo/file_package
python file_transport.py unpack --package demo/file_package --out demo/file_reconstructed.npy
python evaluate.py --reference demo/delta_mu.npy --reconstructed demo/file_reconstructed.npy --package demo/file_package
```

localhost TCP：

```bash
# 终端 1
python receiver.py --host 127.0.0.1 --port 18765 --out demo/tcp_reconstructed.npy

# 终端 2
python sender.py --host 127.0.0.1 --port 18765 --input demo/delta_mu.npy --fps 30
```

## 基础传输性能 benchmark

模拟“静态资产一次传输 + 每帧 correction 传输”：

```bash
python baseline_benchmark.py \
  --frames 60 --points 500000 \
  --static-bytes 502000000 \
  --bandwidth-mbps 100 \
  --fps 30 --codec raw \
  --out results/baseline_100mbps_raw.json
```

`baseline_benchmark.py` 输出静态传输时间、首帧启动延迟、总传输量、动态 payload、按时到达帧比例、接收 FPS 和 decode FPS。当前 `decode_fps` 只表示网络数据解码/重建吞吐，不等于 HUGS GPU 渲染 FPS；真正的观看 FPS 需要在 receiver 解码后接入 HUGS renderer 再测。

对已有的 `output/delta_mu_analysis/<scene>/delta_mu_f16.npy` 测试 Savgol 版本：

```bash
python baseline_benchmark.py \
  --delta ../output/delta_mu_analysis/bike/delta_mu_f16.npy \
  --static-bytes 502000000 \
  --bandwidth-mbps 1000 --fps 30 \
  --codec causal_int8_zlib --savgol-window 11 \
  --out results/bike_savgol11_1000mbps.json
```

`render_received_stream.py` 默认会把静态资产的模拟加载时间写入视频时间轴（黑屏/加载阶段）；使用 `--no-startup-delay` 才会生成只包含内容帧的短片。已有视频可以用 `add_startup_delay.py` 后处理。

当前 packet 使用 JSON header + binary payload，sender 和 receiver 共用 `protocol.py`。后续可以把 zlib 替换成真正的 entropy coder、把 `delta_mu` 替换成 HUGS 导出的 correction，并在 receiver 解码后调用 renderer。

## 当前限制

- 目前 frame payload 是 `delta_mu` 的演示接口，pose 和静态 GS 仍以扩展字段预留。
- 当前使用 causal temporal difference，适合在线 TCP 验证；Savgol 离线实验不在 socket sender 内强行使用。
- TCP localhost 只能验证端到端协议和处理流程，不能代表跨机器网络延迟。
