# RICH 下载与数值整理

**完成状态（2026-09-16 10:25复核）**：本批train图片于02:15下载完，04:42整包校验与解压完成，已发布到`datasets/RICH/extracted/train`。共261429图片、62序列，解压559.593GB；完整图片索引位于`processed/images_v1/images.jsonl`。全部37669个contact人体帧都有至少一个官方metadata允许相机的图片，基础覆盖缺失0。人体/接触数值整理及公共资产也已完成。下载、处理controller均已正常结束，下文运行PID属于历史记录。报告`reports/rich_processing_20260916/image_completion_audit.json`。

下一层监督样本准备已实现，入口为 `scripts/prepare_rich_dataset.py`，配置和产物契约见
[RICH_DATASET.md](RICH_DATASET.md)。它核验完整图片头/EXIF与标注哈希、冻结互斥开发split、
生成图片-人体-相机样本关联；完成度以 `processed/dataset_v1/report.json` 为准。
正式训练仍需真实前端缓存、图像变换适配与独立几何审计；本批不包含官方val/test。

所有命令可从仓库根目录运行。默认数据在 `datasets/RICH/`；处理器支持 `--root` 指定另一台机器的数据目录。代码与官方小型元数据通过 Git 迁移，原始归档、处理数据和认证文件不进入 Git。

## 当前流水线

```bash
python forward_contact_pipeline/scripts/acquire_rich.py --status
python forward_contact_pipeline/scripts/process_rich_assets.py --status
python forward_contact_pipeline/scripts/prepare_rich_annotations.py --status
```

下载器保留 aria2 partial/control 文件；传输程序失败最多尝试 3 次，每次重新建立官方认证会话，验证错误不盲目重试。不能把稀疏 `.part` 文件的逻辑大小当已下载字节数。

归档处理器保留原压缩包，先在独立暂存目录解压，验证 ZIP 每个成员 CRC 或 TAR.GZ 完整 gzip trailer/CRC，记录归档 SHA256，再发布输出目录。不打开或显示图像像素。以下命令在处理任务退出后用于恢复；已有任务运行时锁会拒绝重复启动：

```bash
python forward_contact_pipeline/scripts/process_rich_assets.py --start --wait-hours 36 --stream-jpg
```

该任务最长等待 36 小时；超时或进程退出后需重启命令，并非永久守护服务。
默认仅处理完整归档；`--stream-jpg` 启用下述边下载边解压模式。

## 图片边下载边处理（2026-09-15 已启用）

19:03 启动的处理 controller PID67144 使用 `--stream-jpg`；它替换了仅等待完整下载的旧处理 controller，下载进程保持运行。初始连续完整前缀约160.2GB，可以先处理这一区段。gzip 是顺序压缩格式，不能跳过尚未下载的压缩数据去解后面的片段。

- `rich_streaming.py` 只读取 aria2 完整 piece 位图证明连续到齐的前缀，遇到空洞等待位图更新，保留解压状态后继续。底层文件以无缓冲方式读取，防止预读稀疏空洞导致下载补齐后仍使用旧的零字节。
- 暂存图片在 `extracted/.staging/train_jpg_stream/train/`；只有完整 TAR 成员才从 `.extracting` 改为正常图片文件名。它们仍属于 **待整包校验的暂存数据**，不能绕过训练准入。
- `processed/images_v1/images.provisional.jsonl` 逐文件记录 SHA256、字节数、JPEG 头中的尺寸、sequence/camera/frame，以及对应的人物标注；仅读取 JPEG 标记头，不解码或显示像素。没有对应标注的图片保留并显式记录，不制造标签。
- `processing_logs/train_jpg.json` 记录文件数、已读取压缩字节、连续前缀大小、尺寸统计与状态。`waiting_for_download_piece` 表示在下载缺口等待，不是失败。`pending.json` 在流式过程可能仍列出 `train_jpg`，详细进度以前者为准。
- 当下载器生成正式 `train.tar.gz`，且整个 gzip 的 CRC/trailer 通过、压缩文件全部读完并计算 SHA256 后，直接发布到 `extracted/train/`，索引改名为 `processed/images_v1/images.jsonl`；正常不中断运行无需重解已提取的前缀。
- 进程中断后可重启同一命令，但 gzip 必须从开头重放，已拥有的暂存成员会重写。本机源文件 inode 与标注清单绑定，不能直接把未完成的流式暂存任务当作可跨机续跑的缓存。

`status=complete` 仅代表图片归档提取和整包校验完成；图像解码质量、K/坐标精度、训练划分和真实前端审计仍另行执行。

## GT 标注整理

```bash
# 每个人物轨迹抽查首/中/末三个已标注帧
python forward_contact_pipeline/scripts/prepare_rich_annotations.py --sample
# 全量后台处理，退出后再次执行同命令即可校验已完成分片并续跑
python forward_contact_pipeline/scripts/prepare_rich_annotations.py --start
```

标注处理直接读取完整的 body/contact ZIP，不必等待 JPG 或全量解压。读取的每个 ZIP 成员由 zipfile 校验 CRC。每帧核验人体 PLY 与接触 OBJ 的 10475 个顶点、20908 个三角面的对应关系；最大坐标分量差须不超过 `2e-6 m`。所有帧必须保持与首次核验一致的拓扑，未知格式、未知颜色及不一致坐标会停止处理并记录失败。

默认产物在 `datasets/RICH/processed/annotations_v1/`：

| 文件 | 内容 |
|---|---|
| `catalog.json` | 官方 sequence/subject/capture/scan/gender/camera 映射、帧清单、缺失标签记录、来源与哈希 |
| `calibration.json` | 原始相机标定、场景引用与 SHA256、原始 world 变换及明确矩阵约定 |
| `sample_audit.json` | 每个轨迹的首/中/末抽样核验，不能代替全量审计 |
| `smplx_topology.npz` | 已核对的统一 SMPL-X 三角面索引 |
| `shards/<sequence>/<subject>/*.npz` | 按轨迹分组、每片最多 32 帧的纯数值资产，`allow_pickle=False` 可读 |
| `state.json` | 进度、分片 SHA256、标签统计及缺失颜色帧；只有 `status=complete` 代表全量数值整理结束 |
| `controller.log` / `launch.json` | 后台运行日志、PID 与启动时间 |

分片保留实际 `frame_ids`，不能假定相邻条目时间上连续。字段：

- `body_*`：官方 float32 人体参数，去除每帧最外层长度 1 的维度；手部是 12 维 PCA，`body_pose` 是 63 维。不改变 SMPL-X root 或重新拟合身体。
- `vertices_multicam [T,10475,3]`：原始人体 PLY 顶点，单位米，位于 calibrated multicamera 坐标。
- `contact_smpl [T,6890]` 与 `contact_smpl_valid`：pickle 中的官方 SMPL 接触。不得映射为 SMPL-X 顶点标签。
- `contact_smplx [T,10475]` 与 `contact_smplx_valid`：接触 OBJ 的绿色顶点是 positive，已知灰色顶点是 negative。无颜色 OBJ 的整帧有效 mask 为 false，不能当全负样本。未知颜色报错。
- `s2m_dist_id [T,10475,3]`：原始 float64 位移字段，不以距离阈值制造 binary contact。`closest_triangles_id` 仍保留在原 ZIP/解压 pickle，本阶段不重复打包。

官方资源固定到 `paulchhuang/rich_toolkit@3b6cf83b0f09e2551f5a20cb180073dea6132cfc`，配置在 `configs/data/rich_metadata/`，来源文件已核对 Git blob SHA1 并记录 SHA256。

## 已核验边界与下一步

2026-09-15 清单：62 条序列、64 个人物轨迹，body 37679 人体帧，contact 37669 人体帧；10 个缺 contact 的帧列在 `excluded_missing_contact`，不生成伪标签。

15:23 完成复核：公共包、body、contact 全部解压并通过 ZIP 成员 CRC 与归档 SHA256。
GT 全量 37669 帧已于 14:37 处理完成，生成 1206 分片、压缩后约 10.453 GB；
全量人体/接触顶点和三角面对应通过（最大坐标分量差约 5e-9 m）。索引帧恰好覆盖清单、无重复遗漏，
所有分片存在；首/中/末三片 SHA256、`allow_pickle=False` 回读、帧号与形状复核通过。
全量有 **84 帧** 无 OBJ 顶点颜色，SMPL-X valid=false，有效颜色标签帧为 **37585**，
有效顶点 positive 比例约 3.676%。完整名单为输出目录内 `missing_smplx_color_frames.json`；
完成报告为 `reports/rich_processing_20260915/annotation_completion.json`。
JPG 于19:03启动上述边下载边处理流程；19:06已暂存3548图片/6.500GB并建立索引，
发现4112×3008及3008×4112两类尺寸，473图的相机不属于官方metadata选用集合，均保留记录。
这些检查尚不能代替逐相机K/方向、完整覆盖和真实前端误差审计。

world 公式依据官方例程为 `p_world_row = c * p_multicam_row @ R + t`，对应列向量 4×4 矩阵的左上块为 `c * R.T`。ParkingLot1 的官方 R 只保留三位小数，正交残差最大约 `5.4e-4`，原值保持并标为精度审计待办。不得据此声称接触尺度坐标审计已经通过。相机内参仍属原标定分辨率，JPG 分辨率和缩放必须在下载后另核验。

按照官方 metadata，跨场景 subject 连接成三个不可拆分的隔离组：`{BBQ}`、`{ParkingLot2}`、`{LectureHall,ParkingLot1,Pavallion}`。当前仅记录这些组，未冻结开发 split。官方 validation/test 尚未下载。

本阶段是 **GT 注释和资产准备**。`training_ready=false`：仍需完整 JPG 与逐帧匹配、分辨率/内参核验、冻结开发 split、真实冻结前端输出与独立几何误差审计，然后才能使用 [TRAINING.md](TRAINING.md) 中的训练入口。GT 顶点不得冒充模型前端 inputs；全量解压完成也不自动解除上述门槛。
