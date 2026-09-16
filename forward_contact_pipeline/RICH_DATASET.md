# RICH 监督样本准备

**2026-09-16 全量监督处理完成，真实前端几何门槛未通过。**

- `datasets/RICH/processed/supervision_v2/`：固定 SMPL-X 左/右脚底各64顶点，官方逐顶点接触标签与缺失掩码、可选无符号距离、全量原图/512裁剪画面范围掩码、相机变换、筛选清单和哈希。GT只进入监督或审计，不充当前端输入。
- train保留165920个图像-人体样本，内部val保留74264个；2610个视角双脚均在裁剪外。全部37585个有效人体帧仍至少有一个合格视角。84个无SMPL-X颜色标注帧仍保留invalid，10个body-only帧保持排除接触监督，camera10的14987张图保持缺标定排除。
- 全部1206片及242794候选视角经独立投影/帧号/标签/掩码/哈希回读通过。脚底标签positive比例train55.07%、val69.93%；不是全正标签。画面范围不等于遮挡可见性。
- 冻结GUSH3R纯数值推理完成两个16帧片段，保存真实RGB decoder tokens、SMPL-X、point map及Gaussian诊断；没有渲染、训练或使用未来帧。修正了导出中必须遵循的头部位置平移约定，未修改第三方推理模型。原默认FOV先验K保留，RICH K只用于独立审计。
- 每片段前12帧拟合一个诊断Sim(3)、后4帧留出：全身误差中位数3.39/4.62cm，脚底5.69/2.66cm，均不满足既定2cm门槛。诊断对齐未写回inputs；全量真实特征缓存没有生成，`training_ready=false`。当前阻塞是前端几何精度，下载/GT处理已无后台等待。
- 完成状态：`reports/rich_processing_20260916/data_status.json`；全量审计和几何证据同目录；可迁移说明见`RICH_DATASET.md`。

以下v1记录用于追溯候选清单的生成；可见性筛选进度以上述v2结果为准。

这一层把已解压图片、官方标定与 GT 分片关联起来，产出可迁移的监督样本清单。
它不生成冻结前端特征、不启动训练，也不把 GT 人体或场景作为部署输入。

**2026-09-16 11:00 完成核对**：正式产物已发布至 `processed/dataset_v1/`。
261429 张图片头/尺寸/EXIF 全量检查完成，全部为 identity orientation；1206 个标注分片全部重新校验。
训练为39序列/41人物轨迹/167710候选样本，内部验证为23序列/23轨迹/75084候选样本；
有效 SMPL-X 人体帧分别25071/12514，合计37585。全部样本的图片、帧号、相机、分片/行号再次独立核对通过，
样本ID重复0，scene/sequence/subject跨split重复0。输出哈希复核与10项回归通过。
完成证据：`reports/rich_processing_20260916/dataset_completion_audit.json`。

**可见性仍待筛选**：1290个抽查视角中，17个的GT顶点投影全部在图像边界外，38个少于半数顶点在边界内。
这可能涉及视野范围等因素，尚不能凭数值投影确认实际像素对应或标定误差。
这些标记已记录，不能把242794个候选样本直接等同为最终可训练样本数；
后续须完成逐帧可见性/目标匹配与几何核验。原始数据未因抽样统计而删除或修改。

## 运行与迁移

从仓库根目录运行，使用包含 `numpy`、`Pillow` 的 Python 环境（版本见 `requirements-data.txt`）：

```bash
python forward_contact_pipeline/scripts/prepare_rich_dataset.py \
  --root datasets/RICH \
  --config forward_contact_pipeline/configs/data/rich_dataset_v1.json \
  --workers 8
```

默认输出 `datasets/RICH/processed/dataset_v1/`。所有数据引用都相对 RICH 根目录；
另一台机器保留这个目录结构即可。脚本、配置和说明可以进 Git，数据产物随数据盘迁移。
本轮没有 commit/push，也没有复制或上传原始图片。

处理使用独占锁，在 `dataset_v1.pending/` 写入；完整性及覆盖检查全部通过后才发布正式目录。
失败时保留进度与错误，重跑会重新校验并重建暂存产物，不覆盖已有正式版本。
已完成的图片头记录可在来源 hash、文件大小和 mtime 均相同时复用；变更的文件重新读头，
中断造成的最后一条不完整 JSON 记录会被重新处理。标注分片仍全量重新校验。
已有正式目录时仅核对来源签名和输出文件哈希并返回已有报告；这不是一次原始图片全量重审。
改划分或过滤策略时，应更改版本化配置，并用 `--output ROOT/processed/dataset_v2` 创建新版本。

## 划分和计数口径

`rich_dataset_v1.json` 将 ParkingLot2 全部划为内部开发验证集，其余捕获场景为训练集。
每次生成都检查 capture、scene、sequence、subject 四种身份不跨 split，双人、多相机序列一起归属。
这是从官方 train 划出的开发集，不能称为官方 validation/test。

一个样本表示“某人物、某真实帧、某相机的一张图”，因此多视角样本数大于独立人体帧数。
不能把不同相机拍到的同一动作当成独立泛化证据；后续采样/指标要保留人物轨迹和序列分组。
当前不随机切帧、不把帧号重排成从零开始、不插值补帧，也不按 contact positive 数量选择样本。

## 产物契约

| 文件 | 用途 |
|---|---|
| `report.json` | 总数、split 统计、排除原因、来源签名、产物哈希和待办边界 |
| `splits.json` / `config.json` | 冻结的开发划分和生成配置 |
| `samples_train.jsonl` / `samples_val.jsonl` | 通过帧关联、相机结构与标签检查的候选样本；尚未完成逐帧可见性筛选 |
| `image_headers.jsonl` | 全部图片的尺寸、EXIF orientation、原始哈希引用及检查时 mtime |
| `cameras.json` | 逐场景/相机的图片尺寸、EXIF 分布、原始 K/E、畸变和结构检查 |
| `excluded.jsonl` | 不能进入当前监督清单的样本/图片及原因；原文件保留 |
| `missing_contact_frames.json` | 官方 body 存在但缺 contact 的人体帧 |
| `projection_audit.json` | 每轨迹首/中/末 GT 顶点的数值投影统计，不是实际像素对应或遮挡检查 |
| `pilot_clips.json` | 每个 split 的一条 16 帧连续小样本计划，供后续前端接入；尚未运行 |
| `provenance.json` | 来源清单哈希、官方 metadata revision、归档哈希与校验范围 |

每个样本包含原始图像路径/hash、真实 `frame_id`、subject、camera、split、scene、
`annotation.path/sha256/row` 和有效标签顶点数。读取标签时必须同时检查：

```python
with np.load(rich_root / sample["annotation"]["path"], allow_pickle=False) as shard:
    row = sample["annotation"]["row"]
    assert int(shard["frame_ids"][row]) == sample["frame_id"]
    contact = shard["contact_smplx"][row]          # [10475]
    contact_valid = shard["contact_smplx_valid"][row]
```

`camera_key` 关联本版本 `cameras.json`；`scene` 关联
`processed/annotations_v1/calibration.json` 中的 scan 路径、hash 和 world 变换。
这一层没有确定脚部 ROI。后续应从固定模板与语义定义确定 ROI，再按同一顶点 ID 取标签，
不能用每帧 GT contact 或 GT 最近地面顶点来选择部署输入位置。

## 检查范围

- 重新读取全部 1,206 个标注分片 SHA256，检查真实帧号、shape、有效 mask、逐帧覆盖。
- 全量图片检查文件大小、JPEG 头和 EXIF，禁止调用像素 load/convert/transpose。
  图片逐文件 SHA256 来自之前的完整归档提取；本轮不再读取 559 GB 图片内容重新计算。
- 相机尺寸固定、K/E 有限、相机旋转合法、主点在原始图像范围内、畸变为零。
  Pavallion 的 camera 3/5 保留竖幅，BBQ camera 3 保留其独立尺寸。
- camera 10 无标定/不在官方选用相机中，保留图片并排除监督样本。
  若出现非 identity EXIF，则排除并要求显式图像/相机变换适配，不擅自旋转。
- 无颜色的 SMPL-X contact OBJ 以 unknown 处理，不能当成全负；独立 SMPL 6890 标签仍保留。
- `p_camera = E @ [p_multicam; 1]`；world 行向量公式为 `c * p_multicam @ R + t`，
  列向量矩阵左上块为 `c * R.T`。保留 ParkingLot1 官方有限精度系数，不静默正交化。

结构与数值合理性检查不证明 K 已与实际像素精确对齐，也不证明前端达到接触尺度精度。
`supervision_index_ready=true` 和 `training_ready=false` 可以同时成立。

## 前端接入仍需完成

已检查 `GUSH3R/infer.py`：`prepare_input()` 调用含 EXIF、resize、center crop 的 `load_images()`，
并通过 `get_camera_parameters()` 构造默认相机内参；当前入口没有把 RICH 的 K 直接接入。
后续必须区分模型默认相机先验与 GT 评测相机，记录 raw→resize→crop（另有人体分支 pad）的映射；
不能将原始 K 原封不动用于裁剪后的像素，也不能把改用 GT K 的实验混称为默认前端实验。

`export_contact_state()` 导出按 opacity 选择的 Gaussian 数值以及部分 SMPL/camera 张量，
目前缺少完整 RGB tokens、每个脚部顶点的局部邻域、真实 RICH 帧/相机映射与完整 LBS binding。
导出使用从零开始的序号；接入时须与 `pilot_clips.json` 的真实帧号一一对应。
全局 opacity top-k 也可能丢掉接触附近的场景证据，需要局部覆盖验证。
当前普通 `infer.py` 主流程在导出后还会调用 `render_video()`；数据准备阶段不能直接用它
当成纯数值 exporter，应先提供独立的纯数值推理入口。本轮未调用该推理/渲染流程。

下一步先接两条小样本的真实冻结因果前端，测量人体/场景坐标误差，再构建
`inputs.npz` 与独立 `targets.npz`，满足 [TRAINING.md](TRAINING.md) 的契约后才能训练。
当前 manifest 不可直接送给 `build_training_index.py`，也不可手填 geometry audit 为通过。

## 测试

```bash
python -m unittest discover -s forward_contact_pipeline/tests -p test_rich_dataset.py -v
```

覆盖跨人物泄漏拒绝、非连续真实帧关联、SMPL/SMPL-X 有效 mask 区分、相机缺失过滤、
不解码像素的 EXIF 读取、竖幅投影、失败不发布、路径边界与跨目录迁移后的清单哈希核验。

## 全量脚底监督和画面范围掩码 v2

从仓库根运行（同一输出目录不能并发写入）：

```bash
OPENBLAS_NUM_THREADS=1 python forward_contact_pipeline/scripts/prepare_rich_supervision.py \
  --rich-root datasets/RICH --output datasets/RICH/processed/supervision_v2 \
  --template Human3R/src/models/smplx/SMPLX_NEUTRAL.npz --workers 4
OPENBLAS_NUM_THREADS=1 python forward_contact_pipeline/scripts/audit_rich_supervision.py \
  --rich-root datasets/RICH --supervision datasets/RICH/processed/supervision_v2 \
  --output forward_contact_pipeline/reports/rich_processing_20260916/supervision_completion_audit.json
```

`targets/<sequence>/<subject>/<offset>.npz`保留原分片的真实`frame_ids`，
含`contact/contact_valid [T,2,64]`、固定`vertex_ids [2,64]`。
`unsigned_surface_distance`为官方`s2m_dist_id`位移向量的范数，单位米；不能重命名成signed proximity。
`frustum/`按候选相机/帧保存`sample_ids`、原annotation行号、脚底原图/裁剪mask、
裁剪像素坐标及全身入画顶点数。`contact_supervision_mask = contact_valid & sole_crop_frustum`；
不做遮挡判定，不把出画顶点的标签改为0。筛选清单的`target.path/frustum.path`相对于v2目录，
其他image/annotation路径仍相对于RICH根目录。只要任一有效脚底顶点在裁剪内就保留该候选样本。

ROI只依赖neutral模板的skinning joints 7/10、8/11及法向、模板高度，随后确定性FPS；
不按GT接触值挑选。版本/源文件/实现哈希改变必须使用新输出目录。续跑逐片检查源和已写输出哈希；
`report.json`完成状态与独立审计同时有效才算验收，不仅凭文件夹存在判断。

## 冻结前端数值诊断

需要已安装GUSH3R的CUDA环境、原检查点、SMPL-X资产和本地DINOv2代码，不能用仅numpy环境运行。
以下路径变量由目标机器设置，不把本机conda路径写进实验配置：

```bash
python forward_contact_pipeline/scripts/export_rich_gush3r_pilot.py \
  --repo GUSH3R --rich-root datasets/RICH \
  --output datasets/RICH/processed/frontend_pilot_v1 --dino-repo /path/to/local/dinov2
python forward_contact_pipeline/scripts/audit_rich_gush3r_pilot.py \
  --rich-root datasets/RICH --frontend datasets/RICH/processed/frontend_pilot_v1 \
  --output forward_contact_pipeline/reports/rich_processing_20260916/frontend_geometry_audit.json
```

exporter调用`keep_outputs=False`、逐帧callback与`skip_inference_render_outputs=True`，不调用普通CLI末尾的render。
每个片段独立重置因果状态；没有宣称跨片段连续记忆。记录原图→resize/crop→MHMR pad矩阵，
区分模型默认K与真实标定K。HumanGS的`trans`为头部位置，导出mesh采用
`SMPL-X vertices + trans - posed_head`；中性性别/flat-hand-mean与HumanGS内部层一致。
`binding.npz`保留原sample vertex IDs、LBS weights和拓扑；这仍不是完整的低维修正/LBS部署闭环。
scene只保存当前累计地图中最多8192个高opacity Gaussian用于诊断，不能当作完整局部surface或接触标签。

此轮两个clip各16帧；78.96秒包括加载/校验/推理/落盘，不是全量吞吐保证。
出现检测/身份异常会使审计失败。短片段通过也不代表全量就绪，须扩展多序列测量并实际生成
local relation缓存后才可运行真实训练。本轮未过门槛，后续先诊断/改进前端或明确变更实验协议，
不能修改audit pass标记绕过门槛。
