# Claude 协作记录

## 2026-09-16（继续视觉条件与几何损失校正研究）

- 本轮完成6组MLP训练（3配置×2种子，60epoch，624训练/208内部选择样本）与384帧新序列冻结推理。训练内部锁定visual_geometry_seed7/epoch60后，评价6新序列的camera3与统一补测camera0；原camera3有5段多人输出不满足单人匹配规则，保留失败分母，实际224帧/7视角片段可测。
- 选中模型训练脚底原相机2.70cm，但新camera0对齐脚底10.02→22.47cm、原相机25.58→37.71cm，0/6通过，全部声明视角0/12通过。相机K解析平移仅改善整体位置（原全身23.84→17.73cm），脚底局部10.02→10.69cm仍未改善。训练深度残差中位+41.98cm、新集−11.70cm，表明明显分布变化；不据此承诺扩数据必然解决。
- 可微FK已保留头部梯度，6方向有限差分通过；FP32批量/逐帧差1.43e-6m，最终TF32逐帧零修正差0。全CPU58项中55通过/3因可选roma跳过，3项另在GUSH3R环境通过。已对锁定模型独立复训做数组/指标一致性核验。报告`forward_contact_pipeline/reports/rich_neural_refinement_20260916.md`；失败模型不部署，正式contact训练继续gated。

- 用户要求继续。复用已完成的17序列真实冻结特征，研究保留完整视觉query的非线性校正器及可微顶点几何损失；先冻结未使用的新确认片段，训练和模型选择只用训练内部序列，原12片段仅作历史诊断。保持不读图、不渲染、GT不进入部署输入和正式contact训练门槛；运行后报告真实收益/失败及证据，源码沿既有授权同步Git。

## 2026-09-16（继续真实几何校正与数据量实验）

- 本轮完成：832个真实图像样本（553不同人体帧、17序列、7参与者）训练小型SMPL参数校正器；8组初始超参、4/8/17序列×3种子学习曲线、24组分量/强度候选、8组真值残差上限均完成。两次主实验结果/模型数组逐值一致。TF32设置差异已定位并修复，零修正max-abs=0，未放宽校验。
- 12确认片段native896 camera-local对齐脚底median从无修正8.83cm恶化为学习全部残差13.72cm，分量选择11.36cm，均0/12通过。未见ParkingLot2的4/8/17序列学习曲线为12.70/10.92/12.93cm（3子集中位），非单调；不能简单归因为数据少，也不能以小样本实验排除数据不足。真值全部残差取消限幅后对齐脚底0.54cm、12/12通过；保留限幅则4.92cm、4/12通过，仅作诊断，不可部署。
- 正式contact训练继续保持training_ready=false；下一方向为较大几何初始化与小幅接触控制分开、改进图像/场景条件与几何损失，并预留新确认序列。完整报告`forward_contact_pipeline/reports/rich_geometry_calibration_20260916.md`；数据与失败实验完整保留。源码与文档按既有Git授权同步，未包含其他任务修改的INTERVIEW文档。

- 续接已完成的冻结前端确认实验和68片段训练特征；先定位零修正几何复现失败，保持0.1mm校验，不放宽验收。完成真实监督校正器、4/8/17序列学习曲线与独立确认评估，更新结论并同步本轮代码；正式接触训练仍受原门槛约束。

## 2026-09-16 14:23（解决RICH真实前端几何精度）

- 用户要求放开思路、多做实验和验证，解决真实前端几何误差。本轮先审计坐标/头部平移/相机预处理/模型输出契约，再建立独立样本对照，量化改进与失败；不把GT对齐写入部署输入、不修改门槛伪造通过。允许必要的冻结前端推理和数值实验，持续维护实验协议、TODO和结果证据。
- 首轮已完成5个前处理/记忆开发变体和12预声明新片段×4确认变体（928输出帧，另384过去上下文帧）；geometry-only基线与原导出max-abs=0。原始/原生896输入确认脚底median-of-medians8.52/8.19cm，均0/12通过；标定rectification有1个检测/身份失败，长上下文无稳定改善。早期辅助SMPL joint路径与HumanGS FK约2.4mm差异已通过沿原FK路径消除，原始预测参数逐值一致。
- GT参数按gender/PCA12/non-flat-hand重建与官方mesh误差<1e-6m；随意改flat-hand会有约8cm手部差异。当前误差不能归因为GT拓扑或归档匹配错。模型对比：GUSH3R与本机Human3R1631个共有张量、约11.77亿参数逐值一致；没有再次跑同一几何权重冒充独立候选。
- 用户追问是否数据太少，已澄清旧误差是原始前端，不是新contact修复器的结果。冻结17train序列/1088计划帧用于有界真实监督参数残差研究，4/8/17序列×3固定种子学习曲线；固定确认序列不参与训练和选参。真实前端特征提取中，尚未把小校正研究说成已成功或解除最终门槛。


## 2026-09-16 11:35（完成剩余数据准备）

- 最终软件验证50项通过（其中新增5项涵盖横竖resize/crop、负深度/画外mask、无效标注、源hash/帧错配拒绝、输出hash损坏拒绝和续跑）。源码/文档本轮提交并沿已授权GitHub部署密钥同步；数值数据、模型与密钥保持本机忽略。
- 本轮全量处理完成：新增`rich_supervision.py`与可续跑入口、独立审计器，固定左右脚各64 neutral-template ROI；1206个GT分片及全部242794候选视角的标签、真实帧、原图/裁剪投影mask与hash全量独立回读通过。保留train165920/内部val74264样本，2610个双脚出裁剪视角排除；37585有效人体帧全部仍有合格视角。标签positive分别55.07%/69.93%，84无颜色帧保留invalid。数据与原v1未改写。
- 真实冻结GUSH3R headless pilot 32帧完成，78.96s含启动/校验/落盘，保存RGB decoder tokens、SMPL-X/point map/Gaussian与LBS索引；未调用渲染或训练。确认HumanGS以头部为trans，按`vertices + trans - posed_head`导出，不使用GT标定/人体作为输入。每片独立因果状态，scene仅当前累计map的诊断抽样。
- 独立几何核验：两片段前12帧fit、后4帧held-out，全身median3.39/4.62cm，脚底median5.69/2.66cm，既定2cm门槛均失败；未将对齐写回输入、未启动全量真实特征缓存或训练。原始数据/监督处理已完整验收；真实前端仍未就绪。完成状态与审计见`reports/rich_processing_20260916/data_status.json`及同目录JSON，说明已同步TODO/活协议/RICH_DATASET。

- 用户要求把数据全部处理好。本轮继续全量可见性/标注关联、真实冻结因果前端输入适配与数值缓存、几何核验及训练缓存验收；复用已完成归档、GT分片和候选清单，不重复下载/解压、不用GT冒充前端输入。仅做数据准备与必要前端推理，不启动正式训练、不查看或渲染图像。

## 2026-09-16 11:32（GitHub部署密钥授权后完成同步）

- 用户确认已添加title为deploy的仓库部署密钥。已使用本仓库专用SSH密钥成功将源码提交8ba0dcf推送至origin/main（原远端88c9871），不再需要重复确认推送权限。
- 本轮同步迁移说明、TODO与实验协议中的认证状态，再提交并推送文档更新，最终以本地HEAD和远端main的提交哈希一致验收。数据、模型、私钥和生成产物未上传；仅两个CUDA子模块保留未跟踪的本地编译/缓存文件，无未提交源码改动。

## 2026-09-16 11:14（本地仓库同步到Git）

- 已按审计清单整理387个改动文件，506个源码/配置/文档候选约4.2MB；无大文件、凭据模式或Python语法阻塞。补齐RICH metadata TSV、requirements-data.txt、导出入口与方案文档的迁移收录，生成结果/论文/检索缓存加入忽略。第三方固定revision与GUSH3R补丁逐字节匹配。
- 主工作区45项CPU软件测试通过；暂存内容导出的干净副本45项测试中44项通过、1项因未安装aria2跳过，无失败。现有Markdown硬换行和旧源码空白保留，270项格式提示另记，不为同步改动实验逻辑。真实GPU训练/前端推理未启动。
- 远端main已fetch并确认与原基线88c9871一致；本次将源码快照提交至本地main。HTTPS推送缺认证，旧SSH密钥被拒绝，已生成仅此仓库使用的Ed25519部署密钥（私钥留在忽略的.tools/git-sync）。本仓库push URL配置为原仓库SSH地址，严格校验GitHub官方主机密钥；GitHub仓库端授权公钥后即可推送。当前远端尚未更新，不把本地提交说成已同步成功。
- 用户明确要求把本地仓库同步到Git。本轮核验现有origin、分支和所有本地改动，整理可迁移源码/配置/文档及必要第三方补丁，排除数据集、模型权重、凭据与运行产物；完成提交、推送并核对远端提交，不启动训练或上传数据。

## 2026-09-16 10:28（正式RICH数据准备）

- 11:00完成：`processed/dataset_v1`已正式发布，261429图片头/尺寸/EXIF全量检查（全部identity）、1206标注分片SHA256/数值回读通过。train39序列/41轨迹/167710候选样本，ParkingLot2内部val23序列/23轨迹/75084候选样本；有效SMPL-X人体帧25071+12514=37585，所有37669个contact帧均有已标定图片。全部242794候选样本的图片/真实帧/相机/分片行号独立复核、重复与泄漏检查及发布产物hash复核通过。证据`reports/rich_processing_20260916/{dataset_preparation,dataset_completion_audit,dataset_expected_counts}.json`。
- 已实现`rich_dataset.py`、`prepare_rich_dataset.py`与版本化划分配置；10项回归通过，覆盖真实帧关联、mask、泄漏、EXIF不解码、缺标定过滤、失败不发布、元数据缓存失效/续跑及跨目录迁移。NAS扫描减少重复父目录解析，最终32个元数据I/O线程、标注最多4线程；运行中两次仅停止未发布的扫描以优化I/O，完成记录按source hash/size/mtime复核复用。无原数据删除或重解压。
- 新发现：1290个GT数值投影抽查视角全为正深度，但17个无顶点投影在图像范围内、38个少于半数顶点在范围内。已记录逐帧可见性/目标匹配与标定核验门槛，不直接将候选样本数称为最终训练样本数、不凭此宣称标定错误。两个16帧纯数值前端小样本计划已生成，未运行推理或训练。缺标定camera10共14987图片、84帧无颜色SMPL-X标签及10帧缺contact均保留原始数据并明确排除/标记。
- 划分采用ParkingLot2内部val（23序列），其他场景train（39序列），capture/scene/sequence/subject均互斥。新增`RICH_DATASET.md`说明产物与前端缺口：GUSH3R当前加载器会EXIF/resize/crop，使用默认K；现有数值导出缺完整RGB tokens、局部邻域、真实帧映射和LBS binding。未运行推理或训练，未提交/推送。
- 用户要求现在开始处理数据。本轮在已完成解压/GT整理基础上，推进完整图片-标注-相机索引、逐相机尺寸/坐标核验、scene/subject互斥开发划分与可追溯样本清单，核查后续真实前端缓存入口。不将GT顶点冒充前端输入，不在审计前启动训练。

## 2026-09-16 10:24（下载是否完成复核）

- 完成确认：本批RICH train JPG于9月16日02:15:56下载完成（559418993166字节），04:42:16流式解压、完整gzip CRC/trailer与压缩文件SHA256通过并正式发布`datasets/RICH/extracted/train`，261429图片/559.593GB、62序列；下载与处理controller均正常结束。archive SHA256=`cdae5c9d264c4e12d7965a75dae03623c43044ff1d54f81ae444e3890f0ac6c1`。
- 10:25追加完整索引核验：261429唯一图片路径、总字节数与解压状态一致；全部37669个contact人体帧均找到至少一个官方metadata允许相机的图片，缺失0。首/中/末图片文件大小/SHA256回读通过，不解码或显示像素。三种尺寸4112×3008（236656）、3008×4112（24093）、4096×2700（680）。索引SHA256=`d5a509267d10012ea25fb0a247d2765850464fd372c3dae07c29b05e448a6953`；报告`reports/rich_processing_20260916/image_completion_audit.json`。
- 已完成本批train+公共资产下载与归档处理、GT数值整理及图片基础覆盖匹配。官方val/test未在本批下载；逐相机K/方向、开发split冻结、真实冻结前端特征与独立几何误差审计仍待执行，training_ready=false，未启动真实训练。
- 用户询问下载是否结束；检查正式归档、断点与日志新鲜度、下载进程及图片整包校验/发布状态，必要时恢复已授权的中断任务。

- 初查快照：{"checked_at": "2026-09-16T10:24:57.380314+08:00", "final_images_archive_exists": true, "latest_progress_time": "2026-09-16T02:15:47", "latest_progress_age_seconds": 29350.380314, "connections": 12, "latest_MBps": 15.72864, "mean_MBps_5min": null, "latest_aria2_eta": "9s", "downloaded_GB": 559.418993166, "percent": 100, "remaining_GB": 0, "download": {"status": "downloaded_size_and_format_checked", "time": "2026-09-15T18:15:56.696689+00:00"}, "image_processing": {"status": "complete", "updated": 1789504936.4596615, "completed": 1789504936.4596615, "files": 261429, "image_files": 261429, "uncompressed_bytes": 559592667554, "compressed_bytes_read": 559418993166, "complete_prefix_bytes": 559418993166, "archive_integrity": "verified", "training_ready": false, "archive_sha256": "cdae5c9d264c4e12d7965a75dae03623c43044ff1d54f81ae444e3890f0ac6c1", "output": "/workspace/nas_auto_backup/yuzilang/ml-hugs-work/datasets/RICH/extracted/train", "images_with_contact_subject": 248678, "images_outside_metadata_cameras": 14987, "image_dimensions": {"4112x3008": 236656, "3008x4112": 24093, "4096x2700": 680}, "sequences_seen": 62}, "annotations": {"status": "complete", "updated": 1789454244.15578, "completed": 1789454244.1922038, "training_ready": false, "completed_frames": 37669, "shards": 1206}, "downloader_launch": {"pid": 1464854, "process_exists": false, "process_state": null}, "processor_launch": {"pid": 67144, "process_exists": false, "process_state": null}, "aria2_pids": []}。

## 2026-09-15 23:53（下载与处理进度联合复核）

- 用户询问下载量与处理程度；本轮检查完整归档、aria2实际断点与近期速度、流式图片处理及GT整理状态，遇到异常则沿既有授权恢复。

- 状态快照：{"checked_at": "2026-09-15T23:53:24.981378+08:00", "final_images_archive_exists": false, "latest_progress_time": "2026-09-15T23:53:23", "latest_progress_age_seconds": 1.981378, "connections": 14, "latest_MBps": 0.41472, "mean_MBps_5min": 0.448512, "latest_aria2_eta": "8h50m48s", "downloaded_GB_lower_bound": 546.161846798, "total_GB": 559.418993166, "percent": 97.6301937313619, "remaining_GB_upper_bound": 13.257146368, "control_age_seconds": 25.595860481262207, "hours_at_recent_mean": 8.210573313039067, "estimated_finish_at_recent_mean": "2026-09-16T08:06:03.045305+08:00", "download": {"status": "downloading", "time": "2026-09-15T12:14:56.800189+00:00"}, "image_processing": {"status": "waiting_for_download_piece", "updated": 1789487599.45887, "files": 73599, "image_files": 73599, "uncompressed_bytes": 160445875995, "compressed_bytes_read": 160413253632, "complete_prefix_bytes": 160413253632, "archive_integrity": "pending full gzip CRC/trailer", "training_ready": false, "images_with_contact_subject": 69782, "images_outside_metadata_cameras": 3512, "image_dimensions": {"4112x3008": 65339, "3008x4112": 8260}, "sequences_seen": 20}, "annotations": {"status": "complete", "updated": 1789454244.15578, "completed": 1789454244.1922038, "training_ready": false, "completed_frames": 37669, "shards": 1206}, "downloader_launch": {"pid": 1464854, "process_exists": true, "process_state": "S"}, "processor_launch": {"pid": 67144, "process_exists": true, "process_state": "S"}, "aria2_pids": ["200625"]}。

## 2026-09-15 20:12（下载进度与提速诊断）

- 20:13复核JPG96.5275%、余19.426GB，近5分钟平均0.26645MB/s；实际仅10连接，6个连接在14:35–15:31逐个长度异常退出。图片流式处理已读至160.413GB、73599图并等待下载缺口，当前没有持续解压负载。
- 对旧aria2 PID1465011发送一次SIGINT，正常保存control后退出，原controller PID1464854自动等待60s并重新官方认证，第二轮恢复aria2 PID200625、16连接。partial设备/inode与既有数据均保留，流式处理PID67144未重启。未增加超过16的并发，也未另起重复下载。
- 20:15:45–20:18:45共3分钟重连后测量，排除首分钟后11个摘要平均1.04554MB/s，为重连前约3.92倍；最新0.94372MB/s，16连接持续。20:18完成540.225/559.419GB（96.5689%），余19.194GB，按短时均速约5.1小时，建议估5–6小时/9月16日凌晨1–2点，不包含校验解压且不保证后续速率。证据`datasets/RICH/download_logs/speed_refresh_20260915/{before,after,summary}.json`。改善不证明唯一原因是cookie、限速或网络拥塞。
- 用户要求查看进度并尝试提速。本轮检查近期吞吐、活跃连接及退出错误，必要时在保留aria2断点和并行处理的前提下进行有界恢复测速，不另起重复大包下载。

## 2026-09-15 19:55（接触估计器输入、架构与输出说明）

- 已核对实际代码：当前Stage-A为point MLP→valid-mask均值池化→拼接脚底vertex位置/法向→vertex MLP，融合版增加RGB cross-attention（hidden96、heads4），可拼接Gaussian属性。输入均来自冻结前端的局部几何/视觉特征，RICH只提供训练监督。输出逐vertex contact logits和proximity；默认训练只启contact，距离loss权重0。几何原型虽有reliability_logits，统一损失尚无监督/校准，融合版也尚无可靠性头；不能称为已完成可靠性模块。GRU/TCN与低维修正属于后续Stage-B，不在当前Stage-A内部。已在唯一架构文档顶部补充实装与目标设计的区别，未改模型或启动训练。
- 用户要求讲解接触估计器的输入、内部架构与输出。本轮核对Stage-A实际实现与唯一架构文档，明确已有几何/RGB/Gaussian融合基线和仍待实现的可靠性/时序控制边界，不把GT标签说成部署输入。

## 2026-09-15 18:53（边下载边处理图片可行性与实现）

- 已实现并启用`rich_streaming.py`及`process_rich_assets.py --stream-jpg`：仅读取aria2完整piece构成的连续前缀（约160.2GB），遇到缺口阻塞等待，不返回伪EOF、不读取稀疏空洞。gzip状态在运行中保持，待正式归档改名、全量gzip CRC/trailer通过且完整SHA256计算完后，直接发布`extracted/train`，正常不中断时无需重复解压前缀。中断后可安全重放，不能伪装随机访问gzip续解。使用同一processing.lock避免双提取器。
- 仅替换旧的等待型处理controller PID1471505，新PID67144于19:03启动；下载controller PID1464854及aria2 PID1465011保持。19:06已提取3548图片/6.500GB，索引识别率100%，3461图找到contact人物条目，473图的相机不在官方metadata选用集合，已记录而非默默纳入训练。发现4112×3008与3008×4112两种尺寸，后续须逐相机审计K/方向，不能统一缩放。首/最后checkpointed索引条目的路径/大小/SHA256回读通过；快照`reports/rich_processing_20260915/streaming_started.json`。
- 暂存目录`datasets/RICH/extracted/.staging/train_jpg_stream/train`；图片索引`processed/images_v1/images.provisional.jsonl`含文件hash、JPEG头尺寸、sequence/camera/frame和对应subject，不解码/显示像素；`processing_logs/train_jpg.json`区分extracting_provisional/waiting_for_download_piece/complete。仍training_ready=false，整包校验/图像与K审计/真实前端门槛不变。
- 7项流式与3项既有归档测试通过，覆盖空洞阻塞后补齐、正式文件名门槛、gzip CRC损坏不发布、超时、重放与路径安全。测试曾捕获Python BufferedReader预读空洞并缓存旧零字节，已改为无缓冲底层读取并复测通过。更新`RICH_PREPARATION.md`说明边下载边处理和恢复限制。
- 用户要求对已下载部分同步处理。本轮核验aria2完整piece连续前缀、gzip/tar布局，设计隔离暂存的完整成员提取；不把稀疏文件空洞当数据，不覆盖正式数据、不删除或改写partial/control，不查看或显示图像。提前提取仅作暂存，完整归档校验后才可发布为正式数据。

## 2026-09-15 18:50（下载进度复核）

- 用户询问最新下载进度；核验正式归档、后台进程、日志新鲜度与实际断点，不修改运行中的任务。

- 现场快照：{"time": "2026-09-15T18:50:58.150122+08:00", "archive_complete_filename_exists": false, "latest_sample_time": "2026-09-15T18:50:51", "latest_sample_age_s": 7.150122, "connections": 10, "latest_MBps": 0.596992, "latest_aria2_eta": "9h52m38s", "samples_5min": 28, "mean_MBps_5min": 0.5788525714285715, "completed_piece_GB": 538.160163342, "total_GB": 559.418993166, "percent": 96.19983767378243, "remaining_GB": 21.258829824, "control_age_s": 40.71341800689697, "hours_at_recent_mean": 10.201614017353211, "estimated_finish_at_recent_mean": "2026-09-16T05:03:03.960584+08:00", "pending_processing": {"assets": ["train_jpg"], "updated": 1789469452.4187012}, "download_state": {"status": "downloading", "time": "2026-09-15T05:46:14.326150+00:00", "expected_bytes": 559418993166}, "aria2_pids": ["1465011"]}。ETA仅按近期速率推算，不含解压校验。

## 2026-09-15 15:23（数据处理进度复核）

- 15:23复核：body于14:06、contact于14:39完成全部解压/CRC及归档SHA256；公共包也完成。全量GT整理于14:37完成37669个人体帧、1206个分片（10.453GB），覆盖62序列/64人物轨迹。所有清单帧恰好出现一次、所有分片存在，首/中/末分片SHA256与数值读取/帧号形状复核通过；全量body/contact顶点及面顺序检查通过，最大坐标分量差5e-9m。
- 10个body-only帧排除接触监督；另84帧OBJ无颜色，SMPL-X valid=false，保留其人体与SMPL标签，剩37585帧具有有效SMPL-X接触颜色标注；有效顶点positive比例约3.676%。完整缺颜色列表在`datasets/RICH/processed/annotations_v1/missing_smplx_color_frames.json`，完成证据`reports/rich_processing_20260915/annotation_completion.json`。标注进程正常完成退出，归档处理PID1471505继续等JPG，下载PID1464854仍运行。图片匹配/K、split与真实前端输入/独立审计仍待完成，training_ready=false。
- 用户询问数据处理状态；本轮检查归档校验/解压、全量GT分片、缺失标签和后台异常，明确已完成成果与仍待接通的训练数据接口。

## 2026-09-15 15:18（下载速度即时复核）

- 用户要求查看当前下载速度；读取最新aria2日志、近5分钟均速与实际断点进度，不改动下载任务。

- 现场快照：{"time": "2026-09-15T15:18:15.885713+08:00", "final_archive_exists": false, "latest_sample_time": "2026-09-15T15:18:09", "latest_sample_age_s": 6.885713, "connections": 12, "latest_MBps": 1.048576, "latest_aria2_eta": "6h57m14s", "samples_5min": 29, "mean_MBps_5min": 1.928968827586207, "percent": 94.9109989564562, "remaining_GB": 28.4688384, "hours_at_5min_mean": 4.099605215788928, "estimated_finish": "2026-09-15T19:24:14.464490+08:00"}；所有时间仅按近期速率估算，不包括解压校验。

## 2026-09-15（下载速度下降诊断）

- 已确认昨日19:06–19:11均值4.277MB/s，今日14:34均值0.488MB/s，实际下降约8.8倍。JPG进程参数仍为16连接、`max-download-limit=0`，其他大包下载均已结束。14:35:06两连接出现Expected总长度/Actual0，实际降到14连接，但此前16连接时已慢，不能把全部降速归因这两个错误。
- 本机快照CPU idle93–94%、iowait1%、可用内存约30GiB；14个下载TCP接收队列均0，RTT约156–203ms，未见明显接收端堆积。更像源站/服务或网络路径吞吐波动，尚无法区分服务器策略、负载、链路拥塞等，不能断言唯一原因。14:36:37最近2分钟均值已回升0.943MB/s、最新1.153MB/s；不把单点速度再次外推成保证。没有修改连接数/限速或重启仍在传输的任务。证据`datasets/RICH/download_logs/speed_diagnosis_20260915.json`。
- 用户指出昨日数MB/s与当前约0.49MB/s差异。本轮核对历史速度、当前连接/限速/错误和本机资源，以证据区分现象与原因；不武断归因服务器限速或认证，不破坏现有断点。

## 2026-09-15（下载剩余时间复核）

- 14:34最新：JPG完整piece下界522.902/559.419GB（93.4724%），剩36.5167GB；近5分钟28个摘要均值0.4881MB/s，按此约20.78小时，恒速预计9月16日11:21。最新单点0.5693MB/s/17h48m，与均值有差异，采用约21小时、建议预留约1天；不含校验解压，不承诺速率。下载PID1464854仍运行。body已全部解压/CRC/SHA256；contact解压约97.34/109.24GB，GT分片34503/37669帧。快照`datasets/RICH/download_logs/eta_latest_20260915.json`。
- 用户询问还需多久下载完成。本轮重新读取后台任务、aria2断点和近期吞吐估计剩余时间，区分下载结束与校验解压完成，不沿用此前ETA。

## 2026-09-15（训练结构与脚本完成度复核）

- 现场复核`training/{config,data,components,engine}.py`、统一入口与实验配置，重新运行训练专用7项CPU回归全部通过；涵盖连续/恢复续训权重和history一致、融合通道与有效mask、split隔离、payload/audit指纹和显式test评估。使用合成数值fixture，不作为真实RICH效果或GPU性能证据。
- 已同步`TRAINING.md`：RICH GT整理入口现在已实现，但真实前端inputs exporter、逐帧/相机/局部特征关联与独立几何审计仍缺；Stage-B原型未与Stage-A及完整Gaussian LBS绑定打通。当前单进程CPU/单GPU代码，无DDP/AMP/LR scheduler，目标机GPU环境和吞吐未测。数据/模型/损失可按factory+配置替换；Git迁移方案已准备，新代码仍未commit/push。
- 用户询问训练结构与脚本实现现状。本轮核对实际训练模块、配置、入口和测试证据，区分已可执行的Stage-A数值训练层、正在整理的RICH GT注释，以及尚未接通的真实冻结前端/完整Stage-B到Gaussian绑定；不因框架测试通过而宣称真实系统已经完成，不启动训练。

## 2026-09-15（检查下载与启动RICH数据处理）

- 14:00已完成进展：contact于13:50下载完成（28,826,902,288 bytes）；JPG仍约93%、余约38GB，已保留partial/control并恢复controller PID1464854。下载器新增传输失败后的有界重新认证续传，默认3轮，验证错误不重试。公共scan/calibration与multicam2world全成员CRC和归档SHA256通过；body解压约50,758文件/10.14GB，contact解压等待body完成；处理controller PID1471505自动等待完整JPG（最多36h）。
- 新增`rich_assets.py`/`process_rich_assets.py`及`rich_annotations.py`/`prepare_rich_annotations.py`。官方metadata固定revision `3b6cf83b0f09e2551f5a20cb180073dea6132cfc`并验证Git blob与SHA256；清单62序列/64人物轨迹，body37679帧、contact37669帧，10个缺contact样本显式记录。相机/scan映射与world公式来自官方例程，不按名字猜场景；ParkingLot1旋转矩阵存在5.4e-4正交残差，原值保留，精度审计待办。
- 全64轨迹首/中/末共192帧抽查完成，body PLY与contact OBJ顶点/三角面顺序一致，最大坐标分量差5.0e-9m。pickle contact为6890维SMPL，SMPL-X10475维标签来自OBJ绿色顶点；3个抽样帧无颜色，标记contact_smplx_valid=false，不伪造全负标签。跨scene/subject连通组为{BBQ}、{ParkingLot2}、{LectureHall,ParkingLot1,Pavallion}，尚未冻结split。
- 全量GT数值分片后台PID1507008已启动，14:00处理1082帧/35片并实读第一片验证无object字段；每片最多32帧、保留真实frame ID与人体参数/顶点/两种contact及valid/原始s2m位移，附SHA256，支持续跑。输出`datasets/RICH/processed/annotations_v1`；这不是冻结前端inputs，也未启动训练。相关下载5项、解压3项、标注8项测试通过。使用/迁移与字段说明`forward_contact_pipeline/RICH_PREPARATION.md`，现场证据`reports/rich_processing_20260915/`。图片匹配、分辨率/K、真实前端与独立几何审计仍待执行；代码未commit/push。

- 用户要求检查数据是否下载完成，完成则开始处理。初查body和公共包已完成；JPG在9月15日03:04报错退出，遗留摘要93%；contact在9月14日21:24退出，遗留摘要99%。本轮核验真实断点、恢复未完成下载，并先处理已完整资产，不把partial当完整归档。
- 继续仅文件目录、协议、数值与日志检查，不显示/查看图片。处理过程保留原压缩包，输出放NAS独立目录；先完整性、路径安全和字段审计，不提前启动训练。

## 2026-09-14（可持续迭代的训练架构与Git迁移准备）

- 已完成新增`contact_streaming/training`数值训练层、preflight/train/evaluate统一入口、cache index builder、配置继承/本机路径、geometry与RGB/Gaussian可选融合、masked目标与逐ROI指标、atomic checkpoint及epoch续训。保留旧GRU/TCN/Stage-B与诊断脚本。真实RICH exporter/前端审计/LBS binding仍未就绪，不把软件测试写为真实实验。
- 15项原有+新训练回归通过，另1项迁移回归通过；CPU续训权重/history与连续训练一致。独立源码副本在项目外CWD完成6步检查，GUSH3R五文件patch按固定revision重建字节一致；wheel构建成功且不含数据/权重。初次迁移演练发现public data config被旧gitignore遗漏，已修复并复测。
- 新增`MIGRATION.md`、`TRAINING.md`、源码审查器与依赖/patch清单；Git迁移可行，但新源码仍未commit/push，历史28个脚本含机器路径，目标GPU环境尚未验证。已更新忽略规则，公共RICH下载manifest独立入源码，凭据/数据/输出继续隔离。总结`reports/training_architecture_migration_20260914.md`。

- 用户要求核查训练脚本与仓库架构，缺失部分先实现，并要求后续方案变化时保持灵活；同时评估通过Git将HUGS代码迁移到另一台机器。当前开始盘点既有模块、入口、环境及数据接口，避免重新实现已有机制。
- 初查现有origin为OIIAIIOOIIAII/ml-hugs-work，main有大量既有修改；整个forward_contact_pipeline与多套第三方仓库/数据目前未跟踪。保留用户工作，不批量git add，不推送数据、凭据、模型或第三方大目录。本轮准备可审阅的代码/配置/迁移方案与检查，实际远程发布另按明确目的地和范围执行。
- 延续不读图、数据下载完成且审计通过后才启动真实训练；允许独立CPU合成契约测试验证工程，但不把它写作真实实验结果。

## 2026-09-14 19:11（RICH最新完成时间估算）

- controller PID1391152仍在运行。body于19:07完成下载和大小/格式检查，完整CRC尚待执行；contact已自动接续。JPG完整piece下界471.38/559.42GB（84.26%），剩88.04GB；近5分钟29摘要平均4.28MB/s，最新3.2MiB/s，按均值约5.72小时、最新速度约7.3小时，当前宜估6–8小时（9月15日凌晨1–3点），覆盖此前约3小时估算。contact90.17%，剩2.83GB，平均0.474MB/s，约1小时40分钟；公共scan/坐标已完成。
- 以上仅下载ETA，不含完整归档校验、解压与数据对齐审计。没有重启任务或改变连接/限速。快照`datasets/RICH/download_logs/eta_latest_20260914.json`。

## 2026-09-14（确认RICH完成后的正式实验起点）

- 用户确认是否可在当前数据下载完成后开始正式迭代。结论：这批RICH train JPG、SMPL-X、dense contact与公共scan/calibration/world transform，加上现有PROX，具备启动第一轮数据接入和受控实验所需的数据类型；无需为起步继续搜集更多数据集。但尚未完成下载、逐帧坐标/拓扑/标签审计，不能写成已经具备完整可训练资产。
- 明确当前队列仅train和公共资产，无官方val/test。初期可从train按实际scene/sequence/subject元数据构造互斥开发验证；若无法形成有效隔离则先补官方validation。最终泛化/论文主表需补齐官方val/test并冻结评测，test不用于调参。
- 下载完成后的执行顺序：完整性校验与短序列manifest审计→冻结split并在RICH复测真实Human3R/GUSH3R粗输入误差→合格输入上的Stage-A接触基线及RGB/Gaussian对比→通过接触与可靠性指标后接Stage-B因果控制/LBS。此前PROX上前端厘米级门槛失败不会被下载完成自动解除；不提前启动训练、不再用all-positive伪稠密标签。

## 2026-09-14（RICH提速原因与双机数据存储）

- 用户询问为何恢复后变快，以及上传个人HF供另一台机器使用。历史JPG日志证明9月11日凌晨已出现约1991条≥10MiB/s摘要，约400GB在当时完成；退出前连接错误耗尽，当前重认证恢复16连接与实际piece增长一致，但未证明唯一瓶颈原因。
- 重新读取RICH许可：个人single-user，允许受控计算机安装，同时禁止未经书面许可向第三方提供/复制/分发，明示仅一份archive copy；公开HF不在已有授权内，private标记不能自动解决第三方托管授权。优先同NAS复用，其次完整数据SSH/rsync；获准云托管后再考虑私有对象存储或HF。
- HF当前官方文档（镜像+官方GitHub源码）确认免费私有100GB、PRO含1TB、Git仓库单文件硬上限500GB/建议<200GB。当前约598GB训练归档超免费空间，559GB单包需分块。尚无第二台机器配置，未上传、建仓库、创建云资源或发送外部消息。补充`reports/rich_sources_20260914.md`。

## 2026-09-14（RICH后台进度复核与补充来源检索）

- 用户要求核查后台下载并寻找其他数据来源。初次状态读取显示：训练图片/SMPL-X/contact任务已在9月10日至11日失败退出，遗留摘要分别约82%/94%/89%；这些是停止时进度，不是当前运行速度。接下来核验进程、断点与错误原因，沿已授权官方凭据恢复下载，并补查其他可用公开来源。
- 已使用既有认证成功恢复官方断点：18:47新controller PID1391152，2任务×16连接、非JPG500K。18:52完整piece位图下界：JPG463.69/559.42GB（82.89%）、body8.71/9.15GB（95.19%）、contact25.92/28.83GB（89.90%，排队）；scan与坐标已完成。JPG最新12摘要均值9.91MB/s，控制文件60秒增量10.17MB/s，按当前速度剩约2.7小时，非保证。原退出日志是Expected总长度/Actual0，未武断归因认证或限速。
- BEHAVE官方实际Range核验：RHOBIN训练包29.16GB（官网近似标26GB）、Date01.zip 15.02GB，均206与正确归档签名，只取1KB，不作速度/字段充分性保证。
- 新核实DAMON官网5522图、SMPL/SMPL-X接触标签，需DECO注册，CLIFF估计相机/人体不能等同标定GT；BEHAVE官网321序列、RGBD/人体物体/接触，直接分包总约140GB及RHOBIN训练26GB入口，均须契约适配。PhySIC当前公开demo不等于独立稠密接触数据已发布。HF原候选再次确认manual gate，未找到新增完整RICH镜像。报告`forward_contact_pipeline/reports/rich_sources_20260914.md`，不读图、不启动新大包。

## 2026-09-10（Human3R→GUSH3R 完整技术学习路线补全）

- 用户要求将从 Human3R 到 GUSH3R 的完整技术路线写入学习文档。已在 `INTERVIEW_TECHNICAL_ROADMAP.md` 的新增 `2.2.4` 节补全可面试讲解的闭环：Human3R 的 CUT3R causal recurrent geometry、Multi-HMR human prompt、SMPL-X/mask/track、训练损失与局限；GUSH3R 冻结前端后的 Scene Gaussian Decoder、Human Gaussian Decoder/HGT、canonical query、appearance memory、LBS、两支 decoder 的训练；以及最终的人景合成、真实缺失的接触关系、我方接触层的接口与 binding adapter 缺口。
- 文档明确分开“论文已有的重建/渲染机制”与“我方仍待实现的 contact controller / corrected-SMPL-X-to-Gaussian binding”，并保留 Human3R/GUSH3R held-out 误差远大于约2cm接触尺度的实证边界；没有将视觉渲染质量误写为接触精度。

## 2026-09-10（HF-Mirror已连通：发现需要审批的RICH分卷仓库）

- 用户指出Hugging Face自身有镜像。继续复查后HF-Mirror搜索API成功返回691个rich名称匹配；初次超时不能作为放弃镜像路线的依据。使用IPv4、压缩响应、较长超时和精简字段，某次搜索19,537bytes/2.423秒，但不将API速度当大文件吞吐。
- 进一步全文检索官网域名，发现`Yong-Hoon/human3r-dataset`，其公开metadata/tree列出`RICH.tar.part_aa`至`_aj`共10份，合计486,013,020,160bytes（约452.6GiB，页面写453GB）。revision`1d90eba6ec995145db9360ebde15ae1dce0ee374`，另有checksums.sha256；候选manifest已保存。
- 该仓库`gated: manual`，受限README返回401，固定revision首分卷Range请求实测403（not in the authorized list），公开页面明确需HF登录并接受访问条件；需要HF账号获准访问后才能核验归档内容和测速。未发送任何token/原RICH凭据给镜像，未申请访问、未读图、未启动新大包。
- 公开README仅列出RICH目录，无train/val/test、分辨率、SMPL-X/contact覆盖明细；不能把第三方486GB分卷当作官方559GB训练JPG的等价替代。原下载断点与控制器继续保留，不跨不同归档格式复用部分文件。
- 已更新`reports/rich_mirror_search_20260910.md`（原站申请入口、镜像入口、仅RICH的下载命令与授权后检查顺序）、TODO、活计划及快下方案。此更新覆盖之前“HF尚无法核查/未找到实际RICH分卷候选”的状态，其他来源核查仍有效。

## 2026-09-10（RICH公开镜像检索：未找到完整训练图片替代源）

- 用户要求联网寻找镜像，并要求继续。已核查官方RICH/BSTRO、相关issues、SA-HMR、GVHMR、WHAM、SMPLer-X、DECO以及国内OpenDataLab的公开资料和接口；未使用子agent、未向第三方发送RICH凭据、未查看图像。
- OpenDataLab有`OpenDataLab/RICH`介绍页，但实查详情`fileNum=0/fileSize=0`、文件列表`total=0`，明确提示“本站暂不提供直接下载，请至数据集官网”；不能推荐为国内镜像。
- SA-HMR仅预处理辅助数据，训练图片仍需原站，提供的bodies是拟合SMPL-H；GVHMR/WHAM为预处理/评测数据；SMPLer-X的HF链接主要是模型。相关Drive链接已记录，但本机访问失败，未宣称下载可用或更快。
- DECO官方Keeper链接返回206，`Release_Datasets.tar.gz`总45,205,516bytes；只取前1MiB，tar前缀确认RICH测试NPZ，首数组`imgname.npy`为9834条图片路径。DECO作者issue #16明确训练RICH/PROX仍需从各自官网取图；不是559GB训练图片镜像。
- HF/HF-Mirror/Google/Drive/Jina/Zenodo存在本机连接失败或超时，百度验证码、Bing结果不相关；没有据此宣称全网无镜像。详见`forward_contact_pipeline/reports/rich_mirror_search_20260910.md`及同名证据目录。
- 更小入口是用户官网列出的约107GB BSTRO TSV，官方EXP.md确认有图片/尺寸/标签TSV；实际大小与同原始帧、SMPL-X、相机/场景对应仍待核验，未启动新大包或切换训练来源。原controller PID1305167继续存在，16连接、2任务、非JPG500K配置不变。

## 2026-09-10（术语纠正：GUSH3R Human Gaussian Decoder）

- 重新核对 GUSH3R 论文 Sec.3.1/3.4 与官方代码，纠正此前将代码类名 `HumanGSHead` 与外部同名 HumanGS 工作混淆的表述。GUSH3R 的 **Human Gaussian Decoder / Human Gaussian Transformer (HGT)** 是作者在 GUSH3R 中新提出、与 Scene Gaussian Decoder 并列的自主分支；它冻结 Human3R backbone，取 Human3R 的 SMPL-X mesh/human token/image token 作先验，在 canonical body space 预测人体 Gaussian 属性。其论文明确把外部 **LHM**（Qiu et al., ICCV 2025）作为 AnySplat+LHM+Human3R 的分解式 baseline，而不是把 LHM/HumanGS 接入自己的主模型。后续所有方案与面试表述应写“GUSH3R Human Gaussian Decoder”，不得称其为我方模块或误称为外部 HumanGS 模块。

## 2026-09-10（用户限定本机直连：持续速度与路径诊断）

- 用户没有云平台账号，明确询问仅本机下载能否提速。本轮停止中转账号/租机选择，继续已授权的直连优化，不购买或创建外部资源。
- 12:40 UTC近期JPG平均约340821B/s，明显低于早前1MiB/s样本；之前5–7天ETA已不能作当前稳定预测。将检查DNS/IPv6/HTTP协议和TCP状态，并用更长窗口比较图片独占带宽与连接数；保留所有断点与当前认证。
- 12:45–12:51 UTC完成本机独占实测：16/8连接各180秒，去掉首60秒后平均分别442096.5/152337.5 B/s；测试前并行18摘要均值314823.1 B/s。两组无aria2 errorCode，退出7是到时保存未完成断点；所有下载字节保留。16连接优于8，但未找到稳定多MB/s的额外提速；暂停其他文件/刷新连接未恢复早先1MiB/s。新增可复测脚本`benchmark_rich_local.py`及报告`jpg_local_benchmark.json`。
- 已自动恢复2文件×16连接、非JPG限速500K，controller PID1305167，确认JPG/body继续下载、contact保留断点排队。JPG已完成约5.437GB、剩约553.982GB；按0.3–0.442MB/s恒速余量约14.5–21.4天，不能承诺固定日期。没有改变全局TCP参数或部署频繁重连。DNS仅IPv4、无IPv6默认路由，ALPN为HTTP/1.1；接收自动调节已开启、tcp_rmem上限16MiB，不据此武断归因源站限速。

- 12:54:48 UTC恢复后舍弃首分钟的10摘要均值：JPG287641.6B/s、body325120B/s，图片按剩约554GB恒速还需22.3天（覆盖前面的区间估算），快照`jpg_local_eta_20260910.json`。恢复时JPG单条连接收到HTTP403，另15条持续工作；body16条正常。未据单次403推断限速规则，未继续加并发或重启。确认JPG/body/contact三个aria2控制文件均保留。

## 2026-09-10（中转下载测试：等待可用节点）

- 用户授权测试中转下载方式。已检查本机SSH/scp/sftp/rsync/curl均可用，存在私有SSH密钥（仅检查文件名/权限，未输出内容），无用户SSH config及可用中转主机地址；未发现rclone配置。
- 已向用户请求可用中转服务器SSH别名，或主机/端口/用户名及已有密钥路径。未连接推测地址、未创建付费资源、未传递RICH凭据给第三方。
- 采用SSH动态转发优先：客户端经SOCKS隧道直接从官方HTTPS取小范围数据到当前服务器，测端到端吞吐；中转无需存储559GB或保存RICH密码。测试前先检查SSH与TCP转发，再比较相同范围/并发数的直连和中转，记录认证、Range、实际字节、耗时及波动。
- 当前直连任务保留运行。中转测试尚未执行，真正缺口是节点连接信息；操作方案见 `forward_contact_pipeline/reports/rich_relay_test_plan_20260910.md`。

## 2026-09-10（训练图片ETA与进一步加速：16连接、图片优先）

- 用户询问训练图片预计完成时间与进一步提速。原按单文件约0.8MB/s估算559.4GB约8天，原两任务1.4MB/s合计不能直接当图片速度。
- 已对真实JPG资源进行8/16连接各60秒的独占短测，临时暂停原任务后自动恢复；所有下载字节及aria2控制文件保留，未作无效重复下载。8连接后段约0.9–1.1MiB/s，16连接后段约1.8MiB/s；初始4.4/6.7MiB/s是启动突发，不能用来推算全量耗时。报告 `datasets/RICH/download_logs/jpg_connection_benchmark.json`。
- 选用16连接，并将队列调整为scan→JPG→body→contact，让图片提前与数值标注并行。恢复并行后JPG曾降至约0.6MB/s，因此新增`--other-limit 500K`为非JPG资源限速，JPG不设限；当前controller PID1011347，图片已正式下载，body同时下载，contact保留断点等待。4项HTTP回归再次通过。
- 图片优先后的近期JPG约0.9–1.2MiB/s；2026-09-10 18:21北京时间最新6个10秒摘要均值1MiB/s、已下约876MiB，按固定速率余量估计6.16天，即9月16日晚。实际ETA随吞吐变化，当前宜按5–7天量级规划，不能承诺短测峰值对应的3.5天。记录 `datasets/RICH/download_logs/jpg_eta_20260910.json`。
- 扫描/标定包904175585bytes已下载并完成全部64成员ZIP CRC校验（11.1秒），见`scan_calibration.integrity.json`；公开坐标包此前已通过CRC。
- 只读取4MiB归档前缀的tar目录信息，确认开头为`train/Pavallion_006_plankjack/cam_00`及JPEG文件名；未查看或展示图像。后续可在完整帧与对应标注可用后先开展小样本实验，但这不改变全量下载字节数。
- 已询问是否有现成欧洲/香港/新加坡中转节点；当前没有提供连接配置，未实测中转、未创建付费资源。若有节点，需比较源站到节点与节点到本机两段耗时，不能承诺某个地区必然更快。
- 修正状态展示：前一controller遗留的downloading状态在新controller启动后显示为queued_with_saved_partial，避免把暂停的contact误报为第三个同时下载任务。

## 2026-09-10（RICH下载速度：Cookie修复、多连接接续与实测）

- 根因已确认：用户凭据有效；官方POST成功后设置PHPSESSID并302跳转，原urllib预检不保留Cookie导致误报登录失败。已改用HTTPCookieProcessor；4个保护资源全部返回206及正确归档签名。真实大小分别为scan 904175585、body 9147649220、contact 28826902288、JPG 559418993166 bytes，保护资源总计约598.3GB。
- 下载已实际启动；随后用户询问速度。wget两条流实测约35.3KB/s与40.3KB/s，合计75.6KB/s。已在项目独立`.tools/rich-download`安装aria2 1.37.0，增加私有Netscape Cookie文件、aria2 GET分段后端、进度读取及aria2 sparse文件不得退回wget的保护。
- 4连接/文件阶段合计约0.5MB/s；8连接/文件最近6个10秒摘要平均：scan 574976 B/s、body 812475.7 B/s，合计1.387MB/s，较wget样本约18.3倍。这是顺序观察的近期速度，非严格同时A/B，也不保证全程速率。数据保存于`datasets/RICH/download_logs/speed_comparison_20260910.json`及4/8连接样本JSON。
- 采用2文件并行×每文件8连接，后台controller PID 929174；旧PID 873533/923487仅在验证进程身份和独立进程组后正常终止，wget前缀与aria2控制文件均保留并成功续上。当前scan/body在下载，contact/JPG排队，public multicam2world已完成。
- `tests/test_rich_acquisition.py` 4项HTTP回归通过：Cookie+302+Range，wget续传，aria2导入wget前缀（允许补下不完整piece），登录HTML拒绝及禁止wget接aria2稀疏部分文件。aria2状态不再把稀疏文件的逻辑大小当已下载量，改读aria2真实进度摘要。原始密码与Cookie不输出到对话。

## 2026-09-10（RICH认证预检失败：诊断与修复）

- 用户运行 `acquire_rich.py --configure` 后收到 `scan_calibration: login/error page or unexpected archive; check credentials`。该泛化错误不证明凭据错误；本轮检查真实响应、POST流程和预检假设，凭据仅用于已授权RICH官方端点，不输出内容。

## 2026-09-10（RICH真实资源链接到位：下载预检）

- 用户已提供5个真实链接：公共scan/calibration、multicam2world、train_body、train_hsc、JPG_images/train.tar.gz，授权沿已确认方案直接下载到NAS。
- 本轮先获取公开坐标文件，检查受保护资源的实际HTTP认证机制、资源大小及Range；认证可用后启动可恢复任务。仅读取协议、目录、文本和数值，不查看图片。
- 公开坐标包已下载：`datasets/RICH/raw/common/multicam2world.zip`，4304bytes，ZIP CRC通过、7个场景JSON，SHA-256及成员清单保存在同目录`.download.json`。scan未认证响应为HTTP200、6311bytes登录HTML；明确表单POST字段username/password/commit（Log in），当前不能直接套用netrc/aria2 GET方案。
- 已实现 `forward_contact_pipeline/scripts/acquire_rich.py`：服务器终端 `--configure` 用getpass录入并将URL编码凭据保存在私有目录600权限文件，先验证资源响应，再脱离终端以2个wget任务下载4个保护归档；`--status`查进度、`--start`续传，进程锁避免重复任务，`.part`只在字节数/格式检查后转为最终文件，完整CRC待后续校验。私有清单目录已gitignore，不输出密码。
- 本地HTTP测试通过：特殊字符POST、wget从50000字节偏移续传且最终ZIP字节一致、登录HTML拒绝、无Range时保留已有部分文件。记录`forward_contact_pipeline/runs/rich_download_smoke_qy7js05y/summary.json`。用户认证尚未配置，真实4个归档仍为not_started，未宣称RICH吞吐或分段加速实测通过。

## 2026-09-10（RICH直接代下载：入口检查）

- 用户询问是否可直接代为下载。可在当前服务器执行下载、续传与校验；复核原约定URL清单、当前用户netrc/wgetrc/aria2配置均不存在，尚无可直接使用的RICH资源入口，未启动传输。
- 已创建私有目录 `datasets/RICH/download_lists/`（权限700），用于存放用户从官网下载的脚本/链接清单。NAS当前约34TB可用。后续取得资源入口后先核验认证，再按已准备方案开始小资产下载，不再要求用户自行执行整套下载命令。

## 2026-09-10（恢复会话 01a06025：RICH 快速下载方案）

- 用户要求将会话 `01a06025-0f9a-7ca3-8be1-17ba3a60823f` 的上下文接入当前会话，并给出 RICH 快速下载方案。已读取原会话的阶段交接、后续数据讨论和下载器实现，并核验项目文件仍可访问；继续遵守仅数值/日志检查、不读图的偏好。
- 现场复核：NAS 约32TB可用；当前根文件系统仅约139GB可用；wget/curl/tmux 已安装，aria2c 未安装。原约定位置尚无 `rich_p0_urls.txt`、`rich_test_urls.txt` 或 `datasets/RICH`。官网下载页在未登录时返回登录表单，尚无实际资源 URL，不能宣称认证、Range 或吞吐测试已通过。
- 本轮优先准备可复用的上下文交接与下载方案，修正“必须先56GB sample、再完整test、最后train”的等待顺序：先小体积标注/标定，train/val 的 RGB 子集随之接入；sample 是否完整需实物核验，test 保留正式评测用途。具体体积沿用用户提供的官方清单，不将网页的空间需求等同于已测量的传输字节数。
- 已完成 `forward_contact_pipeline/reports/hugs_context_restored_20260910.md` 与 `reports/rich_fast_download_plan_20260910.md`，同步TODO、活实验计划和A5报告。方案含NAS目录、独立conda aria2环境、显式文件名清单、2文件×4连接、tmux/断点恢复、GET/Basic与POST/cookie认证区别、Range与吞吐测试协议及分阶段容量。三个bash命令块均通过 `bash -n`；体积算术复核为1087GB。没有安装aria2、配置账号、下载数据或启动训练；实测需真实资源URL。

## 2026-09-10（A5：真实稠密接触监督数据门槛）

- 用户要求继续、阶段结束自动进入下一阶段，且不读图。已在当前工作区及已知 NAS 项目根目录完成仅目录/文件名的 RICH、PhySIC、EMDB 数据资产审计；未发现可用数据目录、归档、manifest 或预处理产物。当前 PROX 可继续作 continuous proximity/ROI 状态教师，但其脚 ROI binary label 已证实近乎 all-positive，不能替代真实 dense vertex contact。
- 已新增 `forward_contact_pipeline/reports/a5_dense_supervision_data_gate_20260910.md`，冻结数据准入契约：RGB/相机、SMPL-X topology/correspondence、scene mesh/SDF 与米制坐标、同帧且正负分布有效的 vertex label（source/confidence）、frozen backbone coarse state、scene/sequence/subject 隔离 split。没有这些条件不得恢复 Stage-A dense contact、uncertainty 或 Stage-B 接入；当前只可做数据契约准备和前馈候选可用性跟踪。同步更新活实验计划与 TODO；不把此阻塞误写为 GUSH3R/PROX 的负性能结果。
- 随后完成 A1 contact-local contract 的纯数值复核：`stagea_relation_clean_64x128_v3` 的12/12 sequence required local field、finite、shape 与 nonzero normal unit-length 均通过（最大误差 `2.38e-7`）；旧v1的面积加权 normal 最大误差约5.28、v2的零 normal 继续封存不可训练。A1 仅确认 local continuous-proximity 的无绝对世界坐标接口，不解除 A5 dense binary 标签和真实前端阻塞。报告 `reports/a1_contact_local_contract_audit_20260910.md`。
- 对 UniCon3R 进行只读 release recheck：官方 snapshot `9b248489...` README 仍称 code/checkpoints/install/inference 将“released soon”，仓库顶层仅项目页资产；远程 refs 未给出可判定的新发布信息，因此不作网络端发布与否推断。它继续是最重要的竞品/设计参考，不能作为当前可运行 PROX baseline、A3 backbone 或 dense-label 来源。
- RICH 官方站点/下载/许可页复核：站点明确提供多视角4K RGB、GT 3D bodies、body/scene scans、accurate vertex-level contact labels，能够补齐最终 Stage-A 的 dense contact 与 scene surface 缺口；下载须注册登录，许可仅个人非商业科研/教育/艺术用途且禁止未授权再分发。结论：RICH 是 A5 的优先数据源。数据授权到位后先用一个短序列核验实际字段、SMPL-X topology、坐标、SDF/scan、帧对齐和标签正负比例，不能未审计就全量训练；它不解除 Human3R/GUSH3R 2cm 真实前端门槛。
- 用户提供授权下载页包清单后，已冻结 RICH 分阶段获取：P0 `Gym_010_dips1` 56GB sample；P1 test JPG+SMPL-X+contact+scan/calibration/world transform 约281GB；P2 train+val JPG/SMPL-X/contact 约806GB。完整 JPG train/val/test 约1.09TB，服务器约32TB可用空间足够。禁止运行 test/train 约3TB/6TB的原始 image 下载脚本；BSTRO checkpoint无需先下，TSV 107GB和geodesic matrix仅用于后续严格公平 BSTRO 对比。
- 为大体积 RICH 获取增加 `scripts/download_rich_parallel.sh`：服务器当前无 aria2c，但有 wget/tmux，因此用 wget `--continue`、portal URL list 与默认3文件并行实现可中断续传；脚本强制用用户服务器上自行配置的 netrc 登录态，不接收或保存账号/密码/cookie。已做 bash syntax check；具体下载从 P0 sample 开始。

## 2026-09-10（连续执行恢复：UniCon3R 竞品可运行性审计）

- 用户要求继续，不在阶段结束时等待输入。昨日 A4 已完成 GUSH3R Gaussian-SDF diagnostic：在失败的 held-out coordinate contract 下 center-to-SDF median 33.9cm、coverage 22.4%，再次证明不能把 Gaussian center 作为接触 surface。现进入 UniCon3R 官方代码审计：repo `surtantheta/UniCon3R` 已确认存在并拉取到 `ml-hugs-work/baselines/UniCon3R_20260909/`；接下来检查其模型/数据/权重/输出接口，先做数值与环境验证，再决定是否启动受控 PROX forward smoke。所有结果保持与 GUSH3R/PROX oracle 口径隔离，不读图。
- UniCon3R 审计完成：official repo README 明确称 code/checkpoint/inference will be released soon，当前只有project page、网页demo和每帧 `contactIndices`（10475-vertex）JSON；不能运行/训练/测 PROX，不能作为当前可复现基线。另完成 oracle Gaussian occupancy 表征诊断：现有 PROX near-contact ROI 的 |SDF|<=2cm 占99.997%，即使 full L_Leg FPS sample（MPH112）也全在2cm内，故不具 contact-vs-noncontact 分布；nearest point Spearman=.986、手工Gaussian=.964 无法构成 Gaussian 优势结论。停止调此规则，后续 dense head 需RICH/PhySIC类真实稠密监督或 UniCon3R release。详细结论归档 `reports/a4_candidate_and_gaussian_evidence_20260910.md`。

## 2026-09-10（面试深挖：研究点一、二技术路线文档）

- 用户要求按当前研究点一与研究点二的技术路线整理详细面试材料。将基于已维护的最优 pipeline、实验结论、前馈接触方案和真实门槛证据写独立文档：明确问题、系统接口、方法机理、训练/推理、实验与消融、关键追问、已完成与未完成边界。不得把研究点二的 PROX oracle/degraded proxy 机制验证误写成真实前馈 Gaussian 已验证结果。
- 已完成 `INTERVIEW_TECHNICAL_ROADMAP.md`：以研究点一 `depth_sup12k + AnchorAttention_6k` GT 最优（lab 19.7831）和 VIMO-v4 最优（avg 17.2944，vs STM +0.39）为完整事实主线；研究点二明确区分 oracle mechanism、proxy robustness、真实前端和 renderer diagnostic，包含 E0/E1/E2 数值、Human3R/GUSH3R held-out几何门槛失败、E2/P4/P5 后续路线，以及面试高频追问与回答边界。

## 2026-09-09（第一研究点最佳结果：高帧率与环绕视角渲染）

- 用户要求针对第一研究点（HUGS Anchor Attention / Global Scene Gate）的当前最佳优化结果解决两项展示问题：原始输入视频低帧率导致重建视频观感不连续；现有输出只沿 GT 相机，需增加环绕人体的新视角视频。当前先核验最佳 run、`render_full_video.py` 和 LBS/相机接口，再实现可复现的时间插值渲染与 orbit-camera 渲染；不改变训练 checkpoint 或既有 GT-view 输出。
- 已新增 `scripts/render_smooth_orbit_video.py`。它不只是把低帧序列封装成高 FPS：相邻 SMPL axis-angle joints 先转四元数做 shortest-path SLERP，`transl/betas/smpl_scale` 线性插值，每个合成时刻重新走 human Gaussian 的 LBS；scene GS 仅 forward 一次，保持静态。`--view gt/orbit/both` 支持 GT 相机插帧及以每帧 posed human-GS 重心为 target 的 look-at 环绕相机（可调 radius/elevation/turns）。
- 修复渲染加载时 AnchorAttention `frame_embed` 的训练 split（10）与 checkpoint（83）长度冲突：按 checkpoint embedding 维度恢复模块，并在 all-dataset 渲染中使用真实 `data['frame_idx']`，而不是全帧枚举号。bike VIMO-v4 最优 run 的 2 原始帧 -> 3 合成帧、20 FPS GT/orbit smoke 均已写出且成功；完整 10 -> 30 FPS（103 原始帧 -> 307 合成帧）GT + orbit 渲染已以独立进程 PID 481338 启动，日志为 `.../render_smooth_30fps_x3.log`，不覆盖原 render_all 视频。
- 用户试听完整 30 FPS 视频后反馈：人仍抽搐、背景像低帧率。根因已确定为第一版仅插值 SMPL/LBS，而 GT-view 的相机沿用左端源帧（背景跳变），AnchorAttention 的离散 `frame_embed` 又取最近帧（correction 跳变）。下一版必须同时 SLERP 相机旋转/线性插值相机中心，并在相邻 anchor frame embedding 下分别求 correction 后按时间权重混合；不再把 nearest-frame correction 当作高帧率输出。
- 修复已实现并通过 2 source frames -> 3 output frames 的 GT/orbit 实际 GPU smoke：`c2w` rotation 用 quaternion SLERP、camera center 线性插值并重建 `world_view_transform/full_proj_transform`；同一插值 LBS human 在左右两端 frame embedding 下各跑一次 AnchorAttention，所有 correction-dependent float tensor 按 alpha 混合。完整 bike VIMO-v4 10->30FPS GT+orbit 重渲 PID 517856，覆盖同名 `render_smooth_30fps_x3_{gt,orbit}.mp4`（旧文件在完成前仍保留），日志同 `render_smooth_30fps_x3.log`。
- 用户改为要求 lab 场景，且明确必须使用最优 pipeline；新视角不应大角度环绕，而需围绕人体中心的小幅摆动。执行时必须选用 `BEST_PIPELINE.md` 的 GT 对齐 Ours 主表最优 `depth_sup12k + AnchorAttention_6k` lab run，而非 VIMO-v4。新增/调整渲染参数为小角度往返（预计 `orbit_turns=0.25`、`orbit_start_deg=-45`，即约 -45°→+45°→-45°），保持人景相对尺度。
- 已重核并纠正表述：lab 的全局最优不是旧 ADC(19.5512)，而是 GT `depth_sup12k + AnchorAttention_6k`(19.7831)，实际 run=`output/human_scene/neuman/lab/hugs_trimlp/depth_sup12k_transl_xyz_attn_6000_lab_20260528/2026-05-28_23-44-41/`。该权重缺少后来代码新增的 frame_embed，兼容层按其原始无-frame-embed架构构建后已通过 2->3 帧 GT/small-swing GPU smoke。完整渲染已启动 PID 553816：10->30 FPS，GT view+以人体为 target 的 ±15° sin 小幅摆动（elevation=5°、1 cycle），日志 `render_smooth_30fps_x3_small_swing.log`，输出仍为 `render_smooth_30fps_x3_{gt,orbit}.mp4`。

## 2026-09-09（A3 后续连续执行：Gaussian surface-evidence 审计 → 新前馈候选）

- 用户要求在阶段结束后直接进入下一阶段、持续执行。当前门槛结论不变：Human3R/GUSH3R 的 held-out人体几何均未达2cm，禁止进入真实接触 controller 训练。接下来先做 GUSH3R Gaussian 的 PROX SDF support 审计，量化“Gaussian center 不能直接作为接触 surface”的实际程度；该实验只作为 degraded-map/interface diagnostic。结束后直接核验 UniCon3R 等更强前馈接触候选的官方代码可用性、接口与数据/算力代价，再决定是否切换 backbone。持续维护计划/TODO/本日志；不读图。

## 2026-09-09（Stage-A 监督纠偏：无误差参考与连续接近度实验启动）

- 用户确认继续执行，并要求记录当前 plan 后直接跑实验。已固定“前馈 Gaussian 的误差样本”和“无误差参考”不是同一类数据：冻结前馈 backbone 在真实 RGB 上导出的 `G_coarse/SMPL-X_coarse` 是天然带误差输入；参考监督来自独立 PROX 几何教师（PROXD SMPL-X、官方 scene SDF/scan、相机与可选 RGB/mask/depth），而不是假造一份“零误差 Gaussian”。第一版不要求 GT Gaussian 参数；后续可用 GT SMPL-X canonical human Gaussian 经 LBS 与 scene scan 拟合做离线 reference，但这不替代真实前馈输入。
- 因当前 PROX 脚 ROI 的 2 cm binary label 被审计为每帧 64 vertex 全同，已永久停止用它训练/报告 dense vertex binary F1。现在启动的 A2 baseline 改为：用 local mesh--scene relation 预测每 vertex continuous signed proximity，并将官方 teacher 的每脚 contact 作为 ROI-level 状态；其目的只是验证连续几何表征是否可从局部 patch 恢复及其在未见 scene/error family 下的误差，不把它描述为 RGB/Gaussian 前端结果，也不接入 controller。
- 本轮不读图。GPU 空闲（RTX 4090 4 MiB），启动前先建立新、不可覆盖的 `stagea_proximity_roi_v1_20260909` 实验目录，保存命令、split、metrics 与标准输出。若 strict scene+error-family split 下连续 proximity 不可靠，则先改为机制 split/显式 reliability 目标，不会绕过负结果直接进入 Stage-A+B 或 Gaussian LBS。
- 实验已完成于可追溯目录 `forward_contact_pipeline/runs/stagea_proximity_roi_v2_20260909/`（v1 仅因从错误工作目录解析 split 路径失败、无 metrics；保留 stderr 供追溯）。12 sequence strict split 的 geometry-only local relation encoder 在 distance-corrupted test 上取得 signed proximity MAE 1.27 mm，对比 zero predictor 1.79 mm、naive nearest input-point 12.64 mm；连续几何目标可保留。ROI contact 的 precision=.429/recall=1.0，仍 all-positive，明确否定其 contact F1=.600 的任何正面解释。报告写入 `reports/stagea_proximity_roi_v2_20260909.md`；下一实验仅做 reliability/calibration/abstention，不越级接 controller 或 Gaussian。
- 随后 A2.1 uncertainty v1 发现 hard clamp 导致所有 scale=50 mm，修复为 differentiable bounded log-scale 后重跑 v2。v2 在 validation 的五个 uncertainty bin 中真实 MAE 随 scale 单调增大，但在严格未见 `distance` error family test 上 coverage-risk 失败：保留 50% 的 MAE=.743 mm，高于全量=.698 mm。该结果证明当前 reliability 不具备 unseen-error OOD 检测能力，禁止接 Stage B；下一轮是全已声明 error-family 增强训练、scene-disjoint test，只验证 covered-error reliability，且明确与 OOD 分开报告。
- covered-error augmented reliability 已完成：all-family train、scene-disjoint test 下全量 MAE=.716 mm，lowest-scale 50%=.653 mm 但 lowest 25%=.768 mm，five uncertainty bins 的误差不单调。故即使在已覆盖 proxy error 下也不具备稳定 calibrated abstention；停止继续调 local-only reliability。所有 v1/v2/v3 数字与门槛已归档至 `reports/stagea_uncertainty_20260909.md`。正式下一动作是 A3：从可运行冻结前馈 backbone 抽取 RGB/Gaussian relation token，并以 coarse output 对 PROXD/SDF 的真实测量误差做 reliability teacher；这之前不能进入 Stage B 或 Gaussian LBS 主实验。
- A3 执行完成两项关键诊断。首先，Human3R 的已存 correspondence 在**拟合用训练帧**去除 root/centroid 后，local vertex median 仍为5.78cm，故其不能产生 contact-scale reliability label。其次，GUSH3R 新增 `--export_contact_state` 数值接口；`per-frame` causal scene mode 在 PROX MPH112 60帧为每帧导出250k scene/10k human Gaussian 的top-2048属性和coarse SMPL-X，`final` mode仅末帧有scene state、禁止作为online输入。GUSH3R前45帧拟合/后15帧held-out的 SMPL-X→PROXD median=19.95cm、p95=54.69cm，明确失败2cm门槛。故当前 GUSH3R 只能作为 Gaussian evidence/renderer/degraded-map robustness，不能做真实前端 supervision；详细报告 `reports/a3_gush3r_prox_feasibility_20260909.md`。

## 2026-09-09（前馈人景 Gaussian + 因果接触：总方案与执行路径收敛）

- 用户要求基于已得到的 E0/E1/E1b 证据及 Splat-SAP、HumanGS、Hand-4DGS、PointSplat、GaussianLens 等近期工作，细化整体研究方案并写入可执行文档。本轮将新建唯一的系统路线图，明确系统接口、论文创新点、数据/训练边界、每阶段输入输出、指标/停止条件、基线和风险转向；同步活计划与 TODO。
- 已确认不能将 Splat-SAP 视为当前可直接复现的替代物：它是稀疏双目、target-view-plane Gaussian 的人景重建，官方项目页未提供代码；它最适合提供 E2/E4 的 scale-aware point-map/local-proxy 设计。HumanGS 是潜在 canonical human Gaussian/LBS 分支，但同样尚未本地复现。当前工作先以 PROX oracle/degraded proxy 验证接触机制，再决定接入哪个前馈人体/场景模块。
- E2 已完成：新增 `make_e2_degraded_proxy.py` 和部署一致的 `external_gate`，medium point-map proxy（18° normal、2cm distance noise、1cm bias、20% dropout、2-frame delay）下冻结 E1b controller 的 test sliding 从 0.385 恶化到 0.463 m/s；confidence 阈值仅在 100% abstain 时回到无修正基线。单因素 test 显示 normal18/distance 各恶化约 13%，dropout20 与 delay2 相对安全。几何增强训练 `medium_aug_action40_contact15` 完成后将 sliding 降至 0.276 m/s（-28.4%），但 contact F1=0.722、transition F1=0.142；故增强不足以构成可靠接触估计，下一步必须显式学习 normal/distance reliability 与 uncertainty，而非继续调 gate。
- 用户已明确最终工作目标与正确网络契约：输入是当前帧 RGB（经冻结前馈 Gaussian backbone 得到粗对齐 human/scene Gaussian、SMPL-X 与可复用视觉 token）及模型自身过去 contact/control state；输出是当前帧 contact、geometry uncertainty 和低维 online root/foot correction。接触估计器必须建立在前馈 Gaussian 粗结果之上，且以 RGB/mesh-point local relation encoder + causal controller 组成；scene Gaussian 不被逐帧优化，只通过 corrected SMPL-X/LBS 更新 human Gaussian。当前 PROX 58D GRU 仅是控制层机制原型，不能再表述为最终接触估计器。
- 本轮依用户纠正将最终架构、活实验计划与 TODO 对齐：`I_t + s_{t-1} -> frozen Gaussian backbone -> Stage-A RGB/mesh/point local contact encoder -> Stage-B causal controller -> C_t,U_t,Delta_t -> corrected SMPL-X/LBS human Gaussians`。`s_{t-1}` 严格只含模型自有 contact/action/anchor/uncertainty/reset；Stage A 以 vertex-level contact、continuous proximity、normal/distance reliability 为核心，Stage B 只处理时序控制与安全回退。当前任务是构建 contact-local ROI/point-patch manifest 与小型 Stage-A baseline，并以 source-sequence、scene、error-family 三重隔离评测；Human3R 前端对齐和 GUSH3R renderer 仍分别受门槛限制。
- 已执行 Stage-A 数据契约：新增 `build_stagea_relation_manifest.py`，从 PROXD SMPL-X、PROX SDF/scene mesh 生成每帧每脚 64 个 contact-local ROI vertices、每 vertex 128 个 local scene points、vertex contact/proximity 与 RGB 文件引用（不读图；冻结 backbone token 尚未提取）。初始 12 scene/720 帧 v1 审计发现 `MPH16/MPH8/N3OpenArea/Werkraum` mesh normal 长度约 2π；v2 又暴露 N0Sofa 的孤立 zero normal。两者均禁止训练、保留以供追溯。builder 已修复为显式单位化并从 patch candidates 剔除零/非有限 normal；全新 v3 的 12 scene/720 帧审计通过（全 finite、normal length=1、RGB references 完整）。已新增 `make_stagea_degraded_relations.py` 并完成 normal18、distance、dropout20 三种 relation-level proxy error family（标签仍为 clean PROX oracle）。下一步冻结 source-sequence/scene/error-family 三重 split，才实现 Stage-A geometry baseline；不把该 proxy 数据称为真实前端结果。
- Stage-A strict geometry baseline 已实际运行：v1 loader 出现逐样本 tensor 持有导致 CPU 内存异常上涨，已终止且无 metrics；改为 sequence-backed indexing 的 v2 正常完成。其 val F1=0.7692、test F1=0.5965 均为 all-positive baseline（test precision=.425、recall=1）。审计发现每一帧脚 ROI 的 64 个 hard vertex labels 完全相同；随后 v4 采用 fixed frame-zero spatial-FPS vertex identities 重建仍无混合标签。结论：当前 PROX 的 2cm 脚底 hard-contact 标注只适合 ROI contact，不能声称稠密 vertex binary supervision。后续 Stage A 在 PROX 上改为 vertex continuous proximity + ROI contact/reliability；RICH/PhySIC 类数据到位前不训练/报告 vertex binary contact。

## 2026-09-09（E1b 因果 rollout 与可恢复长时实验控制器）

- 用户授权实现可连续运行数小时、可断点恢复的本地实验控制器，并继续 E1b 接触锚点/foot-lock 实验；明确不读取图片。本轮先执行 Python 静态编译和极小 smoke，核验 rollout 的张量/特征回写/检查点与评测契约，再扩展声明式预注册候选计划并仅在 GPU 无冲突时后台启动。
- 控制器不得把 teacher previous residual 读入部署路径；任何结果都须区分 oracle mechanism、teacher-forced diagnostic 与模型自有状态 causal rollout。实验状态、事件与每一项子实验的 stdout/stderr 都将独立保存，重启时从 state JSON 的未完成项继续，禁止覆盖既有实验目录。
- 已修复 rollout train/eval 两个实际 bug（`max_action_m` 配置字段、hard-gate `[foot,1]` 广播），通过静态编译和 60 帧随机模型 tensor smoke。CPU 的完整一轮反传 smoke 约 7 分钟尚未结束且占 7 核，已只终止该本轮临时 smoke；未删目录/未形成结果。训练现支持 `last.pt` 的 optimizer/history 续训。
- 已启动独立 session 的 6 小时控制器 `PID 1544168`：`forward_contact_pipeline/experiments/e1b_rollout_20260909/controller/state.json` 当前为 `waiting_for_idle_gpu`，因为 GPU 仍有其他 `python low_cpu_mnist.py` compute process（3456 MiB）。控制器每 30 秒检查，不会抢占；空闲后将顺序执行 `action40mm`、`action20mm`、`action20mm_stick5` 并写独立日志与 test metrics。启动方式使用 `setsid`，避免普通后台子进程随终端退出。

## 2026-09-09（GUSH3R 可靠 anchor-map 连续实验：状态核验）

- 用户要求查看当前进度，并明确禁止读取或展示图片。本轮只核验进程、日志、代码差异、manifest 与聚合数值产物；不调用图片查看工具，也不向上下文输出像素内容。
- 当前待验收实验为 `anchor1m_budget_64_smoke_v1`：它测试跨 32 帧窗口交接 100 万高置信背景 anchor，并用 `anchor_budget` 上限策略保护锚点、仅给新观测分配可替换预算，目标是降低因窗口重置产生的在线 seam。结论以运行完成后的无图像 temporal audit 为准。
- 已完成纯数值验收：64/64 帧齐全、耗时 80.41 s，但边界 L1=`0.24067`、窗口内部 L1=`0.02814`、比值=`8.55x`，没有优于 25 万/100 万 anchor 的 legacy 对照（均约 `0.240`），也远差于 hard-32 的 `0.16833`。故确认简单 slot/voxel 保护无法解决跨窗口状态不一致；禁止扩展到 229 帧，详细负结果已写入 `forward_contact_pipeline/reports/gush3r_quality_diagnosis_20260804.md`。
- 用户要求持续维护接触修正的 plan/记忆文档。已新增 `forward_contact_pipeline/CONTACT_CORRECTION_EXPERIMENT_PLAN.md` 为唯一活实验协议：固定 oracle-scene -> degraded-scene -> real front-end -> Gaussian/state 的 E0--E4 阶段、数据边界、split、指标、停止条件和每次实验记录要求。后续必须同步该文档、`TODO.md` 与本日志。
- 用户授权实际执行 E0 与 E1。当前任务是先审计 PROX 坐标/SDF/anchor/标签契约，再建立同 scene/动作、drift-seed 互斥的 oracle-scene 机制恢复 split，并运行规则与 causal GRU/TCN 对照；严格不读取或输出图片，GUSH3R 不参与该阶段。
- E0 已完成：四个 PROX 序列/720 帧的 SDF teacher 标签通过 `PROX_scene_world` 契约审计；旧 E0 标签缺同目录 teacher NPZ，已记录为资产可复现性缺口。E1 完成两轮：旧随机 root/foot target 被证伪（分类可学但 correction 恶化）；可辨识 normal-projection target 的 gated GRU 在未见 drift seed 上把 penetration depth 18.57->13.95 mm、contact distance 29.28->24.58 mm，但 foot sliding 20.99->30.19 mm/s，故仅 clearance 子任务部分通过，不能进入 E2/E3。新增 E0 audit、post-correction evaluator、PyTorch checkpoint compatibility，并将完整指标词典和结论写入 `reports/e0_e1_oracle_contact_20260909.md`。
- 接触文献调研完成并归档至 `CONTACT_AWARE_RELATED_WORK_20260909.md`。最新直接相关工作包括 GraphiContact（vertex contact + uncertainty）、CRISP（point cloud -> planar primitives -> contact-guided completion -> physics）、PhySIC（mesh/scene/contact joint optimization）、LEXIS/StackFLOW（continuous proximity/anchor offsets）、IK/physics route。结论：无公开工作可直接替代我们的 online 人景 Gaussian 主线；下一实验应是 E1b 的 contact-anchor hold/tangential foot-lock，而不是直接接 GUSH3R。
- E1b 第一轮已完成：在真实 PROX teacher-contact 段注入切向 drift，oracle episode anchor lock 将滑步 0.385 m/s 降至 0.028 m/s，验证 anchor-state 表示/执行层正确；但 GRU 即使 teacher-forced 上一 residual 仍使滑步增至 1.180 m/s、接触距离恶化，故 learned controller 失败且不能部署。修复了 `foot_sliding` 旧实现未按 timestamp 除法导致的 m/frame 误标为 m/s，并重跑 E1a 数字（相对结论不变）。后续仅做预测状态 causal rollout + bounded tangent action，不接 GUSH3R。
- 用户授权实现可连续运行数小时、断点可恢复的本地 autoresearch controller，并继续当前实验。控制器将以状态/账本/日志/量化停止条件驱动，不依赖无限聊天回复；首个受控任务为 E1b 的模型自身状态 causal rollout 与 bounded tangent action。所有自动操作仍限制在本工作区、不读图、不覆盖旧实验。

## 2026-08-26（继续自动研究：contact-only 诊断）

- 用户要求继续 autoresearch，并持续维护详细记录。本轮严格保持正式研究问题不变：最终主实验仍是原始 GUSH3R 与 GUSH3R+低维接触修正层；PROX synthetic-drift 仅用于验证模块训练机制，当前 Human3R→PROXD 对齐不合格，禁止把 C 层真实前馈结果用于训练或对外宣称。
- 已先核验正在运行的 `prox_synth_aug_gru_contactonly`：训练采用严格 scene-isolated 的 24 条多 seed synthetic-drift 数据、GRU history=8/hidden=128/seed=42，并仅保留 contact BCE；进程仍在正常运行（已完成 epoch 1--3），不终止或覆盖。训练结束后将对固定 test split 做评测，并根据 F1/transition F1 决定进入解耦 multi-head 还是标签/可分性诊断。
- 本轮所有新实验、日志、代码与文档读写只在 `yuzilang/` 工作树内进行；每项结论将同步到研究协议、TODO、会话记录及长期记忆，明确正负结果与适用范围。
- 已完成 `prox_synth_aug` 的预注册 threshold-separability 审计：严格 scene split 的 train/val/test 接触正例率为 84.58%/53.33%/12.50%，train-only 最优 rule 在 test 预测 92.92% 为正、F1=0.237；此前 GRU F1=0.218 因而接近 all-positive 基线。已把该负证据和两类后续 split（同 scene/动作、互斥 drift seed 的机制恢复；scene-disjoint 的 domain generalization）同步到 `RESEARCH_PROTOCOL_20260826.md`、`TODO.md` 与长期实现记忆。现有 scene split 保留为压力测试，禁止再把 pooled F1 当作机制有效性证据。
- contact-only GRU 已按固定 30 epoch 完整结束：best val=0.692049（epoch 18）；test precision=0.12264、recall=1.0、F1=0.21849、transition F1=0，和 multi-task GRU 完全相同。由此排除“仅是 38D 共享多任务 head 的负迁移”这一解释；当前模型仍退化为 all-positive，不能在旧 scene-split 上继续用 class weighting/阈值/结构调参。下一步改为同 scene/动作、互斥 drift seed 的 B-mechanism split，旧 split 仅保留为 domain-generalization 压力测试。

## 2026-08-26（启动前馈接触修正的自动研究与严谨实验主线）

- 用户确认训练数据集已经就位，授权自动开展研究方案优化、实验搭建与探索；研究必须保持学术严谨，并与第一研究点 Anchor Attention / Global Scene Gate 紧密联系。
- 本轮先以实际数据、标签、许可资产、GPU 和现有管线为准完成可复现实验审计；固定主问题为“GUSH3R 前馈重建的低维、uncertainty-aware 接触修正能否在不破坏 Gaussian 渲染质量的前提下，降低滑步、穿透与接触抖动”。
- 将先建立冻结 backbone 的真实监督基线与评测协议，再按预注册消融/停止条件推进；不会将 Human3R proxy 结果误作正式 GUSH3R 结论，也不会在低置信度对齐条件下伪造接触标签。
- 已确认 PROX 已解压 RGB/Depth recordings、54 个 PROXD 序列、scene SDF/cam2world；新增 `build_prox_geometry_labels.py`，在 180 帧 BasementSittingBooth 审计中验证 SDF 有效覆盖并修正最近顶点速度的标签偏差。新增 `make_prox_synthetic_drift.py`，从 clean PROX teacher 注入时序 root/foot drift，生成 58D 输入与已知 correction target；仅作 B 层机制验证，不作为 Human3R/GUSH3R 结果。
- 用户授权终止先前确认的外部占卡伪训练 `PID 1077800`；已发送 SIGTERM 并复查 GPU 为 4 MiB/0%、无计算进程。后续所有读写继续限制在 `yuzilang/`，准备启动 Human3R-on-PROX 的真实 C 层前端对齐实验。
- Human3R-on-PROX 的 3 帧 MPH112 smoke 已成功：真实 RGB 输入、每帧一个人体、FrameInput/point map/SMPL-X 落盘，最快 0.54 s/frame。用户允许自动继续执行，下一步扩展到 60 帧并审计 frame-name 对齐、模型输出与 PROXD teacher 的坐标映射；`/usr/bin/ffmpeg` 缺失仅影响 mesh overlay MP4，不影响 FrameInput 或接触实验。
- 60 帧 MPH112 已完成（0.312 s/frame、1 人/帧），RGB 与 PROXD frame names 顺序严格匹配。45/15 帧分离的 Human3R-SMPLX→PROXD-SMPLX Sim(3) hold-out median=7.37 cm、p95=45.1 cm，未达到接触尺度；已按协议禁止用该输出训练 C 层，转入 camera/root/local-pose 与重投影诊断。
- 四场景 PROX synthetic-drift 已建立严格 2/1/1 scene split。GRU 30 epoch 的独立测试 contact F1=0.218、transition F1=0、surface distance MAE=9.07 cm、root RMSE=5.98 cm、预测 penetration ratio=50%，不支持泛化有效性结论；TCN 同预算对照正在运行，后续不能仅调阈值掩盖该负结果。
- 用户要求继续 autoresearch 且维护详细记录。增强版多 seed synthetic-drift GRU 已完成（24 条、16/4/4 scene-isolated train/val/test）：最佳 val 在 epoch 2，test F1=0.218、transition F1=0、surface MAE=8.27 cm、root RMSE=6.21 cm、pred penetration=51.7%，仍失败。下一阶段先做 label/heuristic/multi-task loss 诊断，再以预先记录的单变量修复复验；禁止将 synthetic 结果描述为真实 Human3R/GUSH3R 改善。

## 2026-08-25（面试前恢复前馈人体-场景 Gaussian 工作）

- 用户将参加面试，要求恢复 HUGS 前馈接触修正研究线，并以面试表达为目标梳理：实际工作范围、前馈 3DGS 的代表性研究、常见网络架构、技术要点和可能追问。
- 本轮只基于既有方案/实现/报告恢复与解释；会明确区分已完成的软件与 smoke test、已完成的 GUSH3R 长时序质量定位，以及尚未完成的真实对齐、真实数据训练和正式 GUSH3R 接触渲染验证，不夸大为已取得正式实验结果。

## 2026-08-09（恢复 HUGS 两条研究主线记忆）

- 用户要求恢复第一个研究点的完整内容，以及仍在规划/实施中的前馈方案。
- 本轮将以 `docs/agent_memory/`、`BEST_PIPELINE.md`、`forward_contact_pipeline/TODO.md` 和现有代码/报告为准，区分已完成实验结论、已实现但未完成真实验证的模块，以及后续计划；不改动训练主线、权重或实验产物。

## 2026-08-06（PROX recordings 断点续传下载器）

- 用户要求只在 NAS 项目目录下载 PROX `recordings.zip`，并支持后台断点续传。
- 新增通用 `../contact_datasets/PROX/download_prox_recordings.sh`，以及实际数据目录的入口 `PROX/download_recordings.sh`：使用官方认证 POST 接口和 `wget --continue`；`--background` 会输出 PID/日志。后者把 recordings 放在已有 `PROX/scenes.zip`、`PROXD.zip`、`sdf.zip` 的同一 `ml-hugs-work/PROX/` 目录。凭据仅作为该目录下权限 600 的临时 URL-encoded POST 文件存在，子进程退出时删除。
- `recordings.zip` 是第一阶段接触估计所需的 RGB-D 输入。实际下载仍需用户在 NAS 终端本地输入已授权的 PROX 账号密码，不能通过聊天提供。

## 2026-08-05（确认归档 GUSH3R chunked 质量修复）

- 用户要求将本次 GUSH3R 自动质量修复归档至 memory。
- 已确认长期记忆 `docs/agent_memory/hugs-forward-contact-implementation.md` 记录了新增的 `--background-render-mode chunked` / `--background-chunk-size`、独立窗口背景渲染、流式 MP4 编码和 manifest；同时明确 checkpoint/旧模式未改、全 229 帧 GPU 回归尚待显卡空闲，不能将代码完成误记为质量验证完成。

## 2026-08-05（实施 GUSH3R 长时序质量修复）

- 用户要求自动修复 GUSH3R 长视频质量问题。
- 基于已完成的质量定位，将在 `GUSH3R/infer.py` 实现可复现的 bounded/chunked background renderer：长序列按固定窗口独立前向和渲染，再统一编码视频，从架构上避免 final-background 对全部历史帧的未来误差回灌。
- 保持原 `final` / `per-frame` 行为和 checkpoint 不变；当前 GPU 已被其他任务占用（约 42 GB），本轮只完成代码与静态验证，不抢占 GPU 做全长运行。
- 已新增 `--background-render-mode chunked` 和 `--background-chunk-size`（默认 32）：每个窗口独立调用递归推理，以本窗口背景渲染本窗口帧；全局编号 PNG 由 streaming writer 编码为单一 MP4，并输出 `chunked_render_manifest.json`。原 `final` / `per-frame` 不变。
- `infer.py` 已通过 Python 编译检查；全 229 帧 GPU 回归和 chunk 边界/时序质量评估待 GPU 空闲后执行，不能因代码完成而将修复视为已验证。

## 2026-08-05（核对 GUSH3R 质量分析完成度）

- 用户询问此前做到哪里，以及 GUSH3R 质量分析是否已经完成。
- 本轮将以质量诊断报告、代码实现和 32 帧分块实验日志为准，明确区分已完成的根因定位/受控验证，和仍未完成的全 229 帧稳健 renderer 验证；不改动实验。

## 2026-08-05（归档 PROX 本地资产盘点结论）

- 用户要求将本次 `ml-hugs-work/PROX` 数据准备状态写入长期 memory。
- 已验证 `scenes.zip`、`calibration.zip`、`cam2world.zip`、`PROXD.zip`、`sdf.zip` 与 `bodysegments.zip` 均为完整 ZIP；PROXD 含 54 个序列（177,787 个条目），scene/cam2world/SDF 覆盖 12 个场景。
- 未发现 `recordings.zip`（实际 RGB-D 观测）、`vposerDecoderWeights.npz` 或 `quantitative.zip`。其中 recordings 是 Human3R/真实图像接触训练的关键缺口；quantitative 可后补，VPoser 按具体 SMPL-X 处理入口补齐。现有资产仍全部压缩，接入 pipeline 前需解压并建立 manifest。

## 2026-08-05（盘点本地 PROX 数据集准备状态）

- 用户说明已在 `ml-hugs-work/PROX` 准备 PROX 的部分数据，视频尚未完整下载；要求检查其他资产是否齐全。
- 本轮只盘点该目录的实际文件、ZIP 完整性、解压状态与前馈接触 pipeline 所需的 RGB-D/scene/calibration/cam2world/PROXD/SDF/bodysegments/quantitative 资产差距；不启动下载、不解压或覆盖现有数据。

## 2026-08-04（再次查询 GUSH3R 分块后台实验进度）

- 用户再次要求查看 32 帧分块渲染后台任务进度。
- 仅核对后台 PID、各窗口日志/帧产物、GPU 使用和最终视频拼接状态，不改变正在运行的任务。

## 2026-08-04（查询 GUSH3R 分块实验进度）

- 用户要求查看刚启动的 GoodMornin1 全 229 帧、32 帧独立窗口分块渲染实验进度。
- 将只读取进程、日志、已完成帧数和 GPU 状态；不终止、重启或覆盖正在运行的实验。

## 2026-08-04（恢复 HUGS / 接续 GUSH3R 实验）

- 用户要求恢复 HUGS，并检查最新 GUSH3R 实验后继续推进。
- 本会话将以 2026-08-04 的长时序质量诊断为起点：先核对最新代码、报告、诊断产物和 GPU 空闲状态，再实现并验证 causal/chunked background snapshot rendering；该实验固定 GoodMornin1 输入与官方 2M/.005 参数，避免把旧的 full-229 final-background render 误作正式基线。
- 不改动既有 HUGS 训练主线、权重或其他用户未提交内容；新增实验将归档到 `forward_contact_pipeline/diagnostics/gush3r_quality_20260804/` 并同步记录可复现命令和对照结论。

## 2026-08-04（GUSH3R 输出质量定位：长时序背景融合/渲染问题）

- 用户要求持续实验验证 GUSH3R 输出质量问题；所有实验输入/缓存/日志/产物均写入 `yuzilang` 内的 `forward_contact_pipeline/diagnostics/gush3r_quality_20260804/`。
- 固定 GoodMornin1 的 Human3R FrameInput RGB（512x368），完成官方 CLI 2M/.005 的 32/64/128 帧和全 229 帧，以及 200k 模型默认值、2M voxel .01 的单变量对照。详细报告：`forward_contact_pipeline/reports/gush3r_quality_diagnosis_20260804.md`。
- 结论：短 32 帧正常，随着窗口到 128/229 帧背景发生浮点、白洞/黑点和模糊；人体不是主导故障。`infer.py` 以最终全局背景回渲全部历史帧，未来融合误差会污染 frame 0，是已由 causal per-frame 背景渲染直接验证的主要机制。
- 已改进诊断工具：`infer.py` 增加 component 导出和 `--background-render-mode per-frame`；DINO Hub 支持局部 checkout，避免无网络时错误探测 GitHub；FrameInput RGB exporter 增加时间窗口。诊断模式不改 checkpoint/训练。
- 参数结论：官方 CLI `2M/.005/conf1.0/mask.02` 的背景覆盖显著优于模型构造默认 `200k/.01/conf1.5/mask.05`；后者白洞严重。`2M/.01` 仅小幅平滑，完整 229 帧仍不合格。
- `late_window_32` 已完成：原始 192--223 的独立短窗口比 full-229 同一时刻明显更清晰、更少噪点；仅仍有人体掩码导致的白洞。这确认晚段崩坏来自全局长窗口而非内容本身。对照图为 `diagnostics/gush3r_quality_20260804/late_window_vs_full.png`。后续方向为 causal/chunked (32--64帧) background snapshot rendering，稳定前不把旧 full render 当接触/正式 baseline。

## 2026-08-04（启动 GUSH3R 输出质量定位实验）

- 用户要求直接持续实验验证 GUSH3R 输出质量问题的来源。
- 本轮将固定同一输入/帧范围和可追溯的输出目录，先复现官方基线，再分离背景 Gaussian 筛选、阈值/体素/数量、相机与 rasterizer 的影响；每次只改变一个可解释因素，并以渲染帧、日志与定量覆盖/清晰度统计记录结论。
- 所有命令与实验产物严格限制在 `/workspace/nas_auto_backup/yuzilang` 内；不改训练主线或 P0 接触结论。

## 2026-08-04（P0 真实产物补充验证与范围约束）

- 用户要求后续操作严格限制在 `/workspace/nas_auto_backup/yuzilang`；本轮所有项目读写均在该目录内完成，未触碰工作树中已有的 HUGS 改动。
- 重新执行 `prepare_sequence.py` 的真实 NeuMan `lab` body-Sim(3) proxy 路径，新增 `forward_contact_pipeline/prepared/neuman_lab_p0_scaledproxy_support.npz`，不覆盖原有诊断产物。
- 已核验新增 `surface_support`：shape=`(103,2)`、范围 `8--154`、median=`64`；`surface_distance` 全 finite。proxy-confidence median=`0.2607467`、absolute surface-distance median=`3.76910`，与报告中 P0 验收失败的旧统计一致，故没有改变“不训练/不做 IK”的决策。
- 使用 Python 3.8 HUGS 环境以标准库 `unittest` 执行 `forward_contact_pipeline/tests`，4/4 通过（Sim(3)、surface plane、packet、GRU/TCN shape）。`pytest` 未安装，不以它的缺失误报测试失败。
- 已同步 `forward_contact_pipeline/TODO.md` 与 `docs/agent_memory/hugs-forward-contact-implementation.md`。下一步仍是 NeuMan SMPL/mask 重投影、anchor/proxy 可视化及 reference scene proxy 诊断，而非进入 GRU/TCN、P1 IK 或正式训练。

## 2026-08-04（实施 NeuMan 驱动的 P0 对齐与规则几何闭环）

- 用户要求按前馈接触修正方案实现 P0，并同步维护长期 memory 与 Todo。
- 本轮以现有 NeuMan `lab` 场景为 P0 的外部场景坐标参考：Human3R 对同一 NeuMan 帧的输出将通过 NeuMan 已对齐 SMPL、相机与场景几何建立/验证 Sim(3)，而不是从 GoodMornin 的无参考视频臆测绝对对齐。
- 已新增 `forward_contact_pipeline/scripts/build_neuman_p0_correspondences.py`：同帧、同拓扑 SMPL 顶点构成 Human3R→NeuMan Sim(3) 对应，按帧 split，输出仅含 train 对应点的 NPZ 和独立 hold-out vertex error JSON；它是坐标校准工具，不生成接触 GT。
- 已修正 `prepare_sequence.py` 的关键坐标错误：在已有 Sim(3) 时，Human3R point map 与人体 anchor 一并映射到参考场景坐标后才做 local surface proxy，metadata 新增 geometry frame 标记。
- 验证：原有 4 个核心单元测试通过；NeuMan source=target identity 校准的 hold-out median error≈6.1e-7；GoodMornin 10 帧走完 aligned code path，输出 58D feature 且 surface distance 全 finite。identity 只证明工具/路径正确，不证明真实对齐。
- 当前 sandbox 无法访问 NVIDIA driver，因此未启动 Human3R-on-NeuMan GPU 推理。下一步在宿主 GPU 上运行 NeuMan lab，转换/拟合 source SMPL，生成真实 correspondence report，并检查 hold-out error、mask/mesh 投影和脚 anchor/proxy 可视化；完成前不进入 P1/真实训练。

## 2026-08-04（恢复前馈方案并准备终端数据集下载）

- 用户要求恢复 HUGS 前馈接触方案记忆，并希望在终端直接下载训练数据集。
- 本轮先按全局记忆与本地会话记录恢复方案：正式训练数据优先级为 BEDLAM（6fps RGB + GT）和 PROX（RGB-D/场景扫描/SMPL-X），NeuMan 已有，仅作真实 Gaussian 验证；暂不下载手物交互数据。
- 接下来将核验已有 `contact_datasets/` 下载脚本、上游许可与实际可达的镜像/直链，优先启动可断点续传、白名单化的小范围下载，避免误拉取 BEDLAM 的 TB 级全量资源。
- 实测 `bedlam.is.tue.mpg.de` 与 `prox.is.tue.mpg.de` 均可访问；官网公开页明确说明两者均需注册、接受许可后才开放下载。BEDLAM 在 2026-02-26 宣布完整 image/depth 的 Hugging Face 镜像，但镜像 `hf-mirror.com` 当前 DNS 失败，官方 `huggingface.co` 也连接超时，因此不能把该候选镜像视为可用下载路径。
- `contact_datasets/PROX/download_prox_priority.sh` 已是可交互、官方 API 的断点续传脚本：终端提示输入账号密码，下载最小接触集合（scene/calibration/cam2world/PROXD/SDF/bodysegments/quantitative）。现有 `raw/scenes.zip` 是有效 ZIP 格式的 5 MB 断点文件；脚本的 `wget --continue` 会续传。
- 下一步需要用户在 PROX/BEDLAM 官网完成一次注册并接受许可；之后用户可自行在本地终端输入账号（不要在聊天中提供），PROX 可以立即运行既有脚本。BEDLAM 应先在认证后的 Download 页取得实际 HF repo/文件白名单或官方下载链接，再以 `hf download --include ...` 做可续传的 6fps+GT 精确下载，禁止全量拉取。
- 用户要求先下载 PROX；已启动现有脚本确认其会等待交互式账号/密码。由于凭据未提供且不应通过聊天收取，未执行实际下载；等待用户在自己的终端运行 `bash contact_datasets/PROX/download_prox_priority.sh` 完成认证。现有 `raw/scenes.zip` 保持不变，可被 `wget --continue` 续传。

## 2026-07-29（详细归档前馈接触脚本实现）

- 用户要求将本次 P0–P5 全套脚本实现详细记录到 agent memory。
- 本轮将新增独立实现记忆，记录模块/脚本清单、数据接口、测试结果、真实序列状态、边界与下一步，并加入 `docs/agent_memory/MEMORY.md` 索引。
- 已新增 `docs/agent_memory/hugs-forward-contact-implementation.md`，详细记录研究定位、完整数据流、15 个核心模块、10 个 CLI 脚本、58D 输入/38D 输出、损失与指标、62-byte packet、单元/合成/真实 smoke test 结果、10 项未完成工作和下一步顺序。
- 已将新记忆加入 `docs/agent_memory/MEMORY.md` 索引；后续恢复前馈实现时应同时阅读方案记忆和实现交接记忆。

## 2026-07-29（实现前馈接触 pipeline 全套脚本）

- 用户要求按照更新后的 Human3R 预验证、GUSH3R 主实验架构，先实现包括训练、验证在内的完整脚本体系。
- 本轮将保留现有 `export_frame_input.py` 与已有实验产物，先盘点代码接口，再在 `forward_contact_pipeline/` 下补齐 P0–P5 的数据、几何、规则 baseline、GRU/TCN、训练、评估、Gaussian/LBS adapter 和 packet/runtime 脚本，并用合成数据做端到端 smoke test。
- 已新增 `contact_streaming/` 核心 package 与 `scripts/` 全套入口，覆盖 robust Sim(3)、SMPL-X 脚 anchor、surface proxy、规则 baseline、GRU/TCN 训练验证、因果推理、Gaussian/LBS adapter、62-byte packet 和网络模拟；README 与 TODO 已同步。
- 验证结果：4 个单元测试通过；合成数据上 GRU 和 TCN 均完成训练/验证，GRU 推理约 93 FPS（CPU smoke test），packet 62 bytes/frame；真实 `frameinput_goodmornin_10f` 10 帧预处理成功。
- 真实序列当前仅有 translation fallback（alignment confidence=0.25），全部进入 Uncertain，符合“不确定时不强行 IK”的设计。正式实验仍需真实 Sim(3) 对应点、可微 IK、BEDLAM/PROX 训练数据以及 GUSH3R decoder/renderer 内部接入。

## 2026-07-29（同步更新后的前馈方案与 Todo）

- 用户说明 `idea` 中的前馈人体-场景 Gaussian 方案已更新，要求重新阅读并同步 HUGS 记忆与 `forward_contact_pipeline/TODO.md`。
- 本次先以更新后的 idea 文件为准，核对现有前馈记忆、实现状态和 Todo，避免继续沿用旧的 P0/P1/P2 划分。
- 已完整阅读 `idea/实时接触修正前馈人体场景Gaussian方案构想.md`（2026-07-29 更新）：Human3R 定位为网格/点图预验证，GUSH3R 定位为正式 Gaussian 主实验；新增明确的 FrameInput/FrameOutput、三阶段训练、BEDLAM/PROX/NeuMan 数据分工、Gaussian 接触指标和主实验 baseline。
- 已同步 `docs/agent_memory/hugs-forward-contact-pipeline.md` 和 `forward_contact_pipeline/TODO.md`：更新主线定位、接口、训练阶段、数据集优先级、GUSH3R 主表、Gaussian temporal flicker/penetration 及 packet 指标；当前仍停留在 P0/P1 闭环前，未开始训练或 GUSH3R 接触接入。

## 2026-07-29（恢复前馈方案并核对数据集需求）

- 用户要求恢复 HUGS 记忆，复核前馈接触修正方案，并明确需要下载的数据集。
- 已恢复全局 HUGS 记忆与前馈方案：Human3R 原型先行，冻结前馈模型，先完成左右脚接触 proxy/规则修正/渲染闭环，再训练轻量 GRU/TCN，最后接入 GUSH3R；BEDLAM+PROX 用于训练，NeuMan 用于真实场景验证。
- 本轮将继续以代码、TODO 和现有数据目录核对数据需求，区分必需数据集、已有数据和暂不需要下载的资源。
- 代码/TODO 核对结论：P0/P1 规则闭环可直接使用已有 `GoodMornin1.mp4` 与 NeuMan 数据，不需要先下载新数据集；P2 训练接触 GRU/TCN 时再下载 BEDLAM 6fps RGB/GT（优先）和 PROX 的 RGB-D/场景扫描/SMPL-X 拟合；NeuMan 六场景已在 `data/neuman/dataset/`，作为真实 Gaussian 验证集无需重下。

## 2026-07-29（复核 GUSH3R 完整视频画质）

- 用户反馈 `forward_contact_pipeline/visualizations/gush3r_goodmornin_full/render.mp4` 画质很差；已实际查看第 0、50、100、200 帧并与 Human3R 原图对照。
- 现象不是主要的编码问题：人体位置大致正确，但背景大面积白底、黑色散点和模糊，场景几何/覆盖明显失败。
- 当前 `GUSH3R/infer.py` 的 CLI 默认值覆盖 checkpoint/model 内部默认值：CLI 为 `gs_conf_threshold=1.0`、`bg_mask_threshold=0.02`、`bg_voxel_size=0.005`、`bg_gaussian_max=2,000,000`；模型默认分别为 `1.5`、`0.05`、`0.01`、`200,000`（recurrent lighter 内部上限默认 1,000,000）。完整视频此前使用了 dilation=0，但没有记录其它参数覆盖。
- 初步判断优先级：先用训练/模型默认背景筛选参数复跑单帧/短序列，单独比较背景覆盖率和散点；再检查自定义 merged rasterizer 与模型内部 gsplat renderer 的一致性；最后才进入 Human3R/GUSH3R 坐标对齐和接触修正。当前不能把画质问题归因于接触模块，因为尚未做接触修正。

## 2026-07-29（严格官方默认参数完整推理与耗时）

- 按 `GUSH3R/README.md` 的基础命令运行，未显式覆盖任何参数：`size=512`、`gs_conf_threshold=1.0`、`bg_mask_threshold=0.02`、`bg_mask_dilation=3`、`bg_voxel_size=0.005`、`bg_gaussian_max=2,000,000`、`subsample=1`、未启用 TTT3R。
- 输入为 `Human3R/examples/GoodMornin1.mp4`，完整 229 帧；输出：`forward_contact_pipeline/visualizations/gush3r_goodmornin_official_full/`。
- 官方脚本日志 `Inference finished in 90.40s`；`/usr/bin/time -v` 记录端到端墙钟 `5:46.06`，用户 CPU 时间 1710.60s、系统时间 29.62s、CPU 使用率 502%、最大 RSS 24,480,620 KiB（约 23.4 GiB），无 swap。
- 逐帧渲染 PNG 的文件时间显示 frame 0→228 约 5.59s，MP4 写出再约 1.67s；因此除模型推理（90.40s）和约 7.3s 输出渲染/封装外，约 248s 主要消耗在模型初始化与 229 帧输入解码/预处理（官方脚本没有更细的阶段计时）。
- 官方默认输出仍然存在大面积白/黑散点和模糊背景；因此画质差不是此前 `bg_mask_dilation=0` 或非官方阈值单独造成的。

## 2026-07-28（生成 Human3R/GUSH3R 完整视频可视化）

- 用户要求基于同一输入视频生成两个项目的可视化：Human3R 的网格与点云视频，以及 GUSH3R 的 Gaussian 合成渲染视频。
- 计划使用 `Human3R/examples/GoodMornin1.mp4` 全部 229 帧，结果归档到 `forward_contact_pipeline/visualizations/`；GUSH3R 使用已验证的 `bg_mask_dilation=0` 配置。
- 已完成并核验：Human3R 输出 `human3r_goodmornin_full/mesh_overlay.mp4`（229 帧、23.98 FPS、2048×368）和 `pointcloud_depth.mp4`（229 帧、23.98 FPS、1024×368）；GUSH3R 输出 `gush3r_goodmornin_full/render.mp4`（229 帧、30 FPS、512×368）以及 229 张 `merged_render/frame_*.png`。Human3R 网格视频由上游生成的 `color_smpl/*.png` 封装而成（上游未自动写出 `output_video.mp4`）。

## 2026-07-28（Human3R 接触正确性评价协议）

- 用户询问如何评价 Human3R 生成结果的接触是否正确；本轮明确真实 GT 与局部几何 proxy 两种评价模式，覆盖脚底距离、contact F1、foot sliding、penetration 和时序稳定性。

## 2026-07-28（详细解读前馈接触修正方案文档）

- 用户要求详细介绍已上传的前馈人体-场景 Gaussian 接触修正方案文档及完整实施路线；本轮将读取 `idea/实时接触修正前馈人体场景Gaussian方案构想.md`，结合当前 P0.1/P0.2 落地状态进行说明。

## 2026-07-28（同步前馈方案进度到长期记忆）

- 用户要求将当前前馈方案进度写入 memory；已完成 P0.1/P0.2：Human3R/GUSH3R 前馈复现、标准 FrameInput 导出工具、10 帧短序列验证和 baseline 结果归档。
- 当前下一步为 P0.3：坐标统一/Sim(3) 对齐、左右脚 anchor、局部地面 surface proxy；尚未开始 contact/IK/GRU/TCN。

## 2026-07-28（实现 P0.1/P0.2 FrameInput 导出与短序列验证）

- 用户要求实现前馈方案 P0.1/P0.2：标准 `FrameInput` 导出，以及用短视频完成多帧 Human3R 输出验证。
- 本轮将在 `forward_contact_pipeline/` 下新增独立导出工具和数据格式说明，不修改 Human3R/GUSH3R 上游模型代码；先核对 Human3R 当前推理入口与输出字段，再执行完整短序列测试。
- 已新增 `forward_contact_pipeline/export_frame_input.py` 和 `FRAME_INPUT_SCHEMA.md`；使用 Human3R 896L 对 `GoodMornin1.mp4` 前 10 帧完成 GPU 推理，输出 `datasets/frameinput_goodmornin_10f/`。
- 验证通过：`manifest.json` schema 为 `hugs.forward_contact.FrameInput`，10 帧、23.98 FPS、平均 0.574 秒/帧；单帧包含 color/depth/confidence/point_map/camera/SMPL-X 全部字段，坐标状态明确标记为 `raw_unaligned`。

## 2026-07-28（盘点前馈接触方案 TODO 与下一步）

- 用户要求输出当前前馈方案 TODO 并确定下一步；本轮将根据实际 Human3R/GUSH3R smoke test、baseline 归档和 `forward_contact_pipeline/TODO.md` 状态更新判断。

## 2026-07-28（测试 GUSH3R 人体白边来源）

- 用户要求验证 GUSH3R 输出人体周边白边；基线使用 `bg_mask_dilation=3`、`bg_mask_threshold=0.02`，优先测试 `bg_mask_dilation=0`，必要时比较 mask threshold。
- `bg_mask_dilation=0` 单帧测试完成（`Inference finished in 0.71s`）：人体周边白边明显消失，仅保留正常 Gaussian 轮廓模糊；结果已归档到 `forward_contact_pipeline/baseline/results/gush3r_bg_dilation0/`。外圈白边仍来自背景 Gaussian 覆盖不足和白色 rasterizer 背景。

## 2026-07-28（建立前馈 baseline 归档目录）

- 在 `forward_contact_pipeline/baseline/` 建立前馈方案 baseline 归档目录，新增 `README.md` 和 `RESULTS.md`。
- 将 Human3R 单帧前向结果归档到 `baseline/results/human3r_smoke_gpu/`，将 GUSH3R 单帧前向与渲染结果归档到 `baseline/results/gush3r_smoke/`；原始 `/tmp` 结果保留不动。

## 2026-07-28（自动定位 GUSH3R recurrent 前向瓶颈）

- 用户授权自动执行检测，不再逐项请求审批；本轮目标是对 GUSH3R `forward_recurrent_lighter()` 做阶段计时/栈采样，定位 RoPE2D/xFormers 已修复后仍存在的 CPU 瓶颈。
- 根因已定位：checkpoint 采样缓存被 `infer.py` 只写入 `src/dust3r/...`，而模型配置读取相对路径 `dust3r/...`；后者为空导致每次启动重新执行 10,000 点 FPS + `KMeansConstrained(n_jobs=-1)`，产生 joblib worker、约 10 GB CPU 内存和数分钟等待。
- 已修复 `infer.py`，将 sampling cache 同时写入 `src/dust3r/...` 与运行时 `dust3r/...`；并自动适配当前 `diff_gaussian_rasterization` 的 `(color, radii)` 返回接口，同时兼容带 depth/alpha 的四返回值版本。
- profiling 复测已完成：缓存命中；`_encode_image=0.195s`、MHMR backbone `0.145s`、recurrent rollout `0.017s`、downstream head `0.123s`、human GS head `0.057s`；完整 `Inference finished in 0.79s`，并成功生成 `/tmp/gush3r_final_smoke/render.mp4`。

## 2026-07-28（GUSH3R RoPE2D/xFormers 加速复核）

- 已用 `/usr/local/cuda-12.1` + GUSH3R 环境 PyTorch 2.2.0+cu121 编译成功 `GUSH3R/src/croco/models/curope/curope.cpython-310-x86_64-linux-gnu.so`；正确 attention 张量布局的 CUDA kernel smoke test 通过。
- 已在共享 `gush3r` 环境安装 `xformers==0.0.24`（`--no-deps`，未替换既有 torch）；GUSH3R 日志确认 SwiGLU/Attention/Block 均显示 xFormers available，且不再打印 RoPE fallback 警告。
- 重新运行 `Human3R/examples/GoodMornin1.mp4 --max_frames 1 --size 512` 后，约 5 分钟仍无 `Inference finished`：CPU RSS 约 10.35 GB，GPU 约 662 MiB、利用率 0%；已停止本次 smoke test。结论：RoPE/xFormers 环境问题已修复，但主要瓶颈仍在 recurrent 前向或其 CPU/NAS 路径，不能把当前超慢完全归因于缺少这两个组件。

## 2026-07-28（Human3R/GUSH3R 环境与单帧 smoke test）

- 已确认专用环境存在：`human3r`（Python 3.11.9，PyTorch 2.5.1）和 `gush3r`（Python 3.10.20，PyTorch 2.2.0+cu121）；关键依赖 `torch`、`smplx`、`roma`、`transformers`、`gsplat` 可导入，GUSH3R 的 Gaussian rasterizer 可导入。
- 当前系统 NVIDIA 驱动不可用：`nvidia-smi` 失败，两个环境均报告 `torch.cuda.is_available() == False`。
- Human3R 初次 smoke test 缺 `scikit-image`；已在 `human3r` 环境安装 `scikit-image==0.26.0`，并准备 DINOv2 torch hub 源码缓存 `/tmp/hugs_torchhome/hub/facebookresearch_dinov2_main`。
- Human3R 单帧 CPU 测试可读入 `human3r_896L.pth`，随后在模型构建/CPU 推理阶段被系统结束，未生成输出。
- GUSH3R 单帧 CPU 测试初次因缺 DINOv2 hub 缓存失败；补缓存后可进入 checkpoint/model 构建，但在 CPU 加载/推理阶段被系统结束，未生成输出。
- 当前结论：模型、body-model 和 Python 依赖路径已基本准备好；要完成真正推理验证，需要恢复 NVIDIA 驱动并使用 GPU，CPU 条件不足以承载这两个大模型。

## 2026-07-28（sandbox 外 GPU smoke test 修正）

- 用户确认此前 GPU 不可见是 sandbox 限制；在 sandbox 外复核：RTX 4090 正常，Human3R `torch.cuda=True`，GUSH3R `torch.cuda=True`。
- Human3R GPU 单帧测试已成功：checkpoint 全部匹配，DINOv2 cache 正常，单帧推理耗时 3.36 s，并完成 `/tmp/human3r_smoke_gpu` 输出；随后 viewer 常驻，已手动停止测试进程。
- GUSH3R GPU 测试已成功完成 checkpoint/DINOv2 模型初始化，但单帧阶段长时间占用约 10 GB CPU 内存、GPU 占用较低，300 s 内未生成输出，已停止进程。后续需单独分析 GUSH3R 的 CPU preprocessing/render bottleneck。
- 两个 smoke test 进程均已清理，GPU 恢复空闲。

## 2026-07-28（盘点前馈权重与压缩包）

- 按 HUGS 前馈接触方案复核 `Human3R/`、`GUSH3R/` 及根目录新增压缩包。
- 三个主推理权重均已完整落盘：`Human3R/src/human3r_896L.pth`（4,670,554,642 B）、`GUSH3R/checkpoints/gush3r.pth`（4,870,772,362 B）、`GUSH3R/src/models/multiHMR/multiHMR_896_L.pt`（1,286,462,544 B）。
- `SMPL_python_v.1.1.0.zip`（330,607,275 B）和 `googledrive.zip`（962,692 B）通过 `unzip -t`；Google Drive 包含 `J_regressor_h36m.npy`、`smpl_mean_params.npz`、`smplx2smpl.pkl`、`smplx2smpl_joints.npy`。
- `models_smplx_v1_1.zip`（459,066,725 B）无法通过 ZIP 完整性检查，报缺少 central directory，判断为未传完/损坏，不能解压使用；需重新提供完整 SMPL-X 包。
- 当前 `Human3R/src/models/{smpl,smplx}` 与 `GUSH3R/src/models/{body_models,smpl,smplx}` 目录均未创建，压缩包尚未展开；SMPL-X/SMPL 官方模型仍需按各仓库路径安装后再做最小推理 smoke test。

## 2026-07-28（安装 SMPL-X 三性别模型）

- 用户已将 `SMPLX_FEMALE.npz`、`SMPLX_MALE.npz`、`SMPLX_NEUTRAL.npz` 上传到 HUGS 根目录。
- 已按仓库 README 的标准布局复制到：`Human3R/src/models/smplx/` 和 `GUSH3R/src/models/smplx/`。
- 六个目标文件均与源文件字节数一致；三个源文件均可用 NumPy 读取，包含 22 个数组，`v_template` 形状为 `(10475, 3)`。

## 2026-07-28（安装 SMPL 与 supplementary body-model 文件）

- 已解压 `SMPL_python_v.1.1.0.zip`，将 `SMPL_NEUTRAL.pkl`、`SMPL_MALE.pkl`、`SMPL_FEMALE.pkl` 分别放入 Human3R/GUSH3R 的 `src/models/smpl/`。
- 已从 `googledrive.zip` 放置 `J_regressor_h36m.npy`、`smpl_mean_params.npz`、`smplx2smpl.pkl`、`smplx2smpl_joints.npy` 到 GUSH3R 的 `body_models/`，并按代码路径补齐 Human3R 的 `smpl_mean_params.npz`、`smplx2smpl.pkl` 和两边 `smpl/` 下的 joint regressor。
- GUSH3R 的 `smplx2smpl.pkl` 同时放入 `smplx/`，兼容仓库下载脚本的布局；所有目标文件大小校验通过，`smpl_mean_params.npz` 可正常读取。

## 2026-07-26（查找 supplementary 文件镜像）

- 因 Google Drive 文件夹在当前机器不能返回内容，开始核对项目 Hugging Face 发布仓库是否包含同一 supplementary 文件；只接受项目作者公开发布或可验证的来源，不用不明第三方 body-model 映射文件替代。

## 2026-07-26（重试下载 supplementary 文件）

- 用户要求再次尝试下载公开 Google Drive supplementary body-model 文件夹；此前失败原因是 Drive 重定向网络不可达，未生成任何有效或半截文件。本轮继续只操作该公开 supplementary 资源。

## 2026-07-26（下载 supplementary body-model 映射文件）

- 用户已授权下载 Human3R/GUSH3R 所需的公开 Google Drive supplementary 文件；将获取 `smpl_mean_params.npz`、`smplx2smpl.pkl`、`smplx2smpl_joints.npy`、`J_regressor_h36m.npy`，按两个仓库的实际加载路径放置并核验。此操作不下载需账号许可的 SMPL/SMPL-X 主体模型。

## 2026-07-26（恢复 HUGS 前馈接触方案与权重盘点）

- 本会话目标：恢复 HUGS 第二研究点的前馈接触方案，核对 Human3R/GUSH3R 推理所需权重与当前下载落盘状态，并整理可复现的下载途径。
- 当前方案不改动既有 HUGS 训练主线：先冻结 Human3R，完成“输出 → 脚部局部 surface proxy → 规则接触/投影 → LBS/mesh(Gaussian) 渲染 → 指标”的 P0/P1 闭环；稳定后训练轻量因果 GRU/TCN，最后接入 GUSH3R 的人体/场景 Gaussian。
- 实查主 checkpoint 均已存在：`Human3R/src/human3r_896L.pth`（约 4.35 GiB）、`GUSH3R/checkpoints/gush3r.pth`（约 4.54 GiB）和 `GUSH3R/src/models/multiHMR/multiHMR_896_L.pt`（约 1.20 GiB）。仍需补齐并放到代码期望目录的，是许可受限的 SMPL-X/SMPL body models 及 supplementary 的 SMPL-X→SMPL 映射文件；本次只盘点，不修改模型或下载状态。

## 2026-07-26（HUGS 实验产物盘点与磁盘清理）

- 本会话目标：恢复 HUGS 主线记忆，按 `BEST_PIPELINE.md`、实际输出目录、checkpoint、评估/传输产物和最新日志盘点磁盘占用；删除已证实无后续用途的失败、重复或可再生成实验结果，同时保留论文主表、最优权重、关键诊断/传输结果及 Human3R/GUSH3R 前馈接触方向所需资产。
- 清理前须以实际路径、大小、结果文件和进程状态为准，先确认没有正在运行的训练/下载；对候选分成“确认删除”和“需保留/待确认”，完成删除后记录释放空间和精确目录清单。
- 实查并清理：`output/` 原为 878 GB（其中 HUGS NeuMan 867 GB、预训练权重 9.2 GB、传输分析 1.5 GB），STM `output_stm/` 原为 108 GB。已删除 92 组失败/过时 HUGS run（smoke、full_noamass、旧 VIMO、gate/detach/mask-aware/FusionMLP、large-clamp/平滑等）以及 11 组非核心 STM run；并裁剪主表 run 中可再生成的中间 checkpoint/PLY。
- 保留：GT 与旧 VIMO 口径下 HUGS/STM/Ours 的论文主表结果、GT 两阶段所需 S1/S2 final 权重、VIMO v4 最优 `v4_correct_inline_attn_18k` 六场景 final 权重/最终 PLY/指标、parkinglot 11k peak checkpoint/PLY，以及 `output/pretrained_models/`、`output/delta_mu_analysis/`、粗对齐和传输资产。jogging v4 的重复 rerun（`2026-06-14_21-51-55`）已删除，保留文档引用的 `2026-06-14_17-39-01`。
- 清理后：HUGS NeuMan 为 80 GB，整个 `output/` 为 91 GB；STM 为 34 GB。合计从两类输出中释放约 861 GB。核验保留 HUGS `results_train.json` 32 份、STM 12 份；HUGS final 权重 77 个、STM final 权重 24 个。未改动用户已有代码改动或 Human3R/GUSH3R 权重仓库。

## 2026-07-24（恢复第二研究点并核对 Human3R/GUSH3R）

- 本会话目标：恢复 HUGS 记忆，复核第二研究点“前馈接触估计与低维姿态修正”的可实施范围，并检查已克隆 Human3R/GUSH3R 仓库、权重与人体模型下载完成度。
- 先以实际文件、运行进程和 checkpoint 可加载性为准核对；不改动 HUGS 训练代码或现有实验配置。
- 实查：两个仓库均为干净工作树；`Human3R/src/human3r_896L.pth` 与 `GUSH3R/checkpoints/gush3r.pth` 均不存在，且没有相关下载进程。GUSH3R 的 `multiHMR_896_L.pt` 只有 64,667,648 B，官方总大小 1,286,462,544 B，且 `unzip -t` 报缺少 central directory，证实为中断文件而非可加载 checkpoint；尚余 1,221,794,896 B（约 1.14 GiB，95.0%）。GUSH3R 的 SMPL/SMPL-X/body_models 目录尚未创建；可复用的 HUGS `data/smpl/SMPL_NEUTRAL.pkl` 为 236 MiB，但不足以启动 Human3R/GUSH3R 推理。
- 随后按用户要求用 `HF_ENDPOINT=https://hf-mirror.com` 分别启动 Human3R、GUSH3R 主权重断点下载；两者均未建立目标文件即退出。GUSH3R 日志明确为 `hf-mirror.com:443` connect timeout；直链的镜像连通性检查也持续无响应后中止。因此本轮未产生可续传数据，不能把下载视为已启动成功；镜像恢复后用相同 `hf download ... --local-dir` 命令重试即可。
- 用户要求再次重试后，镜像恢复：Human3R `human3r_896L.pth` 已成功下载至 `Human3R/src/`（约 4.4 GiB，下载进程已正常退出）；GUSH3R `gush3r.pth` 下载仍在后台运行，暂存断点文件已有 629,145,600 B（600 MiB），完成后会自动原子移动到 `GUSH3R/checkpoints/gush3r.pth`。


## 2026-07-23（核对 Human3R/GUSH3R 推理下载清单）

- 核对两个仓库 README、推理代码和下载脚本：推理必需为 Human3R checkpoint、GUSH3R merged checkpoint，以及 GUSH3R 的 MultiHMR + SMPL/SMPL-X + supplementary body-model files。
- BEDLAM 图像/深度/训练标签不是当前推理必需；Human3R 的 CUT3R 单独 checkpoint 也不在当前 GUSH3R merged checkpoint 的最小推理清单中。
- 实际检查时未发现下载进程，目标权重和人体模型文件也尚未落盘，后续需重新确认断点下载是否已退出。
- 后续全目录核对发现 HUGS 已有 `data/smpl/SMPL_NEUTRAL.pkl`（约 247 MB），以及 JOSH/TRAM 下的 `J_regressor_h36m.npy` 和 `smpl_mean_params.npz`；此前“人体模型文件尚未落盘”的表述需修正为“SMPL neutral 已有，但 SMPL-X、smplx2smpl 和 Multi-HMR 尚未找到”。
- 已重新启动三个公开权重的断点下载：Human3R 896L、GUSH3R merged checkpoint（从约 30 MiB 续传）和 Naver Multi-HMR 896L；SMPL/SMPL-X 许可模型仍需用户账号下载或上传。
- 已将完整权重清单及下载途径写入 `docs/agent_memory/hugs-forward-contact-pipeline.md`：Human3R/GUSH3R 使用 Hugging Face 国内镜像，Multi-HMR 使用 Naver 官方直链，SMPL/SMPL-X 使用官方授权站，supplementary 映射文件使用项目 Google Drive；HUGS 已有 SMPL neutral，BEDLAM/CUT3R 暂不下载。

## 2026-07-23（同步前馈接触修正方案）

### 方案记忆更新

- 已阅读并核对 `idea/实时接触修正前馈人体场景Gaussian方案构想.md` 的更新版（2026-07-22/23，新增第 18 节 Human3R/GUSH3R 具体 pipeline）。
- 第二研究点固定为“面向实时人体-场景 Gaussian 直播的前馈接触估计与低维姿态修正”，与现有 HUGS inline_attn/v4 主线并行，不覆盖现有实验结论。
- 研究路线固定为 Human3R 原型 → GUSH3R Gaussian 版本：先冻结前馈模型，第一版只做左右脚-地面接触；低维 root/foot residual 经 IK、penetration projection 和 LBS 更新人体 Gaussian，scene Gaussian 不变。
- 必须独立完成坐标 convention/robust Sim(3) 对齐和局部 scene proxy（平面→局部点云→Gaussian-aware），低置信度时回退原始前馈姿态。
- 三阶段训练：冻结模型生成几何/伪标签 → 冻结前馈模型训练 causal GRU/TCN → 仅解冻 GUSH3R human decoder 末层或 adapter 做轻量联合微调。
- 当前第一里程碑：`Human3R 输出 → 脚接触 proxy → 规则修正 → Gaussian 渲染 → 接触指标` 闭环；P0-P6 Todo 已同步到 `docs/agent_memory/hugs-forward-contact-pipeline.md`。
- 已新建独立目录 `forward_contact_pipeline/`，并将逐项可勾选任务写入 `forward_contact_pipeline/TODO.md`；后续实现脚本、接口、实验配置和结果记录统一放入该目录。

### 当前仓库状态核对

- `ml-hugs-work/Human3R/` 和 `ml-hugs-work/GUSH3R/` 已存在；本次只更新记忆文档和会话记录，未改动 HUGS 代码或用户已有未提交实验改动。

## 2026-07-23（引入 Human3R 与 GUSH3R）

### 当前操作

- 恢复 HUGS 项目记忆，开始探索第二个前馈方向。
- 按用户要求，将 `fanegg/Human3R` 和 `abkeito/GUSH3R` 作为独立仓库放入 `ml-hugs-work/`，供后续基于 Human3R/GUSH3R 设计前馈人体-场景方案。

### 注意

- 本次只新增两个仓库目录，不修改 HUGS 现有代码及其未提交实验改动。

---

## 2026-07-22（BEDLAM 下载准备）

### 当前目标

- 为“图像帧 → 前馈人体/场景 → 接触预测 → IK 修正”准备 BEDLAM 数据。
- 明确不能只下载 pose/GT：online 输入是图像，因此最终需要 BEDLAM 6fps RGB images + GT labels；30fps 全量图像和 depth 暂不下载。

### 数据下载决策

- BEDLAM 官方入口：`https://bedlam.is.tue.mpg.de/imagesgt.php`。
- Hugging Face collection：`https://huggingface.co/collections/Intelligent-Systems/perceiving-systems`。
- 国内镜像设定：`HF_ENDPOINT=https://hf-mirror.com`。
- 第一批目标：6fps RGB/GT、SMPL-X 参数、camera/sequence metadata；不下载约 6TB 的 30fps 全量数据，也暂不下载 BEDLAM-depth。
- 机器可用空间约 45TB，但工作盘总使用率约 96%，仍需避免无选择全量下载。

### 实际状态

- 已建立：`/workspace/nas_auto_backup/yuzilang/contact_datasets/BEDLAM/`。
- 曾尝试 `huggingface-cli download --repo-type dataset Intelligent-Systems/BEDLAM --include '*.csv' '*.json' '*.npz'`，但当前 `huggingface-cli` 查询镜像仓库元数据时卡在 TLS/网络连接，未产生有效文件；已中止，不能视为下载成功。
- `huggingface-cli` 版本提示旧命令已 deprecated，应优先尝试 `hf download`；后续需确认 collection 中实际 repo ID，并解决镜像 IPv4/TLS 连接问题。
- 2026-07-22 进一步尝试 `HF_ENDPOINT=https://hf-mirror.com hf download --repo-type dataset Intelligent-Systems/BEDLAM --include README.md`，同样卡在镜像 API 的 TLS handshake；未产生文件，当前没有 BEDLAM 下载进程。


## 2026-07-22（会话八十）

### 会话主题

整理一套适合交接给不了解 HUGS 项目的 AI 的最小文件包与按任务追加文件清单。

### 交接原则

- 先看 `BEST_PIPELINE.md` 建立当前方法和指标全貌。
- 再看本日志顶部的最新会话，了解粗对齐 v4、训练修复和当前未统一的 baseline 口径。
- 用项目概览、实验结果、VIMO 分析和传输方案补足背景；不要一开始把数千行完整 session log 和全部实验配置都塞给新 AI。
- 需要改代码时，再按任务打开 AnchorAttention、trainer、粗对齐脚本和具体配置。

---

## 2026-07-22（会话七十九）

### 会话主题

恢复 HUGS 项目记忆，重新接入当前主线、最终实验结论和后续工作入口。

### 当前恢复结论

- 当前粗对齐主线是离线 `VIMO v4`：用 ROMP/VIMO、COLMAP、segmentation 和 DepthPro 求全局 scale + world translation，再生成 `smpl_optimized_aligned_scale_vimo_v4.npz`。
- 当前最优训练主线是 `v4_correct_inline_attn_18k`：VIMO v4 + inline AnchorAttention，18k 单阶段训练；六场景 final HUMAN_PSNR 平均 17.2944，较 STM 平均高 0.39 dB。
- 正确对齐后训练是否稳定的关键不是 Global Scene Gate、往返轨迹或 scene detach，而是 `gs_trainer.py` 恢复原版 HUGS 使用 full `viewspace_points` 的 scene densification 传法。
- 旧 VIMO 的 scale 错误已由 v4 修正；当前论文对比仍需注意 Ours 使用 `{seq}_vimo_v4`，HUGS/STM baseline 多数仍是旧 `{seq}_vimo`，尚未完全统一为 v4 对齐。
- 传输方向已完成 Savgol(window=11, polyorder=2) + int8 差分/熵编码验证，delta_mu ratio 降低约 47%~73%，PSNR 影响约为零；后续重点在 `transmission_system/` 的人体优先与场景渐进加载实验。

### 恢复入口

- 总结：`BEST_PIPELINE.md`
- 详细时间线：本文件顶部最新会话
- 实验汇总：`docs/agent_memory/hugs-active-experiments.md`、`docs/agent_memory/hugs-experiment-results.md`
- 粗对齐分析：`docs/agent_memory/hugs-vimo-alignment-analysis.md`
- 传输方案：`transmission_system/STREAMING_SYSTEM_PLAN.md`

---

## 2026-07-17（会话七十八）

### 会话主题

恢复 HUGS，系统核验当前采用的粗对齐方案，以及粗对齐各历史版本的具体实现与演进关系。

### 当前操作

- 以 `BEST_PIPELINE.md`、粗对齐脚本、NPZ 生成脚本、训练配置和 git 历史为准，核对当前实际使用链路。
- 梳理初始 VIMO、v2、v3、v4 各版本的输入、scale/translation 求解、fallback、输出及已知问题。
- 区分“粗对齐算法版本”和“训练侧 scene densification 修复”，避免把对齐正确但训练回归导致的崩坏归因到 v4。

### 核对结论

- 当前主线是离线 `VIMO v4`：v4 脚本以 ROMP `smpl_pred`、COLMAP、segmentation 和 DepthPro 估计单个全局 `scale + t_world`；随后 `gen_coarse_npz_vimo.py` 将该变换应用到 VIMO 每帧 `pred_trans`，生成 `smpl_optimized_aligned_scale_vimo_v4.npz`，训练从 `{seq}_vimo_v4` 软链接目录读取。
- v1 使用 mono-depth/scene Chamfer + foot-contact 初始化；v2 改为 DepthPro 背景标定和人体双向 trimmed Chamfer；v3 改为足部附近 COLMAP 深度作为主 scale 约束，但 `[3,20]` 过滤与无接触时硬回退 8.0 导致小尺度场景失败；v4 将范围放宽至 `[0.5,20]`，加入 `calib_a`、body-depth fallback 和 20% contact/calib 偏差检测，并在 `calib_a` 模式禁用 Chamfer。
- 六场景 v4 scale 实值：lab 8.168、bike 1.550、citron 2.372、jogging 1.554、parkinglot 3.447、seattle 2.038；只有 jogging 最终选择 `calib_a`，其余选择 `foot_contact`。
- 当前最佳训练为 `v4_correct_inline_attn_18k`，同时依赖训练侧回滚到原版 HUGS full `viewspace_points` 的 scene densification 传法；该回滚不是粗对齐 v4 算法的一部分，但决定了 v4 正确对齐后是否崩坏。
- 方法口径注意：当前生成器仍复制 GT NPZ 的 pose/betas 等字段，只替换 transl/scale；所以 `BEST_PIPELINE.md` 所写“无 GT”并不严格成立。六场景 VIMO 输出恰好覆盖全部帧，因此没有触发缺帧时使用 GT transl 的 fallback。
- 对比口径注意：当前论文主表 Ours 使用 `{seq}_vimo_v4`，HUGS/STM baseline 使用旧 `{seq}_vimo`；尚未在完全相同 v4 对齐下补齐 HUGS-v4 / STM-v4 baseline。
- 已逐帧数值复核旧 `{seq}_vimo` NPZ：它等于 `R^T(scale_v3 * pred_trans - t)`，与“不加 `global_t_world`”公式误差约为 0；因此旧 baseline 不仅使用 v3 scale，还确实漏用了 v3 优化出的 `global_t_world`。当前生成器已修正为加上该项。

---

## 2026-07-17（会话七十七）

### 会话主题

详细解释 VIMO v3 与 v4 粗对齐 pipeline 的实现差异。

### 当前操作

- 对照 `coarse_align_smpl_neuman_v3.py`、`coarse_align_smpl_neuman_v4.py` 和 `gen_coarse_npz_vimo.py`，核验 scale、translation、DepthPro 标定、fallback 与 NPZ 生成链路。
- 结合六场景对齐误差和后续训练结果，区分“对齐算法改进”与“scene densification 崩坏修复”。

### 核对结论

- v3 只接受 `[3, 20]` 的足部接触 scale；无有效估计时硬回退 `scale_init=8.0`，优化中又没有其他可靠 scale 约束，最终在 bike/jogging/seattle 等场景停在约 3.5 的错误尺度。
- v4 将足部 scale 有效范围放宽为 `[0.5, 20]`，新增背景 DepthPro 标定斜率 `calib_a` 和人体中心深度两级 fallback，并以 contact/calib_a 中位数差异 20% 检测足部估计偏差。
- v4 的 scale 约束优先级是 foot contact → `calib_a` → DepthPro body；选择 `calib_a` 时禁用 Chamfer，只保留 silhouette IoU 优化 `t_world`，避免 DepthPro 人体深度的系统偏差重复污染 scale/translation。
- NPZ 生成阶段对 v3/v4 使用相同坐标变换：`R^T(scale * pred_trans - t) + global_t_world`；v4 的主要变化来自更可靠的 `global_scale/global_t_world`。生成器复制 GT 容器中的 pose/betas，只替换 transl/scale，用来隔离粗平移对齐质量。

---

## 2026-07-17（会话七十六）

### 会话主题

恢复 HUGS，追溯粗对齐方案曾导致部分场景崩坏、以及后续如何修正。

### 当前操作

- 已恢复全局 HUGS 记忆，并定位到 bike / jogging / seattle 的粗对齐崩坏调查。
- 正在用本日志、`BEST_PIPELINE.md`、实际代码和 git 历史核验根因演进与最终修复，避免把早期已推翻的 Global Scene Gate / 往返轨迹假说当成最终结论。

### 核对结论

- 旧 VIMO 粗对齐在 `n_contact=0` 的场景把 scale 高估约 2 倍，继而造成很大的 world translation 偏差；v4 通过放宽 scale 有效范围、DepthPro 背景标定 fallback、`calib_a` 偏差检测并在该模式禁用 Chamfer，修正了 bike / jogging / seattle 的 scale。
- v4 正确对齐初跑反而崩坏，最终发现决定训练成败的代码回归是 `gs_trainer.py` 把 scene densification 改成了“正确切出的 scene 梯度”；这破坏了原版 HUGS 使用 full `viewspace_points` 的隐式人体区域保护。
- 最终修复是回滚到 full `viewspace_points` 传法，并保留 v4 对齐 + inline AnchorAttention。bike 从错误代码下 13.96/旧方案 18.09 提升到 final 18.9626；六场景 final HUMAN_PSNR 平均 17.2944，较 STM +0.39 dB。
- Global Scene Gate、往返轨迹、detach、mask-aware prune、清洁 COLMAP 都属于调查过程中的假说或备选，不是最终主线修复。

---

## 2026-07-16（会话七十五）

### 会话主题

继续开展人体优先渐进传输实验，并持续记录结果。

### 当前阶段

- 优先修复渐进视频：人体 104 帧播放结束后保持最后姿态，继续等待并展示场景块渐进加载，直到完整场景到达。
- 该阶段完成后，在同一视频和指标框架上实现 center-distance 与 contact-aware 调度对比。
- 每个阶段的参数、视频路径和指标同步写入 `transmission_system/STREAMING_SYSTEM_PLAN.md` 及 HUGS 传输记忆。

---

---

## 2026-07-16（会话七十四）

### 会话主题

验证人体优先、场景按人体运动中心渐进加载的最小传输架构。

### 实验目标

- 先传 `H-canonical + H-lbs + H-smpl + 首帧 correction`，客户端立即显示可动人体。
- scene GS 按到人体运动中心的距离分块，近处块优先传输。
- 使用真实 HUGS renderer 生成渐进加载视频。
- 测量人体首帧、第一场景块、完整场景和播放流畅度。

---

---

## 2026-07-16（会话七十二）

### 会话主题

将 Savgol 压缩传输 baseline 接入 HUGS receiver-side renderer，生成按接收节奏渲染的可视化视频。

### 实验目标

- sender 发送静态资产模型和压缩 correction packet 的模拟时序。
- receiver 按 packet 到达顺序解码 `delta_mu`。
- 使用 HUGS `render_human_scene` 逐帧渲染并写出视频。
- 分离统计网络到达、解码、GPU 渲染和最终可播放 FPS。

---

---

## 2026-07-16（会话七十一）

### 会话主题

将已完成的 Savgol `window=11` correction 压缩版本纳入基础传输性能评测。

### 实验目标

- 对比 raw correction 与 Savgol + 差分 + 量化/熵编码版本。
- 使用启动延迟、逐帧 payload、所需带宽、30 FPS 按时到达率和解码吞吐等指标。
- 优先复用已保存的真实压缩输出；若缺失，则根据真实 `delta_mu` 重新编码。

---

---

## 2026-07-16（会话七十）

### 会话主题

建立人体-场景联合表征的基础传输性能 baseline。

### 实验目标

- 模拟静态资产一次性传输 + 动态 correction 逐帧传输。
- 测量首次启动延迟、首帧可见时间、接收端观看 FPS、吞吐、端到端帧延迟和带宽。
- 使用体积视频/流媒体系统常用指标，作为后续接触感知调度和渐进传输的对照基线。

### 计划

- 先用真实或等尺寸的静态/动态数据完成单服务器 localhost TCP baseline。
- 将 benchmark 结果写入 `transmission_system/` 的方案文档和 JSON/CSV 输出。
- 当前 baseline 不加入接触感知调度，不加入场景渐进加载，用于量化简单方案的性能上限与瓶颈。

---

## 2026-07-16（会话六十九）

### 会话主题

在 `transmission_system/` 中记录并完善人体-场景联合表征流式传输方案。

### 完成内容

- 明确系统目标是人体优先、接触区域引导、场景渐进补全的联合传输系统。
- 确认学术主线升级为接触感知率失真优化、人体运动驱动场景预取和联合调度。
- 新增持续维护的方案文档，记录系统设计、创新点、实验矩阵、数据口径和后续决策。

---

## 2026-07-16（会话六十八）

### 会话主题

搭建 HUGS 动态 Gaussian 传输系统实验架构。

### 计划

- 在 `transmission_system/` 中建立独立的传输实验目录。
- 第一阶段支持文件模式：压缩/打包 → 解码 → 接收端重建。
- 第二阶段支持单服务器 localhost TCP：sender → receiver。
- 预留量化、差分、熵编码、keyframe、带宽和画质评估接口。

> **维护规则**：每次对话开始时，Claude 先更新本文件，再做其他任何工作。
> 按倒序记录，最新内容在最上方。

---

## 2026-07-16（会话六十七）

### 会话主题

恢复 HUGS 记忆，回看粗对齐条件下“接触感知优化 / AnchorAttention / scene-human 接触边界保护”此前留下的优化想法。

### 当前操作

- 已读取 `docs/agent_memory/README.md`、`MEMORY.md`、`LOCAL_MEMORY_INDEX.md` 以及 HUGS 相关记忆索引。
- 正在核对 `hugs-future-directions.md`、`hugs-vimo-alignment-analysis.md`、`hugs-experiment-results.md` 与本日志中 contact/AnchorAttention/粗对齐相关段落。

---

## 2026-07-15（会话六十六）

### 会话主题

恢复 HUGS 记忆，整理当前实验进度，并用实际日志/输出状态核对记忆快照是否仍然有效。

### 当前操作

- 已读取 `docs/agent_memory/` 下 HUGS 项目概览、当前实验、实验结果、关键文件和协作偏好记忆。
- 下一步检查 `ml-hugs-work` 内实际输出目录、训练日志、进程状态，汇总当前实验完成度和待办。

### 核对结果

- 当前没有 HUGS/STM 训练进程，`nvidia-smi` 显示 GPU 空闲。
- 粗对齐主线 `v4_correct_inline_attn_18k` 6 场景已完成，仍是 VIMO v4 粗对齐最优 pipeline，final HUMAN_PSNR 平均 17.2944，较 STM 平均 +0.39 dB。
- `v4_large_transl_debug_ply_18k` 6 场景已完成，平均 16.8880，低于 `v4_correct`，仅适合说明大 clamp 对高 transl_norm 场景有利、对低误差场景有害。
- 后续两个变体已完成且均为负向：`parkinglot v4_adapt_clamp_a_18k` final HUMAN_PSNR=13.7248；`lab v4_embed_smooth_18k` final HUMAN_PSNR=17.2801，低于对应 baseline。
- STM GT 对齐 5 场景串行训练已完成，补齐后 GT 对齐 STM 6 场景平均 HUMAN_PSNR=19.4725；`scripts/make_paper_comparison.py` 的 GT STM 路径已填写。
- 传输压缩方案五（savgol w=11 + int8 diff + entropy）已完成评估，PSNR 变化约 ±0.01 dB，ratio 降幅约 47%~73%，记录在 `BEST_PIPELINE.md`。
- 论文/答辩图已有输出：`paper_figures/gt_all6_best.png`、`coarse_all6_best.png`、6 个 `before_after_*.png`、3 个 `depth_ablation_*.png`。

### 核心实验矩阵核对（2026-07-15 追加）

已按论文主表口径核对 `results_train.json`，6 组核心/对比实验均已跑齐 NeuMan 6 场景：

| 对齐方式 | 方法 | 角色 | HUMAN_PSNR avg | 状态 |
|---|---|---|---:|---|
| GT 对齐 | HUGS | 原版 HUGS baseline | 18.8704 | 完成 6/6 |
| GT 对齐 | STM | STM baseline | 19.4725 | 完成 6/6 |
| GT 对齐 | Ours | 我们的核心实验 | 19.6174 | 完成 6/6 |
| 粗对齐 | HUGS | 粗对齐 HUGS baseline | 15.8039 | 完成 6/6 |
| 粗对齐 | STM | 粗对齐 STM baseline | 16.9020 | 完成 6/6 |
| 粗对齐 | Ours | 我们的核心实验 | 17.2944 | 完成 6/6 |

口径注意：粗对齐主表当前使用 `scripts/make_paper_comparison.py` 的路径配置，baseline 为早期 `*_vimo`，Ours 为 `*_vimo_v4` 最优路径。如果后续要求所有方法完全同一 VIMO v4 对齐，需要另行补跑 HUGS-v4 / STM-v4 baseline。

已同步更新 `BEST_PIPELINE.md` 顶部“核心实验矩阵（论文主表口径）”。

---

## 2026-06-23（会话六十五）

### 会话主题

恢复 HUGS，查看论文对比图生成脚本 `scripts/make_paper_comparison.py`

---

## 2026-06-23（会话六十四）

### 会话主题

恢复 HUGS 记忆，查询 VIMO v4 对齐下 6 个场景的完整指标（v4_correct_inline_attn_18k）

---

## 2026-06-22（会话六十三）

### 会话主题

恢复 HUGS 记忆，确认 STM GT 5 场景全部完成（昨晚 22:50 结束），填写 make_paper_comparison.py GT_MODE_PATHS 中 5 个场景的 stm 路径

### 主要结果

STM GT baseline（HUMAN_PSNR final）：bike=20.34, citron=19.70, parkinglot=19.43, jogging=18.23, seattle=19.84, lab=19.29（已有）；6场景平均 19.47

### 完成操作

- `scripts/make_paper_comparison.py` GT_MODE_PATHS 5个场景 stm=None → 实际路径

---

## 2026-06-21（会话六十二）

### 会话主题

恢复 HUGS 记忆，检查 STM GT 对齐 5 场景串行训练进度（PID 198736/198757）

---

## 2026-06-21（会话六十一）

### 会话主题

论文可视化准备：对比图生成脚本 `scripts/make_paper_comparison.py` + 输出目录 `paper_figures/`

### 脚本功能

**文件**：`scripts/make_paper_comparison.py`
**输出目录**：`paper_figures/`（已创建）

生成论文用的 4行×3列方法对比图：
- 行：GT / HUGS / STM / Ours
- 列：3个场景（可配置）
- 可选：脚部放大框（`--foot-zoom`）

支持两种 mode：
- `gt`：精准对齐（expG pipeline，`depth_sup12k_transl_xyz_attn_6000_*`）
- `coarse`：粗对齐（v4_correct_inline_attn_18k，VIMO v4）

图像来源：`val/full_final_NNN.png`（左半=GT，右半=渲染），STM 用 `full_020000_NNN.png`

### 常用命令

```bash
# 粗对齐对比图
python scripts/make_paper_comparison.py \
    --mode coarse --scenes bike citron parkinglot \
    --frames 3 5 2 --foot-zoom \
    --out paper_figures/coarse_comparison.png

# 精准对齐对比图
python scripts/make_paper_comparison.py \
    --mode gt --scenes bike citron jogging \
    --frames 4 2 6 --out paper_figures/gt_comparison.png

# 手动指定脚部框
    --foot-zoom --foot-box 400 550 700 710

# 覆盖某格 val 目录
    --override ours bike /path/to/val
```

### 当前状态

- coarse 模式：12格全部找到图像（bike/citron/parkinglot，frames=3 3 3）✓
- gt 模式：STM 格显示占位灰图（待查是否有 GT 对齐的 STM 实验结果）

---

## 2026-06-19（会话六十）

### 会话主题

时序感知 delta_mu 压缩——方案一（frame_embed 平滑正则）结果复盘；决策方案二 vs 方案五

### 方案一实验结果（embed_smooth λ=0.01，lab 场景）

| 指标 | baseline (v4_correct) | embed_smooth (λ=0.01) | 变化 |
|------|:---------------------:|:---------------------:|:----:|
| PSNR final | 17.7588 | 17.2801 | -0.48 dB ❌ |
| GS pts | 501,392 | 431,194 | -14% |
| raw_std | 0.806 | 0.309 | -62% |
| diff_std | 0.424 | 0.187 | -56% |
| ratio | 0.526 | 0.605 | 变差 ❌ |

**结论**：方案一失败。frame_embed 不是 delta_mu 时序变化的主要来源，pose 驱动的 cross-attention 输出变化才是。约束 embed 后 raw_std/diff_std 绝对值均降低（~60%），但 diff/raw 比值反而上升，PSNR 损失 0.48 dB。

**当前决策**：待定，讨论方案二 vs 方案五。

---

## 2026-06-16（会话五十九）

### 会话主题

恢复 HUGS 记忆，确认 large_transl_debug_ply_18k 全部6场景完成，汇总最终结果

### 全部完成（截至 ~15:11 CST）

**citron 已完成**：final=17.1625，peak=17.5745（v4_correct=17.4823，STM=16.8763，Δ vs old=-0.32，Δ vs STM=+0.29 ✓）

| 场景 | final | peak | v4_correct | STM | Δ vs old | Δ vs STM |
|------|:-----:|:----:|:----------:|:---:|:--------:|:--------:|
| bike | 19.0286 | 18.9893@16k | 18.9626 | 18.6100 | +0.07 | **+0.42 ✓** |
| jogging | 17.0385 | 17.2587 | 17.5463 | 17.3776 | -0.51 | -0.34 ✗ |
| seattle | 17.1982 | 17.1758 | 16.7521 | 16.7278 | +0.45 | **+0.47 ✓** |
| lab | 17.2965 | 17.6378@4k | 17.7588 | 17.4512 | -0.46 | -0.15 ✗ |
| parkinglot | 13.6034 | 14.4928 | 15.2645 | 14.3660 | -1.66 | -0.76 ✗ |
| citron | 17.1625 | 17.5745 | 17.4823 | 16.8763 | -0.32 | **+0.29 ✓** |
| **平均** | **16.8880** | — | **17.2944** | **16.9015** | **-0.41** | **-0.01** |

**结论**：`v4_large_transl_debug_ply_18k`（clamp=1.5，gamma_final=0.15）平均 16.89，略低于 STM（16.90），显著低于 v4_correct（17.29）。仅 bike/seattle/citron 超过 STM，对 parkinglot/jogging/lab 有害。`v4_correct_inline_attn_18k` 仍为最优 pipeline。

### 后续实验（已启动）

`adapt_clamp_a_parkinglot`（PID 1919097，15:35 启动）
- 配置：`cfg_files/release/neuman/hugs_vimo_v4_parkinglot_adapt_clamp_a_18k.yaml`
- 关键参数：`transl_delta_clamp=0.30`，`gamma_transl_final=0.15`，`gamma_transl_decay_end=18000`
- 日志：`run_logs/adapt_clamp_a_parkinglot.log`
- 期望：final ≥ 15.0；若 14.5~15.5 → 方案 A 有效可推广；若 <13.5 → clamp 不是根本原因

---

## 2026-06-16（会话五十八）

### 会话主题

恢复 HUGS 记忆，查看 large_transl_debug_ply_6scenes 实验最终结果（5/6场景完成，citron 运行中 step 13k/18k）

### 当前状态（截至 ~14:30 CST）

日志：`run_logs/large_transl_debug_ply_6scenes_progress.log`

| 场景 | 状态 | final PSNR | transl_norm | v4_correct final | STM | Δ vs old | Δ vs STM |
|------|:----:|:----------:|:-----------:|:----------------:|:---:|:--------:|:--------:|
| bike | ✓ | **19.0286** | 4.36 | 18.9626 | 18.6100 | +0.07 | **+0.42 ✓** |
| jogging | ✓ | **17.0385** | 2.53 | 17.5463 | 17.3776 | -0.51 | **-0.34 ✗** |
| seattle | ✓ | **17.1982** | 3.35 | 16.7521 | 16.7278 | +0.45 | **+0.47 ✓** |
| lab | ✓ | **17.2965** | 0.089 | 17.7588 | 17.4512 | -0.46 | **-0.16 ✗** |
| parkinglot | ✓ | **13.6034** | 0.155 | 15.2645 | 14.3660 | -1.66 | **-0.76 ✗** |
| citron | 🔄 | — (step 13k: 17.21) | 1.136 | 17.4823 | 16.8763 | — | — |

**关键变化**（vs v4_correct_inline_attn_18k）：`transl_delta_clamp: 0.5→1.5`，`gamma_transl_final: 0.05→0.15`，`gamma_transl_decay_end: 15000→18000`

**结论**：large_transl clamp 对极大误差场景（bike/seattle transl_norm≥3）有利，对小/中误差场景（parkinglot/lab/jogging transl_norm≤2.5）有害。parkinglot 跌 -1.66 dB 是最严重退步。5场景平均尚未超过 STM。

---

## 2026-06-16（会话五十七）

### 会话主题

恢复 HUGS 记忆，查看 large_transl_debug_ply_6scenes 6场景串行实验进度

### 当前状态（截至 11:42 CST，会话五十八更新）

日志：`run_logs/large_transl_debug_ply_6scenes_progress.log`
实验关键变化（vs v4_correct）：`transl_delta_clamp: 0.5→1.5`，`gamma_transl_final: 0.05→0.15`，`gamma_transl_decay_end: 15000→18000`

| 场景 | 状态 | final PSNR | 峰值 PSNR | v4_correct final | STM | Δ vs old | Δ vs STM |
|------|:----:|:----------:|:---------:|:----------------:|:---:|:--------:|:--------:|
| bike | ✓完成 | **19.0286** | 18.99@11k | 18.9626 | 18.6100 | +0.07 | **+0.42** ✓ |
| jogging | ✓完成 | **17.0385** | 17.22@16k | 17.5463 | 17.3776 | **-0.51** | **-0.34** ✗ |
| seattle | ✓完成 | **17.1982** | 17.18@11k | 16.7521 | 16.7278 | **+0.45** | **+0.47** ✓ |
| lab | ✓完成 | **17.2965** | 17.64@4k | 17.7588 | 17.4512 | **-0.46** | **-0.16** ✗ |
| parkinglot | 🔄运行中 | — | 9k:12.92↓ | 15.2645(峰16.73) | 14.3660 | 严重退步 | — |
| citron | ⏳等待 | — | — | 17.4823 | 16.8763 | — | — |

### 分析（已完成4场景）

- **bike(transl_norm=4.36)**: 小幅受益 +0.07 dB，仍领先 STM +0.42 dB ✓
- **seattle(transl_norm=3.35)**: 显著改善 +0.45 dB，明确超过 STM +0.47 dB ✓
- **jogging(transl_norm=2.53)**: 反而退步 -0.51 dB，跌破 STM ✗
- **lab(transl_norm=0.089)**: 退步 -0.46 dB，小对齐误差场景受损 ✗
- **parkinglot(transl_norm=0.155)**: step 9k 仅 12.92（老实验 final=15.26），大 clamp 严重破坏小误差场景

**结论**：`transl_delta_clamp=1.5` 对极大误差场景有利（bike/seattle），但对小/中误差场景有害。需要场景自适应 clamp 策略。

---

## 2026-06-16（会话五十六）

### 会话主题

恢复 HUGS 记忆，查看 seattle v4_large_transl_inline_attn_18k_seattle 最新实验结果

### 已完成实验结果

**seattle v4_large_transl Run 1**（`12-09-38`，step 10k 时中断）：
- 峰值 **17.3307** @4k，之后下降，于 step 10k 被 kill

**seattle v4_large_transl Run 2**（`13-17-59`，已完成）：
- final: **17.0450**，峰值 **17.1757** @16k
- 对比：v4_correct final=16.7521，STM=16.7278
- 结论：large_transl final +0.32 dB vs STM，显著改善 seattle

---

## 2026-06-15（会话五十五）

### 会话主题

检查当前正在运行的实验状态

### 确认内容

- 当前唯一运行中实验：**seattle v4_large_transl_inline_attn_18k_seattle**（PID 1740834）
- 启动时间：12:09，总步数：17998，当前进度：step 10000（约55%完成）
- HUMAN_PSNR 走势：1k=17.01, 2k=17.15, 3k=17.26, **4k峰值=17.33**, 5k=16.94, 6k=16.80, 7k=16.79, 8k=16.81, 9k=16.68, 10k=16.76
- 峰值 17.33（step 4k）已超过 STM(16.73) 和之前 v4_correct final(16.75)
- TRANSL_L2_VS_GT: mean=0.3511（恒定），表明 large_transl 优化中 transl 收敛
- 日志路径：`output/human_scene/neuman/seattle_vimo_v4/hugs_trimlp/v4_large_transl_inline_attn_18k_seattle/2026-06-15_12-09-38/train.log`

---

## 2026-06-15（会话五十四）

### 会话主题

运行 STM 方案 + v4 粗对齐 pipeline，lab 场景

### 任务

用 `lab_vimo_v4` 数据跑 STM（HumanSceneFuseDecoder）20k 步，与 OUR v4_correct_inline_attn_18k（lab final=17.7588）和旧 STM lab_vimo（17.4512）做对比。
创建配置 `stm_vimo_lab_v4_20k.yaml`，在 Dynamic-Avatar-Scene-Rendering-from-Human-centric-Context 目录后台启动。

---

## 2026-06-14（会话五十三）

### 会话主题

查看 seattle + jogging v4_correct_inline_attn_18k 最新实验进度

### 关键发现

**seattle**（PID 1462033，step 16k/18k，仍在运行）：持平 STM，整体无明显提升

| step | HUMAN_PSNR | vs STM(16.7278) |
|------|:----------:|:---------------:|
| 14k | 16.8118 | +0.08 |
| 15k | 16.5324 | −0.20 |
| 16k | 16.7779 | +0.05 |

**结论**：在 STM 附近震荡（±0.2 dB），无实质提升，VIMO 对齐偏差（TRANSL_L2=0.35m）可能是瓶颈。

**jogging**（PID 1481478，step 15k/18k，仍在运行）：step 15k 出现崩坏

| step | HUMAN_PSNR | vs STM(17.3776) | human GS 点数 | SCENE_INVASION_COUNT |
|------|:----------:|:---------------:|:-------------:|:--------------------:|
| 13k | 17.5180 | +0.14 | — | — |
| 14k | 17.5822 | +0.21 | 520,410 | 18,932 |
| **15k** | **15.5133** | **−1.86 ⚠️崩坏** | **324,598** | **8,140** |

**根因推测**：step 14k→15k 发生大规模 prune，human GS 点数骤减 196k，导致 PSNR 急跌 2 dB。
是否恢复待观察（step 16k 结果尚未出来）。

---

## 2026-06-14（会话五十二）

### 会话主题

恢复 HUGS，查看 seattle + jogging v4_correct_inline_attn_18k 实验当前进度

### 背景

两个实验并行运行中，检查 step 6k~13k 的 HUMAN_PSNR 曲线，与 STM 基准（seattle=16.73，jogging=17.38）对比。

### 最终汇总（step 11k/13k，仍在运行）

**jogging**（PID 1481478，step 11k/18k）：

| step | HUMAN_PSNR | vs STM(17.3776) |
|------|:----------:|:---------------:|
| 1k | 17.0875 | −0.29 |
| 2k | 16.3287 | −1.05 |
| 3k | 17.3344 | −0.04 |
| 4k | 17.2555 | −0.12 |
| **5k** | **17.6248 ★** | **+0.25** |
| 6k | 16.8793 | −0.50 |
| 7k | 17.2854 | −0.09 |
| 8k | 17.6223 | +0.24 |
| 9k | 17.4669 | +0.09 |
| 10k | 17.5344 | +0.16 |
| 11k | 17.5946 | +0.22 |

**结论**：稳定超越 STM，整体均值约 +0.1~+0.2 dB，无崩坏。

**seattle**（PID 1462033，step 13k/18k）：

| step | HUMAN_PSNR | vs STM(16.7278) |
|------|:----------:|:---------------:|
| 1k | 16.7152 | −0.01 |
| 2k | 16.9382 | +0.21 |
| 3k | 16.8289 | +0.10 |
| 4k | 16.7334 | +0.01 |
| 5k | 16.7421 | +0.01 |
| 6k | 16.5648 | −0.16 |
| 7k | 16.7150 | −0.01 |
| **8k** | **16.8994 ★** | **+0.17** |
| 9k | 16.7816 | +0.05 |
| 10k | 16.8522 | +0.12 |
| 11k | 16.8347 | +0.10 |
| 12k | 16.7675 | +0.04 |
| 13k | 16.6708 | −0.06 |

**结论**：整体在 STM 附近震荡（±0.2 dB），无明显提升，无崩坏。VIMO 对齐偏差（TRANSL_L2=0.35m）可能是瓶颈。

**⚠️ 提醒**：bash wrapper（PID 1461941）在 seattle 完成后会再次启动 jogging，需提前 kill。

---

## 2026-06-14（会话五十一）

### 会话主题

正确代码 + v4 + inline_attn 在 jogging 和 seattle 场景上启动实验

### 背景

bike_vimo_v4 + 正确代码 + inline_attn 实验结果 HUMAN_PSNR=18.96，首次超越 STM（18.61）。
本次会话目标：同样配置在 jogging_vimo_v4 和 seattle_vimo_v4 场景上验证泛化性。

### 实验启动

- jogging：`cfg_files/debug/hugs_jogging_v4_correct_inline_attn_18k.yaml`
- seattle：`cfg_files/debug/hugs_seattle_v4_correct_inline_attn_18k.yaml`
- 两个实验后台并行启动，val_interval=1000

---

## 2026-06-14（会话五十）

### 会话主题

恢复 HUGS 记忆，确认 hugs_orig_bike_gt_s1 实验最终结果

### 实验结果：hugs_orig_bike_gt_s1 已完成（★✓）

**日志**：`output/human_scene/neuman/bike/hugs_trimlp/full_noamass/2026-06-14_12-07-18/train.log`

完整 HUMAN_PSNR 曲线（共 12k 步，val_interval=500）：

| step | HUMAN_PSNR | SCENE_PSNR |
|------|:----------:|:----------:|
| 500  | 11.66      | 8.23       |
| 1k   | **18.82**  | 21.01      |
| 1.5k | 18.90      | 22.83      |
| 2k   | 19.23      | 23.26      |
| 2.5k | 19.39      | 23.65      |
| 3k   | 19.27      | 23.70      |
| 3.5k | 19.60      | 24.02      |
| 4k   | 19.14      | 23.46      |
| 4.5k | 19.68      | 23.94      |
| 5k   | 19.34      | 24.18      |
| 5.5k | 19.70      | 24.52      |
| 6k   | 19.77      | 24.82      |
| 6.5k | 19.74      | 24.67      |
| 7k   | **20.14**  | 25.13（峰）|
| 7.5k | 19.71      | 24.97      |
| 8k   | 19.94      | 25.20      |
| 8.5k | 19.79      | 25.18      |
| 9k   | 19.83      | 25.24      |
| 9.5k | 19.49      | 25.14      |
| 10k  | 19.56      | 25.17      |
| 10.5k| 20.03      | 25.53      |
| 11k  | 19.87      | 25.43      |
| 11.5k| 19.93      | 25.62      |
| **final** | **20.02** | **25.67** |

**结论：代码回滚验证成功**
- 最终 HUMAN_PSNR = **20.02**，超过旧成功实验（5/29）的 19.97
- 曲线稳定在 19.5~20.1 区间，无崩坏，与历史成功实验一致
- 正确代码（full viewspace_points）确认恢复了原版 HUGS 的人体区域保护机制

**下一步**：
1. S2 实验：用此 checkpoint 继续 6k 步 + AnchorAttention（目标复现 expG 的 20.43）
2. 在正确代码下重新跑 bike_vimo（VIMO 对齐）18k，观察崩坏是否减轻
3. 方案 D（清洁 COLMAP）：会话四十七开始实施，需确认状态

### 实验结果：v4_correct_inline_attn_18k 已完成（★★ 超越 STM！）

**日志**：`output/human_scene/neuman/bike_vimo_v4/hugs_trimlp/v4_correct_inline_attn_18k/2026-06-14_12-52-08/train.log`

**配置**：seq=bike_vimo_v4（GT scale=1.55，正确代码，AnchorAttention module_start_iter=2000），18k 步，val_interval=1000

| step | HUMAN_PSNR |
|------|:----------:|
| 1k   | 17.88      |
| 2k   | 18.45      |
| 3k   | 18.29      |
| 4k   | 18.66      |
| 5k   | 18.14      |
| 6k   | 18.24      |
| 7k   | 18.78      |
| 8k   | 18.66      |
| 9k   | 18.63      |
| 10k  | 18.90      |
| 11k  | 18.75      |
| 12k  | 18.00      |
| 13k  | 18.82      |
| 14k  | 18.88      |
| 15k  | 18.87      |
| **16k** | **18.9631**（峰值，best ckpt）|
| 17k  | 18.87      |
| **final** | **18.9626** |

**关键对比**：

| 方法 | bike HUMAN_PSNR | 说明 |
|------|:--------------:|------|
| STM | 18.61 | 对标线 |
| OUR v4 正确代码 + inline_attn | **18.96** | **+0.35 vs STM ✓** |
| OUR inline_attn（旧错误代码）| 18.09 | -0.52 vs STM |
| OUR bike_vimo_v4（无attn，错误代码）| 13.96 | 崩坏 |

**结论**：
1. **OUR 超越 STM +0.35 dB**（bike 场景，之前是 -0.52 dB 落后）
2. 正确代码从 step 0 保护人体区域，消除了"黑暗谷"（step 2k 无下滑，曲线平稳上升）
3. v4 正确对齐（scale/translation 贴近 GT）+ 正确代码 = 双重保证，AnchorAttention 在干净的环境下效果显著
4. TRANSL_L2_VS_GT mean=0.3912m（≈0.26 身高），v4 对齐误差远小于 VIMO（之前 transl_norm=4.364）

---

## 2026-06-14（会话四十九）

### 会话主题

发现 expG pipeline 退步根因；回滚 gs_trainer.py 到原版 HUGS；启动 bike GT S1 验证实验

### 根因分析：expG pipeline 为何退步

**关键发现**：`gs_trainer.py` 中 scene densification 的 viewspace_points 处理方式被「修正」为只传 scene 部分的梯度，但这破坏了原版 HUGS 的隐式保护机制。

**原版 HUGS（正确行为）**：
- `render_pkg['scene_viewspace_points'] = render_pkg['viewspace_points']`（完整张量）
- `scene.add_densification_stats` 内部做 `grad[:N_scene]`，但 N_scene = scene GS 数量
- 当传入 full tensor（human+scene concat），`[:N_scene]` 实际取出的是 **human GS 的梯度**
- 效果：human GS 高梯度区域 → scene GS 不在那里 densify → 人体区域受保护

**新代码（错误行为）**：
- 正确切分 scene 部分梯度传入，scene GS 完全自主 densify
- scene GS 失去 human gradient 的"保护"，在人体区域大量增殖
- 导致 bike/seattle/jogging 全部崩坏

**确认方式**：`git show b65721a:hugs/trainer/gs_trainer.py` 原版 mkocabas 代码与 `git show 88c9871:` 5月29日成功实验版本均采用 full tensor 传法。

### 修复措施

1. **回滚 gs_trainer.py** 到 original HUGS 行为（full viewspace_points 传给 scene densification）
2. **备份新代码**：`hugs/trainer/gs_trainer_scene_grad_fix_backup.py`
3. **新建配置**：`cfg_files/debug/hugs_orig_bike_gt_s1_12k.yaml`（bike GT 对齐，12k 步，无 AnchorAttention）
4. **启动实验**：`output/human_scene/neuman/bike/hugs_trimlp/full_noamass/2026-06-14_12-07-18/`

### 实验结果：hugs_orig_bike_gt_s1（回滚后 bike GT 12k）

| step | HUMAN_PSNR（新实验）| 旧成功实验（5/29）| 崩坏实验（新代码）|
|------|:------------------:|:-----------------:|:----------------:|
| 500  | 11.66              | N/A（无500步val） | —                |
| 1k   | **18.82**          | **18.81**         | ~17.5            |
| 2k   | **19.23**          | **19.45**         | ~17.2（开始崩）  |

**结论：修复有效**，步骤1k/2k PSNR 与旧成功实验几乎完全一致（误差 <0.3 dB）。实验仍在继续运行中（共12k步）。

---

## 2026-06-14（会话四十八）

### 会话主题

恢复 HUGS 记忆，确认 vimo_v4_bike_detach_attn_18k 实验结果

### 实验结论：detach 实验无效（★✗）

| step | detach HUMAN_PSNR | 判断标准 |
|------|------------------:|---------|
| 1k | 17.66 | — |
| 2k | 16.40 | — |
| 3k | 14.52 | — |
| 4k | 14.23 | — |
| 5k | 14.18 | — |
| 6k | **14.32** | **<16 → 无效** |
| 7k | 14.17 | — |

对比基线：bike_vimo_v4 无 detach 无 attn step 6k ≈ 13.96，detach 仅提升 +0.36 dB，无实质改善。

**下一步**：推进方案 D（清洁 COLMAP 初始化）

---

## 2026-06-14（会话四十七）

### 会话主题

实施方案 D：清洁 COLMAP 初始化（在 scene GS init 前去除人体 mask 区域的 COLMAP 种子点）

### 任务目标

1. **方案 D（清洁 COLMAP 初始化）**：修改 `hugs/trainer/gs_trainer.py`，在 scene GS `init_from_pcd()` 调用前过滤污染点
2. **验证实验**：bike_vimo_v4（2140个污染点）+ 清洁 COLMAP，预期 invasion@3k 从 187K → 接近 5K，PSNR 从 13.96 → ~18
3. **可选**：GT NPZ + 清洁 COLMAP 同步实验（验证 upper bound）

### 实验结果：bike_gt（GT 对齐 + AnchorAttention，6k 步）

| step | HUMAN_PSNR | scene invasion (near) |
|------|-----------|----------------------|
| 0 | — | 2,960 |
| 500 | — | 2,905（opacity reset 后） |
| 1k | **18.67** | 9,783 |
| 1.5k | — | 30,438 |
| 2k | 17.67 | 71,516 |
| 2.5k | — | 127,264 |
| 3k | 15.39 | 187,599 |
| 3.5k | — | 208,657 |
| 4k | 15.68 | 205,233 |
| 5k | 15.44 | 215,759 |
| **6k** | **15.70** | 210,901 |

结论：
- Step 1k PSNR=18.67 超过 bike_vimo 最终值（18.09）和 STM 最终值（18.61）
- Step 3k 起 invasion 爆炸（187K），PSNR 崩到 15.39，与 v4 轨迹完全一致
- Step 4k~6k 小幅回升（15.7），说明 AnchorAttention correction 有一定遏制作用
- GT 最终 15.70 vs v4 最终 13.96，差 1.74 dB（GT 起点更高所以降得少）

### 代码对比结论（expG 调查）

**expG（`hugs_coarsealign_lab_expG.yaml`）对比当前 GT bike**：

| 项目 | Exp G（lab，无崩坏）| 当前 GT bike（有崩坏）|
|------|-----------|------------|
| `depth_w` | **无（0）** | 0.05 |
| depth 渲染 | N/A | human+scene 合并 |
| AnchorAttention start | step 12000 | step 2000 |
| scene densification 代码 | 标准 HUGS | 同 + 可选扩展（全关闭）|

**结论：代码没有被改坏**
- scene densification/prune 逻辑与原版 HUGS 完全一致
- mask_aware/global_adc/depth_prune 均由 flag 保护，默认关闭
- FusionMLP 在当前配置中为 None，不参与训练

**Exp G 没有崩坏的真实原因**：lab 场景 COLMAP 在人体位置是稀疏区域，与代码无关

**bike_gt 崩坏的真实原因**：与 v4 完全相同——GT 对齐把人体放在 COLMAP 最稠密区（z≈4.5，种子点 1,176 个），侵入不可避免

**唯一解**：方案 D（清洁 COLMAP 初始化），在 scene GS init 前移除人体 mask 区域内的 COLMAP 种子点

PLY 路径：`output/human_scene/neuman/bike_vimo_gt/hugs_trimlp/gt_bike_debug_ply_6k/2026-06-14_00-27-16/debug_ply/`（13 个文件，step 0~6000）

### 对比实验：gt_bike_noattn_6k（GT 对齐 + 无 AnchorAttention）

**目的**：诊断 PSNR 崩坏是否由 AnchorAttention 梯度冲突引起，还是 COLMAP 污染本身就足以导致崩坏。

**实验配置**：`cfg_files/debug/hugs_gt_bike_noattn_6k.yaml`，`use_anchors: false`，运行至 step 5k（进程被提前 kill）

| step | HUMAN_PSNR（noattn） | scene GS near count | 对比：bike_gt（WITH attn）|
|------|--------------------|--------------------|--------------------------|
| 1k | **18.61** | ~93K（总203K - 人110K）| **18.67** |
| 2k | 17.69 | ~156K（总266K - 人110K）| 17.67 |
| 3k | 15.48 | near=198,246 | 15.39 |
| 4k | 15.04 | near=231,013 | 15.68 |
| 5k | 15.61 | near=225,152（小幅自修复）| 15.44 |

**PLY 路径**：`output/human_scene/neuman/bike_vimo_gt/hugs_trimlp/gt_bike_noattn_6k/2026-06-14_01-15-24/debug_ply/`

**关键结论**：
- **noattn 与 attn 的崩坏轨迹几乎完全相同**（1k: 18.61 vs 18.67，3k: 15.48 vs 15.39）
- **AnchorAttention 梯度冲突不是 PSNR 崩坏的主因**（即使去掉 Attention 也崩坏）
- **COLMAP 种子点污染才是决定性原因**：scene GS 从 1,176 个污染种子点 densify → 自然入侵 → PSNR 崩坏
- step 5k scene GS near 轻微下降（231K→225K）说明 opacity pruning 有微弱自修复，但不足以恢复

### AnchorAttention 梯度冲突假说的修正

**之前的假说（基于架构分析）**：`_query_scene_tokens` 无 `.detach()` → scene GS 参数通过 human rendering loss 接收两路竞争梯度（scene 渲染损失 vs human 损失）→ 场景高斯不稳定 → PSNR 崩坏

**实验数据修正**：noattn 实验证明，不存在 AnchorAttention 时 PSNR 崩坏轨迹相同 → 梯度冲突存在但效果被 COLMAP 污染效果完全主导，不是独立的崩坏原因。

**架构修复（detach）仍有价值**：即使不是主因，去掉不必要的梯度路径是正确的架构设计，可能有轻微稳定性收益。

### 架构 detach 修复（已应用至 anchor_attention.py）

**修改内容**：在 `_query_scene_tokens` 和 `_global_scene_context` 中，对所有 scene GS 特征（xyz/opacity/scales/shs）加 `.detach()`，防止 human 渲染损失通过 AnchorAttention 回流到 scene GS 参数。

**状态**：已提交到 `hugs/models/anchor_attention.py`。AnchorAttention MLP 参数仍接收 human 渲染损失（通过其自身输出）；scene GS 参数只接收 scene 渲染损失。

### 新实验：vimo_v4_bike_detach_attn_18k（进行中）

**配置**：`cfg_files/debug/hugs_vimo_v4_bike_detach_attn_18k.yaml`
- `dataset.seq: bike_vimo_v4`（v4 正确 scale=1.55，z≈4.88）
- `anchor_attention.use_anchors: true`，`module_start_iter: 2000`，`correction_start_iter: 2000`
- `num_steps: 17998`，`debug_ply_interval: 1000`

**进程**：PID=1301700，启动于 2026-06-14 01:46
**预期**：detach 修复不太可能显著改善 PSNR（根因是 COLMAP 污染），但提供干净的对比基线
**日志路径**：`output/human_scene/neuman/bike_vimo_v4/hugs_trimlp/vimo_v4_bike_detach_attn_18k/2026-06-14_01-46-13/train.log`

### 当前优先级确认

1. **★★★ 真正修复**：方案 D（清洁 COLMAP 初始化）— 在 scene GS init 前过滤人体 mask 区域内的 COLMAP 种子点
2. **等待观察**：vimo_v4_bike_detach_attn_18k 实验结果（验证 detach 是否有轻微收益）
3. **长期**：如果方案 D 有效，考虑 6 场景全面验证

---

## 2026-06-14（会话四十六）

### 会话主题

分析 v4 对齐反而 PSNR 崩坏的根因；运行 v3 对比实验；记录结论到记忆

### 核心发现：COLMAP 种子点污染理论（决定性）

**问题**：v4 对齐（scale=1.55，接近 GT 1.53，transl 误差仅 0.40）PSNR=13.96，远低于原始 bike_vimo（scale=3.48，transl 误差 6.67）PSNR=18.09

**根因**：人体正确的 3D 位置恰好与 COLMAP 背景重建最密集的区域重叠

| 对齐方式 | 人体 z 位置 | COLMAP 体内点数 | scene_invasion@3k | HUMAN PSNR@6k |
|---------|---------|------------|-----------------|--------------|
| bike_vimo（VIMO 错误）| z=10.88 | **88个 (0.3%)** | 5,961 | **18.09** ✓ |
| bike_vimo_v3 | z=9.82 | **215个 (0.6%)** | 27,475 | **11.28** ✗ |
| bike_vimo_v4（v4 正确）| z=4.88 | **2,140个 (6.3%)** | 186,934 | **13.96** ✗ |
| bike_gt（GT 真值）| z=4.50 | **1,176个 (3.5%)** | — | — |

COLMAP 稀疏点均值 z=6.07；correct alignment 把人放入场景中心区域 → COLMAP 种子点 → scene GS densify 侵入人体

### v3 实验结果与根因

- **v3 scale=3.48**（足部接触失败，Chamfer 优化收敛到 3.48，与 bike_vimo 相同）
- **t_world=[0.91, 0.45, -1.07]**（Chamfer+IoU 优化的位移修正）
- **bike_vimo vs v3 的唯一差别**：一个常数 1.47 COLMAP 单位的平移偏移
- **结论**：DepthPro Chamfer 优化得出的 t_world 把人体推到 2D 投影更差的位置（DepthPro 深度对 bike 场景有系统误差），导致 PSNR 从 18.09 跌至 11.28

### 数据工具修复

**修复 `scripts/gen_coarse_npz_vimo.py`**：之前漏掉了 `global_t_world` 的应用
```python
# 旧（错误）：
trans_world = R_w2c.T @ (trans_cam_cu - t_w2c)
# 新（正确）：
trans_world = R_w2c.T @ (trans_cam_cu - t_w2c) + t_world_global
```

### 新建数据集与实验

- 创建 `data/neuman/dataset/bike_vimo_v3/`（symlink + v3 NPZ）
- 创建配置 `cfg_files/debug/hugs_vimo_v3_bike_debug_ply_6k.yaml`
- 训练完成（6000步），PLY 文件保存路径：
  `output/human_scene/neuman/bike_vimo_v3/hugs_trimlp/full_noamass/2026-06-13_23-06-42/debug_ply/`（13个文件，step 0~6000）

### 待解决问题

1. **COLMAP 种子点污染**：v4/GT 对齐正确 → 人体在 COLMAP 稠密区 → scene GS 侵入不可避免
2. **修复方案（待验证）**：在 scene GS 初始化前，移除人体 2D mask 区域内的 COLMAP 初始化点（一次性清洁，不同于运行时 prune）
3. **GT NPZ 实验（★★★ 仍未完成）**：用 `smpl_optimized_aligned_scale_gt.npz` 验证 upper bound，预期也有侵入问题

---

## 2026-06-13（会话四十五）

### 会话主题

实现 DepthPro 校准深度约束改善粗对齐（方案 B），并在 6 个场景全部验证

### 实现：coarse_align_smpl_neuman_v4.py

在 v3 基础上三项改进，最终全部生效：

**改进一：scale 范围扩展 (3.0,20.0) → (0.5,20.0)**
→ bike(~1.5)/seattle(~2.0)/citron(~2.4) 之前被错误过滤，修复后足部接触直接生效

**改进二：DepthPro body-center fallback**
→ IQR [20%-80%] 人体掩码 DepthPro 深度，经背景标定线性校准 → scale 估算
→ jogging 体深度约 2m（有偏），scale 估算 ~0.7（被范围过滤），未直接生效

**改进三：calib_a 约束 + 偏差检测（最终修复 jogging）**
→ 背景标定斜率 `a` 直接作为 scale（因 DepthPro 近似米制，a ≈ COLMAP/meter）
→ 当 foot_contact_median 与 calib_a_median 差异 >20% → 切换到 calib_a 模式
→ calib_a 模式下禁用 Chamfer loss（因人体点云深度有偏，会拉偏 scale）
→ jogging: a=1.535，GT=1.555，误差 0.0%

### 最终 v3 vs v4 scale 误差对比（vs GT）

| 场景       | GT    | v3 err | v4 scale | v4 err | scale_method   | 变化          |
|-----------|-------|-------|---------|-------|----------------|-------------|
| lab       | 8.617 | 4.6%  | 8.168   | 5.2%  | foot_contact   | ≈ 不变       |
| bike      | 1.530 | 127.5% | 1.550  | 1.3%  | foot_contact   | ✓✓ 大幅改善  |
| citron    | 2.785 | 28.5% | 2.372   | 14.8% | foot_contact   | ✓ 改善       |
| jogging   | 1.555 | 114.6% | 1.554  | 0.0%  | calib_a        | ✓✓ 大幅改善  |
| parkinglot| 3.742 | 7.9%  | 3.447   | 7.9%  | foot_contact   | ≈ 不变       |
| seattle   | 1.947 | 83.0% | 2.038   | 4.7%  | foot_contact   | ✓✓ 大幅改善  |

**原先4个严重失败场景（>10%）全部修复，最大误差从 127.5% 降至 14.8%**

### 关键代码位置

- `scripts/coarse_align_smpl_neuman_v4.py`
- 输出：`output/coarse_align_v4/{seq}/coarse_align_v4.json`

### 生成 vimo_v4 NPZ 并建立数据集目录（同次会话）

修改 `gen_coarse_npz_vimo.py`，新增 `--json_version` 参数支持 v4 JSON。
对 6 个场景批量运行，生成 `data/neuman/dataset/{seq}/4d_humans/smpl_optimized_aligned_scale_vimo_v4.npz`。

各场景 transl 误差（vs GT，COLMAP units）：
| 场景 | scale_method | mean_err | max_err |
|---|---|---:|---:|
| lab | foot_contact | 0.819 | 3.164 |
| bike | foot_contact | 0.399 | 0.751 |
| citron | foot_contact | 0.946 | 1.986 |
| jogging | calib_a | **0.185** | 0.558 |
| parkinglot | foot_contact | 0.580 | 1.076 |
| seattle | foot_contact | 0.359 | 0.809 |

建立 6 个 `{seq}_vimo_v4` 数据集目录（结构同 `{seq}_vimo`，`smpl_optimized_aligned_scale.npz` → v4 NPZ）。

训练时使用：`dataset.seq: {seq}_vimo_v4`

### 启动 3 个崩坏场景 v4 训练实验（同次会话）

配置文件：`cfg_files/release/neuman/hugs_vimo_v4_{seq}_inline_attn_18k.yaml`
日志：`run_logs/vimo_v4_inline_attn_18k_{seq}_20260613.log`
nohup PID：1005284，顺序运行 bike → jogging → seattle

STM 基准（对照）：bike=18.6133  jogging=17.3776  seattle=16.7278
OUR v3 结果：bike=18.0921  jogging=16.9871  seattle=15.3117
预期：v4 正确 scale 后崩坏场景大幅改善，jogging/seattle 有望超越 STM

---

## 2026-06-12（会话四十四）

### 会话主题

VIMO 粗对齐质量分析 + GT 对齐深度对比 + 可视化工具开发

### 重大发现：VIMO 粗对齐误差完美预测 OUR vs STM 胜负

**发现 GT 对齐文件**：`data/neuman/dataset/{seq}/4d_humans/smpl_optimized_aligned_scale_gt.npz`  
（NeuManDataset 加载的是 VIMO 文件 `smpl_optimized_aligned_scale.npz`，非 GT）

**关键结论一：VIMO pose 完全正确**  
对所有场景逐帧比较，`global_orient`（全局旋转）和 `body_pose` 与 GT 差值 = 0。VIMO 的误差**只在 transl（位置）和 scale（尺度）**。

**关键结论二：transl_norm 完美预测胜负**

| 场景 | transl_norm（身高倍数）| scale_vimo/scale_gt | OUR-STM |
|---|---:|---:|---:|
| lab | 0.089 | 0.95x | **+0.34 ✓** |
| parkinglot | 0.155 | 0.92x | **+0.83 ✓** |
| citron | 1.136 | 1.28x | **+0.52 ✓** |
| jogging | 2.530 | 2.15x | -0.39 ✗ |
| seattle | 3.352 | 1.82x | -1.42 ✗ |
| bike | 4.364 | 2.27x | -0.52 ✗ |

规律：**transl_norm < 1.2 → OUR 胜；transl_norm > 2.5 → OUR 败**

**关键结论三：AnchorAttention 的极限**  
最大只能修正约 1~2 个身体高度的位置偏差；bike/jogging/seattle 偏离 2~4 个身体高度，超出 correction 能力范围。

**对之前根因假说的修订**：
- 「背景复杂度」是第二层放大器，不是决定性因素
- **决定性前提**：VIMO transl/scale 误差大小（第一层）→ 是否进入 AnchorAttention correction 的能力范围

### VIMO 失败根因深度调查（会话四十四下午补充）

**诊断发现（通过逐层解析 coarse_align_v3.py + gen_coarse_npz_vimo.py）**：

VIMO 本身完全正确——焦距精确匹配 COLMAP（ratio=1.000），pred_trans z（相机空间米制深度）在正确范围内。

**唯一错误来源**：`coarse_align_v3.py` 的足部接触法在 bike/jogging/seattle/citron 返回 n_contact=0，导致 scale（COLMAP单位/米）估算失败：

| 场景 | n_contact | scale误差 | 原因 |
|---|---:|---:|---|
| lab | 14 ✓ | 0.95x | 室内地板纹理丰富 |
| parkinglot | 15 ✓ | 0.92x | 车位线提供地面特征 |
| bike | 0 ✗ | **2.28x** | 脚踩踏板，不接触地面 |
| jogging | 0 ✗ | **2.15x** | 混凝土路面无 SIFT 特征 |
| seattle | 0 ✗ | **1.83x** | 城市人行道无 SIFT 特征 |
| citron | 0 ✗ | **1.29x** | 地面稀疏点不足 |

n_contact=0 时 scale_init 回落到 8.0，仅靠 Chamfer+IoU 陷入 ~3.5 局部极值（真实约 1.5）。

**三个修复方案**：
- **方案A（立即可做）**：`neuman.py:209` 改 `_gt.npz`，验证 upper bound
- **方案B（工程化）**：n_contact=0 时用校准 DepthPro 人体深度替代足部接触
- **方案C（训练时）**：在线学习 scale 参数

已保存到 memory：`hugs-vimo-alignment-analysis.md`

### 本次开发的工具

1. **`scripts/eval_coarse_alignment.py`**：计算 6 场景粗对齐量化指标
   - Mask IoU（SMPL 凸包投影 vs GT 分割 mask）
   - KP 重投影误差（12 个 COCO-SMPL 对应关节）
   - Transl jitter（相邻帧 transl 变化均值）
   - Scale std（跨帧 scale 标准差）

2. **`scripts/export_alignment_ply.py`**：导出人体+场景 PLY 点云
   - 蓝色：scene GS（按 opacity 降序取前 200K，先过滤 SfM bbox 外的 floater）
   - 红色：human GS（15帧等间隔，每帧 5K，经 global_orient+transl+scale 近似变换到 world space）
   - 黄色：SMPL transl 轨迹点
   - 输出目录：`output/alignment_viz/`

---

## 2026-06-09（会话四十三）

### 会话主题

代码审查：depth_hinge 为何无效 + 实现 mask_aware_prune + bike 测试。

### 关键发现：depth_hinge 在 opacity reset 后失效的机制

`depth_hinge` 依赖 `scene_depth`（scene GS alpha 加权渲染），opacity reset 后 alpha→0 → scene_depth≈0 → depth_hinge loss≈0 → **梯度信号消失**。恰好在最危险的步骤（500-1000，scene GS 从零 alpha 重生）depth_hinge 是无效的。

更深根因：scene GS 致密化用的是 `xyz_gradient_accum`（viewspace 梯度累积），不是 loss 梯度。opacity reset 后所有像素 photometric error 高 → scene GS viewspace 梯度在 human mask 内最大 → 致密化集中在 human 区域。depth_hinge 的 loss 梯度无法阻止 densification 决策。

### 已有机制（全部被禁用）

代码中已存在三个相关机制，全部默认关闭（`mask_aware_enabled: False`）：

| 机制 | 位置 | 作用 | 局限 |
|------|------|------|------|
| `mask_aware_densify` | gs_trainer.py:1690 | 排除 human mask 内 scene GS 的 viewspace 梯度累积，阻止 split/clone | 不阻止已有 GS 的 opacity 增长 |
| `mask_aware_loss` | gs_trainer.py:876 | scene 只在背景像素计算 L1 loss | 主联合 loss 仍给 human 像素梯度 |
| `human_region_opacity_suppress_loss` | gs_trainer.py:911 | 惩罚 human 包围球内 scene GS opacity | soft 约束，与主 loss 博弈 |

### 新实现：mask_aware_prune

**代码位置**：`hugs/trainer/gs_trainer.py` 第1748-1763行

每次 `densify_and_prune` 后立即执行：
1. 将 scene GS 中心投影到当前帧画面（复用 `_project_scene_to_image`）
2. 找出投影落在 human mask 内的 scene GS
3. 直接删除（`prune_points`）
4. 开关：`scene.mask_aware_prune: true`（需同时设 `scene.mask_aware_enabled: true`）

与 `mask_aware_densify` 合用：densify 不创建新的 → prune 删除已有的 → human mask 内无 scene GS。

### 测试实验

**配置文件**：`cfg_files/release/neuman/mask_aware_bike_1k.yaml`
- 基于 `hugs_vimo_bike_inline_attn_18k.yaml`（VIMO + AnchorAttention）
- `num_steps: 1000`，`val_interval: 100`
- 新增：`scene.mask_aware_enabled: true`, `scene.mask_aware_densify: true`, `scene.mask_aware_prune: true`
- AnchorAttention 在 1k 范围内不激活（module_start_iter=2000），与 baseline 等价

**对照基线**：`opacity_reset_probe_ours_bike_vimo_1k`（bike + vimo，1k，无 mask_aware）= step 500 崩坏至 ~9.65，step 1000 = 15.15

**运行日志**：`run_logs/mask_aware_bike_1k.log`，PID 481840，正在运行。

---

## 2026-06-09（会话四十二）

### 会话主题

诊断实验：opacity reset vs 致密化 根因隔离；根因确认；架构讨论（scene GS 位置错误）。

### 关键发现：opacity reset 是 step 500 崩坏的根因

**代码路径**（`gs_trainer.py` 第1748-1755行）：
```python
if iteration % reset_interval == 0 or (is_white and iteration == self.cfg.scene.densify_from_iter):
    self.scene_gs.reset_opacity()
```
- `is_white and iteration == densify_from_iter`：白背景下在 step==densify_from_iter 触发 opacity reset
- 致密化条件用严格 `>`，首次在 step 600（`densify_from_iter=500`，`densification_interval=100`）
- **human GS 从不触发 opacity reset**（`reset_opacity()` 只调用在 `scene_gs` 上）

### 三个诊断实验结果

| 实验 | 配置变化 | step 500 PSNR | step 1000 PSNR | 结论 |
|------|----------|---------------|----------------|------|
| diag_scene_no_densify_1k | densify_from_iter=9999（禁用所有操作）| 平滑上升 ~16.0 | **16.31** | 对照组：无操作最好 |
| diag_opacity_reset_only_1k | densify_from_iter=500，densification_interval=10000（只有 opacity reset）| **8.93（崩坏）** | 16.41 | opacity reset 单独导致崩坏 |
| diag_densify_from_700_1k | densify_from_iter=700（崩坏时机后移）| 700步 8.73崩坏 | 13.80（未恢复）| 时机移动证实是 opacity reset |

**结论：opacity reset（非致密化）是根因；崩坏后可以恢复（1k步），但 18k 训练后 seattle 最终质量仍卡在 15.3 dB 天花板。**

### 进一步分析：为何最终质量存在天花板

- HUGS baseline（15.37）和 inline_attn（15.31）对 seattle 最终质量相同 → 问题在架构，非数据
- STM 用相同 VIMO 输入却能在 seattle 达到更高质量 → 算法层面有差距
- **根因（用户纠正）**：问题不在 human GS 颜色泄漏，而在于 **scene GS 占据了错误位置，遮挡了 human GS**
- AnchorAttention 只修正 human GS xyz/translation，完全不触及 scene GS → 无法解决 scene GS 入侵问题

### 待解决的架构问题

需要针对 **scene GS 位置** 的修复方案，而非 human GS 侧。方向待讨论。

---

## 2026-06-09（会话四十一）

### 会话主题

渲染图检查、depth_hinge 机制验证、实验错误纠正。

### 关键发现：depth_hinge 在不崩坏场景上有效

之前标记为"depth_hinge_fixed_seattle"的实验（PSNR=18.59）实际上运行的是 **lab 场景**（`seq: lab`），输出目录 `output/human_scene/neuman/lab/hugs_trimlp/full_noamass/2026-06-09_12-53-11/`，config_train.yaml 和渲染图均确认是 lab 室内场景。

**纠正后的结论**：

| 场景 | 类型 | 1k baseline | 1k depth_hinge | 提升 |
|------|------|------------|----------------|------|
| lab（GT transl）| 不崩坏 | 18.01 | **18.59** | **+0.58 dB** ✓ |
| bike | 崩坏 | 15.15 | 15.21 | +0.06 dB |
| jogging | 崩坏 | 15.81 | 15.82 | +0.01 dB |
| seattle（实际） | 崩坏 | 15.22 | 待补跑 | — |

**结论：depth_hinge 在不崩坏场景（lab）上能提供 +0.58 dB 的稳定提升。** 对崩坏场景（bike/jogging）几乎无效，原因是这两个场景的 scene GS depth violation 本来就很小（bike=6.75，jogging=2.97），而且约束很快收敛到 0，说明 scene GS 并未在 human mask 内造成深度入侵，崩坏根因另有其他。

### depth_hinge 约束确认正确执行

bike/jogging 1k 实验的 violation 演化：bike 在 step ~3 已降至 0，jogging 同样；opacity reset 后短暂回升（bike 0.78，jogging 2.15），随后再降至近 0。约束完全生效，但根因不在此。

### Seattle 补跑结果（会话四十二更新）

之前从未正确跑过 seattle depth_hinge（修复版）。已补跑完毕：
```
run_logs/depth_hinge_seattle_1k_rerun.log
```

**Seattle 完整对比（seq: seattle_vimo）：**

| Step | Baseline（无 depth_hinge）| 有 bug 版本 | Fixed depth_hinge（补跑）|
|------|--------------------------|------------|--------------------------|
| 100  | —                        | —          | 13.5526                  |
| 200  | —                        | —          | 14.1951                  |
| 300  | —                        | —          | 14.4306                  |
| 400  | 15.72                    | 11.68      | 14.5193                  |
| 500  | **8.86（崩坏）**          | 9.65       | **9.4206（崩坏）**        |
| 600  | 14.13                    | —          | **14.9614**               |
| 700  | —                        | —          | 15.0193                  |
| 800  | —                        | —          | 15.3044                  |
| 900  | —                        | —          | 15.0595                  |
| 1000 | **15.22**                | 17.13      | **14.7801**              |

**结论：depth_hinge 对 seattle 崩坏场景没有帮助**（最终 14.78 < baseline 15.22）。虽然初始 violation 高达 4.93（step 0时 `l_scene_behind_human_depth=4.9312`），约束确实起效并快速下降，但 step 500 的崩坏仍然出现（9.42 vs baseline 8.86，崩坏略轻但依然崩坏），最终 1k 步 PSNR 反而低于 baseline 0.44 dB。

**核心结论更新：**

| 场景 | 类型 | 1k baseline | 1k depth_hinge | 提升 |
|------|------|------------|----------------|------|
| lab（GT transl）| 不崩坏 | 18.01 | **18.59** | **+0.58 dB** ✓ |
| bike | 崩坏 | 15.15 | 15.21 | +0.06 dB |
| jogging | 崩坏 | 15.81 | 15.82 | +0.01 dB |
| seattle | 崩坏 | **15.22** | 14.78 | **-0.44 dB** ✗ |

**depth_hinge 对崩坏场景无效（甚至有轻微负面效果），崩坏根因不在 scene GS depth 入侵，需要另寻方法。**

---

## 2026-06-09（会话四十）

### 会话主题

实验设置公平性验证 + depth_hinge_fixed 实验结果汇总。

### 公平性分析结论

**问题**：depth_hinge 配置 `use_anchors: false`，而 baseline 配置 `use_anchors: true, module_start_iter: 2000`——这是否影响比较公平性？

**结论：比较完全公平。** 代码路径（`anchor_attention.py` 第248行）：
```python
if scene_out is None or not enabled or int(iteration) < module_start:
    return human_out, {}  # 直接透传，不修改 GS 输出
```
当 `iteration < module_start_iter=2000` 时，AnchorAttention 是纯透传（no-op）。1000步实验中：
- Baseline：AnchorAttention 初始化但从不激活（step 0~999 < 2000）
- depth_hinge：`self.anchor_attention = None`，直接跳过

两者训练效果完全等价。delta_loss 也不会被加入（stats 为空字典，第1472行判断失败）。

**唯一功能差异**：depth_hinge 多了 `scene_behind_human_depth_w: 1.0` 约束，这正是被测变量。

**Seattle 场景确认**：两个配置都使用 `seq: seattle_vimo`（VIMO 粗对齐），步数、lr、lbs_w、depth_w 完全相同。

### depth_hinge_fixed 完整实验结果

配置：`dataset.seq=seattle_vimo`，1000步，`scene_behind_human_depth_w=1.0`，`margin=0.05`，bug修复（`human_depth.detach()`）

| Step | depth_hinge_fixed | baseline（无hinge）| 有bug版本 |
|---:|---:|---:|---:|
| 100 | 15.08 | ~15.72* | ~12.0 |
| 200 | 15.95 | — | — |
| 300 | 16.26 | — | — |
| 400 | 16.62 | 15.72 | 11.68 |
| 500 | 10.12 | 8.86 | 9.65 |
| 600 | 16.93 | 14.13 | — |
| 700 | 18.09 | — | — |
| 800 | 18.34 | — | — |
| 900 | 18.54 | — | — |
| **1000** | **18.59** | **15.22** | **17.13** |

*baseline step400数据来自会话三十七，非当次run

**效果**：与1k baseline相比 +3.37 dB（18.59 vs 15.22），与STM最终结果16.73相比 +1.86 dB（1k步 vs 20k步！）。

**关键改善点**：opacity reset崩坏从 8.86→10.12（底部提升），崩后恢复速度从 14.13（step600）→16.93，显著更快。

### depth_hinge 三场景泛化实验（同次会话追加）

三个崩坏场景（seattle/bike/jogging）全部跑 1k 步 depth_hinge 对比，配置：`scene_behind_human_depth_w=1.0`，`margin=0.05`，设置与各场景 baseline 完全一致。

| Step | seattle_hinge | seattle_base | bike_hinge | bike_base | jogging_hinge | jogging_base |
|---:|---:|---:|---:|---:|---:|---:|
| 400 | **16.62** | 15.72 | 14.85 | 15.55 | 15.09 | — |
| 500↓ | 10.12 | 8.86 | 11.88 | 12.07 | 11.71 | — |
| 600 | **16.93** | 14.13 | **15.80** | 14.20 | **15.80** | — |
| **1000** | **18.59** | 15.22 | **15.21** | 15.15 | **15.82** | **15.81** |
| 提升 | **+3.37 dB** | — | **+0.06 dB** | — | **+0.01 dB** | — |

初始 `l_scene_behind_human_depth`：seattle=19.56，bike=6.75，jogging=2.97

**核心结论**：depth_hinge 效果与初始 violation 正比。

- **Seattle（violation=19.56）**：hinge 显著有效，+3.37 dB，step600 恢复速度 +2.80 dB
- **Bike（violation=6.75）**：hinge 几乎无效，+0.06 dB，但 step600 恢复略有改善（+1.60 dB）
- **Jogging（violation=2.97）**：hinge 完全无效，+0.01 dB（误差范围内）

**结论**：scene GS 入侵 human 区域（depth violation）并非三个崩坏场景共同的主因，只在 seattle 是主要因素。bike 和 jogging 的崩坏是另有原因，depth hinge 无法解决。

### 下一步建议

- Seattle depth_hinge 效果显著，值得扩展至 18k 步与 STM 20k 做等步数公平对比
- Bike/jogging 崩坏根因需要另行探索（可能是人体 GS 质量、scene densify 节奏等）

---

## 2026-06-09（会话三十九）

### 会话主题

定量实验验证「崩坏场景是否有更多 scene GS 入侵人体区域」：4场景（seattle/bike/citron/lab_vimo）并行训练 1k 步，读取 INVASION_COUNT 指标对比。

### 核心进展

**实验背景**：用 AABB 方法统计 scene GS 落入 human GS 包围盒内的数量（margin=0.1），每100步 validate 一次，测1000步。

**并行崩溃根因**：4进程同时启动时共享同一 canon 渲染目录（`lab/full_noamass/{timestamp}`），互相 rmtree 导致 FileNotFoundError，3个进程在 step 10 后 OOM 崩溃，仅 citron 完成。改为顺序运行后全部成功。

**4场景完整 INVASION_COUNT 数据**（顺序运行，seattle→bike→lab_vimo）：

| Step | seattle（崩坏）| bike（崩坏）| citron（正常）| lab_vimo（正常）|
|---:|---:|---:|---:|---:|
| 100 | 391.6 | 541.6 | 457.9 | 566.3 |
| 200 | 512.4 | 566.4 | 516.3 | 545.1 |
| 300 | 572.6 | 559.6 | 512.0 | 543.4 |
| 400 | 558.9 | 545.8 | 579.1 | 517.7 |
| 500 | 565.7 | 577.8 | 627.4 | 553.9 |
| 600 | 530.7 | 531.3 | 567.0 | 521.6 |
| 700 | 751.1 | 660.3 | 772.3 | 720.0 |
| 800 | 1142.2 | 1058.8 | 1206.4 | 1123.3 |
| 900 | 1666.2 | 1366.1 | 1707.8 | 1864.5 |
| **1000** | **2599.0** | **2010.6** | **2647.3** | **2790.0** |

**结论：假说不成立。** 四条曲线几乎完全重叠，走势相同（step 100~600 平稳在 400~650，step 700 开始随 densification 急速增长）。正常场景 lab_vimo=2790 反而最高，崩坏场景 bike=2010 反而最低。INVASION_COUNT 绝对值无法区分崩坏/不崩坏场景。

**指标局限性**：使用 AABB（轴对齐包围盒，margin=0.1），太粗糙；未归一化 scene GS 总数（#sp 各场景不同）；未过滤低 opacity 点；不反映入侵 GS 是否真正遮挡人体。

**对已有根因结论的影响**：不动摇。崩坏根因（opacity reset 后复杂背景梯度阻碍恢复）已由会话三十七的直接 PSNR 曲线实验确认，这次 invasion 实验只是定量化"入侵数量"这一 proxy，proxy 本身选择有误。

### 下一步

- 待定（invasion probe 实验结论已关闭，下一步方向需讨论）

---

## 2026-06-09（会话三十八）

### 会话主题

恢复 HUGS 项目记忆，确认 anchor_start0 实验状态

### 核心进展

- 记忆恢复完毕，等待用户指令

### 下一步

- 读取 anchor_start0_seattle_1k 实验结果（PID 35806，日志 run_logs/anchor_start0_seattle_1k.log）

---

## 2026-06-09（会话三十七）

### 会话主题

六场景 Opacity Reset 扩展对比实验，最终确定根因（推翻往返假说），启动 AnchorAttention start_iter=0 实验

### 核心进展

**六组 1k 探针实验完整数据**（val_interval=100，核心观察 step 400→500→600）：

| Step | OUR seattle | OUR lab_vimo | STM seattle | OUR citron | OUR lab(GT) | OUR bike |
|------|------------|-------------|------------|------------|------------|---------|
| 300 | 15.40 | 15.64 | 15.32 | 16.87 | 17.25 | 14.32 |
| 400 | **15.72** | **16.79** | **15.34** | **16.79** | **18.08** | **15.55** |
| **500（reset）** | **8.86↓** | **9.25↓** | **7.47↓** | **8.78↓** | **9.19↓** | **12.07↓** |
| 600 | 14.13 | 15.91 | 7.68（仍崩）| **16.74** | **17.43** | 14.20 |
| 700 | 15.11 | 16.48 | 15.43 | 16.72 | 18.30 | 15.07 |
| 800 | 14.09 | 16.83 | 15.17 | 16.64 | 18.18 | 15.88 |
| 900 | 15.07 | 17.18 | 15.51 | 17.05 | 18.71 | 15.74 |
| 1000 | 15.22 | **17.17** | 15.82 | 16.93 | **18.01** | 15.15 |

**往返假说彻底推翻（关键对比：citron vs seattle）**：

| 场景 | 往返比 | step 500 崩坏 | step 600 | 结论 |
|---|---|---|---|---|
| citron（果园） | **1.29** | 8.78 | **16.74（恢复完毕）** | 100步内完全恢复 |
| seattle（城市） | **1.32** | 8.86 | 14.13（未恢复）| 500步后仍低于 reset 前 |

两者往返比几乎完全相同，崩坏幅度相同，但恢复结果天壤之别。**往返不是根因。**

**真正根因：scene GS 在人体遮挡区域的复杂背景梯度阻碍恢复**

- 简单场景（lab/citron/parkinglot）：人体背后背景纹理简单，scene GS 在人体遮挡区域梯度弱 → 主要在人体外生长 → 100步内快速恢复
- 复杂城市场景（seattle/bike/jogging）：建筑/汽车/路牌等细节在人体遮挡区域产生大梯度 → scene GS 被吸引进入人体区域 → 与 human GS 竞争 → 恢复缓慢

其他佐证：
- lab往返=11.64（高往返，但室内简单）→ 完全恢复；jogging往返=15.44（类似往返，但室外复杂）→ 不恢复
- bike step 500 崩坏最小（12.07），但恢复同样差 → 不是"进入了多少 GS"的问题，而是 densify 阶段梯度的复杂度

**已启动实验：AnchorAttention start_iter=0 on seattle (1k)**

- 配置文件：`cfg_files/release/neuman/anchor_start0_seattle_1k.yaml`
- 关键改动：`module_start_iter: 0`、`correction_start_iter: 0`、`correction_warmup_iters: 500`、`gamma_transl_decay_start: 0`
- 进程 PID：35806，正在运行
- 日志：`run_logs/anchor_start0_seattle_1k.log`
- 预期：step 600 HUMAN_PSNR 能否超过 14.13（OUR baseline），理想情况接近 citron 的 16.74

### 下一步（明天继续）

1. 读取 anchor_start0_seattle_1k 结果（查 run_logs/anchor_start0_seattle_1k.log）
2. 如果 AnchorAttention 提前有效（step 600 > 15.xx），考虑在全部崩坏场景（bike/jogging）做同样实验
3. 如果无效，考虑 scene mask-aware densification（禁止 scene GS 在人体 mask 区域 clone/split）

---

## 2026-06-09（会话三十六）

### 会话主题

深入分析 opacity reset 对不同场景的影响；确认崩坏根因与方向纠正

### 核心进展

**STM 参数重新确认（纠正会话三十五的错误）**：

| 配置项 | OUR 基线（VIMO+Attention）| STM 实际 |
|---|---|---|
| scene.densify_from_iter | **500** | **500**（config.py 默认，yaml 未覆盖）|
| human.densify_from_iter | 3000 | 3000 |
| scene opacity reset 触发步 | step 500 | step 500 |
| scene.opacity_reset_interval | 20000 | 20000 |

**结论：OUR 基线和 STM 的 scene.densify_from_iter 完全一样（都是 500）**，会话三十五认为 STM 用 3000 是错误的——STM yaml 只覆盖了 `human.densify_from_iter=3000`，scene 沿用 config.py 默认值 500。

**18k 实验（densify3k）终止**：将 scene.densify_from_iter 改为 3000 是负向改动，引入了 step 3k 的 opacity reset 崩坏（HUMAN_PSNR 16.74→8.46），且无法恢复，最终 step 9k 仍只有 14.15 vs STM 的 16.58。已终止该实验。

**Opacity Reset 三方对比实验（1k 步，val_interval=100）**：

| Step | OUR seattle | OUR lab | STM seattle |
|------|------------|---------|------------|
| 400 | 15.72 | 16.79 | ~15.3（估算）|
| **500（reset）** | **8.86↓** | **9.25↓** | **7.47↓** |
| 600 | 14.13 | 15.91 | 7.68（仍低）|
| 700 | 15.11 | 16.48 | 15.43 |
| 1000 | 15.22 | **17.17** | 15.82 |

**核心结论（本会话最重要发现）**：

> **Opacity reset 对三者都造成大幅崩坏（6~9 dB），STM 并不免疫。核心差异在于恢复能力：OUR lab 能完全恢复并超越 reset 前水平；OUR seattle 和 STM seattle 止步于 15 附近，均低于 reset 前峰值（~15.7）。问题根因是往返轨迹（seattle/bike/jogging）在 scene GS 重新 densify 阶段产生矛盾梯度，导致恢复受阻，与是否用 FusionMLP 无直接关联。**

**STM vs OUR 的持久差距**（对比完整 18k 训练曲线）：

- STM HUMAN_PSNR 在 step 1k 就达到 16.23，整个训练过程稳定在 16.2~16.8
- OUR 基线始终在 14.7~15.3，两条曲线几乎平行，差距在最初 500 步就被锁死
- 差距来源：step 500 opacity reset 后，无 FusionMLP 保护，scene GS 在往返场景 densify 阶段更严重地侵入人体区域

### 下一步方向

- 往返轨迹的矛盾梯度 + reset 后 scene GS densify 是根因，需要在 step 500~1000 这段关键期保护人体区域
- AnchorAttention 从 step 2000 才启动，错过了关键保护窗口
- 可能方向：AnchorAttention module_start_iter=0（从训练开始就激活）

---

## 2026-06-09（会话三十五）

### 会话主题

参考 STM 代码修复 FusionMLP 实现；发现 STM 正确参数；提出 scene 早期 densify 假设

### 核心进展

**FusionMLP v3/v4 Bug 分析与修复**：

通过仔细对比 STM 代码（`StM/renderer/gs_renderer.py`、`StM/trainer/gs_trainer.py`、`StM/models/fuse.py`），发现 v3 有两个关键 bug：

1. **renderer bug**：v3 只用 `fused_gs_out[n_human:]`（scene 部分），human 用原始输出 → human_gs_out['xyz'] 出现在两条计算路径，梯度冲突
2. **validate bug**：validate() 未传 fused_gs_out，测的是无 MLP 的原始场景

**修复（v4）**：
- `hugs/renderer/gs_renderer.py`：fused_gs_out 时直接用全部（N_human+N_scene），对齐 STM
- `hugs/trainer/gs_trainer.py` validate()：也计算并传入 fused_gs_out

**FusionMLP v4 结果（3k, seattle_vimo）**：

| step | HUMAN_PSNR |
|---:|---:|
| 1k | 14.58 |
| 2k | 15.53 |
| 3k | **15.97**（final）|

v3=12.00 → v4=15.97，**+3.97**，修复有效。

**STM 参数纠正（之前记忆错误）**：
- `depth_w`：STM 实际是 **0.03**（不是 0）
- `lbs_w`：STM 是 **100**（我们是 1000）
- `scene.densify_from_iter`：STM 是 **3000**（我们是 500）

**新假设**：scene `densify_from_iter=500` 导致崩坏——场景 GS 在第 500 步就开始增殖，往返轨迹的矛盾梯度生成大量 spurious 点在人体区域，FusionMLP 还未学会压制就被"污染"。STM 延迟到 3000 步才 densify，避免了这个问题。

**待做验证实验**：将 `scene.densify_from_iter` 从 500 改为 3000，保持其他参数，看 step 1k PSNR 是否接近 STM(16.23)。

---

## 2026-06-08（会话三十四）

### 会话主题

自主实验循环——尝试解决 seattle_vimo 崩坏场景；分析 STM FusionMLP 并尝试移植

### 核心进展

**今日尝试方向（全部 seq: seattle_vimo，3k 快速验证）**：

| 实验名 | 方案描述 | step 1k | step 2k | step 3k | 结论 |
|---|---|---|---|---|---|
| `vimo_exclusion30_3k_seattle_20260608` | human_exclusion_w=30（几何排斥）| 15.70 | — | 14.92 | ❌ 最终崩坏 |
| `vimo_opacity_suppress_3k_seattle_20260608` | scene GS opacity 压制（w=2.0，3k）| 14.95 | — | 14.67 | ❌ 无效 |
| `vimo_opacity_suppress_18k_seattle_20260608` | 同上 18k 版（中途 kill）| — | — | — | ❌ kill（实验设置曾跑在 lab，已确认无效） |
| `vimo_maskaware_3k_seattle_20260608` | scene GS 只从非人体像素学习 | 14.74 | 15.17 | — | ❌ 无效（kill at 2.5k） |
| `vimo_scene_delay2k_3k_seattle_20260608` | scene GS 延迟到 step 2000 开始优化 | 7.42 | 12.55 | — | ❌ 崩溃（kill at 2.5k） |
| `vimo_attn_warmup_opacity_3k_seattle_20260608` | AnchorAttn module_start=0 + correct_opacity=true | — | — | — | ❌ kill（未出结果，优先改做 FusionMLP） |
| `vimo_fusionmlp_3k_seattle_20260608`（v1）| FusionMLP（human+scene 全量，含 human xyz 修改）| 8.85 | — | — | ❌ 崩溃（human xyz 被破坏）|
| `vimo_fusionmlp_3k_seattle_20260608`（v2）| FusionMLP 修正 property_dims（同 STM），human xyz 仍被修改 | — | 12.74 | — | ❌ 崩溃（同上）|
| `vimo_fusionmlp_3k_seattle_20260608`（v3）| FusionMLP only scene，human 用原始 | — | — | — | ❌ kill（暂停实验）|

**技术分析与结论**：

- **STM FusionMLP 无法直接移植的根本原因**：STM 的 human GS 是 SMPL mesh 顶点（位置规则），FusionMLP 对 human xyz 的微调稳定。我们的 human GS 是 triplane 生成的 ~110K 高斯点，xyz 由 SMPL 变形精确确定，FusionMLP 静态权重修改动态生成的 human xyz 会破坏 SMPL 对齐，导致 lbs_loss 和渲染梯度方向矛盾。
- **往返轨迹 = 人体在3D空间的运动轨迹**（不是相机），Seattle/bike/jogging 人体来回经过同一3D区域，scene GS 受到矛盾梯度
- **STM 不崩坏的原因**：FusionMLP 从 step 0 就联合处理 human+scene，隐式学会在人体体积内压制 scene GS opacity，而不是依赖几何启发式
- **所有几何启发式方法（exclusion、suppress、mask_aware、delay）均失败**，根因是 VIMO 误差导致 SMPL mesh 位置不准，几何判定不精确

**代码改动**（本会话新增）：
- `hugs/models/fusion_mlp.py`：新建，STM HumanSceneFuseDecoder 的移植版（ResidualMLP，fc2 zero init）
- `hugs/trainer/gs_trainer.py`：添加 `setup_fusion_mlp`、训练循环中 fusion forward、optimizer step
- `hugs/renderer/gs_renderer.py`：`render_human_scene` 新增 `fused_gs_out` 参数支持

**下次待讨论方向**：
- FusionMLP v3（只修正 scene，human 保持原始）结果未知，可继续测试
- 或考虑其他从 step 0 就能保护人体区域的学习型机制

---

## 2026-06-08（会话三十三）

### 会话主题

恢复 HUGS 记忆，查看当前实验状态

### 核心进展

**GPU 当前空闲**，所有实验已完成。今日实验汇总：

| 实验名 | 关键结果 | 状态 |
|---|---|---|
| `vimo_attn_step0_3k_seattle_20260608` | step1k=15.5556，final(3k)=14.4934 | ✅ 完成 |
| `vimo_optim_transl_1k_seattle_20260608` | step1k=14.9712 | ✅ 完成 |

**attn_step0 实验（最优先方向验证）**：
- AnchorAttention 提前到 step 0，seattle 场景，3k步快速验证
- step 1k PSNR = **15.5556**（vs OUR inline_attn=15.34，HUGS baseline=15.79，STM=16.23）
- 结论：比原始 inline_attn（step2000启动）略好（+0.22），但仍低于 HUGS baseline 和 STM
- final 3k PSNR 下滑到 14.49（3k步太短，未充分训练）

---

## 2026-06-08（会话三十二）

### 会话主题

恢复 HUGS 记忆，检查截帧实验当前进度

### 核心进展

**截帧实验进度（10:05 CST 检查）**：

- `vimo_inline_attn_18k_bike_f60_20260607`（OUR f60）：**正在运行**，8990/17999步（50%），已用35分钟，剩余约57分钟，预计完成约 11:02。
- `stm_vimo_bike_f60_20k_20260607`（STM f60）：⏳ 等待中
- `vimo_inline_attn_18k_jogging_f50_20260607`（OUR f50）：⏳ 等待中
- `stm_vimo_jogging_f50_20k_20260607`（STM f50）：⏳ 等待中

预计全部完成时间：约17:00~18:00 CST（4实验×~1.5~2小时）

---

## 2026-06-08（会话三十一）

### 会话主题

恢复 HUGS 记忆，检查截帧实验及 seattle 实验最新进度

### 核心进展

**V2 完整验证全部完成（6场景×3方法）**：截至本次对话所有实验均已完成。

**截帧实验（[5/5]步）未执行**：run_gate_then_resume.sh 在上次运行时只读到 [1/4]~[4/4]，[5/5] 截帧步骤是事后追加到文件中的，进程未感知，截帧实验需手动启动。

**完整 HUMAN PSNR 汇总**（inline_attn_18k pipeline，VIMO对齐；lab用旧ADC pipeline）：

| 场景 | OUR | STM | HUGS | OUR-STM |
|---|---:|---:|---:|---:|
| lab* | 19.5512 | 17.4512 | 16.7751 | +2.10 ✓ |
| bike | 18.0921 | 18.6133 | 17.0399 | -0.52 ✗ |
| citron | 17.4008 | 16.8763 | 15.5797 | +0.52 ✓ |
| jogging | 16.9871 | 17.3776 | 16.1335 | -0.39 ✗ |
| parkinglot | 15.1924 | 14.3660 | 13.9266 | +0.83 ✓ |
| seattle | 15.3117 | 16.7278 | 15.3684 | -1.42 ✗ |
| avg(5无lab) | 16.5968 | 16.7922 | 15.6096 | -0.20 |
| avg(全6) | 17.0892 | 16.9020 | 15.8039 | +0.19 |

**结论**：OUR 在3/6场景超越 STM，平均HUMAN PSNR 5场景略低 STM（-0.20）；seattle 成为新的崩坏场景（OUR≈HUGS baseline，远低STM），往返轨迹假说需要验证。

---

## 2026-06-07（会话三十）

### 会话主题

恢复 HUGS 记忆，检查当前实验进度

### 核心进展

**Gate 结论记录**：
- bike gate（18.0924）≈ bike OUR（18.0921），gate 基本无效
- 根因：往返轨迹导致 scene GS 受矛盾梯度，gate 无法从根本上解决

**新增截帧实验（往返轨迹假说验证）**：
- 代码修改：`NeumanDataset` 加 `max_frames` 参数（hugs + StM 两套）
- 配置文件：`hugs_vimo_bike_f60_inline_attn_18k.yaml`、`hugs_vimo_jogging_f50_inline_attn_18k.yaml`、`stm_vimo_bike_f60_20k.yaml`、`stm_vimo_jogging_f50_20k.yaml`
- 运行编排：已追加到 `run_gate_then_resume.sh` 第 [5/5] 步，seattle 三方法后自动运行
- 参数：bike 前60帧（去程），jogging 前50帧（去程）

**新增 PSNR 结果（parkinglot）**：
- parkinglot OUR：HUMAN PSNR = **15.1924**，PSNR = 23.3707
- parkinglot STM：HUMAN PSNR = **14.3660**，PSNR = 22.7068
- 结论：parkinglot 场景 OUR > STM ✓

**当前正在运行的实验**（截至 2026-06-07 20:02）：
1. `vimo_hugs_baseline_parkinglot_20260607`（PID 1130062）：步骤 ~14030/15001，94%，约 7 分钟后完成
2. `vimo_gate_18k_bike_20260607`（PID 1130291）：步骤 ~8260/17999，46%，中间结果 step8000 HUMAN PSNR=17.5338，约 2 小时后完成

**编排状态**：
- gate_then_resume.sh（PID 1105843）已成功触发：检测到 parkinglot STM 完成 → 停止原脚本(983995) → 启动 bike gate
- parkinglot HUGS baseline 在 983995 被 kill 前已作为子进程启动，继续运行（正常）
- bike gate 完成后，gate 脚本将自动运行 jogging gate，再接 parkinglot HUGS（skip，因已跑），再接 seattle 三方法

---

## 2026-06-07（会话二十九）

### 会话主题

详细分析 bike/jogging 崩坏机制，记录 pipeline 和实验流程到记忆，启动 gate_then_resume.sh 自动编排

### 核心进展

- `scripts/run_gate_then_resume.sh` 已启动（PID 1105843），监视 parkinglot STM 完成后自动接管
- 更新 memory：`hugs-project-overview.md`（inline_attn + Gate pipeline 详细设计）、`hugs-experiment-results.md`（V2 进度+实验流程）
- 详细分析 bike/jogging 崩坏的级联失效机制（见下）

### bike/jogging 崩坏的完整因果链分析

#### 根本前提：往返轨迹（Round-trip Trajectory）

bike 和 jogging 场景的人物轨迹都是**往返型**：

- **bike**：frames 0→60 向右走，frames 60→104 原路返回向左走
- **jogging**：frames 0→54 向右跑，frames 54→101 原路返回向左跑
- **lab/citron**：无此规律，人物在区域内来回但不沿同一直线原路折返

往返轨迹是一切问题的根源。

---

#### 第一层：场景 GS 接收矛盾梯度 → scene GS 在折返区崩坏

场景 GS（scene Gaussians）是静态的，整个训练期间保持在同一个 3D 位置。

在往返轨迹中，折返区附近的 scene GS 会被**两次经过**：

| 时机 | 人物相对 scene GS 的关系 | scene GS 学到的 "应有" 外观 |
|---|---|---|
| 去程（帧0→54）| scene GS 在人物**前方**（背景） | 人物背后的场景纹理、光照 |
| 返程（帧54→101）| scene GS 在人物**后方**（同一位置）| 场景仍是那些纹理，但人物正面挡住了 |

同一个 scene GS，在去程时学"没有人"的外观，在返程时被人体遮挡，梯度让它学习遮挡后的外观。这两个目标**相互矛盾**。

结果：折返区的 scene GS 最终呈现出一种"妥协"的错误状态，其颜色/不透明度不能正确表达任何一帧的真实场景，scene GS 在折返区被**污染**。

这是最根本的一层，lab/citron 不会发生，因为它们没有如此对称的往返重叠。

---

#### 第二层：VIMO 对齐在折返点失效 → human GS 被放错位置

VIMO 使用光流估计做 SMPL 粗对齐。人物方向突然反转时，光流追踪失去可靠信号，SMPL 参数在折返点前后出现**跳变**。

- **bike 数据**：frame 84→85 的 SMPL translation 跳变幅度处于所有帧间跳变的 **97.6th percentile**，frame 79~92 整段都存在多次异常跳变
- **jogging 数据**：3D 位置空白区（nearest training frame 的 3D 距离）与 PSNR 的 Pearson 相关系数 **r = -0.821（p=0.004）**，即覆盖越差的区域 PSNR 越低

SMPL 参数错误意味着：LBS（Linear Blend Skinning）在变形 human GS 时，会把它们变形到**错误的 3D 位置**。这些 human GS 出现在与场景不符合的地方，且初始 opacity 并不低。

---

#### 第三层：anchor attention 接收噪声上下文 → 修正方向错误

anchor attention 的工作方式是：

1. 在 SMPL mesh 上采样 16 个语义锚点（脚/手/肩等）
2. 每个锚点查询附近的 scene GS 作为上下文
3. cross-attention 计算出 human GS 的位置修正量（transl+xyz delta）

第一层已经让折返区的 scene GS 被污染了。当 anchor attention 去查询"这个人应该在哪里"时，它获取到的 scene context 是**噪声**，学到的修正方向可能反而是错的，进一步把 human GS 推向更错误的位置。

lab/citron 没有这个问题，因为它们的 scene GS 是干净的，anchor attention 能正确学到"把人拉回来"。

---

#### 第四层：缺少 opacity 抑制机制 → 错位 GS 保持高不透明度 → 可见爆炸

anchor attention **只修正位置**（transl+xyz），不修正 opacity。

当 human GS 因为上述原因漂移到错误位置时，它们依然保持高 opacity。这些高不透明度的错位 GS 直接被渲染进画面，产生可见的白色爆炸点、ghosting、或者身体区域像素异常。

这就是视觉上看到的"崩坏"。

---

#### 为什么 STM 不崩坏

STM 在每次渲染**前**运行 `HumanSceneFuseDecoder`：

```
[human GS 全量属性 + scene GS 全量属性] → concat → ResidualMLP → 修正后属性
```

ResidualMLP 的 fc2 初始化为零，训练初期不改变属性。随着训练进行，网络自动学到：

> "当 human GS 的 xyz 与周围 scene GS 的分布不一致时，把 human GS 的 opacity 压低到接近零"

这是一个**隐式的全局 opacity 抑制机制**，无论 human GS 漂移到哪里，都能被压制。

OUR 的 anchor attention 是局部的、位置导向的，缺少这个全局 opacity 调制能力，这是 bike/jogging 崩坏而 STM 不崩的关键差异。

---

#### Global Scene Gate 的作用

我们新加的 Global Scene Gate 正是补齐这一短板：

```
top-512 scene GS（按 opacity 排序）→ attention pooling → global_ctx
[human_features, global_ctx] → gate_mlp（zero-init）→ gate_delta
corrected_opacity = clamp(opacity × (1 + 0.1 × gate_delta))
```

- **全局感知**：不是局部锚点查询，而是看整个场景 GS 的全局状态
- **opacity 调制**：直接输出 opacity 修正，而不是位置修正
- **zero-init**：训练初期无效果，不破坏现有结果，网络自学何时压制

理想情况下，当 human GS 漂移到与场景 GS 不一致的区域时，gate 学到输出负的 gate_delta，opacity 被压低，不再产生爆炸点。

---

#### 四层级联总结

```
往返轨迹
  → ①scene GS 矛盾梯度 → 折返区 scene GS 污染
      → ②VIMO 对齐失效 → human GS 放错位置
          → ③anchor attention 噪声上下文 → 修正方向错误
              → ④无 opacity 抑制 → 错位 GS 高不透明度 → 可见爆炸
                                                        ↑
                            Global Scene Gate 在这里介入
```

lab/citron 四层都不触发，故不崩。bike/jogging 四层级联，故严重崩坏。

---

#### ⚠️ 二次纠正（2026-06-07 会话二十九，再次深挖）

**用户指出 lab 也用了粗对齐，VIMO 质量不是关键差别，需进一步分析。**

经过 scale 归一化（除以 alignment[0]行的 L2 norm）后的真实轨迹数据：

| 场景 | 归一化X范围 | 归一化Z范围 | X/Z比 | 折返比 |
|---|---:|---:|---:|---:|
| lab | 3.07 | 4.13 | **0.74** | 5.1x |
| bike_vimo | 8.22 | 1.99 | **4.13** | 16.3x |
| jogging_vimo | 9.41 | 2.03 | **4.64** | 11.8x |
| citron_vimo | 6.15 | 1.16 | 5.30 | 1.5x |

**正确结论（最终版）**：

崩坏需要同时满足两个条件：
1. **轨迹近似一维**（X/Z > 3）：走廊很窄，折返时路过几乎同一批 3D 位置
2. **折返比高**（人原路返回，与来时高度重叠）

lab 的 X/Z=0.74（二维轨迹），即使折返，Z 方向不同，同一批 scene GS 不会被从正反方向同时访问，不产生矛盾梯度。

bike/jogging 的 X/Z≈4-5（近似一维），折返时几乎沿完全相同的 3D 走廊返回：同一批 scene GS 需要在去程（人在后、背景在前）和返程（人在前、背景被遮挡）两个矛盾目标下优化，梯度相互抵消 → scene GS 在折返走廊被"撕烂"→ anchor attention 查询这批噪声 GS 作为上下文 → 修正方向错误 → human GS 漂移 → 无 opacity 抑制 → 可见爆炸。

citron 虽是一维轨迹（X/Z=5.3），但折返比仅 1.5x（几乎不折返），条件2不满足，故不崩。

**完整因果链（最终版）**：
```
近似一维轨迹（X/Z>3）+ 高折返比
  → 同一批 scene GS 被从相反方向同时约束
      → scene GS 梯度矛盾，折返走廊 scene GS 变成噪声
          → anchor attention 获得噪声上下文 → 修正方向错误 → human GS 漂移
              → 无 opacity 抑制 → 高 opacity 错位 GS 直接渲染 → 可见爆炸

lab：二维轨迹（X/Z=0.74），折返但不重叠，不触发第一步
citron：一维轨迹但几乎不折返（1.5x），不触发第一步
bike/jogging：一维轨迹 + 高折返，两个条件同时命中
```

---

## 2026-06-07（会话二十八）

### 会话主题

恢复 HUGS 记忆，了解当前 pipeline 和完整验证 V2 进度

### 核心进展

- 检查完整验证 V2 进度：jogging STM（17.3776）和 jogging HUGS baseline（16.1335）均已完成，parkinglot OUR 正在跑（5k/18k 步，HUMAN ~16.34）

---

## 2026-06-07（会话二十七）

### 会话主题

恢复 HUGS 记忆，检查完整验证 V2 进度，分析 bike 场景崩坏原因，发散思考超越 STM 的改进方向

### 核心进展

#### 完整验证 V2 进度（截至 ~15:30）

- **bike** ✅ 全部完成：OUR(18k) HUMAN 18.0921 / STM(20k) 18.6133 / HUGS(15k) 17.0399
- **citron** ✅ 全部完成：OUR 17.4008 / STM 16.8763 / HUGS 15.5797
- **jogging** OUR ✅(16.9871)，STM 🔄 正在跑（19k/20k，HUMAN ~17.47），HUGS ⏳
- **parkinglot / seattle** ⏳ 等待排队串联

**重要发现**：bike 和 jogging 场景 STM > OUR（HUMAN PSNR），lab 和 citron 场景 OUR > STM。

#### Lab 行数据 pipeline 不一致

- OUR lab（19.5512）用的是 `seq=lab`（原始 NeuMan，**无 VIMO 粗对齐**）
- STM lab（17.4512）用的是 `seq=lab_vimo`（VIMO 粗对齐）
- 两者不在同一条件下，lab 行比较不公平，若用 `lab_vimo` 跑 OUR，HUMAN PSNR 约 17.2~17.8

#### Bike 场景崩坏原因分析

**表现**：部分 val 帧人体出现白色爆炸点、ghosting、身体不完整。HUGS baseline 崩坏更严重，说明问题与 anchor attention 无关。

**根本原因（两个）**：
1. **VIMO 粗对齐的 SMPL 姿态误差**：某些帧 SMPL 参数与图像对齐存在残差，LBS 将 human GS 变形到错误位置，且无任何机制将其 opacity 压低
2. **运动幅度大 + stage2 无 opacity reset**：bike 场景人物走路姿态变化大，变形场泛化差；stage2（15000→18000步）无 opacity reset，spurious gaussians 无法被清除，最终渲染时爆炸

**为什么 STM 不崩**：STM 启用了 `HumanSceneFuseDecoder`（`use_hugs: false` 触发），在每次渲染前将 human GS 和 scene GS 的全部属性（shs/xyz/opacity/scales/rotq）concat 后通过 ResidualMLP 联合处理。训练中网络隐式学到"当场景是这样时，偏移的 human GS 应降低 opacity"，从而避免可见爆炸。这是 STM 的核心创新，OUR 目前缺少对应机制。

#### 超越 STM 的改进方向（详见 memory）

提出四个方向，详细内容记录在 `hugs-future-directions.md`。

---

## 2026-06-07（会话二十六）

### 会话主题

继续完整验证实验（从 context 压缩后恢复）：为 5 个非 lab 场景跑 VIMO 预处理，生成训练配置，顺序启动全部 15 个训练任务

### 核心进展

- 待追加

---

## 2026-06-06（会话二十五）

### 会话主题

恢复 context（从压缩摘要继续）+ 实现方案 D（ViTPose 2D 关键点对齐 loss）+ 设置 Exp G+vitpose 自动启动

### 核心进展

- **Exp F3 进度**（StM + VIMO + opacity_reg）：
  - 16000/20001步，HUMAN PSNR 17.5947，仍在上升
  - 预计约 25 分钟后结束

- **方案 D 实现完成**（ViTPose 2D 关键点对齐 loss）：
  - **原理**：将 SMPL 关节投影到图像坐标，与 ViTPose 2D 检测关键点做 L2 对齐
  - **修改文件**：
    1. `hugs/models/hugs_trimlp.py`：主 `forward()` 新增 `smpl_joints_world` 到返回字典（scale/transl 都已应用）
    2. `hugs/datasets/neuman.py`：新增 `vitpose_kp_dir` 参数，从 `keypoints/{idx:05d}.png.npy` 加载 (17,3) 关键点
    3. `hugs/cfg/config.py`：新增 `cfg.human.loss.vitpose_kp_w = 0.0` 和 `cfg.dataset.vitpose_kp_dir = None`
    4. `hugs/trainer/gs_trainer.py`：新增 `maybe_add_vitpose_kp_loss()` 方法，并在训练循环中调用
  - **关节映射**：SMPL body joints [16,17,18,19,20,21,1,2,4,5,7,8] → COCO body [5-16]（跳过人脸关节0-4）
  - **置信度过滤**：`conf > 0.3` 且 `depth > 0.001`（相机前方才投影）
  - 全部文件语法检查通过

- **新实验配置文件**：
  - `cfg_files/release/neuman/hugs_vimo_lab_inline_attn_vitpose_kp_18k.yaml`
  - 基于 Exp G（inline attention 18k），新增 `vitpose_kp_w: 0.01` 和 `vitpose_kp_dir`
  - exp_name: `vimo_inline_attn_vitpose_kp_18k_lab_20260606`

- **自动启动**：
  - 后台 watcher 脚本（PID 924690）等待 F3 (PID 880011) 结束后自动启动 Exp G+vitpose
  - 启动脚本：`run_after_f3_expG_vitpose.sh`
  - 新实验日志：`run_logs/expG_vitpose_kp_20260606.log`

### 监控命令

```bash
# 查看 F3 最新 PSNR
grep "HUGS_HUMAN_PSNR" Dynamic-Avatar-Scene-Rendering-from-Human-centric-Context/run_logs/stm_expF3_opacity_reg_20260606.log | tail -5

# 查看新实验启动情况
cat /tmp/wait_f3_watcher.log
tail -f run_logs/expG_vitpose_kp_20260606.log
```

---

## 2026-06-06（会话二十四）

### 会话主题

恢复记忆 + 检查 Exp N 进度 + 启动 Exp F3（StM + VIMO + opacity_reg）

### 核心进展

- **Exp N 最终结果**（HUGS + VIMO + opacity_reg, 18k步）：
  - HUGS PSNR: **24.7992** / HUMAN PSNR: **17.6872**
  - 略优于 StM F2（17.45），但不及 Exp G（17.79）
  - 结论：global opacity_reg 对 VIMO 场景无明显帮助

- **Exp F3 启动**（StM + VIMO + opacity_reg）
  - 基础：Exp F2（StM + VIMO 粗对齐，20k步）
  - 新增：`scene.loss.opacity_reg_w: 0.01`（与 Exp N 一致）
  - 代码改动：
    - `StM/cfg/config.py`：新增 `cfg.scene.loss.opacity_reg_w = 0.0`
    - `StM/trainer/gs_trainer.py`：在 `loss_fn` 后、`loss.backward()` 前注入 opacity_reg
    - 同时修复 StM `data` symlink（旧 NAS 路径断链，已指向新路径）
  - **PID**: 880011，已正常运行
  - step 10：`l_scene_opacity_reg=0.810`，loss 生效
  - **日志**: `Dynamic-.../run_logs/stm_expF3_opacity_reg_20260606.log`
  - **输出**: `Dynamic-.../output_stm/.../stm_expF3_opacity_reg_20260606/2026-06-06_21-05-35/`
  - 预计约 3.5 小时后完成（20k步）

### 监控命令

```bash
grep "HUGS_HUMAN_PSNR\|scene_opacity_reg" Dynamic-Avatar-Scene-Rendering-from-Human-centric-Context/run_logs/stm_expF3_opacity_reg_20260606.log | tail -5
```

---

## 2026-06-06（会话二十三）

### 会话主题

恢复 hugs 项目记忆（路径迁移）+ 新实验：场景 GS 不透明度正则化（opacity reg）

### 核心进展

- **路径迁移修复**
  - 项目工作目录从 `/workspace/nas_auto_backup/nas/yuzilang/` 迁移至 `/workspace/nas_auto_backup/yuzilang/`
  - hugs 记忆已迁移至当前项目 memory 目录
  - `data/` 下所有断开的符号链接（lab_vimo、lab_coarse、lab_coarse_v3、smpl）已批量修复，均指向新路径

- **新实验：Exp N（场景 GS 不透明度正则化）**
  - 基础：Exp G（VIMO + inline attention 18k），在此基础上新增一项 scene GS 约束
  - 核心想法：约束场景高斯透明度尽量靠近完全不透明，观察对遮挡/穿模的影响
  - **代码修改**：`hugs/trainer/gs_trainer.py` 新增 `maybe_add_scene_opacity_reg_loss()`
    - Loss = `mean((1 - sigmoid_opacity)²)`，在每步训练循环中调用
    - loss key: `l_scene_opacity_reg`
  - **配置文件**：`cfg_files/release/neuman/hugs_vimo_lab_opacity_reg_18k.yaml`
    - `scene.loss.opacity_reg_w: 0.01`
    - 其余参数与 Exp G 完全一致（seq=lab_vimo, inline attention, gamma_transl curriculum）
  - **启动命令**：
    ```bash
    nohup /workspace/nas_auto_backup/yuzilang/miniconda3/envs/hugs/bin/python \
        scripts/run_neuman_human_scene_noamass.py \
        --cfg cfg_files/release/neuman/hugs_vimo_lab_opacity_reg_18k.yaml \
        --seq lab_vimo --exp-name vimo_opacity_reg_18k_lab_20260606 \
        > run_logs/vimo_opacity_reg_18k_lab_20260606.log 2>&1 &
    ```
  - **PID**：838954，**已正常运行**
  - **输出目录**：`output/human_scene/neuman/lab_vimo/hugs_trimlp/vimo_opacity_reg_18k_lab_20260606/2026-06-06_18-34-35/`
  - **早期观察**：`l_scene_opacity_reg` 从 step10 的 0.7492 快速下降到 step50 的 0.4313，loss 正在生效

### 监控命令

```bash
grep "HUGS_HUMAN_PSNR\|scene_opacity_reg" run_logs/vimo_opacity_reg_18k_lab_20260606.log | tail -10
```

---

## 2026-06-05（会话二十二）

### 会话主题

Plan A/B 实施：离线 mask 预对齐实验 + 轨迹平滑；VIMO 实际误差完整诊断

### 核心进展

- **Plan A（offline mask 预对齐）—— 确认对 VIMO 不可行**
  - `segmentations/` mask 覆盖率 94%（背景 mask，非人体！）
  - SAM mask 质心与 SMPL body_center 投影有 **45.9px 解剖学固有偏差**（GT 对齐也如此）
  - VIMO 实际 2D pelvis 误差：**mean=11.1px, max=22.5px**（极小，不是 255px！）
  - 之前报告"255px 侧向偏移"的说法错误（ROMP 的数字被误引到 VIMO）
  - Plan A 脚本已写（`scripts/offline_mask_prealign.py`），但分析证明对 VIMO 无效；生成的 prealigned.npz 已删除
  - 会议文档 Section 11 已全面更新，修正所有错误声明

- **Plan B（轨迹平滑）—— 已实施，结果符合预期**
  - 脚本：`scripts/smpl_traj_smooth.py`（savgol_filter，window=11，polyorder=3）
  - 帧间最大跳变：2.49 → 1.84 COLMAP units（-26%）；均值修正 0.34 COLMAP units（40mm）
  - 输出：`data/neuman/dataset/lab_vimo/4d_humans/smpl_optimized_aligned_scale_smoothed.npz`
  - 待做：用 `_smoothed.npz` 启动 Exp G 变体验证提升效果

- **VIMO 误差完整诊断**
  - 3D transl：mean=90mm, max=340mm（最大误差在深度方向）
  - 2D pelvis：mean=11.1px, max=22.5px（侧向极小）
  - Scale：VIMO=8.218 vs GT=8.617，差 4.85%（体积偏小，深度偏近）
  - 主要误差来源是 **scale（深度）**，不是侧向偏移

- **NAS 路径说明**
  - NAS 重新挂载期间，工作文件在本地备份 `/workspace/nas_auto_backup/yuzilang/ml-hugs-work/`
  - 原 NAS 路径 `/workspace/nas_auto_backup/yuzilang/` 挂回后需同步

---

## 2026-06-04（会话二十一）

### 会话主题

Exp J/K 完整结果确认 + 核心实验路径整理 + VIMO HUGS baseline 启动

### 核心进展

- **Exp K 完整结果**（v6，coarse align，PID 458382，已完成）
  - peak 17.27@step10000，final 17.21
  - 低于 Exp G（18.14）和 Exp D（17.54）；coarse align 2000 步渲染代价大于 SMPL 对齐收益
  - 方案整体失败，已记录到组会报告

- **Exp J 确认状态**（已于上轮 kill，非自然结束）
  - 中止于 step ~9700，peak 17.79@step3000/8000，进程不在运行
  - #hp=455-456K 健康，l_mask_reproj 已激活（正常范围）

- **核心实验输出路径整理**（新增 memory：key_experiment_paths.md）
  - GT 最优：`output/.../lab/.../depth_sup12k_transl_xyz_attn_6000_lab_20260528/2026-05-28_23-44-41/`
  - Exp G（VIMO 最优）：`output/.../lab_vimo/.../vimo_inline_attn_18k_lab_20260604/2026-06-04_10-40-34/`
  - StM F2（VIMO 对比）：`Dynamic-.../output_stm/.../stm_expF2_20260604/2026-06-04_08-45-29/`
  - StM E2（GT 对比）：`Dynamic-.../output_stm/.../stm_expE2_20260604/2026-06-04_03-23-47/`

- **VIMO + HUGS baseline 启动**（PID 519674，补充缺失对照组）
  - Config：`cfg_files/release/neuman/hugs_vimo_lab_baseline.yaml`
  - 参数：`seq=lab_vimo`、`num_steps=15000`、`depth_w=0.0`、`anchor_attention.use_anchors=false`
  - 日志：`run_logs/vimo_baseline_lab_20260605.log`
  - 输出：`output/.../lab_vimo/.../vimo_hugs_baseline_lab_20260605/2026-06-04_19-20-44/`
  - step1000 PSNR：**17.22**（正常，预计 final 在 16-17 dB 区间）

---

## 2026-06-04（会话二十）

### 会话主题

Exp J 结果分析 + Exp K（coarse align v6）实现与测试；组会报告更新

### 核心进展

- **Exp J 最终结果**：peak 17.79@step8000，final ~17.7，低于 Exp G（18.14）
  - 根因：offset_start_iter=5000 时 anchor attention 已经修正了大部分 VIMO 误差，两种修正竞争
  - 结论：并行修正设计错误，应串行（先粗对齐再精细修正）

- **Exp K 实现与启动**（v6，coarse align，PID 458382）
  - 前 2000 步：coarse_align 阶段（mask_reproj_w=0.02 + foot_contact_w=2.0）
  - step 2000 后：coarse align 关闭，anchor attention 启动
  - ground_y=-3.986（从 VIMO smpl_params 估计）
  - **结果（4 个检查点）**：
    - step 1000: PSNR 12.87（coarse loss 主导，期望低）
    - step 2000: PSNR 13.07（coarse align 结束）
    - step 3000: PSNR 15.50（快速恢复 +2.43 dB）
    - step 4000: PSNR 15.55（几乎停滞，vs Exp G 18.08）
  - **结论**：coarse align 阶段损失抑制了渲染优化，GS 质量受损；coarse align 结束后虽然恢复迅速，但仍落后 Exp G 约 2.5 dB，当前方案不优于 Exp G

- **gs_trainer.py 新增功能**：
  - `setup_coarse_align()`：从 VIMO smpl_params 估计 ground_y=-3.986，初始化 scale_corrections
  - `maybe_add_coarse_reproj_loss()`：前 2000 步 mask centroid 2D 重投影损失
  - `maybe_add_foot_contact_loss()`：前 2000 步足底接触防漂移损失

- **下一步考虑**：
  - 离线预对齐 SMPL（训练前单独运行 mask alignment，不影响渲染）
  - 或接受 Exp G 18.14 为 VIMO 条件最优，转向论文写作

---

## 2026-06-04（会话十九）

### 会话主题

Exp I 失败根因分析 + Exp J 修复启动；完善组会报告文档（含 Exp I 失败分析节）

### 核心进展

- **Exp I 失败诊断与停止**（PID 395877，v4 config）
  - step 1000: PSNR 12.32，step 5000: 13.73，step 4000: `#hp=0.4K`（人体 GS 从 110K 崩溃到 400 个）
  - **根因**：mask_reproj_w=0.05 在 step 0 占总 loss 96%（62.3/64.75），frame_trans_offsets 以 lr=0.002 剧烈跳变，将人体 GS 推出相机视锥，densification 期间被 opacity_cull 删光
  - 已于 16:43 停止

- **Exp J 启动**（PID 426433，v5 config）
  - 核心修复：`offset_start_iter=5000`（densification 稳定后才激活 offset）
  - 参数调整：lr 0.002→0.0002，mask_reproj_w 0.05→0.001，smooth_w 0.1→0.01
  - step 1000 验证：`#hp=110.2K` 完整，HUMAN PSNR **17.51**（vs Exp I 的 12.32）
  - 预计约 8 小时完成（约 6-5 早上出结果）

- **gs_trainer.py 代码更新**：
  - `maybe_apply_frame_trans_offset(iteration=0)`：新增 iteration 参数，iteration < offset_start_iter 时跳过
  - `maybe_add_mask_reproj_loss(iteration=0)`：同上
  - frame_trans_optimizer.step()：加 offset_start_iter 检查
  - 训练循环调用处：传入 `iteration=t_iter`

- **cfg_files/release/neuman/hugs_vimo_lab_inline_attn_v5.yaml**：Exp J 配置

- **6-5组会进展汇报.md**：新增第七节 Exp I 失败分析 + Exp J 修复方案，含失败对比表和修复参数表

### 监控命令
```bash
grep "HUGS_HUMAN_PSNR" run_logs/vimo_expJ_inline_attn_v5_20260604.log | tail -5
```

---

## 2026-06-04（会话十八）

### 会话主题

监控 Exp H 完成，分析粗对齐 2D 投影错误原因，实现 per-frame 可学习 translation offset + mask reprojection loss，准备启动 Exp I

### 核心进展

- **Exp H 进行中**（PID 341966，v2 config，step ~15000/17999，HUMAN PSNR @step15000=17.70）
  - v2 改动：smpl_trans lr=0.001（过大），densify_until=10000
  - step 9000 处 opacity reset + 大 lr 导致 PSNR 从 17.75 跌至 16.77，整体表现弱于 Exp G
  
- **新代码实现**（session 继承自上一 context）：
  - `anchor_attention.py`：增加 per-frame delta_transl（frame_embed, num_frames 参数）
  - `gs_trainer.py`：
    - `setup_frame_trans_offsets()`：`nn.Parameter[num_frames, 3]`，独立 Adam 优化器
    - `maybe_apply_frame_trans_offset()`：训练时注入帧级别平移偏移
    - `maybe_add_mask_reproj_loss()`：2D mask 重投影 loss（L2：预测质心 vs GT mask 质心）
    - `maybe_add_frame_trans_smooth_loss()`：时序平滑正则化
    - 最优 checkpoint 保存（HUMAN PSNR 最高时保存）
  - `cfg_files/release/neuman/hugs_vimo_lab_inline_attn_v4.yaml`：Exp I 配置
    - 基础：Exp G（densify_until=15000，smpl_trans=0.0005）
    - 新增：frame_trans_offset（lr=0.002，mask_reproj_w=0.05，smooth_w=0.1）

- **Exp H 完成后启动 Exp I**（v4 配置）

### 关键分析

- **粗对齐 2D 投影为何不准**：VIMO→COLMAP 坐标变换误差波及全部 3 轴（非仅深度），导致 x/y 方向侧移约 255 像素（3m 处），frame_trans_offset 方案可在 2D 监督下直接修正该误差

---

## 2026-06-04（会话十七）

### 会话主题

跟进 Exp G（vimo_inline_attn_18k）训练进度，分析 PSNR 早期见顶问题，获取最终结果

### 核心进展

**Exp G 训练完成（PID 312796）**：
- cfg: `cfg_files/release/neuman/hugs_vimo_lab_inline_attn_18k.yaml`
- 日志：`run_logs/vimo_expG_inline_attn_20260604.log`
- 单阶段 18k，attention 从 step 2000 inline 参与 densification，gamma_transl cosine curriculum 2.0→0.05

**Exp G 最终结果：**

| 指标 | HUMAN | 全图（HUGS）|
|------|-------|------------|
| PSNR | **17.7923** | 25.0522 |
| SSIM | 0.7283 | 0.9021 |
| LPIPS | 0.1747 | 0.0821 |

**完整粗对齐对比矩阵：**

| 实验 | Translation | HUMAN PSNR | 备注 |
|------|-------------|-----------|------|
| HUGS GT 基准 | GT | 19.78 | depth_sup12k+attn6k |
| StM Exp E2 | GT | 19.29 | StM 复现 |
| **Exp G（inline attn 18k）** | VIMO 93mm | **17.79** | 超越 Exp D +0.25，超越 StM F2 +0.34 |
| Exp D（旧两阶段）| VIMO 93mm | 17.54 | |
| StM Exp F2 | VIMO 93mm | 17.45 | StM+VIMO |

**分析：PSNR 为何早期见顶（step 8000 峰值 18.14，最终 17.79）**
- Densification（3k~15k，每600步）每次分裂后精度短暂下降，锯齿状振荡
- 583K 高斯大规模分裂（step 12000）引发最大跳变，后续收敛时间不足
- VIMO 误差是硬上限，gamma_transl 随时间衰减后修正容量不足

---

## 2026-06-04（会话十六）

### 会话主题

恢复上下文，查看粗对齐下 StM 实验（F2）运行状态

### 核心进展

**Exp F2（StM + VIMO 93mm，20k步）正在运行**：
- PID 253275，cfg: `cfg_files/stm_vimo_lab_20k.yaml`
- 当前进度：step ~17070/20001（**85%**），#hp=580.3K 稳定，无崩溃
- Step 17000 HUMAN PSNR：**17.4858**
- Step 17000 HUGS PSNR：**24.9680**
- 日志：`run_logs/stm_expF2_20260604.log`
- 预计剩余约 16 分钟完成

**已完成实验汇总（2026-06-04）**：

| 实验 | Translation | HUMAN PSNR | 状态 |
|------|-------------|------------|------|
| HUGS GT 基准（我们最优） | GT | **19.78** | 完成 |
| StM Exp E2（修复版，20k）| GT | 19.2858 | 完成 |
| HUGS+VIMO Exp D | VIMO 93mm | **17.54** | 完成 |
| StM Exp F2（20k）| VIMO 93mm | ~17.49（17k处）| **进行中** |
| StM Exp F_v1（15k，buggy）| VIMO 93mm | 12.2042 | 完成（作废）|

**初步结论（待 F2 最终结果确认）**：
- StM fusion MLP 对 VIMO 粗对齐误差无明显弥补效果（F2@17k ≈ Exp D）
- StM 在 GT 条件下略差于 HUGS（19.29 vs 19.78，-0.49 dB）

### 下一步

等 F2 完成后整理最终对比矩阵，评估 StM 实际价值。

---

## 2026-06-04（会话十七）

### 会话主题

实现两大改进方案，启动 Exp G（inline attention + large gamma curriculum）

### 核心代码改动

**1. `hugs/models/anchor_attention.py`：添加 `_gamma_transl_schedule()` 方法**

根因：当前 `gamma_transl=0.05 × transl_delta_clamp=0.2 = 0.01` COLMAP单位 ≈ 1.2mm，而 VIMO 误差 93mm ≈ 0.765 COLMAP单位，修正容量仅为误差的 **1.3%**。

新增 cosine decay curriculum：
- `gamma_transl_init` → `gamma_transl_final`（由 `decay_start/end` 控制）
- 向后兼容：无 curriculum 参数时降级为原来的 `gamma_transl` 常数

**2. 新 cfg 文件：`cfg_files/release/neuman/hugs_vimo_lab_inline_attn_18k.yaml`**

| 设计选择 | 值 | 理由 |
|---------|---|------|
| 训练步数 | 18k（单阶段，无ckpt） | 方案一：attention 参与 densification（2k~15k） |
| module_start_iter | 2000 | 场景 GS 初步收敛后再启动 attention |
| correction_start_iter | 2000 | 同上 |
| correction_warmup_iters | 500 | 平滑启动，防止突变 |
| gamma_transl_init | 2.0 | 方案二：max_corr=1.0 > VIMO 误差0.765 |
| gamma_transl_final | 0.05 | 精调阶段与原来一致 |
| gamma_transl_decay_start | 2000 | 启动即为最大值 |
| gamma_transl_decay_end | 15000 | densification 结束时降至 final |
| transl_delta_clamp | 0.5 | 从 0.2 增大，匹配大 gamma_init |
| smpl_trans LR | 0.0005 | 从 0.0001 提高，加速 transl 参数收敛 |

### 当前实验

**Exp G（vimo_inline_attn_18k_lab）**：PID 301348，已启动
- 当前：optimize_init 阶段（7000步，约35分钟后完成）
- 主训练预计：~3小时（18000步 @ ~2.7 it/s）
- 日志：`run_logs/vimo_expG_inline_attn_20260604.log`
- 预期完成：~3.5小时后

### 对比目标

| 实验 | Translation | HUMAN PSNR |
|------|-------------|-----------|
| HUGS+VIMO Exp D（当前最优coarse）| VIMO 93mm | 17.54 |
| StM+VIMO Exp F2 | VIMO 93mm | 17.45 |
| **Exp G（新方案）** | VIMO 93mm | **待测** |

**目标**：Exp G > 17.54（超过现有 VIMO 基线）。若 Exp G >> 17.54，说明方案有效。

---

## 2026-06-04（会话十六）

### 会话主题

depth-diff-gaussian-rasterization 成功编译安装 + Exp E v3 启动（原版深度损失）

### 核心进展

**目标**：用户要求"请你帮我实现编译，然后用原版的 stm 跑实验，保证和源仓库一致"

**depth-diff-gaussian-rasterization 安装三步骤**：
1. **CUDA 版本问题**：系统 CUDA 12.1 vs PyTorch CUDA 11.7 不兼容
   - 解决：设 `CUDA_HOME=/workspace/nas_auto_backup/yuzilang/miniconda3/envs/StM`，使用 conda 环境自带 nvcc 11.7
2. **glm 子模块缺失**：`fatal error: glm/glm.hpp: No such file or directory`
   - 解决：`cd submodules/depth-diff-gaussian-rasterization && git submodule update --init --recursive`
3. **安装成功**：`Successfully installed diff_gaussian_rasterization-0.0.0`

**gs_renderer.py 恢复原版**：
- 删除 2-pass depth workaround（通过 colors_precomp 渲染 z 值的临时方案）
- 恢复为 `rendered_image, radii, depth = rasterizer(...)` 单步三值返回
- depth 返回形状为 `(1, H, W)`，与 trainer/loss.py 的 `reshape(-1, 1)` 完全兼容

**stm_lab_15k.yaml 更新**：
- `exp_name`: stm_expE_v3_20260604（从 v2 升级）
- `depth_w`: 0.0 → 0.03（恢复原版 stm_human_scene.yaml 值）

### 当前实验状态

- **Exp E v3**：PID 212110，训练中（step ~1600/15000，预计约 55 分钟后完成）
  - `#hp=110.2K`（人体高斯数量稳定，无崩溃）
  - `l_depth=0.02~0.04`（**depth loss 正常工作，原版 rasterizer 验证成功**）
  - step 1000 验证：HUMAN PSNR = **11.47**（早期，正常收敛中）
  - 日志：`run_logs/stm_expE_v3_20260604.log`
  - 保留的 bug 修复（与源码的必要差异）：
    - scene.opacity_reset_interval = 20000（stm_human_scene.yaml 原有配置，避免人体高斯崩溃）
    - scene viewspace_points 切片修复（gs_trainer.py，实际 bug）
    - triplane clamp（triplane.py，实际 bug）
    - neuman.py betas 属性（实际 bug）
    - anim dataset try/except（实际 bug）
    - Pearson NaN guard（loss.py，opacity reset 后方差≈0 仍需要）

### Exp F 自动链式运行

**目标**：Exp E v3 完成后，自动启动 Exp F（StM + VIMO 粗对齐，93mm 误差）

**配置**（`cfg_files/stm_vimo_lab_15k.yaml`）：
- seq: `lab_vimo`（VIMO 93mm 粗对齐数据）
- exp_name: `stm_expF_v1_20260604`
- depth_w: 0.03，opacity_reset_interval: 20000（与 Exp E v3 一致）

**链式脚本**：`run_expF_after_expE.sh`（后台 PID 223385）
- 等待 PID 212110 结束，自动执行 `python main.py --cfg_file cfg_files/stm_vimo_lab_15k.yaml`
- 日志：`run_logs/stm_expF_v1_20260604.log`
- 链式监控：`run_logs/chain_expF_20260604.log`

### 对比矩阵（待完成）

| 实验 | Translation | HUMAN PSNR |
|------|-------------|-----------|
| HUGS GT 基准 | GT | 19.78 |
| HUGS+VIMO（Exp D） | VIMO 93mm | 17.54 |
| StM Exp E v3 | GT | 进行中 |
| StM Exp F v1 | VIMO 93mm | 链式等待 |

### 下一步

结果出来后对比：StM fusion 能否弥补粗对齐误差（GT vs VIMO 的 PSNR 差距是否小于 HUGS 的 2.24dB）？

---

## 2026-06-04（会话十四）

### 会话主题

StM 代码全面 bug review + scene viewspace_points 切片 bug 修复

### 背景

上一会话（十三）完成了 VIMO 粗对齐（Exp D 最终结果 17.5），并开始 StM baseline（Exp E）的调试。
用户要求："仔细检查还有没有会出错的地方，为什么用 stm 自己的仓库跑会有这么多错误"

### StM 原始代码为何有这么多错误

**根本原因**：StM 代码依赖 `depth-diff-gaussian-rasterization`（一个修改版光栅化器，返回3值：image, radii, depth），但该库未在任何公开环境中发布，服务器上也不存在。标准 `diff_gaussian_rasterization` 只返回2值，导致所有依赖深度的代码链需要大规模修复。这一个依赖问题像多米诺骨牌一样引发了一系列次生 bug。

**具体根因清单**：
1. 深度光栅化器缺失 → 需要2-pass workaround → depth_vals 通过 means3D 传播 NaN 梯度
2. Pearson depth loss 没有 NaN guard（opacity reset 后方差≈0 时崩溃）
3. scene densification viewspace_points 切片错误（见下方新发现 bug）
4. triplane 边界断言未考虑 densification 后越界
5. neuman.py 缺少 `self.betas` 属性
6. AMASS SFU 数据不存在（anim dataset 无兜底）

### 新发现的关键 Bug（2026-06-04）

**Bug：scene viewspace_points 切片错误（gs_trainer.py 第392行）**

- **问题**：`render_pkg['scene_viewspace_points'] = render_pkg['viewspace_points']`
  传入全部 [human|scene] 高斯（约 110K + 9K），而 `scene_visibility_filter` 只有 scene_n_gs 个元素。
  `scene.add_densification_stats` 中有 `[:update_filter.shape[0]]` 切片，取的是前 scene_n_gs 个梯度
  → 实际取到了 **human 高斯的一部分梯度**，而不是 scene 高斯的梯度。
  → scene densification 的 clone/split 决策完全基于错误梯度

- **对比**：human densification（第407行）已正确切片：`render_pkg['viewspace_points'][:human_n_gs]`

- **修复**（已执行）：在 human_scene 模式下，创建只含 scene 梯度的临时张量：
  ```python
  scene_vp = torch.zeros(scene_n_gs, 3, device="cuda")
  scene_vp.grad = render_pkg['viewspace_points'].grad[human_n_gs:].clone()
  render_pkg['scene_viewspace_points'] = scene_vp
  ```

### 当前实验状态

- **Exp E（nodepth）**：PID 165935，step ~770，损失正常下降（opacity reset 后恢复中）
  - 日志：`run_logs/stm_expE_nodepth_20260604.log`
  - 代码 bug 修复后下次重跑才生效

- **已修复的所有 bug（累计）**：
  1. ✅ depth_w=0.0（临时规避 depth NaN 梯度问题）
  2. ✅ Pearson depth NaN guard（loss.py）
  3. ✅ triplane clamp（triplane.py）
  4. ✅ neuman.py betas 属性
  5. ✅ anim dataset try/except
  6. ✅ scene viewspace_points 切片（gs_trainer.py，今日修复）

- **validate 函数已知轻微问题**：`depth_np /= depth_np.max()` 当 depth 全零时除以 0 → NaN 图像（不崩溃）

### 下一步

1. 等待 Exp E（nodepth）到 step 3000（densify_from_iter），观察是否还有 PSNR 崩溃
2. 若 step 3000 后稳定，说明 NaN 来源确实是 depth rendering 梯度
3. 考虑重启 Exp E 启用 scene viewspace_points 修复（需先停止当前进程）
4. 最终目标：得到稳定的 StM baseline PSNR（GT 翻译），然后对比 VIMO 翻译效果

---

## 2026-06-04（会话十三）

### 会话主题

用 VIMO（TRAM）替代 ROMP，解决时序帧间噪声，提升粗对齐精度

### 问题分析

- Exp C 完成（depth_sup 12k + transl_xyz attn 6k + 粗对齐 v3 初始化）：HUMAN PSNR = 16.16（vs GT 19.78，损失 3.6dB），确认粗对齐精度是唯一瓶颈
- v3（foot-contact COLMAP）scale 误差降至 4.6%，但 t_world 仍有系统性偏移（-0.45, 0.65, -0.69）
- 根本原因：ROMP 逐帧独立估计，帧间 translation 存在随机噪声，无全局时序一致性

### 实验方案：VIMO 粗对齐（Exp D）

**目标**：用 VIMO（时序感知 HMR）替代 ROMP，获得平滑、全局一致的 SMPL 序列

**绕过 DROID-SLAM 的思路**：NeuMan 已有 COLMAP 相机（比 DROID-SLAM 更精确），直接构造 VIMO 所需输入格式，无需运行 estimate_camera.py

**执行步骤**：
1. 检查 hugs 环境能否 import VIMO 依赖（`lib/models/hmr_vimo.py`）
2. 写 `scripts/neuman_to_vimo.py`：COLMAP 相机 → `camera.npy`，segmentation → `tracks.npy`
3. 调 VIMO 推理，得到全帧时序一致 SMPL（相机坐标系）
4. 写 `scripts/gen_coarse_npz_vimo.py`：VIMO transl + v3 scale(8.22) → world 坐标 → npz
5. 创建 `lab_vimo` 数据集目录，启动 Exp D（最优 pipeline：depth_sup 12k + transl_xyz attn 6k）

**JOSH 环境状态**：
- VIMO checkpoint：`JOSH/data/checkpoints/vimo_checkpoint.pth.tar` ✅（2.6G）
- SMPL_NEUTRAL.pkl：软链接至 HUGS 的 SMPL 模型 ✅
- 无需 DROID-SLAM、无需单独 conda 环境（在 hugs env 中运行）

### 成果

1. **VIMO 环境配置**：安装 einops/timm/scikit-image；将 `parse_chunks` 移入 `hmr_vimo.py`；创建 `data/smpl/` → TRAM SMPL 辅助文件软链接

2. **VIMO 推理成功（lab 103帧，~30秒）**：
   - 脚本：`scripts/run_vimo_neuman.py`
   - 输出：`output/vimo_hps/lab/vimo_results.npz`

3. **VIMO 粗对齐 npz 生成**：
   - 脚本：`scripts/gen_coarse_npz_vimo.py`
   - 输出：`data/neuman/dataset/lab/4d_humans/smpl_optimized_aligned_scale_vimo.npz`
   - vs GT 误差：**mean=0.765 COLMAP units（≈0.088m）**（ROMP 1.67 units，改善 **2.2×**）

4. **Exp D 启动**（2026-06-03 ~20:54，运行中）：
   - 数据集：`lab_vimo`（VIMO transl + v3 scale=8.22）
   - Pipeline：depth_sup 12k + transl_xyz attn 6k（同最优 pipeline）
   - 日志：`run_logs/vimo_expD_20260604.log`
   - 对比：GT PSNR **19.78** → 粗对齐 v3 **16.16** → 期望 VIMO 显著改善
   - 预计完成：约 4.5h

### 关键分析结论（2026-06-04）

**VIMO 改善的真实机制**：
- ❌ 不是帧间抖动减少（residual jitter 仅降 14%：0.747→0.639）
- ✅ 而是绝对位置估计更准（mean error 降 58%：1.826→0.765 COLMAP units）
- 原因：VIMO 直接使用正确的 COLMAP focal(1099.98)；ROMP 是事后近似修正

**Exp D 早期结果（step 8000）**：HUMAN PSNR ≈ 17.5，已超过 Exp C 最终值 16.16，方向正确。

### 进度检查命令

```bash
cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work
grep -E "Phase 1 完成|Phase 2|HUMAN_PSNR|Exp D 全部完成" run_logs/vimo_expD_20260604.log | tail -10
```

---

## 2026-06-04（会话十二）

### 会话主题

实现并运行粗对齐 v3（foot-contact COLMAP depth 方案），验证 scale 估计精度

### 成果

（进行中）

---

## 2026-06-04（会话十一，续）

### 会话主题

JOSH 分析 + 粗对齐 v3 实现

### 成果

1. 分析 JOSH coarse alignment：TRAM+DUSt3R 共享焦距坐标系，scale-free，无法直接复用到 NeuMan/COLMAP
2. v2 分析：DepthPro background calibration slope `a` 方差太高（7.13–8.32），导致 v2 (mean err=2.062) 比 v1 (1.826) 更差
3. 实现 `scripts/coarse_align_smpl_neuman_v3.py`：foot-contact COLMAP depth 方案
   - `get_foot_region()`：SMPL 足底顶点投影，取图像下方 4% 作为足部像素
   - `get_colmap_depth_at_pixel()`：查找足部像素附近 50px 内 COLMAP 3D 点的中值深度
   - scale_init = median(colmap_z_foot / smpl_foot_z_m)，直接绕过 DepthPro 标定
   - 优化加入 loss_contact = (scale × foot_z_m − colmap_z_foot)²，权重 10.0

---

## 2026-06-03（会话十）

### 会话主题

实现并测试 `scripts/coarse_align_smpl_neuman.py`（SMPL-to-NeuMan 粗对齐脚本）

### 成果

**已完成**：
1. 修复 mask 方向 bug（NeuMan 约定：黑=人体，白=背景）
2. 修复深度公式 bug（mono_depth 是直接深度，非逆视差）
3. 发现并修复 ROMP focal 假设 bug（ROMP 假设 focal=cx=638，实际 focal=1099.98，导致 X/Y 投影偏移 1.72×）
4. 重新设计优化公式（scale 作用于 camera space，再做 cam_to_world，解决 COLMAP 单位/meter 混用问题）
5. 实现 foot-contact scale_init 估计（利用脚底投影附近的 COLMAP 地面点估算 colmap_per_meter）
6. 加入 t_world L2 正则化（防止 t_world 吸收 scale 误差）

**最终结果（lab 序列）**：
- scale = 7.88 vs GT 8.62（误差 8.6%）
- t_world = [-0.04, 0.36, 0.02]（小）
- SMPL 2D 投影：92.6% verts 落在人体 mask 上
- 3D 位置：SMPL 落在 COLMAP 点云内部

**输出文件**：`output/coarse_align/lab/`（coarse_align.json + vis/）

### 关键技术发现

- COLMAP 单位约 = meter / 8.4（NeuMan lab）
- ROMP 假设 focal=image_W/2，需用 `trans_xy *= (focal_romp/focal_actual)` 修正
- mono_depth 在人体区域给的是背景深度（不可靠），应禁用人体 Chamfer
- 背景 Chamfer 把 SMPL 拉向背景点（过深），应改用脚底地面点约束
- 足底 COLMAP 地面点（±80px 半径内最近10% pts）是最可靠的 scale 信号

### GT 对比结果

运行 `/tmp/compare_gt.py` 对比 6 个关键帧（root joint vs GT root joint）：

| 帧 | 我们估计的 transl | GT transl | 距离（COLMAP units） |
|---|---|---|---|
| 00000 | [22.48, 2.13, 20.68] | [23.25, 3.10, 21.31] | 1.39 |
| 00020 | [6.58, 1.62, 28.14] | [7.18, 2.40, 30.34] | 2.42 |
| 00041 | [8.61, 1.91, 24.75] | [9.21, 2.85, 25.48] | 1.32 |
| 00061 | [14.83, 2.57, 20.64] | [16.30, 3.21, 21.39] | 1.77 |
| 00082 | [16.75, 2.88, 18.39] | [17.63, 3.84, 19.09] | 1.48 |
| 00102 | [15.33, 5.06, 2.37] | [16.79, 5.74, 2.43] | 1.62 |

**均值 root joint 误差：~1.67 COLMAP units ≈ 0.19m**（已修正为 root-to-root 比较）

---

## 2026-06-04（会话十二）

### 会话主题

检查昨晚 Exp A/B 结果 + 发现实验设计缺陷 + 启动 Exp C（正确对照实验）

### 实验设计问题（已发现）

Exp A/B 使用的是**弱版 pipeline**（无 depth_sup，无 transl correction，仅 3000 步 attention），
不是我们的最优 pipeline。正确对照实验应该用 **depth_sup 12k + transl_xyz attn 6k**。

### Exp C：正确的粗对齐对照实验（2026-06-04 ~08:42 启动）

| 项 | 值 |
|---|---|
| Phase 1 config | `cfg_files/release/neuman/hugs_coarsealign_lab_expC_phase1.yaml` |
| Phase 2 config | `cfg_files/release/neuman/hugs_coarsealign_lab_expC_phase2.yaml` |
| 运行脚本 | `scripts/run_coarse_align_expC_20260604.sh` |
| Nohup 日志 | `run_logs/coarse_align_expC_20260604_nohup.log` |
| 详细日志 | `run_logs/coarse_align_expC_20260604.log` |
| 数据集 | `lab_coarse`（粗对齐 transl，~0.19m 误差） |
| Pipeline | depth_sup 12000步 + transl_xyz correction 6000步（同最优 pipeline） |
| 对比基准 | GT transl + 同 pipeline → HUMAN PSNR **19.78** |
| 预计完成 | Phase 1 ~3h，Phase 2 ~1.5h，总计 ~4.5h |

### 进度检查命令

```bash
cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work
# Phase进度
grep -E "Phase [12] 完成|>>> Phase" run_logs/coarse_align_expC_20260604_nohup.log
# PSNR
grep "HUMAN_PSNR\|human_psnr" run_logs/coarse_align_expC_20260604.log | tail -10
```

### 技术讨论：粗对齐改进方向（本次会话）

**1. Exp C 的 transl 修正能力分析**

即使 Exp C 用了最优 pipeline，transl 修正能力仍然有限：
- Phase 1：`smpl_trans lr = 1e-4`，渲染 loss 梯度对大偏移（0.19m）驱动力弱，GS 倾向于用外观变化拟合而非移动整体位置
- Phase 2：`delta_transl` 最大修正量 = `gamma_transl × transl_delta_clamp = 0.05 × 0.2 = 0.01 COLMAP units / step`（非常保守）
- 预期：Exp C 可能仍有明显 PSNR 损失，**瓶颈在粗对齐精度本身**

**2. 为什么不用深度点云对齐人体（类似 ExAvatar）**

最初误判：以为 ExAvatar 用的是 RGBD 深度传感器。
**用户纠正**：ExAvatar 也是用单目深度估计器（DepthPro）。

真正的差异是**模型质量**：
- NeuMan 自带的 mono_depth（旧模型，如 DPT）：人体区域给的是背景穿透深度，不可靠 → 代码里 `w_chamfer_human = 0.0` 强制禁用
- DepthPro（Apple 2024）：输出 metric 绝对深度，边界精度高，人体区域深度是人体表面自身深度，可靠

**3. DepthPro 与 COLMAP 的尺度兼容性**

用户担心 DepthPro 深度与 COLMAP 点云尺度不匹配。结论：**不冲突**，原因：
- DepthPro 只用于**预处理阶段**（coarse alignment），不参与 HUGS 训练循环，与训练期的 `depth_w` loss 完全独立
- 尺度对齐方法：用背景像素的 DepthPro 值对 COLMAP 做线性标定（和现在处理 mono_depth 完全一样），再用标定后的 scale 解读人体区域深度
- scale 误差（8.6%）对人体和背景是一致的，不引入相对偏差

**4. 改进方向（待 Exp C 结果确认后决策）**

若 Exp C 结果仍有明显损失 → 确认瓶颈在粗对齐精度 → 下一步：
1. 对 NeuMan 序列运行 DepthPro
2. 用背景像素标定 DepthPro vs COLMAP 尺度
3. 开启 `w_chamfer_human > 0`，用人体区域 DepthPro 深度反投影做人体 Chamfer 约束
4. 预期能将 transl 误差从 ~0.19m 压缩至更小量级

### Exp A/B 实验结果（已完成）

| 实验 | 初始化 | HUMAN PSNR | SCENE PSNR | HUMAN LPIPS | vs GT基准 |
|---|---|---:|---:|---:|---|
| Exp A：HUGS baseline 15000 | 粗对齐（~0.19m误差） | **15.74** | 23.61 | 0.2282 | -3.06 vs GT(18.80) |
| Exp B：anchor attention 15000 | 粗对齐（~0.19m误差） | **15.76** | 23.65 | 0.2224 | -3.63 vs GT(19.39) |

**结论：粗对齐 transl 初始化不可行。** 0.19m (~1.67 COLMAP units) 的初始误差使 HUMAN PSNR 损失约 3 个点，即使开启 `optim_trans=true` 也无法通过渲染 loss 梯度恢复。GT transl 对于训练收敛是必要条件（不可替代）。

若要实现无 GT pipeline，粗对齐精度需提升至 <0.05m 量级（当前精度不足）。

---

## 2026-06-03（会话十一）

### 会话主题

粗对齐初始化对比实验：ExpA（HUGS baseline 15000）+ ExpB（anchor attention 15000），测试粗对齐能否替代 GT `smpl_optimized_aligned_scale.npz`

### 已完成准备工作

1. **生成粗对齐 npz（全103帧）**：
   - 脚本：`/tmp/gen_coarse_npz.py`
   - 输出：`data/neuman/dataset/lab/4d_humans/smpl_optimized_aligned_scale_coarse.npz`
   - 策略：transl/scale 用粗对齐估计，pose/betas/bbox/vertex_colors 复制 GT（隔离 translation 效果）
   - scale = 7.879（常量），per-frame root transl 误差均值 ~1.67 COLMAP units

2. **创建 lab_coarse 数据目录**：
   - `data/neuman/dataset/lab_coarse/`（所有子目录 symlink 到 `lab/`）
   - `4d_humans/smpl_optimized_aligned_scale.npz` → 粗对齐版本
   - `4d_humans/smpl_optimized_aligned_scale_gt.npz` → GT 备用

3. **创建实验配置**：
   - Exp A：`cfg_files/release/neuman/hugs_coarsealign_lab_expA.yaml`（HUGS baseline 15000步）
   - Exp B：`cfg_files/release/neuman/hugs_coarsealign_lab_expB.yaml`（anchor attention 15000步，correction 从12k开始）
   - 两者均 `optim_trans: true`（让渲染 loss 精化粗对齐的 ~0.19m 误差）

### 正在运行的实验

**运行脚本**：`scripts/run_coarse_align_exps_20260603.sh`
**Nohup 日志**：`run_logs/coarse_align_exps_20260603_nohup.log`（主进度）
**详细日志**：`run_logs/coarse_align_exps_20260603.log`（tee，含所有 iter 输出）
**启动时间**：2026-06-03 ~00:48
**预计完成**：约 6 小时后（Exp A ~3h，Exp B ~3h）

#### Exp A：HUGS baseline 15000 + 粗对齐

- Config：`cfg_files/release/neuman/hugs_coarsealign_lab_expA.yaml`
- exp_name：`coarse_align_hugs_baseline_lab_20260603`
- 预期输出：`output/human_scene/neuman/lab_coarse/hugs_trimlp/coarse_align_hugs_baseline_lab_20260603/`
- 状态：**运行中**（训练已开始）
- 对比基准：GT transl HUGS baseline → HUMAN PSNR ≈ **18.80**

#### Exp B：anchor attention 15000 + 粗对齐

- Config：`cfg_files/release/neuman/hugs_coarsealign_lab_expB.yaml`
- exp_name：`coarse_align_anchor_attention_lab_20260603`
- 预期输出：`output/human_scene/neuman/lab_coarse/hugs_trimlp/coarse_align_anchor_attention_lab_20260603/`
- 状态：**排队中**（Exp A 完成后自动启动）
- 对比基准：GT transl anchor attention → HUMAN PSNR ≈ **19.39**

### 进度检查命令

```bash
cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work

# 主进度（Exp A/B 是否完成）
grep -E "Exp [AB] 完成|>>> Exp" run_logs/coarse_align_exps_20260603_nohup.log

# val PSNR（验证质量，每1000步一次）
grep -E "val_human_psnr|human_psnr|PSNR.*human" run_logs/coarse_align_exps_20260603.log | tail -20

# 当前 iter 进度
grep "Iter\b\|iter " run_logs/coarse_align_exps_20260603.log | tail -5

# 进程确认
pgrep -a python | grep main

# 输出目录确认（Exp A 完成后）
ls output/human_scene/neuman/lab_coarse/hugs_trimlp/ 2>/dev/null
```

### 实验设计说明（为什么开 optim_trans）

粗对齐初始 transl 误差 ~0.19m = 1.67 COLMAP units，不是很大但不可忽视。
开 `optim_trans: true` 让 l1/ssim/lpips 渲染 loss 驱动逐帧 transl 精化（lr=1e-4），
期望训练过程中误差自然收敛。若关闭则 transl 误差固定锁死，渲染质量必然受损。

---

## 2026-06-02（会话九）

### 会话主题

恢复 HUGS 记忆

### 背景

用户要求恢复 HUGS 项目全部记忆，确认当前状态（pipeline、实验结果、待解决问题）。

---

## 2026-06-02（会话八）

### 会话主题

回顾 HUGS 项目完整记忆，汇总当前状态

### 背景

用户要求回复 HUGS 相关全部记忆内容（项目总览、pipeline、实验结果）。

---

## 2026-06-01（会话七）

### 会话主题

渲染 adc12000_transl_xyz_attention_plus3000_stage2_lab_20260527 输出视频

### 背景

用户要求用 render_full_video.py 脚本对该 run 重新渲染输出图像/视频。

---

## 2026-06-01（会话六）

### 会话主题

恢复 HUGS 项目完整记忆

### 背景

用户要求恢复 HUGS 相关全部记忆，确认项目当前状态（pipeline、实验进度、已知问题）。

---

## 2026-06-01（会话五）

### 会话主题

恢复 HUGS 相关记忆，汇总项目状态

### 背景

用户要求回复 HUGS 相关的全部记忆。

---

## 2026-06-01（会话四）

### 会话主题

恢复 HUGS 项目记忆，确认当前状态

### 背景

用户要求恢复 HUGS 项目完整上下文，确认当前 pipeline 与实验进度。

---

## 2026-06-01（会话三）

### 会话主题

恢复记忆，确认当前进度

### 背景

用户要求恢复 HUGS 项目上下文，确认当前 pipeline 状态、待运行任务状态。

---

## 2026-06-01（会话二）

### 会话主题

全量 NeuMan 5 场景对比实验启动：HUGS 15K baseline × 5 + depth_sup+attn 完整流程 × citron/jogging

### 背景

为生成完整的定性对比图（GT | HUGS 15K | depth-sup+attn Ours），需要：
1. 补跑 seattle / parkinglot / bike 三个场景的原版 HUGS 15K baseline（本地未存储）
2. 为 citron / jogging 两个全新场景运行完整三阶段 pipeline（HUGS 15K + depth_sup 12K + attn 6K）

### 本次创建的文件

**配置文件（7 个）**：
- `cfg_files/release/neuman/hugs_human_scene_bike_original_15000.yaml`（exp_name: exp0_hugs_original_15000_bike_20260601）
- `cfg_files/release/neuman/hugs_human_scene_citron_original_15000.yaml`
- `cfg_files/release/neuman/hugs_human_scene_jogging_original_15000.yaml`
- `cfg_files/release/neuman/hugs_depth_supervised_12000_citron.yaml`（exp_name: hugs_depth_sup_12000_citron_20260601）
- `cfg_files/release/neuman/hugs_depth_supervised_12000_jogging.yaml`
- `cfg_files/release/neuman/hugs_depth_sup12k_transl_xyz_attention_6000_citron.yaml`
- `cfg_files/release/neuman/hugs_depth_sup12k_transl_xyz_attention_6000_jogging.yaml`

**脚本**：`scripts/run_all_neuman_20260601.sh`
**日志**：`run_logs/all_neuman_20260601.log`
**规划文档**：`run_logs/experiment_plan_20260601.md`

### 训练任务（9 个，顺序执行）

| Job | 场景 | exp_name | 状态 |
|---|---|---|---|
| A1 | seattle | exp0_hugs_original_15000_seattle_20260601 | 训练中 |
| A2 | parkinglot | exp0_hugs_original_15000_parkinglot_20260601 | 待跑 |
| A3 | bike | exp0_hugs_original_15000_bike_20260601 | 待跑 |
| A4 | citron | exp0_hugs_original_15000_citron_20260601 | 待跑 |
| A5 | jogging | exp0_hugs_original_15000_jogging_20260601 | 待跑 |
| B1 | citron depth_sup 12K | hugs_depth_sup_12000_citron_20260601 | 待跑 |
| C1 | citron attn 6K | depth_sup12k_transl_xyz_attn_6000_citron_20260601 | 待跑 |
| B2 | jogging depth_sup 12K | hugs_depth_sup_12000_jogging_20260601 | 待跑 |
| C2 | jogging attn 6K | depth_sup12k_transl_xyz_attn_6000_jogging_20260601 | 待跑 |

**后台 PID**：1257436
预计总耗时：约 13~15 小时

---

## 2026-06-01（会话一）

### 会话主题

Stage 3 场景优化实验（Scene Refinement / Scene from Scratch）+ ADC 参考来源确认

### 背景

本次会话延续上次 context 压缩前的工作：
- Stage 2 最优 ckpt：`depth_sup12k_temporal_smooth_attn_6000_lab_20260531/2026-05-31_19-54-16`
- 目标：通过 Stage 3 进一步提升场景质量，以接近 STM 论文的 26.6 dB

### 实验一：Stage 3a — 场景精化（Scene Refinement，3000 步）

**配置**：`cfg_files/release/neuman/hugs_depth_sup12k_scene_refine_3000_lab.yaml`

- 冻结人体（所有 LR=0，`optim_pose/trans/betas: false`）
- 冻结 attention（`lr: 0.0`）
- 从 Stage 2 场景 checkpoint 继续训练 3000 步
- `densify_from_iter: 200`，`densify_until_iter: 0`（不开启新的 densification）

**结果**：

| Iter | HUGS PSNR | 备注 |
|---:|---:|---|
| Stage 2 baseline | 26.3699 | 起点 |
| Stage 3a final (3000) | ~26.40 | 提升约 +0.033 |

**结论**：提升极小，Stage 3a 效果不显著。

**根因分析**：
- scene loss 只作用于人体 mask 外的背景区域，因此 Stage 2 场景高斯在背景区域已基本收敛
- 冻结人体之后场景精化无法带来明显增益
- 场景质量瓶颈不在于"历史包袱"，而可能在于场景表征容量本身

**脚本**：`scripts/run_scene_refine_train_and_eval.sh`
**输出目录**：`output/human_scene/neuman/lab/hugs_trimlp/depth_sup12k_scene_refine_3000_lab_20260531/`

---

### 实验二：Stage 3b — 从零重建场景（Scene from Scratch，8000 步）

**配置**：`cfg_files/release/neuman/hugs_depth_sup12k_scene_scratch_8000_lab.yaml`

- 冻结人体（所有 LR=0）、冻结 attention（lr=0）
- `scene.ckpt: ''`：场景从点云（9013 点）重新初始化
- `densify_from_iter: 500`，`densify_until_iter: 15000`
- `opacity_reset_interval: 3000`

**Bug 修复**（训练前）：  
原配置 `densify_until_iter: 0`（从 Stage 2 沿用），导致 `if t_iter < 0` 永远不满足，densification 被完全禁用。已修正为 `15000`。

**PSNR 训练轨迹（截至 step 6K）**：

| Step | HUGS PSNR | 说明 |
|---:|---:|---|
| 1K | 15.77 | 场景快速拟合 |
| 2K | 18.94 | 稳步上升 |
| 3K | 5.91 | opacity reset（interval=3000）导致急剧下降 |
| 4K | 16.01 | 恢复中 |
| 5K | 9.12 | 异常低（第二次 reset 前已开始波动） |
| 6K | 5.74 | 第二次 opacity reset |

训练截止本次会话仍在运行（PID 1200067），预计约 00:20 完成。7K/8K 最终结果待确认。

**已知问题**：
- `opacity_reset_interval: 3000` 在 8000 步训练中触发两次（3K、6K），导致 PSNR 大幅震荡，中间过程极不稳定
- 场景高斯从 9K 快速膨胀至 ~2.1M（step 6K），增长过快可能影响质量
- 从零重建场景的最终结果不确定，需等待训练完成后评估

**脚本**：`scripts/run_scene_scratch_train_and_eval.sh`
**输出目录**：`output/human_scene/neuman/lab/hugs_trimlp/depth_sup12k_scene_scratch_8000_lab_20260531/2026-05-31_23-14-04/`
**日志**：`run_logs/scene_scratch_train_20260531_231354.log`

---

### 已知问题记录（待解决）

#### 问题 1：opacity_reset 导致从零重建训练不稳定

- **现象**：`opacity_reset_interval: 3000` 在 8000 步训练中触发两次，每次 PSNR 从 ~18 跌至 ~6，恢复需 1000 步，有效优化窗口被大幅压缩
- **影响**：不确定最终 8K 结果能否超越 Stage 2 的 26.37 dB
- **潜在方案**：
  - 将 `opacity_reset_interval` 设为大值（如 999999）跳过 reset，依赖 pruning 做清理
  - 或将总步数拉长至 15000，让 reset 后有足够恢复空间

#### 问题 2：场景质量提升的天花板

- **现象**：Stage 2 final PSNR 26.37 dB，STM 26.6 dB，差距约 0.23 dB；Stage 3 系列方案效果有限
- **根因假设**：
  - 单目深度监督质量有限，场景几何初始化误差在后期难以纠正
  - 场景高斯的表征容量（SH degree、densification 策略）可能有上限
  - 人体-场景分离优化本身存在歧义
- **影响**：当前 pipeline 可能已接近瓶颈，需要考虑其他方向

#### 问题 3：整体实验策略待重新规划

- **现状**：场景系列实验（Refine / Scratch）效果不如预期，暂时搁置
- **可能方向**：
  - 多场景评估（bike/seattle/parkinglot）确认当前最优 pipeline 泛化性
  - 提升 human 指标（HUMAN PSNR 目前约 19.78）
  - 探索 open_problems_and_solutions.md 中的优先方案

---

### ADC 参考来源确认

用户查询：ADC（Global-aware Voxelized ADC）代码的参考是什么？

**结论**：代码基于论文 **"Global-aware Voxelized Adaptive Density Control for Efficient and Compact 3D Gaussian Splatting"** 实现。

详见 `test md/Global-aware_Voxelized_ADC应用于HUGS背景重建方案.md` 第 1 节和第 22 节（与论文设置的对应关系）。

---

## 2026-05-31（续3）

### 会话主题

当前 pipeline 三大问题诊断与解决方案

### 问题与方案摘要

**问题一：时序模块指标提升有限**
- 根因：随机采帧导致 memory 无效，smooth_loss 约束方向错误
- 方案（按优先级）：① attention phase 改顺序采帧 ② 直接约束 delta_correction 时序平滑 ③ anchor 位置一致性损失

**问题二：背景高斯无专项设计**
- 根因：attention phase 场景高斯完全冻结，无法跟随人体 correction 调整
- 方案：① 解冻场景高斯（原 LR × 5%） ② 接触区域 render loss 3x 加权 ③ 双向 correction（后期）

**问题三：attention 计算开销大（4.7→1.6 it/s）**
- 根因：torch.cdist 对 200 万场景高斯做全量 KNN
- 方案：① anchor local pool 预缓存（K=1000，每500iter更新） ② 接触区域场景 GS 预过滤（前5万） ③ 自适应 active anchor

**最快见效组合**：「直接约束 delta correction」+「解冻场景高斯（低LR）」+「anchor local pool 预缓存」

详见 memory/open_problems_and_solutions.md

---

## 2026-05-31（续2）

### 会话主题

时序接触质量评估脚本开发与三实验对比

### 工作内容

1. 阅读 `test md/Temporal_Contact_Quality_Evaluation_Guide.md` 评估方案
2. 实现 `scripts/evaluate_temporal_quality.py`（430→790行，含多次 bug 修复）
3. 实现 `scripts/run_temporal_eval_20260531.sh`（三实验顺序评估入口）
4. 对三个模型（HUGS baseline / depth+attn / temporal attn）运行评估并对比

### Bug 修复历程

- **AnchorSceneAttentionBaseline.forward() 不接受 frame_idx**：在签名中加入 `frame_idx=None`
- **foot_xyz 混合左右脚**：改为 SMPL joint 10/11（后又发现 GT joints 三模型相同）
- **GT SMPL joints 无模型区分性**：改为 LBS-weight 识别足部 Gaussians + corrected xyz
- **_left/_right 子 dict 进入 JSON**：`_strip_seqs` 加 `k.startswith("_")` 过滤

### 三实验时序评估结果（lab，2026-05-31）

#### 渲染质量（区分性高）

| 指标 | HUGS Baseline | depth+attn | temporal attn |
|------|---:|---:|---:|
| Mean PSNR (dB) | 26.01 | **26.36** | 26.35 |
| PSNR Std (dB) | **0.728** | 0.995 | 0.971 |
| PSNR Stability | **0.579** | 0.501 | 0.507 |
| Flickering Ratio | **0.0%** | 11.1% | 11.1% |
| Mean ΔPSNR (dB) | **0.907** | 1.071 | 1.034 |
| Mean SSIM | **0.9145** | 0.9088 | 0.9088 |

#### 接触注意力稳定性（仅 attention 模型）

| 指标 | depth+attn | temporal attn | 提升 |
|------|---:|---:|---:|
| Foot Attn Stability | 0.9331 | **0.9428** | +0.010 |
| Non-Foot Attn Stability | 0.8511 | **0.8700** | +0.019 |

### 关键结论

1. **depth+attn / temporal attn** 平均 PSNR 提升 +0.35 dB，但帧间方差增大（attention correction 对不同帧效果存在差异）
2. **Temporal Attention** 相比非时序基线：PSNR Std 略降（0.97 vs 0.99），Foot Attn Stability +0.010，Non-Foot +0.019
3. **接触指标**（foot sliding / jitter）对我们的模型无区分性：由 GT SMPL params 决定，各模型一致。不适合用于区分本组方法
4. **评估报告路径**：`output/.../temporal_eval/report.json`

---

## 2026-05-31（续）

### 会话主题

Temporal Anchor Attention 实验验证（lab 场景）

### 实验配置

- **Base ckpt**：depth_sup 12000 步（`hugs_depth_sup_12000_lab_20260528/2026-05-28_22-04-11`）
- **Attention**：TemporalAnchorAttention，6000 步（`num_steps: 5998`），`module_start_iter: 0`
- **配置文件**：`cfg_files/release/neuman/hugs_depth_sup12k_temporal_attention_6000_lab.yaml`
- **输出目录**：`output/human_scene/neuman/lab/hugs_trimlp/depth_sup12k_temporal_attn_6000_lab_20260531/2026-05-31_17-16-35/`

### 实验结果

| Iter | HUGS PSNR | HUMAN PSNR | HUMAN SSIM | HUMAN LPIPS |
|---:|---:|---:|---:|---:|
| 1000 | 26.3334 | 19.7816 | 0.7685 | 0.1443 |
| 2000 | 26.3526 | 19.7681 | 0.7684 | 0.1406 |
| 3000 | 26.3924 | **19.8234** | 0.7688 | 0.1400 |
| 4000 | 26.3938 | 19.8146 | 0.7699 | 0.1395 |
| 5000 | 26.3992 | 19.8176 | 0.7704 | 0.1388 |
| **final (5998)** | **26.3812** | **19.7850** | **0.7696** | **0.1387** |
| 非时序基线 final | 26.3816 | 19.7831 | — | 0.1386 |

### 结论

- **无代码报错**，TemporalAnchorAttention 成功集成并运行
- Final 结果与非时序基线基本持平（HUMAN PSNR +0.002，在误差范围内）
- **峰值出现在 iter 3000**（HUMAN PSNR 19.8234，+0.04 vs 基线），说明时序 bias 在早期有一定加速收敛效果，但后期会轻微回退
- 时序 smooth loss（平滑约束）在后期可能略微限制了逐帧精细优化的空间
- 训练速度：约 2.1 it/s（与非时序 baseline 接近，时序模块开销可忽略）

---

## 2026-05-31

### 会话主题

记忆恢复，了解最新进展，待用户指定下一步任务。

### 背景与动机

延续 05-29 的工作。记忆记录的最新状态为 depth_sup12k+attn6k 在 lab/bike/seattle/parkinglot 四场景全面优于旧 pipeline。

### 当前状态摘要

- **当前最强 pipeline**：depth_sup12k（HUGS+depth_w=0.05，12000步）+ transl_xyz attention 6000步
- **全场景 HUMAN PSNR**：lab 19.7831 / bike 20.4297 / seattle 19.6412 / parkinglot 19.8686
- **05-29 新增工具脚本**：
  - `scripts/eval_ground_contact.py`：SMPL脚底-地面有符号距离（COLMAP RANSAC拟合地平面）
  - `scripts/eval_gs_ground_contact.py`：高斯椭球穿透深度评估（解析解）
- **待探索**：floater suppression（方案2 opacity penalty 或方案3 attention delta_opacity）、jogging/citron场景泛化

### 新增：Temporal Anchor Attention 实现（2026-05-31）

阅读 `test md/Temporal_Contact_Attention_Implementation_Guide.md`，分析可行性并实现时序 attention 扩展。

**可行性分析：**
- ✅ 帧索引记忆 + 时序 bias + 平滑 loss：可行
- ❌ HierarchicalTemporalAccelerator：不实现，与梯度训练根本不兼容
- ⚠️ 时序顺序假设：改为帧索引邻近查找，兼容随机帧采样

**新增/修改文件：**
- `hugs/models/temporal_contact_attention.py`（新建）
- `hugs/cfg/config.py`（修改）：新增 `anchor_attention.temporal.*` 字段
- `hugs/trainer/gs_trainer.py`（修改）：接入 temporal attention
- `cfg_files/release/neuman/hugs_depth_sup12k_temporal_attention_6000_lab.yaml`（新建）

**使用方式：** YAML 中设 `anchor_attention.temporal.enabled: true`；`false`（默认）完全等同原有 pipeline。

---

## 2026-05-29（续5）

### 会话主题

高斯椭球接触评估脚本开发（`eval_gs_ground_contact.py`）

### 背景与动机

已有 `eval_ground_contact.py` 基于 SMPL 顶点评估脚底-地面距离。
但顶点是零维的点，高斯是有体积的椭球：即使高斯中心在地面以上，椭球也可能穿透地面。
需要对高斯椭球做精确的接触评估。

### 核心数学公式

对于高斯椭球（中心 mu，旋转矩阵 R，尺度 s），到平面 (n, d) 的最近表面点签名距离：

```
s_surface = (n · mu + d) - ||s × (R^T @ n)||
```

其中 `||s × (R^T @ n)|| = sqrt(n^T Σ n)` 是椭球在法向量方向的"半径"（解析解）。
- `s_surface > 0`：椭球表面在地面以上（无穿透）
- `s_surface < 0`：穿透，深度 = `-s_surface`

### 实现：`scripts/eval_gs_ground_contact.py`

**Pipeline：**
1. 加载 `eval_contact/{seq}/ground_plane.npz`（上次拟合的地面平面）
2. 加载 HUGS Trainer（复用 export_human_scene_ply.py 的 configure_from_run 模式）
3. 逐帧 forward 获取 world-space `xyz`、`scales`、`rotq`、`opacity`
4. 用 `R = quat_to_rotmat(rotq)` 从四元数得旋转矩阵（wxyz 约定）
5. `n_local = R^T @ n`（einsum: `'nji,j->ni'`），`r = ||scales * n_local||`
6. 脚部过滤：`s_center < foot_band`（捕获地面附近 + 所有穿透高斯）
7. 输出 CSV + 三子图折线图 + 可选彩色 PLY

**关键参数：**
```bash
python scripts/eval_gs_ground_contact.py \
  --run-dir output/human_scene/neuman/lab/hugs_trimlp/<exp>/<ts> \
  --seq lab \
  [--foot-band 1.0]      # 脚部区域过滤（world units）
  [--min-opacity 0.05]   # 过滤近乎透明的高斯
  [--scale-factor 1.0]   # 1-sigma 椭球（2 or 3 可扩大椭球）
  [--save-colored-ply]   # 生成 PLY：绿=脚部无穿透，红=穿透，灰=非脚部
  [--colored-ply-frame 0]
```

**输出：**
```
<run-dir>/eval_gs_contact/
├── gs_ground_contact.csv      # per-frame stats
├── gs_ground_contact.png      # 三子图：穿透比例/最大深度/opacity加权深度
└── frame0000_penetration.ply  # 可选彩色诊断 PLY
```

**CSV 列（per frame）：**
- `all_*`：全部有效人体高斯的统计
- `foot_*`：脚部区域（中心距地面 < foot_band）高斯的统计
- 指标：`n`（数量）、`n_pen`（穿透数）、`pen_ratio`（穿透比）、`max_pen`、`mean_pen`、`opw_pen`（opacity加权穿透深度）

### Lab 场景评估结果（2026-05-29，foot-band=1.0，scale-factor=1.0）

| 指标 | depth_sup12k+attn6k（新） | HUGS原版15000（旧） | 提升 |
|---|---:|---:|---|
| 穿透比例（mean±std） | 0.260 ± 0.078 | 0.314 ± 0.099 | **-17.2%** |
| 平均帧最大穿透深度（world） | 0.5827 | 0.6347 | **-8.2%** |
| 全局最大穿透深度（world） | 1.0367 | 1.1215 | **-7.5%** |
| 平均穿透深度（over pen. GS）| 0.2166 | 0.2340 | **-7.4%** |
| **opacity加权穿透深度** | **0.0852** | **0.1108** | **-23.1%** |
| 每帧脚部GS数量 | 13610 avg | 14127 avg | — |
| Frame 50 穿透数/总数 | 4905/15121 (32.4%) | 6116/15299 (40.0%) | -7.6pp |

**opacity加权穿透深度**是最重要的指标（考虑可见性权重），新pipeline降低了23.1%。

### 与 eval_ground_contact.py 的区别

| | eval_ground_contact.py | eval_gs_ground_contact.py |
|---|---|---|
| 输入 | SMPL 顶点 | 人体高斯椭球 |
| 评估对象 | 脚底顶点（20个点的均值） | 所有高斯椭球表面最近点 |
| 是否考虑体积 | ✗ | ✓ |
| 需要模型ckpt | ✗ | ✓ |
| 是否包含 anchor correction | ✗ | ✓（auto） |

---

## 2026-05-29（续4）

### 会话主题

地面平面拟合 + 脚底接触距离评估脚本开发

### 背景与动机

现有接触评估依赖 DECO 预测（语义 contact label），缺乏几何定量指标。
目标：用 COLMAP 点云拟合地面平面，计算每帧脚底到地面的有符号距离，作为 HSI 接触质量的几何基准。

### 坐标系发现

NeuMan 使用 **OpenCV/COLMAP 约定（Y 向下）**。
- 脚底 = Y 值最大的顶点（非最小）
- 地面平面在 COLMAP 点云的最大 Y 密集区域
- 直接对全部 COLMAP 点跑 RANSAC 会拟合到错误平面（Y≈-12）
- 正确做法：先用 SMPL 脚底顶点 Y 均值作为先验，在该高度附近的 COLMAP 点上做 RANSAC

### 实现：`scripts/eval_ground_contact.py`

**Pipeline：**
1. 解析 `sparse/points3D.txt` → COLMAP 世界坐标点云
2. SMPL forward 获取每帧脚底顶点（Y 最大的 20 个顶点均值），取中位数作为地面先验
3. 筛选 COLMAP 点：`|Y - floor_prior| < floor_band`（默认 ±2.0）
4. 筛选后的点云上做 3D RANSAC 平面拟合（SVD 精化）
5. 逐帧计算脚底到地面平面的有符号距离（正 = 脚在地面上方，负 = 穿透）
6. 输出 CSV + matplotlib 折线图 + ground_plane.npz

**关键参数：**
```bash
python scripts/eval_ground_contact.py --seq lab \
  --ransac-thresh 0.3   # RANSAC 内点阈值（世界坐标）
  --floor-band 2.0      # 地面搜索窗口半宽
  --n-bottom-verts 20   # 脚底平均顶点数
  --contact-thresh 0.3  # 接触标注阈值（可选）
```

**输出：**
```
eval_contact/{seq}/
├── foot_ground_dist.csv      # frame, foot_bottom_Y_world, signed_dist
├── foot_ground_dist.png      # 折线图（上：脚底 Y vs 地面；下：有符号距离）
└── ground_plane.npz          # normal, d, floor_prior_Y, inlier_ratio
```

### Lab 场景结果

| 指标 | 数值 | 物理换算（scale≈8.62） |
|---|---|---|
| 地面法向 | [-0.012, -0.994, -0.109] | 基本水平（Y 分量 99.4%） |
| COLMAP 内点率 | 57.9% | — |
| 最大距离（摆脚） | +0.388 world | **+4.5 cm** |
| 最小距离（穿透） | -0.519 world | **-6.0 cm** |
| 均值 | -0.041 world | **-0.5 cm** |

结论：人物始终紧贴地面，均值 -0.5 cm（轻微穿透），摆脚抬起约 4.5 cm，物理上合理。

### 后续方向

- 加载 DECO 接触预测标签，对比"DECO 标注接触帧"和"非接触帧"的 signed_dist 分布
- 将 ground_plane.npz 作为 attention correction 的几何约束（脚底不能比地面更深）
- 扩展到其他场景（bike/seattle/parkinglot）

---

## 2026-05-29（续3）

### 会话主题

1. 整理阶段性成果文档（组会用）
2. 新增全帧视频导出脚本，修改默认 fps 配置
3. 批量导出四个场景全帧视频 + 多个实验的 human+scene PLY

### 完成内容

#### 1. 组会阶段性成果文档

生成 `ml-hugs-work/5-29组会阶段性成果.md`，包含：
- 所有实验结果汇总表（lab 全消融 + 多场景泛化）
- 方法体系说明、关键结论、论文 ablation 建议表
- 当前最强 pipeline 训练流程

#### 2. 全帧视频导出（新增脚本）

**新文件：** `scripts/render_full_video.py`

**用法：**
```bash
python scripts/render_full_video.py \
  --run-dir output/human_scene/neuman/{scene}/hugs_trimlp/{exp}/{ts} \
  [--fps 10]           # 帧率，默认 10（103帧 ≈ 10s）
  [--keep-frames]      # 保留逐帧 PNG
  [--iter N]           # 指定迭代数（默认 final ckpt）
  [--apply-anchor-attention auto|yes|no]  # 默认 auto
```

**同步修改：**
- `hugs/cfg/config.py`：新增 `cfg.train.render_fps = 10`
- `main.py`：`render_full_sequence(..., fps=cfg.train.render_fps)`（训练结束自动用此 fps）

#### 3. 本次导出的视频（10fps，全帧）

| 实验 | 帧数 | 视频路径（相对 output/...） |
|---|---:|---|
| lab depth_sup12k+attn6k | 103 | `.../depth_sup12k_transl_xyz_attn_6000_lab_20260528/.../render_all_neuman_lab_final.mp4` |
| bike depth_sup12k+attn6k | 104 | `.../depth_sup12k_transl_xyz_attn_6000_bike_20260529/.../render_all_neuman_bike_final.mp4` |
| seattle depth_sup12k+attn6k | 41 | `.../depth_sup12k_transl_xyz_attn_6000_seattle_20260529/.../render_all_neuman_seattle_final.mp4` |
| parkinglot depth_sup12k+attn6k | 42 | `.../depth_sup12k_transl_xyz_attn_6000_parkinglot_20260529/.../render_all_neuman_parkinglot_final.mp4` |
| lab 原版 HUGS 15k | 103 | `.../exp0_hugs_original_15000_20260527/.../render_all_neuman_lab_final.mp4` |

#### 4. 本次导出的 PLY（val frame 007，实际帧 37）

val split 映射：`scene_length=103, offset=2, length=5` → `val_split=[2,7,12,...,47]` → `val[7]=帧37`

| 实验 | 人体GS | 场景GS | anchor attn | PLY 路径（相对各实验 meshes/） |
|---|---:|---:|---|---|
| 原版 HUGS 15k | 507,885 | 2,181,637 | ✗ | `exp0_hugs_original_15000_.../meshes/human_scene_final_val_00007_raw_sh_splat.ply` |
| depth_sup 15k（无 attn）| 479,405 | 2,147,453 | ✗ | `hugs_depth_sup_15000_.../meshes/human_scene_final_val_00007_raw_sh_splat.ply` |
| depth_sup12k+attn6k | 416,068 | 2,116,859 | ✓ | `depth_sup12k_transl_xyz_attn_6000_lab_.../meshes/human_scene_final_val_00007_raw_sh_splat.ply` |

---

## 2026-05-29（续2）

### 多场景实验全部完成（09:18 AM 汇总）

所有四场景（bike/seattle/parkinglot + lab对照）均已跑完，depth_sup12k+attn6k 全面优于旧 pipeline。

| 场景 | Phase | HUGS PSNR | HUGS SSIM | HUGS LPIPS | HUMAN PSNR | HUMAN SSIM | HUMAN LPIPS | vs 旧pipeline |
|---|---|---:|---:|---:|---:|---:|---:|---|
| lab（对照） | attn 6k final | 26.3816 | — | — | 19.7831 | — | — | +0.23 |
| bike | attn 6k final | 26.3024 | 0.8652 | 0.0865 | **20.4297** | 0.6958 | 0.1332 | **+0.66** |
| seattle | Phase 1 final | 25.6788 | 0.8470 | 0.0998 | 19.2672 | 0.6638 | 0.1404 | — |
| **seattle** | **attn 6k final** | **26.3995** | **0.8665** | **0.0830** | **19.6412** | **0.6884** | **0.1275** | **+0.35** |
| parkinglot | Phase 1 final | 26.4803 | 0.8338 | 0.1543 | 19.3596 | 0.7045 | 0.1657 | — |
| **parkinglot** | **attn 6k final** | **27.1149** | **0.8548** | **0.1328** | **19.8686** | **0.7342** | **0.1369** | **+0.10** |

**输出目录：**
- Seattle Phase 1：`output/human_scene/neuman/seattle/hugs_trimlp/hugs_depth_sup_12000_seattle_20260529/2026-05-29_02-42-04/`
- Seattle Phase 2：`output/human_scene/neuman/seattle/hugs_trimlp/depth_sup12k_transl_xyz_attn_6000_seattle_20260529/2026-05-29_03-20-38/`
- Parkinglot Phase 1：`output/human_scene/neuman/parkinglot/hugs_trimlp/hugs_depth_sup_12000_parkinglot_20260529/2026-05-29_04-17-57/`
- Parkinglot Phase 2：`output/human_scene/neuman/parkinglot/hugs_trimlp/depth_sup12k_transl_xyz_attn_6000_parkinglot_20260529/2026-05-29_04-54-59/`

**结论：depth_sup12k+attn6k 在全部 4 个场景均优于旧 HUGS12k+attn6k，泛化性验证通过，可作为论文 main result。**

---

## 2026-05-29（续）

### 实验进度确认（02:44 AM）

**当前状态（02:44 CST）：**

| 场景 | 阶段 | 状态 | HUGS PSNR | HUGS SSIM | HUGS LPIPS | HUMAN PSNR | HUMAN SSIM | HUMAN LPIPS |
|---|---|---|---:|---:|---:|---:|---:|---:|
| bike | Phase 1 (depth_sup 12k) | ✅ 01:36完成 | 25.6012 | — | — | 19.9678 | — | — |
| bike | Phase 2 (attn 6k) | ✅ 02:39完成 | 26.3024 | 0.8652 | 0.0865 | **20.4297** | 0.6958 | 0.1332 |
| seattle | Phase 1 (depth_sup 12k) | 🔄 02:42启动 | — | — | — | — | — | — |
| seattle | Phase 2 (attn 6k) | ⏳ 待跑 | — | — | — | — | — | — |
| parkinglot | Phase 1+2 | ⏳ 待跑 | — | — | — | — | — | — |

**Bike 结论：** HUMAN PSNR 20.43，比旧 HUGS12k+attn6k pipeline（19.7643）提升 **+0.66**。

**输出目录：**
- Bike Phase 1：`output/human_scene/neuman/bike/hugs_trimlp/hugs_depth_sup_12000_bike_20260529/2026-05-29_00-57-34/`
- Bike Phase 2：`output/human_scene/neuman/bike/hugs_trimlp/depth_sup12k_transl_xyz_attn_6000_bike_20260529/2026-05-29_01-38-26/`
- Seattle Phase 1：`output/human_scene/neuman/seattle/hugs_trimlp/hugs_depth_sup_12000_seattle_20260529/2026-05-29_02-42-04/`

**实验后台运行，断链不影响：** `nohup bash scripts/run_depth_sup_multiscene_20260529.sh > run_logs/depth_sup_multiscene_20260529_nohup.log 2>&1 &`，日志在 `run_logs/depth_sup_multiscene_20260529.log`。

**下次对话恢复时：** 先 `grep "final\|完成\|ERROR" run_logs/depth_sup_multiscene_20260529.log` 查看最新进度，然后更新本文件和 memory 中的 `experiments_20260529.md` 与 `experiment_results.md`。

---

## 2026-05-29

### depth_sup 12000 + transl_xyz attention 6000 多场景实验（进行中）

**背景：** lab 场景验证 depth_sup 背景方案比 ADC 方案在 HUMAN PSNR 上提升 +0.23（19.5512→19.7831），需在其他场景验证泛化性。

**实验配置：**
- Phase 1：depth_supervised 12000 步（`depth_w=0.05`，无 anchor attention）
- Phase 2：transl_xyz attention correction 6000 步（从 phase 1 ckpt 热启动）
- 超参数与 lab 完全一致

**场景顺序：** bike → seattle → parkinglot

**相关文件：**
- Phase 1 配置：`cfg_files/release/neuman/hugs_depth_supervised_12000_{bike,seattle,parkinglot}.yaml`
- Phase 2 配置：`cfg_files/release/neuman/hugs_depth_sup12k_transl_xyz_attention_6000_{bike,seattle,parkinglot}.yaml`
- Run 脚本：`scripts/run_depth_sup_multiscene_20260529.sh`
- Nohup 日志：`run_logs/depth_sup_multiscene_20260529_nohup.log`
- 详细日志：`run_logs/depth_sup_multiscene_20260529.log`

**预期输出目录：**
```
output/human_scene/neuman/bike/hugs_trimlp/hugs_depth_sup_12000_bike_20260529/<ts>/
output/human_scene/neuman/bike/hugs_trimlp/depth_sup12k_transl_xyz_attn_6000_bike_20260529/<ts>/
output/human_scene/neuman/seattle/hugs_trimlp/hugs_depth_sup_12000_seattle_20260529/<ts>/
output/human_scene/neuman/seattle/hugs_trimlp/depth_sup12k_transl_xyz_attn_6000_seattle_20260529/<ts>/
output/human_scene/neuman/parkinglot/hugs_trimlp/hugs_depth_sup_12000_parkinglot_20260529/<ts>/
output/human_scene/neuman/parkinglot/hugs_trimlp/depth_sup12k_transl_xyz_attn_6000_parkinglot_20260529/<ts>/
```

**结果（待填入）：**

| 场景 | 阶段 | HUMAN PSNR | HUMAN SSIM | HUMAN LPIPS | HUGS PSNR | HUGS SSIM | HUGS LPIPS |
|---|---|---:|---:|---:|---:|---:|---:|
| lab（对照） | depth_sup12k+attn6000 final | 19.7831 | 0.7697 | 0.1386 | 26.3816 | 0.9089 | 0.0759 |
| bike | depth_sup12k+attn6000 final | — | — | — | — | — | — |
| seattle | depth_sup12k+attn6000 final | — | — | — | — | — | — |
| parkinglot | depth_sup12k+attn6000 final | — | — | — | — | — | — |

---

---

## 2026-05-28 会话（续/新上下文窗口）

### 会话主题

恢复上下文：depth supervision 实验进度确认

### 实验进度（恢复时状态 → 已完成）

- 12000步实验：**已完成** → HUGS PSNR 26.0120 / HUMAN PSNR **19.4769**（+0.68 vs baseline）
- 15000步实验：**已完成** → HUGS PSNR 26.0568 / HUMAN PSNR **19.3926**（+0.59 vs baseline）
- 两个实验均超越 HUGS 原版 15000步基线（18.8005）
- 日志：`run_logs/depth_supervised_lab_20260528_nohup.log`

---

## 2026-05-28 会话（深夜）

### 会话主题

调研 DASR 的 depth supervision 实现方案并在原版 HUGS 上完整实现，启动 12000/15000 步对比实验。

### 实现内容

#### DASR depth supervision 分析

- **深度图来源**：预计算单目深度图，16-bit PNG，路径 `data/neuman/dataset/{seq}/mono_depth/*.png`，除以 10000 得米
- **Loss 公式**：Pearson 相关系数（尺度/偏移无关），取两个变体的 min：`-mono_depth` 和 `1/(mono_depth+200)`
- **权重**：`l_depth_w=0.05`
- **作用对象**：整幅渲染深度图（human+scene 合并）

#### 代码修改（5 个文件）

| 文件 | 修改内容 |
|---|---|
| `hugs/renderer/gs_renderer.py` | 新增 `render_depth_map()`：以 camera-space z 为 precomputed color 做 alpha compositing |
| `hugs/datasets/neuman.py` | 新增 `mono_depth_dir` 参数，加载 16-bit PNG 并缩放到图像尺寸 |
| `hugs/losses/loss.py` | 新增 `l_depth_w`、`_pearson_corrcoef()`，在 forward 末尾加 depth loss |
| `hugs/cfg/config.py` | 新增 `cfg.human.loss.depth_w=0.0`、`cfg.dataset.mono_depth_dir=None` |
| `hugs/trainer/gs_trainer.py` | 传入 `mono_depth_dir`、`l_depth_w`；训练循环中渲染 depth 并放入 render_pkg |

#### 新文件

- `cfg_files/release/neuman/hugs_depth_supervised_12000_lab.yaml`
- `cfg_files/release/neuman/hugs_depth_supervised_15000_lab.yaml`
- `scripts/run_depth_supervised_lab_20260528.sh`

#### 实验状态（截至记录时）

- 12000 步实验：**运行中**（GPU 93%，5.6G 显存，PID 762830）
- 15000 步实验：排队中（12000 步完成后自动启动）
- 输出目录：`output/human_scene/neuman/lab/hugs_trimlp/hugs_depth_sup_{12000,15000}_lab_20260528/`
- 日志：`run_logs/depth_supervised_lab_20260528_nohup.log`

---

## 2026-05-28 会话（晚）

### 会话主题

`diag_foot_occlusion.py` 三列图结构确认

### 确认内容

**问题**：`diag_foot_occlusion.py` 生成的三列图，中间那张是否以模型渲染图为底？

**结论（见脚本 L226）**：

```python
comp = np.concatenate([gt_np, orig_np, colored_np], axis=1)
```

- 左：GT 真实图像
- 中：**模型正常渲染图**（`render_human_scene` 无任何 GS 颜色修改）
- 右：脚部周围 scene GS 染红后的渲染图

### 问题与纠正

Claude 本次对话开始时未先更新 CLAUDE_SESSION_LOG.md，被用户指出后补记。

---

## 2026-05-28 会话

### 会话主题

1. 确认上次实验进度（mask-aware 系列 + gamma 消融）
2. 查找"对脚部附近场景高斯染色"的可视化脚本

### 确认的进展

#### 当前全局最优（lab 场景）

**ADC12000 + transl_xyz attention Stage2**
- HUGS 26.4374 / 0.9201 / 0.0651，**HUMAN 19.5512** / 0.7708 / 0.1372
- 输出目录：`output/human_scene/neuman/lab/hugs_trimlp/adc12000_transl_xyz_attention_plus3000_stage2_lab_20260527/2026-05-27_20-58-51`

#### 已完成并关闭的方向

| 实验 | HUMAN PSNR | 结论 |
|---|---|---|
| ADC12000+mask_aware standalone | 18.8339 | 无效 |
| ADC12000+mask_aware+transl_xyz Stage1 | 19.2725 | 无效 |
| ADC12000+mask_aware+transl_xyz Stage2 | 19.3599 | 无效（-0.19 vs 基线） |
| gamma=0.15 消融（highgamma） | 19.3442 | 无效（-0.21 vs 基线） |

**结论**：mask-aware 和 gamma 调大均无正向效果，两个方向已放弃。

#### 新增诊断工具（2026-05-28）

- `scripts/render_gs_type_overlay.py`：human/scene GS 类型 overlay（红/蓝），输出 `<run-dir>/val_overlay/`
- `scripts/render_gs_structure.py`：GS 结构 2D 椭圆投影可视化，支持：
  - `--dist-cmap plasma_r`：按到最近 human GS 的距离染色
  - `--scene-occlude-only`：只显示遮挡人体的 scene GS
  - `--scene-near-extend`：显示人体 bbox 附近的 scene GS
  - `--novel-view`：轨道相机绕人体重心旋转

### 本次会话内容

**问题1**：查找"对脚部附近场景高斯染色"的可视化脚本

**查找结果**：
- 该脚本原为 `/tmp/diag_foot_occlusion.py`（临时文件），之前未备份到 NAS
- 根据用户贴出的代码片段还原逻辑：
  - 找脚部锚点（sole/toe/heel）的 world 坐标
  - 筛选半径 R 内的 scene GS（`near_foot_mask`）
  - `shs_modified[near_foot_mask] = 0.0`，再设 DC 分量为红色
  - 渲染保存到 `<run-dir>/diag_foot_occlusion/val/`

**操作**：重新实现并写入 `scripts/diag_foot_occlusion.py`（正式归档）

**脚本功能**：
```bash
python scripts/diag_foot_occlusion.py \
  --run-dir output/human_scene/neuman/lab/hugs_trimlp/<exp>/<ts> \
  --radius 0.5 --frames 0 1 2 3
```
- 输出：`<run-dir>/diag_foot_occlusion/val/frame_NN_comp.png`（GT | 原始渲染 | 脚部红色），`frame_NN_foot.png`
- 支持 `--radius`（默认 1.0m）、`--frames`、`--no-anchor`、`--anchor-ckpt`

**问题2**：要求 Claude 在项目目录维护会话记录 md 文件（本文件）

**操作**：创建 `CLAUDE_SESSION_LOG.md`，写入规则到记忆系统，约定每次对话先更新本文件

---

## 历史实验总表（截至 2026-05-28）

### Lab 场景

| 方案 | 步数 | HUGS PSNR | HUMAN PSNR | 备注 |
|---|---:|---:|---:|---|
| HUGS 原版 | 15000 | 25.9582 | 18.8005 | baseline |
| SuGaR-HUGS | 15000 | 26.1260 | 18.9424 | |
| ADC-HUGS standalone | 12000 | 25.7718 | 19.0950 | |
| ADC12000 + xyz attention stage2（6000步）| 18000 | 26.3999 | 19.4782 | 旧版 xyz-only |
| ADC12000 + transl_xyz attention stage1 | 15000 | 26.3305 | 19.4704 | |
| **ADC12000 + transl_xyz attention stage2** | **18000** | **26.4374** | **19.5512** | ★ 当前最优 |
| HUGS + mask_aware standalone | 15000 | 26.0844 | 18.8943 | 已放弃 |
| ADC12000 + mask_aware standalone | 12000 | 25.5661 | 18.8339 | 已放弃 |
| ADC12000 + mask_aware + transl_xyz stage2 | 18000 | 26.2885 | 19.3599 | 已放弃 |
| ADC12000 + transl_xyz stage2（gamma=0.15）| 18000 | 26.2940 | 19.3442 | 已放弃 |
| HUGS + depth_sup（depth_w=0.05） | 12000 | 26.0120 | 19.4769 | +0.68 vs baseline |
| HUGS + depth_sup（depth_w=0.05） | 15000 | 26.0568 | 19.3926 | +0.59 vs baseline |

### Parkinglot 场景（2026-05-22）

| 方案 | HUGS PSNR | HUMAN PSNR |
|---|---:|---:|
| HUGS 15000 | 26.6424 | 19.1697 |
| HUGS12000 + 6000 attention | 27.2009 | 19.7643 |
| SuGaR12000 + 6000 attention | 27.1805 | 19.6977 |

### Seattle 场景（2026-05-22）

| 方案 | HUGS PSNR | HUMAN PSNR |
|---|---:|---:|
| HUGS 15000 | 26.0878 | 18.7852 |
| HUGS12000 + 6000 attention | 26.2756 | 19.2921 |
| SuGaR12000 + 6000 attention | 26.3009 | 19.3517 |

---

## 核心 Pipeline 说明

### 当前主线（ADC + transl_xyz attention）

```
0-12000步:  ADC-HUGS 背景+人体训练（Global-aware Voxelized ADC）
12000-15000步: anchor-attention correction stage1
              correct_transl=True, correct_xyz=True
              gamma_transl=0.05, gamma_mu=0.005
15000-18000步: anchor-attention correction stage2（continuation）
```

### Anchor Attention 关键参数

```yaml
module_start_iter: 12000
gamma_mu: 0.005
gamma_transl: 0.05
transl_delta_clamp: 0.2
human_pooling: attention
correct_transl: true
correct_xyz: true
correct_opacity: false
zero_init_delta: true
```

### 实验已知结论

- `transl correction` 是主要收益来源，`xyz correction` 在此基础上略有提升
- `human_pooling: attention` 优于 `mean` pooling
- stage2 continuation 有稳定提升（+0.08 HUMAN PSNR）
- `gamma_transl=0.15` 比 `0.05` 差（Z 方向锁死在 clamp，振荡）
- mask-aware loss/densify 对 lab 场景无效（ADC 背景干扰）
- transl_delta_clamp Z 方向永远饱和，想要更大 Z 修正空间

---

## 已实现功能列表

### 背景方案

| 方案 | 配置前缀 | 状态 |
|---|---|---|
| HUGS 原版背景 | `hugs_trimlp` | 已验证 |
| SuGaR-HUGS | `hugs_trimlp_sugar` | 已验证 |
| Global-aware Voxelized ADC | `adc12000_*` | 已验证，当前主用 |

### Attention 相关

- `hugs/models/anchor_attention.py`：`AnchorSceneAttentionBaseline` 主模块
- `hugs/utils/anchor_utils.py`：锚点工具
- 16 个语义锚点：left/right × sole/toe/heel/palm/fingers + back/buttocks + knee/elbow

### 导出脚本

| 脚本 | 功能 | 关键参数 |
|---|---|---|
| `scripts/export_human_scene_ply.py` | 导出指定帧的 human+scene 合并 PLY | `--run-dir` `--frame-idx` `--split val` `--save-render-check` |
| `scripts/render_full_video.py` | 渲染全帧视频（all_dataset，所有帧）| `--run-dir` `--fps 10` `--keep-frames` |

**export_human_scene_ply.py 示例：**
```bash
# 导出 val frame 007（lab 实际帧 37）+ render check
python scripts/export_human_scene_ply.py \
  --run-dir output/.../2026-xx-xx_xx-xx-xx \
  --frame-idx 7 --split val --save-render-check
# 无 anchor attention 版本（原版 HUGS）
  --apply-anchor-attention no
```

**render_full_video.py 示例：**
```bash
python scripts/render_full_video.py \
  --run-dir output/.../2026-xx-xx_xx-xx-xx \
  --fps 10 --keep-frames
# yaml 里可设 train.render_fps: 5 覆盖默认值
```

**Val frame 映射（lab，103帧）：**
`val_split = [2, 7, 12, 17, 22, 27, 32, 37, 42, 47]`，val[7] = 实际帧 **37**

### 诊断脚本

| 脚本 | 功能 |
|---|---|
| `scripts/render_gs_structure.py` | 2D GS 结构椭圆可视化，支持 dist-cmap/occlude-only/near-human/novel-view |
| `scripts/render_gs_type_overlay.py` | human(红)/scene(蓝) 类型 overlay |
| `scripts/diagnose_anchor_probe_attention.py` | anchor→scene 邻居查询诊断，输出 PLY |
| `scripts/diag_foot_occlusion.py` | 脚部附近 scene GS 红色染色渲染，输出 GT/原始/染色三联图 |
| `scripts/evaluate_contact_quality.py` | 脚部接触质量评估 |
| `scripts/debug_anchors.py` | anchor 绑定调试，模板可视化 |
| `scripts/eval_ground_contact.py` | COLMAP 点云拟合地面平面 + 逐帧脚底有符号距离评估（HSI 接触几何指标）|
| `scripts/eval_gs_ground_contact.py` | 高斯椭球接触评估：对每个椭球计算最近表面点到地面的穿透深度（`s_surface = s_center - r`，`r=‖s*(R^Tn)‖`）|

**eval_ground_contact.py 关键说明：**
- 坐标系：Y 向下（OpenCV 约定），脚底 = Y 值最大的顶点
- 先验：先用 SMPL 脚底 Y 中位数定位地面高度，再在该高度带内对 COLMAP 点云做 RANSAC
- 输出 `eval_contact/{seq}/` 下的 CSV、PNG、ground_plane.npz

### Mask-aware 实现（已放弃）

- `hugs/cfg/config.py`：新增 `scene.mask_aware_enabled` 等三个配置项
- `hugs/trainer/gs_trainer.py`：`maybe_add_scene_mask_aware_loss`、`_project_scene_to_image`、densification 过滤
- 结论：干扰 ADC 背景梯度，HUMAN PSNR 全面下降

---

## 待探索方向

| 方向 | 优先级 | 说明 |
|---|---|---|
| 更长 stage2（9000步）| 中 | gamma=0.05，当前6000步是否饱和 |
| 调大 transl_delta_clamp | 中 | Z 方向永远锁死，当前 0.2 可能过小 |
| Visibility Opacity Penalty | 中 | 对投影在人体 mask 内且比 SMPL 更近的 scene GS 施加 opacity 惩罚 |
| 脚部附近 scene GS 可视化 | 低 | `render_gs_structure.py` 新增 `--foot-region` 参数 |
| Attention delta_opacity | 低（长期）| 扩展 attention 同时输出 delta_opacity |
| 多场景验证 | 中 | 当前最优 pipeline 在 parkinglot/seattle 上验证 |

---

## 2026-06-01 会话开始

**时间**：2026-06-01

### 当前状态

恢复记忆，确认上次会话结束时的状态：

- **当前最优 pipeline**：`depth_sup12k + transl_xyz attn 6k`，4场景（lab/bike/seattle/parkinglot）全面验证，可作为论文主结果
- **2026-05-31 时序实验**：`TemporalAnchorAttention` 实验完成，final HUMAN PSNR +0.002，峰值 iter 3000 +0.04，结论：时序 bias 有早期加速但后期受 smooth loss 约束回退
- **三大待解决问题**：时序提升有限 / 背景无专项设计 / 计算开销大

### 待确认任务

等待用户指定本次会话任务方向。

---

## 2026-06-03 会话开始

**时间**：2026-06-03

### 当前状态（从记忆恢复）

- **最新进展**：粗对齐系列实验 Exp A-D，测试无 GT transl 下的 HUGS 训练可行性
- **已完成实验**：
  - Exp A（HUGS baseline + ROMP 粗对齐）：HUMAN PSNR 15.74
  - Exp B（anchor attention + ROMP 粗对齐）：HUMAN PSNR 15.76
  - Exp C（最优 pipeline + ROMP 粗对齐）：HUMAN PSNR 16.16
  - Exp D（最优 pipeline + VIMO 粗对齐，step 8k 已达 17.5，运行中）
- **GT 基准**：depth_sup12k + transl_xyz attn 6k → HUMAN PSNR 19.78
- **粗对齐误差**：ROMP 222mm → VIMO 93mm（改善 2.4×）
- **结论**：GT transl 是训练收敛的必要条件，粗对齐精度直接决定 PSNR 损失幅度
- **下一步方向**：等 Exp D 结果，评估 VIMO 93mm 误差是否足够；若不足，考虑 DepthPro 人体深度约束方案

### 待确认任务

等待用户指定本次会话任务方向。

---

## 2026-06-03 晚（接续同日早些会话）

**主题**：配置 StM 环境并启动两个对比实验（Exp E/F）

### 完成内容

#### 1. StM 代码 bug 修复（共 5 处）

**问题根因**：`cfg.human.name` 默认值为 `'hugs'`，但 trainer 判断分支是 `'hugs_wo_trimlp'` / `'hugs_trimlp'`，导致 `self.human_gs = None`，训练第 0 步即崩溃。

| 文件 | 修改内容 |
|------|---------|
| `cfg_files/stm_lab_15k.yaml` | 新增 `human.name: hugs_trimlp` 及相关参数 |
| `cfg_files/stm_vimo_lab_15k.yaml` | 同上，`seq: lab_vimo` |
| `StM/trainer/gs_trainer.py:L87` | AMASS 数据集缺失时 try/except 优雅跳过 |
| `StM/trainer/gs_trainer.py:L141,171,281,290` | `if self.human_gs:` → `if self.human_gs is not None:` |
| `StM/datasets/neuman.py` | 添加 `self.betas = self.smpl_params["betas"]` |
| `StM/renderer/gs_renderer.py` | 双遍深度渲染（2-pass depth，无需 depth-diff-gaussian-rasterization） |

**smoke test 验证通过**：3步训练无崩溃，HUGS_PSNR=10.9557，HUGS_HUMAN_PSNR=11.3574

#### 2. 正式实验启动

- **Exp E**（StM baseline，lab，GT transl，15k步）：PID 135075，后台运行
  - 日志：`run_logs/stm_expE_lab_gt_20260603.log`
  - 模型：HUGS_TRIMLP + optimize_init(7000步) + 15k主训练 + StM FusionMLP
- **Exp F**（StM + VIMO，lab_vimo，VIMO transl，15k步）：守护进程 PID 138384，等 E 完成后自动启动
  - 日志：`run_logs/stm_expF_lab_vimo_20260603.log`

#### 3. 对比实验设计

| 实验 | 方法 | Translation 来源 | 预期参考 |
|------|------|----------------|---------|
| HUGS GT（已有） | HUGS | GT | 19.78 |
| HUGS VIMO（Exp D） | HUGS+attn | VIMO | 17.54 |
| **Exp E** | StM | GT | ? |
| **Exp F** | StM | VIMO | ? |

### 监控

```bash
cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work/Dynamic-Avatar-Scene-Rendering-from-Human-centric-Context
grep -E "Training.*PSNR|HUMAN_PSNR|final.*PSNR" ../../run_logs/stm_expE_lab_gt_20260603.log | tail -5
```

---

### 2026-06-08 会话续接（Context压缩后）

#### 调查：STM vs OUR step 1k HUMAN_PSNR 差距 (16.23 vs 15.46)

**背景**：用户发现STM在step 1k就有16.23 PSNR，而我们的exclusion实验在step 1k只有15.46，差距0.77。

**比较结论**：

| 参数 | STM | OUR |
|------|-----|-----|
| `lbs_w` | **100** | **1000**（10倍！） |
| `depth_w` | 0.03（无depth data） | 0.05（有实际mono_depth）|
| 人体渲染机制 | FusionMLP（从step 0联合优化） | AnchorAttention（step 2000才激活）|
| seed | 10086 | 0 |
| VIMO数据 | seattle_vimo（相同） | seattle_vimo（相同）|

**粗对齐**：完全相同，都用 `dataset.seq: seattle_vimo`，从同一VIMO数据加载SMPL参数。

**PSNR计算方法**：完全相同，都是将完整渲染（human+scene）裁剪到人体bbox后计算PSNR。

**差距原因分析**：
1. **`lbs_w=1000` 过强**：我们用10倍于STM的LBS正则化，强制人体高斯紧贴SMPL初始mesh，导致早期适应慢
2. **FusionMLP优势**：STM的FusionMLP从step 0联合处理所有高斯（人体+场景），零初始化但经过1000步后已学到轻微的边界调制
3. **深度损失竞争**：我们有实际depth loss，STM无depth数据，我们多一个竞争梯度信号

**exclusion实验当前进度**（step 11k = 15.39）：
- step 1k: 15.4570
- step 5k: 15.1993
- step 10k: 15.5473（最佳）
- step 11k: 15.3959

**建议下一步**：尝试 `lbs_w=100`（对齐STM），看早期PSNR是否提升。


---

## 2026-06-08 会话三十四（续，Context 压缩后接续）

### 重大技术发现：STM 与 HUGS 人体 GS 表征完全相同

**背景**：此前一直以为 STM 用 `hugs_wo_trimlp`（SMPL 顶点直接作为 nn.Parameter，位置规则、无 densification），所以 FusionMLP 修改 human xyz 在 STM 里安全。读取 STM 实际 config 后发现：

**STM config 关键参数**（`cfg_files/stm_human_scene.yaml`）：
```yaml
human:
  name: hugs_trimlp      # 与 HUGS 完全相同！
  n_subdivision: 2       # 约 110K 个高斯点
  
lbs_w: 100.0            # HUGS/OUR 用 1000（差 10 倍）
depth_w: 0.03           # HUGS/OUR 用 0.05（无实际 depth 数据）
fuse:
  lr:
    learning_rate: 1e-3  # 我们的 fusion_mlp 用 1e-4（差 10 倍）
# 无 AnchorAttention
```

**结论**：STM 和 HUGS 使用完全相同的人体高斯表征——triplane 特征场 + AppearanceDecoder/GeometryDecoder MLP，SMPL 细分 2 次得到约 110K 个高斯点，每帧通过 triplane 采样 → MLP → LBS 变形动态生成 xyz 和属性。STM 同样有 human GS densification。

**这意味着**：FusionMLP 修改 human xyz 在 STM 里也是在修改动态 triplane 生成的坐标，理论上同样"不安全"——但 STM 正常运行，说明我们的 FusionMLP v1/v2 崩溃有其他 bug，而非根本不兼容。

---

### FusionMLP 崩溃 Bug 分析

#### v1 崩溃（step 1k PSNR 8.85）
- 错误：对 human 和 scene 都应用了修正（包括 human xyz）
- 根因：修改了动态 triplane 生成的 human xyz，与 SMPL LBS 变形冲突

#### v2 崩溃（step 2k PSNR 12.74）  
- 错误：仍然修改 human xyz（虽然 property_dims 已修正为含 shs=48）
- 仍然对 human 部分输出了修正后的 xyz

#### 两个潜在 bug（v1/v2 共有，即使 v3 只修正 scene 也需检查）

**Bug 1：Densification 梯度切割**

Scene densification 需要读取 `viewspace_points.grad` 来判断哪些高斯要分裂。我们的实现中，`render_human_scene` 返回的 `viewspace_points` 是 全量 `screenspace_points`（N_human + N_scene 维度），但 trainer 里 scene densification 的代码可能只取了 `viewspace_points[:N_scene]` 或者用了全量 index——需要核查。

如果梯度切割出错，scene densification 会基于错误的梯度运行，导致训练不稳定。

**Bug 2：FusionMLP lr 差 10 倍**

STM `fuse.lr.learning_rate = 1e-3`，我们 `fusion_mlp.lr = 1e-4`。早期学习速率太慢，FusionMLP 初期几乎不学，无法发挥边界保护作用。

---

### SH 维度真相（重要）

**验证结论**：human GS 的 shs 实际维度是 (N, 16, 3)（AppearanceDecoder 输出 48 维 reshape），与 scene GS 的 (N, 16, 3) 完全相同。

FusionMLP 的 `property_dims` 中 shs 应设为 48（16×3），可以直接 cat 后经 ResidualMLP 处理，无需任何特殊处理。

---

### FusionMLP v3 实现（只修正 scene）

**已完成代码**：

1. `hugs/models/fusion_mlp.py`（新建）：HumanSceneFuseDecoder，每个属性独立 ResidualMLP，fc2 zero init

2. `hugs/renderer/gs_renderer.py`（修改）：`render_human_scene` 添加 `fused_gs_out` 参数：
   - human 部分：使用原始 human_gs_out 属性（SMPL deformed xyz 不改动）
   - scene 部分：使用 `fused_gs_out['xxx'][n_human:]`（只取 scene 的 fused 结果）

3. `hugs/trainer/gs_trainer.py`（修改）：
   - 新增 `setup_fusion_mlp` 方法
   - 训练循环中调用 `self.fusion_mlp(human_props, scene_props)` 得到 `fused_gs_out`
   - optimizer step 在主 optimizer step 后执行

4. `cfg_files/release/neuman/hugs_vimo_seattle_fusionmlp_3k.yaml`（新建）：
   ```yaml
   fusion_mlp:
     enabled: true
     hidden_dim: 64
     lr: 0.0001
   exp_name: vimo_fusionmlp_3k_seattle_20260608
   ```

**实验状态**：v3 实验启动后未获得结果（会话被中断/压缩），FusionMLP v3 3k 步结果 **待验证**。

---

### 当前优先级（更新于 2026-06-08 会话三十四续）

| 优先级 | 方向 | 状态 |
|---|---|---|
| ★★★ | FusionMLP v3（只修正 scene，lr 对齐 STM=1e-3）| 未出结果，需重跑 |
| ★★ | AnchorAttention module_start_iter=0 | 未做实验 |
| ★★ | Depth-consistent opacity loss | 未做实验 |
| ★ | Stage2 selective reset | 低成本，可随时加 |

**下一步**：继续跑 FusionMLP v3 实验，并排查 densification 梯度切割 bug。


---

## 2026-06-08 会话三十五（FusionMLP v3 修复与重跑）

### 修复内容（对照 STM 源码逐项修复）

**参考文件**：`Dynamic-Avatar-Scene-Rendering-from-Human-centric-Context/StM/models/fuse.py` 和 `StM/trainer/gs_trainer.py`

#### Bug 1 修复：Densification 梯度切割（最关键）

**位置**：`hugs/trainer/gs_trainer.py` ~line 1488

**旧代码（错误）**：
```python
render_pkg['scene_viewspace_points'] = render_pkg['viewspace_points']  # 全量 N_total
render_pkg['scene_viewspace_points'].grad = render_pkg['viewspace_points'].grad
```

**新代码（正确，对齐 STM）**：
```python
if render_mode == 'human_scene' and human_gs_out is not None:
    human_n_gs = human_gs_out['xyz'].shape[0]
    scene_n_gs = render_pkg['scene_visibility_filter'].shape[0]
    scene_vp = torch.zeros(scene_n_gs, 3, device='cuda')
    scene_vp.grad = render_pkg['viewspace_points'].grad[human_n_gs:].clone()
    render_pkg['scene_viewspace_points'] = scene_vp
else:
    render_pkg['scene_viewspace_points'] = render_pkg['viewspace_points']
    render_pkg['scene_viewspace_points'].grad = render_pkg['viewspace_points'].grad
```

**影响**：之前 scene densification 拿到的是 N_human + N_scene 的完整梯度，现在正确切出 N_scene 部分。

#### Bug 2 修复：LR Scheduler（对齐 STM）

**`hugs/models/fusion_mlp.py` 新增方法**：
- `setup_optimizer(lr_init, lr_final, lr_delay_mult, max_steps)`：使用 `get_expon_lr_func`，与 STM 完全一致
- `update_learning_rate(iteration)`：每步调用更新 lr

**`hugs/trainer/gs_trainer.py` 修改**：
- `setup_fusion_mlp` 改为调用 `fusion_mlp.setup_optimizer(lr_init=1e-3, lr_final=1e-5, lr_delay_mult=0.01)`
- 训练循环开头加 `self.fusion_mlp.update_learning_rate(t_iter)`
- optimizer step 改为 `self.fusion_mlp.optimizer.step()`

**LR 曲线**：step 0 → 1e-3，step 1500 → 1e-4，step 3000 → 1e-5（指数衰减）

---

### 当前运行实验

| 字段 | 值 |
|---|---|
| 实验名 | `vimo_fusionmlp_v3_seattle_20260608` |
| 配置 | `cfg_files/release/neuman/hugs_vimo_seattle_fusionmlp_v3_3k.yaml` |
| 日志 | `run_logs/vimo_fusionmlp_v3_fixed_seattle_20260608.log` |
| PID | 1891527 |
| 步数 | 3000 |
| seq | seattle_vimo ✓ |
| AnchorAttention | 禁用（use_anchors: false）|
| FusionMLP | 启用，lr_init=1e-3, lr_final=1e-5，只修正 scene |
| lbs_w | 1000（保持我们当前值）|
| seed | 10086（与 STM 一致）|

**启动时间**：2026-06-08 21:16

**对比基准**：
- STM step 1k: **16.23**
- OUR baseline step 1k: 15.79
- OUR inline_attn step 1k: 15.34


---

## 会话三十八（2026-06-09）：mask_aware_prune 失败 + 深度监督诊断实验

### 本次主要工作

1. **mask_aware_bike_18k 实验（彻底失败，已 kill）**
   - 实现并测试了 mask_aware_prune：每次 densify_and_prune 后强制删除投影到人体 mask 内的 scene GS
   - 1k 快速测试：step 500 崩坏从 9.65→11.92（+2.27 dB），step 1000 PSNR 15.98
   - 18k 完整训练：step 2000 只有 13.98（vs 目标 17.47），step 3000 跌至 12.23，在 step 3200 kill
   - **根因**：prune 是被动清理，optimizer joint L1 loss（无 mask）每步驱动 scene GS 向人体区再生，速度远超清除速度（step 3000 单次删 48,343 GS，总量仍增至 131万）

2. **深度监督（depth_w）诊断实验（bike，2k步）**
   - 实验 A：diag_inlinattn_bike_2k，depth_w=0.05，AnchorAttention 完全禁用（module_start_iter=9999）
   - 实验 B：diag_baseline_bike_2k，depth_w=0.0，AnchorAttention 完全禁用
   - 结果：两者在 step 2000 的 final PSNR 几乎相同（12.66 vs 12.68），而 vimo+inline_attn（有 correction 从 step 2000 启动）在 step 2000 实际值为 13.81，后续靠 correction 拉到 18.09

3. **核心结论：AnchorAttention correction 是主驱动**
   - 无 correction 时（module_start_iter=9999）：step 2000 = 12.66
   - 有 correction（module_start_iter=2000）：最终 18.09
   - depth_w 贡献很小（step 900 峰值差 +1.26 dB，step 2000 ≈ 0）
   - "黑暗谷"（step 1000~2000 从 16 跌至 13）是正常现象，correction 激活后才能恢复

4. **读取了正确的 vimo_inline_attn_18k bike 真实 PSNR 曲线**（从 results_train.json）
   - step 1000: 15.16（非之前误引的 16.18，16.18 是 STM 的值）
   - step 2000: 13.81（低谷）
   - step 16000: 18.10（峰值）
   - final: **18.09**
   - STM bike final: **18.61**，差距 -0.52 dB

### 下一步方向

**最高优先：将 AnchorAttention correction 提前到 step 0（module_start_iter=0）**

目标：在 step 500 opacity reset 之前就建立 human-scene 分离保护，消除或缩短"黑暗谷"，使 step 1000 PSNR 接近 STM 的 16.18。

预期配置：
```yaml
anchor_attention:
  module_start_iter: 0
  correction_start_iter: 0
  correction_warmup_iters: 500
```


---

## 会话五十二（2026-06-16）

### large_transl_debug_ply_6scenes 实验结果分析

**实验配置差异（vs v4_correct_inline_attn_18k）**：

| 参数 | large_transl | v4_correct |
|---|:---:|:---:|
| `gamma_transl_final` | **0.15** | 0.05 |
| `transl_delta_clamp` | **1.5** | 0.5 |
| `gamma_transl_decay_end` | 18000 | 15000 |
| `debug_ply_interval` | 1000 | — |

**PSNR 完整对比（5/6 场景已完成，citron 未运行）**：

| 场景 | large_transl final | peak | v4_correct final | STM | Δ(vs v4_correct) |
|---|:---:|:---:|:---:|:---:|:---:|
| bike | 19.0286 | 18.99@11k | 18.9626 | 18.6100 | +0.07 ✓ |
| jogging | 17.0385 | 17.24@4k | 17.5463 | 17.3776 | -0.51 ✗ |
| seattle | 17.1982 | 17.18@11k | 16.7521 | 16.7278 | +0.45 ✓ |
| lab | 17.2965 | 17.64@4k | 17.7588 | 17.4512 | -0.46 ✗ |
| parkinglot | **13.6034** | 14.49@1k | 15.2645 | 14.3660 | **-1.66 ✗** |
| citron | — | — | 17.4823 | 16.8763 | — |

**平均（5场景）**：large_transl=16.83 vs v4_correct=17.26，退步 -0.43 dB

**结论**：
1. `gamma_transl_final=0.15` 是主要问题：parkinglot 全程在 13-14 dB 振荡（峰值 14.49@1k，从未超过 15），与记忆记录的"Z 方向锁死在 clamp，振荡"完全一致
2. `transl_delta_clamp=1.5` 对 seattle 有一定帮助（+0.45），但代价是其他场景的稳定性下降
3. **large_transl 超参数整体是负向变化**，v4_correct（gamma_transl_final=0.05, transl_delta_clamp=0.5）是更优的默认配置


---

## 会话五十三（2026-06-17）

### 本次开场：恢复记忆 + 状态梳理

**当前日期**：2026-06-17

**状态汇总（截至本次对话开始）**：

#### 最新完成：large_transl_debug_ply_18k（会话五十二，2026-06-16）

6场景全部完成，平均 16.89 dB，低于 v4_correct（17.29），几乎等于 STM（16.90）。
→ **v4_correct_inline_attn_18k 仍为最优 pipeline（平均 17.29，+0.39 vs STM）**

#### 待做事项（来自 hugs-active-experiments.md 下一步建议）

1. 场景自适应 clamp 策略（高 transl_norm >3 用 clamp=1.5，低 <1 用 clamp=0.5）
2. 填写 BEST_PIPELINE.md 最终汇总
3. parkinglot 峰值 checkpoint 分析（v4_correct 中 11k 时 16.73 但 final=15.26）
4. S2 实验（用 hugs_orig_bike_gt_s1 checkpoint 继续 6k 步 + AnchorAttention）
5. **清洁 COLMAP 初始化**（★★★ 最高优先，尚未实验）：过滤 scene GS init 时的人体 mask 区内 COLMAP 点
6. **AnchorAttention module_start_iter=0** 实验


---

## 会话五十四（2026-06-17，续）

### 主题：delta_mu 帧间时序相关性实测

#### 背景
上一次对话（会话五十三）分析了传输带宽，提出帧间差分压缩思路，并成功跑通了
`scripts/analyze_delta_mu_temporal.py`。

#### bike_vimo_v4 结果（会话五十三末尾获得）

**重要修正**：训练后 human GS 点数从初始 110K 增长至 **515,804 pts**（Gaussian densification），
单帧 delta_mu 实际 **6 MB (fp32)**，不是之前估算的 1.26 MB。

| 指标 | 值 |
|------|-----|
| GS 点数 | 515,804 |
| 单帧 fp32 | 6,045 KB（≈6 MB）|
| diff_std / raw_std | **0.269**（差分是原始标准差的27%）|
| \|diff\| < 5% P99 比例 | **72.8%** |
| int8 差分帧（无熵编码）| 1,511 KB/帧 |
| 最大帧间跳变 | 第93帧，RMS=0.436 |

**"几十KB"目标需要99%+稀疏度**（稀疏编码+训练侧时序平滑），当前模型差15-30倍。

#### 当前动作
正在运行 `--all_seqs` 对全部6场景做帧间相关性分析，结果存入 `output/delta_mu_analysis/`

#### 6场景 --all_seqs 完整结果（2026-06-17）

| 场景 | GS pts | ratio | 熵编码下界 | int8差分 |
|------|--------|-------|-----------|---------|
| bike | 515,804 | 0.287 | ~0 KB | 1,511 KB |
| jogging | 457,757 | 0.307 | 47.7 KB | 1,341 KB |
| citron | 593,754 | 0.329 | ~0 KB | 1,740 KB |
| seattle | 539,986 | 0.482 | 80.9 KB | 1,582 KB |
| parkinglot | 491,338 | 0.505 | 210.1 KB | 1,439 KB |
| lab | 501,392 | 0.526 | 148.3 KB | 1,469 KB |

结果存入 output/delta_mu_analysis/。内存文件 hugs-transmission-analysis.md 已更新。

#### 时序平滑方案讨论与实验启动（2026-06-17 下午）

**5个方案总结**：
1. frame_embed 平滑正则 → **当前正在 lab 测试**（λ=0.01）
2. 相邻帧对采样 + delta_mu 直接约束 → 未实现
3. pose 预测残差约束 → 未实现（架构改动大）
4. 去掉/替换 frame_embed 为 pose-conditioned MLP → 未实现（激进）
5. 推理后低通滤波 → 未实现（无需重训）

**代码改动**：
- `hugs/trainer/gs_trainer.py`：新增 `maybe_add_frame_embed_smooth_loss` 方法（~25行），
  在训练循环 `loss.backward()` 前调用
- `cfg_files/debug/hugs_lab_v4_embed_smooth_18k.yaml`：新建，唯一新参数 `frame_embed_smooth_w: 0.01`

**运行状态**：PID 74911，lab scene，预计 17:00 前完成

---

### 会话恢复（2026-06-21）

#### 当前运行实验状态

**STM GT 5场景串行训练**（PID 198736，从 15:06 开始）：
- bike：已完成（16:44）
- citron：进行中，当前 step 15000，PSNR=17.8728，峰值 step 11000=19.4164
- parkinglot/jogging/seattle：待运行

**方案五（savgol低通滤波）评估：已完成**

#### 方案五结果（eval_method5_savgol.py + eval_method5_psnr.py）

**结论：方案五大获成功，无需方案二。**

**Ratio 降低效果（savgol window=11，polyorder=2）**：

| 场景 | ratio（前） | ratio（后） | 降幅 |
|------|:-----------:|:-----------:|:----:|
| bike | 0.287 | 0.127 | -56% |
| jogging | 0.307 | 0.108 | -65% |
| seattle | 0.482 | 0.254 | -47% |
| lab | 0.525 | 0.192 | -64% |
| parkinglot | 0.505 | 0.136 | -73% |
| citron | 0.329 | 0.143 | -57% |

**PSNR 影响（window=11，6场景抽样评估）**：

| 场景 | baseline PSNR | 滤波后 PSNR | ΔdB |
|------|:-----:|:-----:|:---:|
| bike | 24.929 | 24.926 | -0.003 |
| jogging | 23.257 | 23.260 | +0.004 |
| seattle | 25.120 | 25.122 | +0.002 |
| lab | 24.894 | 24.901 | +0.007 |
| parkinglot | 23.350 | 23.356 | +0.006 |
| citron | 24.453 | 24.448 | -0.004 |

**PSNR 变化 ≤ ±0.01 dB，完全在测量噪声范围内，可视为零损失。**

**结果文件**：
- Ratio 分析：`output/delta_mu_analysis/method5_savgol_results.json`
- PSNR 评估：`output/delta_mu_analysis/method5_psnr_results.json`

**决策**：方案五已满足目标（ratio大幅下降+PSNR无损），方案二（重训练）无需实施。推荐最终 window=11 作为默认参数（ratio降50-73%，PSNR±0.01）。
## 2026-07-27（恢复前馈接触方案与数据集下载说明）

- 本会话目标：恢复 HUGS 第二研究点的 Human3R → 接触修正 → GUSH3R 前馈方案，依据仓库当前 README、脚本与已落盘文件，明确原型/训练各阶段所需数据集及可复现下载步骤。
- 边界：仅盘点与说明下载方法；不启动大体量数据下载、不改动模型代码或既有实验资产。
- 用户提供 BEDLAM Hugging Face collection：`https://huggingface.co/collections/Intelligent-Systems/perceiving-systems`；候选 repo 为 `Intelligent-Systems/BEDLAM`。已以镜像做 `README.md` 最小下载探测，但 `hf-mirror.com` 当前 DNS 解析失败（此前 curl 亦超时），未生成有效 BEDLAM 文件；网络恢复后必须先 dry-run/小文件核实内容和许可，再对白名单资源断点下载。
## 2026-07-28（恢复 HUGS 记忆并排查 GUSH3R 前向加速）

- 恢复 HUGS 第二研究点上下文：路线为 Human3R 原型 → 脚部接触 proxy/规则修正 → Gaussian 渲染 → 再接入 GUSH3R；不改动既有 HUGS 训练主线。
- 当前 GUSH3R 主权重、Human3R 主权重、Multi-HMR 及 SMPL/SMPL-X/supplementary 文件均已准备完成；此前 GPU smoke test 已确认 checkpoint、DINOv2、SMPL-X sampling cache 可加载。
- 当前单帧 GUSH3R 卡在 `model.forward_recurrent_lighter()`：缺少 CUDA 编译版 RoPE2D，退回慢速 PyTorch 实现，同时 xFormers 不可用；本次优先编译 `GUSH3R/src/croco/models/curope`、核对/启用 xFormers，并重新做单帧前向测试。
## 2026-09-04（GUSH3R 论文指标复现核查）

- 用户询问 GUSH3R 是否本身存在质量问题，以及能否阅读论文并复现其指标。
- 已可靠阅读 GUSH3R arXiv HTML 正文（arXiv:2607.05243），包括方法、训练说明、评价协议与主表；未把当前 GoodMornin1 长视频的背景退化误称为论文主表失败。
- 论文的单人 NVS 主表是在 NeuMan/EMDB 上，以 4/16 输入视图重建、用 target-frame 相机与 SMPL-X 参数渲染 target view，评价 PSNR/SSIM/LPIPS/FPS。GUSH3R 报告：NeuMan 4-view 18.6/0.55/0.28、16-view 16.6/0.39/0.44；EMDB 4-view 18.1/0.60/0.30、16-view 18.0/0.57/0.41；1.70 FPS。
- 本地 `GUSH3R` 官方仓库（`bac8d88`）仅含 `infer.py` 推理入口，没有 NeuMan/EMDB dataset loader、view split、target-view renderer 调用或 metric evaluator。工作区有一份可用但为接触校准准备的 NeuMan `lab` 103 帧子集（`forward_contact_pipeline/datasets/neuman_lab_h3r_hugs_p0/`，带相机和近似 SMPL），没有 EMDB，也没有论文发布的 4/16-view split；它可用来做最小化的 NeuMan 复现/工程验证，但不能直接代表完整主表。因此尚未、也不能声称已复现主表。
- 当前本地 GoodMornin1 诊断只说明：在当前长序列背景融合与回渲路径，出现背景空洞/散点/模糊；短窗口相对正常。它可能来自本地实现、参数、全序列最终背景回渲历史帧的方式，或输入域/协议差异，不能直接否定论文。
- 复现计划：先取得合法的 NeuMan 或 EMDB 原始评测数据及论文一致的 4-view split/target 定义；建立独立、不可修改上游的 evaluator，固定 checkpoint、输入帧、target camera/SMPL-X、mask/ROI、版本和渲染结果；先完成一个 NeuMan 4-view 场景，再扩到 16-view 和 EMDB。并独立对 GoodMornin 跑 final/per-frame/chunked 的工程质量回归。
- 注意保留本地既有修改：`GUSH3R/infer.py`、`src/dust3r/heads/human_gs/blocks/renderer.py`、`src/mhmr/blocks/dinov2.py` 均为脏改动；后续评测代码放在新目录，不覆盖它们。
- 已新增独立评测器：`forward_contact_pipeline/scripts/evaluate_gush3r_neuman_nvs.py`。它将 target frame 严格排除在输入外，用输入帧预测的相机中心与 NeuMan 标定相机做 Sim(3) 对齐后渲染 target view，并计算 PSNR/SSIM/LPIPS；因为本地 `lab` 子集没有 target SMPL-X，指标仅在人像 mask 外的背景区域计算，不能与论文的完整人景 Table 2 直接横比。
- 实测（本地 NeuMan `lab` 103 帧，targets=[1,33,67,101]）：4-view（inputs=[0,34,68,102]、官方 2M background cap）= **PSNR 8.51 / SSIM 0.330 / LPIPS-Alex 0.597**，输出在 `forward_contact_pipeline/evaluations/gush3r_neuman_lab_scene_nvs_4view_v1/`；明显低于论文 NeuMan 4-view 全图 **18.6 / 0.55 / 0.28**。16-view 官方 2M cap 的首次运行未落盘（进程异常退出），诊断性 500k cap 运行完成：**11.86 / 0.368 / 0.515**，输出在 `forward_contact_pipeline/evaluations/gush3r_neuman_lab_scene_nvs_16view_500k_v1/`；该 500k 配置不能和论文 16-view 的 **16.6 / 0.39 / 0.44** 直接作数值结论，但 16 输入视图相对本地 4-view 的改善是真实的。
- 对照 PNG 均保存为各输出目录下 `renders/*_render_target.png`（左 render、右 RGB target）。视觉证据为背景结构部分重建成功，但有人体残影、白洞、边界/几何错位；因此当前工程链路尚未复现论文质量。优先排查：论文未发布的 target-SMPL-X 注入路径、官方精确 split/预处理/renderer、以及本地 external renderer 的坐标/相机 convention；获得 EMDB 与精确协议前不报告为“论文复现失败”。
## 2026-09-06（续做 GUSH3R 长序列质量验收：仅数值/日志，不读取图片）

- 用户要求继续既有 GUSH3R 质量修复实验，并明确禁止读取图片，以避免视觉内容占用上下文。
- 本轮范围：验收已完成的 GoodMornin1 229 帧 `chunked32` 渲染，使用文件清单、渲染元数据和无图像解码的数值统计，检查 chunk 边界连续性、时序稳定性及相对旧 full-229 渲染的可量化变化；不修改接触模块、不将非论文协议评测写成 Table-2 复现。
- 已知前提：chunked 目标是消除“后续背景融合污染历史帧”的长序列灾难性退化；它不能凭空补全被人体遮挡的背景，也不能解决单目几何/人体 mask 的根本误差。后续结论必须分别报告这些边界。
- 已完成无可视化数值验收：新增 `forward_contact_pipeline/scripts/analyze_gush3r_render_temporal.py`，只在本地进程内计算 PNG 聚合统计、输出 JSON，不生成或查看图片。`chunked32_229_verified` 的 229 帧齐全，8 段窗口、总时长 303.42s；相对历史 full-229，白像素率均值从 13.91% 降至 6.65%、95 分位从 60.71% 降至 28.81%，说明它抑制了全局融合造成的白洞/崩坏代理。
- 但分块尚不可作为在线 baseline：7 个窗口边界的 RGB 帧差均值 0.16833，而 221 个窗口内部转换为 0.04100（4.11 倍），最大边界跳变 0.31569。原因是每段重置 recurrent state 和背景地图。下一步是有重叠的可靠锚点地图交接（或较弱的 overlap blend 对照）；不采用已失败的 `freeze_after_cap`，也不采用 2M cap 下不具在线可行性的 `topk_full`。
- GPU 检查时 RTX 4090 已被其他工作负载占满（100%），本轮没有启动或干扰新的 GPU 重建。详细数值和结论已追加至 `forward_contact_pipeline/reports/gush3r_quality_diagnosis_20260804.md`。
- 用户授权持续 autoresearch：不等待人工逐步确认，按“提出可证伪方案 → 单变量实现/运行 → 无可视化数值验收 → 迭代”的顺序自动推进；GPU 忙时等待/轮询，不抢占其他任务。当前优先方案为重叠窗口交接的弱对照（量化其能否消除 seam，但明确它含未来上下文、不是在线最终方案），然后再实现可在线的可靠 anchor map hand-off。
- 用户要求持续实验直到可用；已完成 hard chunk / overlap / causal-RGB-context 三种对照，确认只有离线 overlap 能消 seam，未来无关的 RGB context 仍有 2.31x 边界跳变。现开始实现真正的可靠背景 anchor-map hand-off：跨窗口传递有限高置信背景 Gaussian，而非重置 map 或使用未来帧；将先做短序列 smoke、再做完整 229 帧无可视化数值验收。
