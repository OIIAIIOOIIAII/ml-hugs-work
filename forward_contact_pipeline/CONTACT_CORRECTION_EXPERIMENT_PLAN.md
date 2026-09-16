# 前馈接触修正：活实验计划

 > **状态：2026-09-16；维护规则：每次启动、结束或否定一项接触/Gaussian 实验时，必须更新本文、`TODO.md` 与 `CLAUDE_SESSION_LOG.md`。**
>
> 本文是第二研究点的当前执行协议；旧的方案文档说明设计动机，本文定义可执行实验、比较口径和停止条件。

## 当前执行补充：RICH获取与处理（更新2026-09-16）

**9月16日Git迁移同步**：用户要求同步本地仓库，本轮整理HUGS/contact/RICH源码、配置、测试与文档快照，并核验第三方固定版本和补丁。RICH数据、模型、凭据及运行产物不进入Git；新增审计覆盖元数据TSV和数据处理依赖。远端仍为既有公开仓库，未改变可见性；GitHub认证缺失，推送等待本机仓库专用部署密钥获授权。未启动真实训练。
**9月16日11:00监督候选清单完成**：`processed/dataset_v1`已正式发布；261429图片头/尺寸/EXIF全部检查（identity），1206标注分片全部重新SHA256/数值校验。冻结train39序列/167710候选样本、ParkingLot2内部val23序列/75084候选样本，capture/scene/sequence/subject互斥；有效人体帧37585。全样本独立核对图片/真实帧/相机/分片行号与计数、产物hash复核、10项回归通过。缺标定的14987张camera10图片、84个无SMPL-X颜色标注帧和10个缺contact帧明确记录，原始数据保留。完整契约见`RICH_DATASET.md`，完成证据`reports/rich_processing_20260916/dataset_completion_audit.json`。

**本轮发现的可见性门槛**：1290个数值投影抽查视角均为正深度，但17个无GT顶点落在图像范围内，38个少于半数顶点在图像内。需核验逐帧可见性/目标匹配及标定，不能把候选清单直接作为最终训练cache或将数值统计当作像素/前端精度验证。两个16帧小样本计划已落地，未启动前端推理。

**前端接入边界**：GUSH3R当前使用默认K并有EXIF/resize/crop/pad处理，现有导出缺完整RGB tokens、局部邻域、真实RICH帧映射与LBS binding；普通推理入口导出后仍渲染。后续先做纯数值因果前端适配与几何测量，不把GT补进inputs。未启动推理/训练，官方val/test未下载。

**9月16日完成确认，覆盖以下全部下载进度/ETA**：本批train JPG于02:15下载完成，04:42完整gzip CRC/trailer及SHA256通过并正式发布`datasets/RICH/extracted/train`。261429图片、62序列、559.593GB；人体/接触GT37669帧/1206片与公共包均完成。全部contact人体帧均找到至少一个官方metadata允许相机的图片（缺失0），完整索引无重复、字节数一致，首/中/末文件hash回读通过。下载与解压进程已正常结束，非异常退出。

**后续门槛保持**：图片有4112×3008、3008×4112、4096×2700三种尺寸；基础文件匹配不等于K/方向或前端误差审计通过。下一步完成逐相机几何核验、开发split冻结、真实冻结前端缓存及独立误差测量；通过后才启动真实训练。官方val/test未在本批下载。完成报告`reports/rich_processing_20260916/image_completion_audit.json`，training_ready=false。

**23:53联合状态复核**：JPG97.6302%、余13.257GB，近5分钟均速0.4485MB/s、14连接，按此约8.2小时，恒速预计9月16日08:06；覆盖此前1–2点ETA且不含校验解压。图片流式处理进程正常等待压缩偏移160.413GB处未完成piece，已暂存20序列的73599图片/160.446GB；gzip不能跳过缺口，不能将后部已下载块计为已解压图片。GT37669帧/1206分片已完成。整包CRC、最终图片覆盖/K与真实前端审计仍待完成，training_ready=false。

**20:18恢复提速结果，覆盖旧下载ETA**：原JPG仅10个存活连接、近5分钟0.266MB/s；一次正常保存断点后由原controller刷新官方会话，恢复16连接（aria2 PID200625）。随后3分钟测量，排除首分钟后均值1.046MB/s、约提高3.92倍。当前96.5689%，余19.194GB，按此约5–6小时（9月16日凌晨1–2点），非保证且不含校验解压。partial原inode保留，图片处理PID67144保持，73599图已暂存、等待160.413GB处缺口。没有把本次短时改善解释为唯一瓶颈已确定，未启动训练。证据`datasets/RICH/download_logs/speed_refresh_20260915/summary.json`。

**19:06图片提前处理更新**：用户授权边下载边处理，已启用`--stream-jpg`，处理PID67144替换仅等待的旧controller，下载未重启。读取aria2位图证明完整的约160.2GB连续前缀，遇空洞等待并保留gzip状态。暂存3548图片/6.500GB，逐文件SHA256、JPEG头尺寸和sequence/camera/frame/subject索引已生成；正常运行至完整gzip CRC/trailer通过后直接发布，不重复解压。首批存在4112×3008/3008×4112横竖尺寸，473图相机不属于官方metadata选用集合，均显式记录；后续相机K/方向和采样仍须审计。临时资产不进入训练，training_ready=false。7项流式及3项归档测试通过，说明见`RICH_PREPARATION.md`。

**15:23处理状态更新**：GT分片与公共/body/contact归档处理均完成。全量37669人体帧→1206分片（10.453GB），逐帧顶点/面顺序核验通过，最大坐标分量差5e-9m；索引恰好覆盖全部contact帧，无重复或遗漏，首/中/末分片校验回读通过。10帧body-only不进入接触监督；84帧无颜色以SMPL-X valid=false保留，SMPL-X有效帧37585、有效顶点positive比例3.676%。这不等于真实前端几何精度通过；图片下载/匹配、K与开发split、真实前端inputs exporter及独立审计仍待完成。处理controller仅等待完整JPG；标注controller已正常结束。证据`reports/rich_processing_20260915/annotation_completion.json`。

**14:34下载时间复核，覆盖本节旧ETA**：JPG93.4724%，剩36.5167GB，近期5分钟28个摘要平均0.4881MB/s，按此约20.8小时、恒速预计9月16日11:21；建议预留约1天，仍不包括完整性检查/解压。body已完整解压并校验，contact解压约97.34/109.24GB，GT数值分片34503/37669帧。三个controller仍在运行，训练门槛不变。快照`datasets/RICH/download_logs/eta_latest_20260915.json`。

**9月15日14:00现场状态，覆盖以下全部历史快照/ETA**：contact、body及公共包已下载；JPG约93%、余约38GB，昨夜长度异常退出后恢复PID1464854，新增有界重认证续传。公共scan/calibration和world包CRC/SHA256完成，body正在解压，PID1471505随后处理contact并自动等候完整JPG（最多36h）。完整归档发布前不使用partial。

**GT注释准备已启动**：官方metadata固定revision并验hash，清单62序列/64人物轨迹、37669个contact人体帧，10个body-only帧显式记录；统一相机、scan及world变换目录已落地。全部轨迹共192帧抽查确认body PLY与contact OBJ顶点/面顺序一致（最大分量误差5e-9m）。严格区分pickle的6890维SMPL标签与OBJ的10475维SMPL-X标签；3个抽样无颜色帧标为contact_smplx_valid=false，不当全负标签。全量numeric分片PID1507008运行，14:00完成1082帧/35片，输出`datasets/RICH/processed/annotations_v1`；详见`RICH_PREPARATION.md`。16项下载/解压/标注测试通过。GT数据不能充当真实冻结前端inputs，训练未启动。

**待完成的真实数据门槛**：完整归档与全量数值整理→JPG逐帧/分辨率/K审计→开发split冻结→真实前端误差复测与独立审计。metadata揭示跨场景subject连接组为{BBQ}、{ParkingLot2}、{LectureHall,ParkingLot1,Pavallion}，按scene+subject隔离不得拆分组。ParkingLot1原world旋转系数只有三位小数，正交残差约5.4e-4，保留原值并记录，不宣称其已通过接触尺度坐标精度审计。

**工程准备完成（9月14日）**：统一数值cache Stage-A训练层已实现，模型/损失/数据通过factory和版本化配置替换，机器路径独立；校验、训练、显式评测、epoch续训与结果追踪可执行，16项回归/独立源码副本迁移通过。原Stage-B脚本保留。后续按`TRAINING.md`接入RICH exporter/真实前端cache及独立审计报告，不能将合成软件fixture当真实实验。完整LBS绑定、目标机GPU环境仍待完成。Git迁移准备见`../MIGRATION.md`，尚未提交或推送。


**19:11下载快照（覆盖本节旧ETA）**：body已下载完成，contact已自动续传至90.17%，约剩1小时40分；JPG84.26%，剩88.04GB，最近5分钟平均4.28MB/s，较18:56减速，按当前均值/最新速度估约6–8小时（9月15日凌晨1–3点），不含完整校验/解压/审计。controller PID1391152仍运行。快照`datasets/RICH/download_logs/eta_latest_20260914.json`。

**下载完成后的首轮迭代范围（用户9月14日确认）**：当前train JPG/SMPL-X/dense contact+公共scan/calibration/world transform与现有PROX足以启动第一轮接入和受控实验，不再把继续搜集新数据集作为起步前提。先完整性与短序列标签/顶点/坐标审计，随后冻结按scene/sequence/subject隔离的开发验证划分并复测RICH真实粗前端；只有输入契约满足才训练Stage-A基线与RGB/Gaussian消融，通过后接Stage-B/LBS。当前未下载官方val/test，若train内部无法形成有效隔离应先补validation，最终正式泛化评测补齐官方val/test且test不调参。数据到位不自动解除既有Human3R/GUSH3R接触尺度几何误差门槛。


**9月14日现场复核覆盖下方旧进度与ETA**：原controller早已退出，已刷新认证并保留断点恢复PID1391152（2任务×16连接，非JPG500K）。18:52 JPG82.89%（463.69/559.42GB）、body95.19%下载中，contact89.90%排队；最新约2分钟JPG均值9.91MB/s，控制文件完整piece增量约10.17MB/s。按当前速度约剩2.7小时，不承诺长期速率。新增DAMON/BEHAVE公开来源说明；前者为图像接触标签补充、相机/人体为CLIFF估计，后者主要为SMPL人体与物体交互，均不能直接替换RICH契约。PhySIC公开demo不等于独立数据集已到位。详见`reports/rich_sources_20260914.md`。

**HF-Mirror最新候选覆盖下面的初次检索结论**：镜像搜索/metadata已可访问，全文检索发现`Yong-Hoon/human3r-dataset`实际发布10个RICH分卷，共486.0GB。它需要HF人工审批访问；当前未核验内部train/val/test、分辨率、原始帧、SMPL-X/contact及场景字段，不能直接替换559GB官方训练JPG。原下载保持。授权后先小范围核对归档及测速，再决定所需文件；入口和清单见`reports/rich_mirror_search_20260910.md`。

**镜像检索更新**：本轮未找到经验证的完整训练JPG替代源。OpenDataLab仅有介绍、文件数0且明确指向官网；SA-HMR/GVHMR/WHAM是预处理或评测资产；DECO Keeper可下载45.2MB包，已读前缀为测试NPZ的图片路径数组，训练图片仍需官网。HF等平台受本机连接失败限制，不能断言全网没有镜像。约107GB官方BSTRO TSV可能减少接触检测实验的图片传输量，但需要核验帧号、SMPL-X和场景对应，不能自动替代当前数据契约。详见`reports/rich_mirror_search_20260910.md`；原下载配置保持。

**当前直连复测覆盖下方历史速度/ETA**：用户没有云平台账号或中转地址，继续本机直连，RICH官方凭据仍有效。20:45–20:51完成16/8连接各180秒图片独占测试，去掉首分钟后平均0.442/0.152MB/s；测试前并行约0.315MB/s，未测到稳定多MB/s的额外提速。已保留全部断点恢复原2文件×16连接、500K非图片限速，controller PID1305167。20:54恢复并行后JPG平均0.288MB/s、body0.325MB/s；JPG剩约554GB按当前恒速约22.3天，早先5–7天ETA失效，线路波动时需重估。JPG单连接403、其余15连接继续传输，未增加并发。证据见 `datasets/RICH/download_logs/jpg_local_benchmark.json`；不因下载速度问题改变训练数据准入门槛。

已恢复来源会话 `01a06025-0f9a-7ca3-8be1-17ba3a60823f` 的阶段上下文，交接见 `reports/hugs_context_restored_20260910.md`。A5继续优先补RICH真实稠密接触监督；数据到位不自动解除Human3R/GUSH3R的真实前端门槛。

以尽早启动字段审计为目标，获取顺序调整为公共scan/calibration/world transform + train bodies/contact（约39.2GB）→同序列train JPG→扩展train/val→冻结test评测。sample 56GB为可选项，内容完整性待查；完整JPG主线约1087GB，实际传输/解压体积待核验。此顺序覆盖旧“强制sample、先全test”的下载建议。NAS约32TB可用，数据与解压产物放NAS。

下载方案及命令见 `reports/rich_fast_download_plan_20260910.md`：先核验授权脚本的实际认证和大包Range，再用aria2从2文件×4连接起测，保留断点；当前只有wget/curl/tmux/conda，尚未安装aria2、获得真实资源URL或实测吞吐。只读文本、目录与数值，不查看图像。到位后按train序列核验标签顶点对应、正负分布和相机/场景坐标，再进入Stage A；test不参与开发调参。

后续现场更新：用户已提供5个资源URL，multicam2world.zip已完成4304字节下载和ZIP CRC校验。受保护下载入口返回POST登录表单，当前改用 `scripts/acquire_rich.py --configure` 交互配置凭据并自动启动两个wget后台任务；有可用GET/cookie认证之前不直接用aria2。认证信息尚未配置，其余4个资源未开始下载。脚本本地HTTP续传与错误页拒绝测试通过；真实站点的Range/吞吐仍待认证后核验。

最新执行更新：凭据已验证，原预检错误为urllib未保留POST→302的会话Cookie，已修复。4个资源Range预检通过，aria2已安装并经Cookie GET启动，当前2文件×8连接、后台PID929174，scan/body在下载，contact/JPG排队。近期合计平均1.387MB/s（较wget样本约18.3倍），断点接续与稀疏文件保护回归通过；以真实进度摘要而非稀疏文件逻辑大小判定完成。Train JPG实际559418993166bytes，全部4个保护包约598.3GB。下载完成后的归档校验/字段审计仍待执行，不能将下载已启动写成数据已到位。

图片ETA优化更新：已对真实JPG作8/16连接短测，当前切到16连接、JPG优先且不限速、非JPG限速500K，PID1011347。scan已完整下载且64成员CRC通过；JPG/body在下载，contact排队。图片近期平均约1MiB/s，按2026-09-10 18:21的余量估算约6.16天，规划5–7天；短测1.8MiB/s与启动6.7MiB/s不能当长期保证。可在首部完整帧和匹配标注到位后先做小样本审计，正式全量训练仍受数据完成度及既定门槛约束。

## 1. 固定研究问题

在不逐帧优化全部 Gaussian 的条件下，能否以**当前 RGB 帧**和过去的**模型自身**接触/控制状态为输入，建立在冻结前馈人体—场景 Gaussian 粗重建之上，因果地估计局部人体—场景接触及其不确定性，并只修正低维的 SMPL-X root/foot 状态，从而降低 foot sliding、penetration 和 contact jitter，同时在场景几何不可靠时安全回退？

第一版仅处理左右脚—地面接触。场景 Gaussian 保持不动；修正后的 SMPL-X 通过 LBS 更新 human Gaussian/mesh。该工作不把“修复背景白洞或全局 Gaussian 漂移”当作接触模块的目标。

## 2. 当前事实与范围边界

| 项目 | 已验证事实 | 对本计划的含义 |
|---|---|---|
| Human3R 人体前端 | 可稳定导出 SMPL-X、point map、depth/confidence；视觉人体结果可用 | 是人体输入候选，但不能把视觉可用等同于接触尺度准确 |
| Human3R -> PROXD 对齐 | 按帧 hold-out Sim(3) median 7.37 cm、p95 45.1 cm | 禁止把当前输出直接作为真实接触监督或直接强制 IK |
| PROX | 已有 recordings、PROXD、scene scan/SDF、相机资产 | 是接触几何教师和真实验证主数据，不需立即换数据集 |
| synthetic drift | 严格跨 scene 测试 GRU/TCN 退化为近似 all-positive，contact F1 约 0.218 | 旧 scene-disjoint split 仅保留为 domain-generalization 压力测试；不能据此调阈值宣称有效 |
| GUSH3R | 短窗可用；长序列背景地图漂移，简单 anchor handoff 失败 | 暂不作为在线接触主表；仅作 renderer/地图退化对照 |

## 3. 数据与标签

### 3.1 立即使用的数据

- **PROX**：RGB/RGB-D、PROXD SMPL-X、scene SDF/scan、相机标定。用于 oracle/degraded scene 实验、真实场景验证及局部几何教师。
- **现有 Human3R 输出**：只用于前端误差与坐标对齐诊断，不能越过对齐门槛生成“真”接触标签。
- **现有 GUSH3R 输出**：只用于局部 Gaussian proxy 的退化输入和 renderer 对照。

### 3.2 后续而非当前阻塞的数据

- **BEDLAM**：用于大规模人体运动/接触预训练，提供更多 pose、速度和遮挡扰动；在 PROX 的机制闭环成立后再接入。
- 不为“修 GUSH3R 背景”盲目下载 TB 级数据。若 P4 需要训练 scene-state，再单独确定带真实相机/深度/静态扫描监督的数据集及许可。

### 3.3 标签定义

用 scene SDF 和脚底 anchor/顶点 ROI 计算：signed surface distance、normal、penetration depth、接触状态、接触期间切向速度。每个标签必须有 `source`、`confidence` 与坐标系字段。

## 4. 模型与在线执行契约

最终每帧输入是当前 RGB `I_t` 和模型自身 `s_{t-1}`。`I_t` 先经过**冻结的前馈人体—场景 Gaussian backbone**，得到可复用 image/crop token、coarse scene/human Gaussian proxy 与 SMPL-X/root；`s_{t-1}` 仅含过去模型预测的 contact、实际施加 action、episode anchor、uncertainty/reset，绝不含 teacher 或未来信息。RGB/mesh-point local contact encoder 先估计 vertex/ROI contact、continuous proximity、geometry reliability；随后 `H=5--8` 的因果控制器用当前 relation token 和 `s_{t-1}` 输出 correction。当前 E1/E2 的 58D 几何输入只是 Stage B 控制原型，不能当作最终接触估计器。

最终网络是 local contact encoder + 小型 causal GRU/TCN 的两层结构；默认时序 hidden=128。输出为：

```text
left/right contact logits
surface distance / normal / penetration risk
root residual (6D) + foot/ankle pose residual
uncertainty + reset flag
```

执行门控：只有 `contact high && local_geometry_reliable && uncertainty low` 时才施加低维修正；否则保留原始姿态并标记 `Uncertain`/`Reset`。禁止将 Gaussian center 直接当作接触表面。

## 5. 分阶段实验与门槛

### E0 — 数据/坐标契约

**目的**：保证每个样本的人体、相机、SDF 和局部 surface 在同一坐标系。

- 建立 sequence manifest、frame-name 对齐、单位/手性检查、SDF 有效覆盖统计。
- 输出足底 anchor 到 SDF 的距离和 mask/投影一致性审计。
- **门槛**：对齐误差达到接触尺度，或显式把训练/评测改为 camera-local/contact-local 坐标；否则不得进入 Human3R 的真实监督。

### E1 — Oracle-scene 机制上界

**目的**：隔离场景重建问题，先问“接触修正机制本身是否成立”。

```text
PROXD clean SMPL-X + true PROX SDF
  -> 注入已知 root/foot 时序 drift、jitter、penetration
  -> causal contact network / rule baseline
  -> 预测 contact 与 residual
  -> 以原始 SDF 评估修正后几何
```

- 对照：原始扰动、EMA/规则投影、GRU、TCN、GRU/TCN+IK/penetration projection。
- split 先采用**同 scene/动作、drift seed 互斥**的机制恢复 split；之后才报告严格 scene-disjoint 泛化 split。
- **通过门槛**：相对未修正输入，在独立 drift seed 上同时降低 penetration、foot sliding、contact jitter；不允许只靠全正 contact 得到表面指标改善。

### E2 — 局部几何退化与不确定性

**目的**：模拟前馈 scene 质量不佳，验证系统不会把人吸到错误表面。

- 从 E1 的 SDF/local proxy 人为加入深度噪声、surface normal 偏差、缺失、错位、延迟和人遮挡洞。
- 训练或校准 uncertainty；报告 calibration、coverage-risk curve、错误修正率与回退率。
- **通过门槛**：几何质量下降时，修正幅度应收缩、回退增多；高不确定帧不得显著增加 penetration 或姿态跳变。

### E3 — 真实前端接入

**目的**：从 oracle human state 过渡到 Human3R/前馈人体估计。

- 先完成 E0 的坐标门槛；必要时在 camera-local contact frame 预测 residual。
- 训练输入使用真实前端误差分布或可量化的 error injection，而不是把 PROXD 直接伪装成前馈输出。
- 对照原始 Human3R、规则、网络修正和 uncertainty gate。
- **停止条件**：若对齐仍大于接触尺度，只做前端误差诊断，不产出真实接触主表。

### E4 — Gaussian/scene-state 接入

**目的**：在冻结人体接触模块的基础上，验证渲染与局部地图接口。

- 先使用稳定的局部 scene proxy；再替换为 GUSH3R 或新 streaming scene-state 的 proxy。
- scene Gaussian 永不被接触模块逐帧优化；只用 corrected SMPL-X LBS 更新 human Gaussian。
- 报告 contact ROI quality、human flicker、foot sliding、penetration、端到端 FPS/latency、失败/回退率。
- **停止条件**：若 scene-state 在接触 ROI 的可靠性不达标，只报告 degraded-map robustness，不把它包装为在线完整人景系统。

## 6. 训练损失与评测

初始多任务损失：contact BCE + surface distance SmoothL1 + normal cosine + root/pose SmoothL1 + penetration + temporal/velocity + uncertainty calibration。时间项必须基于相邻预测或整窗输出计算，不能仅用输出幅度正则冒充时序监督。

每个实验至少报告：contact precision/recall/F1、transition F1、surface distance MAE、penetration ratio/depth、foot sliding、contact jitter、root residual RMSE、uncertainty calibration/abstention。Gaussian 接入后额外报告 contact-ROI PSNR/SSIM/LPIPS、human Gaussian flicker、全图质量变化、FPS、显存、packet bytes。

## 7. 记录与可复现规则

每个实验目录必须保存：命令、git/version、数据 manifest、split、随机种子、标签来源、checkpoint、完整 metric JSON、失败原因。报告中必须标为下列之一：

- `mechanism result`：oracle 或受控扰动下的模块能力；
- `robustness result`：退化场景下的安全性；
- `real-front-end result`：通过坐标门槛的真实前馈结果；
- `renderer diagnostic`：非正式接触主表的 GUSH3R/renderer 结果。

禁止把 synthetic / proxy / 非论文协议结果写成真实在线人景提升，也禁止仅用 pooled frame split 取代 sequence-safe split。

## 8. 当前下一动作

1. 冻结 E1b soft-gate Pareto 表（BCE×1、×1.5、×2）；正式选择必须同时看 contact/transition F1 和 sliding，不能只按训练 loss。
2. 冻结 E2 medium 的第一轮结论：冻结 E1b 在退化 proxy 上失败；几何增强训练可恢复部分控制但未恢复可靠 contact transition。停止继续搜索单一 confidence threshold。
3. 实现最终 Stage-A 数据契约：从 PROX 生成 `RGB token reference + foot ROI vertices + local scene point/Gaussian patch + vertex continuous proximity + ROI contact/reliability label`，且用 source-sequence、scene、error-family 三重隔离。**不再把当前 PROX 2 cm ROI 标签写成 vertex binary contact**：审计已证明一个 ROI 内的 64 个 hard label 完全相同；真正 dense vertex binary contact 留给 RICH/PhySIC 类标注。
4. 先训练小型 local relation encoder，第一项可证伪基线是 continuous signed proximity + ROI contact，再显式输出 normal/distance reliability 与 uncertainty；其 token 通过门槛后才接已有 Stage-B causal controller。E2 的主比较变为 no-correction、frozen E1b、geometry-augmented E1b、Stage-A+B uncertainty/abstain；阈值仅在 validation 固定。
5. 并行完成 Human3R -> PROX 的 camera-local 坐标/重投影诊断；未达门槛不进入 E3。
6. 将 Splat-SAP-style point-map 作为 P4 scene proxy 目标、HumanGS-style canonical LBS 作为 P5 human branch 目标；二者代码/许可未核验前不替换当前可运行接口。

### 8.1 无误差参考的来源（2026-09-09 澄清）

训练样本不需要存在“完美 Gaussian 参数”。对同一真实 RGB 帧，冻结前馈 backbone 输出的 coarse human/scene Gaussian 和 SMPL-X 本身就是有误差的输入；监督参考独立来自 PROX 的 PROXD SMPL-X、官方 scene scan/SDF、相机及可选 RGB/mask/depth。第一版因此监督 corrected geometry/contact/render，而不是回归一个虚构的 GT Gaussian。后续若需要 Gaussian-parameter supervision，可离线将 GT SMPL-X canonical human Gaussian 经 LBS 与 GT scan 拟合为 reference；它只能作补充，不可替代真实 coarse-backbone 输入。

### 8.2 A2 连续几何基线（2026-09-09）

在 12 PROX sequence 的严格 scene/sequence/error-family split 上，geometry-only local relation encoder 的 test signed proximity MAE 为 **1.27 mm**（constant-zero 为 1.79 mm；distance-corrupted local patch 的 naive nearest-point 为 12.64 mm）。这说明连续 SDF 接近度可作为 PROX 上的有效 Stage-A 几何目标。其 ROI contact head precision=.429、recall=1.0，仍退化为 all-positive；不记录为 contact F1 成功。完整协议/数字在 `reports/stagea_proximity_roi_v2_20260909.md`。下一动作是 reliability/calibration/abstention，不接 controller 或 Gaussian LBS。

### 8.3 A2 reliability 停止条件触发（2026-09-09）

局部 mesh/point patch 的 Laplace uncertainty 在 strict unseen-error-family test 上失败（lowest 50% MAE=.743 mm，高于全量=.698 mm）；即使每种已声明 proxy corruption 都加入训练、scene 保持互斥，test 也只有 50% 子集偶然改善（.653 vs .716 mm），25% 反而恶化且误差 bins 不单调。故 local-only uncertainty 不可作为 online safety gate，停止调参。下一阶段必须接入冻结的 current-RGB/Gaussian relation token，并以真实 coarse-front-end 到 PROXD/SDF 的测量残差定义 reliability；完整记录见 `reports/stagea_uncertainty_20260909.md`。

### 8.4 A3 GUSH3R 接口完成、真实几何门槛失败（2026-09-09）

已为 GUSH3R 增加不改变默认推理的 per-frame numerical contact-state export。`per-frame` causal scene mode 能逐帧导出 human/scene Gaussian 的 center、covariance、opacity 与 coarse SMPL-X；`final` mode 只能导出最后一帧累计 scene map，禁止用于 online 接触。PROX MPH112 60帧的前45帧 Sim(3) / 后15帧 held-out 评测显示 GUSH3R coarse SMPL-X median=**19.95 cm**、p95=54.69 cm，远超2cm接触尺度。因此它可作为 Gaussian evidence / degraded-map robustness / renderer diagnostic，但不是当前真实接触监督或 reliability teacher。详情见 `reports/a3_gush3r_prox_feasibility_20260909.md`。

### 8.5 下一连续阶段（2026-09-09）

1. 以已冻结的 GUSH3R per-frame state 和 PROX official SDF 审计 scene Gaussian 的 surface-support / penetration / temporal stability；目标不是用距离阈值判接触，而是量化为何 Gaussian volume 只能成为 relation/reliability evidence。
2. 审计 UniCon3R 或其他可用前馈接触候选的官方代码、输出契约、许可、数据/算力需求；只有人体几何在 held-out PROX 上通过2cm门槛，才可作为 A3 真实 reliability 数据源。
3. 两项均未过门槛时，论文实验保持为 `PROX oracle mechanism + GUSH3R degraded-map robustness`，同时将真实前端接入写为明确未完成风险，不伪造主表。

### 8.6 A4 结果与数据门槛（2026-09-10）

GUSH3R center-to-SDF diagnostic 在失败 held-out coordinate contract 下只有22.4% SDF coverage、有效 |SDF| median=33.9cm；这不是 scene-quality 定量结论，却严格否定“以 Gaussian center/SDF threshold 显式判接触”。现有 PROX near-contact ROI 的 |SDF|<=2cm 比例为99.997%，full L_Leg FPS sample 在 MPH112 也全为正，故不支持 dense contact-vs-noncontact 判别。UniCon3R 官方 repository 当前没有 inference code/weights。后续真实 dense contact head 必须等 RICH/PhySIC 类标注或 UniCon3R release；当前 Gaussian 只通过 AnchorAttention/RGB/SMPL 的 learned interaction field 发挥作用。见 `reports/a4_candidate_and_gaussian_evidence_20260910.md`。

系统级接口、文献定位、阶段产物和风险转向见 `OVERALL_EXECUTION_ROADMAP_20260909.md`。

### 8.7 A5 本地真实稠密监督数据门槛（2026-09-10）

已审计当前工作区和已知 NAS 项目根目录，未发现 RICH、PhySIC 或 EMDB 的可用数据
目录、归档、manifest 或预处理产物。故当前不能合法启动最终 vertex-level binary
contact head；也不能用 PROX 当前 all-positive foot ROI 偷换为稠密监督。A5 的恢复
条件已冻结为：许可明确的 RGB/相机、SMPL-X 拓扑、scene mesh/SDF、同帧 dense vertex
contact（含正负分布和 label source/confidence）、frozen backbone coarse state，以及
scene/sequence/subject 隔离 split。详细资产表、逐帧审计项和恢复顺序见
`reports/a5_dense_supervision_data_gate_20260910.md`。在条件满足前，唯一允许推进的
工作是数据契约/评测接口准备与前馈候选可用性跟踪，禁止继续调 PROX 伪标签、Gaussian
center rule 或 local-only uncertainty。

### 8.8 A1 contact-local 训练接口审计通过（2026-09-10）

`stagea_relation_clean_64x128_v3` 的 12/12 sequence 已只读完成 shape、finite、normal
unit-length 审计：required local relation fields 齐全，非零 normal 最大单位长度误差
`2.38e-7`。旧 v1 的四个面积加权-normal sequence（误差约5.28）及 v2 的零-normal
资产保持禁止训练。A1 因此通过的是“去掉 absolute-world shortcut 的 continuous-proximity
几何接口”，不是 dense binary contact 或真实 RGB/Gaussian front-end 成功；详见
`reports/a1_contact_local_contract_audit_20260910.md`。

## 9. 2026-09-09 E0/E1 实际结果

- **E0 完成（标签契约）**：四个 PROX 序列、共 720 帧的 PROXD -> `PROX_scene_world` SDF 标签通过字段/有限值/法向审计；详情见 `reports/e0_e1_oracle_contact_20260909.md`。旧 E0 目录缺少同目录 teacher NPZ，属于可复现资产缺口，后续 builder 必须同时归档。
- **E1 部分通过（仅法向 clearance）**：可辨识 normal-projection target + GRU + validation-selected contact gate 在互斥 drift-seed test 上将 mean penetration depth 从 18.57 mm 降到 13.95 mm、contact |distance| 从 29.28 mm 降到 24.58 mm；但 foot sliding 从 20.99 mm/s 升到 30.19 mm/s。故不得进入 E2/E3；下一 E1 子任务是 contact hold/tangential stick 与预测-执行闭环。
- **E1 旧随机 root/foot target 已否定**：能获得 contact F1 但应用 residual 使几何变差，原因是 root/foot 分解不可辨识；禁止复用为正式 correction target。
- **边界声明**：E1 训练的是 `PROXD + oracle SDF + 人为注入 drift` 上的机制原型；尚未训练或验证 Human3R/GUSH3R 的真实前端接触模块，尚未做真实 SMPL-X IK/LBS 或真实 Gaussian corrected render。
- **E1b 已立项**：基于接触文献调研，下一子实验从“法向 clearance”扩展为显式 contact-anchor state、切向 foot-lock 和预测—执行 rollout；技术路线与停止条件详见 `CONTACT_AWARE_RELATED_WORK_20260909.md`。
- **E1b 第一轮完成（表示通过，学习控制失败）**：oracle contact anchor lock 将切向滑步 0.385 m/s 降至 0.028 m/s；但 teacher-forced GRU 仍升至 1.180 m/s，不能部署。下一步实现模型自身状态的 causal rollout + bounded tangent action，详见 `reports/e1b_contact_anchor_lock_20260909.md`。
- **E1b 因果 rollout 完成（机制通过）**：模型自有状态回写、切向投影、有界 action 与 soft action gate 构成部署一致闭环，严禁读 teacher previous residual。独立 drift-seed test 的 control-first checkpoint（40 mm、BCE×1）将 sliding 从 0.385 降至 0.218 m/s（-43.4%）、contact F1=0.817；state-first checkpoint（40 mm、BCE×2）给出 F1=0.863、transition F1=0.550、sliding=0.281 m/s。BCE×1.5 位于两者之间（F1=0.849、transition F1=0.416、sliding=0.259）。这是 oracle SDF + injected drift 的 mechanism result；下一步是 E2 point-map proxy 退化、不确定性与 abstention，详见 `OVERALL_EXECUTION_ROADMAP_20260909.md`。
- **E2 第一轮冻结模型失败（有效负结果）**：medium proxy（18° normal、2cm distance noise、1cm bias、20% dropout、2-frame delay）使 control-first test sliding 从 0.385 恶化至 0.463 m/s（+20.3%）。confidence gate 的 0.25 阈值仍为 0.449 m/s；0.75 阈值虽回到 0.385 m/s，但 100% abstain，不是有效修正。单因素 test 定位 normal18/distance 各约 +13% sliding，而 dropout20/delay2 仍安全。
- **E2 几何增强训练完成但不充分**：`medium_aug_action40_contact15` 在同一 medium test 将 sliding 从 0.385 降至 0.276 m/s（-28.4%），但 contact F1=0.722、transition F1=0.142。这说明仅把 normal/distance/noise 作为增强能恢复部分控制，却不能判断何时局部几何不可靠；下一方法必须是显式 normal/distance reliability 与 uncertainty/abstain，而不是继续调 gate 或把该结果写成真实前端提升。
