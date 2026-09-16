# 前馈人体-场景 Gaussian 接触修正方案 Todo

> 方案总文档：`../idea/实时接触修正前馈人体场景Gaussian方案构想.md`
>
> 目标：构建“冻结前馈人体—场景 Gaussian 粗重建 + 当前 RGB 驱动的局部接触估计 + 因果低维修正 + 低维传输”的在线人体—场景一致性层。它不替代前馈重建器，也不逐帧优化场景 Gaussian。
>
> 唯一架构定义：`CONTACT_ESTIMATOR_ARCHITECTURE.md`；唯一活实验协议：`CONTACT_CORRECTION_EXPERIMENT_PLAN.md`；路线图：`OVERALL_EXECUTION_ROADMAP_20260909.md`。下方 2026-07 历史条目只用于追溯，不能覆盖本页顶部的当前任务。

## 当前任务与唯一主线（更新2026-09-16）

- [~] **A5.21 Git源码同步**：本轮整理HUGS及完整contact/RICH源码快照，审计506个源码/配置/文档候选（约4.2MB），补齐元数据TSV/数据依赖/导出入口/方案文档，第三方revision与GUSH3R补丁匹配。数据、权重、认证与生成产物排除；本地提交纳入387个改动文件；主工作区45项通过，干净副本44项通过/1项因缺aria2跳过。HTTPS未登录、原SSH密钥未授权，仓库专用Deploy Key已在本机准备，远端推送等待账号端授权。见根`MIGRATION.md`。
- [x] **A5.19 本批RICH train下载/归档处理完成**：JPG于02:15下载完，04:42整包gzip CRC/SHA256通过并发布`extracted/train`。261429图片/62序列、559.593GB；GT37669帧/1206片、公共包均已完成。10:25核验图片索引无重复、总字节一致；全部37669个contact帧均有至少一个metadata允许相机图片，无缺失，抽查3个文件hash通过。报告`reports/rich_processing_20260916/image_completion_audit.json`。下方下载ETA与运行PID全部为历史状态。
- [x] **A5.20a RICH监督候选清单准备**：`processed/dataset_v1`已正式发布。261429图片头/尺寸/EXIF全量检查（全部identity），1206片SHA256/数值复核通过；train39序列/167710候选样本，内部val23序列/75084候选样本，有效人体帧37585。全样本图片/帧/相机/分片行号独立核对、重复与泄漏检查、产物hash复核和10项回归通过。见`RICH_DATASET.md`及`reports/rich_processing_20260916/dataset_completion_audit.json`。
- [ ] **A5.20b 真实训练输入准备**：先核验逐帧可见性（1290个数值投影抽查中17个无顶点在图像内、38个少于半数，尚不能直接断言标定错误），再接通RICH raw→resize/crop/pad变换及保留真实帧号的纯数值冻结因果前端exporter，生成local geometry/RGB/Gaussian缓存并独立测量几何误差，通过后才训练。当前GUSH3R使用默认K，既有export缺RGB tokens/局部邻域/完整LBS；官方val/test仍需另行补齐，training_ready=false。

> **23:53下载与处理快照**：JPG97.6302%（546.162/559.419GB），余13.257GB；近5分钟0.4485MB/s、14连接，按此约8.2小时（9月16日08:06），覆盖旧1–2点ETA。流式处理正常等待第160.413GB处未完成piece，已暂存73599图片/160.446GB、涉及20序列；gzip不能跳过缺口，整包CRC尚未通过。GT37669帧/1206片已完成；两个controller均存活，无新任务修改。快照`datasets/RICH/processing_logs/joint_status_latest.json`。

> **20:18下载恢复测速**：原实际连接降到10、均速0.266MB/s；已保存断点并触发原controller刷新会话，aria2 PID200625恢复16连接。重连后3分钟测量、排除首分钟均速1.046MB/s（约3.92倍），JPG96.5689%、余19.194GB，按当前速率约5–6小时（9月16日凌晨1–2点），非保证、不含校验解压。流式处理仍在原PID67144，已73599图片，等待前缀160.413GB处下载缺口。证据`datasets/RICH/download_logs/speed_refresh_20260915/summary.json`。

- [~] **A5.18 图片边下载边整理**：19:03已切换到`process_rich_assets.py --stream-jpg`（处理PID67144，下载保持）。从约160.2GB连续完整piece前缀顺序解压，空洞等待补齐；正式归档与全gzip校验通过才原子发布。19:06完成3548图片/6.500GB，并生成路径/SHA256/尺寸/帧/相机/标注人物索引。横竖尺寸并存，473图片的相机不在metadata选用集合，均记录待审计。7项流式+3项归档回归通过。临时图片不进入正式训练；操作见`RICH_PREPARATION.md`。

> **15:23数据处理完成快照**：公共包、body、contact均已全部解压并通过成员CRC/归档SHA256。GT整理完成37669帧、1206分片（10.453GB），全量顶点/面对应通过、索引覆盖无重复遗漏，3片SHA256及数值回读通过。10帧缺contact；另84帧无OBJ颜色，SMPL-X valid=false（有效37585帧），未伪造负标签。处理器仅剩等待JPG下载；图片匹配/内参、split冻结和真实前端缓存/审计仍待完成。

> **14:34最新下载快照**：JPG93.4724%（522.902/559.419GB），剩36.5167GB，近5分钟均值0.4881MB/s，约还需20.8小时（恒速预计9月16日11:21），建议预留约1天；不含解压/校验。body已完整解压校验，contact解压约97.34/109.24GB，GT数值处理34503/37669帧。此快照覆盖下方旧进度/ETA。

- [~] **A5.17 RICH完整性与GT数值整理**：contact/body/公共包已下载；JPG仍约93%、余约38GB，恢复下载PID1464854（有界重认证续传）。公共包CRC/SHA256及解压通过，body/contact完整解压由PID1471505执行，自动等待完整JPG最多36小时。官方metadata、相机/scan/world映射与62序列/64轨迹清单已落地；10帧缺contact记录排除。192帧跨全部轨迹的顶点/三角面核验通过，3个无颜色帧以SMPL-X valid=false保留。全量GT数值分片PID1507008运行，14:00完成1082帧/35片；完成度以`state.json`为准。详见`RICH_PREPARATION.md`，16项相关测试通过。

- [x] **A5.15 训练工程与迁移准备**：新增版本化numeric-cache训练层、可替换组件/配置、独立本机路径、split/hash/audit检查、验证选模/显式test/epoch续训；16项回归及源码副本迁移演练通过。GUSH3R本地5处改动按固定revision保存patch。见`TRAINING.md`、`../MIGRATION.md`与`reports/training_architecture_migration_20260914.md`。未commit/push，未启动真实训练。
- [ ] **A5.16 RICH真实缓存export与执行绑定**：原始归档→同帧/同顶点标签与真实冻结前端cache的exporter、测量审计报告、GUSH3R corrected-SMPL-X到Gaussian binding仍需实现/验证。数值训练框架就绪不解除这些门槛。旧路径28处按目标机实际使用适配，目标GPU环境另验。


> 9月15日状态覆盖下方历史下载快照及所有旧ETA。当前GT准备不代表训练就绪：JPG完整性与逐帧/K检查、开发split冻结、真实冻结前端与独立误差审计仍待执行。官方val/test未下载。

- [ ] **A5.14 下载完成后启动首轮迭代**：CRC/归档与同短序列RGB-body-contact-camera-scan审计→有效scene/sequence/subject开发验证隔离→RICH真实前端误差复测→契约合格的Stage-A基线与RGB/Gaussian消融→通过后Stage-B/LBS。当前train+公共资产够起步，无需继续增加数据集；官方val/test仍待补齐，最终主表不得复用开发数据。下载未结束前不启动训练。

- [ ] **A5.13 双机数据访问**：等待目标机器平台/NAS可达性；优先同一NAS受控访问，其次已完成且校验归档的SSH/rsync。RICH许可禁止未经授权再分发，私有HF/对象存储托管不能自动视为获准。HF免费100GB、PRO含1TB、单文件500GB限制已核实，尚未上传或建云资源。见`reports/rich_sources_20260914.md`。

- [~] **A5.12 官方RICH断点恢复（2026-09-14，覆盖旧进度/ETA）**：原任务9月10–11日长度异常退出；已用既有认证恢复PID1391152、2任务×16连接、非JPG500K。18:52图片463.69/559.42GB（82.89%），body95.19%下载中，contact89.90%排队。近期JPG均值9.91MB/s，按当前速率约剩2.7小时，非保证。新来源DAMON/BEHAVE仅作补充候选，不能直接替换完整RICH契约；PhySIC暂无确认独立数据发布。详见`reports/rich_sources_20260914.md`。

- [~] **A5.11 HF-Mirror找到RICH分卷候选（覆盖A5.9初次结论）**：搜索与公开metadata已连通，全文检索发现`Yong-Hoon/human3r-dataset`，10个RICH分卷共486.0GB。`gated: manual`，需HF账号取得访问权限；内部train/SMPL-X/contact覆盖和速度未验证，原下载未切换。manifest及后续仅RICH下载命令见`reports/rich_mirror_search_20260910.md`。

- [x] **A5.9 RICH镜像检索**：未找到已验证的完整训练图片镜像。OpenDataLab文件数0且明确转官网；SA-HMR/GVHMR/WHAM是辅助或评测数据；DECO Keeper 45.2MB包前缀是测试NPZ而非全量图像。部分平台本机无法访问，不作“全网没有”的断言。报告`reports/rich_mirror_search_20260910.md`，原下载继续。
- [ ] **A5.10 较小官方图片入口核验**：BSTRO约107GB TSV已由官方EXP.md确认有img/hw/label文件；实际归档大小、分辨率、帧号、SMPL-X/相机/场景对应未核验，尚不能替代当前训练数据契约，也未切换下载队列。

### 最终在线契约（必须遵守）

```text
当前 RGB I_t + 模型自身 s_{t-1}
  -> 冻结前馈 human-scene Gaussian backbone
  -> coarse scene/human Gaussians + SMPL-X + RGB/crop tokens
  -> Stage A: RGB–mesh–point local contact encoder
  -> Stage B: causal contact controller
  -> contact / uncertainty / bounded root-foot correction
  -> corrected SMPL-X -> LBS 更新 human Gaussians；scene Gaussians 不动
```

- `s_{t-1}` 只能包含上一帧模型预测的 contact、已施加 action、episode anchor、uncertainty/reset；禁止 teacher state 与未来帧。
- Stage A 是最终接触估计器：脚底 mesh ROI vertex（64--128）+ 每 vertex 的 local scene point/Gaussian patch（128--256）+ 当前 RGB token，输出 vertex contact、continuous proximity、normal/distance reliability。
- Stage B 是小型因果控制器：使用当前 Stage-A ROI token、低维 pose/root dynamics 与 `s_{t-1}`，输出 bounded tangent/root/ankle correction 与 uncertainty/abstain/reset。
- 现有 58D GRU 仅为 Stage-B mechanism prototype。它没有 RGB、vertex-level relation 或真实前端输入，不能称为最终 contact estimator。

### 已冻结的事实

- [x] E0：四个 PROX 序列/720 帧的 SDF 标签与坐标契约通过审计。
- [x] E1b：oracle SDF + injected drift 下的 causal controller mechanism 通过；最佳 control-first `sliding 0.385 -> 0.218 m/s`，但不是真实前端结果。
- [x] E2：medium degraded proxy 下冻结 controller 失败（`0.385 -> 0.463 m/s`）；增强训练部分恢复到 `0.276 m/s`，但 F1=`0.722`、transition F1=`0.142`，不能称为可靠接触估计。
- [x] Human3R -> PROXD 的当前 hold-out Sim(3) median=`7.37 cm`、p95=`45.1 cm`；未通过接触尺度门槛，禁止训练/报告真实前端主表。
- [x] GUSH3R 长序列 scene state 质量不稳；只作为 E4 renderer diagnostic 或 degraded proxy，不能借此替代接触验证。

### 下一批必须按顺序完成

- [~] **A0 数据清单与 split**：已构建并审计 12 个互斥 PROX scene、720 帧的 clean-oracle Stage-A relation `v3`（RGB 引用、SMPL-X foot ROI、local scene patch、vertex contact/proximity），全字段 finite、patch normal 均单位化。v1（非单位 normal）与 v2（零 normal 未过滤）均保留为失败资产，禁止训练。已生成 `normal18`、`distance`、`dropout20` 三个可追溯 relation-level error family；下一步冻结三重隔离 split。
- [x] **A0 严格 robustness split**：已冻结 `experiments/stagea_relation_degrade_20260909/strict_split.json`：train 为 8 scenes 的 clean+normal18，val 为未见的 2 scenes/dropout20，test 为未见的 2 scenes/distance；scene、source sequence、error family 均不跨 split。它不替代后续同分布机制 split。
- [x] **A0.1 否定伪稠密标签**：严格 split 的 geometry-only Stage-A 首训 test F1=0.5965、precision=0.425、recall=1.0，等于 all-positive baseline；审计确认每个 foot ROI 帧的 64 个 binary vertex labels 全相同。v4 fixed-vertex 重建仍如此，说明是 PROX 脚底 2cm hard-contact 标签的粒度限制，不是模型/threshold 问题。当前 PROX 版本改用 vertex continuous proximity + ROI contact；真正 vertex binary contact 需 RICH/PhySIC 类稠密监督。
- [x] **A1 contact-local 坐标**：v3 clean 12/12 sequence 的 `roi_vertex_local/scene_point_local` 与 normal/distance schema、finite 和 unit-normal 审计通过；非零 local normal 最大单位长度误差 `2.38e-7`。旧 v1 面积加权 normal（最大误差约5.28）和 v2 零 normal 资产继续禁止训练。报告：`reports/a1_contact_local_contract_audit_20260910.md`。该通过只适用于 continuous-proximity 几何契约，不解除 A5 的 dense binary contact 数据阻塞。
- [~] **A2 Stage-A baseline**：不含 RGB 的小型 vertex MLP + PointNet 连续 signed proximity / ROI-contact 基线已完成（strict test signed MAE=**1.27 mm**，优于 zero predictor 1.79 mm，见 `reports/stagea_proximity_roi_v2_20260909.md`）。ROI head F1=.600 但 precision=.429/recall=1.0，仍是 all-positive，禁止作为 contact 结论。下一步接 explicit normal/distance reliability + calibration/abstention；之后才接 frozen RGB token。最终输出为 dense supervision 可得时的 vertex contact、continuous proximity、normal/distance reliability。
- [~] **A2.1 reliability / abstention**：启动连续 proximity 的 heteroscedastic Laplace reliability 实验；在相同 strict split 上用 train clean+normal18、val dropout20、test distance 的未见 error family 做 coverage-risk/calibration 检验。门槛：低 uncertainty 保留子集的 MAE 必须低于全量，且高 uncertainty bin 的真实误差应单调更高；否则 reliability head 不接 Stage B。
- [x] **A2.1-v1 implementation failure**：v1 使用 hard `exp(...).clamp`，所有 test scale 饱和为 50 mm，coverage-risk 无效；不作实验结论，保留 artifact 后改为 differentiable bounded log-scale 并重跑 v2。
- [x] **A2.1-v2 strict error-family OOD 负结果**：可导 bounded log-scale 修复后，val 的 uncertainty bins 与误差单调对应，但未见 `distance` error-family test 的 coverage-risk 失效（50% 保留 MAE=.743 mm，高于全量=.698 mm）。因此不能声称 OOD reliability 或接 Stage B。下一项改为所有已声明 corruption 都参与训练、但 scene 仍互斥的 augmented reliability protocol；它只能证明 covered-error robustness。
- [x] **A2.1-v3 covered-error 仍未通过**：all-family augmentation + scene-disjoint test 的全量 MAE=.716 mm，最低 uncertainty 50% 为 .653 mm，但最低 25%=.768 mm 且 uncertainty bins 不单调；没有稳定 calibrated abstention。停止调 local-only reliability head。下一项 A3 是提取 frozen RGB/Gaussian relation token，并以真实 coarse-front-end 与 PROX teacher 的测量误差定义 reliability target；未完成前不接 Stage B。
- [~] **A3 Gaussian evidence interface（GUSH3R 已完成，几何门槛失败）**：已实现 `GUSH3R/infer.py --export_contact_state`，per-frame causal map 在 PROX 60帧导出 250k scene / 10k human GS 的 top-2048 数值属性与 coarse SMPL-X；`final` mode 只给末帧 scene state，禁止用于 online 接触。60帧前45拟合、后15 held-out 的 coarse SMPL-X→PROXD median=**19.95 cm**（p95=54.69 cm），未达2cm contact gate。GUSH3R 仅保留为 degraded Gaussian proxy/renderer interface，不能生成真实 A3 reliability teacher；详见 `reports/a3_gush3r_prox_feasibility_20260909.md`。
- [ ] **A3 真正的前端证据接口（待新 backbone）**：选定能在目标数据上通过 contact-scale human geometry gate 的前馈 backbone，导出 `G_coarse/SMPL-X_coarse`、current-frame visual token、contact-local Gaussian patch（中心、协方差、opacity、view/temporal confidence）；与 PROXD/SDF 计算真实 coarse geometry error。以该 error 而非 proxy family 身份作为 reliability supervision，再复验 calibration/abstention。
- [x] **A4 Gaussian center hard-rule 否定**：GUSH3R per-frame top-2048 GS 在失败 held-out coordinate contract 下对 PROX SDF 仅22.4% coverage、有效 |SDF| median=33.9cm；严禁将 Gaussian center/SDF threshold 作为接触。Gaussian 只作 learned relation/reliability evidence，见 `reports/a4_candidate_and_gaussian_evidence_20260910.md`。
- [x] **A4 UniCon3R 候选审计**：官方仓库已拉取但 README 明确 code/checkpoint/inference “will be released soon”；仅有网页 demo 的每帧 10475-vertex `contactIndices`，无可运行模型/监督契约，当前不能复现或作 PROX baseline。
- [~] **A5 dense supervision data gate**：已完成本地工作区与已知 NAS 根目录的只读资产审计：未发现 RICH/PhySIC/EMDB 数据、归档或 manifest，故不能启动最终 dense contact head；审计报告为 `reports/a5_dense_supervision_data_gate_20260910.md`。数据到位且许可明确后，先建立 RGB/camera、SMPL-X topology、scene mesh/SDF、同帧 dense vertex labels（source/confidence）、coarse front-end state、scene/sequence/subject split 的 manifest，并做正负分布/坐标/topology audit；之后才重启 Gaussian evidence vs point/mesh relation 和 Stage-A contact head。禁止继续使用当前 PROX near-contact ROI 的伪 dense binary 标签。
- [~] **A5.2 RICH priority adapter**：官方站点复核确认 RICH 提供多视角 RGB、GT 3D body/body+scene scan 与 accurate vertex-level body contact；须注册并接受单用户非商业科研许可，不能擅自共享/再分发。获得授权数据后先以 1 个短序列检查实际 body topology、camera/world、scene surface、label frame alignment 和正负比例，再扩展训练；RICH 可解 dense supervision 缺口，但不替代真实 Gaussian backbone 的 2 cm gate。
- [ ] **A5.3 RICH staged acquisition（2026-09-10修订）**：公共scan/calibration/world transform + train bodies/contact约39.2GB先到位，接同序列train JPG；有序列级链接则先1–2条序列，否则按官方实际拆包获取。56GB sample仅在字段完整时作可选smoke，不强制先下；随后扩train/val（含公共资产约806.7GB），test留作冻结评测（再约280.3GB）。全量JPG主线约1087GB，传输与解压体积尚待实物核验。不启动原始图片全量脚本；BSTRO公平对比前补官方评测协议/索引。详见 `reports/rich_fast_download_plan_20260910.md`。
- [x] **A5.4 RICH resumable downloader**：既有 `scripts/download_rich_parallel.sh` 使用wget文件级并行与断点续传，原版仅做语法检查，不能加速单个文件。`.netrc`仅适合对应HTTP认证，未证明兼容官网POST/cookie流程。当前有wget/tmux/conda、无aria2c；已准备独立aria2安装与2文件×4连接命令，尚未安装或实测。NAS约32TB可用，系统盘约139GB；数据落NAS。
- [~] **A5.5 RICH authorized-link preflight**：5个真实URL已到位，公开坐标ZIP已下载（4304bytes、CRC通过、7个场景JSON）；scan链接未认证返回HTTP200/6311字节登录HTML，表单要求POST username/password/commit。新增 `scripts/acquire_rich.py --configure`：在服务器交互配置私有凭据，验证后自动启动2个wget后台任务，保留断点与状态。POST/续传/HTML拒绝/不支持Range保留部分文件的本地HTTP测试通过；当前待用户配置认证，受保护4个归档、实际Range与吞吐未验证。aria2方案仅在取得可用GET/cookie链接后适用。
- [~] **A5.6 RICH实际下载与加速（覆盖A5.5旧状态）**：用户凭据已验证，修复urllib遗漏302登录Cookie的问题；4保护资源均通过206 Range/格式预检。已安装aria2并接入私有Cookie GET，每文件8连接、2文件并行，PID929174。scan/body下载中、contact/JPG排队；近期平均合计1.387MB/s，约为wget 75.6KB/s的18.3倍。已下载坐标包CRC通过，其他归档未完成。完整回归4项通过，aria2稀疏文件必须保留控制文件；用脚本`--status`看实际进度。真实训练JPG约559.4GB，4保护包共约598.3GB。
- [~] **A5.7 JPG优先与ETA（当前覆盖A5.6）**：独占8/16连接短测后启用16连接，JPG提前与body并行，其他资产限速500K，PID1011347；contact保留断点排队。图片近期0.9–1.2MiB/s，18:21最新平均1MiB/s，余量恒速约6.16天（9/16晚），建议按5–7天规划。扫描包64成员CRC已通过；JPG首部tar目录已识别Pavallion_006_plankjack/cam_00，仅结构读取。中转需用户已有节点配置后实测两段吞吐。状态展示已避免把前一controller的暂停任务误报为正在下载。
- [~] **A5.8 本机直连复测完成、下载继续（覆盖A5.7旧ETA）**：无云账号/中转地址，RICH官方凭据仍有效。16/8连接独占JPG各180秒、舍弃首分钟后均值0.442/0.152MB/s，测试前并行0.315MB/s；未找到稳定多MB/s提速。保留全部断点恢复2文件×16连接、非JPG500K，PID1305167；JPG/body下载中、contact排队。20:54恢复后JPG0.288MB/s、body0.325MB/s；JPG剩约554GB按当前恒速约22.3天，5–7天旧ETA失效，快照`jpg_local_eta_20260910.json`。JPG单连接403、其余15条持续下载。DNS/ALPN未发现IPv6或HTTP/2替代路径，未改全局TCP。报告`jpg_local_benchmark.json`，脚本`benchmark_rich_local.py`。
- [x] **A5.1 UniCon3R release recheck**：官方 snapshot `9b248489...` 的 README 仍明确 code/checkpoints/inference “will be released soon”，顶层仍仅项目页资产；远程 refs 未返回可判定的新发布信息。它保持为竞品/架构参考，不能成为当前可运行 baseline 或 A3 backbone。
- [ ] **A3 uncertainty**：训练/校准 reliability 与 uncertainty/abstain head；固定 validation threshold，报告 calibration、coverage-risk、sliding/penetration-vs-abstention。
- [ ] **B1 Stage-A+B rollout**：将 ROI pooled token 接到既有 causal controller；只回写 model-owned state，比较 no-correction、frozen E1b、geometry-augmented E1b 与 Stage-A+B。
- [ ] **C1 真实前端门槛**：完成 Human3R-on-PROX 的 camera/root/local-pose 分解、foot ROI reprojection 与 camera-local contact contract。未通过门槛只产出 diagnostic，禁止 C 层训练。
- [ ] **D1 Gaussian 接入**：仅在 C1 与 Stage-A+B 达标后，接 corrected SMPL-X -> LBS human Gaussian；scene Gaussian 始终冻结。GUSH3R 只在局部 scene proxy 合格后接入。

### 停止与表述规则

- [ ] 不把 PROX oracle / synthetic drift 的 `mechanism result` 写成 Human3R/GUSH3R 在线结果。
- [ ] 不继续用仅四个短序列、仅换随机 drift seed 的结果声称跨场景泛化。
- [ ] 不继续围绕单一 confidence gate 调阈值；E2 已证明这不能在非全 abstain 时保证安全。
- [ ] 不移动或重新优化 scene Gaussian；只允许 low-D SMPL-X correction 经 LBS 更新 human Gaussians。

---

## 历史工程记录（2026-07--08；以顶部当前主线为准）

### 2026-08-26 自动研究更新（以 `RESEARCH_PROTOCOL_20260826.md` 为准）

### 2026-09-09 E1b causal rollout / 长时控制器更新

- [x] 明确 E1b oracle anchor lock 是表示与执行层机制证据（sliding 0.385 -> 0.028 m/s），teacher-forced GRU 则不可部署（约 1.180 m/s）。
- [x] 实现 `anchorlock_rollout.py`：每帧仅回写模型自身上一 contact/action、切向投影、bounded action；评测不读取 teacher previous state。
- [x] 实现 `train_e1b_rollout.py` / `evaluate_e1b_rollout.py`，并加入 checkpoint + optimizer + history 的 `--resume`；静态编译与 60 帧 hard-gate tensor smoke 通过。
- [x] 实现 `autoresearch_controller.py` 与三项预注册候选；状态 JSON、事件、独立日志、metric gate、时间预算和 shared-GPU idle wait 均已定义。
- [x] 完成 E1b rollout 三候选（40 mm、20 mm、20 mm + stick=5）的实际训练/独立 test；只以 `test_metrics.json` 判定，不以训练损失或 oracle 数字判定。

### 2026-09-09 系统路线收敛更新

- [x] 完成 E1b 三组初始候选以及 40 mm contact-BCE `1/1.5/2` 的软门控 test；冻结为 mechanism-only Pareto，不混入 Human3R/GUSH3R 表。
- [x] 核验 Splat-SAP、HumanGS、Hand-4DGS、PointSplat、GaussianLens 的系统接口并形成总体路线图：`OVERALL_EXECUTION_ROADMAP_20260909.md`。
- [ ] P2/E2：实现 PROX teacher -> degraded point-map proxy 构造（normal rotation、distance bias/noise、dropout/occlusion、latency），每帧保存 severity/confidence/source。
- [ ] P2/E2：实现 confidence-rule abstain、coverage-risk 与 sliding/penetration-vs-abstention 评测；阈值仅在 validation 选择。
- [ ] 重构最终 contact estimator：当前 RGB token + coarse human/scene Gaussian proxy -> mesh ROI / point patch local contact encoder -> causal controller；详见 `CONTACT_ESTIMATOR_ARCHITECTURE.md`。
- [ ] 将 Stage B 输入转换到 contact-local invariant frame，移除绝对 world-coordinate 记忆捷径；采用 source-sequence、scene、error-family 三重隔离，而不是仅换 drift seed。
- [ ] Stage A：构建脚底 ROI vertex (64--128) 与 scene point/Gaussian patch (128--256) 的 relation dataset，监督 vertex contact、continuous proximity、normal/distance reliability。
- [ ] P3：完成 Human3R-on-PROX camera/root/local-pose error decomposition 与 foot ROI reprojection；未通过接触尺度门槛不得训练真实前端接触网络。
- [ ] P4/P5：仅在外部代码/许可可用后，对 Splat-SAP scene proxy、HumanGS canonical human branch 做 10--60 帧接口 smoke；禁止未验证替换当前 baseline。

- [x] PROX 真实资产已到位并解压：54 个 PROXD 序列、RGB/Depth recordings、12 个 scene meshes、官方 SDF、cam2world 与 body segments。
- [x] 新增 `scripts/build_prox_geometry_labels.py`：PROXD SMPL-X + cam2world + official SDF 构造带来源的脚部几何教师标签；不伪造 pose/root residual 标签。
- [x] 首个 180 帧 CPU 坐标/标签审计：`BasementSittingBooth_00142_01` 的脚部 SDF 有效率 100%；已发现并修复最近顶点身份切换造成的速度标签偏差。
- [ ] 在至少四个不同 PROX 场景完成 E0 数据审计、可视化抽检和冻结 split；通过前不得训练/比较 GRU/TCN。
- [ ] 构造有已知 root/foot drift 的 PROX synthetic-error 数据，提供真实 correction residual 监督；禁止用 clean PROXD 生成全零 residual 训练网络。
- [x] 实现 `make_prox_synthetic_drift.py` 并以 MPH112 60 帧验证输出为 58D input + 非零且已知的 root/foot residual；待跨序列 E0 后建立 sequence-safe B 层 split。
- [ ] GPU 空闲后运行 Human3R-on-PROX 并建立 predicted-to-PROXD 对齐，才进入真实前馈 C 层实验。
- [x] Human3R-on-PROX MPH112 60 帧导出与 held-out Sim(3) 已运行；hold-out vertex median=7.37 cm、p95=45.1 cm，未通过接触几何验收。
- [ ] 将 Human3R-to-PROXD 误差拆分为 root/camera/local-pose，并完成重投影与脚 anchor-SDF 可视化；通过前 C 层训练保持禁止。
- [x] 建立四个互斥 PROX scene 的 B 层 synthetic-drift 2/1/1 split，并完成 GRU 30 epoch 与独立 test；结果不通过泛化验收（F1=0.218、transition F1=0、distance MAE=9.07 cm、penetration ratio=50%）。
- [x] 审计 B 层当前 scene split 的接触先验：train/val/test 正例率 84.58%/53.33%/12.50%，Rule/GRU F1 接近 all-positive；`diagnostics/prox_synth_aug_contact_separability.json` 已归档。此 split 保留为 domain-generalization 压力测试，不能单独证明机制不可学习。
- [x] contact-only GRU 30 epochs 完成：best val epoch=18，test precision=0.12264、recall=1.0、F1=0.21849、transition F1=0；与 multi-task GRU 相同，否定“仅共享多任务负迁移”为首要原因。`runs/prox_synth_aug_gru_contactonly/`。
- [ ] 构建第二个预注册 B 层 split：同一 scene/动作跨 train/val/test 使用不重叠 drift seed（无 sequence/seed 泄漏），用于机制恢复；仍保留现有 2/1/1 scene split 作为独立 domain-generalization 报告。
- [ ] 在两个 split 上执行同一训练预算的 Rule、contact-only、multi-head/两阶段 GRU；报告正类比例、per-scene F1、PR-AUC、threshold calibration 与 correction residual，禁止仅报 pooled F1。
- [ ] TCN 已完成首轮且回归发散（test distance MAE 1.036 m、root RMSE 3.69 m），在数据协议修正前不继续扩模型/调参。
- [ ] GUSH3R corrected-SMPL-X/LBS renderer 与 contact-ROI rendering 是 D 层正式实验，不能由 Human3R 指标替代。

### 2026-07-29 脚本实现状态

- [x] 建立独立 `contact_streaming/` package 和统一 58D 几何/时序特征、38D 网络输出 schema。
- [x] 实现 robust Sim(3)、translation fallback、SMPL-X 左右脚 anchor 和局部加权 PCA 平面 proxy。
- [x] 实现规则 contact hysteresis、Tracking/Uncertain/Reset、root/foot residual 和 penetration projection target。
- [x] 实现 sequence-safe 数据集、GT 标签合并、GRU/causal TCN、损失、训练、验证和因果推理脚本。
- [x] 实现通用 Gaussian LBS adapter、62-byte contact packet 和延迟/丢包模拟。
- [x] 用合成数据跑通 GRU/TCN 的训练→验证→推理→packet 全链路；4 个核心单元测试通过。
- [x] 用 `frameinput_goodmornin_10f` 跑通真实 Human3R FrameInput→SMPL-X anchor→surface proxy→规则 baseline。
- [x] 对 NeuMan `lab` 103 帧完成真实 Human3R FrameInput、HUGS SMPL 转换、SMPL vertex/camera Sim(3) 对照、对齐 point-map proxy 与诊断报告（`reports/neuman_lab_p0_20260804.md`）。
- [x] 新增 NeuMan P0 correspondence builder：同帧同拓扑 SMPL 顶点拟合 Human3R→NeuMan Sim(3)，输出仅供训练的对应点与 frame-level hold-out 验证报告。
- [x] 修正 `prepare_sequence.py`：参考场景 Sim(3) 同时作用于人体 anchor 与 Human3R point map，禁止混用两个坐标系拟合 surface proxy。
- [x] 重新生成 `prepared/neuman_lab_p0_scaledproxy_support.npz`，其中保存每帧每脚 `surface_support`；103×2 字段完整、surface distance 全 finite，核心单元测试 4/4 通过。
- [ ] GoodMornin 仍只有低置信度 translation fallback；NeuMan 已提供真实 correspondence runner，但其 proxy 接受率不足，尚未通过 P0 几何验收。
- [ ] 当前 foot residual 是几何 target，尚未接入可微 SMPL-X IK 求解器。
- [ ] 当前 Gaussian adapter 已提供 LBS/NPZ 接口，尚未完成 GUSH3R decoder 内 corrected SMPL-X 重算和正式 renderer 对照。
- [ ] 尚未准备 BEDLAM/PROX 数据，也未进行真实训练和论文指标实验。

### 主实验定位

- Human3R 是网格/点图接触预验证平台，不是最终 Gaussian 主实验平台。
- GUSH3R 是正式主基线和最终 Gaussian 实验平台。
- 主表必须比较 `原始 GUSH3R` 与 `GUSH3R + 我们的接触修正层`。
- 接触模块只修正低维 SMPL-X 状态，并通过 LBS 更新人体 Gaussian；scene Gaussian 保持不变。

### 已完成

- 前馈仓库和主权重已准备：Human3R 896L、GUSH3R merged checkpoint、Multi-HMR、SMPL/SMPL-X 及 supplementary 文件。
- Human3R 单帧 GPU smoke test 已完成，结果已归档到 `baseline/results/human3r_smoke_gpu/`。
- GUSH3R 单帧 GPU 前向和 Gaussian 渲染已完成，结果已归档到 `baseline/results/gush3r_smoke/`。
- GUSH3R CUDA RoPE2D、xFormers 和 DINOv2 本地 cache 已配置。
- 已定位并修复 GUSH3R sampling cache 路径错误；已兼容当前 rasterizer 的返回接口。
- 已测试 `bg_mask_dilation=0`，人体周边白边明显减弱；对照结果在 `baseline/results/gush3r_bg_dilation0/`。
- 已实现 `export_frame_input.py`，完成 P0.1/P0.2 的标准 FrameInput 导出。
- 已用 `GoodMornin1.mp4` 前 10 帧完成多帧验证：23.98 FPS 输入、10 帧输出、平均 0.574 秒/帧，结果在 `datasets/frameinput_goodmornin_10f/`。

### 尚未完成

- 当前已有第一版标准化 `FrameInput`，但仍是 `raw_unaligned`，尚未完成坐标对齐和接触字段。
- 尚未完成 Human3R/GUSH3R 坐标统一、Sim(3) 对齐、脚 anchor、局部场景 surface proxy。
- 尚未实现规则 contact、脚部投影/IK、penetration 指标和 corrected Gaussian/mesh 渲染。
- 尚未训练 GRU/TCN，也尚未做低维 packet 和客户端实时传输。

### 更新后的模块接口

第一阶段 `K=2`，只处理左右脚。输入固定为 SMPL-X pose/root、anchor 位置/速度/朝向、局部 surface 点/法向/置信度、人体置信度、上一帧 contact 和 residual；正式版本可选加入 image feature，第一版几何 baseline 不依赖 image feature。

输出固定为 contact logits、contact point、surface distance/normal、pose residual、`root_residual[6]`、penetration risk、uncertainty 和 reset flag。`pose_residual` 只覆盖 root、ankle 和足部相关关节，不预测完整 SMPL-X pose；连续输出控制在约 20–40 个变量。

### 下一步（当前唯一主线）

1. 选一个短序列确认第一版 FrameInput 格式和时间戳稳定。
2. 明确 Human3R camera/world 与 SMPL-X 的坐标轴、单位和手性，完成 robust Sim(3) 对齐并输出 residual/confidence。
3. 从 SMPL-X 得到左右脚底 anchor，使用 depth/point map 拟合局部地面平面，输出 signed distance 和法向。
4. 在此基础上实现规则 contact baseline，再进行 root vertical correction 和脚部 IK。
5. 用 mesh/点图完成指标闭环后，再训练 GRU/TCN，最后接入 GUSH3R Gaussian。

完成 P0/P1 闭环前，暂不进入 GRU/TCN 训练、GUSH3R 接触接入或复杂传输协议。

## 总体路线

```text
Human3R 输出（网格/点图预验证）
  -> 坐标统一与对齐
  -> 脚 anchor + 局部场景 surface proxy
  -> 规则接触判断与脚部投影
  -> mesh 接触指标与规则 baseline
  -> causal GRU/TCN
  -> GUSH3R 主实验：scene/human Gaussian
  -> corrected SMPL-X -> human Gaussian LBS
  -> 低维 contact packet 与实时传输
```

## 当前里程碑

- [ ] M0：完成 Human3R 单序列离线输出和最小数据格式
- [ ] M1：完成 `Human3R 输出 → 脚接触 proxy → 规则修正 → 渲染 → 指标` 闭环
- [ ] M2：冻结 Human3R/GUSH3R，训练并验证 causal GRU/TCN
- [ ] M3：接入 GUSH3R human/scene Gaussian 主实验
- [ ] M4：完成低维 packet、客户端 LBS 和实时性能验证

---

## P0：Human3R 输出、坐标统一和局部几何

### P0.1 核对前馈仓库和推理入口（已完成）

- [x] 阅读 `Human3R/README.md`、推理脚本和 checkpoint 加载方式
- [x] 阅读 `GUSH3R/README.md`，确认它继承 Human3R 的哪些输出
- [x] 确认当前环境、依赖、GPU、checkpoint 路径和最小可运行命令
- [x] 用一个短 RGB 序列分别跑通 Human3R 推理
- [x] 记录推理速度、显存和输出帧数

### P0.2 固定 `FrameInput` 数据格式（已完成第一版）

每帧至少保存：

- [x] camera intrinsics/extrinsics
- [x] SMPL/SMPL-X pose、root translation、betas
- [x] 人体 mask 或人体置信度
- [x] depth、confidence、point map
- [x] 原始图像尺寸、帧号、时间戳和坐标系说明
- [x] 版本号、数据源和 checkpoint 信息

建议统一为：

```text
FrameInput[t] = {
    image_path,
    camera,
    smplx_pose,
    root_translation,
    anchor_position,
    anchor_orientation,
    depth,
    point_map,
    confidence,
    coordinate_convention,
}
```

### P0.3 实现坐标对齐模块

- [ ] 明确 Human3R camera/world、SMPL-X 和 renderer 的轴向、手性、单位
- [ ] 实现 convention conversion
- [x] 实现 robust Sim(3) 对齐接口（真实对应点尚待验证）
- [ ] 用场景点云/人体 anchor/地面候选平面估计 scale、rotation、translation
- [x] 为 NeuMan 实现同帧 SMPL 顶点对应的 scale/rotation/translation 估计与独立 hold-out error 报告；尚待以实际 Human3R-on-NeuMan 输出运行。
- [ ] 检查 SMPL-X 投影与人体 mask 的重叠率
- [x] 输出每帧 `alignment_residual` 和 `alignment_confidence`
- [x] 对齐失败时禁止强行 IK，保留原始姿态并标记 uncertain/reset

### P0.4 实现脚部 anchor

- [x] 定义左右脚底/脚踝 joint anchor（脚部顶点 ROI 尚待补充）
- [x] 从 SMPL-X forward kinematics 得到世界坐标
- [x] 计算 anchor 速度和脚底朝向（加速度尚待指标接入）
- [ ] 与人体 mask、depth 和投影位置做可视化核验
- [ ] 保存左右脚 anchor 轨迹和置信度

### P0.5 实现局部场景 proxy

按由简到难的顺序实现并分别评估：

- [x] V0：脚 anchor 邻域深度/point map 的局部平面拟合
- [x] V1：KNN 局部点云 + 加权 PCA，输出 signed distance 和法向（MLS 尚待补充）
- [ ] V2：过滤 opacity、scale、viewing confidence 后的 Gaussian-aware proxy
- [ ] 不直接将 Gaussian center 当作真实表面
- [x] 保存局部法向、距离、置信度和 support（局部点/版本元数据尚待补充）

### P0 验收

- [x] 能对一个完整 NeuMan 序列逐帧导出标准 `FrameInput`（lab，103 帧）
- [ ] 左右脚 anchor、局部平面和坐标系可视化正确
- [x] 对齐残差和局部表面置信度可统计；当前统计显示 proxy 不足，详见 `reports/neuman_lab_p0_20260804.md`
- [ ] 固定同一输入可重复推理，结果和版本信息可追溯

---

## P1：规则接触 baseline 和指标闭环

### P1.1 规则接触估计

- [x] 用脚到 surface 的距离、脚速度和置信度判断接触（法向兼容性尚待加入规则）
- [x] 输出 `contact_state`、`contact_probability`、`surface_distance`
- [x] 加入进入/保持/退出 hysteresis
- [x] 实现 Tracking / Uncertain / Reset 三种状态
- [x] 阈值集中在 `RuleConfig`（proxy 版本元数据尚待补充）

### P1.2 低维脚部修正

- [x] 先实现 root translation 修正
- [ ] 再实现脚踝/足部相关关节的 IK 修正
- [x] 加入 root residual clamp（pose joint-limit clamp 尚待 IK 实现）
- [ ] 实现脚底贴面和 penetration projection
- [x] 高不确定性时回退原始姿态
- [ ] 通过 SMPL-X forward kinematics 生成 corrected pose

### P1.3 渲染和指标

- [ ] 实现 corrected mesh 的可视化
- [ ] 接入已有 Gaussian deformation/LBS；若暂时不能接入，先完成 mesh 闭环
- [ ] 固定同一输入比较原始前馈、规则 contact、规则 contact+IK
- [x] 计算 foot-ground/surface distance
- [x] 计算 contact precision/recall/F1/transition F1
- [x] 计算 foot sliding
- [x] 计算 penetration ratio/depth
- [ ] 计算 human PSNR/SSIM/LPIPS 和 contact ROI 指标
- [ ] 记录每帧接触模块延迟和端到端 FPS

### P1 验收

- [ ] 完成第一里程碑闭环
- [ ] 规则方法相对原始前馈在至少一个真实序列上减少 foot sliding 和 penetration
- [ ] 指标脚本能够批量运行并输出 CSV/JSON
- [ ] 失败帧可定位到坐标、proxy、接触判断或 IK 阶段

---

## P2：因果接触网络

### P2.1 数据和标签

- [ ] 先确认 BEDLAM 6fps RGB/GT 可用性和本地目录结构
- [ ] 准备 BEDLAM + PROX 的脚接触训练/验证数据
- [ ] NeuMan 作为真实 Gaussian 管线验证集，不把不完整标签当作无条件 GT
- [ ] 暂不下载 BEHAVE/GRAB/ARCTIC/HOSNeRF；仅在手-物体或复杂交互扩展时加入
- [ ] 保存 `label_source`、`label_confidence` 和伪标签规则
- [x] 建立按序列划分的 train/val/test，避免相邻帧泄漏

### P2.2 网络接口

- [x] 第一版只用几何和时序输入，不接 image feature
- [x] 输入窗口支持过去 5–8 帧
- [x] 实现轻量 causal GRU baseline
- [x] 实现 causal TCN 对照版本
- [x] 输出左右脚 contact logits、surface geometry、root/foot residual、uncertainty
- [x] 固定初始损失：state + surface + velocity + penetration + pose + temporal + uncertainty
- [x] 输出连续变量控制为 38 个

### P2.3 训练

- [x] 训练脚本只加载离线特征，Human3R/GUSH3R 保持冻结
- [x] 使用 AdamW、混合精度、梯度裁剪和固定随机种子
- [ ] 初始损失：state + surface + velocity + penetration + pose + temporal + uncertainty
- [ ] 以 contact F1、foot sliding、penetration 和 ROI LPIPS 联合选择 checkpoint
- [ ] 检查网络是否只学会视觉贴合而没有真实接触
- [ ] 与规则 baseline、EMA baseline 做正式真实数据对照（规则评估脚本已实现）

### P2 验收

- [ ] 网络在线推理达到至少 30 FPS（不含前馈 backbone 时单独统计）
- [ ] 相对规则 baseline 接触 F1 或 temporal consistency 提升
- [ ] 相对原始前馈 foot sliding 和 contact jitter 下降目标 ≥30%
- [ ] 不确定性高时能正确回退，不产生大幅姿态跳变

---

## P3：接入人体 Gaussian 和 contact ROI 渲染

- [ ] 明确 Human3R 输出到 canonical human Gaussian/LBS 的转换接口
- [ ] 先用已有 HUGS human Gaussian 做 corrected SMPL/LBS 验证
- [ ] 只更新人体 Gaussian 的低维运动位置，不重新优化 scene Gaussian 的颜色/尺度/opacity
- [ ] 实现原始 pose 与 corrected pose 的同步渲染
- [ ] 评估全图和脚部 contact ROI 的 PSNR/SSIM/LPIPS
- [ ] 验证 correction 不引入明显人体形变或外观漂移
- [ ] 完成客户端侧 LBS + local Gaussian update 的最小 demo
- [ ] 统计 contact ROI Gaussian penetration 和人体 Gaussian temporal flicker

### P3 验收

- [ ] Contact ROI 视觉质量不下降或提升
- [ ] 全图 PSNR 下降不超过 0.1 dB
- [ ] corrected human Gaussian 可以稳定播放完整序列

---

## P4：GUSH3R 接入

- [ ] 固定 GUSH3R 的 scene Gaussian、human Gaussian、SMPL-X 和相机输出接口
- [ ] 复用 P0 的坐标对齐、脚 anchor、局部 proxy 和接触模块
- [ ] 对比 Human3R mesh/原型和 GUSH3R Gaussian 两条路线
- [ ] 场景 Gaussian 保持不变，只修正人体低维状态和 LBS deformation
- [ ] 将 GUSH3R 作为正式主实验：原始 GUSH3R、规则/EMA、GRU/TCN 无 IK、GRU/TCN+IK+penetration
- [ ] 建立与 HUGS 离线优化的对照主表
- [ ] 统计端到端 FPS、显存、延迟和失败率

### P4 验收

- [ ] GUSH3R 版本复现同样的接触质量提升
- [ ] 明确提升来自接触网络，而不是单纯更好的 scene proxy
- [ ] 高不确定性帧可回退原始 GUSH3R 输出

---

## P5：低维传输和实时系统

- [x] 定义 contact packet：状态、root/foot residual、置信度、序号、时间戳、reset/keyframe 标志
- [ ] 实现人体优先、场景缓存和低维增量传输
- [x] 测量 packet 大小、bitrate、模拟延迟和丢包恢复
- [x] 测量 contact packet bitrate、late ratio 和 recovery frames（first-contact visible time 尚待 renderer）
- [x] packet 支持 keyframe、reset 和 Uncertain flags
- [ ] 在客户端完成 LBS、Gaussian update 和 renderer 播放
- [ ] 与现有 HUGS transmission system 的 Savgol/差分方案做 rate-distortion 对比

### P5 验收

- [ ] 单帧 contact packet <10 KB
- [ ] 客户端 LBS/修正/渲染达到 30 FPS
- [ ] 丢包、乱序和重新连接不会导致人体永久卡死或跳变

---

## P6：扩展任务（第一版完成后）

- [ ] 手-物体接触
- [ ] 臀部-椅面、背部-墙面和坐姿
- [ ] 多接触点与全身 penetration
- [ ] image feature / human crop 融合
- [ ] GUSH3R human decoder adapter 轻量联合微调
- [ ] 更复杂的物体 proxy 和交互数据集

## 暂不做

- [ ] 不逐帧优化全部 Gaussian
- [ ] 不一开始联合微调 Human3R/GUSH3R backbone
- [ ] 不一开始处理手-物体、坐姿和完整人体碰撞
- [ ] 不在闭环之前投入复杂网络协议和大规模模型训练

## 2026-09-09：活实验协议同步

- [x] 建立 `CONTACT_CORRECTION_EXPERIMENT_PLAN.md`，作为接触修正当前阶段、数据/标签、split、指标、停止条件与实验记录的唯一活协议。
- [x] E0：PROX SDF/PROXD 的四序列标签契约审计通过；旧标签缺 teacher NPZ 的资产缺口已记录，后续重建必须一并归档。
- [x] E1（法向 clearance 子任务）：以 PROX SDF + clean PROXD 建立同 scene/动作、drift-seed 互斥的机制 split，并固定 zero/rule/oracle/GRU/TCN 对照；结果为部分通过，详见 `reports/e0_e1_oracle_contact_20260909.md`。
- [ ] E1（接触保持子任务）：增加可辨识的 tangential contact anchor/hold target、预测-执行闭环和真实 velocity/temporal loss；通过条件为 penetration、contact distance、foot sliding 同时不恶化。
- [x] E1b oracle anchor-lock：切向 drift + episode anchor 的表示/执行上界通过，但 GRU teacher-forced 控制失败；报告 `reports/e1b_contact_anchor_lock_20260909.md`。
- [ ] E1b rollout controller：模型自身维护 previous contact/applied residual/anchor state；输出 bounded tangent action，训练 multi-step rollout、stick/release、slew-rate loss；严禁使用 teacher residual 作为测试时状态。
- [x] 调研并归档 mesh/point-cloud/contact-aware 前馈/半前馈工作；`CONTACT_AWARE_RELATED_WORK_20260909.md` 给出 GraphiContact、CRISP、PhySIC、POSA、LEXIS、StackFLOW、IK/physics 等路线与 E1b 可迁移设计。
- [ ] E2：建立局部 surface 的噪声/缺失/错位/延迟退化协议，并评测 uncertainty calibration、错误修正率与安全回退。
- [ ] E3：完成 Human3R -> PROX 的 camera-local 对齐/重投影门槛；此前不得把 Human3R 输出用于真实接触监督或正式主表。
- [ ] E4：仅在 E1--E3 通过后接入 GUSH3R 或替代 scene-state；报告接触 ROI 和在线稳定性，不将 renderer diagnostic 误作接触主表。

## 第一版最终验收标准

- [ ] 接触模块 ≥30 FPS
- [ ] foot sliding、contact jitter 相对原始前馈下降 ≥30%
- [ ] penetration ratio/depth 明显下降
- [ ] contact ROI PSNR 不下降，全图 PSNR 下降不超过 0.1 dB
- [ ] contact ROI Gaussian penetration 和人体 Gaussian temporal flicker 有统计结果
- [ ] packet <10 KB
- [ ] 记录 packet late ratio、recovery time 和 first-contact visible time
- [ ] 客户端 LBS + 修正 + 渲染 ≥30 FPS
- [ ] 高不确定性帧可自动回退或请求 keyframe
