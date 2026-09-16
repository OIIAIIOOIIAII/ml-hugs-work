# RICH下载进度与其他数据来源（2026-09-14）

## 当前结论

官方RICH续传优先。原任务已经在9月10–11日因aria2响应长度异常退出，本次18:47北京时间使用原有私有认证与全部断点成功恢复，controller PID1391152，2个任务并行、每文件16连接、非图片500K限速。没有重新从零下载，也没有启动HF或其他数据集大包。

18:52复核：JPG已保存完整分块至少463.69GB/559.42GB（82.89%），body至少8.71GB/9.15GB（95.19%），contact至少25.92GB/28.83GB（89.90%，等待body完成后续传）；公共scan/calibration 0.904GB及坐标包已完成，历史CRC证据仍保留。

恢复后舍弃启动段的最新12条约10秒摘要：JPG平均9.91MB/s（9.45MiB/s），范围7.24–12.58MB/s；另用控制文件完整分块增量得到60.13秒平均10.17MB/s，相互吻合。按剩余约95.7GB与近期速度恒速估算约2.7小时；只反映当前几分钟窗口，不能保证完成时间，旧22.3天ETA已不适用于当前速率。

状态读取来自aria2日志与控制文件位图；位图统计只计完整piece，是下载字节下界，控制文件约每60秒保存。没有把预分配/稀疏文件的逻辑长度或磁盘占用当进度。快照：`datasets/RICH/download_logs/resume_20260914_before.json`、`resume_20260914_progress_1.json`、`resume_20260914_latest.json`。控制格式核对aria2 1.37.0官方源码`DefaultBtProgressInfoFile.cc`。


最后复核（2026-09-14T18:56:45.476312+08:00）：JPG完整piece至少466.17GB（83.33%），body 96.39%，contact89.90%排队；最新12摘要图片均值8.20MB/s。动态快照为`resume_20260914_final_snapshot.json`。

## RICH本身的其他入口

- [Human3R HF仓库](https://huggingface.co/datasets/Yong-Hoon/human3r-dataset)：9月14日再次确认`gated: manual`与原revision `1d90eba6ec995145db9360ebde15ae1dce0ee374`。公开全文检索官网域名和RICH.tar没有发现第二个可确认的RICH归档仓库；当前唯一分卷候选仍为这10份共486.0GB的数据。没有获得新的访问凭据或下载归档，内部覆盖及大文件速度仍未验证。
- [BSTRO官方TSV](https://github.com/paulchhuang/bstro/blob/main/docs/EXP.md)：重新核对文档，含train/val/test的img/hw/label TSV，官网标示约107GB。它可能适合只做接触检测，但原始帧、分辨率、SMPL-X与场景对应仍需审计。JPG现在仅剩约96GB，单为减少剩余字节而切换107GB包已经没有明显优势。
- OpenDataLab的“仅目录、无实际文件”结论见9月10日报告。此次ModelScope尝试的Search/Keyword/Name字段仍返回默认推荐列表，搜索过滤没有生效，不能作为不存在RICH的证据。HF有些请求超时，但官网域名全文检索和精简仓库metadata成功；不声称全面排除所有镜像。

## 可补充研究的其他数据集

### DAMON / DECO

入口：[项目及数据说明](https://deco.is.tue.mpg.de/)；[实际字段及下载要求](https://github.com/sha2nkt/deco#damon-data-description)。

官网明确共5522张图像，提供密集人体顶点接触标注，包含人体受场景支撑与人支撑物体两类接触。官方代码说明已发布SMPL和SMPL-X接触标签，以及按物体拆分的标签。下载须在DECO网站注册并登录，本次只读公开说明，没有将RICH凭据发送给DECO。

适合图像到顶点接触估计的补充监督。README里的pose/transl/shape与cam_k由CLIFF估计，scene_seg是语义分割；这些不能直接当作RICH级别的标定相机、米制场景扫描与GT人体配准。不能单独解除本项目最终接触修正数据契约。此前45MB DECO Keeper包只证明有预处理NPZ，不等于完整DAMON图像已经取得。

### BEHAVE

入口：[官网](https://virtualhumans.mpi-inf.mpg.de/behave/)；[官方分包下载页](https://virtualhumans.mpi-inf.mpg.de/behave/license.html)。

官网明确包含321条序列、8位参与者、20种物体、5个环境、4路Kinect RGB-D、人体和物体配准、相机参数与接触标注。下载页直接列出Date01至Date07压缩包（共约140GB）以及扫描物体、相机标定、split JSON；另有RHOBIN 2024训练26GB、验证测试3.3GB或视频25GB的精简包。以上为页面标示，精简包不能默认包含全部原始接触/场景字段。

适合人体与物体交互及局部几何接触研究，主体采用SMPL，接触对象主要是具体物体；仍需SMPL-X映射与场景契约适配，不能直接替换RICH人体与完整场景数据。范围读取结果另存`rich_sources_20260914/behave_range_probes.json`；实测两包均HTTP206且归档签名正确：RHOBIN训练tar **29,162,670,080bytes（29.16GB，27.16GiB）**，页面“26GB”为近似旧标注；Date01.zip 15,019,046,742bytes。每个仅读取1024bytes，未下载整包或校验内部字段，不把1KB访问耗时当稳定下载速度。

### PhySIC状态纠正

[PhySIC官方README](https://github.com/YuxuanSnow/Phy-SIC)目前给出demo/优化代码；fetch_data所需访问包括SMPL/SMPL-X/AGORA/CameraHMR，evaluation code仍未勾选。该公开说明没有确认一个可直接下载、满足本项目稠密接触契约的独立“PhySIC数据集”。因此不能将此前计划中的“RICH/PhySIC类标注”解释为PhySIC现成数据已可用。

## 下一步

继续已恢复的官方JPG/body/contact下载，完成后按归档格式作完整性校验，再以同一短序列核验图像、人体、相机、场景、接触标签对应。DAMON/BEHAVE保留为补充来源，不自动换训练数据，不读图、不运行新训练。原始网页、请求状态及元数据均保存在`rich_sources_20260914/`。

## 两台机器使用与云端保存（同日后续）

用户询问本次提速原因、是否能上传自己HF、如何供另一台机器使用。已再次核查RICH许可与HF当前存储文档；尚未提供第二台机器地址/平台，未创建仓库、上传数据或创建付费存储。

### 速度证据

原JPG日志并非一直低速：9月11日01:51已有12MiB/s（14连接），07:40仍有10MiB/s（4连接），历史共1991个摘要达到10MiB/s以上；当晚约400GB已传完。最后连接陆续因Expected总长/Actual0失败，降到1连接后退出。9月14日重新认证和恢复16连接后回到多MB/s，控制位图增长与下载日志吻合。可以确认连接恢复以及链路/源站时变，不足以将唯一根因归结为cookie过期或并发数，也不据此部署频繁重连。

### RICH许可边界

当前原文：[RICH License](https://rich.is.tue.mpg.de/license.html)。许可为个人single-user、non-transferable；允许安装到owned/leased/otherwise controlled by you and/or your organization的计算机。同时明确：

> The Data & Software may not be reproduced, modified and/or made available in any form to any third party without Max-Planck’s prior written permission.

“No Distribution”还规定不得copy/share/distribute/transfer等，唯一明示复制例外是one copy for archive purposes only。公开上传HF、给其他人共享数据不在已有许可内；私有仓库本身不证明第三方托管及复制已获授权。自己受控机器使用符合installation条款的方向，若采用第三方云备份/托管，应让许可方澄清相关条款范围。没有代用户联系许可方。

### 存储与转移选择

1. 两机能挂载同一个NAS：优先复用当前NAS目录，可读挂载减少重复副本和公网传输；实际平台权限/挂载能力待用户提供。
2. 两台个人受控机器，主要一次性搬运：完整归档校验后通过SSH/rsync可断点传输；只传需要的归档或序列，目标校验SHA256。不要把仍在写入的aria2稀疏.part当完整数据，也不要传download_lists内凭据或cookie。不启用--delete。
3. 若获准使用云托管且经常跨平台：可选私有OSS/COS/S3类对象存储，区域靠近计算节点、禁止匿名访问，按实际存储与公网流出计费；rclone/provider CLI作分段传输。训练前落到机器本地盘/NAS，不把公网对象存储上的大量小JPEG随机读取当成本地训练性能。
4. HF私有数据仓库技术可行但不优先：当前官方文档免费private storage100GB，PRO包括1TB且超额收费，约598GB训练归档已超过免费额。当前单文件hard limit500GB，建议分块<200GB，故559.4GB的train.tar.gz必须分块。文件限制针对Git-backed repos；HF Storage Buckets有独立规则，不可混用。不保证HF-Mirror改善私有上传或下载速度。

HF文档：[Storage limits](https://huggingface.co/docs/hub/storage-limits)，本机原站超时，已通过镜像页面及官方huggingface/hub-docs仓库原文交叉核对，证据保存于同名目录。
