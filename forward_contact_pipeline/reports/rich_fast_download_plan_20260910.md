# RICH 快速下载与实验启动方案（2026-09-10）

目标：尽早得到同序列的人体、接触、场景、相机与RGB闭环，然后扩展train/val，最后进行冻结的test评测。项目上下文见 `hugs_context_restored_20260910.md`。

镜像最新进展见 `rich_mirror_search_20260910.md`：HF-Mirror已连通，发现`Yong-Hoon/human3r-dataset`的10个RICH分卷（486.0GB），需要HF人工审批访问，内容覆盖与速度待核验。OpenDataLab明确无直接下载；约107GB官方BSTRO TSV仍是另一待核验入口。未改变当前下载队列。

## 最新本机直连复测（北京时间2026-09-10 20:45–20:55）

用户明确没有云平台账号/中转地址，本轮仅继续本机直连。已保存的RICH官方凭据有效，无需重新提供。20:40前后图片约0.3MB/s，早先5–7天预计已不能代表当前速度。

DNS仅发现IPv4 `192.124.27.139`，未发现IPv6默认路由；TLS ALPN选择HTTP/1.1。TCP接收自动调节已启用，`tcp_rmem`上限16MiB，未作全局网络参数修改。这些检查没有找到可立即切换的更快路径，也不足以断言源站限速。

`scripts/benchmark_rich_local.py` 对已有JPG断点作16/8连接各180秒的独占传输，每组舍弃前60秒摘要，保留实际下载字节与aria2控制文件。测试结束恢复原连接数和非JPG限速，不修改队列；结果写入 `datasets/RICH/download_logs/jpg_local_benchmark.json`。本次为顺序现场比较，仍受线路随时间波动影响。

结果（MB/s按十进制计算）：

| 条件 | 近期/去掉首分钟后的平均速率 |
|---|---:|
| 测试前：16连接JPG与body并行，最近18个摘要 | 0.315 MB/s |
| JPG独占，16连接，后11个摘要 | 0.442 MB/s |
| JPG独占，8连接，后11个摘要 | 0.152 MB/s |

两组未记录aria2错误码；退出码7来自180秒主动结束的未完成下载。16连接较8连接更好，但没有测到稳定的多MB/s；暂停其他文件和重建连接也没有恢复早先1MiB/s。不能将独占与并行的差异全部归因于某个设置，因为比较是顺序进行的。

已恢复**2文件并行、每文件16连接、非JPG限速500K**，controller PID **1305167**，JPG与body均确认继续下载，contact仍排队。继续保留并行以推进标注包，未部署定时重启或修改全局TCP设置。

20:54:48恢复并行后的10个摘要（已舍弃首分钟）显示JPG平均**0.288MB/s**、body平均**0.325MB/s**，按约554GB图片余量恒速估算还需**22.3天**。如果一直维持独占测试的0.442MB/s，则约14.5天；两者都不是完成日期承诺，不能继续沿用5–7天。最新快照见`jpg_local_eta_20260910.json`。JPG恢复时一条连接收到HTTP403，其余15条持续传输；未作反复重启或继续增加并发。单条拒绝不足以确定服务器限速规则，也不表示官方凭据失效。一天内下完该余量需平均约**6.41MB/s**，本轮没有测到。若目标是尽早实验，可先用匹配的小子集；普通单包tar.gz不能任意按序列跳读，这不等同于全量数据已到位。

## 历史图片优先配置与ETA（北京时间2026-09-10 18:21，已失效）

已对真实Train JPG作8/16连接各60秒的独占比较，保留下载数据后自动恢复。16连接后段约1.8MiB/s，但初始6.7MiB/s属于启动突发；并行任务恢复后图片一度只有约0.6MB/s，因此不能按短测峰值承诺全量完成时间。

当前采用**每文件16连接、JPG不限速、其他标注包限速500K、两任务并行**，队列为scan→JPG→body→contact，PID **1011347**。scan已完成并通过64成员完整ZIP CRC校验；JPG/body正在下载，contact保留断点等待。图片近期0.9–1.2MiB/s，最新6个10秒摘要均值1MiB/s，按剩余量恒速计算约6.16天，约9月16日晚；建议按**5–7天**规划，网络变化时重新估计。比较和ETA记录分别为`jpg_connection_benchmark.json`、`jpg_eta_20260910.json`。

```bash
cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work
python forward_contact_pipeline/scripts/acquire_rich.py --status
# 任务退出后恢复，保留图片优先设置：
python forward_contact_pipeline/scripts/acquire_rich.py --start --engine aria2 --connections 16 --other-limit 500K
```

仍可能进一步提速的方向是实测已有中转节点的两段吞吐。当前尚未提供欧洲/香港/新加坡节点配置，未购买或启动新节点，不能将假设带宽当实测。也可在归档前部的完整帧与对应人体/contact标签到位后先做小实验；已只读tar目录前缀，开头序列为`Pavallion_006_plankjack/cam_00`，未查看图像。

本节覆盖下方8连接配置、旧PID和旧队列状态。

## 最新执行状态：认证修复、aria2已启动

凭据有效。原预检失败来自未保留POST成功后302跳转所需的PHPSESSID，已修复Cookie处理；4个保护资源的Range均返回206且归档签名正确。已安装项目独立aria2 1.37.0，脚本支持以私有Cookie作GET多连接下载，保留wget连续前缀与aria2控制文件；当前采用**2文件并行×每文件8连接**，后台PID **929174**。

速度实测：wget合计约75.6KB/s；aria2每文件4连接约0.5MB/s；8连接最近六个10秒摘要合计平均约1.387MB/s（scan 0.575MB/s、body 0.812MB/s），顺序观察约18.3倍提升，不保证全程或其他资源同速。证据：`datasets/RICH/download_logs/speed_comparison_20260910.json`。当前scan/body正在下载，contact/JPG排队。扫描/人体当时进度约20%/2%，近期ETA约20分钟/3小时；这些数字会变化。

服务端实际传输大小：scan **904175585 bytes**、body **9147649220 bytes**、contact **28826902288 bytes**、Train JPG **559418993166 bytes**。训练JPG实际约559.4GB，四个保护包约598.3GB。前文使用的500GB是官网近似标注。

常用命令（有正在运行的任务时`--start`会拒绝重复启动）：

```bash
cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work
python forward_contact_pipeline/scripts/acquire_rich.py --status
# 仅在任务已退出、需要恢复时：
python forward_contact_pipeline/scripts/acquire_rich.py --start --engine aria2 --connections 8
```

aria2部分文件可能稀疏扩展到接近总大小，**不能用ls文件大小判断下载完成度**；`--status`会显示aria2真正的已下载量、连接数、速度和ETA。保留对应`.part.aria2`文件，不能用wget续接aria2的稀疏文件。四项本地HTTP回归通过，涵盖Cookie跳转、两种客户端的续传、错误页拒绝及稀疏断点保护。

以下章节保留前期决策和获取过程；“尚无凭据/aria2未安装”的历史状态已由本节覆盖。

## 实际资源预检与当前启动方法（后续更新）

用户已提供全部5个真实链接。公开坐标包 `https://rich.is.tue.mpg.de/media/upload/multicam2world.zip` 已下载至 `datasets/RICH/raw/common/multicam2world.zip`：4304 bytes，ZIP CRC通过，包含7个场景JSON；SHA-256为 `9d5aeb542311049d2fc5b60586a8f30ef16a98aef7ff86c99b781445fa57df77`（本地复查基线）。

受保护的scan下载链接未经认证返回HTTP 200、6311字节HTML登录表单，而非归档。表单使用POST，字段为`username/password/commit`，commit值为`Log in`。因此实际启动入口改用 `scripts/acquire_rich.py`，先验证POST认证，再启动两个wget后台任务；后文aria2方案仅保留为获取可用GET/cookie链接之后的条件方案，不能直接应用于当前未认证URL。

在服务器终端执行一次：

```bash
cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work
python forward_contact_pipeline/scripts/acquire_rich.py --configure
```

交互输入RICH用户名和密码，密码隐藏。脚本将URL编码后的POST表单保存到权限600的 `datasets/RICH/download_lists/credentials.post`；父目录权限700，整个私有清单目录已通过`.gitignore`排除。认证成功即自动启动独立后台进程，按扫描→人体参数→接触→JPG队列、最多两个文件并行下载，关闭终端后继续运行。无需先安装aria2。

```bash
python forward_contact_pipeline/scripts/acquire_rich.py --status
python forward_contact_pipeline/scripts/acquire_rich.py --start
```

`--status`查看当前状态与已下载部分字节；`--start`在无同类任务运行时复用已有凭据重试/续传。`.part`保留未完成数据，仅在下载成功且预期字节数（若服务端提供）及格式检查通过后改为最终文件名。ZIP目录可读性、gzip签名检查不等价于大归档完整CRC校验；状态显式记录完整CRC尚未检查。日志位于`datasets/RICH/download_logs/`。

本地HTTP测试已验证：含特殊字符凭据的POST编码、真实wget从50000字节偏移继续下载并与源ZIP逐字节一致、登录HTML被拒绝、服务器不支持Range时保留已有部分文件。记录：`runs/rich_download_smoke_qy7js05y/summary.json`。这些是本地行为测试，不是RICH站点认证/吞吐的实测。当前未取得用户凭据，4个受保护归档仍未开始。

## 已核验与尚未核验

- 官方入口：https://rich.is.tue.mpg.de/ ，下载入口：https://rich.is.tue.mpg.de/download.php 。公开说明确认多视角RGB、3D bodies、场景扫描和body vertex-level contact。当前未登录的下载入口返回登录表单。
- 用户已注册，并提供下面的下载包清单。体积是官网下载页的近似标注，可能描述解压空间；尚未用资源的Content-Length核验真实传输量。
- NAS `/workspace/nas_auto_backup/yuzilang` 当前约32TB可用；当前系统盘约139GB可用。压缩包与解压输出都放NAS。NAS可用空间属于共享容量，正式启动时再查一次。
- 当前有wget、curl、tmux、conda，无aria2c。旧wget脚本只有文件级并行，不能提速单个大包。
- 目前没有真实资源URL，尚未进行登录、Range、吞吐或归档完整性测试。本文准备的是执行方案，没有启动数据传输。

## 下载顺序与规模

| 顺序 | 资产 | 按用户清单估计 | 可开始的工作 |
|---|---|---:|---|
| 1 | Scans/calibration + multiview-to-world | 约1GB + 4KB | 检查场景单位、坐标转换和相机结构 |
| 2 | Train SMPL-X + Train contact | 9.2 + 29 = 38.2GB | 检查人体拓扑、接触标签正负分布、帧编号；连同公共资产共39.2GB |
| 3 | 上述序列对应的Train JPG | 按实际拆包方式确定；整个train约500GB | 先建立1–2条序列的单目RGB闭环，跑adapter与前端数值审计 |
| 4 | Validation bodies/contact/JPG | 4.3 + 13.2 + 250 = 267.5GB | 验证、阈值选择；完整train+val+公共资产约806.7GB |
| 5 | Test bodies/contact/JPG | 7.3 + 23 + 250 = 280.3GB | 冻结后的最终评测；全量合计约1087GB |

所有split的bodies/contact加公共资产合计约87GB。下载快的小资产先完成，标注与大JPG包可并行；没有必要等全量RGB下载完才开始字段检查。仅有标注时还不能进行RGB接触网络训练。

`Gym_010_dips1` 56GB sample是可选加速入口：若官网下载包说明或归档目录确认其提供同帧RGB、body、contact，则可用于最小闭环；缺公共scan/calibration时补公共包。尚未检查实际归档，不能承诺sample必然完整。无需把它设为所有正式下载的硬性前置。

若JPG按序列/相机提供链接，只取训练split中的1–2条序列、一个相机先跑通；保留原始frame ID和时间戳。若只有单个巨大归档，下载工具无法凭空生成序列级URL。ZIP在支持Range且中央目录可读时可以另行评估选择性获取；普通tar.gz通常不能任意按序列跳读，不能承诺通用“只下载其中几帧”。

完整JPG主线约1.09TB，原始图像清单约12TB（6+3+3）。优先JPG；不启动原始图像全量获取脚本。JPG自身的官方授权脚本仍可使用。解压空间按实际归档内容估算，不把“下载量=解压量”当事实。

## 加速方式

优先在NAS目录中建立独立aria2环境，不改训练环境。在HUGS项目根目录执行：

```bash
cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work
conda create -y --prefix "$PWD/.tools/rich-download" --override-channels -c conda-forge aria2
mkdir -p datasets/RICH/raw/train datasets/RICH/download_lists
chmod 700 datasets/RICH/download_lists
```

这条安装命令尚未执行。如果conda源不可达，先使用已有wget获取小标注包即可，无需阻塞字段审计。

把官网下载页或官方脚本里的真实资源URL存到 `datasets/RICH/download_lists/train.aria2`。aria2支持显式输出名，建议每个资源按下面格式填写，避免多个PHP入口写成同一个文件名；扩展名应使用实际归档格式：

```text
<Train SMPL-X的真实HTTPS资源URL>
  out=<对应实际归档文件名>
<Train contact的真实HTTPS资源URL>
  out=<对应实际归档文件名>
<Train JPG的真实HTTPS资源URL>
  out=<对应实际归档文件名>
```

尖括号是待替换内容，不能原样执行。公共资产、val、test分别建清单并输出到各自目录。清单若含临时签名链接，权限设为600。

先核对认证机制，再启动下载：

- 直接GET资源链接，或HTTP Basic认证：可使用aria2；Basic认证可从当前服务器用户的 `.netrc` 读取，`machine`必须匹配实际资源域名，而非猜测首页域名。
- 浏览器cookie登录、一次性签名或官方脚本POST认证：按官方脚本保留所需cookie、POST字段和编码；不能假设“复制URL + .netrc”一定等价。若POST仅用于换取临时GET链接，在获得该链接后才适合交给aria2。
- 不将账号密码写入命令行或会话报告；认证信息在服务器的私有文件或交互提示中配置。已有有效授权配置可直接复用。

对适用GET/Basic认证的清单，启动：

```bash
tmux new -s rich_download
cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work
umask 077
.tools/rich-download/bin/aria2c \
  --input-file=datasets/RICH/download_lists/train.aria2 \
  --dir=datasets/RICH/raw/train \
  --continue=true --auto-file-renaming=false \
  --max-concurrent-downloads=2 \
  --max-connection-per-server=4 --split=4 --min-split-size=32M \
  --file-allocation=none --max-tries=10 --retry-wait=15 \
  --connect-timeout=30 --timeout=60 \
  --save-session=datasets/RICH/download_lists/train.unfinished.aria2 \
  --save-session-interval=60
```

初始配置同时下载2个文件、每文件最多4个连接。按`Ctrl+B`再按`D`离开tmux；`tmux attach -t rich_download`查看。保留 `.aria2` 控制文件；重跑原命令或将输入换成非空的 `train.unfinished.aria2` 恢复未完成任务。不要在没有校验的情况下删除部分归档/控制文件。

`--file-allocation=none`避免下载前为大文件做完整预分配。完成后仍需校验归档；aria2状态“完成”不等于下载到了正确数据。

若必须立即使用已安装工具，GET/Basic资源的纯URL清单可用：

```bash
cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work/datasets/RICH/raw/train
wget --continue --content-disposition --trust-server-names --netrc \
  --tries=10 --timeout=60 --waitretry=15 \
  --input-file=../../download_lists/train.urls.txt
```

这是顺序、断点续传的备用命令；与aria2的URL加`out=`清单格式不同。旧 `scripts/download_rich_parallel.sh` 可做多个wget并行，但本轮没有重新验证其失败汇总或实际站点行为。

## 用真实链接验证加速是否有效

1. 先完成约4KB的坐标文件，确认内容确为坐标数据，而不是HTTP 200的登录HTML。它不能只因文件小就被判错；大归档只有几KB时才重点排查认证。
2. 在一个较大资源上用同样认证方式发送 `Range: bytes=0-0`，只读取一小段后关闭响应。收到 `206 Partial Content` 且 `Content-Range`正确才算Range测试通过；仅有`Accept-Ranges`响应头不足以证明。
3. 在同一个资源上，比较单连接和4连接各约60秒的稳态吞吐。Range有效且4连接确有收益，再考虑每文件8连接；建议同时文件数维持2，观察站点的429/503和总速度。
4. 若Range不支持，单文件只能单流传输，重点靠多个独立文件并行与减少下载量；若账号/服务器总带宽限速，增加连接数也无法按倍数提速。
5. 若国内直连路径慢，且已有可用的欧洲节点，可比较“官方→欧洲节点→当前服务器”的两段端到端时间。只有两段和费用都合适才使用；新开付费机器不是本方案的默认动作。

无实际URL时不能测量以上项目，也不声称可保证特定提速倍数。

## 理论耗时与到位后的验收

若1087GB确实是要传输的十进制字节量，并且以下速度持续稳定：

| 总有效吞吐 | 1087GB理论传输耗时 |
|---|---:|
| 10 MB/s | 约30.2小时 |
| 50 MB/s | 约6.0小时 |
| 100 MB/s | 约3.0小时 |

这里是MB/s，不是Mbps；不含解压、重试和预处理。未测得当前站点速度。

到位一个完整包就核验官方hash（若提供）、实际字节量、归档类型与目录，再按需解压。官方没有hash时本地SHA-256只能形成后续复查基线，不能独自证明来源完整性。坐标小文件需解析结构；大归档要识别HTML错误页。只读目录/文本/数值，不查看图片。

最小实验验收：SMPL-X/标签实际拓扑和顶点顺序（不能预设BSTRO的SMPL标签与SMPL-X一一对应）、同帧RGB/人体/contact、scene与camera的米制坐标、标签正负分布、train/val/test归属。通过后再训练Stage A并连接Stage B。RICH补齐监督数据，不自动证明现有Human3R/GUSH3R已经通过真实前端几何门槛。

BSTRO checkpoint、107GB TSV、geodesic matrix不作为第一轮adapter前置；正式宣称与BSTRO公平比较前，必须获得其官方评测索引/协议，不能把任意自选帧的结果当同一benchmark。
