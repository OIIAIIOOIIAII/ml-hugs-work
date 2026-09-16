# RICH 备用下载源核查（2026-09-10）

> 2026-09-14更新：原官方后台曾失败，现已恢复；最新进度/速度及新增DAMON、BEHAVE来源见 [新报告](rich_sources_20260914.md)。以下运行状态与ETA均为9月10日历史快照。

目标：查找可替代当前官方 Train JPG（559,418,993,166 bytes）、train_body、train_hsc 的备用资源。现有下载与断点保持不变，仅作公开网页、仓库文档和小范围文件检查，不读取或展示图像。

最新结论：通过 HF-Mirror 已找到包含 RICH 分卷的第三方候选仓库 `Yong-Hoon/human3r-dataset`。它需要人工审批访问，已确认分卷名称和字节数，尚未验证内部内容、完整 train/SMPL-X/contact 覆盖或下载速度。因此现在有可核实的候选资源，但还不能直接替换原下载。

## HF-Mirror 复查与新候选（覆盖初次连接失败结论）

用户提醒 Hugging Face 自身有镜像后，继续测试。HF-Mirror 首页一度只返回部分内容，随后用 IPv4、压缩响应、较长超时和较小元数据字段成功读取搜索 API。不能将短暂连接失败解释为平台永久不可达，也不能将这次成功全部归因于某个参数。

- `https://hf-mirror.com/api/datasets?search=rich&limit=1000&expand%5B%5D=gated` 返回完整 JSON，691 个名称匹配项；该次 HTTP200、传输19,537bytes、2.423秒。**这不是大文件下载测速。**
- 对疑似候选读取公开 metadata/files，排除了文本训练、图文质量评分等同名数据；另外检索 BSTRO、HMR、SMPL、human-scene、DECO。
- 全文检索 `"rich.is.tue.mpg.de"` 得到 `humanmovevqa/HumanMoveVQA` 和 `Yong-Hoon/human3r-dataset`。前者只有 JSONL/视频来源说明，后者实际列出 RICH 分卷。

### 可申请访问的 RICH 分卷仓库

- 原仓库及申请入口：<https://huggingface.co/datasets/Yong-Hoon/human3r-dataset>
- 镜像页面：<https://hf-mirror.com/datasets/Yong-Hoon/human3r-dataset>
- 已核查 revision：`1d90eba6ec995145db9360ebde15ae1dce0ee374`。
- 公开文件列表：`RICH.tar.part_aa` 至 `RICH.tar.part_aj`，共10份，另有 `checksums.sha256`。
- 前9份各53,687,091,200bytes，最后一份2,829,199,360bytes，合计 **486,013,020,160bytes（486.0GB，约452.6GiB）**。页面写“453GB”，实际与GiB口径接近。
- API标记 `gated: manual`；公开页面要求登录并接受访问条件，直接读取受限README返回HTTP401。因此访问需要HF账号获得该仓库权限，当前没有使用或发送任何HF token。 对固定revision首分卷请求前1024bytes，实测HTTP403，返回“you are not in the authorized list…ask for access”，确认当前无法取到归档字节。
- README称该合集包括3DPW、EMDB、RICH、processed BEDLAM；RICH展开后的结构仅写`RICH/...`。**未说明train/val/test细节、分辨率或contact覆盖，不能按标题推断它等价于官方559GB train图片包。**
- 公开LFS列表没有暴露可用校验哈希，不能将其占位符当作SHA256。分卷清单保存于 `rich_mirror_search_20260910/rich_hf_candidate_manifest.json`。

获准访问后的顺序：先验证小范围读取、tar目录和split/帧号/人体/contact覆盖，再测真实归档速度；只有确认适用才启动所需RICH分卷。原`.part`与`.aria2`不删除，不把原`.tar.gz`断点直接续到这个不同的分卷tar。

仅RICH的下载命令可按如下准备（**当前未执行；须先获得权限并在本机配置HF登录**）：

```bash
HF_ENDPOINT=https://hf-mirror.com hf download \
  Yong-Hoon/human3r-dataset --repo-type dataset \
  --revision 1d90eba6ec995145db9360ebde15ae1dce0ee374 \
  --include 'RICH.tar.part_*' checksums.sha256 \
  --local-dir /workspace/nas_auto_backup/yuzilang/ml-hugs-work/datasets/RICH/hf_candidate
```

原下载继续运行。以下保留其他候选的核查结果。

## 已核实的候选

| 来源 | 发布内容及证据 | 能否替代完整训练图片包 |
|---|---|---|
| [HF-Mirror / Human3R Dataset](https://hf-mirror.com/datasets/Yong-Hoon/human3r-dataset) | 实际列出10个RICH分卷，合计486.0GB；需要人工批准访问。 | 候选，内部覆盖及速度待授权后核验 |
| [OpenDataLab / RICH](https://opendatalab.com/OpenDataLab/RICH) | 已通过关键词搜索定位并读取详情及文件列表：`fileNum=0`、`fileSize=0`、文件列表`total=0`。接口明确提示本站暂不提供直接下载，转到RICH官网。 | 否，仅目录介绍 |
| [RICH 官方工具库](https://github.com/paulchhuang/rich_toolkit#download-necessary-files) | README 要求从 RICH 官网获取图片、人体和接触数据；官网指向该工具库及 BSTRO。 | 未提供独立镜像 |
| [SA-HMR](https://github.com/zju3dv/SA-HMR#weights-and-data) | 提供最小预处理评测辅助数据；说明中有 `RICH.zip`、`RICH_train.zip`，训练图片仍要求从原作者下载。其 `bodies` 为拟合的 SMPL-H 参数，不能当作原始 SMPL-X 包。 | 否 |
| [GVHMR 数据准备](https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md) | Google Drive 提供 `RICH_hmr4d_support.tar.gz`；明确不分发原始数据，原始图像/视频和标注仍从官网获取。 | 否 |
| [WHAM 数据准备](https://github.com/yohanshin/WHAM/blob/main/docs/DATASET.md) | Google Drive 提供解析后的 RICH 评测数据。 | 否 |
| [SMPLer-X](https://github.com/MotrixLab/SMPLer-X#preparation) | Hugging Face 链接主要是模型权重或演示，RICH 数据入口仍指向 RICH 官网。 | 否 |
| [DECO 下载脚本](https://github.com/sha2nkt/deco/blob/main/fetch_data.sh) | Keeper 上的 `Release_Datasets.tar.gz` 确实可按范围下载，45,205,516 bytes；归档前缀含 RICH 测试集 NPZ。 | 已核实部分是元数据，不是完整图片镜像 |

### OpenDataLab 实际接口证据

读取公开前端使用的接口（仓库标识以逗号分隔）：

- <https://opendatalab.com/datasets/api/v3/datasets/OpenDataLab,RICH>
- <https://opendatalab.com/datasets/api/v3/datasets/OpenDataLab,RICH/r/main>

文件列表返回 `code=0`、`total=0`、`hasNext=false`，提示：

> 本站暂不提供直接下载，如需下载请至数据集官网: https://rich.is.tue.mpg.de/index.html

不是未登录导致无法列出文件的推测；接口本身明确给出了无直接下载的提示。未创建账号或申请下载权限。

### DECO 实际文件检查

- 发布链接：<https://keeper.mpdl.mpg.de/f/81c3ec9997dd440b8db3/?dl=1>。
- 请求前 1 MiB 得到 HTTP 206，`Content-Range: bytes 0-1048575/45205516`，gzip 签名正确。
- tar 前缀中识别到 `Release_Datasets/rich/rich_test_smplx_cropped_bmp.npz`，该成员未压缩大小 560,461,320 bytes。
- 该 NPZ 第一项为 `imgname.npy`，NumPy 头显示 Unicode 图片路径数组，shape `(9834,)`；没有打开任何图片。
- DECO 的 `data/base_dataset.py` 从 NPZ 读取 `imgname`，随后另外读取图片文件；[作者在 issue #16 的回复](https://github.com/sha2nkt/deco/issues/16#issuecomment-1971323751)也明确要求从 RICH/PROX 官网下载训练数据。
- 仅读取 1 MiB 前缀，没有完成整个 45 MB 包的下载/校验，不能声称已穷举全部成员。

### 相关项目的公开 Drive 链接

- SA-HMR：<https://drive.google.com/drive/folders/1CluXFrJliem1awumjBt7gvitSbSI92cZ?usp=sharing>
- GVHMR：<https://drive.google.com/drive/folders/10sEef1V_tULzddFxzCmDUpsIqfv7eP-P?usp=drive_link>
- WHAM：<https://drive.google.com/drive/folders/13T2ghVvrw_fEk3X-8L0e6DVSYx_Og8o3?usp=sharing>

以上链接来自项目文档，但本机访问 Drive 失败，未验证当前文件列表、直接下载速度或是否需要登录。不能将文档中的链接存在等同于本机已能高速下载。

## 较小的官方数据入口

用户提供的官网下载页列有 **BSTRO train/val/test TSV 数据库约107GB**。[BSTRO 官方训练说明](https://github.com/paulchhuang/bstro/blob/main/docs/EXP.md)确认其包括 `train.img.tsv`、`train.hw.tsv`、`train.label.tsv` 以及 val/test 对应文件。

这不是独立镜像，也没有实测更高下载速度，但它有可能减少接触检测实验所需的图片传输量。当前 HUGS 实验还需核验图片/标签字段、原始帧号、SMPL-X 顶点对应、相机与场景坐标，不能直接将 BSTRO TSV 宣称为完整 RICH 的等价替代。约107GB沿用用户提供的官网标注，尚未验证真实归档字节数；未启动新大包或改变现有下载队列。

## 检索边界

- 已读取 RICH、BSTRO 的官方 README、下载说明及相关 issues，并追踪 SA-HMR、GVHMR、WHAM、SMPLer-X、DECO 的实际文档链接。
- 初次访问Hugging Face、HF-Mirror、Google搜索、Jina Reader、Drive和Zenodo存在连接失败或超时；**后续HF-Mirror搜索、metadata及公开页面已成功读取**，见顶部更新。其他平台未完成的核查仍不能当作“没有镜像”。
- 百度返回安全验证；Bing 返回的结果与检索主题不符，未将其当作证据。
- Kaggle `rich` 查询的首批结果主要是无关同名数据；不代表完整平台排查。
- ModelScope 初次接口请求未正确应用关键词过滤，因此其列表不作为“没有RICH”的依据。
- 原始公开资料与小范围检查证据保存于同名目录 `rich_mirror_search_20260910/`。
