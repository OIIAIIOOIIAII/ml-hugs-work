# 完整验证实验协议

> **用法**：每次需要完整验证某个 pipeline 版本时，把本文档交给 Claude 阅读，
> 再告知"待验证 pipeline 名称"和"所用的数据设置（是否含粗对齐等）"，
> Claude 即可按本协议自动安排实验、跳过已有结果、收集指标。

---

## 1. 场景清单（6 个 NeuMan 场景）

| seq 名称 | 数据目录 | VIMO 预处理 | 备注 |
|---|---|---|---|
| `lab` | `data/neuman/dataset/lab` | `lab_vimo` | 主要开发场景 |
| `bike` | `data/neuman/dataset/bike` | 未制作 | |
| `citron` | `data/neuman/dataset/citron` | 未制作 | |
| `jogging` | `data/neuman/dataset/jogging` | 未制作 | |
| `parkinglot` | `data/neuman/dataset/parkinglot` | 未制作 | |
| `seattle` | `data/neuman/dataset/seattle` | 未制作 | |

**VIMO 数据说明**：使用 VIMO 粗对齐时，需使用对应的 `{seq}_vimo` 数据目录。
目前只有 `lab_vimo` 完整制作，其余场景如需 VIMO 需先运行预处理。

---

## 2. 实验角色定义

每次完整验证包含三类角色，每类都要在全部 6 个场景上跑：

| 角色 | 含义 | 何时可复用已有结果 |
|---|---|---|
| **OUR** | 待测试的我方 pipeline（版本由用户指定）| 新版本必须重新跑 |
| **STM** | StM baseline（需与 OUR 对齐数据设置）| 配置完全相同时可复用 |
| **HUGS** | HUGS baseline（需与 OUR 对齐数据设置）| 配置完全相同时可复用 |

### 数据设置对齐规则

- 若 OUR pipeline **不含粗对齐（no VIMO）**：STM 和 HUGS 也使用原始数据（`{seq}`）
- 若 OUR pipeline **含 VIMO 粗对齐**：STM 和 HUGS 也必须使用 `{seq}_vimo` 数据，重新跑（结果不可与无 VIMO 版本混用）
- 若 OUR pipeline **含某额外设置 X**（如 opacity_reg、vitpose_kp 等）：STM/HUGS 的对照组**不加 X**，只对齐数据设置

---

## 3. 验证矩阵（完整一轮 = 18 个实验）

```
角色 × 场景 = 3 × 6 = 18 个实验

         lab   bike  citron  jogging  parkinglot  seattle
OUR      [ ]   [ ]    [ ]     [ ]       [ ]         [ ]
STM      [ ]   [ ]    [ ]     [ ]       [ ]         [ ]
HUGS     [ ]   [ ]    [ ]     [ ]       [ ]         [ ]
```

每个实验收集指标：`HUGS PSNR / SSIM / LPIPS`，`HUMAN PSNR / SSIM / LPIPS`

---

## 4. Pipeline 类型规范

### 4.1 OUR pipeline（HUGS + anchor attention）

**训练入口**：`main.py`（工作目录：`ml-hugs-work/`）

**配置文件位置**：`cfg_files/release/neuman/`

**关键参数**（在 yaml 里设置）：
- `dataset.seq`：场景名（`lab_vimo` / `bike` / ...）
- `dataset.mono_depth_dir`：`data/neuman/dataset/{seq}/mono_depth`（有时是 `{seq_vimo}/mono_depth`）
- `exp_name`：`{version_tag}_{seq}_{date}`
- 其余超参由版本 yaml 决定

**启动命令模板**：
```bash
cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work
nohup /workspace/nas_auto_backup/yuzilang/miniconda3/envs/hugs/bin/python main.py \
    --cfg_file cfg_files/release/neuman/{CONFIG_FILE} \
    > run_logs/{EXP_NAME}.log 2>&1 &
echo "PID: $!"
```

**结果位置**：`output/human_scene/neuman/{seq}/hugs_trimlp/{exp_name}/{timestamp}/`

**指标读取命令**：
```bash
grep "HUGS_PSNR\|HUGS_HUMAN_PSNR\|HUGS_SSIM\|HUGS_HUMAN_SSIM\|LPIPS" run_logs/{EXP_NAME}.log | tail -20
```

---

### 4.2 STM pipeline（Scene-to-Motion baseline）

**训练入口**：`main.py`（工作目录：`Dynamic-Avatar-Scene-Rendering-from-Human-centric-Context/`）

**配置文件位置**：`Dynamic-.../cfg_files/`

**关键参数**：
- `dataset.seq`：场景名（需与 OUR 数据设置对齐）
- `use_hugs: false`（StM fusion 模式）
- `exp_name`：`stm_{variant}_{seq}_{date}`

**启动命令模板**：
```bash
cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work/Dynamic-Avatar-Scene-Rendering-from-Human-centric-Context
nohup /workspace/nas_auto_backup/yuzilang/miniconda3/envs/StM/bin/python main.py \
    --cfg_file cfg_files/{CONFIG_FILE} \
    > run_logs/{EXP_NAME}.log 2>&1 &
echo "PID: $!"
```

**结果位置**：`output_stm/human_scene/neuman/{seq}/hugs_trimlp/{exp_name}/{timestamp}/`

---

### 4.3 HUGS pipeline（原版 HUGS baseline）

STM 仓库中同样实现了 HUGS 原版（`use_hugs: true`），也可用 `ml-hugs-work/main.py` 跑原版 HUGS。

**配置方式**：与 OUR pipeline 相同入口，但不启用 anchor attention，不启用粗对齐等改进。

**关键区别**（相对于 OUR）：
- `anchor_attention: null` 或不写
- `dataset.seq`：与 OUR 对齐数据设置

---

## 5. 复用已有结果的判定规则

Claude 在安排实验前，先检查以下条件，全部满足则可复用：

1. 结果目录存在且不为空
2. 训练日志有最终 step 的 val 指标（搜索 `{max_steps - 2000}` 到 `{max_steps}` 的 PSNR 行）
3. 配置文件中 `dataset.seq` 与本次要求一致
4. 数据设置（VIMO vs 原始）与本次要求一致
5. 额外设置（如 opacity_reg_w、vitpose_kp_w 等）与本次要求一致

**复用时操作**：直接从日志中读取指标填表，不重新启动训练。

---

## 6. 指标汇总表格式

每轮验证结束后，用以下格式汇总（Claude 自动生成）：

```
验证轮次：{验证名称}
数据设置：{原始 / VIMO 粗对齐 / ...}
日期：{date}

| 场景     | OUR HUGS | OUR HUMAN | STM HUGS | STM HUMAN | HUGS HUGS | HUGS HUMAN |
|----------|---:|---:|---:|---:|---:|---:|
| lab      |  |  |  |  |  |  |
| bike     |  |  |  |  |  |  |
| citron   |  |  |  |  |  |  |
| jogging  |  |  |  |  |  |  |
| parkinglot|  |  |  |  |  |  |
| seattle  |  |  |  |  |  |  |
| **avg**  |  |  |  |  |  |  |
```

---

## 7. 验证输出结构

### 7.1 目录布局

每轮完整验证在 `output/validation_{tag}/` 下统一存放，`{tag}` 命名规则：
`{pipeline技术核心}_{date}`，例如 `vimo_inline_attn_20260606`。

```
output/
└── validation_{tag}/
    ├── our/
    │   ├── lab   -> 符号链接 → 对应训练输出的 val/ 目录
    │   ├── bike  -> ...
    │   └── ...
    ├── stm/
    │   ├── lab   -> ...
    │   └── ...
    ├── hugs/
    │   ├── lab   -> ...
    │   └── ...
    ├── comparisons/
    │   ├── lab/
    │   │   ├── full_compare_000.png    # GT | OUR | STM | HUGS（全场景）
    │   │   ├── full_compare_001.png
    │   │   ├── human_compare_000.png   # GT | OUR | STM | HUGS（人体 crop）
    │   │   └── ...
    │   ├── bike/
    │   └── ...（每个场景各一个子目录）
    └── summary.md                      # 指标汇总表 + 路径记录
```

**`our/stm/hugs/` 下使用符号链接**，不复制数据，指向各自训练输出的 `val/` 目录。
`comparisons/` 下是实际生成的对比图。

### 7.2 对比图格式

- **`full_compare_{idx:03d}.png`**：全图对比，4 列横向拼接
  - 列顺序：`GT | OUR | STM | HUGS`
  - 每帧顶部有颜色标签条（白=GT，红=OUR，蓝=STM，绿=HUGS）

- **`human_compare_{idx:03d}.png`**：人体区域 crop 对比，格式相同

- 帧序号与各 pipeline 的 val 帧保持一致（`full_final_{idx:03d}.png`）

### 7.3 对比图生成命令

训练全部完成后，运行：

```bash
cd /workspace/nas_auto_backup/yuzilang/ml-hugs-work

python scripts/make_validation_comparison.py \
    --tag vimo_inline_attn_20260606 \
    --scenes lab bike citron jogging parkinglot seattle \
    --our   "output/human_scene/neuman/{seq}/hugs_trimlp/OUR_EXP_NAME/TIMESTAMP" \
    --stm   "Dynamic-Avatar-Scene-Rendering-from-Human-centric-Context/output_stm/human_scene/neuman/{seq}/hugs_trimlp/STM_EXP_NAME/TIMESTAMP" \
    --hugs  "output/human_scene/neuman/{seq}/hugs_trimlp/HUGS_EXP_NAME/TIMESTAMP"
```

路径中的 `{seq}` 为占位符，脚本会对每个场景自动替换。
未完成的角色/场景用 `none` 代替，对比图中对应列显示灰色占位。

---

## 9. 历史验证记录

> 每次完整验证完成后，Claude 在此追加一条记录。

### V0：初始基线（无粗对齐，2026-05-22）

数据设置：原始场景（无 VIMO）

| 场景      | OUR HUMAN | STM HUMAN | HUGS HUMAN |
|-----------|---:|---:|---:|
| lab       | — | — | 18.8005 |
| parkinglot | — | — | — |
| seattle   | — | — | — |
| bike      | — | — | — |
| citron    | — | — | — |
| jogging   | — | — | — |

> 注：V0 阶段 OUR pipeline 尚未成型，以 HUGS 原版为 baseline。

---

### V1：当前最优（ADC12k + transl_xyz attention stage2，2026-05-27）

数据设置：无 VIMO，原始场景

| 场景       | OUR HUMAN | OUR HUGS | STM HUMAN | HUGS HUMAN |
|------------|---:|---:|---:|---:|
| lab        | **19.5512** | 26.4374 | — | 18.8005 |
| parkinglot | 19.7643 | 27.2009 | — | — |
| seattle    | 19.2921 | 26.2756 | — | — |
| bike       | — | — | — | — |
| citron     | — | — | — | — |
| jogging    | — | — | — | — |

> 注：parkinglot/seattle 为旧版 xyz-only attention 结果（非最新 transl_xyz stage2 版本），其余场景待跑。
> STM 对应列需用原始场景数据重新对齐后补充。

---

## 10. 下一次完整验证计划

**待验证 pipeline**：`vimo_inline_attn_18k`（Exp G 最优，含 VIMO 粗对齐 + anchor attention）

**数据设置**：VIMO（使用 `{seq}_vimo` 数据，需为其余 5 个场景制作 VIMO 预处理）

**对照组**：
- STM + VIMO（对应 Exp F2/F3 的其余场景版本）
- HUGS + VIMO（对应 Exp N 的其余场景版本）

**现有可复用结果**：
- OUR lab：Exp G（`adc12000_transl_xyz_attention_plus3000_stage_lab_20260527`）✅
- STM lab：Exp F2/F3（20k 步，待 F3 完成）
- HUGS lab：Exp N（18k 步，已完成 17.6872）

**待运行**：其余 5 个场景的 VIMO 预处理 + OUR/STM/HUGS 训练（共 15 个实验）

---

## 11. Claude 执行指引

当用户说"用 X 版本跑完整验证"时，Claude 按以下步骤操作：

1. **读本文件**，确认 6 个场景和三类角色
2. **询问用户**（如未指明）：数据设置是否含 VIMO？STM/HUGS 对照组是否需要修改？
3. **检查已有结果**：逐场景逐角色检查，标注哪些可复用、哪些需要跑
4. **准备配置文件**：若目标场景无现成 yaml，从参考场景（lab）复制并修改 `seq` 和 `exp_name`
5. **检查 VIMO 预处理**：若需要且不存在，提示用户先运行 VIMO 处理
6. **顺序启动实验**：每次启动一个（GPU 资源有限），后台 watcher 等待上一个完成再启动下一个
7. **等待完成**：监控日志，收集最终指标
8. **汇总填表**：按第 6 节格式输出结果，并在第 7 节追加历史记录
9. **更新记忆文件**：`memory/hugs-experiment-results.md`
