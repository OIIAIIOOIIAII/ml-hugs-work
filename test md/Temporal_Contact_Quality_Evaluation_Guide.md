# 时序接触质量评估方案

> **创建日期**: 2026-05-27  
> **目标**: 评估现有模型的时序渲染质量和接触稳定性  
> **适用范围**: 所有时序相关的模型（Baseline、StM、Temporal Attention 等）

---

## 一、评估背景与目标

### 1.1 为什么需要时序评估？

传统的新视角合成评估指标（PSNR、SSIM、LPIPS）只关注**单帧质量**，但动态场景重建还需要关注**时序质量**：

```
问题 1：时序闪烁
  - 单帧 PSNR 都很高（27 dB）
  - 但相邻帧之间质量波动大
  - 视觉上出现"闪烁"效果
  - 人眼对闪烁非常敏感

问题 2：接触不稳定
  - 脚底接触地面时，位置在"滑动"
  - 接触深度在"抖动"
  - 接触状态在"闪烁"（接触 → 离开 → 接触）
  - 违反物理规律

问题 3：运动不自然
  - 人体运动不连续
  - 关节位置突变
  - 速度/加速度不合理
```

### 1.2 评估目标

| 目标 | 具体指标 | 说明 |
|------|---------|------|
| **整体时序质量** | Temporal PSNR Stability > 0.95 | 渲染质量稳定，不闪烁 |
| **接触稳定性** | Foot Sliding < 1 cm/frame | 脚底接触时几乎不动 |
| **接触平滑性** | Contact Jitter < 2 mm | 脚底高度稳定 |
| **接触一致性** | Contact Consistency > 90% | 接触状态不频繁切换 |

---

## 二、评估指标体系

### 2.1 整体时序质量指标

#### 指标 1：Temporal PSNR Stability（时序 PSNR 稳定性）

**物理意义**：衡量渲染质量在时序上的一致性。如果 PSNR 波动大，说明存在闪烁。

**计算方法**：

```
步骤 1：渲染测试序列的每一帧
  - 输入：测试序列（如 100 帧）
  - 输出：每帧的渲染图像

步骤 2：计算每帧的 PSNR
  - 对比渲染图像和 Ground Truth
  - 得到 PSNR 序列：[PSNR_0, PSNR_1, ..., PSNR_T]

步骤 3：计算统计量
  - Mean PSNR：平均值（越高越好）
  - PSNR Std：标准差（越小越好，表示稳定）
  - PSNR Stability：1 / (1 + Std)（越大越好）
  - PSNR Range：最大值 - 最小值（越小越好）

公式：
  Mean = (1/T) × Σ PSNR_t
  Std = sqrt((1/T) × Σ (PSNR_t - Mean)²)
  Stability = 1 / (1 + Std)
  Range = max(PSNR) - min(PSNR)
```

**解读标准**：

```
优秀：
  Mean PSNR > 27 dB
  PSNR Std < 0.3 dB
  PSNR Stability > 0.95
  PSNR Range < 1.0 dB
  → 质量稳定，无闪烁

良好：
  Mean PSNR > 26 dB
  PSNR Std < 0.5 dB
  PSNR Stability > 0.90
  PSNR Range < 2.0 dB
  → 轻微波动，可接受

较差：
  PSNR Std > 1.0 dB
  PSNR Stability < 0.80
  PSNR Range > 3.0 dB
  → 明显闪烁，需要优化
```

**可视化建议**：

```
绘制 PSNR 时序曲线：
  X 轴：帧编号（0-100）
  Y 轴：PSNR（dB）
  添加：
    - Mean 线（虚线）
    - Mean ± Std 区域（阴影）
    - 标注闪烁帧（PSNR 突变 > 2 dB）
```

---

#### 指标 2：Flickering Score（闪烁分数）

**物理意义**：检测相邻帧之间的质量突变。即使平均 PSNR 高，如果闪烁严重也不行。

**计算方法**：

```
步骤 1：计算相邻帧的 PSNR 差异
  ΔPSNR_t = |PSNR_t - PSNR_{t-1}|，t = 1, 2, ..., T

步骤 2：定义闪烁阈值
  通常：ΔPSNR > 2 dB 认为是"闪烁"

步骤 3：统计闪烁帧
  Flickering Count = Σ I(ΔPSNR_t > 2 dB)
  Flickering Ratio = Flickering Count / (T - 1)

步骤 4：计算其他统计量
  Mean ΔPSNR：平均相邻帧变化
  Max ΔPSNR：最大相邻帧变化
  95th Percentile ΔPSNR：95% 分位数
```

**解读标准**：

```
优秀：
  Flickering Ratio < 5%
  Mean ΔPSNR < 0.3 dB
  Max ΔPSNR < 2.0 dB
  → 几乎无闪烁

良好：
  Flickering Ratio < 10%
  Mean ΔPSNR < 0.5 dB
  Max ΔPSNR < 3.0 dB
  → 偶尔闪烁

较差：
  Flickering Ratio > 15%
  Max ΔPSNR > 5.0 dB
  → 频繁闪烁
```

---

#### 指标 3：Temporal SSIM/LPIPS Consistency（时序结构一致性）

**物理意义**：SSIM 和 LPIPS 的时序稳定性，衡量结构和感知质量的连续性。

**计算方法**：

```
类似 PSNR，计算每帧的 SSIM 和 LPIPS：

SSIM 序列：[SSIM_0, SSIM_1, ..., SSIM_T]
  - Mean SSIM
  - SSIM Std（越小越好）
  - SSIM Stability = 1 / (1 + Std)

LPIPS 序列：[LPIPS_0, LPIPS_1, ..., LPIPS_T]
  - Mean LPIPS（越小越好）
  - LPIPS Std（越小越好）
  - LPIPS Stability = 1 / (1 + Std)
```

**解读标准**：

```
SSIM：
  Mean > 0.92
  Std < 0.01
  → 结构稳定

LPIPS：
  Mean < 0.08
  Std < 0.02
  → 感知质量稳定
```

---

### 2.2 接触时序质量指标（核心！）

#### 指标 1：Foot Sliding（脚部滑动）

**物理意义**：脚底接触地面时的水平移动速度。正常情况下，脚接触地面后应该固定不动（如站立），或者匀速移动（如走路）。如果脚底在接触时"滑动"，说明接触关系不稳定。

**计算方法**：

```
步骤 1：检测每一帧的接触状态
  对于每个脚底 anchor：
    - 获取脚底的世界坐标（xyz）
    - 找到最近的场景高斯（地面）
    - 计算垂直距离（y 轴）
    - 如果距离 < 5 cm，认为"接触"

步骤 2：提取接触帧
  Contact Frames = {t | distance_t < 5 cm}

步骤 3：计算水平速度
  对于每个接触帧 t：
    - 计算脚底的水平位移（x, z 轴）
      Δx = x_t - x_{t-1}
      Δz = z_t - z_{t-1}
    - 计算水平速度
      Velocity_t = sqrt(Δx² + Δz²)

步骤 4：统计滑动指标
  Mean Sliding = (1/N) × Σ Velocity_t（接触帧的平均速度）
  Max Sliding = max(Velocity_t)（接触帧的最大速度）
  Sliding Ratio = (Velocity_t > 1 cm 的比例)

单位转换：
  如果坐标单位是米（m），转换为厘米（cm）：
    Sliding (cm) = Sliding (m) × 100
```

**解读标准**：

```
物理参考：
  - 静止站立：Foot Sliding ≈ 0 cm/frame
  - 正常走路：Foot Sliding ≈ 0.5-1.5 cm/frame（接触初期）
  - 滑动/滑冰：Foot Sliding > 2 cm/frame

评估标准：
  优秀：
    Mean Sliding < 0.5 cm/frame
    Max Sliding < 1.0 cm/frame
    Sliding Ratio < 5%
    → 脚底非常稳定
  
  良好：
    Mean Sliding < 1.0 cm/frame
    Max Sliding < 2.0 cm/frame
    Sliding Ratio < 15%
    → 轻微滑动，可接受
  
  较差：
    Mean Sliding > 2.0 cm/frame
    Max Sliding > 5.0 cm/frame
    → 明显滑动，不真实
```

**可视化建议**：

```
1. 脚底位置时序曲线：
   X 轴：帧编号
   Y 轴：脚底 X/Z 坐标（m）
   标注接触帧（高亮显示）

2. 脚底速度时序曲线：
   X 轴：帧编号
   Y 轴：水平速度（cm/frame）
   标注阈值线（1 cm/frame）
```

---

#### 指标 2：Contact Jitter（接触抖动）

**物理意义**：接触时脚底到地面的距离在时序上的波动。正常情况下，脚底接触地面后，距离应该稳定（如 2 cm）。如果距离在"上下跳动"，说明抖动。

**计算方法**：

```
步骤 1：识别接触周期
  接触周期 = 连续接触的帧序列
  例如：
    帧 50-60：接触（周期 1）
    帧 61-70：不接触
    帧 71-85：接触（周期 2）

步骤 2：对于每个接触周期，记录深度序列
  周期 1：[depth_50, depth_51, ..., depth_60]
  周期 2：[depth_71, depth_72, ..., depth_85]

步骤 3：计算每个周期的抖动
  Jitter_1 = std([depth_50, ..., depth_60])
  Jitter_2 = std([depth_71, ..., depth_85])

步骤 4：统计所有周期
  Mean Jitter = (Jitter_1 + Jitter_2 + ...) / N_periods
  Max Jitter = max(Jitter_1, Jitter_2, ...)
  Jitter Std = std([Jitter_1, Jitter_2, ...])

单位转换：
  如果距离单位是米（m），转换为毫米（mm）：
    Jitter (mm) = Jitter (m) × 1000
```

**解读标准**：

```
物理参考：
  - 稳定接触：Jitter < 1 mm（几乎不动）
  - 轻微抖动：Jitter ≈ 1-3 mm（可接受）
  - 明显抖动：Jitter > 5 mm（脚在"弹跳"）
  - 穿透地面：Jitter > 10 mm（严重问题）

评估标准：
  优秀：
    Mean Jitter < 1.0 mm
    Max Jitter < 2.0 mm
    → 脚底高度非常稳定
  
  良好：
    Mean Jitter < 2.0 mm
    Max Jitter < 4.0 mm
    → 轻微抖动，可接受
  
  较差：
    Mean Jitter > 5.0 mm
    Max Jitter > 10.0 mm
    → 明显抖动，不真实
```

**可视化建议**：

```
1. 接触深度时序曲线：
   X 轴：帧编号
   Y 轴：脚底到地面距离（mm）
   按接触周期分段显示（不同颜色）
   标注每个周期的 Jitter 值

2. Jitter 分布直方图：
   X 轴：Jitter 值（mm）
   Y 轴：周期数量
   标注 Mean 和 Max
```

---

#### 指标 3：Contact Consistency（接触一致性）

**物理意义**：接触状态在时序上的一致性。接触是一个持续过程，不应该频繁切换（接触 → 离开 → 接触）。

**计算方法**：

```
步骤 1：记录每一帧的接触状态
  Contact Status_t = 1（接触）或 0（不接触）
  序列：[0, 0, 1, 1, 1, 0, 0, 1, 1, ...]

步骤 2：检测状态切换
  Transitions_t = |Status_t - Status_{t-1}|
  如果 Transitions_t = 1，说明发生了切换

步骤 3：统计切换次数
  Total Transitions = Σ Transitions_t
  Transition Ratio = Total Transitions / (T - 1)

步骤 4：计算一致性
  Consistency = 1 - Transition Ratio
  或者：
  Consistency = (T - Total Transitions) / T

步骤 5：检测虚假接触
  虚假接触 = 只持续 1 帧的接触
  例如：[0, 0, 1, 0, 0]（第 2 帧是虚假接触）
  
  False Contact Count = 统计这种模式
  False Contact Rate = False Contact Count / Total Contact Frames
```

**解读标准**：

```
物理参考：
  - 正常步态：脚接触地面会持续 10-30 帧
  - 不会"接触 1 帧 → 离开 1 帧 → 接触 1 帧"

评估标准：
  优秀：
    Consistency > 95%
    Total Transitions < 10（100 帧序列）
    False Contact Rate < 3%
    → 接触非常稳定
  
  良好：
    Consistency > 90%
    Total Transitions < 20
    False Contact Rate < 10%
    → 偶尔切换，可接受
  
  较差：
    Consistency < 80%
    Total Transitions > 30
    False Contact Rate > 20%
    → 频繁切换，不真实
```

**可视化建议**：

```
1. 接触状态时序图：
   X 轴：帧编号
   Y 轴：接触状态（0 或 1）
   用柱状图显示
   标注切换点

2. 接触周期长度分布：
   X 轴：周期长度（帧数）
   Y 轴：周期数量
   标注平均周期长度
```

---

#### 指标 4：Contact Attention Stability（接触注意力稳定性）

**物理意义**：接触区域的 attention weights 在时序上的稳定性。接触区域的 attention 应该更稳定（因为接触关系稳定），非接触区域可以灵活变化。

**计算方法**：

```
步骤 1：提取每帧的 attention weights
  Attention_t：[N_human, N_scene]

步骤 2：计算相邻帧的 attention 差异
  ΔAttention_t = |Attention_t - Attention_{t-1}|
  或者用余弦相似度：
  Cosine Similarity_t = cos(Attention_t, Attention_{t-1})

步骤 3：分离接触和非接触区域
  Contact Mask_t：哪些人体高斯在接触区域
  Non-Contact Mask_t：哪些人体高斯不在接触区域
  
  Contact ΔAttention_t = ΔAttention_t[Contact Mask_t]
  Non-Contact ΔAttention_t = ΔAttention_t[Non-Contact Mask_t]

步骤 4：分别计算稳定性
  Contact Stability = 1 / (1 + mean(Contact ΔAttention))
  Non-Contact Stability = 1 / (1 + mean(Non-Contact ΔAttention))
  
  或者用余弦相似度：
  Contact Stability = mean(Cosine Similarity[Contact])
  Non-Contact Stability = mean(Cosine Similarity[Non-Contact])

步骤 5：计算整体变化率
  Attention Change Rate = mean(ΔAttention)
```

**解读标准**：

```
预期行为：
  接触区域的 attention 应该更稳定（变化小）
  非接触区域可以变化大（因为视角在变）

评估标准：
  优秀：
    Contact Stability > 0.90
    Non-Contact Stability > 0.80
    Contact Stability > Non-Contact Stability
    → 接触区域更稳定！
  
  良好：
    Contact Stability > 0.80
    Non-Contact Stability > 0.70
    Contact Stability ≥ Non-Contact Stability
  
  较差：
    Contact Stability < 0.70
    Contact Stability < Non-Contact Stability
    → 接触区域反而不稳定（有问题）
```

**可视化建议**：

```
1. Attention 变化率时序图：
   X 轴：帧编号
   Y 轴：Attention Change Rate
   两条线：接触区域 vs 非接触区域
   期望：接触区域的线在下方（变化小）

2. Attention 热力图对比：
   选择几帧（如 t, t+1, t+2）
   显示 attention weights 热力图
   观察接触区域是否稳定
```

---

### 2.3 运动质量指标

#### 指标 1：Motion Smoothness（运动平滑性）

**物理意义**：人体关节运动的平滑性。正常运动应该是连续的，速度和加速度不应该突变。

**计算方法**：

```
步骤 1：提取人体关节位置序列
  Joints_t：[N_joints, 3]，每帧的关节位置

步骤 2：计算速度
  Velocity_t = Joints_t - Joints_{t-1}

步骤 3：计算加速度
  Acceleration_t = Velocity_t - Velocity_{t-1}

步骤 4：统计平滑性
  Mean Velocity：平均速度
  Mean Acceleration：平均加速度（越小越平滑）
  Velocity Std：速度标准差（越小越稳定）
  Acceleration Std：加速度标准差（越小越稳定）
```

**解读标准**：

```
优秀：
  Mean Acceleration < 0.01 m/frame²
  → 运动非常平滑

良好：
  Mean Acceleration < 0.05 m/frame²
  → 轻微抖动

较差：
  Mean Acceleration > 0.1 m/frame²
  → 运动不平滑，有突变
```

---

#### 指标 2：Temporal Depth Consistency（时序深度一致性）

**物理意义**：渲染深度的时序稳定性。深度不应该闪烁或突变。

**计算方法**：

```
步骤 1：提取每帧的渲染深度图
  Depth_t：[H, W]

步骤 2：计算相邻帧的深度差异
  ΔDepth_t = |Depth_t - Depth_{t-1}|

步骤 3：统计
  Mean ΔDepth：平均深度变化
  Max ΔDepth：最大深度变化
  Depth Stability = 1 / (1 + mean(ΔDepth))
```

**解读标准**：

```
优秀：
  Mean ΔDepth < 0.01 m
  → 深度非常稳定

良好：
  Mean ΔDepth < 0.05 m
  → 轻微变化

较差：
  Mean ΔDepth > 0.1 m
  → 深度闪烁
```

---

## 三、评估流程

### 3.1 完整评估流程

```
Phase 1：准备阶段（半天）
  1. 选择测试序列
     - 推荐：NeuMan 数据集的标准测试序列
     - 长度：50-100 帧
     - 包含：站立、走路、转身等动作
  
  2. 渲染测试序列
     - 使用训练好的模型
     - 保存每帧的渲染图像
     - 保存每帧的深度图
     - 保存每帧的 attention weights（如果有）
  
  3. 准备 Ground Truth
     - GT 图像
     - GT 深度（如果有）
     - 人体 mask（DensePose 或手动标注）

Phase 2：计算指标（1-2 天）
  4. 计算整体时序指标
     - Temporal PSNR Stability
     - Flickering Score
     - Temporal SSIM/LPIPS Consistency
  
  5. 计算接触时序指标
     - Foot Sliding
     - Contact Jitter
     - Contact Consistency
     - Contact Attention Stability
  
  6. 计算运动质量指标
     - Motion Smoothness
     - Temporal Depth Consistency

Phase 3：分析与可视化（1-2 天）
  7. 生成评估报告（JSON）
  8. 生成可视化图表（PNG）
  9. 对比不同模型（Baseline vs StM vs Ours）
  10. 撰写分析总结
```

### 3.2 评估脚本结构

```
hugs/utils/temporal_evaluation.py
│
├── 整体时序评估
│   ├── evaluate_temporal_psnr_stability()
│   ├── evaluate_flickering_score()
│   └── evaluate_temporal_ssim_lpips()
│
├── 接触时序评估
│   ├── evaluate_foot_sliding()
│   ├── evaluate_contact_jitter()
│   ├── evaluate_contact_consistency()
│   └── evaluate_contact_attention_stability()
│
├── 运动质量评估
│   ├── evaluate_motion_smoothness()
│   └── evaluate_temporal_depth_consistency()
│
└── 综合评估
    ├── evaluate_all()
    ├── generate_report()
    └── visualize_results()
```

---

## 四、评估实验设计

### 4.1 对比实验

| 实验 | 模型 | 预期结果 |
|------|------|---------|
| **实验 1** | Baseline（HUGS） | Foot Sliding ~3.5 cm，Contact Jitter ~5 mm |
| **实验 2** | + Depth Supervision | Foot Sliding ~2.1 cm，Contact Jitter ~3 mm |
| **实验 3** | StM | Foot Sliding ~2.5 cm，Contact Jitter ~3.1 mm |
| **实验 4** | + Temporal Attention | Foot Sliding ~0.7 cm，Contact Jitter ~0.9 mm |

### 4.2 消融实验

| 实验 | 配置 | 目的 |
|------|------|------|
| **实验 A** | 无 Temporal Memory | 验证时序记忆的作用 |
| **实验 B** | 无 Temporal Bias | 验证时序 bias 的作用 |
| **实验 C** | 无 Contact Gate | 验证接触门控的作用 |
| **实验 D** | 完整模型 | 作为对比基准 |

### 4.3 预期结果表格

#### 整体时序质量

| 指标 | Baseline | +Depth | StM | Ours | 目标 |
|------|----------|--------|-----|------|------|
| Mean PSNR | 26.09 | 26.40 | 26.60 | **26.70** | > 26.60 |
| PSNR Std | 0.85 | 0.62 | 0.55 | **0.28** | < 0.3 |
| PSNR Stability | 0.85 | 0.88 | 0.89 | **0.96** | > 0.95 |
| Flickering Ratio | 18% | 12% | 10% | **4%** | < 5% |
| Mean ΔPSNR | 0.92 | 0.65 | 0.58 | **0.25** | < 0.3 |

#### 接触时序质量

| 指标 | Baseline | +Depth | StM | Ours | 目标 |
|------|----------|--------|-----|------|------|
| Foot Sliding (cm) | 3.5 | 2.1 | 2.5 | **0.7** | < 1.0 |
| Contact Jitter (mm) | 5.2 | 3.1 | 3.1 | **0.9** | < 2.0 |
| Contact Consistency | 72% | 82% | 82% | **96%** | > 90% |
| Contact Attn Stability | 0.70 | 0.78 | 0.78 | **0.94** | > 0.90 |
| False Contact Rate | 25% | 15% | 15% | **3%** | < 5% |

---

## 五、可视化工具

### 5.1 时序曲线图

```python
# 示例：PSNR 时序曲线
import matplotlib.pyplot as plt

def plot_psnr_timeline(psnr_sequence, model_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    frames = range(len(psnr_sequence))
    ax.plot(frames, psnr_sequence, linewidth=2)
    
    # 添加统计信息
    mean_psnr = np.mean(psnr_sequence)
    std_psnr = np.std(psnr_sequence)
    
    ax.axhline(y=mean_psnr, color='r', linestyle='--', label=f'Mean: {mean_psnr:.2f} dB')
    ax.fill_between(frames, mean_psnr - std_psnr, mean_psnr + std_psnr, 
                     alpha=0.2, color='r', label=f'±{std_psnr:.2f} dB')
    
    ax.set_xlabel('Frame', fontsize=14)
    ax.set_ylabel('PSNR (dB)', fontsize=14)
    ax.set_title(f'Temporal PSNR - {model_name}', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'psnr_timeline_{model_name}.png', dpi=300)
```

### 5.2 接触深度时序图

```python
# 示例：接触深度时序
def plot_contact_depth_timeline(contact_periods, model_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, period in enumerate(contact_periods):
        frames = range(period['start'], period['end'])
        depths = period['depths']
        
        ax.plot(frames, depths, linewidth=2, label=f'Period {i+1}')
        
        # 标注 Jitter
        jitter = np.std(depths) * 1000  # mm
        ax.text(period['start'], max(depths) + 0.001, 
                f'Jitter: {jitter:.1f} mm', fontsize=10)
    
    ax.set_xlabel('Frame', fontsize=14)
    ax.set_ylabel('Foot-Ground Distance (m)', fontsize=14)
    ax.set_title(f'Contact Depth Over Time - {model_name}', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'contact_depth_{model_name}.png', dpi=300)
```

### 5.3 综合对比图

```python
# 示例：多模型对比
def plot_model_comparison(results_dict):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    metrics = ['PSNR Stability', 'Foot Sliding', 'Contact Jitter', 
               'Contact Consistency', 'Flickering Ratio', 'Attn Stability']
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 3, i % 3]
        
        models = list(results_dict.keys())
        values = [results_dict[model][metric] for model in models]
        
        ax.bar(models, values, width=0.5)
        ax.set_title(metric, fontsize=14)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
```

---

## 六、评估报告模板

### 6.1 JSON 报告格式

```json
{
  "model_name": "Temporal Attention",
  "dataset": "NeuMan - jogging",
  "num_frames": 100,
  "evaluation_date": "2026-05-27",
  
  "overall_temporal_quality": {
    "mean_psnr": 26.70,
    "psnr_std": 0.28,
    "psnr_stability": 0.96,
    "psnr_range": 0.95,
    "flickering_ratio": 0.04,
    "mean_delta_psnr": 0.25,
    "max_delta_psnr": 1.82,
    "mean_ssim": 0.925,
    "ssim_stability": 0.97,
    "mean_lpips": 0.072,
    "lpips_stability": 0.96
  },
  
  "contact_temporal_quality": {
    "foot_sliding_cm": 0.72,
    "max_foot_sliding_cm": 1.45,
    "sliding_ratio": 0.08,
    "contact_jitter_mm": 0.92,
    "max_jitter_mm": 2.15,
    "contact_consistency": 0.96,
    "total_transitions": 4,
    "false_contact_rate": 0.03,
    "contact_attn_stability": 0.94,
    "non_contact_attn_stability": 0.87
  },
  
  "motion_quality": {
    "mean_acceleration": 0.008,
    "velocity_std": 0.012,
    "depth_stability": 0.95
  },
  
  "summary": {
    "overall_score": 92.5,
    "strengths": [
      "Excellent temporal PSNR stability (0.96)",
      "Very low foot sliding (0.72 cm/frame)",
      "High contact consistency (96%)"
    ],
    "weaknesses": [
      "Minor jitter in some contact periods (max 2.15 mm)"
    ]
  }
}
```

### 6.2 文本报告模板

```
========================================
时序接触质量评估报告
========================================

模型：Temporal Attention
数据集：NeuMan - jogging
帧数：100
评估日期：2026-05-27

----------------------------------------
1. 整体时序质量
----------------------------------------
Mean PSNR:          26.70 dB
PSNR Std:           0.28 dB ✓ (优秀)
PSNR Stability:     0.96 ✓ (优秀)
Flickering Ratio:   4% ✓ (优秀)
Mean ΔPSNR:         0.25 dB ✓ (优秀)

评价：时序质量优秀，无明显闪烁。

----------------------------------------
2. 接触时序质量
----------------------------------------
Foot Sliding:       0.72 cm/frame ✓ (优秀)
Max Foot Sliding:   1.45 cm/frame ✓ (良好)
Contact Jitter:     0.92 mm ✓ (优秀)
Max Jitter:         2.15 mm ✓ (良好)
Contact Consistency: 96% ✓ (优秀)
False Contact Rate: 3% ✓ (优秀)

评价：接触稳定性优秀，脚底几乎不滑动。

----------------------------------------
3. 运动质量
----------------------------------------
Mean Acceleration:  0.008 m/frame² ✓ (优秀)
Depth Stability:    0.95 ✓ (优秀)

评价：运动平滑，深度稳定。

----------------------------------------
4. 综合评价
----------------------------------------
总体评分：92.5/100 ✓

优势：
  ✓ 时序 PSNR 稳定性优秀
  ✓ 脚底滑动极低
  ✓ 接触一致性高

不足：
  - 个别接触周期有轻微抖动

改进建议：
  - 可增加 temporal smoothness loss 权重
  - 可优化接触检测阈值

========================================
```

---

## 七、实施建议

### 7.1 优先级

```
Phase 1（必须，1-2 天）：
  ✓ Temporal PSNR Stability
  ✓ Flickering Score
  ✓ Foot Sliding
  ✓ Contact Jitter
  ✓ Contact Consistency

Phase 2（建议，2-3 天）：
  ✓ Contact Attention Stability
  ✓ Motion Smoothness
  ✓ Temporal Depth Consistency

Phase 3（可选，1-2 天）：
  ✓ 可视化图表
  ✓ 综合评分系统
  ✓ 自动报告生成
```

### 7.2 测试数据

```
推荐测试序列：
  1. NeuMan - jogging（中等运动，接触丰富）
  2. NeuMan - parkinglot（站立为主，接触稳定）
  3. NeuMan - seattle（复杂运动，挑战大）

每个序列：
  - 长度：50-100 帧
  - 包含：多种接触类型（脚-地面、手-物体等）
```

### 7.3 验收标准

```
评估工具验收：
  ✓ 所有指标计算正确
  ✓ 可视化图表清晰
  ✓ 报告格式完整
  ✓ 支持多模型对比

模型验收（Ours）：
  ✓ PSNR Stability > 0.95
  ✓ Foot Sliding < 1.0 cm/frame
  ✓ Contact Jitter < 2.0 mm
  ✓ Contact Consistency > 90%
```

---

## 八、常见问题

### Q1：如何定义"接触"？

```
A：接触的定义取决于阈值。推荐：
  - 脚底到地面距离 < 5 cm：认为接触
  - 可以根据数据集调整（3-7 cm）
  - 关键是保持一致性（所有模型用相同阈值）
```

### Q2：Foot Sliding 只计算接触帧吗？

```
A：是的！只计算接触帧的滑动速度。
  - 非接触帧脚可以移动（正常走路）
  - 接触帧脚应该静止（或匀速）
```

### Q3：如何处理多个脚底 anchor？

```
A：分别计算每个 anchor 的指标，然后平均。
  - 左脚、右脚分别计算
  - Mean Sliding = (左脚滑动 + 右脚滑动) / 2
```

### Q4：Contact Jitter 的接触周期怎么划分？

```
A：连续接触的帧序列为一个周期。
  - 帧 50-60 接触 → 周期 1
  - 帧 61-70 不接触
  - 帧 71-85 接触 → 周期 2
  - 分别计算每个周期的 Jitter
```

### Q5：如何对比不同模型的指标？

```
A：使用相同的测试序列和相同的评估脚本。
  - 确保公平对比
  - 生成对比图表
  - 计算相对提升（%）
```

---

## 九、文件清单

```
需要创建的文件：
  hugs/utils/temporal_evaluation.py（评估核心）
  hugs/utils/visualization_temporal.py（可视化工具）
  scripts/evaluate_temporal_quality.py（评估脚本）
  cfg_files/temporal_evaluation.yaml（评估配置）

输出文件：
  output/temporal_reports/{model_name}.json（JSON 报告）
  output/temporal_reports/{model_name}.txt（文本报告）
  output/temporal_visualizations/{model_name}/（可视化图表）
```

---

## 十、参考资料

1. **UniCon3R**（CVPR 2026）：接触感知的人体场景重建
   - Foot Sliding 评估方法
   - Contact Jitter 定义

2. **EHMR**（SIGGRAPH 2024）：高效人体运动重建
   - Foot Sliding 物理参考值
   - 运动平滑性评估

3. **TRiGS**（2026）：时序刚体运动的 4DGS
   - Motion Smoothness 计算
   - Temporal Consistency 评估

4. **4DSTR**（AAAI 2025）：时序关联的 4D 生成
   - Temporal Flickering 检测
   - PSNR Stability 定义

---

**祝评估顺利！** 📊
