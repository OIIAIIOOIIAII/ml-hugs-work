# Temporal Contact Attention 实现指南

> **创建日期**: 2026-05-27  
> **目标**: 实现时序感知的接触注意力机制，同时提升渲染质量和训练效率  
> **预期效果**: PSNR +0.1-0.2 dB，Foot Sliding -60%，训练加速 3-5x

---

## 一、项目背景与目标

### 1.1 当前问题

我们现有的 attention 机制是**逐帧独立计算**的，存在三个核心问题：

1. **时序闪烁**：相邻帧的 attention 权重可能突变，导致渲染结果闪烁
2. **接触不稳定**：接触状态在帧之间频繁切换（接触 → 离开 → 接触），不符合物理规律
3. **计算冗余**：相邻帧人体位置相近，但每次都重新计算 attention，浪费计算资源

### 1.2 核心洞察

1. **人体运动是连续的**：相邻帧的人体位移有限（通常 < 5cm），不会突变
2. **场景是静态的**：地面、墙壁、家具不会动，场景高斯的位置不变
3. **接触是持续的**：脚踩到地面后会持续接触多帧（通常 > 10 帧），不会频繁切换

### 1.3 设计目标

| 目标 | 具体指标 | 说明 |
|------|---------|------|
| **质量提升** | PSNR +0.1-0.2 dB | 整体渲染质量提升 |
| **时序稳定性** | Foot Sliding < 1 cm/frame | 脚底接触时几乎不动 |
| **接触稳定性** | Contact Jitter < 2 mm | 脚底高度稳定 |
| **一致性** | Contact Consistency > 90% | 接触状态不闪烁 |
| **训练加速** | 3-5x 加速 | 利用时序缓存减少计算 |

---

## 二、设计原理

### 2.1 为什么 Temporal Attention 可行？

#### 物理可行性
- **人体运动学约束**：人体不能瞬间移动（有惯性），相邻帧位移有限
- **接触持续性**：脚接触地面后会持续多帧，不会"接触 1 帧 → 离开 1 帧"
- **场景静态性**：地面、家具不会动，场景的"可接触性"是固定的

#### 数学可行性
- **Attention 连续性**：Attention(Q, K, V) 是 Q 和 K 的连续函数，如果 Q 和 K 变化小，Attention 也变化小
- **时序平滑性可优化**：可以添加损失函数 L_smooth = ||Attention_t - Attention_{t-1}||²
- **历史信息是正则化**：防止过拟合到当前帧的噪声，提升泛化能力

#### 计算可行性
- **缓存机制**：历史的 attention 可以缓存，只需存储和读取，开销很小
- **插值可行**：相邻帧的 attention 相似，可以线性插值，计算量极小
- **加速训练**：60% 的帧可复用历史，25% 可插值，只有 15% 需要完整计算

### 2.2 核心机制

#### 机制 1：时序记忆（Temporal Memory）

```
原理：维护一个"历史缓冲区"，保存最近 K 帧的信息

存储内容：
  - Contact features（接触特征）
  - Attention weights（注意力权重）
  - Contact status（接触状态）

工作方式：
  第 t 帧查询历史（第 t-K 到 t-1 帧）
  → 计算时序加权平均（越近的帧权重越大）
  → 得到"历史上下文"（Temporal Context）

示例（K=5）：
  第 10 帧查询：
    第 9 帧：权重 0.85（最近）
    第 8 帧：权重 0.72
    第 7 帧：权重 0.61
    第 6 帧：权重 0.52
    第 5 帧：权重 0.44（最远）
```

#### 机制 2：时序 Bias（Temporal Bias）

```
原理：在计算当前帧的 attention 时，加入"历史 bias"

标准 attention：
  Attention(Q, K, V) = softmax(Q·K/√d) · V

加入时序 bias：
  Attention(Q, K, V) = softmax(Q·K/√d + Bias_temporal) · V

Bias_temporal 来源：
  - 从历史 contact features 投影得到
  - 逻辑：历史上接触过的场景区域，当前也应该关注

效果：
  第 9 帧：脚底关注地面的 A 区域
  第 10 帧：脚底移动了 1cm
  
  如果没有 temporal bias：
    Attention 完全重新计算，可能关注 B 区域 → 突变！
  
  如果有 temporal bias：
    Bias 告诉 attention："第 9 帧关注了 A 区域"
    Attention 会倾向于继续关注 A 区域附近 → 平滑过渡！
```

#### 机制 3：接触门控（Contact Gate）

```
原理：接触区域和非接触区域，对历史的依赖程度不同

门控值 gate ∈ [0, 1]：
  - Gate = 1：完全使用历史的 attention
  - Gate = 0：完全使用当前帧的 attention
  - Gate = 0.7：70% 历史 + 30% 当前

Gate 由什么决定？
  1. 接触程度：
     - 接触区域：gate 大（强依赖历史）
     - 非接触区域：gate 小（弱依赖历史）
  
  2. 运动速度：
     - 运动慢：gate 大（变化小，可以依赖历史）
     - 运动快：gate 小（变化大，需要重新计算）

示例：
  脚底接触地面，运动缓慢：
    gate = 0.8（80% 历史 + 20% 当前）→ 非常稳定！
  
  手部快速挥动：
    gate = 0.2（20% 历史 + 80% 当前）→ 灵活响应！
  
  胸部（不接触）：
    gate = 0.3（30% 历史 + 70% 当前）→ 适中
```

#### 机制 4：时序融合（Temporal Fusion）

```
原理：将"当前帧 attention"和"历史 attention"融合

融合公式：
  Attention_final = Gate · Attention_history + (1 - Gate) · Attention_current

示例：
  第 10 帧：
    Attention_current：第 10 帧独立计算的结果
    Attention_history：第 5-9 帧的加权平均
    Gate = 0.7（接触区域，运动慢）
    
    Attention_final = 0.7 × Attention_history + 0.3 × Attention_current

效果：
  - 保留了历史信息（70%），保证稳定
  - 又融入了当前信息（30%），保证响应性
  - 不会突变，也不会过于滞后
```

---

## 三、模块架构设计

### 3.1 整体流程图

```
时间 t-K          时间 t-1          时间 t
  │                │                │
  ├─ Human_{t-K}   ├─ Human_{t-1}   ├─ Human_t ──────────┐
  ├─ Scene_{t-K}   ├─ Scene_{t-1}   ├─ Scene_t           │
  │                │                │                    │
  └─ Attn_{t-K}    └─ Attn_{t-1}    │                    │
       │                │           │                    │
       └────────────────┴───────────┘                    │
                    │                                    │
              Temporal Memory                           │
              (缓存历史)                                 │
                    │                                    ▼
              ┌─────────────────────────────────────────────┐
              │   Contact-Guided Temporal Attention Layer   │
              │                                              │
              │  Query:  Human_t                            │
              │  Key:    Scene_t                            │
              │  Value:  Scene_t                            │
              │  Bias:   Temporal_Context (来自 Memory)     │
              │  Gate:   Contact + Velocity                 │
              └─────────────────────────────────────────────┘
                            │
                            ▼
                    Temporal Attention Weights
                            │
                            ▼
                    Update Gaussian Attributes
                            │
                            ▼
                    Render Image + Depth
```

### 3.2 模块组成

```
TemporalContactAttention (主模块)
│
├── TemporalContactMemory (时序记忆)
│   ├── 存储：contact_features, attention_weights, contact_status
│   ├── 更新：FIFO 缓冲区，decay 加权
│   └── 查询：时序加权平均
│
├── ContactGuidedTemporalAttentionLayer (时序注意力层)
│   ├── Query/Key/Value 投影
│   ├── Temporal Bias 计算
│   ├── Contact Gate 计算
│   └── Temporal Fusion
│
├── HierarchicalTemporalAccelerator (层次加速器)
│   ├── Level 1: 缓存复用（最快）
│   ├── Level 2: 时空插值（快）
│   └── Level 3: 完整计算（中等）
│
└── TemporalContactLoss (时序损失)
    ├── 时序平滑性损失
    ├── 时序一致性损失
    └── 接触区域正则化
```

---

## 四、具体实现方案

### Phase 1：时序记忆模块（1-2 天）

#### 4.1.1 TemporalContactMemory

**文件位置**: `hugs/models/modules/temporal_memory.py`

**核心功能**:
- 维护一个 FIFO 缓冲区，存储最近 K 帧的信息
- 支持更新和查询操作
- 使用时序 decay 加权

**实现要点**:

```
输入：
  - feature_dim: 特征维度（如 64）
  - memory_size: 缓冲区大小（如 5）
  - decay: 衰减率（如 0.85）

存储结构：
  - contact_features: [memory_size, N_anchors, feature_dim]
  - attention_weights: [memory_size, N_human, N_scene]
  - contact_status: [memory_size, N_anchors]
  - timestamps: [memory_size]

更新操作：
  1. 缓冲区未满：直接写入
  2. 缓冲区已满：向左移位，写入最新

查询操作：
  1. 计算时间权重：weight_t = decay ^ (current_time - timestamp_t)
  2. 归一化权重：weight_t / sum(weights)
  3. 时序加权平均：sum(weight_t * feature_t)
  4. 返回 temporal_context dict

关键设计：
  - 使用 register_buffer 存储（不更新梯度）
  - 支持动态大小的 batch
  - 处理边界情况（缓冲区为空）
```

#### 4.1.2 测试用例

```
测试 1：基本功能
  - 创建 memory（size=5）
  - 更新 10 次
  - 验证缓冲区只保留最近 5 帧

测试 2：时序加权
  - 插入 5 帧不同时间戳的数据
  - 查询，验证最近的帧权重更大

测试 3：空缓冲区
  - 创建 memory 但不更新
  - 查询，应返回 None 或默认值
```

---

### Phase 2：时序注意力层（2-3 天）

#### 4.2.1 ContactGuidedTemporalAttentionLayer

**文件位置**: `hugs/models/modules/temporal_attention.py`

**核心功能**:
- 标准 attention 计算（Q, K, V）
- 加入 temporal bias
- 计算 contact gate
- 时序融合

**实现要点**:

```
输入：
  - human_features: [N_human, feature_dim]
  - scene_features: [N_scene, feature_dim]
  - temporal_context: dict（来自 TemporalContactMemory）
  - human_velocity: [N_human, 3]

组件：
  1. Query/Key/Value 投影层（Linear）
  2. 多头 attention 机制
  3. Temporal Bias 投影（MLP）
  4. Contact Gate 计算（MLP + Sigmoid）
  5. Velocity Encoder（MLP + Sigmoid）
  6. LayerNorm + Residual Connection

前向流程：
  Step 1: 标准 attention 计算
    Q = query_proj(human_features)
    K = key_proj(scene_features)
    V = value_proj(scene_features)
    attention_scores = Q·K / √d
  
  Step 2: 加入 temporal bias
    temporal_bias = temporal_bias_proj(temporal_context['contact_features'])
    attention_scores += temporal_bias
  
  Step 3: Softmax
    attention_weights = softmax(attention_scores)
  
  Step 4: 计算 contact gate
    contact_degree = compute_contact_degree(human_features, temporal_context)
    gate_input = concat(human_features, contact_degree)
    contact_gate = sigmoid(contact_gate_mlp(gate_input))
  
  Step 5: 速度调制
    velocity_weight = sigmoid(velocity_encoder(human_velocity))
    final_gate = contact_gate * (1 - velocity_weight)
  
  Step 6: 时序融合
    if temporal_context exists:
      blended_attention = final_gate * history_attention + (1 - final_gate) * current_attention
    else:
      blended_attention = current_attention
  
  Step 7: 加权求和 + 输出投影
    attended_features = blended_attention · V
    output = output_proj(attended_features)
    output = layer_norm(output + human_features)  # Residual
  
  返回：
    - output: [N_human, feature_dim]
    - attention_weights: [N_human, N_scene]

关键设计：
  - 多头 attention（num_heads=4）
  - Residual connection 保证梯度流
  - Gate 在 [0, 1] 范围内
  - 处理 temporal_context 为 None 的情况
```

#### 4.2.2 测试用例

```
测试 1：基本前向传播
  - 创建 attention 层
  - 输入随机特征
  - 验证输出维度正确

测试 2：Temporal Bias
  - 提供 temporal_context
  - 验证 attention_scores 包含 bias

测试 3：Contact Gate
  - 测试不同 contact_degree 下的 gate 值
  - 验证 contact_degree 高时 gate 大

测试 4：速度调制
  - 测试不同 human_velocity 下的 gate 值
  - 验证 velocity 大时 gate 小

测试 5：时序融合
  - 提供/不提供 temporal_context
  - 验证融合逻辑正确
```

---

### Phase 3：层次加速器（1-2 天）

#### 4.3.1 HierarchicalTemporalAccelerator

**文件位置**: `hugs/models/modules/temporal_accelerator.py`

**核心功能**:
- 智能选择计算策略（复用/插值/完整计算）
- 管理缓存
- 加速训练

**实现要点**:

```
输入：
  - model: ContactGuidedTemporalAttentionLayer
  - config: 配置参数

策略选择逻辑：
  策略 1：检查缓存（最快）
    条件：frame_id-1 在缓存中，且人体位移 < reuse_threshold（5cm）
    动作：直接返回缓存的 output 和 attention
    加速：~100x（只需读取）
  
  策略 2：时空插值（快）
    条件：前后关键帧在缓存中
    动作：线性插值
    加速：~20x（只需插值计算）
  
  策略 3：完整计算（中等）
    条件：不满足策略 1 和 2
    动作：调用 model 完整计算
    加速：1x
  
缓存管理：
  - 存储：output, attention, human_xyz
  - 限制大小：max_cache_size=10
  - 淘汰策略：删除最旧的帧

关键设计：
  - 人体位移计算：mean(||human_xyz_t - human_xyz_{t-1}||)
  - 插值权重：alpha = (frame_id - prev_keyframe) / keyframe_interval
  - 缓存命中统计：记录各策略的使用次数
```

#### 4.3.2 测试用例

```
测试 1：缓存复用
  - 连续调用 10 帧，人体位移小
  - 验证策略 1 被触发

测试 2：时空插值
  - 设置 keyframe_interval=5
  - 在第 3 帧调用，验证策略 2 被触发

测试 3：完整计算
  - 人体位移大
  - 验证策略 3 被触发

测试 4：缓存管理
  - 调用 15 帧，max_cache_size=10
  - 验证缓存只保留最近 10 帧

测试 5：加速统计
  - 运行 100 帧
  - 打印各策略使用比例
```

---

### Phase 4：时序损失函数（1-2 天）

#### 4.4.1 TemporalContactLoss

**文件位置**: `hugs/losses/temporal_loss.py`

**核心功能**:
- 时序平滑性损失
- 时序一致性损失
- 接触区域正则化

**实现要点**:

```
输入：
  - render_pkg: 渲染结果
  - data: 训练数据
  - attention_history: attention 权重历史
  - frame_id: 当前帧 ID

损失组成：
  1. 标准 RGB/Depth Loss（已有）
     rgb_loss = MSE(render, gt)
     depth_loss = 1 - Pearson(render_depth, mono_depth)
  
  2. 时序平滑性损失（新增）
     条件：len(attention_history) >= 3
     计算：
       velocity_t = attention_t - attention_{t-1}
       velocity_{t-1} = attention_{t-1} - attention_{t-2}
       temporal_smooth_loss = MSE(velocity_t, velocity_{t-1})
     权重：lambda_temporal_smooth = 0.1
  
  3. 时序一致性损失（新增）
     条件：len(attention_history) >= 2
     计算：
       temporal_consistency_loss = MSE(attention_t, attention_{t-1})
     权重：lambda_temporal_consistency = 0.05
  
  4. 接触区域正则化（新增）
     条件：contact_mask 存在
     计算：
       contact_var = attention[contact_mask].var()
       non_contact_var = attention[~contact_mask].var()
       contact_stability_loss = ReLU(contact_var - non_contact_var)
     权重：0.1
     目的：接触区域的 attention 应该更稳定

总损失：
  total_loss = rgb_loss + depth_loss + 
               temporal_smooth_loss + 
               temporal_consistency_loss + 
               contact_stability_loss

关键设计：
  - 只在历史足够时计算时序损失
  - 接触区域和非接触区域分别处理
  - 权重可调（通过 config）
```

#### 4.4.2 测试用例

```
测试 1：基本损失计算
  - 提供 3 帧 attention_history
  - 验证所有损失项都能计算

测试 2：历史不足
  - 只提供 1 帧 attention_history
  - 验证时序损失为 0

测试 3：接触区域
  - 提供 contact_mask
  - 验证接触区域正则化生效

测试 4：损失权重
  - 调整各损失权重
  - 验证总损失变化
```

---

### Phase 5：集成到训练流程（2-3 天）

#### 4.5.1 修改训练循环

**文件位置**: `hugs/trainer/gs_trainer.py`

**集成要点**:

```
初始化阶段：
  1. 创建 TemporalContactMemory
  2. 创建 ContactGuidedTemporalAttentionLayer
  3. 创建 HierarchicalTemporalAccelerator
  4. 初始化 attention_history 列表

训练循环（每帧）：
  Step 1: 提取特征
    human_features = extract_human_features(batch)
    scene_features = extract_scene_features(batch)
    human_velocity = compute_human_velocity(frame_id, batch)
  
  Step 2: 查询时序记忆
    temporal_context = temporal_memory.query(
      query_timestamp=float(frame_id),
      query_human_xyz=human_features[:, :3]
    )
  
  Step 3: 加速 forward
    output, attention_weights = accelerator.forward(
      frame_id, 
      human_features, 
      scene_features, 
      temporal_context,
      human_velocity
    )
  
  Step 4: 渲染
    render_pkg = render_with_attention(
      human_gs_out=batch['human_gs'],
      scene_gs_out=batch['scene_gs'],
      attention_weights=attention_weights
    )
  
  Step 5: 计算损失
    loss, loss_dict = temporal_loss_fn(
      render_pkg, batch, attention_history, frame_id
    )
  
  Step 6: 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
  Step 7: 更新历史
    attention_history.append(attention_weights.detach())
    if len(attention_history) > 10:
      attention_history.pop(0)
  
  Step 8: 更新时序记忆
    contact_features = extract_contact_features(output)
    contact_status = detect_contact_status(render_pkg)
    temporal_memory.update(
      contact_features,
      attention_weights.detach(),
      contact_status,
      timestamp=float(frame_id)
    )

关键修改：
  - 在 trainer.__init__ 中初始化时序模块
  - 在 trainer.training_step 中集成时序逻辑
  - 在 trainer.save_checkpoint 中保存时序模块状态
  - 在 trainer.load_checkpoint 中加载时序模块状态
```

#### 4.5.2 配置文件

**文件位置**: `cfg_files/temporal_attention.yaml`

```yaml
# Temporal Attention 配置

temporal_attention:
  enable: true
  
  # Temporal Memory
  memory_size: 5
  decay: 0.85
  
  # Temporal Attention Layer
  feature_dim: 64
  num_heads: 4
  temporal_window: 5
  
  # Accelerator
  reuse_threshold: 0.05  # 5cm
  interpolation_interval: 5
  max_cache_size: 10
  
  # Loss Weights
  lambda_temporal_smooth: 0.1
  lambda_temporal_consistency: 0.05
  lambda_contact_stability: 0.1
  
  # Training
  warmup_frames: 10  # 前 10 帧不使用时序损失
```

---

### Phase 6：评估指标实现（2-3 天）

#### 4.6.1 时序评估模块

**文件位置**: `hugs/utils/temporal_evaluation.py`

**核心功能**:
- 整体时序质量评估
- 接触时序质量评估
- 可视化

**实现要点**:

```
整体时序指标：
  1. Temporal PSNR Stability
     - 计算每帧 PSNR
     - 计算 mean, std, stability_score = 1/(1+std)
     - 期望：stability > 0.95
  
  2. Flickering Score
     - 计算相邻帧 PSNR 差异
     - 统计 ΔPSNR > 2 dB 的帧占比
     - 期望：< 10%

接触时序指标：
  1. Foot Sliding
     - 检测接触帧
     - 计算脚底水平速度
     - 期望：< 1 cm/frame
  
  2. Contact Jitter
     - 计算接触深度的时序标准差
     - 期望：< 2 mm
  
  3. Contact Consistency
     - 统计接触状态切换次数
     - 计算一致性比例
     - 期望：> 90%
  
  4. Contact Attention Stability
     - 计算 attention 权重的时序变化
     - 分离接触/非接触区域
     - 期望：接触区域 > 0.90

评估流程：
  1. 渲染测试序列（100 帧）
  2. 计算每帧指标
  3. 计算时序统计量
  4. 生成报告（JSON）
  5. 可视化（PNG）

可视化内容：
  - PSNR 时序曲线
  - 脚部位置时序
  - 接触深度时序
  - 接触状态时序
  - Attention 变化率
  - 综合评分对比
```

---

## 五、实验设计

### 5.1 对比实验

| 实验 | 配置 | 预期效果 |
|------|------|---------|
| **Baseline** | 逐帧 attention | PSNR 26.60, Foot Sliding 1.8 cm |
| **+ Temporal Memory** | 只加时序记忆 | Foot Sliding 降低到 1.2 cm |
| **+ Temporal Bias** | 加入时序 bias | PSNR Stability 提升 |
| **+ Contact Gate** | 加入接触门控 | Contact Consistency 提升 |
| **Full Model** | 所有组件 | PSNR 26.70, Foot Sliding 0.7 cm |

### 5.2 消融实验

| 实验 | 变量 | 范围 |
|------|------|------|
| **Memory Size** | K | 3, 5, 7, 10 |
| **Decay Rate** | decay | 0.7, 0.85, 0.95 |
| **Reuse Threshold** | threshold | 0.03, 0.05, 0.1 m |
| **Keyframe Interval** | interval | 3, 5, 10 |

### 5.3 预期结果表格

| 指标 | Baseline | StM | Ours | 提升 |
|------|----------|-----|------|------|
| Overall PSNR | 26.60 | 26.60 | **26.70** | +0.10 dB |
| PSNR Stability | 0.90 | 0.88 | **0.96** | +0.06 |
| Foot Sliding (cm) | 1.8 | 2.5 | **0.7** | -61% |
| Contact Jitter (mm) | 2.4 | 3.1 | **0.9** | -63% |
| Contact Consistency | 88% | 82% | **96%** | +8% |
| Training Time | 10 min | 12 min | **8 min** | -20% |

---

## 六、实施时间线

| Phase | 任务 | 时间 | 依赖 |
|-------|------|------|------|
| **Phase 1** | Temporal Memory | 1-2 天 | 无 |
| **Phase 2** | Temporal Attention Layer | 2-3 天 | Phase 1 |
| **Phase 3** | Hierarchical Accelerator | 1-2 天 | Phase 2 |
| **Phase 4** | Temporal Loss | 1-2 天 | Phase 2 |
| **Phase 5** | 集成到训练流程 | 2-3 天 | Phase 1-4 |
| **Phase 6** | 评估指标 | 2-3 天 | Phase 5 |
| **总计** | - | **9-15 天** | - |

---

## 七、关键注意事项

### 7.1 实现细节

1. **梯度处理**：
   - 缓存的 attention weights 使用 `.detach()`，不反传梯度
   - 只损失函数的梯度会反传到当前帧

2. **内存管理**：
   - 缓冲区大小要合理（K=5-10）
   - 缓存要限制大小（max_cache_size=10）
   - 及时释放不再需要的缓存

3. **边界情况**：
   - 前几帧没有历史：跳过时序损失
   - 缓冲区为空：返回默认值
   - 人体数量变化：处理 padding/truncation

4. **调试技巧**：
   - 打印各策略使用比例
   - 可视化 attention weights 时序变化
   - 监控 loss 曲线（时序损失应该逐渐下降）

### 7.2 性能优化

1. **GPU 加速**：
   - 所有计算在 GPU 上进行
   - 使用 torch.einsum 高效计算
   - 避免 CPU-GPU 数据传输

2. **批处理**：
   - 支持 batch size > 1
   - 时序维度可以并行处理

3. **缓存优化**：
   - 使用 pinned memory
   - 预分配缓冲区
   - 避免频繁创建 tensor

### 7.3 常见错误

1. **错误 1：梯度泄漏**
   - 问题：缓存的 tensor 带有梯度，导致内存泄漏
   - 解决：使用 `.detach()` 和 `.cpu()` 存储

2. **错误 2：时序错位**
   - 问题：时间戳不正确，导致查询错误的历史
   - 解决：严格管理 frame_id 和 timestamp

3. **错误 3：插值越界**
   - 问题：关键帧不存在时插值失败
   - 解决：检查关键帧存在性，fallback 到完整计算

---

## 八、文件清单

```
需要创建的文件：
  hugs/models/modules/temporal_memory.py
  hugs/models/modules/temporal_attention.py
  hugs/models/modules/temporal_accelerator.py
  hugs/losses/temporal_loss.py
  hugs/utils/temporal_evaluation.py
  cfg_files/temporal_attention.yaml

需要修改的文件：
  hugs/trainer/gs_trainer.py（集成时序模块）
  hugs/models/__init__.py（导出新模块）
  main.py（加载配置）
```

---

## 九、验收标准

### 9.1 功能验收

- [ ] Temporal Memory 能正确存储和查询
- [ ] Temporal Attention 能正确计算
- [ ] Accelerator 能智能选择策略
- [ ] Temporal Loss 能正确计算
- [ ] 训练流程能正常运行

### 9.2 性能验收

- [ ] PSNR 提升 ≥ 0.1 dB
- [ ] Foot Sliding < 1.0 cm/frame
- [ ] Contact Jitter < 2.0 mm
- [ ] Contact Consistency > 90%
- [ ] 训练加速 ≥ 2x

### 9.3 代码质量

- [ ] 有完整的单元测试
- [ ] 有详细的注释
- [ ] 符合项目代码规范
- [ ] 通过代码审查

---

## 十、参考资料

1. **UniCon3R**（CVPR 2026）：接触感知的时序人类场景重建
   - Temporal Momentum 机制
   - Contact-guided Latent Refinement

2. **TRiGS**（2026）：时序刚体运动的 4D Gaussian Splatting
   - Bézier Curve 运动平滑
   - Local Motion Coherence

3. **4DSTR**（AAAI 2025）：时序关联的 4D 生成
   - Temporal Buffer + Mamba
   - Scale/Rotation Residual

4. **ST-4DGS**（2024）：时空一致的 4DGS
   - Cubic Hermite Splines
   - Temporal Curvature Regularization

---

## 十一、联系与支持

如有问题，请参考：
- 本文档的设计原理部分
- 相关论文的原始代码
- 项目的 issue tracker

**祝开发顺利！** 🚀
