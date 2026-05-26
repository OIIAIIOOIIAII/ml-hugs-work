# Human3R 用于 HUGS 初始化的质量分析与后续方案

本文档记录当前将 Human3R 输出用于 HUGS 初始化时观察到的质量变化、可能原因，以及后续更合理的使用路线。

## 1. 背景

HUGS 原版流程中，人体和场景的初始化基本是分开进行的。我们的动机是利用 Human3R 输出中相对合理的人体、场景空间关系，让 HUGS 训练一开始就拥有更好的全局人-场景相对位置。

当前已经实现并测试过一条 Human3R 初始化路线：

- 使用 Human3R 输出的人体/场景信息构造 HUGS 可读数据。
- 将 Human3R 的人体参数拟合或转换到 HUGS 使用的 SMPL 表达。
- 在 HUGS 中用 Human3R 转换后的数据进行 human + scene 联合训练。
- 与原版 HUGS 在相同分辨率、相同步数下进行对比。

## 2. 实验对比结果

我们重点比较了两个 6000 step 结果：

1. 原版 HUGS 路线，但输入图像下采样到 Human3R 路线相同分辨率。
2. Human3R 初始化路线，使用相同分辨率和相同步数。

结果如下：

```text
原版 HUGS 同分辨率 6000 step:
PSNR:         26.3263
SSIM:         0.9208
LPIPS:        0.0392
Human PSNR:   19.2890
Human SSIM:   0.7276
Human LPIPS:  0.0843

Human3R 初始化 6000 step:
PSNR:         18.9311
SSIM:         0.6439
LPIPS:        0.1806
Human PSNR:   15.2528
Human SSIM:   0.5131
Human LPIPS:  0.1648
```

结论：在相同输入分辨率、相同训练步数下，Human3R 初始化路线仍然明显差于原版 HUGS 路线。因此当前质量下降不能简单归因于 Human3R 输出分辨率较低，也不能简单归因于训练步数不足。

## 3. 已排除或弱化的原因

### 3.1 输入分辨率不是主要原因

一开始怀疑 Human3R 输出分辨率较低限制了 HUGS 的人体细节恢复。为此我们将原版 HUGS 输入也下采样到 Human3R 路线相同分辨率，并跑了 6000 step 对比。

实验结果显示，即使分辨率一致，原版 HUGS 仍显著优于 Human3R 初始化路线。因此分辨率可能有影响，但不是当前质量差距的主因。

### 3.2 训练步数不是主要原因

原版 HUGS 曾跑过 15000 step，而 Human3R 初始化路线早期结果只跑过 9000 step，因此曾怀疑步数不足导致质量下降。

但在 6000 step 对齐实验中，原版 HUGS 仍明显更好。因此更多步数可能能改善 Human3R 路线，但大概率不能从根本上解决当前差距。

## 4. 当前观察到的关键现象

### 4.1 Human3R 的人-场景相对位置有价值

从点云检查看，Human3R 给出的人体与场景相对位置是有意义的。Human3R 的优势主要体现在：

- 人体和场景处在同一个合理坐标关系中。
- 人体整体朝向、尺度、深度有较好初值。
- 可以为 HUGS 提供全局空间对齐先验。

这说明 Human3R 并不是没有价值。问题更可能是我们把 Human3R 输出使用得过重，让它影响了 HUGS 原本擅长的人体初始化和人体外观优化。

### 4.2 Human3R 路线的人体高斯数量更少

导出的 posed human point cloud 统计显示：

```text
原版 HUGS 人体点数:   253,000
Human3R 人体点数:    164,286
```

Human3R 路线最终人体高斯数量明显更少。这可能意味着：

- 人体区域梯度不足。
- human densification 没有充分发生。
- 人体 mask、相机、SMPL 姿态或尺度存在偏差。
- 初始人体位置虽然全局合理，但与图像中的人体投影仍有误差，导致 photometric 优化效率下降。

人体高斯数量不足会直接影响衣服纹理、面部、手臂和身体轮廓等细节。

## 5. 可能导致质量下降的原因

### 5.1 SMPL-X 到 SMPL 的拟合损失

Human3R 输出更接近 SMPL-X 体系，而 HUGS 原版使用 SMPL。SMPL-X 的参数更多，包含更细的人体表达；但当前为了接入 HUGS，需要把它转换或拟合到 SMPL。

这个过程可能带来身体关节姿态误差、身体比例偏差、肩膀和手臂区域对齐不准，以及 global orientation、translation、scale 的小偏差。即使 Human3R 在自己的体系中看起来正确，转换到 HUGS 的 SMPL 后也可能变成一个对 HUGS 优化不友好的初值。

### 5.2 坐标、尺度和相机体系存在细微不一致

Human3R 和 HUGS 对世界坐标系方向、相机外参约定、人体 translation 参考点、scale 定义、SMPL body root 使用方式、scene point cloud 坐标尺度等内容的定义可能不同。

这些差异不一定会导致明显崩溃，但会造成投影层面的偏差。对于 HUGS 这种依赖图像 photometric loss 和 mask 的优化流程，几个像素到几十个像素的初始偏差就可能明显影响人体高斯增长和外观学习。

### 5.3 Human3R 初始化可能破坏 HUGS 原版人体先验

HUGS 原版的人体初始化、canonical 高斯、SMPL 参数和 NeuMan 数据格式是配套设计的。Human3R 初始化引入后，如果直接替代 HUGS 原本的人体状态，可能会破坏这些隐含先验。

表现为人体高斯从不合适的 canonical 或 posed 状态开始，SMPL 参数和 HUGS 模型内部假设不完全一致，后续优化需要先修正初始化偏差，再学习外观细节，导致 6000 或 9000 step 内无法恢复到原版 HUGS 的人体质量。

### 5.4 Human3R 的优势主要是全局对齐，不是 HUGS 人体外观初始化

Human3R 能帮助确定“人在哪里、朝向哪里、相对场景在哪里”。但 HUGS 的人体外观质量来自 HUGS 原版 canonical human gaussians、SMPL 驱动的人体形变、人体 mask 和 RGB photometric 优化、human densification，以及 HUGS 自己的人体先验和网络结构。

因此，直接让 Human3R 接管完整人体初始化，可能会拿到全局位置好处，但损失 HUGS 原本的人体细节恢复能力。

## 6. 当前判断

当前更可信的判断是：

```text
Human3R 初始化质量下降不是单一原因造成的。
主要问题可能来自 SMPL-X -> SMPL 拟合误差、坐标/尺度/相机体系不一致，
以及 Human3R 初始化过度干预了 HUGS 原版人体先验和 densification 流程。
```

Human3R 的人-场景相对位置仍然有价值，但它不适合直接完整替代 HUGS 的人体初始化。

## 7. 推荐后续路线

### 7.1 推荐方案 A：只使用 Human3R 的全局对齐信息

这是当前最推荐的方向。

核心思想：

```text
保留 HUGS 原版人体初始化质量；
只从 Human3R 中提取人体-场景的全局 Sim(3) 对齐信息。
```

具体做法：

1. 先按原版 HUGS 流程初始化人体高斯和场景高斯。
2. 从 Human3R 输出中估计人体和场景之间的全局位姿关系。
3. 将该全局 transform 应用到 HUGS 的人体整体位姿或场景坐标中。
4. 不直接用 Human3R 的 SMPL-X/SMPL 拟合结果替代 HUGS 原版人体状态。

优点：

- 最大程度保留 HUGS 原版人体重建质量。
- 利用 Human3R 最可靠的人-场景相对位置。
- 避免 SMPL-X 到 SMPL 细节拟合误差直接污染人体外观训练。

### 7.2 方案 B：严格重新设计 SMPL-X 到 SMPL 拟合

如果仍希望使用 Human3R 的人体参数初始化 HUGS，则不应做简单参数映射，而应做优化式拟合。

优化变量：

```text
SMPL betas
SMPL body_pose
SMPL global_orient
SMPL transl
SMPL scale
```

建议 loss：

```text
3D joints loss
body surface Chamfer loss
pose prior
shape prior
temporal smoothness
feet/contact consistency
camera reprojection loss
```

目标是让 SMPL 的身体关节点和表面点尽可能贴近 Human3R 的 SMPL-X 身体部分，同时保持 HUGS 可优化的合理人体先验。

### 7.3 方案 C：Human3R prior 逐步释放

Human3R 可以作为训练早期的 soft prior，但不应长期锁死人体。

可以采用：

```text
0 - 1000 step:
Human3R pose/transl/scale prior 权重较高

1000 step 之后:
逐渐降低 Human3R prior 权重，让 RGB/mask photometric loss 接管
```

这样做可以利用 Human3R 的初始对齐，同时允许 HUGS 自己修正 Human3R 或 SMPL-X -> SMPL 转换带来的偏差。

### 7.4 方案 D：加强 Human3R 路线的人体 densification

由于 Human3R 路线人体点数明显少于原版，可以尝试：

- 更早开启 human densification。
- 延长 human densification 区间。
- 调低人体 densification 阈值。
- 提高 human RGB/mask loss 权重。
- 检查人体 mask 是否与 Human3R 转换后的投影一致。
- 单独可视化每个训练阶段的人体高斯数量变化。

这条路线不能解决所有坐标和拟合问题，但可能改善人体细节不足。

## 8. 建议的下一步实验

优先级从高到低：

1. 用点云 PLY 验证 Human3R 人体与场景相对位置是否确实优于原版初始化。
2. 基于 Human3R 估计一个全局 Sim(3) transform，只用于移动 HUGS 原版人体或场景。
3. 使用原版 HUGS 人体初始化 + Human3R 全局对齐，跑 6000 step 对比。
4. 如果方案 3 明显改善位置且保留人体质量，再扩展到完整 15000 step。
5. 并行记录 human gaussian 数量、mask 投影误差、人体 bbox 投影偏差等中间指标。

## 9. 当前结论

Human3R 不应被简单视为 HUGS 的完整替代初始化。更合理的定位是：

```text
Human3R 提供全局人-场景相对位置先验；
HUGS 保留原版人体初始化、外观学习和 densification 流程。
```

这条路线最符合当前实验观察：Human3R 的相对位置有价值，但直接接管人体初始化会导致人体重建质量明显下降。

