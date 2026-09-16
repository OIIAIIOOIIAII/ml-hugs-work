# 训练架构与Git迁移准备（2026-09-14）

## 结论

已有GRU/TCN、Stage-B因果rollout和Stage-A几何原型，但此前缺统一可复现训练层。
本轮补齐可执行的数值缓存Stage-A工程：配置继承/覆盖与独立本机路径、缓存分片和
数据审计、可替换数据/模型/损失factory、geometry与RGB/Gaussian融合基线、掩码损失、
聚合和逐区域指标、验证选模、显式测试、epoch边界断点恢复及实验快照。

现有原型未删除或重写。真实RICH原始字段/同帧顶点标签到缓存的exporter、真实前端
精度审计和GUSH3R完整LBS binding仍未完成；该版本不是最终系统结果，也未启动真实训练。

## 具体交付

- `contact_streaming/training/{config,data,components,engine}.py`。
- `scripts/run_experiment.py`：preflight/train/evaluate，训练不读取test指标。
- `scripts/build_training_index.py`：检查明确对齐的input/target分片、划分身份和SHA256；不自动生成已通过几何审计。
- `configs/experiments/{base,rich_rgb_gaussian,smoke}.yaml`与`configs/paths.example.yaml`。
- `pyproject.toml`与`requirements-training.txt`：单独安装数值训练包，不合并HUGS/GUSH3R GPU环境。
- `TRAINING.md`、根目录`MIGRATION.md`、`scripts/audit_git_migration.py`。
- `migration/dependencies.json`固定Human3R/GUSH3R/两个CUDA子模块版本；GUSH3R五个本地修改保存成补丁。
- `.gitignore`隔离数据/权重/凭据/临时产物；公共RICH清单迁至`configs/data`，download脚本在无私有manifest时使用它。

## 验证

- 既有4个核心测试、4个下载HTTP测试、新增7个训练测试全部通过；另有1个迁移回归通过，共16项。
- CPU上完整两epoch与一epoch后恢复续训的模型权重逐Tensor相同、history相同。
- 覆盖跨scene/sequence/subject泄漏、过时审计/变更payload、单类标签拒绝、掩码标签不影响损失、RGB/Gaussian通道、分片采样每样本恰好一次、显式测试隔离。
- 仅复制Git候选源码到独立目录，从项目外CWD执行fixture生成/preflight/train/resume/evaluate均通过；无原有数据或私有manifest时下载status也通过。初次演练发现公共data配置被旧gitignore遗漏，已修复并增加回归。
- GUSH3R补丁在指定base的五个源文件上apply/check成功，重建文件与现有改动逐字节相同。
- wheel成功构建，26个条目，含training模块，无数据/图像/checkpoint混入。
- Git tracked diff whitespace检查通过；没有git add/commit/push，没有修改现有训练数据或下载任务。

## Git迁移边界

本次审查快照包含477个源码/文档候选，其中340个未跟踪；新代码尚不在远端。现有Git历史包括原项目teaser等资产，忽略规则不会自动清除历史。源码候选检查没有发现所检查模式的凭据，不等于完整历史安全审计。

历史28个脚本含机器绝对路径，完整文件与行号在`git_migration_20260914/audit.json`；新训练入口用paths配置，旧脚本迁移按实际使用逐项适配。尚未在目标机器测试GPU/驱动、renderer与CUDA扩展。

推荐主仓库继续保存自有代码/配置/测试/必要文档，第三方版本和修改另有可重放记录；数据、SMPL模型、checkpoint和特征缓存走NAS/受控文件传输。远端发布前按source.paths选择性审阅，不能盲目git add整个工作区。

## 下一步

下载完成并通过归档校验后，先完成RICH短序列帧/相机/顶点/坐标与标签审计、真实前端缓存export，产生版本化index及独立审计报告。通过后运行preflight，再启动首轮Stage-A。
新算法以新组件/新配置/新run承载，科学配置改变不复用旧run的resume；仅跨机器迁移同一实验允许更换本机路径。
