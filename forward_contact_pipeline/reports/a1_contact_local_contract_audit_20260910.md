# A1：contact-local 坐标契约审计

## 目的

排除 Stage-A 通过绝对房间坐标、scene identity 或全局朝向记忆接触的捷径。所有
mesh/scene patch 特征须以每个脚的 surface tangent-normal frame 表达，训练输入中不
保留绝对 world coordinate。

## 审计对象与方法

只读取 NPZ 数值字段和 manifest；不读取 RGB。审计当前可用的
`stagea_relation_clean_64x128_v3`（12 sequence × 60 frame）及其同 schema 的退化
资产。检查 required feature、有限性、shape，以及 vertex / scene normal 的单位长度。

## 结果

- v3 clean 的 12/12 sequence payload 均含
  `roi_vertex_local`、`roi_vertex_normal_local`、`scene_point_local`、
  `scene_normal_local`、`scene_point_distance`、`vertex_proximity`、`vertex_contact`；
  shape 为 `[60, 2, 64, 3]` 的 ROI，并且所有检查字段 finite。
- v3 中非零 local normal 的最大单位长度误差不超过 `2.38e-7`。
- 旧 v1 的 `MPH16/MPH8/N3OpenArea/Werkraum` normal 误差约 `5.28`，正是此前发现的
  PLY 面积加权 normal 问题；它们不是可训练资产。v2 的零 normal 问题同样不回收。
- builder 只保存 local relative position / normal / distance；RGB 仅保留为路径引用，
  尚未读取或训练视觉编码器。故该结论只验证几何坐标契约，不是 RGB/Gaussian 前端结果。

## 结论

A1 对当前 PROX continuous-proximity data contract **通过**。它可作为未来真实 dense
dataset adapter 的不可变接口：新数据也必须把人体 ROI 与 scene patch 转入本地 frame，
并保存 topology/correspondence 与 label source。此通过不改变 A5 阻塞：PROX 仍不具备
可报告的 dense binary contact 正负分布，不能启动最终 contact head。
