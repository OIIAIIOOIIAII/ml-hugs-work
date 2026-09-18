# 本项目训练权重

本目录只保存本项目新增且体积较小的研究模块权重，供另一台机器 clone 后直接取得。它不包含训练数据、SMPL/SMPL-X、上游 GUSH3R 权重、RICH cache，也不包含研究点一按场景生成的 human/scene Gaussian checkpoint。

| 研究点 | 文件 | 内容 | 大小 | SHA256 |
|---|---|---|---:|---|
| 一：Anchor Attention（粗对齐） | `research_point_1/*.pth` | VIMO v4 六场景 final Anchor Attention state dict；每个72 tensors | 约0.87MB/个 | 见下表 |
| 一：Anchor Attention（GT 对齐） | `research_point_1_gt/*.pth` | GT 对齐六场景最优 run 的 Anchor Attention state dict；每个72 tensors | 约0.71MB/个 | 见下表 |
| 一：Anchor Attention（粗对齐峰值） | `research_point_1_vimo_v4_peak/*.pth` | parkinglot VIMO v4 step-11k peak state dict | 872,877 bytes | 见下表 |
| 二：RICH Stage-A Contact | `research_point_2/rich_full_contact_v1_seed42_v2_best.pt` | epoch-5 best checkpoint，含模型、optimizer、随机状态和训练状态；`state.epoch=6` 表示下一训练轮从 6 开始 | 6,577,126 bytes | `645ef6e02c075a4da5578bffd61bea0561f26cf5cb46f613cd76568dc5b87406` |

## 研究点一：每场景模块

| 场景 | 文件 | SHA256 |
|---|---|---|
| bike | `research_point_1/bike_vimo_v4_anchor_attention_final.pth` | `72c6c84dde76425d354ee9a5762d9a77b0b399c3eb6a9fcfd82db34cf29d162d` |
| seattle | `research_point_1/seattle_vimo_v4_anchor_attention_final.pth` | `51cf7e45b299dbed7419d24ce670ac0dbeca0cfc0a5b46ea393ce2d493c18923` |
| jogging | `research_point_1/jogging_vimo_v4_anchor_attention_final.pth` | `ec85944e2b49356aae0d1eb8d33a47331fe6c6104f022c40be0f4200d00d3975` |
| lab | `research_point_1/lab_vimo_v4_anchor_attention_final.pth` | `84173836350397df25bb8fa510ef4feb6c8ef00341dd66b0501ea70df1510c39` |
| parkinglot | `research_point_1/parkinglot_vimo_v4_anchor_attention_final.pth` | `297edc0f5a033f50d9300fac397f04fff257ea92aa3f37b2a392a90b7b1e3d4e` |
| citron | `research_point_1/citron_vimo_v4_anchor_attention_final.pth` | `2a7d5dd388483551e5192613181802f012b5d0ad12e24507002da404b26f01be` |

## 研究点一：GT 对齐模块与粗对齐峰值模块

| 路线 / 场景 | 文件 | SHA256 |
|---|---|---|
| GT 对齐 / bike | `research_point_1_gt/bike_gt_anchor_attention_final.pth` | `b397a40c5a323e11c8f8af33c2c4a019b0dbefd17b3ab349a4bf8fa54894481b` |
| GT 对齐 / seattle | `research_point_1_gt/seattle_gt_anchor_attention_final.pth` | `4952a307712ddbfef48f53cb7855e08bfa88bbc1e14347221a7d5d344c2b611d` |
| GT 对齐 / jogging | `research_point_1_gt/jogging_gt_anchor_attention_final.pth` | `32fdd84ebf79e19e7d34a0c50b57f457b208f9e2cedbd29142044cea8909d521` |
| GT 对齐 / lab | `research_point_1_gt/lab_gt_anchor_attention_final.pth` | `03f81f4d21bdfe7005cc7e91adc9bcf53f66e17254521b06e8eeab3aa98052b2` |
| GT 对齐 / parkinglot | `research_point_1_gt/parkinglot_gt_anchor_attention_final.pth` | `71390d2a3fa95726638f1ed8da4575f5fd414d9348d5b2a2f95ca5d460b548e4` |
| GT 对齐 / citron | `research_point_1_gt/citron_gt_anchor_attention_final.pth` | `4591f7b4c4a728845c1a410205932cd53166fa5d2f2103cf2a4a2190a309cef9` |
| VIMO v4 / parkinglot step 11k peak | `research_point_1_vimo_v4_peak/parkinglot_vimo_v4_peak_011000_anchor_attention.pth` | `ad29be57d9e07f16c5019a7a74739e827f75b6842bfc43d00783b206eb9e60ec` |

parkinglot VIMO v4 的 11k 峰值 HUMAN_PSNR=16.7302，明显高于18k final=15.2645；加载该场景最佳结果时，必须把此模块和同一 step 的 scene/human checkpoint 一起使用。

研究点一需要与对应场景的 HUGS 配置和 human/scene Gaussian checkpoint 配套使用；后两者较大且是每场景重建产物，仍按 [MIGRATION.md](../MIGRATION.md) 受控迁移。

研究点二使用 [`forward_contact_pipeline/scripts/evaluate_rich_full_contact.py`](../forward_contact_pipeline/scripts/evaluate_rich_full_contact.py) 复评时，可把 `--checkpoint` 指向本目录的 `.pt`。它仍要求本地存在具有相同 contract 的 RICH plan/cache；不允许拿不同数据、cache 或前端版本强行恢复。
