# 研究点一：NeuMan GT 对齐最优结果发布清单

Release tag：`research-point-1-gt-alignment-best-results`。

本发布是研究点一的 **GT 对齐（无 VIMO 粗对齐）** 论文主表 Ours 路线：`depth_sup12k + AnchorAttention_6k` 两阶段 pipeline。它包含 bike、seattle、jogging、lab、parkinglot、citron 六个场景的最优已完成 run 的最终可加载状态：每场景的 `scene_final.pth`、`human_final.pth`、`anchor_attention_final.pth` 和 `config_train.yaml`，共24个文件、约9.3GB。

该路线的六场景 final HUMAN_PSNR 平均为 **19.6174**；逐场景为 bike 20.4297、seattle 19.6412、jogging 17.8027、lab 19.7831、parkinglot 19.8686、citron 20.1794。它与 VIMO v4 粗对齐路线是两套独立实验条件，必须分别使用对应的 Release。

下载后在仓库根目录执行：

```bash
sha256sum -c releases/research_point_1_gt_best_results/SHA256SUMS
```

验证通过后，将以 `<scene>_gt_scene_final.pth`、`<scene>_gt_human_final.pth`、`<scene>_gt_anchor_attention_final.pth` 和 `<scene>_gt_config_train.yaml` 命名的资产，按 `SHA256SUMS` 所示相对路径放回仓库根目录。
