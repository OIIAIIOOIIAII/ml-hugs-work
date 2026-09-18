# 研究点一：VIMO v4 parkinglot 峰值 checkpoint

此目录记录 VIMO v4 粗对齐路线中 parkinglot 的唯一显著“峰值优于 final”例外。`v4_correct_inline_attn_18k` 在 step 11,000 的 HUMAN_PSNR 为 **16.7302**，而 step 18k final 为 15.2645。因此完整复刻最佳结果时应使用下列三件套，而不是该场景的 final 三件套：`scene_011000.pth`、`human_011000.pth`、`anchor_attention_011000.pth`。

这些文件将作为 `research-point-1-v4-final-results` Release 的额外资产发布，命名为 `parkinglot_vimo_v4_peak_011000_{scene,human,anchor_attention}.pth`。`config_train.yaml` 与该 Release 已有的 parkinglot config 相同。
