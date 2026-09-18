# 研究点一：NeuMan v4 最终场景结果发布清单

Release tag：`research-point-1-v4-final-results`。

本发布包含 bike、seattle、jogging、lab、parkinglot、citron 六个 NeuMan v4 场景的最终可加载结果：每场景的 `scene_final.pth`、`human_final.pth`、`anchor_attention_final.pth` 和 `config_train.yaml`，共 24 个文件、约 9.3GB。它不包含中间 checkpoint、渲染图片、debug、训练日志或数据集。

每个文件均小于 GitHub Release 的 2GB 单资产限制。下载后在仓库根目录执行：

```bash
sha256sum -c releases/research_point_1_v4_final_results/SHA256SUMS
```

验证通过后，将 Release 中以 `<scene>_scene_final.pth`、`<scene>_human_final.pth`、`<scene>_anchor_attention_final.pth` 和 `<scene>_config_train.yaml` 命名的资产按 `SHA256SUMS` 内的相对路径放回仓库根目录，即可恢复六个场景的最终结果。Anchor Attention 副本也已作为小型模块权重在 `model_weights/research_point_1/` 中跟踪；Release 中的副本使每个场景结果包自包含。

发布前必须运行 `scripts/upload_research_point_1_v4_release.sh --verify-only`。实际上传要求 GitHub API 写权限；本机 deploy key 仅能推送 Git，不能替代该登录权限。
