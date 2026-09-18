#!/usr/bin/env bash
set -euo pipefail

tag=research-point-1-gt-alignment-best-results
manifest=releases/research_point_1_gt_best_results/SHA256SUMS
notes=releases/research_point_1_gt_best_results/README.md
gh_bin=${GH_BIN:-}
if [[ ${1:-} == --verify-only ]]; then sha256sum -c "$manifest"; exit 0; fi
if [[ -z "$gh_bin" ]]; then
  if [[ -x .tools/bin/gh ]]; then gh_bin=.tools/bin/gh
  elif command -v gh >/dev/null; then gh_bin=$(command -v gh)
  else echo 'Install GitHub CLI and run: gh auth login' >&2; exit 2; fi
fi
"$gh_bin" auth status >/dev/null
sha256sum -c "$manifest"
stage=$(mktemp -d); trap 'rm -rf "$stage"' EXIT
while read -r _ path; do
  scene=$(awk -F/ '{for(i=1;i<=NF;i++) if($i=="neuman") {print $(i+1); exit}}' <<<"$path")
  case "$path" in
    */ckpt/scene_final.pth) name="${scene}_gt_scene_final.pth" ;;
    */ckpt/human_final.pth) name="${scene}_gt_human_final.pth" ;;
    */ckpt/anchor_attention_final.pth) name="${scene}_gt_anchor_attention_final.pth" ;;
    */config_train.yaml) name="${scene}_gt_config_train.yaml" ;;
    *) echo "Unexpected manifest path: $path" >&2; exit 2 ;;
  esac
  ln -s "$(realpath "$path")" "$stage/$name"
done < "$manifest"
"$gh_bin" release create "$tag" --target main --title 'Research Point 1: NeuMan GT-alignment best results' --notes-file "$notes" "$stage"/*
