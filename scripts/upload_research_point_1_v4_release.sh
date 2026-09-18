#!/usr/bin/env bash
set -euo pipefail

tag=research-point-1-v4-final-results
manifest=releases/research_point_1_v4_final_results/SHA256SUMS
notes=releases/research_point_1_v4_final_results/README.md

if [[ ${1:-} == --verify-only ]]; then
  sha256sum -c "$manifest"
  exit 0
fi

command -v gh >/dev/null || { echo 'Install GitHub CLI and run: gh auth login' >&2; exit 2; }
gh auth status >/dev/null
sha256sum -c "$manifest"

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
while read -r _ path; do
  scene=$(awk -F/ '{for(i=1;i<=NF;i++) if($i=="neuman") {print $(i+1); exit}}' <<<"$path")
  case "$path" in
    */ckpt/scene_final.pth) name="${scene}_scene_final.pth" ;;
    */ckpt/human_final.pth) name="${scene}_human_final.pth" ;;
    */ckpt/anchor_attention_final.pth) name="${scene}_anchor_attention_final.pth" ;;
    */config_train.yaml) name="${scene}_config_train.yaml" ;;
    *) echo "Unexpected manifest path: $path" >&2; exit 2 ;;
  esac
  ln -s "$(realpath "$path")" "$stage/$name"
done < "$manifest"
gh release create "$tag" --target main --title 'Research Point 1: NeuMan v4 final scene results' --notes-file "$notes" "$stage"/*
