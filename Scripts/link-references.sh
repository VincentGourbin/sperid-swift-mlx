#!/usr/bin/env bash
# Crée reference/ : liens symboliques vers les dépôts frères (sources à copier) et clones
# superficiels des références Python. Tout est gitignoré ; commits épinglés dans plan/12-ecosysteme.md.
set -euo pipefail
cd "$(dirname "$0")/.."
DEV="${SPERID_SIBLINGS_DIR:-$(cd .. && pwd)}"
mkdir -p reference
link() { [ -e "reference/$1" ] || ln -s "$DEV/$2" "reference/$1"; }
link flux-2   flux-2-swift-mlx
link yue2     YuE2-mlx-swift
link gemma-4  gemma-4-swift-mlx
link ltx      ltx-video-swift-mlx
link profiler swift-mlx-profiler
clone() {  # clone <dir> <url> <commit>
  if [ ! -d "reference/$1" ]; then
    git clone --quiet "$2" "reference/$1"
    git -C "reference/$1" checkout --quiet "$3"
  fi
  echo "$1 @ $(git -C "reference/$1" rev-parse --short HEAD)"
}
clone idol-sdk https://github.com/speridlabs/idol-sdk.git 8db7003
# DINOv3 (modeling_dinov3_vit.py) : seule la référence transformers fait foi, pas la mémoire.
if [ ! -d reference/transformers ]; then
  git clone --quiet --depth 1 --filter=blob:none --sparse https://github.com/huggingface/transformers.git reference/transformers
  git -C reference/transformers sparse-checkout set src/transformers/models/dinov3_vit
fi
echo "transformers @ $(git -C reference/transformers rev-parse --short HEAD)"
ls -l reference
