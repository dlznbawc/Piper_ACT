#!/usr/bin/env bash
# 训练 ACT：scripts/train.sh [数据集目录]（默认 datasets/charge_align）
# 可用环境变量覆盖：DEVICE=cuda|cpu  STEPS=100000  BATCH=8
# 输出 outputs/train/act_<数据集名>/，已存在会报错（换名或删掉旧的）。
set -euo pipefail
cd "$(dirname "$0")/.."
DS=${1:-datasets/charge_align}
NAME=$(basename "$DS")
export PYTHONNOUSERSITE=1   # ~/.local 里的包（transformers 等）会串进 conda 环境导致 import 失败

lerobot-train \
  --dataset.repo_id="local/$NAME" --dataset.root="$DS" \
  --policy.type=act --policy.push_to_hub=false --policy.device="${DEVICE:-cuda}" \
  --output_dir="outputs/train/act_$NAME" --job_name="act_$NAME" \
  --batch_size="${BATCH:-8}" --steps="${STEPS:-100000}" --save_freq=20000 \
  --wandb.enable=false
