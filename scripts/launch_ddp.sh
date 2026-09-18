#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/pretrain_120m.yaml}"
GPU_COUNT="${GPU_COUNT:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}"
RUN_DIR="${RUN_DIR:-runs/ddp-${GPU_COUNT}gpu}"

python scripts/collect_environment.py --output "${RUN_DIR}/environment.json"
torchrun --standalone --nproc-per-node="${GPU_COUNT}" \
  -m miniscale.train --config "${CONFIG}" --run-dir "${RUN_DIR}"

