#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
exec "${SCRIPT_DIR}/train_2gpu.sh" --config project/jepa_distill/config/toy_2gpu.yaml \
    --set model.encoder_initialization=random \
    --set wandb.group=two-gpu-random-encoder-toy "$@"
