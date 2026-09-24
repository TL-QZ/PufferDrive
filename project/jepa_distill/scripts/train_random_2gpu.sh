#!/usr/bin/env bash
# Default to GPUs 2,3; GPUs 0,1 are occupied by the warm-start experiment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
exec "${SCRIPT_DIR}/train_2gpu.sh" --config project/jepa_distill/config/condition_b_random.yaml "$@"
