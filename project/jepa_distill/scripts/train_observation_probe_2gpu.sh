#!/usr/bin/env bash
# Supply CUDA_VISIBLE_DEVICES explicitly. Budgets remain in the selected YAML.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to two available GPUs}"
exec torchrun --standalone --nproc_per_node=2 \
  -m project.jepa_distill.observation_probe.train "$@"
