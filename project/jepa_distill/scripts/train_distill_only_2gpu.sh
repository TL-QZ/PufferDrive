#!/usr/bin/env bash
# Fresh ablation; DRY_RUN=1 prints the launch command without starting training.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec "${REPO_ROOT}/project/jepa_distill/scripts/train_2gpu.sh" \
    --config project/jepa_distill/config/condition_b_distill_only.yaml "$@"
