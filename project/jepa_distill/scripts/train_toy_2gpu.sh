#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/train_2gpu.sh" --config project/jepa_distill/config/toy_2gpu.yaml "$@"
