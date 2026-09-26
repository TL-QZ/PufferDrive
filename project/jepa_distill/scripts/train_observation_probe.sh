#!/usr/bin/env bash
# Read-only preflight: train_observation_probe.sh --dry-run
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"
exec python -m project.jepa_distill.observation_probe.train "$@"
