#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"
exec python -m project.jepa_distill.train "$@"
