#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/../.." && pwd)"

THREADS="${OMP_NUM_THREADS:-4}"
if [[ ! "${THREADS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "OMP_NUM_THREADS must be a positive integer." >&2
    exit 2
fi

if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    echo "Missing project venv: ${REPO_ROOT}/.venv/bin/python" >&2
    exit 1
fi

export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv/bin/activate"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exec python -m project.jepa_distill.evaluate_carla "$@" --dry-run
fi
if [[ "${DRY_RUN:-0}" != "0" ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 2
fi

exec python -m project.jepa_distill.evaluate_carla "$@"
