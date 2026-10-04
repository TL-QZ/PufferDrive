#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/../.." && pwd)"

THREADS="${OMP_NUM_THREADS:-1}"
if [[ ! "${THREADS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "OMP_NUM_THREADS must be a positive integer." >&2
    exit 2
fi

if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    echo "Missing project venv: ${REPO_ROOT}/.venv/bin/python" >&2
    exit 1
fi

DRY_RUN_VALUE="${DRY_RUN:-0}"
if [[ "${DRY_RUN_VALUE}" != "0" && "${DRY_RUN_VALUE}" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 2
fi

export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv/bin/activate"

if [[ "${DRY_RUN_VALUE}" == "1" ]]; then
    exec python -m project.jepa_distill.render_carla "$@" --dry-run
fi

exec python -m project.jepa_distill.render_carla "$@"
