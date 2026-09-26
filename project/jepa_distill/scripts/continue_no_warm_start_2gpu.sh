#!/usr/bin/env bash
# Inspect with DRY_RUN=1. CLEANUP_CACHED=1 authorizes deleting all non-active training caches on launch.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"
RUN_ROOT="experiments/jepa_distill/runs/condition_b_distill_no_warm_start"
RESUME_CONFIG="project/jepa_distill/config/continue_no_warm_start.yaml"
if [[ "$#" != 0 ]]; then
    echo 'This launcher preserves the saved recipe. Set CUDA_VISIBLE_DEVICES, DRY_RUN, or CLEANUP_CACHED as needed.' >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
if [[ ! "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+,[0-9]+$ ]]; then
    echo 'CUDA_VISIBLE_DEVICES must specify exactly two different GPU indices.' >&2
    exit 1
fi
IFS=, read -r FIRST_GPU_IDX SECOND_GPU_IDX <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${FIRST_GPU_IDX}" == "${SECOND_GPU_IDX}" ]]; then
    echo 'Choose two different GPUs.' >&2
    exit 1
fi
PREPARE_COMMAND=(python -m project.jepa_distill.prepare_resume
    --config "${RESUME_CONFIG}" --checkpoint "${RUN_ROOT}/checkpoint.pt")
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    PREPARE_COMMAND+=(--require-ready)
    if [[ "${CLEANUP_CACHED:-0}" == 1 ]]; then
        PREPARE_COMMAND+=(--cleanup-cached)
    fi
fi
"${PREPARE_COMMAND[@]}"
exec "${REPO_ROOT}/project/jepa_distill/scripts/train_2gpu.sh" \
    --config "${RESUME_CONFIG}" \
    --run-id condition_b_distill_no_warm_start \
    --set "training.resume_checkpoint=${RUN_ROOT}/checkpoint.pt"
