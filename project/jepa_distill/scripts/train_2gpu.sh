#!/usr/bin/env bash
# Full recipe by default; the user starts the official run explicitly.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
if [[ ! "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+,[0-9]+$ ]]; then
    echo 'CUDA_VISIBLE_DEVICES must specify exactly two GPU indices, e.g. 0,1.' >&2
    exit 1
fi
IFS=, read -r FIRST_GPU_IDX SECOND_GPU_IDX <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${FIRST_GPU_IDX}" == "${SECOND_GPU_IDX}" ]]; then
    echo 'Choose two different GPUs.' >&2
    exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
TRAIN_COMMAND=(torchrun --standalone --nnodes=1 --nproc_per_node=2
    -m project.jepa_distill.train "$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_VISIBLE_DEVICES}"
    printf '%q ' "${TRAIN_COMMAND[@]}"
    printf '\n'
    exit 0
fi
exec "${TRAIN_COMMAND[@]}"
