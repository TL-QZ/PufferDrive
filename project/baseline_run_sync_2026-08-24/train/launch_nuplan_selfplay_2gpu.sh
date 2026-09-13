#!/usr/bin/env bash
set -euo pipefail

# Train one synced-baseline nuPlan self-play seed on exactly two GPUs.
#
# Usage:
#   ./project/baseline_run_sync_2026-08-24/train/launch_nuplan_selfplay_2gpu.sh 0
#   CUDA_VISIBLE_DEVICES=4,5 ./project/baseline_run_sync_2026-08-24/train/launch_nuplan_selfplay_2gpu.sh 1
#   DRY_RUN=1 ./project/baseline_run_sync_2026-08-24/train/launch_nuplan_selfplay_2gpu.sh 2

if (( $# != 1 )) || [[ ! "$1" =~ ^[012]$ ]]; then
    echo "Usage: $0 <seed: 0|1|2>" >&2
    exit 2
fi

DRY_RUN="${DRY_RUN:-0}"
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 2
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
if [[ ! "${CUDA_VISIBLE_DEVICES}" =~ ^(0|[1-9][0-9]*),(0|[1-9][0-9]*)$ ]]; then
    echo "CUDA_VISIBLE_DEVICES must name exactly two comma-separated GPU indices." >&2
    exit 2
fi
IFS=, read -r FIRST_GPU_IDX SECOND_GPU_IDX <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${FIRST_GPU_IDX}" == "${SECOND_GPU_IDX}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must name two different GPU indices." >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES

SEED="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/../.." && pwd)"
CONFIG="${PROJECT_DIR}/override_config/nuplan_selfplay.yaml"
RUN_NAME_PREFIX="baseline_run_sync_2026-08-24_nuplan_selfplay_dt03"
DEFAULT_RUN_NAME="${RUN_NAME_PREFIX}_$(date +%Y-%m-%d_%H-%M-%S)_seed${SEED}"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
EXPERIMENT_ROOT="${REPO_ROOT}/experiments/baseline_run_sync_2026-08-24/nuplan_selfplay_dt03"
DATA_DIR="${EXPERIMENT_ROOT}/${RUN_NAME}"

if [[ ! "${RUN_NAME}" =~ ^${RUN_NAME_PREFIX}_[A-Za-z0-9._-]+_seed${SEED}$ ]]; then
    echo "RUN_NAME must start with ${RUN_NAME_PREFIX}_ and end with _seed${SEED}; the middle may contain only letters, digits, dots, underscores, and hyphens." >&2
    exit 2
fi
if [[ ! -f "${CONFIG}" || ! -f "${SCRIPT_DIR}/selfplay_config.py" || ! -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    echo "Missing self-play config, selfplay_config.py, or project .venv." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"

# Capture first: configuration failures must stop the launcher, including dry runs.
CONFIG_OUTPUT="$(python "${SCRIPT_DIR}/selfplay_config.py" --seed "${SEED}" --run-dir "${DATA_DIR}")"
mapfile -t CONFIG_ARGS <<< "${CONFIG_OUTPUT}"

TRAIN_COMMAND=(
    torchrun --standalone --nnodes=1 --nproc-per-node=2 --max_restarts=0
    -m pufferlib.pufferl train puffer_drive "${CONFIG_ARGS[@]}"
)

printf 'Run:        %s (seed %s)\n' "${RUN_NAME}" "${SEED}"
printf 'GPUs:       %s\n' "${CUDA_VISIBLE_DEVICES}"
printf 'Config:     %s\n' "${CONFIG}"
printf 'Output:     %s\n' "${DATA_DIR}"
if [[ -f "${DATA_DIR}/trainer_state.pt" ]]; then
    printf 'Resume:     %s/trainer_state.pt\n' "${DATA_DIR}"
else
    printf 'Resume:     none (random weights, fresh optimizer and counters)\n'
fi
printf 'Command:'
printf ' %q' "${TRAIN_COMMAND[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
fi

VISIBLE_GPUS="$(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"
if (( VISIBLE_GPUS != 2 )); then
    echo "Expected exactly two visible CUDA GPUs, found ${VISIBLE_GPUS}." >&2
    exit 1
fi

export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

mkdir -p "${DATA_DIR}"
exec "${TRAIN_COMMAND[@]}"
