#!/usr/bin/env bash
set -euo pipefail

# Train one third-run CARLA/Gigaflow seed on exactly two GPUs.
#
# Usage:
#   ./project/third_run_nightly_best_config/train/launch_2gpu.sh 0
#   CUDA_VISIBLE_DEVICES=4,5 ./project/third_run_nightly_best_config/train/launch_2gpu.sh 1
#   DRY_RUN=1 ./project/third_run_nightly_best_config/train/launch_2gpu.sh 2

if (( $# != 1 )) || [[ ! "$1" =~ ^[012]$ ]]; then
    echo "Usage: $0 <seed: 0|1|2>" >&2
    exit 2
fi

DRY_RUN="${DRY_RUN:-0}"
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 2
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
if [[ ! "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+,[0-9]+$ ]]; then
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
CONFIG="${PROJECT_DIR}/override_config/nightly_best.yaml"
DEFAULT_RUN_NAME="nightly_best_local_2gpu_$(date +%Y-%m-%d_%H-%M-%S)_seed${SEED}"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
EXPERIMENT_ROOT="${REPO_ROOT}/experiments/third_run_nightly_best_config"
DATA_DIR="${EXPERIMENT_ROOT}/${RUN_NAME}"

if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_NAME may contain only letters, digits, dots, underscores, and hyphens." >&2
    exit 2
fi
if [[ ! -f "${CONFIG}" || ! -f "${PROJECT_DIR}/yaml_overrides.py" || ! -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    echo "Missing third-run config, yaml_overrides.py, or project .venv." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"

mapfile -t CONFIG_ARGS < <(python "${PROJECT_DIR}/yaml_overrides.py" "${CONFIG}")
CONFIG_ARGS+=(
    "vec.num_envs=20"
    "vec.batch_size=10"
    "train.minibatch_size=128000"
    "train.max_minibatch_size=32000"
    "train.seed=${SEED}"
    "train.data_dir=${DATA_DIR}"
    "run_name=${RUN_NAME}"
)

TRAIN_COMMAND=(
    torchrun --standalone --nnodes=1 --nproc-per-node=2 --max_restarts=0
    -m pufferlib.pufferl train puffer_drive "${CONFIG_ARGS[@]}"
)

printf 'Run:        %s (seed %s)\n' "${RUN_NAME}" "${SEED}"
printf 'GPUs:       %s\n' "${CUDA_VISIBLE_DEVICES}"
printf 'Config:     %s\n' "${CONFIG}"
printf 'Output:     %s\n' "${DATA_DIR}"
printf 'Per rank:   envs=20, vec_batch=10, minibatch=128000, max_minibatch=32000\n'
printf 'Command:'
printf ' %q' "${TRAIN_COMMAND[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
fi

VISIBLE_GPUS="$(python -c 'import torch; print(torch.cuda.device_count())')"
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
