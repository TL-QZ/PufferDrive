#!/usr/bin/env bash
set -euo pipefail

# Start or resume a fresh two-GPU nuPlan SDC fine-tune from one third-run CARLA seed.
# Source trainer state is never staged; only this new run's trainer_state.pt may resume.
#
# Usage:
#   ./project/third_run_nightly_best_config/train/launch_nuplan_sdc_finetune_2gpu.sh 0
#   CUDA_VISIBLE_DEVICES=4,5 ./project/third_run_nightly_best_config/train/launch_nuplan_sdc_finetune_2gpu.sh 1
#   DRY_RUN=1 ./project/third_run_nightly_best_config/train/launch_nuplan_sdc_finetune_2gpu.sh 2

if (( $# != 1 )) || [[ ! "$1" =~ ^[012]$ ]]; then
    echo "Usage: $0 <source-seed: 0|1|2>" >&2
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

SOURCE_SEED="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/../.." && pwd)"
CONFIG="${PROJECT_DIR}/override_config/nuplan_sdc_finetune.yaml"
EXPERIMENT_ROOT="${REPO_ROOT}/experiments/third_run_nightly_best_config"
FINETUNE_ROOT="${EXPERIMENT_ROOT}/nuplan_sdc_finetune"
DEFAULT_RUN_NAME="nuplan_sdc_finetune_$(date +%Y-%m-%d_%H-%M-%S)_seed${SOURCE_SEED}"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"

if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_NAME may contain only letters, digits, dots, underscores, and hyphens." >&2
    exit 2
fi
if [[ ! -f "${CONFIG}" || ! -f "${PROJECT_DIR}/yaml_overrides.py" || ! -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    echo "Missing third-run config, yaml_overrides.py, or project .venv." >&2
    exit 1
fi

shopt -s nullglob
SOURCE_RUN_DIRS=("${EXPERIMENT_ROOT}"/nightly_best_local_2gpu_*_seed"${SOURCE_SEED}")
shopt -u nullglob
if (( ${#SOURCE_RUN_DIRS[@]} != 1 )); then
    echo "Expected exactly one third-run CARLA source run for seed ${SOURCE_SEED}; found ${#SOURCE_RUN_DIRS[@]}." >&2
    exit 1
fi

SOURCE_RUN_DIR="${SOURCE_RUN_DIRS[0]}"
SOURCE_MODEL_PATH="${SOURCE_RUN_DIR}/final_model.pt"
SOURCE_CONFIG_PATH="${SOURCE_RUN_DIR}/config.yaml"
if [[ ! -f "${SOURCE_MODEL_PATH}" ]]; then
    echo "Missing source checkpoint: ${SOURCE_MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -f "${SOURCE_CONFIG_PATH}" ]]; then
    echo "Missing source configuration: ${SOURCE_CONFIG_PATH}" >&2
    exit 1
fi

RUN_DIR="${FINETUNE_ROOT}/${RUN_NAME}"
INITIAL_CHECKPOINT_DIR="${RUN_DIR}/initial_checkpoint"
INITIAL_MODEL_PATH="${INITIAL_CHECKPOINT_DIR}/final_model.pt"
INITIAL_CONFIG_PATH="${INITIAL_CHECKPOINT_DIR}/config.yaml"
RESUME_STATE_PATH="${RUN_DIR}/trainer_state.pt"

if [[ -e "${INITIAL_CHECKPOINT_DIR}/trainer_state.pt" ]]; then
    echo "Refusing initial checkpoint bundle containing trainer_state.pt: ${INITIAL_CHECKPOINT_DIR}" >&2
    exit 1
fi
if [[ -f "${RESUME_STATE_PATH}" ]]; then
    if [[ ! -f "${INITIAL_MODEL_PATH}" || ! -f "${INITIAL_CONFIG_PATH}" ]]; then
        echo "Existing run is missing its staged initial checkpoint bundle: ${INITIAL_CHECKPOINT_DIR}" >&2
        exit 1
    fi
    if ! cmp -s -- "${SOURCE_MODEL_PATH}" "${INITIAL_MODEL_PATH}" \
        || ! cmp -s -- "${SOURCE_CONFIG_PATH}" "${INITIAL_CONFIG_PATH}"; then
        echo "Existing run's initial checkpoint does not match source seed ${SOURCE_SEED}." >&2
        exit 1
    fi
fi

# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"

mapfile -t CONFIG_ARGS < <(python "${PROJECT_DIR}/yaml_overrides.py" "${CONFIG}")
CONFIG_ARGS+=(
    "load_model_path=${INITIAL_MODEL_PATH}"
    "train.seed=${SOURCE_SEED}"
    "train.data_dir=${RUN_DIR}"
    "run_name=${RUN_NAME}"
)
if [[ -f "${RESUME_STATE_PATH}" ]]; then
    CONFIG_ARGS+=("train.resume_state_path=${RESUME_STATE_PATH}")
fi

TRAIN_COMMAND=(
    torchrun --standalone --nnodes=1 --nproc-per-node=2 --max_restarts=0
    -m pufferlib.pufferl train puffer_drive "${CONFIG_ARGS[@]}"
)

printf 'Source run: %s\n' "${SOURCE_RUN_DIR}"
printf 'Run:        %s\n' "${RUN_DIR}"
printf 'GPUs:       %s\n' "${CUDA_VISIBLE_DEVICES}"
printf 'Stage model:  %s -> %s\n' "${SOURCE_MODEL_PATH}" "${INITIAL_MODEL_PATH}"
printf 'Stage config: %s -> %s\n' "${SOURCE_CONFIG_PATH}" "${INITIAL_CONFIG_PATH}"
if [[ -f "${RESUME_STATE_PATH}" ]]; then
    printf 'Resume:     %s\n' "${RESUME_STATE_PATH}"
else
    printf 'Resume:     none (fresh optimizer, scheduler, epoch, and global-step counters)\n'
fi
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

mkdir -p "${INITIAL_CHECKPOINT_DIR}"
if [[ ! -f "${RESUME_STATE_PATH}" ]]; then
    cp -- "${SOURCE_MODEL_PATH}" "${INITIAL_MODEL_PATH}"
    cp -- "${SOURCE_CONFIG_PATH}" "${INITIAL_CONFIG_PATH}"
fi

export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

exec "${TRAIN_COMMAND[@]}"
