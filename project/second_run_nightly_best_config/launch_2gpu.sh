#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./project/second_run_nightly_best_config/launch_2gpu.sh 0
#   ./project/second_run_nightly_best_config/launch_2gpu.sh 1
#   CUDA_VISIBLE_DEVICES=4,5 ./project/second_run_nightly_best_config/launch_2gpu.sh 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SEED="${1:-0}"
RUN_NAME="nightly_best_local_2gpu_$(date +%Y-%m-%d_%H-%M-%S)_seed${SEED}"
DATA_DIR="${REPO_ROOT}/experiments/second_run_nightly_best_config/${RUN_NAME}"

if (( $# > 1 )) || [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 [non-negative-seed]" >&2
    exit 2
fi

# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"

mapfile -t CONFIG_ARGS < <(python "${SCRIPT_DIR}/yaml_overrides.py" "${SCRIPT_DIR}/nightly_best.yaml")
CONFIG_ARGS+=(
    "vec.num_envs=20"
    "vec.batch_size=10"
    "train.minibatch_size=128000"
    "train.max_minibatch_size=32000"
    "train.seed=${SEED}"
    "train.data_dir=${DATA_DIR}"
    "run_name=${RUN_NAME}"
)

export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

mkdir -p "${DATA_DIR}"
exec torchrun --standalone --nnodes=1 --nproc-per-node=2 --max_restarts=0 \
    -m pufferlib.pufferl train puffer_drive "${CONFIG_ARGS[@]}"
