#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   ./project/second_run_nightly_best_config/launch.sh 0       # seed 0, 8 GPUs
#   ./project/second_run_nightly_best_config/launch.sh 0 2     # seed 0, 2 GPUs
#   NUM_GPUS=2 ./project/second_run_nightly_best_config/launch.sh 0
#   CUDA_VISIBLE_DEVICES=4,5 ./project/second_run_nightly_best_config/launch.sh 0 2
#   RUN_NAME=my_run DATA_ROOT=/tmp/runs ./project/second_run_nightly_best_config/launch.sh 0 2
#   MAX_MINIBATCH_SIZE=16000 ./project/second_run_nightly_best_config/launch.sh 0 2
#   DRY_RUN=1 ./project/second_run_nightly_best_config/launch.sh 0 2
#
# GPU scaling preserves the repository nightly recipe's 40 global environments
# and 256k global logical minibatch. MAX_MINIBATCH_SIZE controls the per-GPU
# memory/gradient-accumulation tradeoff without changing that logical minibatch.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${SCRIPT_DIR}/nightly_best.yaml"
SEED="${1:-0}"
GPU_COUNT="${2:-${NUM_GPUS:-8}}"
RUN_NAME="${RUN_NAME:-nightly_best_local_$(date +%Y-%m-%d_%H-%M-%S)_seed${SEED}}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/experiments/second_run_nightly_best_config}"

if (( $# > 2 )) || [[ ! "${SEED}" =~ ^[0-9]+$ ]] || [[ ! "${GPU_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 [non-negative-seed] [positive-gpu-count]" >&2
    exit 2
fi

GLOBAL_ENV_COUNT=40
GLOBAL_MINIBATCH_SIZE=256000
BPTT_HORIZON=128

if (( GLOBAL_ENV_COUNT % GPU_COUNT != 0 || GLOBAL_MINIBATCH_SIZE % GPU_COUNT != 0 )); then
    echo "GPU count ${GPU_COUNT} must divide ${GLOBAL_ENV_COUNT} and ${GLOBAL_MINIBATCH_SIZE}." >&2
    exit 2
fi

PER_RANK_ENV_COUNT=$((GLOBAL_ENV_COUNT / GPU_COUNT))
PER_RANK_MINIBATCH_SIZE=$((GLOBAL_MINIBATCH_SIZE / GPU_COUNT))
DEFAULT_MAX_MINIBATCH_SIZE=$((PER_RANK_MINIBATCH_SIZE / 2))
if (( DEFAULT_MAX_MINIBATCH_SIZE > 32000 )); then
    DEFAULT_MAX_MINIBATCH_SIZE=32000
fi
PER_RANK_MAX_MINIBATCH_SIZE="${MAX_MINIBATCH_SIZE:-${DEFAULT_MAX_MINIBATCH_SIZE}}"
if (( PER_RANK_ENV_COUNT % 2 == 0 )); then
    PER_RANK_VEC_BATCH_SIZE=$((PER_RANK_ENV_COUNT / 2))
else
    # Five environments on each of eight ranks cannot be split into equal
    # zero-copy batches, so process the complete per-rank pool together.
    PER_RANK_VEC_BATCH_SIZE=${PER_RANK_ENV_COUNT}
fi

if [[ ! "${PER_RANK_MAX_MINIBATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || (( PER_RANK_MINIBATCH_SIZE % PER_RANK_MAX_MINIBATCH_SIZE != 0 || PER_RANK_MAX_MINIBATCH_SIZE % BPTT_HORIZON != 0 )); then
    echo "Derived minibatch sizes are incompatible with gradient accumulation or BPTT horizon ${BPTT_HORIZON}." >&2
    exit 2
fi

if [[ ! -f "${CONFIG}" || ! -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    echo "Missing nightly_best.yaml or the project .venv." >&2
    exit 1
fi

if [[ "${DATA_ROOT}" != /* ]]; then
    DATA_ROOT="${REPO_ROOT}/${DATA_ROOT}"
fi
DATA_DIR="${DATA_ROOT}/${RUN_NAME}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"

CONFIG_OUTPUT="$(python "${SCRIPT_DIR}/yaml_overrides.py" "${CONFIG}")"
mapfile -t CONFIG_ARGS <<< "${CONFIG_OUTPUT}"
CONFIG_ARGS+=(
    "vec.num_envs=${PER_RANK_ENV_COUNT}"
    "vec.batch_size=${PER_RANK_VEC_BATCH_SIZE}"
    "train.minibatch_size=${PER_RANK_MINIBATCH_SIZE}"
    "train.max_minibatch_size=${PER_RANK_MAX_MINIBATCH_SIZE}"
    "train.seed=${SEED}"
    "train.data_dir=${DATA_DIR}"
    "run_name=${RUN_NAME}"
)

export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

COMMAND=(
    torchrun --standalone --nnodes=1 --nproc-per-node="${GPU_COUNT}" --max_restarts=0
    -m pufferlib.pufferl train puffer_drive
    "${CONFIG_ARGS[@]}"
)

echo "Config: ${CONFIG}"
echo "Run: ${RUN_NAME} (seed ${SEED}, GPUs ${GPU_COUNT})"
echo "Per rank: envs=${PER_RANK_ENV_COUNT}, vec_batch=${PER_RANK_VEC_BATCH_SIZE}, minibatch=${PER_RANK_MINIBATCH_SIZE}, max_minibatch=${PER_RANK_MAX_MINIBATCH_SIZE}"
echo "Output: ${DATA_DIR}"
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

VISIBLE_GPUS="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( VISIBLE_GPUS < GPU_COUNT )); then
    echo "Requested ${GPU_COUNT} CUDA GPUs, found ${VISIBLE_GPUS} visible." >&2
    exit 1
fi

mkdir -p "${DATA_DIR}"
exec "${COMMAND[@]}"
