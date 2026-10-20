#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
source .venv/bin/activate

RUN_DIR=""
SEED=""
SEED_SET=0
GPUS=0,1
DRY_RUN=0
SMOKE=0
NO_WANDB=0
SOURCE_CONFIG="${REPO_ROOT}/project/soft_actor_critic_warm_start/config/warm_start.yaml"
while (($#)); do
    case "$1" in
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --config) SOURCE_CONFIG="$2"; shift 2 ;;
        --seed) SEED="$2"; SEED_SET=1; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --smoke) SMOKE=1; shift ;;
        --no-wandb) NO_WANDB=1; shift ;;
        *) echo "Usage: $0 --run-dir PATH [--config YAML] [--seed N] [--gpus 0,1] [--dry-run] [--smoke] [--no-wandb]" >&2; exit 2 ;;
    esac
done
if [[ -z "${RUN_DIR}" ]]; then
    echo "--run-dir must name a new experiment directory" >&2
    exit 2
fi
if [[ -e "${RUN_DIR}" ]]; then
    echo "Refusing existing run directory: ${RUN_DIR}" >&2
    exit 2
fi
if (( ! SMOKE )) && [[ ! "${GPUS}" =~ ^[0-9]+,[0-9]+$ || "${GPUS%,*}" == "${GPUS#*,}" ]]; then
    echo "Full training requires two distinct GPU IDs, e.g. --gpus 0,1" >&2
    exit 2
fi
CONFIG_PATH="${RUN_DIR}/warm_start_config.yaml"
PREPARE=(python -m project.soft_actor_critic_warm_start.scripts.prepare
    --run-dir "${RUN_DIR}" --output "${CONFIG_PATH}" --config "${SOURCE_CONFIG}")
if (( SEED_SET )); then PREPARE+=(--seed "${SEED}"); fi
if (( SMOKE )); then PREPARE+=(--smoke); fi
if (( NO_WANDB )); then PREPARE+=(--no-wandb); fi
"${PREPARE[@]}"
if (( SMOKE )); then
    COMMAND=(python -m pufferlib.pufferl_sac_ppo_warm_start --config "${CONFIG_PATH}")
else
    COMMAND=(torchrun --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0
        -m pufferlib.pufferl_sac_ppo_warm_start --config "${CONFIG_PATH}")
fi
printf 'Command: '
printf '%q ' "${COMMAND[@]}"
printf '\n'
if (( DRY_RUN )); then exit 0; fi
export CUDA_VISIBLE_DEVICES="${GPUS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "${COMMAND[@]}"
