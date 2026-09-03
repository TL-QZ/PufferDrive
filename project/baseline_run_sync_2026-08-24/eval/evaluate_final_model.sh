#!/usr/bin/env bash
set -euo pipefail

# Evaluate one completed synced-baseline CARLA seed on CARLA and both nuPlan modes.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./project/baseline_run_sync_2026-08-24/eval/evaluate_final_model.sh 0
#   CUDA_VISIBLE_DEVICES=1 DRY_RUN=1 ./project/baseline_run_sync_2026-08-24/eval/evaluate_final_model.sh 1

if (( $# != 1 )) || [[ ! "$1" =~ ^[012]$ ]]; then
    echo "Usage: CUDA_VISIBLE_DEVICES=<gpu-index> $0 <seed: 0|1|2>" >&2
    exit 2
fi
if [[ ! "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]]; then
    echo "CUDA_VISIBLE_DEVICES must name exactly one GPU index." >&2
    exit 2
fi

DRY_RUN="${DRY_RUN:-0}"
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 2
fi

SEED="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/../.." && pwd)"
EXPERIMENT_ROOT="${REPO_ROOT}/experiments/baseline_run_sync_2026-08-24"
BENCHMARK_CONFIG="${PROJECT_DIR}/override_config/evaluation_benchmarks.yaml"

shopt -s nullglob
RUN_DIRS=("${EXPERIMENT_ROOT}"/baseline_run_sync_2026-08-24_*_seed"${SEED}")
shopt -u nullglob
if (( ${#RUN_DIRS[@]} != 1 )); then
    echo "Expected exactly one synced-baseline CARLA run for seed ${SEED}; found ${#RUN_DIRS[@]}." >&2
    exit 1
fi

RUN_DIR="${RUN_DIRS[0]}"
MODEL_PATH="${RUN_DIR}/final_model.pt"
CHECKPOINT_CONFIG="${RUN_DIR}/config.yaml"
if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Missing final checkpoint: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -f "${CHECKPOINT_CONFIG}" ]]; then
    echo "Missing checkpoint configuration: ${CHECKPOINT_CONFIG}" >&2
    exit 1
fi
if [[ ! -f "${BENCHMARK_CONFIG}" ]]; then
    echo "Missing evaluation benchmark configuration: ${BENCHMARK_CONFIG}" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"

EVAL_COMMAND=(
    puffer eval puffer_drive carla,nuplan_single,nuplan_multi
    "load_model_path=${MODEL_PATH}"
    "eval.benchmark_config=${BENCHMARK_CONFIG}"
    "eval.output_name=final_model_mean_metrics"
    "eval.num_agents=300"
    "eval.action_selection=mean"
    "eval.render_scenarios=false"
    "eval.render_filter=null"
    "eval.capture_observations=false"
    "vec.num_envs=20"
    "wandb=True"
)

printf 'Run:        %s\n' "${RUN_DIR}"
printf 'Checkpoint: %s\n' "${MODEL_PATH}"
printf 'Command:'
printf ' %q' "${EVAL_COMMAND[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
fi

wandb login --verify
exec "${EVAL_COMMAND[@]}"
