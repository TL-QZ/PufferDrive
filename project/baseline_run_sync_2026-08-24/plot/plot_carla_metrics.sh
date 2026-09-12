#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )) || [[ ! "$1" =~ ^[012]$ ]]; then
    echo "Usage: $0 <seed: 0|1|2>" >&2
    exit 2
fi

SEED="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/../.." && pwd)"
EXPERIMENT_ROOT="${REPO_ROOT}/experiments/baseline_run_sync_2026-08-24"

shopt -s nullglob
RUN_DIRS=("${EXPERIMENT_ROOT}"/baseline_run_sync_2026-08-24_*_seed"${SEED}")
shopt -u nullglob
if (( ${#RUN_DIRS[@]} != 1 )); then
    echo "Expected one baseline run for seed ${SEED}; found ${#RUN_DIRS[@]}." >&2
    exit 1
fi

shopt -s nullglob
CARLA_CSVS=("${RUN_DIRS[0]}"/eval/carla_final_model_mean_metrics/*/episode_metrics.csv)
shopt -u nullglob
if (( ${#CARLA_CSVS[@]} != 1 )); then
    echo "Expected one final CARLA CSV; found ${#CARLA_CSVS[@]}." >&2
    exit 1
fi

source "${REPO_ROOT}/.venv/bin/activate"
python "${REPO_ROOT}/project/metric_analysis/plot_carla_metrics.py" \
    --carla-csv "${CARLA_CSVS[0]}" \
    --model-seed "${SEED}" \
    --output-dir "${PROJECT_DIR}/output/carla/seed${SEED}"
