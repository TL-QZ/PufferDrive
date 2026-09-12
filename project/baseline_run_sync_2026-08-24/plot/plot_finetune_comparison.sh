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
BEFORE_RUNS=("${EXPERIMENT_ROOT}"/baseline_run_sync_2026-08-24_*_seed"${SEED}")
AFTER_RUNS=("${EXPERIMENT_ROOT}"/nuplan_sdc_finetune_dt03/baseline_run_sync_2026-08-24_nuplan_sdc_finetune_dt03_*_seed"${SEED}")
shopt -u nullglob
if (( ${#BEFORE_RUNS[@]} != 1 || ${#AFTER_RUNS[@]} != 1 )); then
    echo "Expected one before and one after run for seed ${SEED}; found ${#BEFORE_RUNS[@]} and ${#AFTER_RUNS[@]}." >&2
    exit 1
fi

JSON_ARGS=()
for STAGE in before after; do
    [[ "${STAGE}" == "before" ]] && RUN_DIR="${BEFORE_RUNS[0]}" || RUN_DIR="${AFTER_RUNS[0]}"
    for BENCHMARK in carla nuplan_single nuplan_multi; do
        shopt -s nullglob
        SUMMARIES=("${RUN_DIR}"/eval/"${BENCHMARK}"_final_model_mean_metrics/*/evaluation_summary.json)
        shopt -u nullglob
        if (( ${#SUMMARIES[@]} != 1 )); then
            echo "Expected one ${STAGE} ${BENCHMARK} summary; found ${#SUMMARIES[@]}." >&2
            exit 1
        fi
        JSON_ARGS+=("--${STAGE}-${BENCHMARK//_/-}-json" "${SUMMARIES[0]}")
    done
done

source "${REPO_ROOT}/.venv/bin/activate"
python "${REPO_ROOT}/project/metric_analysis/plot_finetune_comparison.py" \
    --seed "${SEED}" \
    "${JSON_ARGS[@]}" \
    --output-dir "${PROJECT_DIR}/output/finetune_comparison/seed${SEED}"
