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
EXPERIMENT_ROOT="${REPO_ROOT}/experiments/baseline_run_sync_2026-08-24/nuplan_sdc_finetune_dt03"

shopt -s nullglob
RUN_DIRS=("${EXPERIMENT_ROOT}"/baseline_run_sync_2026-08-24_nuplan_sdc_finetune_dt03_*_seed"${SEED}")
shopt -u nullglob
if (( ${#RUN_DIRS[@]} != 1 )); then
    echo "Expected one fine-tuned run for seed ${SEED}; found ${#RUN_DIRS[@]}." >&2
    exit 1
fi

JSON_ARGS=()
for BENCHMARK in carla nuplan_single nuplan_multi; do
    shopt -s nullglob
    SUMMARIES=("${RUN_DIRS[0]}"/eval/"${BENCHMARK}"_final_model_mean_metrics/*/evaluation_summary.json)
    shopt -u nullglob
    if (( ${#SUMMARIES[@]} != 1 )); then
        echo "Expected one final ${BENCHMARK} summary; found ${#SUMMARIES[@]}." >&2
        exit 1
    fi
    JSON_ARGS+=("--${BENCHMARK//_/-}-json" "${SUMMARIES[0]}")
done

source "${REPO_ROOT}/.venv/bin/activate"
python "${REPO_ROOT}/project/metric_analysis/plot_benchmark_comparison.py" \
    "${JSON_ARGS[@]}" \
    --model-seed "${SEED}" \
    --output-dir "${PROJECT_DIR}/output/finetuned_benchmarks/seed${SEED}"
