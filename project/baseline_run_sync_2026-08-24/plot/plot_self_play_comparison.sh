#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 )) || [[ ! "$1" =~ ^[012]$ ]]; then
    echo "Usage: $0 <seed: 0|1|2> [--models <carla_trained|nuplan_finetuned|nuplan_self_play> ...]" >&2
    exit 2
fi
SEED="$1"
shift
MODELS=(carla_trained nuplan_finetuned nuplan_self_play)
if (( $# > 0 )); then
    if [[ "$1" != "--models" ]] || (( $# < 2 )); then
        echo "Expected --models followed by one to three distinct model names." >&2
        exit 2
    fi
    shift
    MODELS=("$@")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/../.." && pwd)"
EXPERIMENT_ROOT="${REPO_ROOT}/experiments/baseline_run_sync_2026-08-24"
declare -A RUN_PATTERNS=(
    [carla_trained]="${EXPERIMENT_ROOT}/baseline_run_sync_2026-08-24_*_seed${SEED}"
    [nuplan_finetuned]="${EXPERIMENT_ROOT}/nuplan_sdc_finetune_dt03/baseline_run_sync_2026-08-24_nuplan_sdc_finetune_dt03_*_seed${SEED}"
    [nuplan_self_play]="${EXPERIMENT_ROOT}/nuplan_selfplay_dt03/baseline_run_sync_2026-08-24_nuplan_selfplay_dt03_*_seed${SEED}"
)
declare -A SELECTED_MODELS=()
SELECTION_NAME=""
for MODEL in "${MODELS[@]}"; do
    case "${MODEL}" in
        carla_trained|nuplan_finetuned|nuplan_self_play) ;;
        *) echo "Unknown model: ${MODEL}" >&2; exit 2 ;;
    esac
    if [[ -n "${SELECTED_MODELS[${MODEL}]:-}" ]]; then
        echo "Duplicate model: ${MODEL}" >&2
        exit 2
    fi
    SELECTED_MODELS["${MODEL}"]=1
    SELECTION_NAME+="${SELECTION_NAME:+__}${MODEL}"
done

shopt -s nullglob
JSON_ARGS=()
for MODEL in "${MODELS[@]}"; do
    # Split the directory from the basename glob to preserve spaces in paths.
    RUN_PATTERN="${RUN_PATTERNS[${MODEL}]}"
    RUN_PARENT="${RUN_PATTERN%/*}"
    RUN_GLOB="${RUN_PATTERN##*/}"
    RUN_DIRS=("${RUN_PARENT}"/${RUN_GLOB})
    if (( ${#RUN_DIRS[@]} != 1 )) || [[ ! -d "${RUN_DIRS[0]}" ]]; then
        echo "Expected one ${MODEL} run for seed ${SEED}; found ${#RUN_DIRS[@]}. Pattern: ${RUN_PATTERN}" >&2
        printf '  %s\n' "${RUN_DIRS[@]}" >&2
        exit 1
    fi
    for BENCHMARK in carla nuplan_single nuplan_multi; do
        SUMMARIES=("${RUN_DIRS[0]}"/eval/"${BENCHMARK}"_final_model_mean_metrics/*/evaluation_summary.json)
        if (( ${#SUMMARIES[@]} != 1 )); then
            echo "Expected one ${MODEL} / ${BENCHMARK} summary; found ${#SUMMARIES[@]} in ${RUN_DIRS[0]}/eval/${BENCHMARK}_final_model_mean_metrics." >&2
            printf '  %s\n' "${SUMMARIES[@]}" >&2
            exit 1
        fi
        printf '%s / %s: %s\n' "${MODEL}" "${BENCHMARK}" "${SUMMARIES[0]}"
        JSON_ARGS+=("--${MODEL//_/-}-${BENCHMARK//_/-}-json" "${SUMMARIES[0]}")
    done
done

source "${REPO_ROOT}/.venv/bin/activate"
python "${REPO_ROOT}/project/metric_analysis/plot_finetune_comparison.py" \
    --seed "${SEED}" --models "${MODELS[@]}" "${JSON_ARGS[@]}" \
    --output-dir "${PROJECT_DIR}/output/self_play_comparison/seed${SEED}/${SELECTION_NAME}"
