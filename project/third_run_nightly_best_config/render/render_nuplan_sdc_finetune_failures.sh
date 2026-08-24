#!/usr/bin/env bash
set -euo pipefail

# Render selected failures from an existing third-run fine-tuned evaluation.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./project/third_run_nightly_best_config/render/render_nuplan_sdc_finetune_failures.sh 0 carla
#   CUDA_VISIBLE_DEVICES=1 ./project/third_run_nightly_best_config/render/render_nuplan_sdc_finetune_failures.sh 1 nuplan_single offroad
#   CUDA_VISIBLE_DEVICES=2 ./project/third_run_nightly_best_config/render/render_nuplan_sdc_finetune_failures.sh 2 nuplan_multi collision 10

if (( $# < 2 || $# > 4 )); then
    echo "Usage: CUDA_VISIBLE_DEVICES=<gpu-index> $0 <seed: 0|1|2> <carla|nuplan_single|nuplan_multi> [failure-mode] [max-rendered-failures]" >&2
    exit 2
fi
if [[ ! "$1" =~ ^[012]$ ]]; then
    echo "Seed must be 0, 1, or 2." >&2
    exit 2
fi

case "$2" in
    carla|nuplan_single|nuplan_multi) DATASET="$2" ;;
    *)
        echo "Dataset must be carla, nuplan_single, or nuplan_multi." >&2
        exit 2
        ;;
esac

FAILURE_MODE="${3:-all_infractions}"
case "${FAILURE_MODE}" in
    all_infractions) RENDER_FILTER="all_infractions" ;;
    collision) RENDER_FILTER="collision_rate" ;;
    at_fault_collision) RENDER_FILTER="at_fault_collision_rate" ;;
    offroad) RENDER_FILTER="offroad_rate" ;;
    red_light) RENDER_FILTER="red_light_violation_rate" ;;
    *)
        echo "Failure mode must be all_infractions, collision, at_fault_collision, offroad, or red_light." >&2
        exit 2
        ;;
esac

if [[ ! "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]]; then
    echo "CUDA_VISIBLE_DEVICES must name exactly one GPU index." >&2
    exit 2
fi

DRY_RUN="${DRY_RUN:-0}"
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 2
fi

MAX_RENDERED_FAILURES="${4:-${MAX_RENDERED_FAILURES:-null}}"
if [[ "${MAX_RENDERED_FAILURES}" != "null" && ! "${MAX_RENDERED_FAILURES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_RENDERED_FAILURES must be a positive integer or null." >&2
    exit 2
fi

SEED="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/../.." && pwd)"
FINETUNE_ROOT="${REPO_ROOT}/experiments/third_run_nightly_best_config/nuplan_sdc_finetune"
BENCHMARK_CONFIG="${PROJECT_DIR}/override_config/evaluation_benchmarks.yaml"

shopt -s nullglob
RUN_DIRS=("${FINETUNE_ROOT}"/nuplan_sdc_finetune_*_seed"${SEED}")
shopt -u nullglob
if (( ${#RUN_DIRS[@]} != 1 )); then
    echo "Expected exactly one completed third-run fine-tune for seed ${SEED}; found ${#RUN_DIRS[@]}." >&2
    exit 1
fi

RUN_DIR="${RUN_DIRS[0]}"
MODEL_PATH="${RUN_DIR}/final_model.pt"
CHECKPOINT_CONFIG="${RUN_DIR}/config.yaml"

shopt -s nullglob
METRICS_FILES=("${RUN_DIR}"/eval/"${DATASET}"_final_model_mean_metrics/*/episode_metrics.csv)
shopt -u nullglob
if (( ${#METRICS_FILES[@]} == 0 )); then
    echo "No standalone final-model metrics CSV found for seed ${SEED}, dataset ${DATASET}." >&2
    echo "Run eval/evaluate_nuplan_sdc_finetune.sh for this seed first." >&2
    exit 1
fi
METRICS_CSV="${METRICS_FILES[${#METRICS_FILES[@]} - 1]}"

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

RUN_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
METRICS_DIR="$(dirname "${METRICS_CSV}")"
ANALYSIS_DIR="${METRICS_DIR}/failure_analysis/${FAILURE_MODE}/${RUN_TIMESTAMP}"
ANALYSIS_CSV="${ANALYSIS_DIR}/episode_metrics.csv"
INDEX_PATH="${ANALYSIS_DIR}/failures/rendered_replays/index.html"

if [[ "${DRY_RUN}" == "0" ]]; then
    mkdir -p "$(dirname "${ANALYSIS_DIR}")"
    mkdir "${ANALYSIS_DIR}"
    cp -- "${METRICS_CSV}" "${ANALYSIS_CSV}"
fi

# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"

EVAL_COMMAND=(
    puffer eval puffer_drive "${DATASET}"
    "load_model_path=${MODEL_PATH}"
    "eval.benchmark_config=${BENCHMARK_CONFIG}"
    "eval.failure_replay_csv=${ANALYSIS_CSV}"
    "eval.render_filter=${RENDER_FILTER}"
    "eval.max_rendered_failures=${MAX_RENDERED_FAILURES}"
    "eval.num_agents=300"
    "eval.action_selection=mean"
    "eval.capture_observations=true"
    "vec.num_envs=20"
    "wandb=False"
)

printf 'Fine-tune:      %s\n' "${RUN_DIR}"
printf 'Source metrics: %s\n' "${METRICS_CSV}"
printf 'Failure mode:  %s (%s)\n' "${FAILURE_MODE}" "${RENDER_FILTER}"
printf 'Render limit:  %s\n' "${MAX_RENDERED_FAILURES}"
printf 'HTML index:    %s\n' "${INDEX_PATH}"

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'Command:'
    printf ' %q' "${EVAL_COMMAND[@]}"
    printf '\n'
    exit 0
fi

exec "${EVAL_COMMAND[@]}"
