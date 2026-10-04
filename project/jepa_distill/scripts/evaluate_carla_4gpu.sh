#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT=""
OUTPUT_DIR=""
BENCHMARK="carla"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS=()

while (( $# )); do
    case "$1" in
        --checkpoint|--output-dir|--benchmark)
            if (( $# < 2 )) || [[ "$2" == --* ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            case "$1" in
                --checkpoint) CHECKPOINT="$2" ;;
                --output-dir) OUTPUT_DIR="$2" ;;
                --benchmark) BENCHMARK="$2" ;;
            esac
            shift 2
            ;;
        --smoke) EXTRA_ARGS+=(--smoke); shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help)
            echo "Usage: CUDA_VISIBLE_DEVICES=2,3,4,5 $0 --checkpoint PATH --output-dir NEW_DIR [--benchmark carla|nuplan_single|nuplan_multi] [--smoke] [--dry-run]"
            exit 0
            ;;
        *) echo "Unknown argument: $1 (k and device are assigned by this launcher)" >&2; exit 2 ;;
    esac
done

case "${BENCHMARK}" in
    carla|nuplan_single|nuplan_multi) ;;
    *) echo "Unsupported benchmark: ${BENCHMARK}" >&2; exit 2 ;;
esac

if [[ ! "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]]; then
    echo "CUDA_VISIBLE_DEVICES must contain exactly four distinct GPU indices, e.g. 2,3,4,5." >&2
    exit 2
fi
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
declare -A SEEN_GPUS=()
for GPU_ID in "${GPU_IDS[@]}"; do
    if [[ -n "${SEEN_GPUS[$GPU_ID]:-}" ]]; then
        echo "Duplicate GPU index: ${GPU_ID}" >&2
        exit 2
    fi
    SEEN_GPUS[$GPU_ID]=1
done
if [[ "${DRY_RUN}" != 0 && "${DRY_RUN}" != 1 ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 2
fi
if [[ ! -f "${CHECKPOINT}" || -z "${OUTPUT_DIR}" ]]; then
    echo "Provide --checkpoint pointing to an existing file and --output-dir pointing to a new directory." >&2
    exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Output path already exists: ${OUTPUT_DIR}" >&2
    exit 2
fi
# Resolve caller-relative paths before the single-GPU launcher changes directory.
CHECKPOINT="$(realpath -- "${CHECKPOINT}")"
OUTPUT_DIR="$(realpath -m -- "${OUTPUT_DIR}")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
if [[ ! "${OMP_NUM_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "OMP_NUM_THREADS must be a positive integer." >&2
    exit 2
fi

PIDS=()
if [[ "${DRY_RUN}" == 0 ]]; then
    command -v setsid >/dev/null
    mkdir -p -- "$(dirname "${OUTPUT_DIR}")"
    mkdir -- "${OUTPUT_DIR}"
    # Each job owns a process group, including its simulator workers.
    trap 'trap - INT TERM; for PID in "${PIDS[@]}"; do kill -TERM -- "-${PID}" 2>/dev/null || true; done; wait || true; exit 130' INT TERM
fi

for GPU_IDX in 0 1 2 3; do
    EXECUTION_K=$((GPU_IDX + 1))
    COMMAND=("${SCRIPT_DIR}/evaluate_carla.sh" --checkpoint "${CHECKPOINT}"
        --output-dir "${OUTPUT_DIR}/run_k${EXECUTION_K}" --k "${EXECUTION_K}" --device cuda:0
        --benchmark "${BENCHMARK}"
        "${EXTRA_ARGS[@]}")
    printf 'k=%s GPU=%s log=%s\n' "${EXECUTION_K}" "${GPU_IDS[$GPU_IDX]}" "${OUTPUT_DIR}/k${EXECUTION_K}.log"
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU_IDS[$GPU_IDX]}"
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    if [[ "${DRY_RUN}" == 0 ]]; then
        CUDA_VISIBLE_DEVICES="${GPU_IDS[$GPU_IDX]}" setsid "${COMMAND[@]}" >"${OUTPUT_DIR}/k${EXECUTION_K}.log" 2>&1 &
        PIDS+=("$!")
    fi
done

STATUS=0
for JOB_IDX in "${!PIDS[@]}"; do
    if wait "${PIDS[$JOB_IDX]}"; then
        printf 'k=%s completed\n' "$((JOB_IDX + 1))"
    else
        JOB_STATUS=$?
        printf 'k=%s failed (exit %s); see %s/k%s.log\n' "$((JOB_IDX + 1))" "${JOB_STATUS}" "${OUTPUT_DIR}" "$((JOB_IDX + 1))" >&2
        STATUS=1
    fi
done
exit "${STATUS}"
