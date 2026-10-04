#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT=""
OUTPUT_DIR=""
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS=()
ARGUMENT_COUNT=$#
for ((ARGUMENT_IDX=0; ARGUMENT_IDX<ARGUMENT_COUNT && $#>0; ARGUMENT_IDX++)); do
    case "$1" in
        --eval-root|--output-dir|--failure-mode|--max-failures|--workers)
            if (( $# < 2 )) || [[ "$2" == --* ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            case "$1" in
                --eval-root) EVAL_ROOT="$2" ;;
                --output-dir) OUTPUT_DIR="$2" ;;
                *) EXTRA_ARGS+=("$1" "$2") ;;
            esac
            shift 2
            ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help)
            echo "Usage: CUDA_VISIBLE_DEVICES=2,3,4,5 $0 --eval-root DIR --output-dir NEW_DIR [--failure-mode MODE] [--max-failures 10|all] [--workers 2] [--dry-run]"
            exit 0
            ;;
        *) echo "Unknown argument: $1 (k and device are assigned by this launcher)" >&2; exit 2 ;;
    esac
done

if [[ ! "${CUDA_VISIBLE_DEVICES:-}" =~ ^(0|[1-9][0-9]*),(0|[1-9][0-9]*),(0|[1-9][0-9]*),(0|[1-9][0-9]*)$ ]]; then
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
if [[ ! -d "${EVAL_ROOT}" || -z "${OUTPUT_DIR}" ]]; then
    echo "Provide --eval-root pointing to an existing directory and --output-dir pointing to a new directory." >&2
    exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Output path already exists: ${OUTPUT_DIR}" >&2
    exit 2
fi
EVAL_ROOT="$(realpath -- "${EVAL_ROOT}")"
OUTPUT_DIR="$(realpath -m -- "${OUTPUT_DIR}")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Validate every source before starting any GPU job or creating output.
for EXECUTION_K in 1 2 3 4; do
    DRY_RUN=1 "${SCRIPT_DIR}/render_carla.sh" --eval-root "${EVAL_ROOT}" \
        --output-dir "${OUTPUT_DIR}/run_k${EXECUTION_K}" --k "${EXECUTION_K}" \
        --dry-run "${EXTRA_ARGS[@]}"
done

PIDS=()
if [[ "${DRY_RUN}" == 0 ]]; then
    command -v setsid >/dev/null
    mkdir -p -- "$(dirname "${OUTPUT_DIR}")"
    mkdir -- "${OUTPUT_DIR}"
    # Separate process groups include the Python process and its simulator workers.
    trap 'trap - INT TERM; for PID in "${PIDS[@]}"; do kill -TERM -- "-${PID}" 2>/dev/null || true; done; wait || true; exit 130' INT TERM
fi

for GPU_IDX in 0 1 2 3; do
    EXECUTION_K=$((GPU_IDX + 1))
    COMMAND=("${SCRIPT_DIR}/render_carla.sh" --eval-root "${EVAL_ROOT}"
        --output-dir "${OUTPUT_DIR}/run_k${EXECUTION_K}" --k "${EXECUTION_K}" --device cuda:0
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
        printf 'k=%s rendering completed\n' "$((JOB_IDX + 1))"
    else
        JOB_STATUS=$?
        printf 'k=%s failed (exit %s); see %s/k%s.log\n' "$((JOB_IDX + 1))" "${JOB_STATUS}" "${OUTPUT_DIR}" "$((JOB_IDX + 1))" >&2
        STATUS=1
    fi
done
exit "${STATUS}"
