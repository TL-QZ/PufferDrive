#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT=""
OUTPUT_DIR=""
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS=()
CHILD_PID=""

while (( $# )); do
    case "$1" in
        --checkpoint|--output-dir)
            if (( $# < 2 )) || [[ "$2" == --* ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            if [[ "$1" == --checkpoint ]]; then CHECKPOINT="$2"; else OUTPUT_DIR="$2"; fi
            shift 2
            ;;
        --smoke) EXTRA_ARGS+=(--smoke); shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help)
            echo "Usage: CUDA_VISIBLE_DEVICES=2,3,4,5 $0 --checkpoint PATH --output-dir NEW_DIR [--smoke] [--dry-run]"
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ "${DRY_RUN}" != 0 && "${DRY_RUN}" != 1 ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 2
fi
if [[ ! -f "${CHECKPOINT}" || -z "${OUTPUT_DIR}" ]]; then
    echo "Provide --checkpoint pointing to an existing file and --output-dir pointing to a new directory." >&2
    exit 2
fi
if [[ -e "${OUTPUT_DIR}" || -L "${OUTPUT_DIR}" ]]; then
    echo "Output path already exists: ${OUTPUT_DIR}" >&2
    exit 2
fi
# Resolve caller-relative paths before the delegated launcher changes directory.
CHECKPOINT="$(realpath -- "${CHECKPOINT}")"
OUTPUT_DIR="$(realpath -m -- "${OUTPUT_DIR}")"

terminate_active_child() {
    local signal_name="$1"
    local exit_status="$2"
    if [[ -n "${CHILD_PID}" ]]; then
        kill -s "${signal_name}" "${CHILD_PID}" 2>/dev/null || true
        wait "${CHILD_PID}" 2>/dev/null || true
        CHILD_PID=""
    fi
    exit "${exit_status}"
}
trap 'terminate_active_child TERM 130' INT
trap 'terminate_active_child TERM 143' TERM

run_benchmark() {
    local benchmark="$1"
    local command=("${SCRIPT_DIR}/evaluate_carla_4gpu.sh"
        --checkpoint "${CHECKPOINT}"
        --output-dir "${OUTPUT_DIR}/${benchmark}"
        --benchmark "${benchmark}"
        "${EXTRA_ARGS[@]}")
    if [[ "${DRY_RUN}" == 1 ]]; then
        "${command[@]}" --dry-run
        return
    fi

    "${command[@]}" &
    CHILD_PID="$!"
    local status=0
    if wait "${CHILD_PID}"; then
        status=0
    else
        status="$?"
    fi
    CHILD_PID=""
    return "${status}"
}

# Each benchmark's four k jobs finish before the next benchmark begins.
run_benchmark nuplan_single
run_benchmark nuplan_multi
