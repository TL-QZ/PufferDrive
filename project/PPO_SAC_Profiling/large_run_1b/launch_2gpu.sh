#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
if [[ ! "${CUDA_VISIBLE_DEVICES}" =~ ^([0-9]+),([0-9]+)$ ]] || [[ "${BASH_REMATCH[1]}" == "${BASH_REMATCH[2]}" ]]; then
    echo 'CUDA_VISIBLE_DEVICES must specify two different GPU indices.' >&2
    exit 2
fi
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
RUN_ROOT="${1:?Usage: launch_2gpu.sh NEW_OUTPUT_DIRECTORY}"
if [[ -e "${RUN_ROOT}" ]]; then
    echo "Output already exists: ${RUN_ROOT}" >&2
    exit 2
fi
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
trap 'printf "%s\n" failed > "${RUN_ROOT}/status.txt"' ERR
printf '%s\n' running > "${RUN_ROOT}/status.txt"
cp "${SCRIPT_DIR}"/*.yaml "${RUN_ROOT}/"
git diff --binary > "${RUN_ROOT}/working_tree.patch"
# The optimized sources include untracked additions; preserve their exact bytes.
tar --exclude=results --exclude=__pycache__ --exclude=.pytest_cache \
    --exclude=compile_cache -czf "${RUN_ROOT}/profiling_source.tar.gz" \
    project/PPO_SAC_Profiling
date -u +%FT%TZ > "${RUN_ROOT}/started_utc.txt"
for CASE_NAME in stock_baseline optimized; do
    printf '%s\n' "${CASE_NAME}" > "${RUN_ROOT}/active_case.txt"
    torchrun --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
        project/PPO_SAC_Profiling/profile_training.py \
        --profile-config "${RUN_ROOT}/${CASE_NAME}.yaml" \
        --source-config "${RUN_ROOT}/source_sac.yaml" \
        --output-dir "${RUN_ROOT}/${CASE_NAME}" \
        2>&1 | tee "${RUN_ROOT}/${CASE_NAME}.log"
done
python project/PPO_SAC_Profiling/analyze_runs.py "${RUN_ROOT}" \
    --output "${RUN_ROOT}/comparison"
python "${SCRIPT_DIR}/compare_progress.py" "${RUN_ROOT}" > "${RUN_ROOT}/speedup.json"
printf '%s\n' completed > "${RUN_ROOT}/active_case.txt"
printf '%s\n' completed > "${RUN_ROOT}/status.txt"
date -u +%FT%TZ > "${RUN_ROOT}/completed_utc.txt"
