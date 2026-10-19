#!/usr/bin/env bash
set -euo pipefail

# Full CARLA evaluation followed by selected failure replay rendering.
# Usage: CUDA_VISIBLE_DEVICES=0 bash round3_individual_eval_render.sh 1
# Optional: MAX_RENDERED_FAILURES=20 RENDER_FILTER=offroad_rate DRY_RUN=1
# Set SWEEP_DIR explicitly if runs/round3 contains more than one sweep.
if (( $# != 1 )) || [[ ! "$1" =~ ^[1-9][0-9]{0,2}$ ]]; then
    echo "Usage: CUDA_VISIBLE_DEVICES=0 bash $0 <round-three-trial-number>" >&2
    exit 2
fi

TRIAL_ID="$1"
case "$TRIAL_ID" in
    [1-9]|1[0-2]|49|5[0-9]|60|9[7-9]|10[0-8]) ;;
    *) echo "Round-three trial must be 1–12, 49–60, or 97–108." >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_RENDERED_FAILURES="${MAX_RENDERED_FAILURES:-10}"
RENDER_FILTER="${RENDER_FILTER:-all_infractions}"
DRY_RUN="${DRY_RUN:-0}"
if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]; then
    echo "CUDA_VISIBLE_DEVICES must name exactly one GPU index." >&2
    exit 2
fi
if [[ ! "$MAX_RENDERED_FAILURES" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_RENDERED_FAILURES must be a positive integer." >&2
    exit 2
fi
case "$RENDER_FILTER" in
    all_infractions|collision_rate|at_fault_collision_rate|offroad_rate|red_light_violation_rate) ;;
    *) echo "Unsupported RENDER_FILTER: $RENDER_FILTER" >&2; exit 2 ;;
esac
if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/../.." && pwd)"
if [[ -z "${SWEEP_DIR:-}" ]]; then
    shopt -s nullglob
    SWEEP_DIRS=("${PROJECT_DIR}"/runs/round3/sac108_grid_*)
    shopt -u nullglob
    if (( ${#SWEEP_DIRS[@]} != 1 )); then
        echo "Expected one round-three sweep; found ${#SWEEP_DIRS[@]}. Set SWEEP_DIR to the desired sweep directory." >&2
        exit 1
    fi
    SWEEP_DIR="${SWEEP_DIRS[0]}"
fi
SWEEP_DIR="$(cd "$SWEEP_DIR" && pwd)"
printf -v TRIAL_NAME 'trial_%02d' "$TRIAL_ID"
LEARNER_DIR="${SWEEP_DIR}/${TRIAL_NAME}/learner"
BENCHMARK_CONFIG="${REPO_ROOT}/project/baseline_run_sync_2026-08-24/override_config/evaluation_benchmarks.yaml"
for REQUIRED_FILE in "${LEARNER_DIR}/final_model.pt" "${LEARNER_DIR}/config.yaml" "$BENCHMARK_CONFIG"; do
    if [[ ! -f "$REQUIRED_FILE" ]]; then
        echo "Missing required file: $REQUIRED_FILE" >&2
        exit 1
    fi
done

source "${REPO_ROOT}/.venv/bin/activate"
cd "$REPO_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
OUTPUT_NAME="round3_${TRIAL_NAME}_full"
EVAL_COMMAND=(
    python -B -m pufferlib.pufferl_sac eval puffer_drive carla
    "load_model_path=${LEARNER_DIR}/final_model.pt"
    "eval.benchmark_config=${BENCHMARK_CONFIG}"
    "eval.output_name=${OUTPUT_NAME}"
    eval.output_dir_name=eval
    eval.num_agents=300
    eval.action_selection=mean
    eval.render_scenarios=false
    eval.failure_replay_csv=null
    "eval.render_filter=${RENDER_FILTER}"
    "eval.max_rendered_failures=${MAX_RENDERED_FAILURES}"
    eval.capture_observations=true
    vec.num_envs=20
    vec.num_workers=auto
    train.device=cuda
    train.amp=true
    train.precision=bfloat16
    train.compile=false
    wandb=false
)

printf 'Checkpoint: %s\n' "${LEARNER_DIR}/final_model.pt"
printf 'Full CARLA: 1,000 scenarios, eight towns, seed 42, 6,000 steps, dt=0.1\n'
printf 'Render: up to %s failures matching %s, GPU %s\n' "$MAX_RENDERED_FAILURES" "$RENDER_FILTER" "$CUDA_VISIBLE_DEVICES"
printf 'Reports: %s/eval/carla_%s/<timestamp>/\n' "$LEARNER_DIR" "$OUTPUT_NAME"
printf 'Gallery: failures/rendered_replays/index.html inside the report directory\n'
if [[ "$DRY_RUN" == 1 ]]; then
    printf 'Command:'
    printf ' %q' "${EVAL_COMMAND[@]}"
    printf '\n'
    exit 0
fi
exec "${EVAL_COMMAND[@]}"
